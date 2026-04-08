from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import time

import torch
from torch import nn
import yaml

from dgfm.config import RunRoots
from dgfm.datasets import build_image_dataloaders
from dgfm.models import ModelEMA, build_velocity_model
from dgfm.paths import build_path, ensure_flow_matching_on_path
from dgfm.utils import build_experiment_archive


def _device_from_config(config: dict) -> torch.device:
    requested = config.get("runtime", {}).get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _autocast_context(device: torch.device, enabled: bool):
    if device.type != "cuda" or not enabled:
        return torch.autocast(device_type="cpu", enabled=False)
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)


def _skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    p_mean = -1.2
    p_std = 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma = (rnd_normal * p_std + p_mean).exp()
    time = 1.0 / (1.0 + sigma)
    return torch.clamp(time, min=1.0e-4, max=1.0)


@dataclass(slots=True)
class BaselineTrainer:
    config: dict
    roots: RunRoots
    archive: object | None = field(init=False, default=None)

    def prepare(self) -> None:
        self.roots.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.roots.sample_dir.mkdir(parents=True, exist_ok=True)
        self.roots.log_dir.mkdir(parents=True, exist_ok=True)
        with (self.roots.log_dir / "config_resolved.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.config, handle, sort_keys=False)
        self.archive = build_experiment_archive(self.roots)
        self.archive.dump_yaml("config_resolved.yaml", self.config)

    def _run_epoch(
        self,
        model: nn.Module,
        ema: ModelEMA | None,
        loader,
        optimizer,
        scaler: torch.amp.GradScaler,
        path,
        device: torch.device,
        train: bool,
    ) -> float:
        model.train(train)
        total = 0.0
        count = 0
        train_cfg = self.config.get("train", {})
        batch_limit_key = "max_train_batches" if train else "max_val_batches"
        batch_limit = int(train_cfg.get(batch_limit_key, 0) or 0)
        use_amp = bool(self.config.get("runtime", {}).get("amp", True))
        use_skewed_timesteps = bool(train_cfg.get("skewed_timesteps", True))
        class_cond = self.config.get("model", {}).get("num_classes") is not None
        ctx = torch.enable_grad if train else torch.no_grad
        with ctx():
            for batch_idx, (images, labels) in enumerate(loader):
                if batch_limit > 0 and batch_idx >= batch_limit:
                    break
                images = images.to(device)
                labels = labels.to(device)
                images = images * 2.0 - 1.0
                noise = torch.randn_like(images)
                if use_skewed_timesteps:
                    t = _skewed_timestep_sample(images.shape[0], device=device)
                else:
                    t = torch.rand(images.shape[0], device=device)
                path_sample = path.sample(x_0=noise, x_1=images, t=t)
                model_extra = {"label": labels} if class_cond else {}
                with _autocast_context(device, use_amp):
                    pred = model(path_sample.x_t, path_sample.t, extra=model_extra)
                    loss = torch.mean((pred - path_sample.dx_t) ** 2)
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    if ema is not None:
                        ema.update(model)
                total += float(loss.detach().item())
                count += 1
        return total / max(1, count)

    def run(self, resume: str | None = None, verbose: bool = False) -> None:
        del verbose
        self.prepare()
        ensure_flow_matching_on_path()
        device = _device_from_config(self.config)
        dataloaders = build_image_dataloaders(self.config)
        path = build_path(self.config)
        model = build_velocity_model(self.config).to(device)
        ema = ModelEMA(model, decay=float(self.config["train"].get("ema_decay", 0.9999)))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.config["train"].get("lr", 1.0e-4)),
            weight_decay=float(self.config["train"].get("weight_decay", 0.0)),
            betas=tuple(self.config["train"].get("optimizer_betas", [0.9, 0.95])),
        )
        scaler = torch.amp.GradScaler(device="cuda", enabled=device.type == "cuda" and bool(self.config.get("runtime", {}).get("amp", True)))
        start_epoch = 0
        best_val = float("inf")
        if resume:
            ckpt = torch.load(resume, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            if "ema_model" in ckpt:
                ema.load_state_dict(ckpt["ema_model"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_val = float(ckpt.get("best_val", best_val))
            if "scaler" in ckpt and scaler.is_enabled():
                scaler.load_state_dict(ckpt["scaler"])

        epochs = int(self.config["train"].get("epochs", 1))
        history_path = self.roots.log_dir / "train.jsonl"
        for epoch in range(start_epoch, epochs):
            t0 = time.time()
            train_loss = self._run_epoch(model, ema, dataloaders["train"], optimizer, scaler, path, device, train=True)
            eval_model = ema.shadow if ema is not None else model
            val_loss = self._run_epoch(eval_model, None, dataloaders["val"], optimizer, scaler, path, device, train=False)
            elapsed = time.time() - t0
            payload = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "elapsed_sec": elapsed,
            }
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")
            self.archive.append_jsonl("train.jsonl", payload)
            last_ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "ema_model": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                "best_val": best_val,
                "config": self.config,
            }
            torch.save(last_ckpt, self.roots.checkpoint_dir / "last.pt")
            self.archive.save_checkpoint("last.pt", last_ckpt)
            if val_loss <= best_val:
                best_val = val_loss
                last_ckpt["best_val"] = best_val
                torch.save(last_ckpt, self.roots.checkpoint_dir / "best.pt")
                self.archive.save_checkpoint("best.pt", last_ckpt)
            print(
                f"epoch={epoch + 1}/{epochs} train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} elapsed_sec={elapsed:.2f}",
                flush=True,
            )
