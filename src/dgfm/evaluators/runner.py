from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import re
import time

import torch
from torchvision.utils import save_image

from dgfm.datasets import build_image_dataloaders
from .common import (
    device_from_config,
    load_model_from_checkpoint,
    load_timewarp_from_checkpoint,
    objective_mode,
    sample_condition_labels,
    sample_from_model_batched,
    solver_nfe,
    to_unit_interval,
)
from dgfm.schedulers import build_runtime_time_grid, summarize_time_grid
from .fid import InceptionFeatureExtractor, compute_dataset_stats, compute_generator_stats, frechet_distance, load_stats, save_stats


@dataclass(slots=True)
class EvaluationRunner:
    config: dict
    checkpoint: Path
    eval_root: Path

    def _eval_cfg(self) -> dict:
        return self.config.get("eval", {})

    def _stats_cache_path(self) -> Path:
        dataset_cfg = self.config["dataset"]
        eval_cfg = self._eval_cfg()
        split = str(eval_cfg.get("reference_split", "test"))
        image_size = int(dataset_cfg["image_size"])
        dataset_name = str(dataset_cfg["name"])
        fid_protocol = str(eval_cfg.get("fid_protocol", "torch_fidelity_inceptionv3_2048"))
        protocol_slug = re.sub(r"[^a-zA-Z0-9]+", "_", fid_protocol).strip("_").lower()
        cache_root = Path(dataset_cfg["data_root"]) / ".dgfm_cache"
        return cache_root / f"fid_stats_{dataset_name}_{split}_{image_size}_{protocol_slug}.npz"

    def _prepare_reference_stats(self, feature_extractor, device: torch.device):
        cache_path = self._stats_cache_path()
        if cache_path.exists():
            return load_stats(cache_path), cache_path
        loaders = build_image_dataloaders(self.config)
        split = str(self._eval_cfg().get("reference_split", "test"))
        if split not in loaders:
            raise ValueError(f"Unsupported reference split: {split}")
        stats = compute_dataset_stats(
            loaders[split],
            feature_extractor=feature_extractor,
            device=device,
            image_limit=int(self._eval_cfg().get("reference_num_samples", 0)) or None,
        )
        save_stats(cache_path, stats)
        return stats, cache_path

    def _fid_sample_mode(self, fid_samples: int) -> str:
        return "full" if fid_samples >= 50000 else "approximate"

    def _sample_batch_size(self, fid_batch_size: int) -> int:
        eval_cfg = self._eval_cfg()
        configured = int(eval_cfg.get("sample_batch_size", 0) or 0)
        return configured if configured > 0 else fid_batch_size

    def _fixed_grid_batch_size(self, sample_batch_size: int) -> int:
        eval_cfg = self._eval_cfg()
        configured = int(eval_cfg.get("fixed_grid_batch_size", 0) or 0)
        return configured if configured > 0 else sample_batch_size

    def _save_fixed_grid(
        self,
        model,
        timewarp,
        device: torch.device,
        step_count: int,
        step_dir: Path,
        solver_method: str,
        sample_batch_size: int,
    ) -> dict:
        eval_cfg = self._eval_cfg()
        fixed_seed = int(eval_cfg.get("fixed_seed", 42))
        fixed_grid_size = int(eval_cfg.get("fixed_grid_size", 64))
        nrow = int(eval_cfg.get("grid_nrow", 8))
        channels = int(self.config["dataset"]["channels"])
        image_size = int(self.config["dataset"]["image_size"])
        generator = torch.Generator(device=device).manual_seed(fixed_seed)
        noise = torch.randn(fixed_grid_size, channels, image_size, image_size, generator=generator, device=device)
        labels = sample_condition_labels(self.config, fixed_grid_size, device=device, generator=generator)
        samples = sample_from_model_batched(
            config=self.config,
            model=model,
            x_init=noise,
            step_count=step_count,
            method=solver_method,
            timewarp=timewarp,
            max_batch_size=self._fixed_grid_batch_size(sample_batch_size),
            move_to_cpu=True,
            extra={"label": labels} if labels is not None else None,
        )
        samples = to_unit_interval(samples)
        torch.save(samples, step_dir / "fixed_seed_samples.pt")
        if labels is not None:
            torch.save(labels.detach().cpu(), step_dir / "fixed_seed_labels.pt")
        save_image(samples, step_dir / "fixed_seed_grid.png", nrow=nrow)
        dump_count = int(eval_cfg.get("dump_image_count", 64))
        image_dir = step_dir / "fixed_seed_images"
        image_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(min(dump_count, samples.shape[0])):
            save_image(samples[idx], image_dir / f"{idx:06d}.png")
        return {
            "fixed_seed": fixed_seed,
            "fixed_grid_size": fixed_grid_size,
            "dump_image_count": min(dump_count, samples.shape[0]),
            "class_cond": labels is not None,
        }

    def run(self, step_counts: list[int]) -> None:
        self.eval_root.mkdir(parents=True, exist_ok=True)
        report_dir = self.eval_root / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        device = device_from_config(self.config)
        model = load_model_from_checkpoint(self.config, self.checkpoint, device=device)
        timewarp = load_timewarp_from_checkpoint(self.config, self.checkpoint, device=device)
        feature_extractor = InceptionFeatureExtractor().to(device)
        reference_stats, cache_path = self._prepare_reference_stats(feature_extractor, device)

        eval_cfg = self._eval_cfg()
        fid_samples = int(eval_cfg.get("num_fid_samples", 50000))
        fid_batch_size = int(eval_cfg.get("fid_batch_size", 256))
        sample_batch_size = self._sample_batch_size(fid_batch_size)
        fid_protocol = str(eval_cfg.get("fid_protocol", "torch_fidelity_inceptionv3_2048"))
        solver_method = str(eval_cfg.get("solver_method", "midpoint"))
        mode = objective_mode(self.config)
        channels = int(self.config["dataset"]["channels"])
        image_size = int(self.config["dataset"]["image_size"])
        reference_count = int(reference_stats.count)

        records = []
        for step_count in step_counts:
            step_dir = self.eval_root / f"steps{step_count}"
            step_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            nfe = solver_nfe(step_count=step_count, method=solver_method, mode=mode)

            def sample_fn(batch_size: int) -> torch.Tensor:
                noise = torch.randn(batch_size, channels, image_size, image_size, device=device)
                labels = sample_condition_labels(self.config, batch_size, device=device)
                with torch.no_grad():
                    samples = sample_from_model_batched(
                        config=self.config,
                        model=model,
                        x_init=noise,
                        step_count=step_count,
                        method=solver_method,
                        timewarp=timewarp,
                        max_batch_size=sample_batch_size,
                        extra={"label": labels} if labels is not None else None,
                    )
                return to_unit_interval(samples)

            fake_stats = compute_generator_stats(
                sample_fn=sample_fn,
                feature_extractor=feature_extractor,
                batch_size=sample_batch_size,
                total_samples=fid_samples,
                device=device,
            )
            fid = frechet_distance(reference_stats, fake_stats)
            grid_meta = self._save_fixed_grid(
                model=model,
                timewarp=timewarp,
                device=device,
                step_count=step_count,
                step_dir=step_dir,
                solver_method=solver_method,
                sample_batch_size=sample_batch_size,
            )
            elapsed = time.time() - t0
            save_stats(step_dir / "generated_stats.npz", fake_stats)
            time_grid = build_runtime_time_grid(
                config=self.config,
                step_count=step_count,
                device=device,
                dtype=torch.float32,
                timewarp=timewarp,
            )
            record = {
                "step_count": step_count,
                "integration_steps": step_count,
                "nfe": nfe,
                "nfe_per_step": nfe / step_count,
                "fid": fid,
                "fid_protocol": fid_protocol,
                "num_fid_samples": fid_samples,
                "fid_sample_mode": self._fid_sample_mode(fid_samples),
                "fid_batch_size": fid_batch_size,
                "sample_batch_size": sample_batch_size,
                "reference_count": reference_count,
                "reference_stats": str(cache_path),
                "checkpoint": str(self.checkpoint),
                "solver_method": solver_method,
                "objective_mode": mode,
                "elapsed_sec": elapsed,
                "samples_per_sec": fid_samples / max(elapsed, 1.0e-8),
                "timewarp_enabled": timewarp is not None,
                **summarize_time_grid(time_grid),
                **grid_meta,
            }
            with (step_dir / "metrics.json").open("w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2)
            records.append(record)
            print(
                f"eval step_count={step_count} nfe={nfe} fid={fid:.4f} "
                f"num_fid_samples={fid_samples} fid_mode={record['fid_sample_mode']} elapsed_sec={elapsed:.2f}",
                flush=True,
            )

        with (report_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2)
        with (report_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        best = min(records, key=lambda item: item["fid"])
        with (report_dir / "best.json").open("w", encoding="utf-8") as handle:
            json.dump(best, handle, indent=2)
        print("dgfm evaluation runner completed")
        print(f"eval_root: {self.eval_root}")
        print(f"best_step: {best['step_count']} fid={best['fid']:.4f}")
