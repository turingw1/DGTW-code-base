from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image

from dgfm.datasets import build_image_dataloaders
from dgfm.schedulers import build_time_grid
from .common import device_from_config, load_model_from_checkpoint, sample_with_ode, solver_nfe, to_unit_interval
from .fid import InceptionFeatureExtractor, compute_dataset_stats, compute_generator_stats, frechet_distance, load_stats, save_stats
from .qualitative import build_strategy_panel


@dataclass(slots=True)
class TimewarpSamplingRunner:
    config: dict
    checkpoint: Path
    eval_root: Path

    def _eval_cfg(self) -> dict:
        return self.config.get("eval", {})

    def _tw_cfg(self) -> dict:
        return self.config.get("timewarp_sampling", {})

    def _strategy_dir_name(self, strategy: str) -> str:
        return (
            strategy.replace("@", "_at_")
            .replace(".", "p")
            .replace("/", "_")
            .replace(":", "_")
        )

    def _stats_cache_path(self) -> Path:
        from .runner import EvaluationRunner
        dummy = EvaluationRunner(config=self.config, checkpoint=self.checkpoint, eval_root=self.eval_root)
        return dummy._stats_cache_path()

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

    def _time_grid(self, step_count: int, strategy: str, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return build_time_grid(
            step_count,
            strategy=strategy,
            device=device,
            dtype=dtype,
            power_gamma=float(self._tw_cfg().get("power_gamma", 2.0)),
            random_concentration=float(self._tw_cfg().get("random_concentration", 1.0)),
            random_seed=int(self._tw_cfg().get("random_seed", 123)),
        )

    def _save_strategy_grid(
        self,
        model: torch.nn.Module,
        *,
        device: torch.device,
        step_count: int,
        strategy: str,
        step_dir: Path,
        solver_method: str,
    ) -> dict:
        eval_cfg = self._eval_cfg()
        fixed_seed = int(eval_cfg.get("fixed_seed", 42))
        fixed_grid_size = int(eval_cfg.get("fixed_grid_size", 64))
        nrow = int(eval_cfg.get("grid_nrow", 8))
        channels = int(self.config["dataset"]["channels"])
        image_size = int(self.config["dataset"]["image_size"])
        generator = torch.Generator(device=device).manual_seed(fixed_seed)
        noise = torch.randn(fixed_grid_size, channels, image_size, image_size, generator=generator, device=device)
        time_grid = self._time_grid(step_count, strategy, device=device, dtype=noise.dtype)
        samples = sample_with_ode(
            model=model,
            x_init=noise,
            step_count=step_count,
            method=solver_method,
            time_grid=time_grid,
        )
        samples = to_unit_interval(samples)
        torch.save(samples.detach().cpu(), step_dir / "fixed_seed_samples.pt")
        torch.save(time_grid.detach().cpu(), step_dir / "time_grid.pt")
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
            "time_grid": [float(item) for item in time_grid.detach().cpu().tolist()],
        }

    def _save_strategy_panel(
        self,
        model: torch.nn.Module,
        *,
        device: torch.device,
        strategy_names: list[str],
        step_count: int,
        report_dir: Path,
        solver_method: str,
    ) -> None:
        tw_cfg = self._tw_cfg()
        fixed_seed = int(self._eval_cfg().get("fixed_seed", 42))
        num_examples = int(tw_cfg.get("qualitative_num_examples", 8))
        channels = int(self.config["dataset"]["channels"])
        image_size = int(self.config["dataset"]["image_size"])
        generator = torch.Generator(device=device).manual_seed(fixed_seed)
        noise = torch.randn(num_examples, channels, image_size, image_size, generator=generator, device=device)
        samples_by_strategy: dict[str, torch.Tensor] = {}
        for strategy in strategy_names:
            time_grid = self._time_grid(step_count, strategy, device=device, dtype=noise.dtype)
            samples = sample_with_ode(
                model=model,
                x_init=noise.clone(),
                step_count=step_count,
                method=solver_method,
                time_grid=time_grid,
            )
            samples_by_strategy[strategy] = to_unit_interval(samples).detach().cpu()

        panel = build_strategy_panel(
            noise=noise.detach().cpu(),
            samples_by_strategy=samples_by_strategy,
            strategy_names=strategy_names,
            include_noise=True,
        )
        panel_path = report_dir / f"strategy_panel_steps{step_count}.png"
        save_image(panel, panel_path, nrow=len(strategy_names) + 1, padding=2)
        payload = {
            "step_count": step_count,
            "fixed_seed": fixed_seed,
            "strategy_names": strategy_names,
            "noise": noise.detach().cpu(),
            "samples_by_strategy": samples_by_strategy,
        }
        torch.save(payload, report_dir / f"strategy_panel_steps{step_count}.pt")

    def _save_plot(self, records: list[dict], report_dir: Path) -> None:
        mpl_dir = report_dir / ".mplconfig"
        mpl_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
        plt.figure(figsize=(7, 4.5))
        strategy_names = sorted({str(row["strategy"]) for row in records})
        for strategy in strategy_names:
            subset = sorted(
                [row for row in records if row["strategy"] == strategy],
                key=lambda item: (item["nfe"], item["step_count"]),
            )
            plt.plot(
                [row["nfe"] for row in subset],
                [row["fid"] for row in subset],
                marker="o",
                label=strategy,
            )
        plt.xlabel("NFE")
        plt.ylabel("FID")
        plt.title("fm_timewarp_sampling Phase A")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(report_dir / "fid_vs_nfe.png", dpi=180)
        plt.close()

    def _write_reports(self, records: list[dict], report_dir: Path) -> None:
        with (report_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2)
        with (report_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

        by_strategy: dict[str, dict] = {}
        for row in records:
            current = by_strategy.get(row["strategy"])
            if current is None or float(row["fid"]) < float(current["fid"]):
                by_strategy[row["strategy"]] = row

        with (report_dir / "best_by_strategy.json").open("w", encoding="utf-8") as handle:
            json.dump(by_strategy, handle, indent=2)

        uniform_map = {int(row["step_count"]): float(row["fid"]) for row in records if row["strategy"] == "uniform"}
        compact_rows = []
        for row in sorted(records, key=lambda item: (item["strategy"], item["step_count"])):
            baseline_fid = uniform_map.get(int(row["step_count"]))
            delta_vs_uniform = None if baseline_fid is None else float(row["fid"]) - baseline_fid
            compact_rows.append(
                {
                    "strategy": row["strategy"],
                    "step_count": row["step_count"],
                    "nfe": row["nfe"],
                    "fid": row["fid"],
                    "delta_fid_vs_uniform": delta_vs_uniform,
                }
            )
        with (report_dir / "compact_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(compact_rows, handle, indent=2)
        with (report_dir / "compact_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(compact_rows[0].keys()))
            writer.writeheader()
            writer.writerows(compact_rows)

        best = min(records, key=lambda item: float(item["fid"]))
        with (report_dir / "best_overall.json").open("w", encoding="utf-8") as handle:
            json.dump(best, handle, indent=2)

    def _evaluate_strategy_step(
        self,
        *,
        model: torch.nn.Module,
        feature_extractor: torch.nn.Module,
        reference_stats,
        cache_path: Path,
        device: torch.device,
        strategy: str,
        step_count: int,
        eval_root: Path,
        fid_samples: int,
        fid_batch_size: int,
        fid_protocol: str,
        solver_method: str,
        channels: int,
        image_size: int,
        reference_count: int,
    ) -> dict:
        step_dir = eval_root / self._strategy_dir_name(strategy) / f"steps{step_count}"
        step_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        nfe = solver_nfe(step_count=step_count, method=solver_method)

        def sample_fn(batch_size: int) -> torch.Tensor:
            noise = torch.randn(batch_size, channels, image_size, image_size, device=device)
            time_grid = self._time_grid(step_count, strategy, device=device, dtype=noise.dtype)
            with torch.no_grad():
                samples = sample_with_ode(
                    model=model,
                    x_init=noise,
                    step_count=step_count,
                    method=solver_method,
                    time_grid=time_grid,
                )
            return to_unit_interval(samples)

        fake_stats = compute_generator_stats(
            sample_fn=sample_fn,
            feature_extractor=feature_extractor,
            batch_size=fid_batch_size,
            total_samples=fid_samples,
            device=device,
        )
        fid = frechet_distance(reference_stats, fake_stats)
        grid_meta = self._save_strategy_grid(
            model=model,
            device=device,
            step_count=step_count,
            strategy=strategy,
            step_dir=step_dir,
            solver_method=solver_method,
        )
        elapsed = time.time() - t0
        save_stats(step_dir / "generated_stats.npz", fake_stats)
        record = {
            "strategy": strategy,
            "step_count": step_count,
            "integration_steps": step_count,
            "nfe": nfe,
            "fid": fid,
            "fid_protocol": fid_protocol,
            "num_fid_samples": fid_samples,
            "fid_batch_size": fid_batch_size,
            "reference_count": reference_count,
            "reference_stats": str(cache_path),
            "checkpoint": str(self.checkpoint),
            "solver_method": solver_method,
            "elapsed_sec": elapsed,
            "samples_per_sec": fid_samples / max(elapsed, 1.0e-8),
            **grid_meta,
        }
        with (step_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
        print(
            f"timewarp_eval strategy={strategy} step_count={step_count} nfe={nfe} "
            f"fid={fid:.4f} elapsed_sec={elapsed:.2f}",
            flush=True,
        )
        return record

    def run(self, *, step_counts: list[int], strategy_names: list[str]) -> None:
        self.eval_root.mkdir(parents=True, exist_ok=True)
        report_dir = self.eval_root / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        device = device_from_config(self.config)
        model = load_model_from_checkpoint(self.config, self.checkpoint, device=device)
        feature_extractor = InceptionFeatureExtractor().to(device)
        reference_stats, cache_path = self._prepare_reference_stats(feature_extractor, device)

        eval_cfg = self._eval_cfg()
        fid_samples = int(eval_cfg.get("num_fid_samples", 50000))
        fid_batch_size = int(eval_cfg.get("fid_batch_size", 256))
        fid_protocol = str(eval_cfg.get("fid_protocol", "torch_fidelity_inceptionv3_2048"))
        solver_method = str(eval_cfg.get("solver_method", "heun2"))
        channels = int(self.config["dataset"]["channels"])
        image_size = int(self.config["dataset"]["image_size"])
        reference_count = int(reference_stats.count)

        records: list[dict] = []
        for strategy in strategy_names:
            for step_count in step_counts:
                record = self._evaluate_strategy_step(
                    model=model,
                    feature_extractor=feature_extractor,
                    reference_stats=reference_stats,
                    cache_path=cache_path,
                    device=device,
                    strategy=strategy,
                    step_count=step_count,
                    eval_root=self.eval_root,
                    fid_samples=fid_samples,
                    fid_batch_size=fid_batch_size,
                    fid_protocol=fid_protocol,
                    solver_method=solver_method,
                    channels=channels,
                    image_size=image_size,
                    reference_count=reference_count,
                )
                records.append(record)

        self._write_reports(records, report_dir)
        self._save_plot(records, report_dir)
        panel_step = int(self._tw_cfg().get("qualitative_panel_step", 16))
        if panel_step in step_counts:
            self._save_strategy_panel(
                model=model,
                device=device,
                strategy_names=strategy_names,
                step_count=panel_step,
                report_dir=report_dir,
                solver_method=solver_method,
            )
        best = min(records, key=lambda item: float(item["fid"]))
        print("dgfm timewarp sampling runner completed")
        print(f"eval_root: {self.eval_root}")
        print(f"best_strategy: {best['strategy']}")
        print(f"best_step: {best['step_count']}")
        print(f"best_fid: {best['fid']:.4f}")
