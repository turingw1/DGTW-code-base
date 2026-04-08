from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image

from .common import device_from_config, load_model_from_checkpoint
from .timewarp_sampling import TimewarpSamplingRunner
from .fid import InceptionFeatureExtractor
from .qualitative import build_strategy_panel


@dataclass(slots=True)
class TimewarpSearchRunner(TimewarpSamplingRunner):
    def _search_cfg(self) -> dict:
        return self.config.get("timewarp_search", {})

    def _candidate_gammas(self) -> list[float]:
        gammas = self._search_cfg().get("candidate_gammas", [1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0])
        return [float(item) for item in gammas]

    def _objective_steps(self) -> list[int]:
        return [int(item) for item in self._search_cfg().get("objective_steps", [8, 16])]

    def _objective_weights(self) -> list[float]:
        weights = [float(item) for item in self._search_cfg().get("objective_weights", [0.5, 0.5])]
        if len(weights) != len(self._objective_steps()):
            raise ValueError("objective_weights must match objective_steps length")
        return weights

    def _baseline_reference_strategies(self) -> list[str]:
        return [str(item) for item in self._search_cfg().get("reference_strategies", ["uniform", "data_dense_power2"])]

    def _search_family(self) -> str:
        return str(self._search_cfg().get("family", "data_dense_power"))

    def _strategy_from_gamma(self, gamma: float) -> str:
        return f"{self._search_family()}@{gamma:.4f}"

    def _search_dir(self) -> Path:
        return self.eval_root / "reports"

    def _detailed_log_path(self) -> Path:
        return self._search_dir() / "search_trace.jsonl"

    def _log_detailed(self, payload: dict) -> None:
        self._search_dir().mkdir(parents=True, exist_ok=True)
        with self._detailed_log_path().open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def _aggregate_gamma_rows(self, gamma_rows: list[dict]) -> dict:
        objective_steps = self._objective_steps()
        objective_weights = self._objective_weights()
        row_by_step = {int(row["step_count"]): row for row in gamma_rows}
        objective = 0.0
        for step_count, weight in zip(objective_steps, objective_weights):
            objective += float(weight) * float(row_by_step[step_count]["fid"])
        return {
            "gamma": float(gamma_rows[0]["gamma"]),
            "search_strategy": str(gamma_rows[0]["strategy"]),
            "objective": objective,
            "objective_steps": objective_steps,
            "objective_weights": objective_weights,
            "rows": gamma_rows,
        }

    def _save_objective_plots(self, gamma_records: list[dict], report_dir: Path) -> None:
        mpl_dir = report_dir / ".mplconfig"
        mpl_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

        gammas = [float(item["gamma"]) for item in gamma_records]
        objectives = [float(item["objective"]) for item in gamma_records]
        plt.figure(figsize=(6.5, 4.0))
        plt.plot(gammas, objectives, marker="o")
        plt.xlabel("gamma")
        plt.ylabel("weighted few-step FID objective")
        plt.title("fm_timewarp_sampling Phase B1 objective")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(report_dir / "objective_vs_gamma.png", dpi=180)
        plt.close()

        objective_steps = self._objective_steps()
        plt.figure(figsize=(6.5, 4.0))
        for step_count in objective_steps:
            fid_values = []
            for item in gamma_records:
                row_by_step = {int(row["step_count"]): row for row in item["rows"]}
                fid_values.append(float(row_by_step[step_count]["fid"]))
            plt.plot(gammas, fid_values, marker="o", label=f"step={step_count}")
        plt.xlabel("gamma")
        plt.ylabel("FID")
        plt.title("few-step FID vs gamma")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(report_dir / "fid_vs_gamma.png", dpi=180)
        plt.close()

    def _save_best_gamma_panel(
        self,
        *,
        model: torch.nn.Module,
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
            samples = self._sample_for_panel(model, noise, step_count, solver_method, time_grid)
            samples_by_strategy[strategy] = samples
        panel = build_strategy_panel(
            noise=noise.detach().cpu(),
            samples_by_strategy=samples_by_strategy,
            strategy_names=strategy_names,
            include_noise=True,
        )
        save_image(panel, report_dir / f"best_gamma_panel_steps{step_count}.png", nrow=len(strategy_names) + 1, padding=2)
        torch.save(
            {
                "step_count": step_count,
                "strategy_names": strategy_names,
                "fixed_seed": fixed_seed,
                "noise": noise.detach().cpu(),
                "samples_by_strategy": samples_by_strategy,
            },
            report_dir / f"best_gamma_panel_steps{step_count}.pt",
        )

    def _sample_for_panel(
        self,
        model: torch.nn.Module,
        noise: torch.Tensor,
        step_count: int,
        solver_method: str,
        time_grid: torch.Tensor,
    ) -> torch.Tensor:
        from .common import sample_with_ode, to_unit_interval

        samples = sample_with_ode(
            model=model,
            x_init=noise.clone(),
            step_count=step_count,
            method=solver_method,
            time_grid=time_grid,
        )
        return to_unit_interval(samples).detach().cpu()

    def run(self, *, step_counts: list[int]) -> None:
        self.eval_root.mkdir(parents=True, exist_ok=True)
        report_dir = self.eval_root / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        search_dir = self._search_dir()
        search_dir.mkdir(parents=True, exist_ok=True)
        if self._detailed_log_path().exists():
            self._detailed_log_path().unlink()

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

        reference_records: list[dict] = []
        for strategy in self._baseline_reference_strategies():
            for step_count in step_counts:
                row = self._evaluate_strategy_step(
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
                row["group"] = "reference"
                reference_records.append(row)
                self._log_detailed(row)

        gamma_records: list[dict] = []
        candidate_rows: list[dict] = []
        for gamma in self._candidate_gammas():
            strategy = self._strategy_from_gamma(gamma)
            gamma_rows = []
            for step_count in step_counts:
                row = self._evaluate_strategy_step(
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
                row["gamma"] = gamma
                row["group"] = "candidate"
                gamma_rows.append(row)
                candidate_rows.append(row)
                self._log_detailed(row)
            gamma_record = self._aggregate_gamma_rows(gamma_rows)
            gamma_records.append(gamma_record)
            self._log_detailed(
                {
                    "event": "objective",
                    "gamma": gamma_record["gamma"],
                    "search_strategy": gamma_record["search_strategy"],
                    "objective": gamma_record["objective"],
                    "objective_steps": gamma_record["objective_steps"],
                    "objective_weights": gamma_record["objective_weights"],
                }
            )
            print(
                f"timewarp_search gamma={gamma_record['gamma']:.4f} objective={gamma_record['objective']:.4f}",
                flush=True,
            )

        best_gamma_record = min(gamma_records, key=lambda item: float(item["objective"]))
        best_strategy = str(best_gamma_record["search_strategy"])

        with (report_dir / "gamma_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(gamma_records, handle, indent=2)
        with (report_dir / "gamma_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["gamma", "search_strategy", "objective", "objective_steps", "objective_weights"])
            writer.writeheader()
            for item in gamma_records:
                writer.writerow(
                    {
                        "gamma": item["gamma"],
                        "search_strategy": item["search_strategy"],
                        "objective": item["objective"],
                        "objective_steps": json.dumps(item["objective_steps"]),
                        "objective_weights": json.dumps(item["objective_weights"]),
                    }
                )

        uniform_map = {
            int(row["step_count"]): float(row["fid"])
            for row in reference_records
            if row["strategy"] == "uniform"
        }
        best_map = {
            int(row["step_count"]): float(row["fid"])
            for row in best_gamma_record["rows"]
        }
        compact_rows = []
        for step_count in sorted(best_map):
            compact_rows.append(
                {
                    "step_count": step_count,
                    "nfe": int([row for row in best_gamma_record["rows"] if int(row["step_count"]) == step_count][0]["nfe"]),
                    "uniform_fid": uniform_map.get(step_count),
                    "best_gamma_fid": best_map[step_count],
                    "delta_fid_vs_uniform": best_map[step_count] - uniform_map.get(step_count, best_map[step_count]),
                    "best_gamma": best_gamma_record["gamma"],
                    "best_strategy": best_strategy,
                }
            )
        with (report_dir / "compact_search_results.json").open("w", encoding="utf-8") as handle:
            json.dump(compact_rows, handle, indent=2)
        with (report_dir / "compact_search_results.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(compact_rows[0].keys()))
            writer.writeheader()
            writer.writerows(compact_rows)

        with (report_dir / "best_gamma.json").open("w", encoding="utf-8") as handle:
            json.dump(best_gamma_record, handle, indent=2)

        self._save_objective_plots(gamma_records, report_dir)

        panel_steps = [int(item) for item in self._search_cfg().get("panel_steps", self._objective_steps())]
        compare_strategies = ["uniform", "data_dense_power2", best_strategy]
        for step_count in panel_steps:
            if step_count in step_counts:
                self._save_best_gamma_panel(
                    model=model,
                    device=device,
                    strategy_names=compare_strategies,
                    step_count=step_count,
                    report_dir=report_dir,
                    solver_method=solver_method,
                )

        print("dgfm timewarp Phase B1 search completed")
        print(f"eval_root: {self.eval_root}")
        print(f"best_gamma: {best_gamma_record['gamma']}")
        print(f"best_strategy: {best_strategy}")
        print(f"best_objective: {best_gamma_record['objective']:.4f}")
