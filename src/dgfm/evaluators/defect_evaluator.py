from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch

from .common import device_from_config, load_model_from_checkpoint, load_timewarp_from_checkpoint, sample_condition_labels
from dgfm.samplers import rollout_trajectory_with_map
from dgfm.schedulers import build_runtime_time_grid


@dataclass(slots=True)
class HeldOutDefectReport:
    defect_mean: float
    defect_std: float
    defect_median: float
    defect_by_t_bin: dict[str, float]
    defect_by_step_count: dict[str, float]
    num_samples: int
    num_triplets: int
    checkpoint: str
    triplet_source: str


def _default_triplets(step_count: int) -> list[tuple[int, int, int]]:
    triplets: list[tuple[int, int, int]] = []
    for start in range(step_count - 2):
        for stop in range(start + 2, step_count + 1):
            mid = (start + stop) // 2
            if mid <= start:
                mid = start + 1
            if mid >= stop:
                mid = stop - 1
            triplets.append((start, mid, stop))
    return triplets


def _load_triplets(path: str | Path, *, grid_size: int) -> list[tuple[int, int, int]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    triplets: list[tuple[int, int, int]] = []
    for item in payload:
        if isinstance(item, dict):
            t_idx = int(item["t_idx"])
            s_idx = int(item["s_idx"])
            u_idx = int(item["u_idx"])
        else:
            t_idx, s_idx, u_idx = [int(value) for value in item]
        if not (0 <= t_idx < s_idx < u_idx <= grid_size):
            raise ValueError(f"Invalid triplet indices: {(t_idx, s_idx, u_idx)} for grid_size={grid_size}")
        triplets.append((t_idx, s_idx, u_idx))
    return triplets


def evaluate_held_out_defect(
    *,
    config: dict,
    checkpoint: str | Path,
    out_path: str | Path,
    num_samples: int,
    grid_steps: int,
    triplets_path: str | Path | None = None,
    seed: int = 42,
) -> HeldOutDefectReport:
    device = device_from_config(config)
    model = load_model_from_checkpoint(config, checkpoint, device=device)
    timewarp = load_timewarp_from_checkpoint(config, checkpoint, device=device)
    if str(config.get("train", {}).get("objective", "flow_matching_velocity")) not in {"explicit_map", "map_branch"}:
        raise ValueError("Held-out defect evaluator currently supports explicit_map checkpoints only")

    generator = torch.Generator(device=device).manual_seed(seed)
    channels = int(config["dataset"]["channels"])
    image_size = int(config["dataset"]["image_size"])
    x_init = torch.randn(num_samples, channels, image_size, image_size, generator=generator, device=device)
    labels = sample_condition_labels(config, num_samples, device=device, generator=generator)
    extra = {"label": labels} if labels is not None else None

    time_grid = build_runtime_time_grid(
        config=config,
        step_count=grid_steps,
        device=device,
        dtype=x_init.dtype,
        timewarp=timewarp,
    )
    states, _time_grid = rollout_trajectory_with_map(
        model=model,
        x_init=x_init,
        step_count=grid_steps,
        time_grid=time_grid,
        extra=extra,
    )
    triplets = _load_triplets(triplets_path, grid_size=grid_steps) if triplets_path else _default_triplets(grid_steps)
    defects: list[torch.Tensor] = []
    t_bin_values: dict[str, list[float]] = {}
    span_values: dict[str, list[float]] = {}

    with torch.no_grad():
        for t_idx, s_idx, u_idx in triplets:
            x_t = states[:, t_idx]
            t = time_grid[t_idx].expand(num_samples)
            s = time_grid[s_idx].expand(num_samples)
            u = time_grid[u_idx].expand(num_samples)
            model_extra = {"label": labels} if labels is not None else {}
            direct = model(x_t, t, u, extra=model_extra)
            via_mid = model(model(x_t, t, s, extra=model_extra), s, u, extra=model_extra)
            defect = ((direct - via_mid) ** 2).flatten(1).mean(dim=1)
            defects.append(defect)
            defect_mean = float(defect.mean().item())
            t_key = f"{float(time_grid[t_idx].item()):.4f}"
            span_key = str(int(u_idx - t_idx))
            t_bin_values.setdefault(t_key, []).append(defect_mean)
            span_values.setdefault(span_key, []).append(defect_mean)

    all_defects = torch.stack(defects, dim=1).reshape(-1)
    report = HeldOutDefectReport(
        defect_mean=float(all_defects.mean().item()),
        defect_std=float(all_defects.std(unbiased=False).item()),
        defect_median=float(all_defects.median().item()),
        defect_by_t_bin={key: float(sum(values) / len(values)) for key, values in t_bin_values.items()},
        defect_by_step_count={key: float(sum(values) / len(values)) for key, values in span_values.items()},
        num_samples=num_samples,
        num_triplets=len(triplets),
        checkpoint=str(Path(checkpoint)),
        triplet_source=str(Path(triplets_path)) if triplets_path else "default_dense_triplets",
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(report), handle, indent=2)
    return report
