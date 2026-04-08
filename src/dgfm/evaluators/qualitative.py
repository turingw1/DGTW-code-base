from __future__ import annotations

from pathlib import Path

import torch
from torchvision.utils import save_image

from .common import sample_from_model_batched, to_unit_interval


def build_multistep_panel(
    noise: torch.Tensor,
    samples_by_step: dict[int, torch.Tensor],
    step_counts: list[int],
    include_noise: bool = True,
) -> torch.Tensor:
    rows = noise.shape[0]
    columns: list[torch.Tensor] = []
    if include_noise:
        columns.append(to_unit_interval(noise))
    for step_count in step_counts:
        columns.append(samples_by_step[step_count])

    panel_rows = []
    for row_idx in range(rows):
        for column in columns:
            panel_rows.append(column[row_idx])
    return torch.stack(panel_rows, dim=0)


def build_strategy_panel(
    noise: torch.Tensor,
    samples_by_strategy: dict[str, torch.Tensor],
    strategy_names: list[str],
    include_noise: bool = True,
) -> torch.Tensor:
    rows = noise.shape[0]
    columns: list[torch.Tensor] = []
    if include_noise:
        columns.append(to_unit_interval(noise))
    for name in strategy_names:
        columns.append(samples_by_strategy[name])

    panel_rows = []
    for row_idx in range(rows):
        for column in columns:
            panel_rows.append(column[row_idx])
    return torch.stack(panel_rows, dim=0)


@torch.no_grad()
def save_multistep_qualitative_panel(
    config: dict,
    model: torch.nn.Module,
    output_dir: str | Path,
    *,
    channels: int,
    image_size: int,
    step_counts: list[int],
    num_examples: int = 8,
    fixed_seed: int = 42,
    solver_method: str = "heun2",
    include_noise: bool = True,
    sample_batch_size: int = 0,
    device: torch.device,
    timewarp: torch.nn.Module | None = None,
    sample_extra: dict | None = None,
) -> dict[str, str | int | list[int]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device=device).manual_seed(fixed_seed)
    noise = torch.randn(
        num_examples,
        channels,
        image_size,
        image_size,
        generator=generator,
        device=device,
    )

    samples_by_step: dict[int, torch.Tensor] = {}
    for step_count in step_counts:
        samples = sample_from_model_batched(
            config=config,
            model=model,
            x_init=noise.clone(),
            step_count=step_count,
            method=solver_method,
            timewarp=timewarp,
            max_batch_size=sample_batch_size,
            move_to_cpu=True,
            extra=sample_extra,
        )
        samples_by_step[step_count] = to_unit_interval(samples)

    panel = build_multistep_panel(
        noise=noise.detach().cpu(),
        samples_by_step=samples_by_step,
        step_counts=step_counts,
        include_noise=include_noise,
    )
    columns = len(step_counts) + (1 if include_noise else 0)
    panel_path = output_dir / "multistep_panel.png"
    save_image(panel, panel_path, nrow=columns, padding=2)

    payload = {
        "step_counts": step_counts,
        "fixed_seed": fixed_seed,
        "include_noise": include_noise,
        "num_examples": num_examples,
        "solver_method": solver_method,
        "noise": noise.detach().cpu(),
        "samples_by_step": samples_by_step,
        "sample_extra": {key: value.detach().cpu() for key, value in (sample_extra or {}).items()},
    }
    payload_path = output_dir / "multistep_panel.pt"
    torch.save(payload, payload_path)

    return {
        "panel_path": str(panel_path),
        "payload_path": str(payload_path),
        "num_examples": num_examples,
        "step_counts": step_counts,
        "fixed_seed": fixed_seed,
    }
