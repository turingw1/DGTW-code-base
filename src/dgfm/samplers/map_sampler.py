from __future__ import annotations

import torch


def _normalize_time_grid(
    x_init: torch.Tensor,
    step_count: int,
    time_grid: torch.Tensor | None = None,
) -> torch.Tensor:
    if time_grid is None:
        return torch.linspace(0.0, 1.0, steps=step_count + 1, device=x_init.device, dtype=x_init.dtype)
    time_grid = time_grid.to(device=x_init.device, dtype=x_init.dtype)
    if time_grid.ndim != 1 or time_grid.shape[0] != step_count + 1:
        raise ValueError(f"time_grid must have shape ({step_count + 1},), got {tuple(time_grid.shape)}")
    return time_grid


def rollout_with_map(
    model: torch.nn.Module,
    x_init: torch.Tensor,
    step_count: int,
    time_grid: torch.Tensor | None = None,
    extra: dict | None = None,
) -> torch.Tensor:
    if step_count <= 0:
        raise ValueError(f"step_count must be positive, got {step_count}")
    time_grid = _normalize_time_grid(x_init, step_count, time_grid)
    x = x_init
    batch = x.shape[0]
    for idx in range(step_count):
        t = time_grid[idx].to(dtype=x.dtype).expand(batch)
        s = time_grid[idx + 1].to(dtype=x.dtype).expand(batch)
        x = model(x, t, s, extra=dict(extra or {}))
    return x


def rollout_trajectory_with_map(
    model: torch.nn.Module,
    x_init: torch.Tensor,
    step_count: int,
    time_grid: torch.Tensor | None = None,
    extra: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if step_count <= 0:
        raise ValueError(f"step_count must be positive, got {step_count}")
    time_grid = _normalize_time_grid(x_init, step_count, time_grid)
    x = x_init
    states = [x]
    batch = x.shape[0]
    for idx in range(step_count):
        t = time_grid[idx].to(dtype=x.dtype).expand(batch)
        s = time_grid[idx + 1].to(dtype=x.dtype).expand(batch)
        x = model(x, t, s, extra=dict(extra or {}))
        states.append(x)
    return torch.stack(states, dim=1), time_grid
