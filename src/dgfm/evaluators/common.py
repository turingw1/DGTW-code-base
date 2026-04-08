from __future__ import annotations

from pathlib import Path

import torch

from dgfm.models import build_map_model, build_velocity_model
from dgfm.schedulers import TimeWarpMonotone, build_runtime_time_grid, build_timewarp_module
from dgfm.samplers import rollout_with_map


def objective_mode(config: dict) -> str:
    objective = str(config.get("train", {}).get("objective", "flow_matching_velocity"))
    if objective in {"flow_matching_velocity", "velocity_fm"}:
        return "velocity_fm"
    if objective in {"explicit_map", "map_branch"}:
        return "explicit_map"
    raise ValueError(f"Unsupported train.objective: {objective}")


@torch.no_grad()
def device_from_config(config: dict) -> torch.device:
    requested = config.get("runtime", {}).get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


@torch.no_grad()
def load_model_from_checkpoint(config: dict, checkpoint: str | Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(checkpoint, map_location=device)
    if objective_mode(config) == "explicit_map":
        model = build_map_model(config).to(device)
    else:
        model = build_velocity_model(config).to(device)
    state_dict = ckpt.get("ema_model") or ckpt["model"]
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def load_timewarp_from_checkpoint(
    config: dict,
    checkpoint: str | Path,
    device: torch.device,
) -> TimeWarpMonotone | None:
    module = build_timewarp_module(config, device=device, dtype=torch.float32)
    if module is None:
        return None
    ckpt = torch.load(checkpoint, map_location=device)
    state_dict = ckpt.get("timewarp")
    if state_dict is not None:
        module.load_state_dict(state_dict)
    module.eval()
    return module


def solver_nfe(step_count: int, method: str = "midpoint", mode: str = "velocity_fm") -> int:
    if step_count <= 0:
        raise ValueError(f"step_count must be positive, got {step_count}")
    if mode == "explicit_map":
        return step_count
    if method == "euler":
        return step_count
    if method in {"midpoint", "heun2"}:
        return 2 * step_count
    raise ValueError(f"Unsupported solver method: {method}")


@torch.no_grad()
def sample_with_ode(
    model: torch.nn.Module,
    x_init: torch.Tensor,
    step_count: int,
    method: str = "midpoint",
    time_grid: torch.Tensor | None = None,
    extra: dict | None = None,
) -> torch.Tensor:
    x = x_init
    if time_grid is None:
        time_grid = torch.linspace(0.0, 1.0, steps=step_count + 1, device=x_init.device, dtype=x_init.dtype)
    else:
        time_grid = time_grid.to(device=x_init.device, dtype=x_init.dtype)
        if time_grid.ndim != 1 or time_grid.shape[0] != step_count + 1:
            raise ValueError(f"time_grid must have shape ({step_count + 1},), got {tuple(time_grid.shape)}")
    batch = x.shape[0]
    for idx in range(step_count):
        t0 = time_grid[idx]
        t1 = time_grid[idx + 1]
        dt = t1 - t0
        t0_vec = t0.to(dtype=x.dtype).expand(batch)
        if method == "euler":
            k1 = model(x, t0_vec, extra=dict(extra or {}))
            x = x + dt * k1
            continue
        if method == "heun2":
            k1 = model(x, t0_vec, extra=dict(extra or {}))
            t1_vec = t1.to(dtype=x.dtype).expand(batch)
            k2 = model(x + dt * k1, t1_vec, extra=dict(extra or {}))
            x = x + 0.5 * dt * (k1 + k2)
            continue
        if method == "midpoint":
            k1 = model(x, t0_vec, extra=dict(extra or {}))
            tmid = t0 + 0.5 * dt
            tmid_vec = tmid.to(dtype=x.dtype).expand(batch)
            k2 = model(x + 0.5 * dt * k1, tmid_vec, extra=dict(extra or {}))
            x = x + dt * k2
            continue
        raise ValueError(f"Unsupported solver method: {method}")
    return x


def sample_from_model(
    config: dict,
    model: torch.nn.Module,
    x_init: torch.Tensor,
    step_count: int,
    method: str = "midpoint",
    time_grid: torch.Tensor | None = None,
    timewarp: TimeWarpMonotone | None = None,
    extra: dict | None = None,
) -> torch.Tensor:
    if time_grid is None:
        time_grid = build_runtime_time_grid(
            config=config,
            step_count=step_count,
            device=x_init.device,
            dtype=x_init.dtype,
            timewarp=timewarp,
        )
    mode = objective_mode(config)
    if mode == "explicit_map":
        return rollout_with_map(model=model, x_init=x_init, step_count=step_count, time_grid=time_grid, extra=extra)
    return sample_with_ode(model=model, x_init=x_init, step_count=step_count, method=method, time_grid=time_grid, extra=extra)


@torch.no_grad()
def sample_from_model_batched(
    config: dict,
    model: torch.nn.Module,
    x_init: torch.Tensor,
    step_count: int,
    *,
    method: str = "midpoint",
    time_grid: torch.Tensor | None = None,
    timewarp: TimeWarpMonotone | None = None,
    max_batch_size: int = 0,
    move_to_cpu: bool = False,
    extra: dict | None = None,
) -> torch.Tensor:
    batch_size = int(x_init.shape[0])
    if max_batch_size <= 0 or batch_size <= max_batch_size:
        out = sample_from_model(
            config=config,
            model=model,
            x_init=x_init,
            step_count=step_count,
            method=method,
            time_grid=time_grid,
            timewarp=timewarp,
            extra=extra,
        )
        return out.detach().cpu() if move_to_cpu else out

    outputs: list[torch.Tensor] = []
    for start in range(0, batch_size, max_batch_size):
        stop = min(batch_size, start + max_batch_size)
        out = sample_from_model(
            config=config,
            model=model,
            x_init=x_init[start:stop],
            step_count=step_count,
            method=method,
            time_grid=time_grid,
            timewarp=timewarp,
            extra={key: value[start:stop] for key, value in (extra or {}).items()},
        )
        outputs.append(out.detach().cpu() if move_to_cpu else out)
    return torch.cat(outputs, dim=0)


def sample_condition_labels(
    config: dict,
    batch_size: int,
    *,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor | None:
    num_classes = config.get("model", {}).get("num_classes")
    if num_classes is None:
        return None
    return torch.randint(int(num_classes), (batch_size,), device=device, generator=generator)


@torch.no_grad()
def to_unit_interval(images: torch.Tensor) -> torch.Tensor:
    return torch.clamp(images * 0.5 + 0.5, 0.0, 1.0)
