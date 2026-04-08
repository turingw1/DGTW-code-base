from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


DEFAULT_STRATEGIES = (
    "uniform",
    "source_dense_power2",
    "data_dense_power2",
    "random_dirichlet",
    "spline_mass",
)


def list_timewarp_strategies() -> tuple[str, ...]:
    return DEFAULT_STRATEGIES


def _validate_time_grid(time_grid: torch.Tensor, step_count: int) -> torch.Tensor:
    if time_grid.ndim != 1:
        raise ValueError(f"time_grid must be 1D, got shape={tuple(time_grid.shape)}")
    if time_grid.shape[0] != step_count + 1:
        raise ValueError(f"time_grid length must be {step_count + 1}, got {time_grid.shape[0]}")
    if not torch.all(time_grid[1:] >= time_grid[:-1]):
        raise ValueError("time_grid must be monotone non-decreasing")
    if abs(float(time_grid[0].item())) > 1.0e-8 or abs(float(time_grid[-1].item()) - 1.0) > 1.0e-8:
        raise ValueError("time_grid must start at 0 and end at 1")
    return time_grid


class TimeWarpMonotone(nn.Module):
    """Piecewise-linear monotone time warp on `[0, 1]`."""

    def __init__(self, num_bins: int = 64, init_bias: float = 0.0) -> None:
        super().__init__()
        if num_bins < 4:
            raise ValueError("num_bins must be at least 4")
        self.num_bins = int(num_bins)
        self.logits = nn.Parameter(torch.full((self.num_bins,), float(init_bias)))

    def _bin_heights(self) -> Tensor:
        heights = F.softplus(self.logits) + 1.0e-4
        return heights / heights.sum()

    def _cdf(self) -> Tensor:
        heights = self._bin_heights()
        cdf = torch.cat(
            [
                torch.zeros(1, device=heights.device, dtype=heights.dtype),
                torch.cumsum(heights, dim=0),
            ],
            dim=0,
        )
        cdf[-1] = 1.0
        return cdf

    def _grid(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.linspace(0.0, 1.0, steps=self.num_bins + 1, device=device, dtype=dtype)

    def forward(self, t: Tensor) -> Tensor:
        t = t.clamp(0.0, 1.0)
        cdf = self._cdf().to(device=t.device, dtype=t.dtype)
        flat_t = t.reshape(-1)
        scaled = flat_t * self.num_bins
        indices = torch.clamp(scaled.floor().long(), max=self.num_bins - 1)
        alpha = scaled - indices.to(dtype=t.dtype)
        u0 = cdf[indices]
        u1 = cdf[indices + 1]
        warped = u0 + alpha * (u1 - u0)
        return warped.reshape_as(t)

    @torch.no_grad()
    def inverse(self, u: Tensor) -> Tensor:
        u = u.clamp(0.0, 1.0)
        cdf = self._cdf().to(device=u.device, dtype=u.dtype)
        grid = self._grid(u.device, u.dtype)
        flat_u = u.reshape(-1)
        indices = torch.searchsorted(cdf, flat_u, right=True) - 1
        indices = indices.clamp(0, self.num_bins - 1)
        u0 = cdf[indices]
        u1 = cdf[indices + 1]
        t0 = grid[indices]
        t1 = grid[indices + 1]
        alpha = (flat_u - u0) / torch.clamp(u1 - u0, min=1.0e-6)
        restored = t0 + alpha * (t1 - t0)
        return restored.reshape_as(u)

    def derivative(self, t: Tensor) -> Tensor:
        t = t.clamp(0.0, 1.0)
        heights = self._bin_heights().to(device=t.device, dtype=t.dtype)
        flat_t = t.reshape(-1)
        scaled = flat_t * self.num_bins
        indices = torch.clamp(scaled.floor().long(), max=self.num_bins - 1)
        derivative = heights[indices] * float(self.num_bins)
        return derivative.reshape_as(t)

    def warp_grid(self, t_grid: Tensor) -> Tensor:
        return self.forward(t_grid)

    @torch.no_grad()
    def grid_cache(self) -> tuple[Tensor, Tensor]:
        t_grid = self._grid(self.logits.device, self.logits.dtype)
        u_grid = self._cdf().to(dtype=t_grid.dtype)
        return t_grid, u_grid


class SplineWarp(TimeWarpMonotone):
    """Minimum-viable monotone spline warp with fixed knots and learnable mass."""

    pass


def _uniform_grid(step_count: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.linspace(0.0, 1.0, steps=step_count + 1, device=device, dtype=dtype)


def _source_dense_power_grid(step_count: int, *, gamma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    base = _uniform_grid(step_count, device=device, dtype=dtype)
    return base.pow(gamma)


def _data_dense_power_grid(step_count: int, *, gamma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    base = _uniform_grid(step_count, device=device, dtype=dtype)
    return 1.0 - (1.0 - base).pow(gamma)


def _random_dirichlet_grid(
    step_count: int,
    *,
    concentration: float,
    random_seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if concentration <= 0.0:
        raise ValueError(f"concentration must be positive, got {concentration}")
    rng = np.random.default_rng(random_seed)
    increments_np = rng.dirichlet(np.full(step_count, float(concentration), dtype=np.float64))
    increments = torch.tensor(increments_np, device=device, dtype=dtype)
    time_grid = torch.cat(
        [
            torch.zeros(1, device=device, dtype=dtype),
            torch.cumsum(increments, dim=0),
        ],
        dim=0,
    )
    time_grid[-1] = torch.tensor(1.0, device=device, dtype=dtype)
    return time_grid


def build_time_grid(
    step_count: int,
    strategy: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    power_gamma: float = 2.0,
    random_concentration: float = 1.0,
    random_seed: int = 123,
) -> torch.Tensor:
    if step_count <= 0:
        raise ValueError(f"step_count must be positive, got {step_count}")

    strategy_name = strategy
    gamma_override: float | None = None
    if "@" in strategy_name:
        strategy_name, raw_gamma = strategy_name.split("@", 1)
        gamma_override = float(raw_gamma)
    gamma = power_gamma if gamma_override is None else gamma_override

    if strategy_name == "uniform":
        return _validate_time_grid(_uniform_grid(step_count, device=device, dtype=dtype), step_count)
    if strategy_name == "source_dense_power2":
        return _validate_time_grid(
            _source_dense_power_grid(step_count, gamma=2.0, device=device, dtype=dtype),
            step_count,
        )
    if strategy_name == "data_dense_power2":
        return _validate_time_grid(
            _data_dense_power_grid(step_count, gamma=2.0, device=device, dtype=dtype),
            step_count,
        )
    if strategy_name == "source_dense_power":
        return _validate_time_grid(
            _source_dense_power_grid(step_count, gamma=gamma, device=device, dtype=dtype),
            step_count,
        )
    if strategy_name == "data_dense_power":
        return _validate_time_grid(
            _data_dense_power_grid(step_count, gamma=gamma, device=device, dtype=dtype),
            step_count,
        )
    if strategy_name == "random_dirichlet":
        return _validate_time_grid(
            _random_dirichlet_grid(
                step_count,
                concentration=random_concentration,
                random_seed=random_seed,
                device=device,
                dtype=dtype,
            ),
            step_count,
        )
    raise ValueError(f"Unsupported timewarp strategy: {strategy}")


def _coerce_logits(logits: Iterable[float] | None, *, num_bins: int, device: torch.device, dtype: torch.dtype) -> Tensor | None:
    if logits is None:
        return None
    values = list(logits)
    if len(values) != num_bins:
        raise ValueError(f"timewarp logits length must be {num_bins}, got {len(values)}")
    return torch.tensor(values, device=device, dtype=dtype)


def _timewarp_cfg(config: dict) -> dict:
    return config.get("scheduler", {}).get("timewarp", {})


def build_timewarp_module(config: dict, *, device: torch.device, dtype: torch.dtype) -> TimeWarpMonotone | None:
    timewarp_cfg = _timewarp_cfg(config)
    if not bool(timewarp_cfg.get("enabled", False)):
        return None
    warp_type = str(timewarp_cfg.get("type", "learnable_monotone"))
    if warp_type not in {"learnable_monotone", "spline_mass"}:
        return None
    num_bins = int(timewarp_cfg.get("num_bins", 64))
    init_bias = float(timewarp_cfg.get("init_bias", 0.0))
    module_cls = SplineWarp if warp_type == "spline_mass" else TimeWarpMonotone
    module = module_cls(num_bins=num_bins, init_bias=init_bias).to(device=device, dtype=dtype)
    logits_tensor = _coerce_logits(timewarp_cfg.get("logits"), num_bins=num_bins, device=device, dtype=dtype)
    if logits_tensor is not None:
        module.logits.data.copy_(logits_tensor)
    return module


def apply_time_warp(
    t_linear: Tensor,
    *,
    warp_type: str,
    num_bins: int = 64,
    init_bias: float = 0.0,
    logits: Iterable[float] | None = None,
) -> Tensor:
    if warp_type == "identity":
        return t_linear
    if warp_type == "learnable_monotone":
        module = TimeWarpMonotone(num_bins=num_bins, init_bias=init_bias).to(device=t_linear.device, dtype=t_linear.dtype)
        logits_tensor = _coerce_logits(logits, num_bins=num_bins, device=t_linear.device, dtype=t_linear.dtype)
        if logits_tensor is not None:
            module.logits.data.copy_(logits_tensor)
        return module(t_linear)
    if warp_type == "spline_mass":
        module = SplineWarp(num_bins=num_bins, init_bias=init_bias).to(device=t_linear.device, dtype=t_linear.dtype)
        logits_tensor = _coerce_logits(logits, num_bins=num_bins, device=t_linear.device, dtype=t_linear.dtype)
        if logits_tensor is not None:
            module.logits.data.copy_(logits_tensor)
        return module(t_linear)
    return build_time_grid(
        int(t_linear.shape[0] - 1),
        warp_type,
        device=t_linear.device,
        dtype=t_linear.dtype,
    )


def build_config_time_grid(
    config: dict,
    step_count: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    linear = _uniform_grid(step_count, device=device, dtype=dtype)
    scheduler_cfg = config.get("scheduler", {})
    timewarp_cfg = scheduler_cfg.get("timewarp", {})
    if not bool(timewarp_cfg.get("enabled", False)):
        return _validate_time_grid(linear, step_count)
    warp_type = str(timewarp_cfg.get("type", "learnable_monotone"))
    warped = apply_time_warp(
        linear,
        warp_type=warp_type,
        num_bins=int(timewarp_cfg.get("num_bins", max(16, step_count * 4))),
        init_bias=float(timewarp_cfg.get("init_bias", 0.0)),
        logits=timewarp_cfg.get("logits"),
    )
    warped = warped.to(device=device, dtype=dtype)
    warped[0] = torch.tensor(0.0, device=device, dtype=dtype)
    warped[-1] = torch.tensor(1.0, device=device, dtype=dtype)
    return _validate_time_grid(warped, step_count)


def build_runtime_time_grid(
    config: dict,
    step_count: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    timewarp: TimeWarpMonotone | None = None,
) -> Tensor:
    if timewarp is None:
        return build_config_time_grid(config=config, step_count=step_count, device=device, dtype=dtype)
    linear = _uniform_grid(step_count, device=device, dtype=dtype)
    warped = timewarp(linear).to(device=device, dtype=dtype)
    warped[0] = torch.tensor(0.0, device=device, dtype=dtype)
    warped[-1] = torch.tensor(1.0, device=device, dtype=dtype)
    return _validate_time_grid(warped, step_count)


@torch.no_grad()
def summarize_time_grid(time_grid: Tensor) -> dict[str, float | list[float]]:
    deltas = time_grid[1:] - time_grid[:-1]
    return {
        "time_grid": [float(item) for item in time_grid.detach().cpu().tolist()],
        "delta_min": float(deltas.min().item()),
        "delta_max": float(deltas.max().item()),
        "delta_mean": float(deltas.mean().item()),
        "delta_std": float(deltas.std(unbiased=False).item()) if deltas.numel() > 1 else 0.0,
    }
