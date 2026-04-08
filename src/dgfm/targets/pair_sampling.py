from __future__ import annotations

import torch


def _sample_jump_delta(max_delta: int, target_cfg: dict, device: torch.device) -> int:
    short_max = min(max_delta, int(target_cfg.get("pair_short_max", 4)))
    mid_max = min(max_delta, int(target_cfg.get("pair_mid_max", 12)))
    long_max = min(max_delta, int(target_cfg.get("pair_long_max", 32)))
    choices: list[tuple[int, int, float]] = []
    if short_max >= 1:
        choices.append((1, short_max, float(target_cfg.get("pair_short_weight", 0.55))))
    if mid_max >= short_max + 1:
        choices.append((short_max + 1, mid_max, float(target_cfg.get("pair_mid_weight", 0.30))))
    if long_max >= mid_max + 1:
        choices.append((mid_max + 1, long_max, float(target_cfg.get("pair_long_weight", 0.15))))
    if not choices:
        return 1
    weights = torch.tensor([weight for _, _, weight in choices], dtype=torch.float32, device=device)
    weights = weights / weights.sum()
    bucket_idx = int(torch.multinomial(weights, 1).item())
    low, high, _ = choices[bucket_idx]
    return int(torch.randint(low, high + 1, (1,), device=device).item())


def _sample_pair_indices_legacy(
    num_points: int,
    target_cfg: dict,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if num_points < 2:
        raise ValueError(f"num_points must be at least 2, got {num_points}")
    endpoint_prob = float(target_cfg.get("pair_endpoint_weight", 0.35))
    high_noise_weight = float(target_cfg.get("high_noise_t_weight", 0.75))
    high_noise_fraction = float(target_cfg.get("high_noise_t_fraction", 0.35))
    max_delta = num_points - 1
    t_indices = torch.empty(batch_size, dtype=torch.long, device=device)
    s_indices = torch.empty(batch_size, dtype=torch.long, device=device)
    for idx in range(batch_size):
        if torch.rand(1, device=device).item() < endpoint_prob:
            s_index = num_points - 1
            max_start = max(1, s_index)
            high_noise_limit = max(1, int(round(max_start * high_noise_fraction)))
            if torch.rand(1, device=device).item() < high_noise_weight:
                t_index = int(torch.randint(0, high_noise_limit, (1,), device=device).item())
            else:
                t_index = int(torch.randint(0, max_start, (1,), device=device).item())
        else:
            delta = _sample_jump_delta(max_delta=max_delta, target_cfg=target_cfg, device=device)
            max_start = num_points - delta
            if max_start <= 1:
                t_index = 0
            else:
                high_noise_limit = max(1, int(round(max_start * high_noise_fraction)))
                if torch.rand(1, device=device).item() < high_noise_weight:
                    t_index = int(torch.randint(0, high_noise_limit, (1,), device=device).item())
                else:
                    t_index = int(torch.randint(0, max_start, (1,), device=device).item())
            s_index = t_index + delta
        t_indices[idx] = t_index
        s_indices[idx] = s_index
    return t_indices, s_indices


def _sample_num_heun_steps(batch_size: int, num_points: int, target_cfg: dict, device: torch.device) -> torch.Tensor:
    max_delta = num_points - 1
    max_heun_step = int(target_cfg.get("num_heun_step", max_delta))
    max_heun_step = max(1, min(max_heun_step, max_delta))
    random_heun = bool(target_cfg.get("num_heun_step_random", True))
    if not random_heun:
        return torch.full((batch_size,), max_heun_step, dtype=torch.long, device=device)

    strategy = str(target_cfg.get("heun_step_strategy", "weighted"))
    if strategy == "uniform":
        return torch.randint(1, max_heun_step + 1, (batch_size,), dtype=torch.long, device=device)
    if strategy == "weighted":
        multiplier = float(target_cfg.get("heun_step_multiplier", 1.0))
        weights = torch.arange(1, max_heun_step + 1, dtype=torch.float32, device=device) ** multiplier
        probs = weights / weights.sum()
        return torch.multinomial(probs, num_samples=batch_size, replacement=True).to(torch.long) + 1
    raise ValueError(f"Unsupported heun_step_strategy: {strategy}")


def _sample_pair_indices_ctm_discrete(
    num_points: int,
    target_cfg: dict,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if num_points < 2:
        raise ValueError(f"num_points must be at least 2, got {num_points}")
    num_heun_steps = _sample_num_heun_steps(
        batch_size=batch_size,
        num_points=num_points,
        target_cfg=target_cfg,
        device=device,
    )
    max_start = num_points - num_heun_steps
    t_indices = torch.floor(torch.rand(batch_size, device=device) * max_start.to(torch.float32)).to(torch.long)

    sample_s_strategy = str(target_cfg.get("sample_s_strategy", "uniform"))
    lower = t_indices + num_heun_steps
    if sample_s_strategy == "smallest":
        s_indices = torch.full((batch_size,), num_points - 1, dtype=torch.long, device=device)
        return t_indices, s_indices
    if sample_s_strategy == "uniform":
        span = num_points - lower
        offsets = torch.floor(torch.rand(batch_size, device=device) * span.to(torch.float32)).to(torch.long)
        s_indices = lower + offsets
        return t_indices, s_indices
    raise ValueError(f"Unsupported sample_s_strategy: {sample_s_strategy}")


def sample_target_triplet_indices(
    num_points: int,
    target_cfg: dict,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if num_points < 3:
        raise ValueError(f"num_points must be at least 3 for CTM-style triplets, got {num_points}")
    sampling_mode = str(target_cfg.get("sampling_mode", "ctm_discrete"))
    if sampling_mode != "ctm_discrete":
        t_indices, s_indices = sample_target_pair_indices(
            num_points=num_points,
            target_cfg=target_cfg,
            batch_size=batch_size,
            device=device,
        )
        midpoint = torch.clamp(t_indices + ((s_indices - t_indices) // 2), min=t_indices + 1, max=s_indices)
        return t_indices, midpoint, s_indices
    num_heun_steps = _sample_num_heun_steps(
        batch_size=batch_size,
        num_points=num_points,
        target_cfg=target_cfg,
        device=device,
    )
    max_start = num_points - num_heun_steps
    t_indices = torch.floor(torch.rand(batch_size, device=device) * max_start.to(torch.float32)).to(torch.long)
    t_dt_indices = t_indices + num_heun_steps

    sample_s_strategy = str(target_cfg.get("sample_s_strategy", "uniform"))
    if sample_s_strategy == "smallest":
        s_indices = torch.full((batch_size,), num_points - 1, dtype=torch.long, device=device)
        return t_indices, t_dt_indices, s_indices
    if sample_s_strategy == "uniform":
        span = num_points - t_dt_indices
        offsets = torch.floor(torch.rand(batch_size, device=device) * span.to(torch.float32)).to(torch.long)
        s_indices = t_dt_indices + offsets
        return t_indices, t_dt_indices, s_indices
    raise ValueError(f"Unsupported sample_s_strategy: {sample_s_strategy}")


def sample_target_pair_indices(
    num_points: int,
    target_cfg: dict,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    sampling_mode = str(target_cfg.get("sampling_mode", "ctm_discrete"))
    if sampling_mode == "ctm_discrete":
        return _sample_pair_indices_ctm_discrete(
            num_points=num_points,
            target_cfg=target_cfg,
            batch_size=batch_size,
            device=device,
        )
    if sampling_mode == "heuristic_pairs":
        return _sample_pair_indices_legacy(
            num_points=num_points,
            target_cfg=target_cfg,
            batch_size=batch_size,
            device=device,
        )
    raise ValueError(f"Unsupported target sampling_mode: {sampling_mode}")


def sample_pair_indices(num_points: int, target_cfg: dict, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return sample_target_pair_indices(
        num_points=num_points,
        target_cfg=target_cfg,
        batch_size=batch_size,
        device=device,
    )
