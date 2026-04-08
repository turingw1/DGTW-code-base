from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch

from dgfm.schedulers import TimeWarpMonotone, build_runtime_time_grid
from dgfm.targets.pair_sampling import sample_target_pair_indices, sample_target_triplet_indices
from dgfm.teachers import build_teacher


@dataclass(slots=True)
class TargetBatch:
    x_t: torch.Tensor
    x_s_target: torch.Tensor
    t: torch.Tensor
    s: torch.Tensor
    x_0: torch.Tensor
    x_1: torch.Tensor
    x_t_dt: torch.Tensor | None = None
    t_dt: torch.Tensor | None = None
    target_construction: str = "trajectory_regression"
    target_source: str = "teacher"
    target_stop_grad: bool = True
    bridge_source: str = "teacher"
    labels: torch.Tensor | None = None


def _skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    p_mean = -1.2
    p_std = 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma = (rnd_normal * p_std + p_mean).exp()
    time = 1.0 / (1.0 + sigma)
    return torch.clamp(time, min=1.0e-4, max=1.0)


class AnalyticPathTargetBuilder:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.target_cfg = config.get("target", {})
        self.train_cfg = config.get("train", {})

    def sample_times(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        min_time = float(self.target_cfg.get("min_time", 1.0e-4))
        max_time = float(self.target_cfg.get("max_time", 0.999))
        min_gap = float(self.target_cfg.get("min_time_gap", 0.05))
        use_skewed = bool(self.train_cfg.get("skewed_timesteps", True))
        if use_skewed:
            s = _skewed_timestep_sample(batch_size, device=device)
        else:
            s = torch.rand(batch_size, device=device)
        s = torch.clamp(s, min=min_time + min_gap, max=max_time)
        fraction = torch.rand(batch_size, device=device)
        slack = torch.clamp(s - min_time - min_gap, min=0.0)
        t = s - min_gap - fraction * slack
        t = torch.maximum(t, torch.full_like(t, min_time))
        t = torch.minimum(t, s - min_gap)
        return t, s

    loader_mode = "images"
    needs_path = True
    uses_dataset_images = True

    def build(self, x_0: torch.Tensor, x_1: torch.Tensor, path) -> TargetBatch:
        t, s = self.sample_times(batch_size=x_0.shape[0], device=x_0.device)
        path_sample_t = path.sample(x_0=x_0, x_1=x_1, t=t)
        path_sample_s = path.sample(x_0=x_0, x_1=x_1, t=s)
        return TargetBatch(
            x_t=path_sample_t.x_t,
            x_s_target=path_sample_s.x_t,
            t=t,
            s=s,
            x_0=x_0,
            x_1=x_1,
        )

    def build_from_batch(self, batch, *, device: torch.device, path, model=None, target_model=None) -> TargetBatch:
        del model, target_model
        images, labels = batch
        x_1 = images.to(device) * 2.0 - 1.0
        x_0 = torch.randn_like(x_1)
        target_batch = self.build(x_0=x_0, x_1=x_1, path=path)
        target_batch.labels = labels.to(device) if labels is not None else None
        return target_batch


class TrajectoryShardTargetBuilder:
    loader_mode = "trajectory_shard"
    needs_path = False
    uses_dataset_images = True

    def __init__(self, config: dict) -> None:
        self.config = config

    def build_from_batch(self, batch, *, device: torch.device, path=None, model=None, target_model=None) -> TargetBatch:
        del path, model, target_model
        x_t = batch["x_t"].to(device)
        x_s = batch["x_s"].to(device)
        t = batch["t"].to(device)
        s = batch["s"].to(device)
        return TargetBatch(
            x_t=x_t,
            x_s_target=x_s,
            t=t,
            s=s,
            x_0=batch["x_0"].to(device),
            x_1=batch["x_1"].to(device),
            labels=batch.get("labels").to(device) if batch.get("labels") is not None else None,
        )


class TeacherSamplerTargetBuilder:
    loader_mode = "images"
    needs_path = False
    uses_dataset_images = False

    def __init__(self, config: dict) -> None:
        self.config = config
        self.target_cfg = config.get("target", {})
        self.teacher_cfg = config.get("teacher", {})
        self.teacher = build_teacher(config)
        start_scales = int(self.target_cfg.get("start_scales", self.teacher_cfg.get("retain_num_points", 33)))
        self.start_scales = start_scales
        self.timewarp: TimeWarpMonotone | None = None
        self.target_construction = str(self.target_cfg.get("target_construction", "ctm_consistency"))
        self.target_source = str(self.target_cfg.get("target_source", "ema_model"))
        self.target_stop_grad = bool(self.target_cfg.get("target_stop_grad", True))
        self.bridge_source = str(self.target_cfg.get("bridge_source", "ema_model_rollout"))
        self.bridge_steps = max(1, int(self.target_cfg.get("bridge_steps", 1)))
        self.bridge_stop_grad = bool(self.target_cfg.get("bridge_stop_grad", True))

    def set_timewarp(self, timewarp: TimeWarpMonotone | None) -> None:
        self.timewarp = timewarp

    def current_u_grid(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return build_runtime_time_grid(
            config=self.config,
            step_count=self.start_scales - 1,
            device=device,
            dtype=dtype,
            timewarp=self.timewarp,
        )

    def _batch_size_from_batch(self, batch) -> int:
        if isinstance(batch, (tuple, list)):
            return int(batch[0].shape[0])
        if isinstance(batch, dict):
            for value in batch.values():
                if isinstance(value, torch.Tensor):
                    return int(value.shape[0])
        raise TypeError("Unsupported batch type for teacher_sampler target builder")

    def _sample_online_teacher_pairs(self, batch_size: int, device: torch.device) -> TargetBatch:
        self.teacher.prepare(device)
        sub_batch = int(self.target_cfg.get("teacher_sampler_sub_batch", 0) or 0)
        if sub_batch <= 0:
            sub_batch = batch_size
        all_x_t = []
        all_x_t_dt = []
        all_x_s = []
        all_t = []
        all_t_dt = []
        all_s = []
        all_x_0 = []
        all_x_1 = []
        for start in range(0, batch_size, sub_batch):
            chunk = min(sub_batch, batch_size - start)
            x_0 = self.teacher.sample_x0(batch_size=chunk, device=device)
            u_grid = self.current_u_grid(device=device, dtype=torch.float32)
            teacher_batch = self.teacher.sample_trajectory_from_x0(x_0=x_0, u_grid=u_grid, device=device)
            t_grid = teacher_batch.t_grid.to(device=device, dtype=torch.float32)
            x_grid = teacher_batch.x_grid.to(device=device, dtype=torch.float32)
            if self.target_construction == "ctm_consistency":
                t_indices, t_dt_indices, s_indices = sample_target_triplet_indices(
                    num_points=t_grid.shape[0],
                    target_cfg=self.target_cfg,
                    batch_size=chunk,
                    device=device,
                )
            else:
                t_indices, s_indices = sample_target_pair_indices(
                    num_points=t_grid.shape[0],
                    target_cfg=self.target_cfg,
                    batch_size=chunk,
                    device=device,
                )
                t_dt_indices = torch.clamp(t_indices + 1, max=s_indices)
            batch_indices = torch.arange(chunk, device=device)
            all_x_t.append(x_grid[batch_indices, t_indices])
            all_x_t_dt.append(x_grid[batch_indices, t_dt_indices])
            all_x_s.append(x_grid[batch_indices, s_indices])
            all_t.append(t_grid[t_indices])
            all_t_dt.append(t_grid[t_dt_indices])
            all_s.append(t_grid[s_indices])
            all_x_0.append(x_grid[:, 0])
            all_x_1.append(x_grid[:, -1])
        return TargetBatch(
            x_t=torch.cat(all_x_t, dim=0),
            x_s_target=torch.cat(all_x_s, dim=0),
            t=torch.cat(all_t, dim=0),
            t_dt=torch.cat(all_t_dt, dim=0),
            s=torch.cat(all_s, dim=0),
            x_0=torch.cat(all_x_0, dim=0),
            x_1=torch.cat(all_x_1, dim=0),
            x_t_dt=torch.cat(all_x_t_dt, dim=0),
            target_construction=self.target_construction,
            target_source=self.target_source,
            target_stop_grad=self.target_stop_grad,
            bridge_source="teacher",
        )

    def _resolve_bridge_model(self, *, model, target_model):
        if self.bridge_source == "teacher":
            return None
        if self.bridge_source == "ema_model_rollout":
            return target_model if target_model is not None else model
        if self.bridge_source == "current_model_rollout":
            return model
        raise ValueError(f"Unsupported bridge_source: {self.bridge_source}")

    def _rollout_bridge_state(
        self,
        *,
        source_model,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_dt: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        grad_ctx = torch.no_grad() if self.bridge_stop_grad else nullcontext()
        with grad_ctx:
            current = x_t
            for step_idx in range(self.bridge_steps):
                alpha0 = float(step_idx) / float(self.bridge_steps)
                alpha1 = float(step_idx + 1) / float(self.bridge_steps)
                step_t = t + (t_dt - t) * alpha0
                step_s = t + (t_dt - t) * alpha1
                extra = {"label": labels} if labels is not None else {}
                current = source_model(current, step_t, step_s, extra=extra)
        return current.detach() if self.bridge_stop_grad else current

    def build_from_batch(self, batch, *, device: torch.device, path=None, model=None, target_model=None) -> TargetBatch:
        del path
        batch_size = self._batch_size_from_batch(batch)
        target_batch = self._sample_online_teacher_pairs(batch_size=batch_size, device=device)
        if isinstance(batch, (tuple, list)) and len(batch) > 1 and isinstance(batch[1], torch.Tensor):
            target_batch.labels = batch[1].to(device)
        if self.target_construction != "ctm_consistency":
            return target_batch
        bridge_model = self._resolve_bridge_model(model=model, target_model=target_model)
        if bridge_model is None or target_batch.x_t_dt is None or target_batch.t_dt is None:
            return target_batch
        target_batch.x_t_dt = self._rollout_bridge_state(
            source_model=bridge_model,
            x_t=target_batch.x_t,
            t=target_batch.t,
            t_dt=target_batch.t_dt,
            labels=target_batch.labels,
        )
        target_batch.bridge_source = self.bridge_source
        return target_batch


def build_target_builder(config: dict):
    target_cfg = config.get("target", {})
    mode = str(target_cfg.get("builder", "analytic_path"))
    if mode == "analytic_path":
        return AnalyticPathTargetBuilder(config)
    if mode == "trajectory_shard":
        return TrajectoryShardTargetBuilder(config)
    if mode == "teacher_sampler":
        return TeacherSamplerTargetBuilder(config)
    raise ValueError(f"Unsupported target builder mode: {mode}")
