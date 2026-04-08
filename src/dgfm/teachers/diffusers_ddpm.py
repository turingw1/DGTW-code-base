from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(slots=True)
class TeacherTrajectoryBatch:
    t_grid: Tensor
    x_grid: Tensor


class DiffusersDDPMTeacher:
    """Diffusers DDPM teacher adapted to dgfm map-branch time semantics.

    Internal teacher time uses tau in [0, 1]:
    - tau=1.0: noisiest state
    - tau=0.0: cleanest state

    dgfm map branch keeps u in [0, 1]:
    - u=0.0: noisiest state
    - u=1.0: cleanest state

    We therefore retain trajectories on u-grid and map them internally via
    tau = 1 - u.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.teacher_cfg = config.get("teacher", {})
        self.dataset_cfg = config.get("dataset", {})
        self.pipeline = None
        self.unet = None
        self.scheduler = None
        self._loaded_device: str | None = None

    def is_enabled(self) -> bool:
        return True

    def _require_name_or_path(self) -> str:
        name_or_path = self.teacher_cfg.get("name_or_path")
        if not name_or_path:
            raise ValueError("teacher.name_or_path must be set for diffusers_ddpm teacher rollout")
        return str(name_or_path)

    def prepare(self, device: torch.device | str) -> None:
        device = torch.device(device)
        if self.pipeline is not None and self._loaded_device == str(device):
            return
        try:
            from diffusers import DDPMPipeline
            from diffusers.schedulers import DDIMScheduler
        except ImportError as exc:
            raise ImportError(
                "Diffusers teacher requires optional dependencies. "
                "Install project extras or use scripts/experiments/create_map_branch_env.sh."
            ) from exc

        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.pipeline = DDPMPipeline.from_pretrained(
            self._require_name_or_path(),
            local_files_only=bool(self.teacher_cfg.get("local_files_only", True)),
            torch_dtype=torch_dtype,
        ).to(device)
        self.unet = self.pipeline.unet.eval()
        if str(self.teacher_cfg.get("solver", "ddim")) == "ddim":
            self.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
        else:
            self.scheduler = self.pipeline.scheduler
        self.scheduler.set_timesteps(int(self.teacher_cfg.get("num_inference_steps", 128)), device=device)
        self._loaded_device = str(device)

    def sample_x0(self, batch_size: int, device: torch.device | str) -> Tensor:
        device = torch.device(device)
        return torch.randn(
            batch_size,
            int(self.dataset_cfg["channels"]),
            int(self.dataset_cfg["image_size"]),
            int(self.dataset_cfg["image_size"]),
            device=device,
        ) * float(self.teacher_cfg.get("x0_std", 1.0))

    def _noise_time_to_inference_index(self, tau: float) -> int:
        assert self.scheduler is not None
        total = int(self.scheduler.config.num_train_timesteps) - 1
        timestep = int(round(float(tau) * total))
        timesteps = self.scheduler.timesteps.detach().cpu().to(torch.long)
        return int(torch.argmin(torch.abs(timesteps - timestep)).item())

    def _forward_eps(self, x_t: Tensor, timestep_ids: Tensor) -> Tensor:
        assert self.unet is not None
        timestep_ids = timestep_ids.to(device=x_t.device, dtype=torch.long).view(-1)
        if x_t.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return self.unet(x_t, timestep_ids).sample
        return self.unet(x_t, timestep_ids).sample

    def _rollout_between_indices(self, x_t: Tensor, start_index: int, end_index: int) -> Tensor:
        assert self.scheduler is not None
        if end_index <= start_index:
            return x_t
        current = x_t
        timesteps = self.scheduler.timesteps.to(x_t.device)
        for step_index in range(start_index, end_index):
            step_value = timesteps[step_index]
            eps_pred = self._forward_eps(current, step_value.expand(current.shape[0]))
            current = self.scheduler.step(eps_pred, int(step_value.item()), current).prev_sample
        return current

    @torch.no_grad()
    def sample_trajectory(
        self,
        batch_size: int,
        u_grid: Tensor,
        device: torch.device | str,
    ) -> TeacherTrajectoryBatch:
        device = torch.device(device)
        self.prepare(device)
        x_0 = self.sample_x0(batch_size=batch_size, device=device)
        return self.sample_trajectory_from_x0(x_0=x_0, u_grid=u_grid, device=device)

    @torch.no_grad()
    def sample_trajectory_from_x0(
        self,
        x_0: Tensor,
        u_grid: Tensor,
        device: torch.device | str,
    ) -> TeacherTrajectoryBatch:
        device = torch.device(device)
        self.prepare(device)
        if u_grid.ndim != 1:
            raise ValueError("u_grid must be a 1D tensor")
        u_grid = torch.sort(u_grid.float()).values.to(device)
        tau_grid_desc = 1.0 - u_grid
        index_grid = [self._noise_time_to_inference_index(float(tau.item())) for tau in tau_grid_desc]

        current = x_0.to(device=device)
        x_desc = [current.detach()]
        for idx in range(len(index_grid) - 1):
            current = self._rollout_between_indices(current, index_grid[idx], index_grid[idx + 1])
            x_desc.append(current.detach())

        x_grid = torch.stack(x_desc, dim=1).float()
        return TeacherTrajectoryBatch(
            t_grid=u_grid.detach().cpu(),
            x_grid=x_grid.detach().cpu(),
        )
