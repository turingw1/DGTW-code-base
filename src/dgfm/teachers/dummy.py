from __future__ import annotations

import torch
from torch import Tensor

from .diffusers_ddpm import TeacherTrajectoryBatch


class DummyTeacher:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.dataset_cfg = config.get("dataset", {})
        self.teacher_cfg = config.get("teacher", {})

    def is_enabled(self) -> bool:
        return True

    def prepare(self, device) -> None:
        del device

    def sample_x0(self, batch_size: int, device) -> Tensor:
        device = torch.device(device)
        return torch.randn(
            batch_size,
            int(self.dataset_cfg["channels"]),
            int(self.dataset_cfg["image_size"]),
            int(self.dataset_cfg["image_size"]),
            device=device,
        ) * float(self.teacher_cfg.get("x0_std", 1.0))

    @torch.no_grad()
    def sample_trajectory_from_x0(self, x_0: Tensor, u_grid: Tensor, device) -> TeacherTrajectoryBatch:
        device = torch.device(device)
        u_grid = torch.sort(u_grid.float()).values.to(device)
        x_1 = torch.tanh(0.6 * x_0 + 0.15)
        x_grid = []
        for u in u_grid:
            x_u = (1.0 - u) * x_0 + u * x_1
            x_grid.append(x_u.detach())
        return TeacherTrajectoryBatch(
            t_grid=u_grid.detach().cpu(),
            x_grid=torch.stack(x_grid, dim=1).detach().cpu(),
        )

    @torch.no_grad()
    def sample_trajectory(self, batch_size: int, u_grid: Tensor, device) -> TeacherTrajectoryBatch:
        x_0 = self.sample_x0(batch_size=batch_size, device=device)
        return self.sample_trajectory_from_x0(x_0=x_0, u_grid=u_grid, device=device)
