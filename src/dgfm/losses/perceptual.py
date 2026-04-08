from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class MultiScaleL1(nn.Module):
    def __init__(self, levels: int = 3) -> None:
        super().__init__()
        self.levels = max(1, int(levels))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = pred.new_tensor(0.0)
        cur_pred = pred
        cur_target = target
        for _ in range(self.levels):
            total = total + torch.mean(torch.abs(cur_pred - cur_target))
            if min(cur_pred.shape[-2:]) <= 8:
                break
            cur_pred = F.avg_pool2d(cur_pred, kernel_size=2, stride=2)
            cur_target = F.avg_pool2d(cur_target, kernel_size=2, stride=2)
        return total / float(self.levels)


class PIQLPIPS(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from piq import LPIPS

        self.metric = LPIPS(replace_pooling=True, reduction="none")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_input = pred
        target_input = target
        if pred_input.shape[-2] < 256:
            pred_input = F.interpolate(pred_input, size=224, mode="bilinear", align_corners=False)
            target_input = F.interpolate(target_input, size=224, mode="bilinear", align_corners=False)
        losses = self.metric((pred_input + 1.0) / 2.0, (target_input + 1.0) / 2.0)
        return torch.mean(losses)


def build_perceptual_metric(config: dict) -> nn.Module | None:
    loss_cfg = config.get("loss", {})
    weight = float(loss_cfg.get("perceptual_weight", 0.0))
    if weight <= 0.0:
        return None
    loss_type = str(loss_cfg.get("perceptual_type", "lpips_piq"))
    if loss_type == "lpips_piq":
        try:
            metric = PIQLPIPS()
        except ImportError as exc:
            fallback = str(loss_cfg.get("perceptual_fallback", "multiscale_l1"))
            if fallback != "multiscale_l1":
                raise ImportError(
                    "Perceptual loss requested with perceptual_type=lpips_piq, "
                    "but package 'piq' is not installed."
                ) from exc
            metric = MultiScaleL1(levels=int(loss_cfg.get("perceptual_fallback_levels", 3)))
    elif loss_type == "multiscale_l1":
        metric = MultiScaleL1(levels=int(loss_cfg.get("perceptual_fallback_levels", 3)))
    else:
        raise ValueError(f"Unsupported perceptual_type: {loss_type}")
    metric.requires_grad_(False)
    metric.eval()
    return metric
