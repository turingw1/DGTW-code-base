from __future__ import annotations

from pathlib import Path
import sys

from torch import nn


def ensure_flow_matching_image_models_on_path() -> None:
    root = Path(__file__).resolve().parents[3]
    image_root = root / "flow_matching" / "examples" / "image"
    if str(image_root) not in sys.path:
        sys.path.insert(0, str(image_root))


class OfficialVelocityUNet(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        ensure_flow_matching_image_models_on_path()
        from models.unet import UNetModel

        self.model = UNetModel(**kwargs)

    def forward(self, x, t, extra: dict | None = None):
        return self.model(x, t, extra or {})
