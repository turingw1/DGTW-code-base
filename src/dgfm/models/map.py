from __future__ import annotations

import torch
from torch import Tensor, nn

from .official_unet import ensure_flow_matching_image_models_on_path


class OfficialExplicitMapUNet(nn.Module):
    """Official image-example UNet adapted to explicit map prediction.

    The model keeps dgfm's current time semantics:
    - x_t is the state at time t
    - s > t is the target time
    - the model predicts x_s directly or through a residual parameterization
    """

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: tuple[int, ...],
        dropout: float,
        channel_mult: tuple[int, ...],
        conv_resample: bool,
        dims: int,
        num_classes,
        use_checkpoint: bool,
        num_heads: int,
        num_head_channels: int,
        num_heads_upsample: int,
        use_scale_shift_norm: bool,
        resblock_updown: bool,
        use_new_attention_order: bool,
        with_fourier_features: bool,
        map_conditioning_channels: int = 2,
        prediction_type: str = "residual",
        residual_scale_by_delta: bool = True,
        residual_tanh_scale: float = 1.0,
        use_preconditioning: bool = False,
        sigma_data: float = 0.5,
        sigma_min: float = 1.0e-3,
        time_embed_mode: str = "time",
        inner_parametrization: str = "no",
        outer_parametrization: str = "euler",
    ) -> None:
        super().__init__()
        ensure_flow_matching_image_models_on_path()
        from models.unet import UNetModel

        self.prediction_type = prediction_type
        self.residual_scale_by_delta = residual_scale_by_delta
        self.residual_tanh_scale = residual_tanh_scale
        self.map_conditioning_channels = map_conditioning_channels
        self.use_preconditioning = use_preconditioning
        self.sigma_data = float(sigma_data)
        self.sigma_min = float(sigma_min)
        self.time_embed_mode = str(time_embed_mode)
        self.inner_parametrization = str(inner_parametrization)
        self.outer_parametrization = str(outer_parametrization)

        self.model = UNetModel(
            in_channels=in_channels + map_conditioning_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            conv_resample=conv_resample,
            dims=dims,
            num_classes=num_classes,
            use_checkpoint=use_checkpoint,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            with_fourier_features=with_fourier_features,
        )

    def _conditioning_maps(self, x_t: Tensor, t: Tensor, s: Tensor) -> Tensor:
        delta = (s - t).view(-1, 1, 1, 1)
        s_map = s.view(-1, 1, 1, 1).expand(-1, 1, x_t.shape[-2], x_t.shape[-1])
        delta_map = delta.expand(-1, 1, x_t.shape[-2], x_t.shape[-1])
        return torch.cat([s_map, delta_map], dim=1)

    def _noise_level(self, time_value: Tensor) -> Tensor:
        return torch.clamp(1.0 - time_value, min=self.sigma_min)

    def _time_embedding_input(self, t: Tensor) -> Tensor:
        if self.time_embed_mode == "time":
            return t
        if self.time_embed_mode == "log_noise":
            noise_t = self._noise_level(t)
            return 1000.0 * 0.25 * torch.log(noise_t + 1.0e-44)
        raise ValueError(f"Unsupported time_embed_mode: {self.time_embed_mode}")

    def _inner_scalings(self, noise_t: Tensor) -> tuple[Tensor, Tensor]:
        if self.inner_parametrization == "no":
            return torch.zeros_like(noise_t), torch.ones_like(noise_t)
        if self.inner_parametrization == "edm":
            c_skip = self.sigma_data**2 / (noise_t**2 + self.sigma_data**2)
            c_out = noise_t * self.sigma_data / torch.sqrt(noise_t**2 + self.sigma_data**2)
            return c_skip, c_out
        raise ValueError(f"Unsupported inner_parametrization: {self.inner_parametrization}")

    def _outer_scalings(self, noise_t: Tensor, noise_s: Tensor) -> tuple[Tensor, Tensor]:
        if self.outer_parametrization == "no":
            return torch.zeros_like(noise_t), torch.ones_like(noise_t)
        if self.outer_parametrization == "euler":
            ratio = torch.clamp(noise_s / noise_t, min=0.0, max=1.0)
            return ratio, 1.0 - ratio
        raise ValueError(f"Unsupported outer_parametrization: {self.outer_parametrization}")

    def forward(self, x_t: Tensor, t: Tensor, s: Tensor, extra: dict | None = None) -> Tensor:
        model_extra = dict(extra or {})
        model_extra["concat_conditioning"] = self._conditioning_maps(x_t, t, s)
        model_input = x_t
        model_time = t
        if self.use_preconditioning:
            noise_t = self._noise_level(t)
            c_in = 1.0 / torch.sqrt(noise_t**2 + self.sigma_data**2)
            model_input = x_t * c_in.view(-1, 1, 1, 1)
            model_time = self._time_embedding_input(t)
        raw = self.model(model_input, model_time, model_extra)

        if self.use_preconditioning:
            noise_t = self._noise_level(t)
            noise_s = self._noise_level(s)
            c_skip_inner, c_out_inner = self._inner_scalings(noise_t)
            g_theta = c_out_inner.view(-1, 1, 1, 1) * raw + c_skip_inner.view(-1, 1, 1, 1) * x_t
            c_skip_outer, c_out_outer = self._outer_scalings(noise_t, noise_s)
            return c_out_outer.view(-1, 1, 1, 1) * g_theta + c_skip_outer.view(-1, 1, 1, 1) * x_t

        if self.prediction_type == "direct":
            return raw
        if self.prediction_type != "residual":
            raise ValueError(f"Unsupported map prediction_type: {self.prediction_type}")

        residual = torch.tanh(raw) * self.residual_tanh_scale
        if self.residual_scale_by_delta:
            residual = residual * (s - t).view(-1, 1, 1, 1)
        return x_t + residual


def build_map_model(config: dict) -> nn.Module:
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    family = str(model_cfg.get("family", "official_map_unet"))
    if family != "official_map_unet":
        raise ValueError(f"Unsupported explicit map model family: {family}")
    return OfficialExplicitMapUNet(
        in_channels=int(dataset_cfg["channels"]),
        model_channels=int(model_cfg.get("hidden_channels", 128)),
        out_channels=int(dataset_cfg["channels"]),
        num_res_blocks=int(model_cfg.get("num_res_blocks", 4)),
        attention_resolutions=tuple(model_cfg.get("attention_resolutions", [2])),
        dropout=float(model_cfg.get("dropout", 0.3)),
        channel_mult=tuple(model_cfg.get("channel_mult", [2, 2, 2])),
        conv_resample=bool(model_cfg.get("conv_resample", False)),
        dims=int(model_cfg.get("dims", 2)),
        num_classes=model_cfg.get("num_classes", None),
        use_checkpoint=bool(model_cfg.get("use_checkpoint", False)),
        num_heads=int(model_cfg.get("num_heads", 1)),
        num_head_channels=int(model_cfg.get("num_head_channels", -1)),
        num_heads_upsample=int(model_cfg.get("num_heads_upsample", -1)),
        use_scale_shift_norm=bool(model_cfg.get("use_scale_shift_norm", True)),
        resblock_updown=bool(model_cfg.get("resblock_updown", False)),
        use_new_attention_order=bool(model_cfg.get("use_new_attention_order", True)),
        with_fourier_features=bool(model_cfg.get("with_fourier_features", False)),
        map_conditioning_channels=int(model_cfg.get("map_conditioning_channels", 2)),
        prediction_type=str(model_cfg.get("prediction_type", "residual")),
        residual_scale_by_delta=bool(model_cfg.get("residual_scale_by_delta", True)),
        residual_tanh_scale=float(model_cfg.get("residual_tanh_scale", 1.0)),
        use_preconditioning=bool(model_cfg.get("use_preconditioning", False)),
        sigma_data=float(model_cfg.get("sigma_data", 0.5)),
        sigma_min=float(model_cfg.get("sigma_min", 1.0e-3)),
        time_embed_mode=str(model_cfg.get("time_embed_mode", "time")),
        inner_parametrization=str(model_cfg.get("inner_parametrization", "no")),
        outer_parametrization=str(model_cfg.get("outer_parametrization", "euler")),
    )
