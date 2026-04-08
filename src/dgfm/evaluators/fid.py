from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable
from urllib.error import URLError, HTTPError
from urllib.parse import urlparse

import numpy as np
import torch
from torch import nn
from torch.hub import download_url_to_file, get_dir
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from torch_fidelity.feature_extractor_inceptionv3 import URL_INCEPTION_V3
from torch_fidelity.metric_fid import KEY_METRIC_FID, fid_statistics_to_metric


FID_FEATURE_LAYER = "2048"
FID_WEIGHTS_FILENAME = Path(URL_INCEPTION_V3).name
DEFAULT_FID_MIRROR_PREFIX = "https://githubfast.com/"


def _resolve_mirror_prefixed_url(prefix: str, source_url: str) -> str:
    parsed = urlparse(source_url)
    if parsed.scheme in {"http", "https"} and parsed.netloc == "github.com":
        mirrored_path = parsed.path.lstrip("/")
        return prefix.rstrip("/") + "/" + mirrored_path
    return prefix.rstrip("/") + "/" + source_url.lstrip("/")


def _resolve_inception_weights_path() -> str | None:
    explicit_path = os.environ.get("DGFM_TORCH_FIDELITY_WEIGHTS_PATH")
    if explicit_path:
        return explicit_path

    hub_dir = Path(get_dir())
    checkpoint_dir = hub_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target_path = checkpoint_dir / FID_WEIGHTS_FILENAME
    if target_path.exists():
        return str(target_path)

    mirror_url = os.environ.get("DGFM_TORCH_FIDELITY_MIRROR_URL")
    mirror_prefix = os.environ.get("DGFM_TORCH_FIDELITY_MIRROR_PREFIX", DEFAULT_FID_MIRROR_PREFIX)

    candidate_urls: list[str] = []
    if mirror_url:
        candidate_urls.append(mirror_url)
    if mirror_prefix:
        candidate_urls.append(_resolve_mirror_prefixed_url(str(mirror_prefix), URL_INCEPTION_V3))
    candidate_urls.append(URL_INCEPTION_V3)

    last_error: Exception | None = None
    for resolved_url in candidate_urls:
        try:
            download_url_to_file(resolved_url, str(target_path), progress=True)
            return str(target_path)
        except (URLError, HTTPError, OSError, RuntimeError) as exc:
            last_error = exc
            if target_path.exists():
                target_path.unlink()

    raise RuntimeError(
        "Unable to download torch-fidelity Inception weights. "
        f"Tried: {candidate_urls}. "
        "Set DGFM_TORCH_FIDELITY_WEIGHTS_PATH to a local file, or set "
        "DGFM_TORCH_FIDELITY_MIRROR_URL / DGFM_TORCH_FIDELITY_MIRROR_PREFIX "
        "to a reachable mirror."
    ) from last_error


class InceptionFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        weights_path = _resolve_inception_weights_path()
        self.model = FeatureExtractorInceptionV3(
            name="inception-v3-compat",
            features_list=[FID_FEATURE_LAYER],
            feature_extractor_weights_path=weights_path,
        )

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = to_uint8(images)
        feats = self.model(images)[0]
        if feats.ndim > 2:
            feats = torch.flatten(feats, start_dim=1)
        return feats


@dataclass(slots=True)
class GaussianStats:
    mu: np.ndarray
    sigma: np.ndarray
    count: int


class RunningStats:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.count = 0
        self.sum = np.zeros(dim, dtype=np.float64)
        self.sum_sq = np.zeros((dim, dim), dtype=np.float64)

    def update(self, feats: torch.Tensor) -> None:
        arr = feats.detach().cpu().numpy().astype(np.float64)
        self.count += arr.shape[0]
        self.sum += arr.sum(axis=0)
        self.sum_sq += arr.T @ arr

    def finalize(self) -> GaussianStats:
        mu = self.sum / max(1, self.count)
        sigma = self.sum_sq / max(1, self.count) - np.outer(mu, mu)
        return GaussianStats(mu=mu, sigma=sigma, count=self.count)


def to_uint8(images: torch.Tensor) -> torch.Tensor:
    if images.dtype == torch.uint8:
        return images
    images = torch.clamp(images, 0.0, 1.0)
    return torch.round(images * 255.0).to(torch.uint8)


@torch.no_grad()
def compute_dataset_stats(
    loader: Iterable,
    feature_extractor: nn.Module,
    device: torch.device,
    image_limit: int | None = None,
) -> GaussianStats:
    running: RunningStats | None = None
    seen = 0
    for images, _labels in loader:
        images = images.to(device)
        feats = feature_extractor(images)
        if running is None:
            running = RunningStats(int(feats.shape[1]))
        if image_limit is not None and seen + images.shape[0] > image_limit:
            keep = image_limit - seen
            feats = feats[:keep]
            images = images[:keep]
        running.update(feats)
        seen += images.shape[0]
        if image_limit is not None and seen >= image_limit:
            break
    if running is None:
        raise RuntimeError("No images available for FID statistics")
    return running.finalize()


@torch.no_grad()
def compute_generator_stats(
    sample_fn,
    feature_extractor: nn.Module,
    batch_size: int,
    total_samples: int,
    device: torch.device,
) -> GaussianStats:
    running: RunningStats | None = None
    produced = 0
    while produced < total_samples:
        current = min(batch_size, total_samples - produced)
        samples = sample_fn(current)
        samples = samples.to(device)
        feats = feature_extractor(samples)
        if running is None:
            running = RunningStats(int(feats.shape[1]))
        running.update(feats)
        produced += current
    if running is None:
        raise RuntimeError("No generated samples available for FID statistics")
    return running.finalize()


def frechet_distance(real: GaussianStats, fake: GaussianStats) -> float:
    metric = fid_statistics_to_metric(
        {
            "mu": np.asarray(real.mu, dtype=np.float64),
            "sigma": np.asarray(real.sigma, dtype=np.float64),
        },
        {
            "mu": np.asarray(fake.mu, dtype=np.float64),
            "sigma": np.asarray(fake.sigma, dtype=np.float64),
        },
        verbose=False,
    )
    return float(metric[KEY_METRIC_FID])


def save_stats(path: str | Path, stats: GaussianStats) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mu=stats.mu, sigma=stats.sigma, count=stats.count)


def load_stats(path: str | Path) -> GaussianStats:
    data = np.load(Path(path))
    return GaussianStats(mu=data["mu"], sigma=data["sigma"], count=int(data["count"]))
