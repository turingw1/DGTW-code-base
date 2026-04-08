from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_fidelity import calculate_metrics


@dataclass(slots=True)
class OfficialMetricsResult:
    fid: float | None
    inception_score_mean: float | None
    inception_score_std: float | None
    precision: float | None
    recall: float | None
    num_samples: int
    sample_path: str
    reference_path: str


class NPZImageDataset(Dataset):
    def __init__(self, npz_path: str | Path) -> None:
        self.npz_path = Path(npz_path)
        payload = np.load(self.npz_path)
        if "arr_0" not in payload:
            raise KeyError(f"{self.npz_path} does not contain arr_0")
        self.images = np.asarray(payload["arr_0"])
        self.labels = np.asarray(payload["labels"]) if "labels" in payload else None
        if self.images.ndim != 4:
            raise ValueError(f"Expected arr_0 to have shape [N,H,W,C], got {self.images.shape}")

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int):
        image = self.images[index]
        tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        if self.labels is None:
            return tensor
        return tensor, int(self.labels[index])


def save_samples_npz(
    images_uint8: np.ndarray,
    out_path: str | Path,
    *,
    labels_int64: np.ndarray | None = None,
    shuffle: bool = True,
    seed: int = 42,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    images = np.asarray(images_uint8)
    labels = None if labels_int64 is None else np.asarray(labels_int64)
    if shuffle:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(images.shape[0])
        images = images[perm]
        if labels is not None:
            labels = labels[perm]
    payload = {"arr_0": images}
    if labels is not None:
        payload["labels"] = labels.astype(np.int64, copy=False)
    np.savez(out_path, **payload)
    return out_path


def _metric_value(metrics: dict, *keys: str) -> float | None:
    for key in keys:
        if key in metrics:
            return float(metrics[key])
    return None


def evaluate_npz_metrics(
    *,
    samples_path: str | Path,
    reference_path: str | Path,
    metrics: list[str],
    cuda: bool,
    batch_size: int,
    prc_neighborhood: int = 3,
) -> OfficialMetricsResult:
    normalized = {item.strip().lower() for item in metrics}
    sample_ds = NPZImageDataset(samples_path)
    reference_ds = NPZImageDataset(reference_path)
    raw = calculate_metrics(
        input1=sample_ds,
        input2=reference_ds,
        cuda=cuda,
        batch_size=batch_size,
        fid="fid" in normalized,
        isc="is" in normalized or "isc" in normalized or "inception_score" in normalized,
        prc="precision" in normalized or "recall" in normalized or "prc" in normalized,
        prc_neighborhood=prc_neighborhood,
        verbose=False,
        samples_find_deep=False,
    )
    return OfficialMetricsResult(
        fid=_metric_value(raw, "frechet_inception_distance", "fid"),
        inception_score_mean=_metric_value(raw, "inception_score_mean"),
        inception_score_std=_metric_value(raw, "inception_score_std"),
        precision=_metric_value(raw, "precision"),
        recall=_metric_value(raw, "recall"),
        num_samples=len(sample_ds),
        sample_path=str(Path(samples_path)),
        reference_path=str(Path(reference_path)),
    )


def write_metrics_report(result: OfficialMetricsResult, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(result.__dict__, handle, indent=2)
    return out_path
