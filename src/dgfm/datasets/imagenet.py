from __future__ import annotations

from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
import zipfile

from PIL import Image
import torch
from torch.utils.data import Dataset, random_split
from torchvision import datasets


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ZipImageFolderDataset(Dataset):
    """ImageFolder-style dataset backed by a zip archive."""

    def __init__(self, archive_path: Path, *, split: str, transform=None) -> None:
        self.archive_path = Path(archive_path)
        self.split = split.strip("/")
        self.transform = transform
        self._archive = zipfile.ZipFile(self.archive_path)
        names = [name for name in self._archive.namelist() if not name.endswith("/")]
        split_prefix = f"{self.split}/"
        candidates = [name for name in names if name.startswith(split_prefix)]
        classes = sorted({Path(name).parts[1] for name in candidates if len(Path(name).parts) >= 3})
        if not classes:
            raise FileNotFoundError(f"No class folders found under split '{self.split}' in {self.archive_path}")
        self.class_to_idx = {name: idx for idx, name in enumerate(classes)}
        self.samples: list[tuple[str, int]] = []
        for name in candidates:
            path = Path(name)
            if len(path.parts) < 3 or path.suffix.lower() not in IMG_EXTENSIONS:
                continue
            class_name = path.parts[1]
            self.samples.append((name, self.class_to_idx[class_name]))
        if not self.samples:
            raise FileNotFoundError(f"No image samples found under split '{self.split}' in {self.archive_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        rel_path, label = self.samples[index]
        with self._archive.open(rel_path, "r") as handle:
            image = Image.open(BytesIO(handle.read())).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def _resolve_imagenet64_source(dataset_cfg: dict) -> tuple[str, Path]:
    preprocessed = dataset_cfg.get("preprocessed_zip_or_folder")
    if preprocessed:
        return "preprocessed", Path(preprocessed)

    raw_root = dataset_cfg.get("raw_ilsvrc_root")
    if raw_root:
        return "raw_ilsvrc", Path(raw_root)

    return "preprocessed", Path(dataset_cfg["data_root"])


def _find_raw_ilsvrc_train_root(raw_root: Path) -> Path:
    candidates = [
        raw_root / "ILSVRC" / "Data" / "CLS-LOC" / "train",
        raw_root / "Data" / "CLS-LOC" / "train",
        raw_root / "train",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Unable to locate raw ILSVRC train split under {raw_root}. "
        "Expected one of: ILSVRC/Data/CLS-LOC/train, Data/CLS-LOC/train, train."
    )


def _find_preprocessed_roots(source_root: Path, *, train_split: str, val_split: str) -> tuple[Path, Path | None]:
    train_root = source_root / train_split
    val_root = source_root / val_split
    if train_root.is_dir():
        return train_root, val_root if val_root.is_dir() else None
    return source_root, None


def _split_dataset(dataset: Dataset, *, val_fraction: float, seed: int) -> tuple[Dataset, Dataset]:
    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError(f"Dataset too small for validation split: len={len(dataset)} val_fraction={val_fraction}")
    return random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )


def build_imagenet64_datasets(
    dataset_cfg: dict,
    *,
    train_transform,
    eval_transform,
    val_fraction: float,
    seed: int,
) -> tuple[Dataset, Dataset, Dataset]:
    source_kind, source_path = _resolve_imagenet64_source(dataset_cfg)
    train_split = str(dataset_cfg.get("train_split", "train"))
    val_split = str(dataset_cfg.get("val_split", "val"))

    if source_kind == "raw_ilsvrc":
        train_root = _find_raw_ilsvrc_train_root(source_path)
        full_train = datasets.ImageFolder(root=train_root, transform=train_transform)
        train_set, val_set = _split_dataset(full_train, val_fraction=val_fraction, seed=seed)
        val_set.dataset.transform = eval_transform
        test_set = val_set
        return train_set, val_set, test_set

    if source_path.is_file():
        full_train = ZipImageFolderDataset(source_path, split=train_split, transform=train_transform)
        if val_split != train_split:
            try:
                val_set = ZipImageFolderDataset(source_path, split=val_split, transform=eval_transform)
                return full_train, val_set, val_set
            except FileNotFoundError:
                pass
        train_set, val_set = _split_dataset(full_train, val_fraction=val_fraction, seed=seed)
        val_set.dataset.transform = eval_transform
        test_set = val_set
        return train_set, val_set, test_set

    train_root, val_root = _find_preprocessed_roots(source_path, train_split=train_split, val_split=val_split)
    full_train = datasets.ImageFolder(root=train_root, transform=train_transform)
    if val_root is not None:
        val_set = datasets.ImageFolder(root=val_root, transform=eval_transform)
        return full_train, val_set, val_set
    train_set, val_set = _split_dataset(full_train, val_fraction=val_fraction, seed=seed)
    val_set.dataset.transform = eval_transform
    test_set = val_set
    return train_set, val_set, test_set
