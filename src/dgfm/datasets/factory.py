from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from .imagenet import build_imagenet64_datasets
from .trajectory import build_trajectory_dataloaders


def _require_path(path: Path, message: str) -> None:
    if not path.exists():
        raise FileNotFoundError(message)


def _build_cifar10_transforms():
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    eval_transform = transforms.ToTensor()
    return train_transform, eval_transform


def _build_imagefolder_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )
    return train_transform, eval_transform


def build_image_dataloaders(config: dict) -> dict[str, DataLoader]:
    dataset_cfg = config["dataset"]
    train_cfg = config["train"]
    image_size = int(dataset_cfg["image_size"])
    batch_size = int(train_cfg["batch_size"])
    num_workers = int(train_cfg.get("num_workers", 4))
    val_split = float(train_cfg.get("val_fraction", 0.05))
    if dataset_cfg["name"] == "cifar10":
        root = Path(dataset_cfg["data_root"])
        train_transform, eval_transform = _build_cifar10_transforms()
        _require_path(
            root / "cifar-10-batches-py",
            (
                f"CIFAR-10 data not found under {root}. "
                "Prepare it manually or run `python scripts/build_dataset.py --dataset cifar10 "
                f"--data-root {root} --download` once."
            ),
        )
        full_train = datasets.CIFAR10(root=root, train=True, download=False, transform=train_transform)
        test_set = datasets.CIFAR10(root=root, train=False, download=False, transform=eval_transform)
        val_size = max(1, int(len(full_train) * val_split))
        train_size = len(full_train) - val_size
        train_set, val_set = random_split(
            full_train,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(int(config["experiment"].get("seed", 42))),
        )
        val_set.dataset.transform = eval_transform
    elif dataset_cfg["name"] == "imagenet32":
        root = Path(dataset_cfg["data_root"])
        train_root = root / dataset_cfg.get("train_split", "train")
        val_root = root / dataset_cfg.get("val_split", "val")
        train_transform, eval_transform = _build_imagefolder_transforms(image_size=image_size)
        _require_path(train_root, f"ImageNet train split not found: {train_root}")
        _require_path(val_root, f"ImageNet val split not found: {val_root}")
        train_set = datasets.ImageFolder(root=train_root, transform=train_transform)
        val_set = datasets.ImageFolder(root=val_root, transform=eval_transform)
        test_set = val_set
    elif dataset_cfg["name"] == "imagenet64":
        train_transform, eval_transform = _build_imagefolder_transforms(image_size=image_size)
        train_set, val_set, test_set = build_imagenet64_datasets(
            dataset_cfg,
            train_transform=train_transform,
            eval_transform=eval_transform,
            val_fraction=val_split,
            seed=int(config["experiment"].get("seed", 42)),
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_cfg['name']}")

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": bool(train_cfg.get("pin_memory", True)),
    }
    if num_workers > 0:
        common["persistent_workers"] = bool(train_cfg.get("persistent_workers", True))
        common["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 4))
    return {
        "train": DataLoader(train_set, shuffle=True, drop_last=True, **common),
        "val": DataLoader(val_set, shuffle=False, drop_last=False, **common),
        "test": DataLoader(test_set, shuffle=False, drop_last=False, **common),
    }


def build_map_training_dataloaders(config: dict) -> dict[str, DataLoader]:
    target_mode = str(config.get("target", {}).get("builder", "analytic_path"))
    if target_mode == "trajectory_shard":
        return build_trajectory_dataloaders(config)
    return build_image_dataloaders(config)
