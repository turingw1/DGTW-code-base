from __future__ import annotations

import argparse
from pathlib import Path

from torchvision import datasets, transforms


def _cifar10_ready(root: Path) -> bool:
    return (root / "cifar-10-batches-py").is_dir()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare datasets for dgfm experiments")
    parser.add_argument("--dataset", required=True, choices=["cifar10", "imagenet32", "imagenet64"], help="Dataset name")
    parser.add_argument("--data-root", required=True, help="Dataset root directory")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Explicitly download CIFAR-10 if it is missing. Default behavior is manual/check-only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    if args.dataset == "cifar10":
        if _cifar10_ready(data_root):
            print(f"cifar10 already prepared at {data_root}")
            return
        if not args.download:
            print(f"cifar10 not found at {data_root}", flush=True)
            print("manual mode: place `cifar-10-batches-py/` under the data root, or rerun with `--download`.", flush=True)
            return
        transform = transforms.ToTensor()
        datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
        datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
        print(f"prepared cifar10 at {data_root}")
        return

    if args.dataset == "imagenet64":
        train_root = data_root / "train"
        val_root = data_root / "val"
        if train_root.is_dir():
            print(f"imagenet64 preprocessed folder detected at {data_root}", flush=True)
            return
        print(f"imagenet64 not prepared at {data_root}", flush=True)
        print(
            "manual mode: either place preprocessed train/val folders under the data root, "
            "or run `python scripts/prepare_imagenet64.py --source-root <raw_train_dir> --output-root "
            f"{data_root}` after obtaining raw ILSVRC2012 data.",
            flush=True,
        )
        return

    print(
        f"dataset scaffold prepared for {args.dataset} at {data_root}. "
        "Manual ImageNet population is still required in Phase 1.",
        flush=True,
    )


if __name__ == "__main__":
    main()
