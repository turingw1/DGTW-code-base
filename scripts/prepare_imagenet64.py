from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess ImageNet-style class folders into 64x64 image folders")
    parser.add_argument("--source-root", required=True, help="Source class-folder root, e.g. ILSVRC/Data/CLS-LOC/train")
    parser.add_argument("--output-root", required=True, help="Output directory containing class folders")
    parser.add_argument("--image-size", type=int, default=64, help="Target square image size")
    parser.add_argument("--limit-per-class", type=int, default=0, help="Optional cap per class, 0 means no cap")
    return parser.parse_args()


def _iter_class_images(source_root: Path):
    for class_dir in sorted(source_root.iterdir()):
        if not class_dir.is_dir():
            continue
        files = [
            path
            for path in sorted(class_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS
        ]
        if files:
            yield class_dir.name, files


def _prepare_image(path: Path, image_size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    side = min(width, height)
    left = max(0, (width - side) // 2)
    top = max(0, (height - side) // 2)
    image = image.crop((left, top, left + side, top + side))
    return image.resize((image_size, image_size), Image.BICUBIC)


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    written = 0
    classes = 0
    for class_name, files in _iter_class_images(source_root):
        classes += 1
        class_out = output_root / class_name
        class_out.mkdir(parents=True, exist_ok=True)
        limit = len(files) if args.limit_per_class <= 0 else min(len(files), args.limit_per_class)
        for idx, source_path in enumerate(files[:limit]):
            image = _prepare_image(source_path, args.image_size)
            target_path = class_out / f"{idx:06d}.png"
            image.save(target_path)
            written += 1

    print("imagenet64 preprocessing completed")
    print(f"source_root: {source_root}")
    print(f"output_root: {output_root}")
    print(f"classes: {classes}")
    print(f"images_written: {written}")


if __name__ == "__main__":
    main()
