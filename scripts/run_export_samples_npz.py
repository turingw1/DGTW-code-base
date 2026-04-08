from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "debug" / ".mplconfig"))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dgfm.config import load_experiment_config
from dgfm.evaluators.common import (
    device_from_config,
    load_model_from_checkpoint,
    load_timewarp_from_checkpoint,
    sample_condition_labels,
    sample_from_model_batched,
    to_unit_interval,
)
from dgfm.evaluators.official_metrics import save_samples_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export generated samples into official-style npz format")
    parser.add_argument("--config", required=True, help="Experiment config path")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--out", required=True, help="Output .npz path")
    parser.add_argument("--num-samples", type=int, default=0, help="Number of generated samples, 0 uses config default")
    parser.add_argument("--steps", type=int, default=1, help="Sampling step count")
    parser.add_argument("--sample-batch-size", type=int, default=0, help="Maximum per-forward batch size")
    parser.add_argument("--seed", type=int, default=42, help="Generation seed")
    parser.add_argument("--set", action="append", default=[], help="Config override in key=value form")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config, overrides=args.set)
    device = device_from_config(config)
    model = load_model_from_checkpoint(config, args.checkpoint, device=device)
    timewarp = load_timewarp_from_checkpoint(config, args.checkpoint, device=device)
    channels = int(config["dataset"]["channels"])
    image_size = int(config["dataset"]["image_size"])
    num_samples = int(args.num_samples or config.get("eval", {}).get("official_num_samples", config.get("eval", {}).get("num_fid_samples", 50000)))

    sample_batch_size = int(args.sample_batch_size)
    if sample_batch_size <= 0:
        sample_batch_size = int(
            config.get("eval", {}).get("official_export_batch_size", 0)
            or config.get("eval", {}).get("sample_batch_size", 0)
            or 64
        )

    generator = torch.Generator(device=device).manual_seed(args.seed)
    images_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    remaining = num_samples
    while remaining > 0:
        current = min(sample_batch_size, remaining)
        noise = torch.randn(current, channels, image_size, image_size, generator=generator, device=device)
        labels = sample_condition_labels(config, current, device=device, generator=generator)
        samples = sample_from_model_batched(
            config=config,
            model=model,
            x_init=noise,
            step_count=args.steps,
            timewarp=timewarp,
            max_batch_size=sample_batch_size,
            move_to_cpu=True,
            extra={"label": labels} if labels is not None else None,
        )
        samples = to_unit_interval(samples)
        images_uint8 = torch.round(samples * 255.0).to(torch.uint8).permute(0, 2, 3, 1).contiguous().numpy()
        images_chunks.append(images_uint8)
        if labels is not None:
            label_chunks.append(labels.detach().cpu().numpy().astype(np.int64, copy=False))
        remaining -= current

    images_all = np.concatenate(images_chunks, axis=0)
    labels_all = np.concatenate(label_chunks, axis=0) if label_chunks else None
    out_path = save_samples_npz(images_all, args.out, labels_int64=labels_all, shuffle=True, seed=args.seed)
    print("dgfm npz export completed")
    print(f"checkpoint: {args.checkpoint}")
    print(f"out: {out_path}")
    print(f"num_samples: {images_all.shape[0]}")
    print(f"steps: {args.steps}")
    print(f"class_cond: {labels_all is not None}")


if __name__ == "__main__":
    main()
