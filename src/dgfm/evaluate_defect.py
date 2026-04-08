from __future__ import annotations

import argparse

from .config import load_experiment_config
from .evaluators.defect_evaluator import evaluate_held_out_defect


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run held-out semigroup defect evaluation for explicit-map checkpoints")
    parser.add_argument("--config", required=True, help="Experiment config path")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--out", required=True, help="Output json path")
    parser.add_argument("--num-samples", type=int, default=4096, help="Held-out seed count")
    parser.add_argument("--grid-steps", type=int, default=16, help="Dense runtime grid step count")
    parser.add_argument("--triplets", default=None, help="Optional json triplet preset")
    parser.add_argument("--seed", type=int, default=42, help="Evaluation seed")
    parser.add_argument("--set", action="append", default=[], help="Config override in key=value form")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config, overrides=args.set)
    report = evaluate_held_out_defect(
        config=config,
        checkpoint=args.checkpoint,
        out_path=args.out,
        num_samples=args.num_samples,
        grid_steps=args.grid_steps,
        triplets_path=args.triplets,
        seed=args.seed,
    )
    print("dgfm held-out defect evaluation completed")
    print(f"checkpoint: {args.checkpoint}")
    print(f"out: {args.out}")
    print(f"defect_mean: {report.defect_mean:.6f}")
    print(f"defect_std: {report.defect_std:.6f}")
    print(f"num_triplets: {report.num_triplets}")


if __name__ == "__main__":
    main()
