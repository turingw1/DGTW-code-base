from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "debug" / ".mplconfig"))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dgfm.config import load_experiment_config
from dgfm.evaluators import build_evaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run refactored DGFM evaluation")
    parser.add_argument("--config", required=True, help="Experiment config path")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--eval-root", required=True, help="Evaluation root directory")
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 2, 4, 8, 16], help="Few-step evaluation list")
    parser.add_argument("--fid-samples", type=int, default=None, help="Override eval.num_fid_samples")
    parser.add_argument("--fid-batch-size", type=int, default=None, help="Override eval.fid_batch_size")
    parser.add_argument("--sample-batch-size", type=int, default=None, help="Override eval.sample_batch_size")
    parser.add_argument("--set", action="append", default=[], help="Config override in key=value form")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TORCH_HOME", str(Path(args.eval_root) / ".torch"))
    overrides = list(args.set)
    if args.fid_samples is not None:
        overrides.append(f"eval.num_fid_samples={args.fid_samples}")
    if args.fid_batch_size is not None:
        overrides.append(f"eval.fid_batch_size={args.fid_batch_size}")
    if args.sample_batch_size is not None:
        overrides.append(f"eval.sample_batch_size={args.sample_batch_size}")
    config = load_experiment_config(args.config, overrides=overrides)
    runner = build_evaluator(
        config=config,
        checkpoint=Path(args.checkpoint),
        eval_root=Path(args.eval_root),
    )
    runner.run(step_counts=args.steps)


if __name__ == "__main__":
    main()
