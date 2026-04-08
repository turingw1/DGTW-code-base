from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dgfm.config import load_experiment_config, resolve_run_roots
from dgfm.trainers import build_trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run refactored DGFM training")
    parser.add_argument("--config", required=True, help="Experiment config path")
    parser.add_argument("--run-root", required=True, help="Run root directory")
    parser.add_argument("--resume", default=None, help="Optional checkpoint to resume from")
    parser.add_argument("--verbose", action="store_true", help="Print extended diagnostics in addition to core metrics")
    parser.add_argument("--set", action="append", default=[], help="Config override in key=value form")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config, overrides=args.set)
    if bool(config.get("runtime", {}).get("cudnn_benchmark", False)) and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    roots = resolve_run_roots(args.run_root)
    trainer = build_trainer(config=config, roots=roots)
    trainer.run(resume=args.resume, verbose=args.verbose)


if __name__ == "__main__":
    main()
