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
from dgfm.evaluators import save_multistep_qualitative_panel
from dgfm.evaluators.common import (
    device_from_config,
    load_model_from_checkpoint,
    load_timewarp_from_checkpoint,
    sample_condition_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-style qualitative panel across multiple step counts")
    parser.add_argument("--config", required=True, help="Experiment config path")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--output-dir", required=True, help="Output directory for the panel")
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 2, 4, 8, 16], help="Step counts to compare")
    parser.add_argument("--num-examples", type=int, default=8, help="Number of fixed-seed examples (rows)")
    parser.add_argument("--fixed-seed", type=int, default=42, help="Fixed seed for qualitative generation")
    parser.add_argument("--solver-method", default=None, help="Override eval.solver_method")
    parser.add_argument("--no-noise-column", action="store_true", help="Do not include the initial noise column")
    parser.add_argument("--set", action="append", default=[], help="Config override in key=value form")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = list(args.set)
    if args.solver_method:
        overrides.append(f"eval.solver_method='{args.solver_method}'")
    config = load_experiment_config(args.config, overrides=overrides)
    device = device_from_config(config)
    model = load_model_from_checkpoint(config, args.checkpoint, device=device)
    timewarp = load_timewarp_from_checkpoint(config, args.checkpoint, device=device)
    labels = sample_condition_labels(config, args.num_examples, device=device)

    result = save_multistep_qualitative_panel(
        config=config,
        model=model,
        output_dir=Path(args.output_dir),
        channels=int(config["dataset"]["channels"]),
        image_size=int(config["dataset"]["image_size"]),
        step_counts=args.steps,
        num_examples=args.num_examples,
        fixed_seed=args.fixed_seed,
        solver_method=str(config.get("eval", {}).get("solver_method", "heun2")),
        include_noise=not args.no_noise_column,
        device=device,
        timewarp=timewarp,
        sample_extra={"label": labels} if labels is not None else None,
    )
    print("dgfm multistep qualitative panel completed")
    print(f"checkpoint: {args.checkpoint}")
    print(f"output_dir: {args.output_dir}")
    print(f"panel_path: {result['panel_path']}")
    print(f"payload_path: {result['payload_path']}")
    print(f"step_counts: {result['step_counts']}")
    print(f"num_examples: {result['num_examples']}")
    print(f"fixed_seed: {result['fixed_seed']}")
    if labels is not None:
        print(f"sample_labels: {labels.detach().cpu().tolist()}")


if __name__ == "__main__":
    main()
