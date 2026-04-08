from __future__ import annotations

import argparse

from .config import load_experiment_config
from .evaluators.official_metrics import evaluate_npz_metrics, write_metrics_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated sample npz files with official-style metrics")
    parser.add_argument("--config", default=None, help="Optional experiment config path for default reference/metrics")
    parser.add_argument("--samples", required=True, help="Generated samples .npz")
    parser.add_argument("--reference", default=None, help="Reference samples/statistics .npz")
    parser.add_argument("--metrics", default=None, help="Comma-separated metrics")
    parser.add_argument("--batch-size", type=int, default=64, help="Metric batch size")
    parser.add_argument("--cpu", action="store_true", help="Force CPU evaluation")
    parser.add_argument("--out", required=True, help="Output json path")
    parser.add_argument("--set", action="append", default=[], help="Optional config override in key=value form")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config, overrides=args.set) if args.config else {}
    reference = args.reference or cfg.get("eval", {}).get("official_reference_npz")
    if not reference:
        raise ValueError("--reference is required unless eval.official_reference_npz is set in --config")
    metric_source = args.metrics
    if metric_source is None:
        default_metrics = cfg.get("eval", {}).get("official_metrics", ["fid", "is", "precision", "recall"])
        metric_source = ",".join(default_metrics)
    metrics = [item.strip() for item in metric_source.split(",") if item.strip()]
    batch_size = int(args.batch_size or cfg.get("eval", {}).get("official_metric_batch_size", 64))
    result = evaluate_npz_metrics(
        samples_path=args.samples,
        reference_path=reference,
        metrics=metrics,
        cuda=not args.cpu,
        batch_size=batch_size,
    )
    report_path = write_metrics_report(result, args.out)
    print("dgfm official-style evaluation completed")
    print(f"samples: {args.samples}")
    print(f"reference: {reference}")
    print(f"metrics: {metrics}")
    print(f"report_path: {report_path}")


if __name__ == "__main__":
    main()
