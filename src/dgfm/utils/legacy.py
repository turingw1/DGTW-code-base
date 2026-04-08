from __future__ import annotations


def print_legacy_notice(entrypoint: str) -> None:
    print(
        f"{entrypoint}: legacy DG-TWFD entrypoint. "
        "New refactor target is scripts/run_train.py, scripts/run_eval.py, and scripts/run_sample.py.",
        flush=True,
    )
