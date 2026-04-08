from __future__ import annotations

from pathlib import Path
import sys


def ensure_flow_matching_on_path() -> None:
    root = Path(__file__).resolve().parents[3]
    vendor_root = root / "flow_matching"
    if str(vendor_root) not in sys.path:
        sys.path.insert(0, str(vendor_root))


def build_path(config: dict):
    ensure_flow_matching_on_path()
    from flow_matching.path import CondOTProbPath

    path_name = config["path"]["name"]
    if path_name == "condot":
        return CondOTProbPath()
    if path_name == "ot":
        return CondOTProbPath()
    raise ValueError(f"Unsupported path: {path_name}")
