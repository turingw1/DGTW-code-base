from __future__ import annotations

from ast import literal_eval
from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[3]
_ENV_DEFAULT_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\:?-([^}]*)\}")


@dataclass(slots=True)
class RunRoots:
    run_root: Path
    checkpoint_dir: Path
    sample_dir: Path
    log_dir: Path
    archive_root: Path | None
    archive_checkpoint_dir: Path | None
    archive_sample_dir: Path | None
    archive_log_dir: Path | None


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _merge_dicts(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _expand_bash_style_defaults(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        default_value = match.group(2)
        env_value = os.environ.get(var_name)
        if env_value is None or env_value == "":
            return default_value
        return env_value

    return _ENV_DEFAULT_PATTERN.sub(replace, text)


def _expand_env_vars(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(_expand_bash_style_defaults(value))
    return value


def _resolve_include(path_like: str) -> Path:
    include_path = ROOT / "configs" / path_like
    if include_path.suffix != ".yaml":
        include_path = include_path.with_suffix(".yaml")
    return include_path


def _resolve_config_path(path_like: str | Path, *, base_dir: Path | None = None) -> Path:
    candidate = Path(path_like)
    if candidate.is_absolute():
        return candidate
    if base_dir is not None:
        local = (base_dir / candidate).resolve()
        if local.exists():
            return local
    rooted = (ROOT / candidate).resolve()
    if rooted.exists():
        return rooted
    if "configs" not in candidate.parts:
        return _resolve_include(str(candidate)).resolve()
    return rooted


def _load_config_tree(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    resolved_path = path.resolve()
    active = set() if seen is None else set(seen)
    if resolved_path in active:
        raise ValueError(f"Config include cycle detected at: {resolved_path}")
    active.add(resolved_path)

    loaded = _read_yaml(resolved_path)
    base_path = loaded.pop("base", None)
    includes = loaded.pop("includes", [])

    merged: dict[str, Any] = {}
    if base_path:
        merged = _merge_dicts(
            merged,
            _load_config_tree(_resolve_config_path(base_path, base_dir=resolved_path.parent), active),
        )

    for include in includes:
        merged = _merge_dicts(
            merged,
            _load_config_tree(_resolve_config_path(include, base_dir=resolved_path.parent), active),
        )

    return _merge_dicts(merged, loaded)


def _parse_override_value(raw_value: str) -> Any:
    lowered = raw_value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered == "null":
        return None
    try:
        return literal_eval(raw_value)
    except (ValueError, SyntaxError):
        return raw_value


def _apply_overrides(config_dict: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    merged = dict(config_dict)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got: {override}")
        key_path, raw_value = override.split("=", 1)
        value = _parse_override_value(raw_value)
        target = merged
        keys = key_path.split(".")
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        target[keys[-1]] = value
    return merged


def load_experiment_config(config_path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    path = _resolve_config_path(config_path)
    merged = _load_config_tree(path)
    if overrides:
        merged = _apply_overrides(merged, overrides)
    return _expand_env_vars(merged)


def resolve_run_roots(run_root: str | Path) -> RunRoots:
    root = Path(run_root)
    archive_root_raw = os.environ.get("DGFM_ARCHIVE_ROOT")
    archive_root = Path(archive_root_raw) if archive_root_raw else None
    return RunRoots(
        run_root=root,
        checkpoint_dir=root / "checkpoints",
        sample_dir=root / "samples",
        log_dir=root / "logs",
        archive_root=archive_root,
        archive_checkpoint_dir=(archive_root / "checkpoints") if archive_root else None,
        archive_sample_dir=(archive_root / "samples") if archive_root else None,
        archive_log_dir=(archive_root / "logs") if archive_root else None,
    )
