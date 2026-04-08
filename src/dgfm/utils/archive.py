from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from dgfm.config import RunRoots


@dataclass(slots=True)
class ExperimentArchive:
    roots: RunRoots

    @property
    def enabled(self) -> bool:
        return self.roots.archive_root is not None

    def prepare(self) -> None:
        if not self.enabled:
            return
        assert self.roots.archive_root is not None
        assert self.roots.archive_checkpoint_dir is not None
        assert self.roots.archive_sample_dir is not None
        assert self.roots.archive_log_dir is not None
        self.roots.archive_root.mkdir(parents=True, exist_ok=True)
        self.roots.archive_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.roots.archive_sample_dir.mkdir(parents=True, exist_ok=True)
        self.roots.archive_log_dir.mkdir(parents=True, exist_ok=True)

    def dump_yaml(self, filename: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        assert self.roots.archive_log_dir is not None
        with (self.roots.archive_log_dir / filename).open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)

    def append_jsonl(self, filename: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        assert self.roots.archive_log_dir is not None
        with (self.roots.archive_log_dir / filename).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def save_checkpoint(self, filename: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        assert self.roots.archive_checkpoint_dir is not None
        torch.save(payload, self.roots.archive_checkpoint_dir / filename)


def build_experiment_archive(roots: RunRoots) -> ExperimentArchive:
    archive = ExperimentArchive(roots=roots)
    archive.prepare()
    return archive
