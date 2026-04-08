from __future__ import annotations

import bisect
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
import yaml

from dgfm.targets.pair_sampling import sample_pair_indices


class TrajectoryShardPairDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, config: dict, split: str) -> None:
        self.config = config
        self.split = split
        self.target_cfg = config.get("target", {})
        self.root = Path(str(self.target_cfg.get("shard_root", "")))
        if not self.root.exists():
            raise FileNotFoundError(
                f"Teacher trajectory shard root not found: {self.root}. "
                "Run scripts/prepare_teacher_trajectories.py first."
            )
        split_root = self.root / split
        if split_root.exists():
            self.shard_files = sorted(split_root.glob(str(self.target_cfg.get("trajectory_file_glob", "*.pt"))))
            self.split_root = split_root
        else:
            self.shard_files = sorted(self.root.glob(str(self.target_cfg.get("trajectory_file_glob", "*.pt"))))
            self.split_root = self.root
        if not self.shard_files:
            raise FileNotFoundError(f"No trajectory shards found under {self.split_root}")

        self.cache_limit = int(self.target_cfg.get("cache_limit", 2))
        self._cache: OrderedDict[int, list[dict[str, Any]]] = OrderedDict()
        self.shard_lengths = self._resolve_shard_lengths()
        self.cumulative_sizes: list[int] = []
        running = 0
        for length in self.shard_lengths:
            running += int(length)
            self.cumulative_sizes.append(running)
        self.total_samples = running

    def __len__(self) -> int:
        return self.total_samples

    def _resolve_shard_lengths(self) -> list[int]:
        manifest_path = self.split_root / "manifest.yaml"
        if manifest_path.exists():
            payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
            shards = payload.get("shards", [])
            count_by_file = {
                str(item["file"]): int(item["count"])
                for item in shards
                if isinstance(item, dict) and "file" in item and "count" in item
            }
            lengths = [count_by_file.get(path.name, -1) for path in self.shard_files]
            if lengths and all(length > 0 for length in lengths):
                return lengths
        first_samples = torch.load(self.shard_files[0], map_location="cpu", weights_only=False)
        last_samples = torch.load(self.shard_files[-1], map_location="cpu", weights_only=False)
        if not isinstance(first_samples, list) or not isinstance(last_samples, list):
            raise TypeError("Trajectory shard must contain a list of sample dicts")
        lengths = [len(first_samples)] * len(self.shard_files)
        lengths[-1] = len(last_samples)
        return lengths

    def _load_shard(self, shard_index: int) -> list[dict[str, Any]]:
        cached = self._cache.get(shard_index)
        if cached is not None:
            self._cache.move_to_end(shard_index)
            return cached
        samples = torch.load(self.shard_files[shard_index], map_location="cpu", weights_only=False)
        if not isinstance(samples, list):
            raise TypeError(f"Trajectory shard must contain a list of sample dicts: {self.shard_files[shard_index]}")
        self._cache[shard_index] = samples
        self._cache.move_to_end(shard_index)
        while len(self._cache) > self.cache_limit:
            self._cache.popitem(last=False)
        return samples

    def _get_sample(self, index: int) -> dict[str, Any]:
        if index < 0:
            index = self.total_samples + index
        shard_index = bisect.bisect_right(self.cumulative_sizes, index)
        shard_start = 0 if shard_index == 0 else self.cumulative_sizes[shard_index - 1]
        sample_index = index - shard_start
        return self._load_shard(shard_index)[sample_index]

    def _sample_pair(self, t_grid: torch.Tensor, x_grid: torch.Tensor) -> dict[str, torch.Tensor]:
        t_indices, s_indices = sample_pair_indices(
            num_points=len(t_grid),
            target_cfg=self.target_cfg,
            batch_size=1,
            device=t_grid.device,
        )
        t_index = int(t_indices[0].item())
        s_index = int(s_indices[0].item())
        return {
            "x_t": x_grid[t_index].float(),
            "x_s": x_grid[s_index].float(),
            "t": t_grid[t_index].float(),
            "s": t_grid[s_index].float(),
            "x_0": x_grid[0].float(),
            "x_1": x_grid[-1].float(),
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self._get_sample(index)
        t_grid = torch.as_tensor(sample["t_grid"]).float().view(-1)
        x_grid = torch.as_tensor(sample["x_grid"]).float()
        if x_grid.ndim == 5 and x_grid.shape[0] == 1:
            x_grid = x_grid.squeeze(0)
        if x_grid.ndim != 4:
            raise ValueError("x_grid must have shape [M, C, H, W] or [1, M, C, H, W]")
        order = torch.argsort(t_grid)
        return self._sample_pair(t_grid[order], x_grid[order])


def build_trajectory_dataloaders(config: dict) -> dict[str, DataLoader]:
    train_cfg = config["train"]
    batch_size = int(train_cfg["batch_size"])
    num_workers = int(train_cfg.get("num_workers", 4))
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": bool(train_cfg.get("pin_memory", True)),
    }
    if num_workers > 0:
        common["persistent_workers"] = bool(train_cfg.get("persistent_workers", True))
        common["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 4))
    train_set = TrajectoryShardPairDataset(config=config, split="train")
    val_set = TrajectoryShardPairDataset(config=config, split="val")
    return {
        "train": DataLoader(train_set, shuffle=True, drop_last=True, **common),
        "val": DataLoader(val_set, shuffle=False, drop_last=False, **common),
    }
