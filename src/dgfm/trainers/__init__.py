from .baseline import BaselineTrainer
from .map import MapTrainer


def build_trainer(config, roots):
    objective = str(config.get("train", {}).get("objective", "flow_matching_velocity"))
    if objective in {"flow_matching_velocity", "velocity_fm"}:
        return BaselineTrainer(config=config, roots=roots)
    if objective in {"explicit_map", "map_branch"}:
        return MapTrainer(config=config, roots=roots)
    raise ValueError(f"Unsupported train.objective: {objective}")


__all__ = ["BaselineTrainer", "MapTrainer", "build_trainer"]
