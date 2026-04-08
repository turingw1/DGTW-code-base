from .map_eval import MapEvaluationRunner
from .runner import EvaluationRunner
from .qualitative import save_multistep_qualitative_panel
from .timewarp_sampling import TimewarpSamplingRunner
from .timewarp_search import TimewarpSearchRunner


def build_evaluator(config, checkpoint, eval_root):
    objective = str(config.get("train", {}).get("objective", "flow_matching_velocity"))
    if objective in {"explicit_map", "map_branch"}:
        return MapEvaluationRunner(config=config, checkpoint=checkpoint, eval_root=eval_root)
    return EvaluationRunner(config=config, checkpoint=checkpoint, eval_root=eval_root)


__all__ = [
    "EvaluationRunner",
    "MapEvaluationRunner",
    "save_multistep_qualitative_panel",
    "TimewarpSamplingRunner",
    "TimewarpSearchRunner",
    "build_evaluator",
]
