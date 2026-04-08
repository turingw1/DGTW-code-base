from __future__ import annotations

from .runner import EvaluationRunner


class MapEvaluationRunner(EvaluationRunner):
    """Explicit map branch evaluation runner.

    Currently reuses the shared evaluation implementation and relies on the
    config-driven objective dispatch in evaluators.common.
    """

    pass
