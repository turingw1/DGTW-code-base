# Public Docs

This directory collects the public-facing operational documents for the
standalone `dgfm` map-branch release.

## Main documents

- experiment registry:
  [experiments/map_branch/EXPERIMENT_LOG.md](experiments/map_branch/EXPERIMENT_LOG.md)
- execution pipeline:
  [experiments/map_branch/A100_PIPELINE.md](experiments/map_branch/A100_PIPELINE.md)

## Main entry points

- environment bootstrap:
  [../scripts/experiments/create_map_branch_env.sh](../scripts/experiments/create_map_branch_env.sh)
- experiment activation:
  [../scripts/experiments/activate_fm_cifar10.sh](../scripts/experiments/activate_fm_cifar10.sh)
- train:
  [../scripts/run_train.py](../scripts/run_train.py)
- eval:
  [../scripts/run_eval.py](../scripts/run_eval.py)
- multi-step panel:
  [../scripts/run_multistep_panel.py](../scripts/run_multistep_panel.py)
- official sample export:
  [../scripts/run_export_samples_npz.py](../scripts/run_export_samples_npz.py)
- official metrics:
  [../scripts/run_evaluate_metrics.py](../scripts/run_evaluate_metrics.py)
- held-out defect:
  [../scripts/run_evaluate_defect.py](../scripts/run_evaluate_defect.py)

## Recommended reading order

1. Read [experiments/map_branch/EXPERIMENT_LOG.md](experiments/map_branch/EXPERIMENT_LOG.md)
   to choose a committed experiment variant.
2. Follow [experiments/map_branch/A100_PIPELINE.md](experiments/map_branch/A100_PIPELINE.md)
   for environment setup, activation, training, evaluation, and reporting.
3. Inspect outputs under the run root and eval root recorded by the activation
   script.
