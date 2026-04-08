# DGFM Map Branch Public

This repository is the stripped public release of the explicit-map `dgfm`
experiment system. It keeps only the files needed to run the formal
experiments recorded in:

- [docs/experiments/map_branch/EXPERIMENT_LOG.md](docs/experiments/map_branch/EXPERIMENT_LOG.md)

The stable operational procedure lives in:

- [docs/experiments/map_branch/A100_PIPELINE.md](docs/experiments/map_branch/A100_PIPELINE.md)

Public scope:

- committed experiment configs
- train / eval / multi-step panel entrypoints
- official `.npz` sample export and metrics bridge
- held-out defect evaluator
- ImageNet64 preprocessing and baseline smoke path

Omitted from the public release:

- private development notes and planning docs
- local artifacts, checkpoints, and one-off diagnostics
- unused research snapshots not required by `EXPERIMENT_LOG`

## Environment

Two installation paths are included:

1. Exact project script:
   - [scripts/experiments/create_map_branch_env.sh](scripts/experiments/create_map_branch_env.sh)
2. Reviewable conda spec:
   - [environment.yml](environment.yml)

The script is the recommended path for CUDA servers because it installs
the expected torch/torchvision wheels and project dependencies.

## Main code entry points

Core runtime scripts:

- train: [scripts/run_train.py](scripts/run_train.py)
- eval: [scripts/run_eval.py](scripts/run_eval.py)
- qualitative multi-step panel: [scripts/run_multistep_panel.py](scripts/run_multistep_panel.py)
- official sample export: [scripts/run_export_samples_npz.py](scripts/run_export_samples_npz.py)
- official metrics: [scripts/run_evaluate_metrics.py](scripts/run_evaluate_metrics.py)
- held-out defect: [scripts/run_evaluate_defect.py](scripts/run_evaluate_defect.py)
- experiment activation: [scripts/experiments/activate_fm_cifar10.sh](scripts/experiments/activate_fm_cifar10.sh)

Main source modules:

- config loader: [src/dgfm/config/loader.py](src/dgfm/config/loader.py)
- map model: [src/dgfm/models/map.py](src/dgfm/models/map.py)
- map trainer: [src/dgfm/trainers/map.py](src/dgfm/trainers/map.py)
- target construction: [src/dgfm/targets/builder.py](src/dgfm/targets/builder.py)
- map sampler: [src/dgfm/samplers/map_sampler.py](src/dgfm/samplers/map_sampler.py)
- evaluation runner: [src/dgfm/evaluators/runner.py](src/dgfm/evaluators/runner.py)

## Quick start

```bash
bash scripts/experiments/create_map_branch_env.sh dgfm_map
conda activate /cache/Zhengwei/conda_envs/dgfm_map
source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s1_e6_budget_full e602a
python scripts/run_train.py --config $FM_CONFIG --run-root $RUN_ROOT --verbose
```
