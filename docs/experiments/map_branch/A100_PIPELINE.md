# MAP Branch Experiment Pipeline

## 1. Environment

If the server does not already have a usable conda environment, create one
first:

```bash
cd <public-repo-root>
bash scripts/experiments/create_map_branch_env.sh dgfm_map
```

Then use:

```bash
cd <public-repo-root>
conda activate /cache/Zhengwei/conda_envs/dgfm_map
```

## 2. Activate experiment

Before entering the pipeline, first update
[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

Then activate the selected experiment once:

```bash
source scripts/experiments/activate_fm_cifar10.sh <EXP_VARIANT> <EXP_TAG>
```

`<EXP_VARIANT>` can be:
- any config stem under `configs/experiment/`
- for example:
  - `fm_cifar10_map_branch_s1_e1_ctm_ema`
  - `fm_cifar10_map_branch_s2_official_metrics`
  - `fm_imagenet64_baseline_smoke`

Current policy:
- do not use `--set` in formal experiments
- encode every experiment difference in a committed config file
- treat
  [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)
  as the only experiment-entry source of truth

This sets stable environment variables for all later commands:
- `FM_CONFIG`
- `RUN_ROOT`
- `CKPT_DIR`
- `SAMPLE_ROOT`
- `LOG_ROOT`
- `METRIC_ROOT`
- `HF_HOME`
- `HF_HUB_CACHE`
- `HF_ENDPOINT`
- `TORCH_HOME`
- `REF_ROOT`
- `OFFICIAL_REFERENCE_NPZ`
- `IMAGENET_RAW_ROOT`
- `IMAGENET64_PREPROCESSED`
- `IMAGENET64_REFERENCE_NPZ`
- `IMAGENET64_TEACHER_CKPT`

## 3. Dataset preparation

### CIFAR-10

Manual-first check:

```bash
python scripts/build_dataset.py --dataset cifar10 --data-root $DATA_ROOT/cifar10
```

If missing and you explicitly want download:

```bash
python scripts/build_dataset.py --dataset cifar10 --data-root $DATA_ROOT/cifar10 --download
```

### ImageNet64

Check whether the preprocessed folder is already present:

```bash
python scripts/build_dataset.py --dataset imagenet64 --data-root $IMAGENET64_PREPROCESSED
```

If you already have raw ILSVRC-style class folders, preprocess them into the
ImageFolder layout used by the baseline smoke:

```bash
python scripts/prepare_imagenet64.py \
  --source-root $IMAGENET_RAW_ROOT/ILSVRC/Data/CLS-LOC/train \
  --output-root $IMAGENET64_PREPROCESSED/train
```

Notes:
- the current baseline smoke uses a preprocessed ImageFolder-style layout
- the current map-branch ImageNet64 teacher path is still treated as external
  metadata, not as a finished in-framework training backend

## 4. Stable command families

### Train

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_train.py \
  --config $FM_CONFIG \
  --run-root $RUN_ROOT \
  --verbose
```

### Eval

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_eval.py \
  --config $FM_CONFIG \
  --checkpoint $CKPT_DIR/best.pt \
  --eval-root $METRIC_ROOT \
  --steps 1 2 4 8 16 32 64 128 256
```

### Multi-step panel

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_multistep_panel.py \
  --config $FM_CONFIG \
  --checkpoint $CKPT_DIR/best.pt \
  --output-dir $SAMPLE_ROOT/multistep_panel \
  --steps 1 2 4 8 16 32 64 128 256 \
  --num-examples 8 \
  --fixed-seed 42
```

### Official sample export

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_export_samples_npz.py \
  --config $FM_CONFIG \
  --checkpoint $CKPT_DIR/best.pt \
  --out $METRIC_ROOT/official/step16_samples.npz \
  --steps 16
```

This writes:
- `arr_0=[N,H,W,C] uint8`
- optional `labels`

### Official metrics

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_evaluate_metrics.py \
  --config $FM_CONFIG \
  --samples $METRIC_ROOT/official/step16_samples.npz \
  --reference ${OFFICIAL_REFERENCE_NPZ:-$IMAGENET64_REFERENCE_NPZ} \
  --out $METRIC_ROOT/official/step16_metrics.json
```

Use:
- `OFFICIAL_REFERENCE_NPZ` for the current dataset if you have a dataset-specific
  reference batch
- `IMAGENET64_REFERENCE_NPZ` for ImageNet64 official-style evaluation

### Held-out defect

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_evaluate_defect.py \
  --config $FM_CONFIG \
  --checkpoint $CKPT_DIR/best.pt \
  --out $METRIC_ROOT/defect/heldout.json
```

## 5. What to inspect after each run

Stage 1:
- `$LOG_ROOT/train.jsonl`
- `$METRIC_ROOT/reports/summary.json`
- `$SAMPLE_ROOT/multistep_panel/multistep_panel.png`

Stage 2:
- `$METRIC_ROOT/official/*.json`
- `$METRIC_ROOT/defect/*.json`

Key readout:
- few-step FID trend
- `train_update_ratio / val_update_ratio`
- `train_update_cosine / val_update_cosine`
- `timewarp_time_grid`
- `timewarp_interval_defects`
- official `fid / inception_score_mean / precision / recall`
- held-out `defect_mean / defect_by_t_bin / defect_by_step_count`

## 6. Cache notes

- `TORCH_HOME` defaults to `/cache/Zhengwei/torch_home`
- `HF_HOME` defaults to `/cache/huggingface`
- `DGFM_ARCHIVE_ROOT` mirrors logs/checkpoints/samples into `/temp/Zhengwei/...`
- `REF_ROOT` defaults to `/cache/Zhengwei/dgfm_refs`
