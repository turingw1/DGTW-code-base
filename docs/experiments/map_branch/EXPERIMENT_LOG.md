# MAP Branch Experiment Log

This file is the single operational ledger for the public experiment phase.

Rules:
- every formal run must map to one committed config under `configs/experiment/`
- activate experiments only through
  [activate_fm_cifar10.sh](../../../scripts/experiments/activate_fm_cifar10.sh)
- do not use `--set` in formal runs
- use the stable
  [A100_PIPELINE.md](A100_PIPELINE.md)
  command families after activation
- Stage 1 chooses the winning CIFAR-10 recipe
- Stage 2 validates that winner with official metrics and held-out defect

## Naming convention

- `EXP_VARIANT`
  - exactly the config stem under `configs/experiment/`
- `EXP_TAG`
  - run id such as `e103a` or `e701a`
- `FM_EXP`
  - `${EXP_VARIANT}_${EXP_TAG}`

## Stable command families

After activation, use these commands without extra overrides.

Train:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_train.py \
  --config $FM_CONFIG \
  --run-root $RUN_ROOT \
  --verbose
```

Eval:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_eval.py \
  --config $FM_CONFIG \
  --checkpoint $CKPT_DIR/best.pt \
  --eval-root $METRIC_ROOT \
  --steps 1 2 4 8 16 32 64 128 256
```

Panel:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_multistep_panel.py \
  --config $FM_CONFIG \
  --checkpoint $CKPT_DIR/best.pt \
  --output-dir $SAMPLE_ROOT/multistep_panel \
  --steps 1 2 4 8 16 32 64 128 256 \
  --num-examples 8 \
  --fixed-seed 42
```

Official sample export:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_export_samples_npz.py \
  --config $FM_CONFIG \
  --checkpoint $CKPT_DIR/best.pt \
  --out $METRIC_ROOT/official/step16_samples.npz \
  --steps 16
```

Official metrics:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_evaluate_metrics.py \
  --config $FM_CONFIG \
  --samples $METRIC_ROOT/official/step16_samples.npz \
  --reference ${OFFICIAL_REFERENCE_NPZ:-$IMAGENET64_REFERENCE_NPZ} \
  --out $METRIC_ROOT/official/step16_metrics.json
```

Held-out defect:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_evaluate_defect.py \
  --config $FM_CONFIG \
  --checkpoint $CKPT_DIR/best.pt \
  --out $METRIC_ROOT/defect/heldout.json
```

ImageNet64 preprocessing:

```bash
python scripts/prepare_imagenet64.py \
  --source-root $IMAGENET_RAW_ROOT/ILSVRC/Data/CLS-LOC/train \
  --output-root $IMAGENET64_PREPROCESSED/train
```

## Stage 1. CIFAR-10 algorithm selection

| Group | EXP_TAG | EXP_VARIANT | FM_CONFIG | Activate | Purpose | Status |
| --- | --- | --- | --- | --- | --- | --- |
| E1 | e101a | `fm_cifar10_map_branch_s1_e1_traj_reg` | `configs/experiment/fm_cifar10_map_branch_s1_e1_traj_reg.yaml` | `source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s1_e1_traj_reg e101a` | target-construction baseline: plain trajectory regression | planned |
| E1 | e102a | `fm_cifar10_map_branch_s1_e1_ctm_teacher` | `configs/experiment/fm_cifar10_map_branch_s1_e1_ctm_teacher.yaml` | `source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s1_e1_ctm_teacher e102a` | CTM contract with teacher bridge | planned |
| E1 | e103a | `fm_cifar10_map_branch_s1_e1_ctm_ema` | `configs/experiment/fm_cifar10_map_branch_s1_e1_ctm_ema.yaml` | `source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s1_e1_ctm_ema e103a` | CTM contract with EMA rollout bridge | planned |
| E1 | e104a | `fm_cifar10_map_branch_s1_e1_ctm_current` | `configs/experiment/fm_cifar10_map_branch_s1_e1_ctm_current.yaml` | `source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s1_e1_ctm_current e104a` | CTM contract with current-model rollout bridge | planned |
| E2 | e201a | `fm_cifar10_map_branch_s1_e2_defect_probe` | `configs/experiment/fm_cifar10_map_branch_s1_e2_defect_probe.yaml` | `source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s1_e2_defect_probe e201a` | defect-probe run with learnable timewarp diagnostics | planned |
| E3 | e301a | `fm_cifar10_map_branch_s1_e3_pred_residual` | `configs/experiment/fm_cifar10_map_branch_s1_e3_pred_residual.yaml` | `source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s1_e3_pred_residual e301a` | prediction target ablation: residual map | planned |
| E3 | e302a | `fm_cifar10_map_branch_s1_e3_pred_direct` | `configs/experiment/fm_cifar10_map_branch_s1_e3_pred_direct.yaml` | `source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s1_e3_pred_direct e302a` | prediction target ablation: direct endpoint map | planned |
| E4 | e401a | `fm_cifar10_map_branch_s1_e3_pred_residual` | `configs/experiment/fm_cifar10_map_branch_s1_e3_pred_residual.yaml` | `source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s1_e3_pred_residual e401a` | auxiliary ablation baseline: endpoint off | planned |
| E4 | e402a | `fm_cifar10_map_branch_s1_e4_aux_endpoint_on` | `configs/experiment/fm_cifar10_map_branch_s1_e4_aux_endpoint_on.yaml` | `source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s1_e4_aux_endpoint_on e402a` | auxiliary ablation: endpoint on | planned |
| E5 | e501a | `fm_cifar10_map_branch_s1_e5_warp_identity` | `configs/experiment/fm_cifar10_map_branch_s1_e5_warp_identity.yaml` | `source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s1_e5_warp_identity e501a` | warp ablation: identity clock | planned |
| E5 | e502a | `fm_cifar10_map_branch_s1_e5_warp_data_dense` | `configs/experiment/fm_cifar10_map_branch_s1_e5_warp_data_dense.yaml` | `source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s1_e5_warp_data_dense e502a` | warp ablation: static data-dense power warp | planned |
| E5 | e503a | `fm_cifar10_map_branch_s1_e5_warp_source_dense` | `configs/experiment/fm_cifar10_map_branch_s1_e5_warp_source_dense.yaml` | `source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s1_e5_warp_source_dense e503a` | warp ablation: static source-dense power warp | planned |
| E5 | e504a | `fm_cifar10_map_branch_s1_e5_warp_learned` | `configs/experiment/fm_cifar10_map_branch_s1_e5_warp_learned.yaml` | `source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s1_e5_warp_learned e504a` | warp ablation: learnable monotone warp | planned |
| E5 | e505a | `fm_cifar10_map_branch_s1_e5_warp_spline` | `configs/experiment/fm_cifar10_map_branch_s1_e5_warp_spline.yaml` | `source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s1_e5_warp_spline e505a` | warp ablation: spline-mass monotone warp | planned |
| E6 | e601a | `fm_cifar10_map_branch_s1_e6_budget_quick` | `configs/experiment/fm_cifar10_map_branch_s1_e6_budget_quick.yaml` | `source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s1_e6_budget_quick e601a` | budget sensitivity: quick budget | planned |
| E6 | e602a | `fm_cifar10_map_branch_s1_e6_budget_full` | `configs/experiment/fm_cifar10_map_branch_s1_e6_budget_full.yaml` | `source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s1_e6_budget_full e602a` | budget sensitivity: full budget | planned |

## Stage 2. External-facing validation

| Group | EXP_TAG | EXP_VARIANT | FM_CONFIG | Activate | Purpose | Status |
| --- | --- | --- | --- | --- | --- | --- |
| E7 | e701a | `fm_cifar10_map_branch_s2_official_metrics` | `configs/experiment/fm_cifar10_map_branch_s2_official_metrics.yaml` | `source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s2_official_metrics e701a` | official-style `.npz` export and FID/IS/Precision/Recall bridge on the selected CIFAR-10 checkpoint | planned |
| E8 | e801a | `fm_cifar10_map_branch_s2_defect_eval` | `configs/experiment/fm_cifar10_map_branch_s2_defect_eval.yaml` | `source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s2_defect_eval e801a` | held-out semigroup defect evaluation on selected CIFAR-10 checkpoints | planned |
| E9 | e901a | `fm_imagenet64_baseline_smoke` | `configs/experiment/fm_imagenet64_baseline_smoke.yaml` | `source scripts/experiments/activate_fm_cifar10.sh fm_imagenet64_baseline_smoke e901a` | ImageNet64 data/class-cond baseline smoke and optional official metrics path | planned |

## Execution order

1. Run `E1` first and pick the best target-construction recipe.
2. Run `E2` once to check whether defect diagnostics are coherent.
3. Run `E3` to lock prediction parameterization.
4. Run `E4` to decide whether endpoint should stay.
5. Run `E5` to decide whether any warp strategy, including spline, is worth keeping.
6. Use the winning recipe to interpret `E6`.
7. Activate `E7` and reuse the selected checkpoint to produce official `.npz` metrics.
8. Activate `E8` and reuse the selected checkpoint(s) to produce held-out defect reports.
9. Run `E9` to verify the ImageNet64 data / baseline / official-eval bridge.

## Output inspection

For Stage 1 rows, inspect:
- `$LOG_ROOT/train.jsonl`
- `$METRIC_ROOT/reports/summary.json`
- `$SAMPLE_ROOT/multistep_panel/multistep_panel.png`

For Stage 2 rows, additionally inspect:
- `$METRIC_ROOT/official/*.json`
- `$METRIC_ROOT/defect/*.json`

Primary decision fields:
- FID trend from `1` to `16`
- whether improvement continues beyond `16`
- `train_update_ratio / val_update_ratio`
- `train_update_cosine / val_update_cosine`
- `timewarp_time_grid` and `timewarp_interval_defects` when warp is enabled
- official `fid / inception_score_mean / precision / recall`
- held-out `defect_mean / defect_by_t_bin / defect_by_step_count`
