#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "This script must be sourced, not executed." >&2
  echo "Example: source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s1_e6_budget_full e602a" >&2
  exit 1
fi

variant="${1:-fm_cifar10_map_branch_s1_e6_budget_full}"
tag="${2:-dev}"

case "${variant}" in
  baseline)
    export FM_CONFIG="configs/experiment/fm_cifar10_baseline.yaml"
    exp_prefix="fm_cifar10_baseline"
    ;;
  map_branch)
    export FM_CONFIG="configs/experiment/fm_cifar10_map_branch.yaml"
    exp_prefix="fm_cifar10_map_branch"
    ;;
  map_branch_quick)
    export FM_CONFIG="configs/experiment/fm_cifar10_map_branch_quick.yaml"
    exp_prefix="fm_cifar10_map_branch_quick"
    ;;
  map_branch_timewarp_probe)
    export FM_CONFIG="configs/experiment/fm_cifar10_map_branch_timewarp_probe.yaml"
    exp_prefix="fm_cifar10_map_branch_timewarp_probe"
    ;;
  map_branch_timewarp_smoke)
    export FM_CONFIG="configs/experiment/fm_cifar10_map_branch_timewarp_smoke.yaml"
    exp_prefix="fm_cifar10_map_branch_timewarp_smoke"
    ;;
  stable)
    export FM_CONFIG="configs/experiment/fm_cifar10_stable.yaml"
    exp_prefix="fm_cifar10_stable"
    ;;
  *)
    candidate="configs/experiment/${variant}.yaml"
    if [[ -f "${candidate}" ]]; then
      export FM_CONFIG="${candidate}"
      exp_prefix="${variant}"
    else
      echo "Unknown FM variant: ${variant}" >&2
      echo "Expected any config stem under configs/experiment/ that is listed in docs/experiments/map_branch/EXPERIMENT_LOG.md" >&2
      return 1
    fi
    ;;
esac

export PROJ="${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export ENV_NAME="${ENV_NAME:-consistency}"
export DATA_ROOT="${DATA_ROOT:-/cache/Zhengwei/datasets}"
export RUNS_ROOT="${RUNS_ROOT:-/cache/Zhengwei/dgfm_runs}"
export EVAL_ROOT="${EVAL_ROOT:-/cache/Zhengwei/dgfm_eval}"
export REF_ROOT="${REF_ROOT:-/cache/Zhengwei/dgfm_refs}"
export TRAJ_ROOT="${TRAJ_ROOT:-/cache/Zhengwei/dgfm_teacher_traj/cifar10_ddpm128_p33}"
export IMAGENET_RAW_ROOT="${IMAGENET_RAW_ROOT:-/cache/Zhengwei/datasets/imagenet_raw}"
export IMAGENET64_PREPROCESSED="${IMAGENET64_PREPROCESSED:-/cache/Zhengwei/datasets/imagenet64}"
export IMAGENET64_REFERENCE_NPZ="${IMAGENET64_REFERENCE_NPZ:-${REF_ROOT}/VIRTUAL_imagenet64_labeled.npz}"
export OFFICIAL_REFERENCE_NPZ="${OFFICIAL_REFERENCE_NPZ:-}"
export IMAGENET64_TEACHER_CKPT="${IMAGENET64_TEACHER_CKPT:-checkpoints/teachers/edm_imagenet64_ema.pt}"
export HF_HOME="${HF_HOME:-/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export DGFM_TORCH_FIDELITY_MIRROR_PREFIX="${DGFM_TORCH_FIDELITY_MIRROR_PREFIX:-https://githubfast.com/}"
export EXP_VARIANT="${variant}"
export EXP_TAG="${tag}"
export EXP_NAME="${exp_prefix}_${tag}"
export FM_EXP="${exp_prefix}_${tag}"
export EXP_SOURCE="${FM_CONFIG}"
export RUN_ROOT="${RUNS_ROOT}/${FM_EXP}"
export CKPT_DIR="${RUN_ROOT}/checkpoints"
export SAMPLE_ROOT="${RUN_ROOT}/samples"
export LOG_ROOT="${RUN_ROOT}/logs"
export METRIC_ROOT="${EVAL_ROOT}/${FM_EXP}"
export ARCHIVE_ROOT="${ARCHIVE_ROOT:-/temp/Zhengwei/dgfm_runs/${FM_EXP}}"
export DGFM_ARCHIVE_ROOT="${DGFM_ARCHIVE_ROOT:-${ARCHIVE_ROOT}}"
export TORCH_CACHE_ROOT="${TORCH_CACHE_ROOT:-/cache/Zhengwei/torch_home}"
export TORCH_HOME="${TORCH_HOME:-${TORCH_CACHE_ROOT}}"
mkdir -p "${CKPT_DIR}" "${SAMPLE_ROOT}" "${LOG_ROOT}" "${METRIC_ROOT}" "${DGFM_ARCHIVE_ROOT}" 2>/dev/null || true
mkdir -p "${TORCH_HOME}" 2>/dev/null || true
mkdir -p "${REF_ROOT}" 2>/dev/null || true

echo "Activated dgfm experiment"
echo "  variant=${variant}"
echo "  EXP_VARIANT=${EXP_VARIANT}"
echo "  EXP_TAG=${EXP_TAG}"
echo "  EXP_NAME=${EXP_NAME}"
echo "  EXP_SOURCE=${EXP_SOURCE}"
echo "  FM_CONFIG=${FM_CONFIG}"
echo "  FM_EXP=${FM_EXP}"
echo "  RUN_ROOT=${RUN_ROOT}"
echo "  METRIC_ROOT=${METRIC_ROOT}"
echo "  TRAJ_ROOT=${TRAJ_ROOT}"
echo "  REF_ROOT=${REF_ROOT}"
echo "  IMAGENET_RAW_ROOT=${IMAGENET_RAW_ROOT}"
echo "  IMAGENET64_PREPROCESSED=${IMAGENET64_PREPROCESSED}"
echo "  IMAGENET64_REFERENCE_NPZ=${IMAGENET64_REFERENCE_NPZ}"
echo "  OFFICIAL_REFERENCE_NPZ=${OFFICIAL_REFERENCE_NPZ}"
echo "  IMAGENET64_TEACHER_CKPT=${IMAGENET64_TEACHER_CKPT}"
echo "  HF_HOME=${HF_HOME}"
echo "  HF_HUB_CACHE=${HF_HUB_CACHE}"
echo "  HF_ENDPOINT=${HF_ENDPOINT}"
echo "  DGFM_TORCH_FIDELITY_MIRROR_PREFIX=${DGFM_TORCH_FIDELITY_MIRROR_PREFIX}"
echo "  DGFM_ARCHIVE_ROOT=${DGFM_ARCHIVE_ROOT}"
