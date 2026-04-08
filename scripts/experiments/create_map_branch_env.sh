#!/usr/bin/env bash

set -euo pipefail

ENV_NAME="${1:-dgfm_map}"
ENV_ROOT="${2:-/cache/Zhengwei/conda_envs}"
ENV_PREFIX="${ENV_ROOT}/${ENV_NAME}"
WHEEL_ROOT="${WHEEL_ROOT:-/cache/Zhengwei/wheels}"
TORCH_WHL_DIR="${TORCH_WHL_DIR:-${WHEEL_ROOT}/torch-cu128}"
TORCH_WHL_URL="${TORCH_WHL_URL:-}"
PIP_INDEX_URL_DEFAULT="${PIP_INDEX_URL_DEFAULT:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_EXTRA_INDEX_URL_DEFAULT="${PIP_EXTRA_INDEX_URL_DEFAULT:-}"

echo "Creating conda environment:"
echo "  name:   ${ENV_NAME}"
echo "  prefix: ${ENV_PREFIX}"

mkdir -p "${ENV_ROOT}"
conda create -p "${ENV_PREFIX}" python=3.10 -y

eval "$(conda shell.bash hook)"
conda activate "${ENV_PREFIX}"

python -m pip install --upgrade pip setuptools wheel
python -m pip config set global.index-url "${PIP_INDEX_URL_DEFAULT}"
if [[ -n "${PIP_EXTRA_INDEX_URL_DEFAULT}" ]]; then
  python -m pip config set global.extra-index-url "${PIP_EXTRA_INDEX_URL_DEFAULT}"
fi

mkdir -p "${TORCH_WHL_DIR}"

if compgen -G "${TORCH_WHL_DIR}/torch-2.10.0+cu128-*.whl" >/dev/null && compgen -G "${TORCH_WHL_DIR}/torchvision-0.25.0+cu128-*.whl" >/dev/null; then
  echo "Using local torch wheel cache under ${TORCH_WHL_DIR}"
else
  if [[ -n "${TORCH_WHL_URL}" ]]; then
    echo "Downloading torch wheels from TORCH_WHL_URL"
    wget -c -P "${TORCH_WHL_DIR}" "${TORCH_WHL_URL%/}/torch-2.10.0%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl"
    wget -c -P "${TORCH_WHL_DIR}" "${TORCH_WHL_URL%/}/torchvision-0.25.0%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl"
  else
    echo "Falling back to download.pytorch.org for torch wheels"
    python -m pip download \
      --dest "${TORCH_WHL_DIR}" \
      --index-url https://download.pytorch.org/whl/cu128 \
      --no-deps \
      torch==2.10.0 torchvision==0.25.0
  fi
fi

python -m pip install \
  "${TORCH_WHL_DIR}"/torch-2.10.0+cu128-*.whl \
  "${TORCH_WHL_DIR}"/torchvision-0.25.0+cu128-*.whl

python -m pip install \
  PyYAML==6.0.3 \
  numpy==2.2.3 \
  scipy==1.15.3 \
  torch-fidelity==0.4.0 \
  'diffusers>=0.30' \
  'transformers>=4.40' \
  'accelerate>=0.30' \
  'safetensors>=0.4' \
  'piq>=0.8' \
  matplotlib \
  pillow \
  pytest

python -m pip install -e .

echo
echo "Environment ready:"
echo "  prefix: ${ENV_PREFIX}"
echo "  pip index: ${PIP_INDEX_URL_DEFAULT}"
echo "  torch wheel dir: ${TORCH_WHL_DIR}"
echo "Next:"
echo "  conda activate ${ENV_PREFIX}"
echo "  source scripts/experiments/activate_fm_cifar10.sh fm_cifar10_map_branch_s1_e6_budget_full e602a"
