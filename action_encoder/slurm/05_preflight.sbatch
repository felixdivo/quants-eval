#!/usr/bin/env bash
#SBATCH --job-name=quants-ae-smoke
#SBATCH --account=jackal_ai
#SBATCH --partition=h200,all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=action_encoder/logs/%x-%j.out
#SBATCH --error=action_encoder/logs/%x-%j.err

set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root before submitting}"
ACTION_ROOT="${PROJECT_ROOT}/action_encoder"
export HF_HOME="${ACTION_ROOT}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCH_HOME="${ACTION_ROOT}/.cache/torch"
export PYTHONUNBUFFERED=1

SMOKE_PARENT="${ACTION_ROOT}/.cache/gpu_preflight"
mkdir -p "${SMOKE_PARENT}"
SMOKE_OUTPUT="$(mktemp -d "${SMOKE_PARENT}/job-${SLURM_JOB_ID}-XXXXXX")"
cleanup() {
  if [[ "${SMOKE_OUTPUT}" == "${SMOKE_PARENT}/"* ]]; then
    rm -rf -- "${SMOKE_OUTPUT}"
  fi
}
trap cleanup EXIT

cd "${PROJECT_ROOT}"
srun "${ACTION_ROOT}/.venv/bin/quants-action-train" \
  --task binary \
  --output-dir "${SMOKE_OUTPUT}" \
  --learning-rate 1e-4 \
  --seed 42 \
  --device cuda \
  --smoke-test

export SMOKE_MODEL="${SMOKE_OUTPUT}/smoke_model"
srun "${ACTION_ROOT}/.venv/bin/python" - <<'PY'
import os

import torch

from quants_action_encoder.model import load_action_encoder, predict_action_ids

assert torch.cuda.is_available(), "CUDA is unavailable in the GPU preflight"
past_values = torch.randn(1, 80, 72, device="cuda")
model = load_action_encoder(os.environ["SMOKE_MODEL"], device="cuda")
prediction = predict_action_ids(model, past_values)
assert prediction.shape == (1,)
assert 0 <= int(prediction.item()) < 19
print("real xLSTMMixer CUDA train/save/reload/predict preflight passed", flush=True)
PY
