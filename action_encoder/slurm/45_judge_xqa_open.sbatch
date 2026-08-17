#!/usr/bin/env bash
#SBATCH --job-name=quants-xqa-judge
#SBATCH --account=jackal_ai
#SBATCH --partition=all
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=action_encoder/logs/%x-%A_%a.out
#SBATCH --error=action_encoder/logs/%x-%A_%a.err

set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root before submitting}"
: "${DATA_ROOT:?Set DATA_ROOT to the verified local QuAnTS snapshot}"
: "${JUDGE_BASE_URL:?Set JUDGE_BASE_URL to the judge OpenAI-compatible endpoint}"
: "${JUDGE_MODEL:?Set JUDGE_MODEL to the served judge model ID}"
: "${JUDGE_MODEL_REVISION:?Set JUDGE_MODEL_REVISION to the immutable judge weights revision}"
: "${JUDGE_DEPLOYMENT_FINGERPRINT:?Set JUDGE_DEPLOYMENT_FINGERPRINT to the judge server manifest digest}"
ACTION_ROOT="${PROJECT_ROOT}/action_encoder"
MODES=(gt predicted)
MODE="${MODES[${SLURM_ARRAY_TASK_ID}]}"
SPLIT="${SPLIT:-test}"
CONFIG="xqa_${MODE}_open"
export HF_HOME="${ACTION_ROOT}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}"
srun "${ACTION_ROOT}/.venv/bin/quants-xqa-judge" \
  --predictions "${ACTION_ROOT}/results/xqa_${MODE}_open_${SPLIT}.csv" \
  --split "${SPLIT}" \
  --config "${CONFIG}" \
  --data-root "${DATA_ROOT}" \
  --cache-dir "${HF_DATASETS_CACHE}" \
  --preflight-manifest "${ACTION_ROOT}/data/preflight_manifest.json" \
  --output "${ACTION_ROOT}/results/judge_${CONFIG}_${SPLIT}.csv"
