#!/usr/bin/env bash
#SBATCH --job-name=quants-ae-predict
#SBATCH --account=jackal_ai
#SBATCH --partition=h200,all
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=action_encoder/logs/%x-%A_%a.out
#SBATCH --error=action_encoder/logs/%x-%A_%a.err

set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root before submitting}"
: "${DATA_ROOT:?Set DATA_ROOT to the verified local QuAnTS snapshot}"
ACTION_ROOT="${PROJECT_ROOT}/action_encoder"
TASKS=(binary multi open)
TASK="${TASKS[${SLURM_ARRAY_TASK_ID}]}"
SPLIT="${SPLIT:-test}"
export HF_HOME="${ACTION_ROOT}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}"
srun "${ACTION_ROOT}/.venv/bin/quants-action-predict" \
  --checkpoint "${ACTION_ROOT}/artifacts/action_encoder/${TASK}/best" \
  --task "${TASK}" \
  --split "${SPLIT}" \
  --data-root "${DATA_ROOT}" \
  --cache-dir "${HF_DATASETS_CACHE}" \
  --preflight-manifest "${ACTION_ROOT}/data/preflight_manifest.json" \
  --output "${ACTION_ROOT}/results/action_encoder_${TASK}_${SPLIT}.csv"
