#!/usr/bin/env bash
#SBATCH --job-name=quants-ae-metrics
#SBATCH --account=jackal_ai
#SBATCH --partition=all
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=action_encoder/logs/%x-%A_%a.out
#SBATCH --error=action_encoder/logs/%x-%A_%a.err

set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root before submitting}"
ACTION_ROOT="${PROJECT_ROOT}/action_encoder"
TASKS=(binary multi open)
TASK="${TASKS[${SLURM_ARRAY_TASK_ID}]}"
SPLIT="${SPLIT:-test}"
export HF_HOME="${ACTION_ROOT}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

DATA_ARGS=()
if [[ -n "${DATA_ROOT:-}" ]]; then
  DATA_ARGS+=(--data-root "${DATA_ROOT}")
fi

cd "${PROJECT_ROOT}"
srun "${ACTION_ROOT}/.venv/bin/quants-action-metrics" \
  --predictions "${ACTION_ROOT}/results/action_encoder_${TASK}_${SPLIT}.csv" \
  --task "${TASK}" \
  --split "${SPLIT}" \
  --output-dir "${ACTION_ROOT}/results/action_metrics/${TASK}_${SPLIT}" \
  "${DATA_ARGS[@]}"
