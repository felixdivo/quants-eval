#!/usr/bin/env bash
#SBATCH --job-name=quants-xqa-metrics
#SBATCH --account=jackal_ai
#SBATCH --partition=all
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=action_encoder/logs/%x-%A_%a.out
#SBATCH --error=action_encoder/logs/%x-%A_%a.err

set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root before submitting}"
ACTION_ROOT="${PROJECT_ROOT}/action_encoder"
TASKS=(binary multi open)
INDEX="${SLURM_ARRAY_TASK_ID}"
if (( INDEX < 3 )); then
  MODE=gt
  TASK="${TASKS[${INDEX}]}"
else
  MODE=predicted
  TASK="${TASKS[$((INDEX - 3))]}"
fi
SPLIT="${SPLIT:-test}"
CONFIG="xqa_${MODE}_${TASK}"
export HF_HOME="${ACTION_ROOT}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export NLTK_DATA="${ACTION_ROOT}/.cache/nltk_data"

DATA_ARGS=()
if [[ -n "${DATA_ROOT:-}" ]]; then
  DATA_ARGS+=(--data-root "${DATA_ROOT}")
fi
JUDGE_ARGS=()
if [[ "${TASK}" == open ]]; then
  JUDGE_ARGS+=(
    --judge-ratings
    "${ACTION_ROOT}/results/judge_${CONFIG}_${SPLIT}.csv"
  )
fi

cd "${PROJECT_ROOT}"
srun "${ACTION_ROOT}/.venv/bin/quants-xqa-metrics" \
  --predictions "${ACTION_ROOT}/results/xqa_${MODE}_${TASK}_${SPLIT}.csv" \
  --task "${TASK}" \
  --split "${SPLIT}" \
  --config "${CONFIG}" \
  --cache-dir "${HF_DATASETS_CACHE}" \
  --output "${ACTION_ROOT}/results/xqa_metrics/${CONFIG}_${SPLIT}_metrics.csv" \
  --parsed-output "${ACTION_ROOT}/results/xqa_metrics/${CONFIG}_${SPLIT}_parsed.csv" \
  "${JUDGE_ARGS[@]}" \
  "${DATA_ARGS[@]}"
