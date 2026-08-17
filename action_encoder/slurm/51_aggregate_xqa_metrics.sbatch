#!/usr/bin/env bash
#SBATCH --job-name=quants-xqa-aggregate
#SBATCH --account=jackal_ai
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=action_encoder/logs/%x-%j.out
#SBATCH --error=action_encoder/logs/%x-%j.err

set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root before submitting}"
ACTION_ROOT="${PROJECT_ROOT}/action_encoder"
SPLIT="${SPLIT:-test}"
shopt -s nullglob
INPUTS=("${ACTION_ROOT}"/results/xqa_metrics/xqa_*_"${SPLIT}"_metrics.csv)
shopt -u nullglob
if [[ "${#INPUTS[@]}" -ne 6 ]]; then
  printf 'expected 6 xQA metric files, found %s\n' "${#INPUTS[@]}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
srun "${ACTION_ROOT}/.venv/bin/quants-xqa-aggregate" \
  --inputs "${INPUTS[@]}" \
  --output "${ACTION_ROOT}/results/xqa_metrics_by_config_and_split.csv"
