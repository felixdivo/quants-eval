#!/usr/bin/env bash
#SBATCH --job-name=quants-ae-data
#SBATCH --account=jackal_ai
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=action_encoder/logs/%x-%j.out
#SBATCH --error=action_encoder/logs/%x-%j.err

set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root before submitting}"
: "${DATA_ROOT:?Set DATA_ROOT to the workspace-local dataset destination}"
ACTION_ROOT="${PROJECT_ROOT}/action_encoder"
export HF_HOME="${ACTION_ROOT}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}"
srun "${ACTION_ROOT}/.venv/bin/python" \
  "${ACTION_ROOT}/analysis/download_data.py" \
  --destination "${DATA_ROOT}" \
  --cache-dir "${HF_HOME}/hub"
