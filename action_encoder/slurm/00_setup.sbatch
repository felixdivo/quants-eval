#!/usr/bin/env bash
#SBATCH --job-name=quants-ae-setup
#SBATCH --account=jackal_ai
#SBATCH --partition=all
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=action_encoder/logs/%x-%j.out
#SBATCH --error=action_encoder/logs/%x-%j.err

set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root before submitting}"
ACTION_ROOT="${PROJECT_ROOT}/action_encoder"
UV_BIN="${UV_BIN:-uv}"
export HF_HOME="${ACTION_ROOT}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export UV_CACHE_DIR="${ACTION_ROOT}/.cache/uv"
export UV_PROJECT_ENVIRONMENT="${ACTION_ROOT}/.venv"
export NLTK_DATA="${ACTION_ROOT}/.cache/nltk_data"

cd "${PROJECT_ROOT}"
"${UV_BIN}" sync --frozen --project "${ACTION_ROOT}" --extra dev --extra analysis
"${UV_BIN}" run --frozen --project "${ACTION_ROOT}" \
  python -m nltk.downloader -d "${NLTK_DATA}" wordnet omw-1.4
"${UV_BIN}" run --frozen --project "${ACTION_ROOT}" pytest "${ACTION_ROOT}/tests"
