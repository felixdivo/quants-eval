#!/usr/bin/env bash
#SBATCH --job-name=quants-ae-train
#SBATCH --account=jackal_ai
#SBATCH --partition=h200,all
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=action_encoder/logs/%x-%A_%a.out
#SBATCH --error=action_encoder/logs/%x-%A_%a.err

set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root before submitting}"
: "${DATA_ROOT:?Set DATA_ROOT to the verified local QuAnTS snapshot}"
ACTION_ROOT="${PROJECT_ROOT}/action_encoder"
TASKS=(binary multi open)
TASK="${TASKS[${SLURM_ARRAY_TASK_ID}]}"
export HF_HOME="${ACTION_ROOT}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCH_HOME="${ACTION_ROOT}/.cache/torch"
export PYTHONUNBUFFERED=1

OUTPUT_DIR="${ACTION_ROOT}/artifacts/action_encoder/${TASK}"
RESUME_ARGS=()
shopt -s nullglob
for manifest in "${OUTPUT_DIR}"/checkpoint-epoch-*/manifest.json; do
  checkpoint_dir="${manifest%/manifest.json}"
  if [[ -f "${checkpoint_dir}/training_state.pt" && -f "${checkpoint_dir}/model/config.json" ]]; then
    RESUME_ARGS=(--resume-from-checkpoint)
    break
  fi
done
shopt -u nullglob

cd "${PROJECT_ROOT}"
srun "${ACTION_ROOT}/.venv/bin/quants-action-train" \
  --task "${TASK}" \
  --output-dir "${OUTPUT_DIR}" \
  --data-root "${DATA_ROOT}" \
  --cache-dir "${HF_DATASETS_CACHE}" \
  --preflight-manifest "${ACTION_ROOT}/data/preflight_manifest.json" \
  --epochs 5 \
  --batch-size 512 \
  --learning-rate 1e-4 \
  --seed 42 \
  --skip-data-validation \
  "${RESUME_ARGS[@]}"
