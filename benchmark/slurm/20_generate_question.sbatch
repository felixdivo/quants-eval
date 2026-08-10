#!/usr/bin/env bash
#SBATCH --job-name=quants_generate_question
#SBATCH --account=jackal_ai
#SBATCH --partition=all
#SBATCH --array=0-2%3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
source "${QUANTS_ABLATION_ROOT:?}/slurm/common.sh"

TASKS=(binary multi open)
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

"$UV_BIN" run quants-generate \
    --ablation question_only \
    --task "$TASK" \
    --data-root "$QUANTS_ABLATION_ROOT/data/raw/quants" \
    --cache-root "$QUANTS_ABLATION_ROOT/data/prepared" \
    --model "$QUANTS_ABLATION_ROOT/models/llama-3.1-8b" \
    --adapter "$QUANTS_ABLATION_ROOT/artifacts/question_only_${TASK}/seed42/final_adapter" \
    --results-root "$QUANTS_ABLATION_ROOT/results"
