#!/usr/bin/env bash
#SBATCH --job-name=quants_generate_ts
#SBATCH --account=jackal_ai
#SBATCH --partition=all
#SBATCH --array=0-2%3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:h100:4
#SBATCH --time=4-00:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
source "${QUANTS_ABLATION_ROOT:?}/slurm/common.sh"

TASKS=(binary multi open)
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

"$UV_BIN" run torchrun --standalone --nproc-per-node=4 \
    -m quants_ablation.generate \
    --ablation ts_only \
    --task "$TASK" \
    --data-root "$QUANTS_ABLATION_ROOT/data/raw/quants" \
    --cache-root "$QUANTS_ABLATION_ROOT/data/prepared" \
    --model "$QUANTS_ABLATION_ROOT/models/llama-3.1-8b" \
    --adapter "$QUANTS_ABLATION_ROOT/artifacts/ts_only_${TASK}/seed42/final_adapter" \
    --results-root "$QUANTS_ABLATION_ROOT/results"
