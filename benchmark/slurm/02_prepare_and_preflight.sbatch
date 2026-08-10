#!/usr/bin/env bash
#SBATCH --job-name=quants_preflight
#SBATCH --account=jackal_ai
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
source "${QUANTS_ABLATION_ROOT:?}/slurm/common.sh"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

DATA_ROOT="$QUANTS_ABLATION_ROOT/data/raw/quants"
CACHE_ROOT="$QUANTS_ABLATION_ROOT/data/prepared"
MODEL="$QUANTS_ABLATION_ROOT/models/llama-3.1-8b"

"$UV_BIN" run quants-prepare \
    --data-root "$DATA_ROOT" \
    --cache-root "$CACHE_ROOT" \
    --model "$MODEL"
"$UV_BIN" run pytest -q

for ablation in question_only ts_only naive; do
    if [[ "$ablation" == question_only ]]; then
        micro_batch=8
        accumulation=2
    else
        micro_batch=1
        accumulation=16
    fi
    for task in binary multi open; do
        "$UV_BIN" run quants-train \
            --ablation "$ablation" \
            --task "$task" \
            --data-root "$DATA_ROOT" \
            --cache-root "$CACHE_ROOT" \
            --model "$MODEL" \
            --micro-batch-size "$micro_batch" \
            --gradient-accumulation "$accumulation" \
            --max-steps 1 \
            --save-steps 1 \
            --output "$QUANTS_ABLATION_ROOT/artifacts/preflight/${ablation}_${task}"
    done
done
