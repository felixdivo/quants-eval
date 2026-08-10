#!/usr/bin/env bash
#SBATCH --job-name=quants_train_naive
#SBATCH --account=jackal_ai
#SBATCH --partition=all
#SBATCH --array=0-2%3
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --gres=gpu:h100:4
#SBATCH --time=4-00:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
source "${QUANTS_ABLATION_ROOT:?}/slurm/common.sh"

TASKS=(binary multi open)
export TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((29600 + SLURM_ARRAY_TASK_ID))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

srun --kill-on-bad-exit=1 bash -c '
    "$UV_BIN" run torchrun \
        --nnodes="$SLURM_NNODES" \
        --nproc-per-node=4 \
        --node-rank="$SLURM_NODEID" \
        --master-addr="$MASTER_ADDR" \
        --master-port="$MASTER_PORT" \
        -m quants_ablation.train \
        --ablation naive \
        --task "$TASK" \
        --data-root "$QUANTS_ABLATION_ROOT/data/raw/quants" \
        --cache-root "$QUANTS_ABLATION_ROOT/data/prepared" \
        --model "$QUANTS_ABLATION_ROOT/models/llama-3.1-8b" \
        --output "$QUANTS_ABLATION_ROOT/artifacts/naive_${TASK}/seed42" \
        --micro-batch-size 1 \
        --gradient-accumulation 1
'
