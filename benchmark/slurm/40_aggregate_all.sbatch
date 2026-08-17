#!/usr/bin/env bash
#SBATCH --job-name=quants_aggregate_all
#SBATCH --account=jackal_ai
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
source "${QUANTS_ABLATION_ROOT:?}/slurm/common.sh"

"$UV_BIN" run quants-aggregate \
    --data-root "$QUANTS_ABLATION_ROOT/data/raw/quants" \
    --results-root "$QUANTS_ABLATION_ROOT/results" \
    --ablations question_only ts_only naive
