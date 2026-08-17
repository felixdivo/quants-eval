#!/usr/bin/env bash
#SBATCH --job-name=quants_download
#SBATCH --account=jackal_ai
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
source "${QUANTS_ABLATION_ROOT:?}/slurm/common.sh"

DATA_ROOT="$QUANTS_ABLATION_ROOT/data/raw/quants"
mkdir -p "$DATA_ROOT"

"$UV_BIN" tool run --from 'huggingface_hub==0.35.1' \
    hf download dasyd/quants \
    --repo-type dataset \
    --revision 0e78849313d3b043a0b08697dc004fb6cba15df9 \
    --local-dir "$DATA_ROOT"

python3 "$QUANTS_ABLATION_ROOT/src/quants_ablation/download_manifest.py" \
    --root "$DATA_ROOT" \
    --output "$DATA_ROOT/manifest.json"
