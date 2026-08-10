#!/usr/bin/env bash
#SBATCH --job-name=quants_setup
#SBATCH --account=jackal_ai
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
source "${QUANTS_ABLATION_ROOT:?}/slurm/common.sh"

if [[ -z "${HF_TOKEN:-}" && ! -s "$HF_HOME/token" ]]; then
    echo 'Authenticate with Hugging Face under HF_HOME or export HF_TOKEN.' >&2
    echo 'Access to meta-llama/Llama-3.1-8B is required.' >&2
    exit 2
fi

"$UV_BIN" sync --frozen --extra test
"$UV_BIN" venv --clear --python 3.12 "$QUANTS_ABLATION_ROOT/.venv-sglang"
"$UV_BIN" pip install \
    --python "$QUANTS_ABLATION_ROOT/.venv-sglang/bin/python" \
    --prerelease=allow \
    --requirement "$QUANTS_ABLATION_ROOT/sglang-requirements.txt"
"$UV_BIN" pip freeze \
    --python "$QUANTS_ABLATION_ROOT/.venv-sglang/bin/python" \
    > "$QUANTS_ABLATION_ROOT/artifacts/sglang-environment.txt"
if ! cmp -s \
    "$QUANTS_ABLATION_ROOT/sglang-requirements.txt" \
    "$QUANTS_ABLATION_ROOT/artifacts/sglang-environment.txt"; then
    diff -u \
        "$QUANTS_ABLATION_ROOT/sglang-requirements.txt" \
        "$QUANTS_ABLATION_ROOT/artifacts/sglang-environment.txt" >&2 || true
    echo 'The installed judge environment does not match its lock file.' >&2
    exit 1
fi

"$UV_BIN" run hf download meta-llama/Llama-3.1-8B \
    --revision d04e592bb4f6aa9cfee91e2e20afa771667e1d4b \
    --local-dir "$QUANTS_ABLATION_ROOT/models/llama-3.1-8b"
"$UV_BIN" run hf download Qwen/Qwen3-8B-AWQ \
    --revision 4da05a8edb55c6046cce958586c33b61da07bb79 \
    --local-dir "$QUANTS_ABLATION_ROOT/models/qwen3-8b-awq"

"$UV_BIN" run python -m nltk.downloader \
    -d "$NLTK_DATA" wordnet omw-1.4
"$UV_BIN" run python -c \
    "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('$QUANTS_ABLATION_ROOT/models/llama-3.1-8b', local_files_only=True); print(type(t).__name__, t.model_max_length)"
