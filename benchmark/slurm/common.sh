#!/usr/bin/env bash

# Shared runtime setup for jobs submitted by submit_pipeline.sh.
: "${QUANTS_ABLATION_ROOT:?submit jobs through slurm/submit_pipeline.sh}"

UV_BIN=${UV_BIN:-uv}
if ! command -v "$UV_BIN" >/dev/null 2>&1; then
    echo "uv executable not found: $UV_BIN" >&2
    exit 2
fi

export UV_BIN
export HF_HOME=${HF_HOME:-"$QUANTS_ABLATION_ROOT/cache/huggingface"}
export HF_HUB_CACHE=${HF_HUB_CACHE:-"$HF_HOME/hub"}
export UV_CACHE_DIR=${UV_CACHE_DIR:-"$QUANTS_ABLATION_ROOT/cache/uv"}
export NLTK_DATA=${NLTK_DATA:-"$QUANTS_ABLATION_ROOT/cache/nltk"}
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

mkdir -p \
    "$QUANTS_ABLATION_ROOT/artifacts" \
    "$QUANTS_ABLATION_ROOT/cache" \
    "$QUANTS_ABLATION_ROOT/data" \
    "$QUANTS_ABLATION_ROOT/logs" \
    "$QUANTS_ABLATION_ROOT/models" \
    "$QUANTS_ABLATION_ROOT/results"
cd "$QUANTS_ABLATION_ROOT"
