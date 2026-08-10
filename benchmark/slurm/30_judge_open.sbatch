#!/usr/bin/env bash
#SBATCH --job-name=quants_judge_open
#SBATCH --account=jackal_ai
#SBATCH --partition=all,h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1
#SBATCH --time=4-00:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
source "${QUANTS_ABLATION_ROOT:?}/slurm/common.sh"

PORT=$((30000 + SLURM_JOB_ID % 10000))
SGLANG_PYTHON="$QUANTS_ABLATION_ROOT/.venv-sglang/bin/python"

"$SGLANG_PYTHON" -m sglang.launch_server \
    --model-path "$QUANTS_ABLATION_ROOT/models/qwen3-8b-awq" \
    --served-model-name qwen3-judge \
    --host 127.0.0.1 \
    --port "$PORT" \
    --tp-size 1 \
    --dp-size 1 \
    --quantization awq \
    --reasoning-parser qwen3 \
    --mem-fraction-static 0.85 &
SERVER_PID=$!

cleanup() {
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT TERM INT

ready=0
for _ in $(seq 1 180); do
    if curl --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null; then
        ready=1
        break
    fi
    sleep 5
done
if [[ "$ready" != 1 ]]; then
    echo 'SGLang judge did not become healthy within 15 minutes.' >&2
    exit 1
fi

"$UV_BIN" run quants-judge \
    --endpoint "http://127.0.0.1:$PORT/v1/chat/completions" \
    --concurrency 32 \
    --data-root "$QUANTS_ABLATION_ROOT/data/raw/quants" \
    --results-root "$QUANTS_ABLATION_ROOT/results" \
    --ablations question_only ts_only naive
