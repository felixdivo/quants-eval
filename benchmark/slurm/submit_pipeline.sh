#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export QUANTS_ABLATION_ROOT
QUANTS_ABLATION_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

if [[ -z "${UV_BIN:-}" ]]; then
    UV_BIN=$(command -v uv || true)
fi
if [[ -z "$UV_BIN" || ! -x "$UV_BIN" ]]; then
    echo 'uv was not found. Put it on PATH or set UV_BIN=/path/to/uv.' >&2
    exit 2
fi
export UV_BIN
export HF_HOME=${HF_HOME:-"$QUANTS_ABLATION_ROOT/cache/huggingface"}
export HF_HUB_CACHE=${HF_HUB_CACHE:-"$HF_HOME/hub"}
export UV_CACHE_DIR=${UV_CACHE_DIR:-"$QUANTS_ABLATION_ROOT/cache/uv"}
export NLTK_DATA=${NLTK_DATA:-"$QUANTS_ABLATION_ROOT/cache/nltk"}

if [[ -z "${HF_TOKEN:-}" && ! -s "$HF_HOME/token" ]]; then
    echo 'Hugging Face authentication is missing.' >&2
    echo 'Log in under HF_HOME or export HF_TOKEN before submitting.' >&2
    exit 2
fi

mkdir -p "$QUANTS_ABLATION_ROOT/logs"
cd "$QUANTS_ABLATION_ROOT"

submit() {
    sbatch --parsable \
        --chdir="$QUANTS_ABLATION_ROOT" \
        --export=ALL \
        "$@"
}

download_job=$(submit slurm/00_download_data.sbatch)
setup_job=$(submit --dependency="afterok:$download_job" slurm/01_setup_and_models.sbatch)
preflight_job=$(submit --dependency="afterok:$setup_job" slurm/02_prepare_and_preflight.sbatch)

question_train_job=$(submit --dependency="afterok:$preflight_job" slurm/10_train_question.sbatch)
ts_train_job=$(submit --dependency="afterok:$preflight_job" slurm/11_train_ts.sbatch)
naive_train_job=$(submit --dependency="afterok:$preflight_job" slurm/12_train_naive.sbatch)

question_generate_job=$(submit --dependency="afterok:$question_train_job" slurm/20_generate_question.sbatch)
ts_generate_job=$(submit --dependency="afterok:$ts_train_job" slurm/21_generate_ts.sbatch)
naive_generate_job=$(submit --dependency="afterok:$naive_train_job" slurm/22_generate_naive.sbatch)

judge_job=$(submit \
    --dependency="afterok:$question_generate_job:$ts_generate_job:$naive_generate_job" \
    slurm/30_judge_open.sbatch)
aggregate_job=$(submit --dependency="afterok:$judge_job" slurm/40_aggregate_all.sbatch)

printf '%-26s %s\n' \
    download "$download_job" \
    setup_and_models "$setup_job" \
    preflight_9_configs "$preflight_job" \
    question_train_array "$question_train_job" \
    ts_train_array "$ts_train_job" \
    naive_train_array "$naive_train_job" \
    question_generate_array "$question_generate_job" \
    ts_generate_array "$ts_generate_job" \
    naive_generate_array "$naive_generate_job" \
    open_judge "$judge_job" \
    aggregate_all "$aggregate_job"
