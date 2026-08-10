#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root}"
: "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL before submitting xQA jobs}"
: "${OPENAI_MODEL:?Set OPENAI_MODEL before submitting xQA jobs}"
: "${OPENAI_MODEL_REVISION:?Set OPENAI_MODEL_REVISION before submitting xQA jobs}"
: "${OPENAI_DEPLOYMENT_FINGERPRINT:?Set OPENAI_DEPLOYMENT_FINGERPRINT before submitting xQA jobs}"
: "${JUDGE_BASE_URL:?Set JUDGE_BASE_URL before submitting open-answer judge jobs}"
: "${JUDGE_MODEL:?Set JUDGE_MODEL before submitting open-answer judge jobs}"
: "${JUDGE_MODEL_REVISION:?Set JUDGE_MODEL_REVISION before submitting open-answer judge jobs}"
: "${JUDGE_DEPLOYMENT_FINGERPRINT:?Set JUDGE_DEPLOYMENT_FINGERPRINT before submitting judge jobs}"
export DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/action_encoder/data/quants}"

cd "${PROJECT_ROOT}"
mkdir -p action_encoder/logs action_encoder/results action_encoder/artifacts

setup_job="$(sbatch --parsable action_encoder/slurm/00_setup.sbatch)"
download_job="$(sbatch --parsable --dependency="afterok:${setup_job}" action_encoder/slurm/01_download_data.sbatch)"
data_preflight_job="$(sbatch --parsable --dependency="afterok:${download_job}" action_encoder/slurm/02_data_preflight.sbatch)"
preflight_job="$(sbatch --parsable --dependency="afterok:${data_preflight_job}" action_encoder/slurm/05_preflight.sbatch)"
train_job="$(sbatch --parsable --dependency="afterok:${preflight_job}" action_encoder/slurm/10_train.sbatch)"
gt_job="$(sbatch --parsable --dependency="afterok:${data_preflight_job}" action_encoder/slurm/30_xqa_gt.sbatch)"
predict_job="$(sbatch --parsable --dependency="afterok:${train_job}" action_encoder/slurm/20_predict_actions.sbatch)"
predicted_job="$(sbatch --parsable --dependency="afterok:${predict_job}" action_encoder/slurm/31_xqa_predicted.sbatch)"
metrics_job="$(sbatch --parsable --dependency="afterok:${predict_job}" action_encoder/slurm/40_metrics.sbatch)"
judge_job="$(sbatch --parsable --dependency="afterok:${gt_job}:${predicted_job}" action_encoder/slurm/45_judge_xqa_open.sbatch)"
xqa_metrics_job="$(sbatch --parsable --dependency="afterok:${judge_job}" action_encoder/slurm/50_xqa_metrics.sbatch)"
aggregate_job="$(sbatch --parsable --dependency="afterok:${xqa_metrics_job}" action_encoder/slurm/51_aggregate_xqa_metrics.sbatch)"

printf 'setup=%s download=%s data_preflight=%s gpu_preflight=%s train=%s gt_xqa=%s predict=%s predicted_xqa=%s action_metrics=%s judge=%s xqa_metrics=%s aggregate=%s\n' \
  "${setup_job}" "${download_job}" "${data_preflight_job}" "${preflight_job}" "${train_job}" "${gt_job}" "${predict_job}" \
  "${predicted_job}" "${metrics_job}" "${judge_job}" "${xqa_metrics_job}" "${aggregate_job}"
