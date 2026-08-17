# QuAnTS xLSTMMixer action encoder and xQA

This package implements the xLSTMMixer action encoder and xQA workflows for QuAnTS.
The active action-classification pipeline uses xLSTMMixer throughout.

## Fixed inputs

- Dataset: `dasyd/quants` at
  `0e78849313d3b043a0b08697dc004fb6cba15df9`.
- Action labels: one fixed global 19-class order, checked against the released dataset
  and shared by training and xQA decoding.
- Backend: `xLSTMMixerForTimeSeriesClassification` from
  `mauricekraus/transformers` commit
  `e97fc33f97155b79658af37fde29b39b5139e066`. The fork identifies itself as
  Transformers 4.57.0. The project uses the commit's HTTPS source archive, so installs
  neither depend on a mutable branch nor require GitHub SSH access.
- xLSTM runtime: `xlstm==2.0.5`, the latest release predating that fork commit.

The direct dependency pin is in `pyproject.toml`, and `uv.lock` fixes the complete
environment. Changing either defines a different
experiment. The model uses four contiguous 80-frame segments. Each segment
is flattened from `[80,24,3]` to `[80,72]`, and the model predicts one action ID. A
trajectory therefore produces four action IDs.

Training preserves per-QA-row weighting: a trajectory contributes its
four segments once for every question row in a task. Prediction deliberately
deduplicates by `sample_id`, producing exactly four rows per sample.

## Installation

From the repository root:

```bash
uv sync --frozen --project action_encoder --extra dev --extra analysis
uv run --frozen --project action_encoder pytest action_encoder/tests
```

Use `HF_TOKEN` in the environment for gated or authenticated Hub access. Tokens are
never accepted as command-line values and must not be committed.

Download and verify the pinned Parquet snapshot on CPU before requesting a GPU:

```bash
export DATA_ROOT="$PWD/action_encoder/data/quants"
action_encoder/.venv/bin/python action_encoder/analysis/download_data.py \
  --destination "$DATA_ROOT" \
  --cache-dir action_encoder/.cache/huggingface/hub
action_encoder/.venv/bin/quants-action-data-preflight \
  --data-root "$DATA_ROOT" \
  --output action_encoder/data/preflight_manifest.json
```

The expected layout is `<root>/{binary,multi,open}/{train,val,test}-*.parquet`.
The CPU preflight validates every trajectory/action row, exact shard and row counts,
split keys, duplicate consistency, and content hashes.

## Train and predict actions

Train one independent classifier per answer format:

```bash
action_encoder/.venv/bin/quants-action-train \
  --task binary \
  --output-dir action_encoder/artifacts/action_encoder/binary \
  --data-root "$DATA_ROOT" \
  --preflight-manifest action_encoder/data/preflight_manifest.json \
  --skip-data-validation \
  --epochs 5 \
  --batch-size 512 \
  --learning-rate 1e-4 \
  --seed 42
```

Run the lowest-validation-loss checkpoint on a split:

```bash
action_encoder/.venv/bin/quants-action-predict \
  --checkpoint action_encoder/artifacts/action_encoder/binary/best \
  --task binary \
  --split test \
  --data-root "$DATA_ROOT" \
  --preflight-manifest action_encoder/data/preflight_manifest.json \
  --output action_encoder/results/action_encoder_binary_test.csv
```

The separate action-prediction CSV is long-form and sorted:

```csv
sample_id,segment_id,predicted_action_id,predicted_action_name
27000,0,11,bowing
27000,1,9,waving
27000,2,16,picking something up with both hands
27000,3,16,picking something up with both hands
```

There must be exactly four unique segment rows (`segment_id` 0 through 3) per sample.
The sidecar `.meta.json` contains the dataset/configuration fingerprint and checkpoint
SHA-256.

## Run xQA with GT or predicted actions

The xQA prompt uses binary, multi, and open task-specific few-shot examples and a
structured `actions`, `steps`, `answer` output. The default OpenAI-compatible request uses a strict JSON
schema, temperature 0 for the first request, temperature 0.3 for validation retries,
and at most 2,048 output tokens. It does not depend on SGLang's
Python package; an SGLang server can still be used through its OpenAI-compatible HTTP
endpoint.

Configure the SGLang-compatible endpoint with an explicit server model ID, immutable
weights revision, and deployment/container digest:

```bash
export OPENAI_BASE_URL=http://inference-host:30000
export OPENAI_MODEL=served-model-id-from-v1-models
export OPENAI_MODEL_REVISION=immutable-weights-revision
export OPENAI_DEPLOYMENT_FINGERPRINT=sha256:immutable-server-manifest
export OPENAI_API_KEY=replace-if-required
```

GT-action mode reads the released `action_sequence`:

```bash
action_encoder/.venv/bin/quants-action-xqa \
  --task binary \
  --split test \
  --mode gt \
  --data-root "$DATA_ROOT" \
  --preflight-manifest action_encoder/data/preflight_manifest.json \
  --output action_encoder/results/xqa_gt_binary_test.csv
```

Predicted-action mode consumes the separate action CSV:

```bash
action_encoder/.venv/bin/quants-action-xqa \
  --task binary \
  --split test \
  --mode predicted \
  --action-predictions action_encoder/results/action_encoder_binary_test.csv \
  --data-root "$DATA_ROOT" \
  --preflight-manifest action_encoder/data/preflight_manifest.json \
  --output action_encoder/results/xqa_predicted_binary_test.csv
```

If an endpoint lacks strict JSON-schema support, explicitly use
`--response-format json-object`; `--response-format none` is available only as a final
compatibility fallback.

Every canonical QA prediction CSV has exactly these columns:

```csv
sample_id,question_id,prediction_text
```

`prediction_text` is the untouched model continuation. It is not stripped, normalized,
or replaced by a parsed label. Validation failures are retried, but if all attempts are
malformed, the final real malformed output is retained. If the endpoint returns no
continuation at all, the job stops and leaves a resumable `.partial` CSV; it never
substitutes `false`, `A`, or an empty answer. Parsing `answer` for metrics happens
post-hoc via `parse_structured_response` and never changes this canonical file.

Resume metadata fingerprints the task, split, dataset revision, action CSV, prompt
version, endpoint model, and decoding configuration. Resuming with different metadata
fails rather than mixing experiments.

## Evaluate the action encoder

Post-hoc evaluation reports segment accuracy, exact four-action sequence accuracy,
position accuracy, and per-class precision/recall/F1:

```bash
action_encoder/.venv/bin/quants-action-metrics \
  --predictions action_encoder/results/action_encoder_binary_test.csv \
  --task binary \
  --split test \
  --data-root "$DATA_ROOT" \
  --output-dir action_encoder/results/action_metrics/binary_test
```

It writes `action_metrics.json` plus two metric views:

- `action_metrics_unique_sample_{summary,by_position,by_class}.csv` counts every
  distinct trajectory once.
- `action_metrics_qa_row_weighted_{summary,by_position,by_class}.csv` repeats a
  trajectory once per released QA row, matching the dataset's question-row weighting.
  `action_metrics_qa_row_weighted_by_question_type.csv` additionally breaks this view
  down by question type.

The plotting wrapper consumes the JSON without loading a model:

```bash
action_encoder/.venv/bin/python \
  action_encoder/analysis/plot_action_encoder_metrics.py \
  action_encoder/results/action_metrics/binary_test/action_metrics.json \
  --output action_encoder/results/action_metrics/binary_test/action_metrics.png
```

## Evaluate xQA

Closed-answer xQA evaluation reports accuracy, macro precision/recall/F1, and invalid
output rate. Open-answer evaluation reports ROUGE-L F1, METEOR, invalid output rate,
and normalized LLMJudge. Parsing is post-hoc; canonical `prediction_text` remains
unchanged.

Judge both open configurations with the pinned judge deployment identity:

```bash
export JUDGE_BASE_URL=http://judge-host:30000
export JUDGE_MODEL=Qwen/Qwen3-8B-AWQ
export JUDGE_MODEL_REVISION=4da05a8edb55c6046cce958586c33b61da07bb79
export JUDGE_DEPLOYMENT_FINGERPRINT=sha256:immutable-judge-server-manifest

action_encoder/.venv/bin/quants-xqa-judge \
  --predictions action_encoder/results/xqa_gt_open_test.csv \
  --config xqa_gt_open \
  --split test \
  --data-root "$DATA_ROOT" \
  --preflight-manifest action_encoder/data/preflight_manifest.json \
  --output action_encoder/results/judge_xqa_gt_open_test.csv
```

Then compute one configuration's metrics:

```bash
action_encoder/.venv/bin/quants-xqa-metrics \
  --predictions action_encoder/results/xqa_gt_open_test.csv \
  --judge-ratings action_encoder/results/judge_xqa_gt_open_test.csv \
  --config xqa_gt_open \
  --task open \
  --split test \
  --data-root "$DATA_ROOT" \
  --output action_encoder/results/xqa_metrics/xqa_gt_open_test_metrics.csv
```

This writes a one-row configuration/split metric CSV, a `_parsed.csv` table keyed by
`sample_id,question_id` for judge/audit input, and a `_by_question_type.csv` table. The
six GT/predicted × binary/multi/open metric files are combined by
`quants-xqa-aggregate` into `xqa_metrics_by_config_and_split.csv`. Judge CSVs require
exact canonical keys and ratings 1–3; normalization is `(rating−1)/2`.

## SLURM

The examples in `slurm/` use account `jackal_ai`, workspace-local environments/caches,
non-exclusive jobs, H200 or general GPU partitions, resumable outputs, and SLURM
requeue support. `submit_pipeline.sh` connects the pinned environment, CPU download and
full data preflight, real-backend CUDA smoke, three parallel training jobs, action
prediction, both xQA modes, open judging, and both metric views with `afterok`
dependencies. Endpoint jobs must already be running.

Set at least `PROJECT_ROOT`, `OPENAI_BASE_URL`, `OPENAI_MODEL`, an immutable
`OPENAI_MODEL_REVISION`, and `OPENAI_DEPLOYMENT_FINGERPRINT` before submission. These
values are fingerprinted in result metadata; the endpoint operator remains responsible
for serving those exact weights.
The submitter also requires the four analogous `JUDGE_*` values. `DATA_ROOT` defaults
to `action_encoder/data/quants` and is populated by the CPU download job.
