# Naive, Q2, and Q3 experiments

This directory contains the implementation of three QuAnTS ablations for
binary, multiple-choice, and open-answer questions:

- `question_only`: Q2, using only the question
- `ts_only`: Q3, using only the serialized motion trajectory
- `naive`: the motion trajectory followed by the question

Each of the nine ablation/task combinations is fine-tuned independently.

## Prompts and time-series serialization

The trajectories have shape `[320, 24, 3]`: 320 time steps, 24 human-body joints, and
three spatial coordinates. Q3 uses this prompt:

```text
### Instruction: You are provided with a 3-dimensional dataset of shape [320, 24, 3], representing 3-dimensional spatial locations of 24 human-body joints over 320 time steps. Using this data, please analyze the movements and provide the answer associated with this human activity sequence.

### Data: {scaled time series data}

### Answer:
```

For every example, the complete trajectory tensor is min-max scaled to `[-999, 999]`,
rounded to the nearest integer, and serialized as a compact nested JSON list. A constant
tensor maps to zeros. Compact JSON only removes whitespace; it retains all three axes and
all values. Preprocessing fails rather than truncating an input that exceeds Llama's
131,072-token context.

Q2 retains the paper prompt:

```text
### Question: {question}

### Answer:
```

The Naive prompt uses the same data description and compact trajectory, then
adds `### Question: {question}` before `### Answer:`. Binary targets are `Yes` and `No`
from the released numeric labels (`1` and `0`); multiple-choice targets are `A`, `B`, and
`C`; open targets use `answer_text`. Training loss is applied only to answer tokens and
EOS, without a chat template.

## Experiment settings

The workflow pins:

- dataset `dasyd/quants` at `0e78849313d3b043a0b08697dc004fb6cba15df9`
- base model `meta-llama/Llama-3.1-8B` at
  `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`
- judge `Qwen/Qwen3-8B-AWQ` at
  `4da05a8edb55c6046cce958586c33b61da07bb79`

Fine-tuning uses five epochs, effective batch size 16, seed 42, NF4 QLoRA with double
quantization and BF16 computation, and LoRA rank 16, alpha 32, and dropout 0.1 on all
attention and MLP linear projections. The optimizer is AdamW with learning rate `2e-4`,
linear decay, betas `(0.9, 0.999)`, weight decay `0.01`, no warmup, and maximum gradient
norm `1.0`. Gradient checkpointing and SDPA attention are enabled.

For the long prompts, the trainer projects vocabulary logits only at positions needed by
the answer-token loss. Since every prompt label is masked, this is algebraically the same
completion loss as projecting logits at all prompt positions, while avoiding a very large
unused logits tensor.

## Run the SLURM pipeline

Run the submission script from this directory. It computes and exports
`QUANTS_ABLATION_ROOT`, uses workspace-local caches, and submits the complete dependency
chain under account `jackal_ai`:

```bash
cd benchmark
export HF_HOME="$PWD/cache/huggingface"
uv tool run --from 'huggingface_hub==0.35.1' hf auth login
bash slurm/submit_pipeline.sh
```

The Llama checkpoint is gated, so the Hugging Face account used above must have access.
An environment-provided `HF_TOKEN` is also supported. Do not put a token in a script,
configuration file, or Git history. If `uv` is not on `PATH`, select it without editing
the scripts:

```bash
UV_BIN=/path/to/uv bash slurm/submit_pipeline.sh
```

The submission is non-exclusive; array elements can run concurrently when resources are
available. Its resource topology is:

| Stage | Array | Resources per element |
| --- | ---: | --- |
| Download data | no | CPU (`short`) |
| Set up environments and pinned models | no | CPU (`short`) |
| Prepare, test, and smoke-test all nine configs | no | 1 H100 |
| Train Q2 | 3 tasks | 1 H100, microbatch 8, accumulation 2 |
| Train Q3 | 3 tasks | 4 nodes × 4 H100, microbatch 1, accumulation 1 |
| Train Naive | 3 tasks | 4 nodes × 4 H100, microbatch 1, accumulation 1 |
| Generate Q2 | 3 tasks | 1 H100 |
| Generate Q3 | 3 tasks | 4 H100 |
| Generate Naive | 3 tasks | 4 H100 |
| Judge all open answers | no | 1 GPU (`all,h200`) |
| Validate and aggregate | no | CPU (`short`) |

Training resumes from its latest checkpoint. Generation resumes from rank-local files in
`results/.parts/`, and the judge checkpoints completed ratings after each chunk. Every
downstream stage uses an `afterok` dependency, so failed prerequisites do not silently
produce partial aggregate results. Job IDs are printed after submission; logs are written
under `logs/`.

## Outputs

A complete run writes 27 canonical prediction files to `results/`:

```text
{question_only,ts_only,naive}_{binary,multi,open}_{train,val,test}.csv
```

Every prediction CSV is UTF-8/RFC 4180, sorted by `(sample_id, question_id)`, and contains
exactly these columns:

```csv
sample_id,question_id,prediction_text
```

`prediction_text` is the raw decoded continuation, including Unicode and embedded
newlines. The evaluator reads these canonical CSVs directly. It additionally writes nine
open-answer judge CSVs, `metrics_by_config_and_split.csv`,
`rebuttal_test_metrics.csv`, and a run manifest. Predictions, metrics, model snapshots,
adapters, caches, logs, and rank-local parts remain untracked and can be distributed as
a complete release artifact.

Run the local test suite after changing prompts, serialization, CSV handling, or metrics:

```bash
uv sync --frozen --extra test
uv run pytest
```
