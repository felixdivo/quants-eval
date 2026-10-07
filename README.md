# QuAnTS: Question Answering on Time Series – Evaluation

[![arXiv](https://img.shields.io/badge/arXiv-2511.05124-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2511.05124)
[![Project Website](https://img.shields.io/badge/Project-Website-1f4e79.svg)](https://mauricekraus.github.io/quants-generate/)
[![Dataset on Hugging Face](https://img.shields.io/badge/Hugging%20Face-Dataset-ffd21e.svg?logo=huggingface)](https://huggingface.co/datasets/dasyd/quants)
[![Dataset Generation on GitHub](https://img.shields.io/badge/GitHub-Dataset%20Generation-blue.svg?logo=github)](https://github.com/mauricekraus/quants-generate)
[![Evaluation on GitHub](https://img.shields.io/badge/GitHub-Evaluation-blue.svg?logo=github)](https://github.com/felixdivo/quants-eval)

Training and evaluation code for [QuAnTS](https://huggingface.co/datasets/dasyd/quants),
a question-answering dataset built from human-motion trajectories. The experiments cover
binary, multiple-choice, and open-ended answers.

## Repository map

| Path | Contents |
| --- | --- |
| [`benchmark/`](benchmark/README.md) | Llama 3.1 8B Naive, Q2 question-only, and Q3 time-series-only experiments |
| [`action_encoder/`](action_encoder/README.md) | xLSTMMixer action encoder and xQA with ground-truth or predicted actions |
| [`analysis/`](analysis/README.md) | Paper tables, figures, and dataset analysis |

## Experiments

The Llama experiments train one QLoRA adapter per input configuration and answer format.
Time-series prompts contain compact nested JSON for the full `[320, 24, 3]` trajectory,
scaled to `[-999, 999]` and checked against the model context limit.

The xQA pipeline splits each trajectory into four 80-frame segments. An xLSTMMixer
classifier predicts one of 19 actions for each segment, and the language model answers
questions from either the released action sequence or the predicted sequence.

Each experiment directory contains its own pinned `uv.lock`, tests, command-line tools,
and SLURM submission pipeline. Start with the README in the corresponding directory.

## Development checks

```bash
uv lock --check --project benchmark
uv lock --check --project action_encoder
uv lock --check --project analysis

uv run --frozen --project benchmark --extra test pytest benchmark/tests
uv run --frozen --project action_encoder --extra dev pytest action_encoder/tests
```

Model and dataset credentials are read from environment variables. The SLURM workflows
use workspace-local environments, caches, logs, checkpoints, and result directories.

## Citation

Citation metadata for the paper and software is available in [`CITATION.cff`](CITATION.cff).
