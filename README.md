# ts-qa

Question Answering for Time Series 🚀

## Howto use
To run the project one can either build the vscode devcontainer or run a headless compose version. For this one can leverage the given makefile. 
Executing the make file requires a unix/linux environment.

- Running `make` builds the container and then attaches to the shell of the container
- One must manually stop and/or remove the running container. This can be done by running `make stop` and/or `make remove` or `make stop_remove`.

Also run `pip install -e .`.

## Naive, Q2, and Q3 experiments

The Q2 (question-only), Q3 (time-series-only), and Naive
(time-series + question) Llama 3.1 experiments and complete SLURM workflow are
documented in [`benchmark/`](benchmark/README.md).

## Analysis notebooks

The evaluation of the trained models and all paper figures live in `eval-all/`. They expect the
repository root as the working directory.

| Notebook | Purpose |
| --- | --- |
| `consolidate_data.ipynb` | Joins the raw per-experiment CSVs in `eval-all/data/<section>/` with the `dasyd/quants` dataset into `all_joined.h5`. |
| `eval_llm_judge.ipynb` (+ `run_llm_judge.py`) | Scores the open answers with the LLM judge, checks its agreement with the human reference ratings, and yields `all_joined_judged.h5`. |
| `compute-metrics.ipynb` | Computes the result tables (`all_results_{binary,multi,open}.csv`) and the finetuning-dataset-size figures. Its last section breaks Humans and xQA-Qwen on AE down over all 45 question types (`question_type_breakdown.{csv,pdf}`). |
| `dataset-diversity-overview.ipynb` | Question-type distributions of QuAnTS and of the datasets it is compared against. |
| `action-difficulty.ipynb` | Per-action recognition F1 from the human identification sub-study, and how human QA performance changes when a hard-to-recognise action is named in the question or answer. |

`action-difficulty.ipynb` needs neither a GPU nor the training dependencies, so it can also be run
outside the container:

```bash
uv venv .venv-host --python 3.11
uv pip install --python .venv-host "numpy<2" "pandas<2.2" tables matplotlib seaborn scipy torchmetrics jupyter
.venv-host/bin/jupyter nbconvert --execute --inplace eval-all/action-difficulty.ipynb
```
