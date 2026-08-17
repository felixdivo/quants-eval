# Analysis

These notebooks build the QuAnTS paper tables and figures from the experiment outputs and
the released dataset.

| Notebook | Purpose |
| --- | --- |
| `consolidate_data.ipynb` | Join predictions with the released QuAnTS examples |
| `compute-metrics.ipynb` | Compute answer-format metrics and question-type breakdowns |
| `dataset-diversity-overview.ipynb` | Compare question-type distributions across datasets |
| `action-difficulty.ipynb` | Analyze per-action recognition and human QA performance |
| `data-nesy-raw/combine.ipynb` | Combine NeSy prediction shards |
| `data-nesy-xlstm-raw/combine.ipynb` | Combine xLSTM-NeSy prediction shards |

Create the pinned environment from the repository root:

```bash
uv sync --frozen --project analysis
uv run --frozen --project analysis jupyter lab
```

Open Jupyter at the repository root so the notebooks resolve `analysis/data/` and
`analysis/plots/` consistently.
