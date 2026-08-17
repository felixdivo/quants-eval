from __future__ import annotations

import os
from pathlib import Path


_project_root_override = os.environ.get("QUANTS_ABLATION_ROOT")
PROJECT_ROOT = (
    Path(_project_root_override).expanduser().resolve()
    if _project_root_override
    else Path(__file__).resolve().parents[2]
)
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "quants"
CACHE_ROOT = PROJECT_ROOT / "data" / "prepared"
RESULTS_ROOT = PROJECT_ROOT / "results"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
MODEL_ROOT = PROJECT_ROOT / "models"

DATASET_REPO = "dasyd/quants"
DATASET_REVISION = "0e78849313d3b043a0b08697dc004fb6cba15df9"
LLAMA_REPO = "meta-llama/Llama-3.1-8B"
LLAMA_REVISION = "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
JUDGE_REPO = "Qwen/Qwen3-8B-AWQ"
JUDGE_REVISION = "4da05a8edb55c6046cce958586c33b61da07bb79"

TASKS = ("binary", "multi", "open")
ABLATIONS = ("question_only", "ts_only", "naive")
SPLITS = ("train", "val", "test")
SPLIT_ID_RANGES = {
    "train": (0, 23_999),
    "val": (24_000, 26_999),
    "test": (27_000, 29_999),
}
EXPECTED_ROWS = {
    "binary": {"train": 49_454, "val": 6_235, "test": 6_115},
    "multi": {"train": 34_927, "val": 4_326, "test": 4_346},
    "open": {"train": 35_619, "val": 4_439, "test": 4_539},
}
MAX_CONTEXT_TOKENS = 131_072
SEED = 42


def validate_choice(value: str, choices: tuple[str, ...], label: str) -> str:
    if value not in choices:
        raise ValueError(f"invalid {label} {value!r}; expected one of {choices}")
    return value


def config_name(ablation: str, task: str) -> str:
    validate_choice(ablation, ABLATIONS, "ablation")
    validate_choice(task, TASKS, "task")
    return f"{ablation}_{task}"
