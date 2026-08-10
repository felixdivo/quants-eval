from __future__ import annotations

import hashlib
from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from datasets import Dataset

from .constants import (
    DATA_ROOT,
    EXPECTED_ROWS,
    SPLIT_ID_RANGES,
    TASKS,
    validate_choice,
)

BASE_COLUMNS = ("sample_id", "question_id", "question", "answer_text")


def parquet_shards(task: str, split: str, data_root: Path = DATA_ROOT) -> list[Path]:
    validate_choice(task, TASKS, "task")
    if split not in SPLIT_ID_RANGES:
        raise ValueError(f"unknown split: {split}")
    paths = sorted((data_root / task).glob(f"{split}-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no shards found for {task}/{split} below {data_root}")
    return paths


def load_rows(
    task: str,
    split: str,
    data_root: Path = DATA_ROOT,
    include_trajectory: bool = False,
    include_description: bool = False,
) -> Dataset:
    paths = parquet_shards(task, split, data_root)
    columns = list(BASE_COLUMNS)
    if task in ("binary", "multi"):
        columns.append("answer")
    if include_trajectory:
        columns.append("trajectory")
    if include_description:
        columns.append("textual_description")
    table = pq.read_table(paths, columns=columns)
    indices = pc.sort_indices(
        table, sort_keys=[("sample_id", "ascending"), ("question_id", "ascending")]
    )
    table = pc.take(table, indices)
    dataset = Dataset(table)
    validate_rows(dataset, task, split)
    return dataset


def validate_rows(dataset: Dataset, task: str, split: str) -> None:
    expected = EXPECTED_ROWS[task][split]
    if len(dataset) != expected:
        raise ValueError(
            f"expected {expected} {task}/{split} rows, found {len(dataset)}"
        )
    lower, upper = SPLIT_ID_RANGES[split]
    keys: set[tuple[int, int]] = set()
    for sample_id, question_id in zip(
        dataset["sample_id"], dataset["question_id"], strict=True
    ):
        sample_id = int(sample_id)
        question_id = int(question_id)
        if not lower <= sample_id <= upper:
            raise ValueError(f"sample_id {sample_id} outside {split} range")
        if not 0 <= question_id <= 4:
            raise ValueError(f"question_id {question_id} outside [0, 4]")
        key = (sample_id, question_id)
        if key in keys:
            raise ValueError(f"duplicate key {key} in {task}/{split}")
        keys.add(key)


def trajectory_array(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (320, 24, 3):
        raise ValueError(f"expected trajectory shape (320, 24, 3), got {array.shape}")
    return np.ascontiguousarray(array)


def trajectory_sha256(value: Any) -> str:
    return hashlib.sha256(trajectory_array(value).tobytes()).hexdigest()


def expected_keys(
    task: str, split: str, data_root: Path = DATA_ROOT
) -> list[tuple[int, int]]:
    dataset = load_rows(task, split, data_root=data_root)
    return [
        (int(sample_id), int(question_id))
        for sample_id, question_id in zip(
            dataset["sample_id"], dataset["question_id"], strict=True
        )
    ]


def load_ts_prompt_cache(path: Path) -> Dataset:
    table = pq.read_table(path)
    expected_columns = {"sample_id", "input_ids", "trajectory_sha256", "token_count"}
    if set(table.column_names) != expected_columns:
        raise ValueError(
            f"invalid TS prompt cache schema in {path}: {table.column_names}"
        )
    sample_ids = table.column("sample_id").to_pylist()
    if any(left >= right for left, right in pairwise(sample_ids)):
        raise ValueError(f"TS prompt cache is not strictly sorted by sample_id: {path}")
    return Dataset(table)


def rows_as_dicts(dataset: Dataset) -> Iterable[dict[str, Any]]:
    for index in range(len(dataset)):
        yield dataset[index]
