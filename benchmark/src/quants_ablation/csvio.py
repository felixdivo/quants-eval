from __future__ import annotations

import csv
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable, Mapping

from .constants import SPLIT_ID_RANGES


PREDICTION_COLUMNS = ("sample_id", "question_id", "prediction_text")


def _normalized_prediction_rows(
    rows: Iterable[Mapping[str, object]], split: str
) -> list[dict[str, object]]:
    if split not in SPLIT_ID_RANGES:
        raise ValueError(f"unknown split: {split}")
    lower, upper = SPLIT_ID_RANGES[split]
    normalized: list[dict[str, object]] = []
    seen: set[tuple[int, int]] = set()
    for row in rows:
        sample_id = int(row["sample_id"])
        question_id = int(row["question_id"])
        prediction = row["prediction_text"]
        if not isinstance(prediction, str):
            raise TypeError("prediction_text must be a Unicode string")
        if not lower <= sample_id <= upper:
            raise ValueError(f"sample_id {sample_id} is outside the {split} range")
        if not 0 <= question_id <= 4:
            raise ValueError(f"question_id {question_id} is outside [0, 4]")
        key = (sample_id, question_id)
        if key in seen:
            raise ValueError(f"duplicate prediction key: {key}")
        seen.add(key)
        normalized.append(
            {
                "sample_id": sample_id,
                "question_id": question_id,
                "prediction_text": prediction,
            }
        )
    normalized.sort(key=lambda row: (int(row["sample_id"]), int(row["question_id"])))
    return normalized


def write_prediction_csv(
    destination: Path, rows: Iterable[Mapping[str, object]], split: str
) -> None:
    normalized = _normalized_prediction_rows(rows, split)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=destination.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(PREDICTION_COLUMNS),
            extrasaction="raise",
            lineterminator="\r\n",
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        writer.writerows(normalized)
        temporary = Path(handle.name)
    validate_prediction_csv(temporary, split, expected_count=len(normalized))
    os.replace(temporary, destination)


def read_prediction_csv(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PREDICTION_COLUMNS:
            raise ValueError(f"invalid columns in {path}: {reader.fieldnames}")
        return [
            {
                "sample_id": int(row["sample_id"]),
                "question_id": int(row["question_id"]),
                "prediction_text": row["prediction_text"],
            }
            for row in reader
        ]


def validate_prediction_csv(
    path: Path, split: str, expected_count: int | None = None
) -> None:
    rows = read_prediction_csv(path)
    normalized = _normalized_prediction_rows(rows, split)
    if rows != normalized:
        raise ValueError(f"prediction rows are not canonically sorted in {path}")
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} rows in {path}, found {len(rows)}")
