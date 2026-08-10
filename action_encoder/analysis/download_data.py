#!/usr/bin/env python3
"""Download the pinned QuAnTS Parquet snapshot and write a verified manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from quants_action_encoder.constants import DATASET_ID, DATASET_REVISION, TASKS
from quants_action_encoder.csvio import file_sha256, write_json_atomic
from quants_action_encoder.data import (
    action_names_from_sequence,
    load_question_rows,
    normalize_split,
    trajectory_array,
    validate_sample_id,
)

SHARD_COUNTS = {
    "binary": {"train": 13, "val": 2, "test": 2},
    "multi": {"train": 9, "val": 2, "test": 2},
    "open": {"train": 9, "val": 2, "test": 2},
}


def build_download_manifest(
    destination: Path,
    split_reports: dict[str, dict[str, int]],
) -> dict[str, object]:
    """Build a stable manifest from only the intended snapshot files."""

    files: list[Path] = []
    for task, split_counts in SHARD_COUNTS.items():
        for split, expected_count in split_counts.items():
            shards = sorted((destination / task).glob(f"{split}-*.parquet"))
            if len(shards) != expected_count:
                raise ValueError(
                    f"expected {expected_count} {task}/{split} shards, found {len(shards)}"
                )
            files.extend(shards)
    for name in ("README.md", "LICENSE"):
        path = destination / name
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append(path)
    return {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "splits": split_reports,
        "files": {
            path.relative_to(destination).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in sorted(files)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()

    destination = args.destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        revision=DATASET_REVISION,
        local_dir=destination,
        cache_dir=args.cache_dir,
        allow_patterns=[
            "binary/*.parquet",
            "multi/*.parquet",
            "open/*.parquet",
            "README.md",
            "LICENSE",
        ],
    )

    splits: dict[str, dict[str, int]] = {}
    for task in TASKS:
        for split in ("train", "val", "test"):
            rows = load_question_rows(task, split, data_root=destination)
            required = {
                "sample_id",
                "question_id",
                "trajectory",
                "action_sequence",
                "question",
                "answer_text",
                "question_type",
            }
            missing_columns = required - set(rows.column_names)
            if missing_columns:
                raise ValueError(f"{task}/{split} misses columns {sorted(missing_columns)}")
            keys: set[tuple[int, int]] = set()
            for sample_id, question_id in zip(
                rows["sample_id"], rows["question_id"], strict=True
            ):
                typed_sample = validate_sample_id(sample_id, split)
                typed_question = int(question_id)
                if not 0 <= typed_question <= 4:
                    raise ValueError(
                        f"{task}/{split} question_id {typed_question} outside [0, 4]"
                    )
                key = (typed_sample, typed_question)
                if key in keys:
                    raise ValueError(f"duplicate {task}/{split} key {key}")
                keys.add(key)
            if not keys:
                raise ValueError(f"{task}/{split} is empty")
            for index in {0, len(rows) - 1}:
                row = rows[index]
                trajectory_array(row["trajectory"])
                action_names_from_sequence(row["action_sequence"])
            splits[f"{task}/{normalize_split(split)}"] = {
                "rows": len(rows),
                "unique_samples": len({sample_id for sample_id, _ in keys}),
            }

    manifest = build_download_manifest(destination, splits)
    write_json_atomic(destination / "manifest.json", manifest)
    print(f"verified {len(manifest['files'])} files in {destination}", flush=True)


if __name__ == "__main__":
    main()
