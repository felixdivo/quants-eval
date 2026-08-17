"""Validated QuAnTS loading and 320-frame-to-action segmentation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import (
    ACTION_TO_ID,
    DATASET_ID,
    DATASET_REVISION,
    EXPECTED_PARQUET_SHARDS,
    EXPECTED_ROWS,
    INPUT_CHANNELS,
    SEGMENT_COUNT,
    SEGMENT_LENGTH,
    SPLIT_ID_RANGES,
    SPLITS,
    TASKS,
    TRAJECTORY_SHAPE,
)

TaskName = Literal["binary", "multi", "open"]
SplitName = Literal["train", "val", "test"]
_SPLIT_ALIASES = {"validation": "val"}


def normalize_split(split: str) -> SplitName:
    """Normalize the common ``validation`` alias to the released ``val`` split."""

    normalized = _SPLIT_ALIASES.get(split, split)
    if normalized not in SPLIT_ID_RANGES:
        raise ValueError(f"unsupported split {split!r}; expected train, val, or test")
    return normalized  # type: ignore[return-value]


def validate_sample_id(sample_id: Any, split: str) -> int:
    """Return a typed ID after checking the fixed released split boundaries."""

    normalized_split = normalize_split(split)
    try:
        typed = int(sample_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sample_id must be an integer, got {sample_id!r}") from exc
    if isinstance(sample_id, float | np.floating) and not float(sample_id).is_integer():
        raise ValueError(f"sample_id must be an integer, got {sample_id!r}")
    lower, upper = SPLIT_ID_RANGES[normalized_split]
    if not lower <= typed <= upper:
        raise ValueError(
            f"sample_id {typed} is outside the {normalized_split} range "
            f"[{lower}, {upper}]"
        )
    return typed


def trajectory_array(value: Any) -> np.ndarray:
    """Validate one released trajectory and return contiguous float32 values."""

    array = np.asarray(value, dtype=np.float32)
    if array.shape != TRAJECTORY_SHAPE:
        raise ValueError(
            f"expected trajectory shape {TRAJECTORY_SHAPE}, got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("trajectory contains a non-finite value")
    return np.ascontiguousarray(array)


def action_names_from_sequence(action_sequence: Any) -> tuple[str, ...]:
    """Normalize both Hugging Face's dict-of-lists and a list-of-dicts form."""

    names: Any
    if isinstance(action_sequence, Mapping):
        names = action_sequence.get("action")
        for key in ("start", "end", "action_sentence"):
            values = action_sequence.get(key)
            if values is not None and len(values) != SEGMENT_COUNT:
                raise ValueError(
                    f"action_sequence.{key} must contain {SEGMENT_COUNT} values"
                )
    elif isinstance(action_sequence, Sequence) and not isinstance(
        action_sequence, str | bytes
    ):
        if not all(isinstance(item, Mapping) for item in action_sequence):
            raise ValueError("action_sequence entries must be mappings")
        names = [item.get("action") for item in action_sequence]
    else:
        raise ValueError("action_sequence must be a mapping or a sequence of mappings")

    if not isinstance(names, Sequence) or isinstance(names, str | bytes):
        raise ValueError("action_sequence.action must be a sequence")
    if len(names) != SEGMENT_COUNT:
        raise ValueError(
            f"action_sequence.action must contain {SEGMENT_COUNT} values, got {len(names)}"
        )
    normalized = tuple(str(name) for name in names)
    unknown = [name for name in normalized if name not in ACTION_TO_ID]
    if unknown:
        raise ValueError(f"unknown QuAnTS action label {unknown[0]!r}")
    return normalized


def action_ids_from_sequence(action_sequence: Any) -> tuple[int, ...]:
    """Map an action sequence to the stable 0--18 label IDs."""

    return tuple(ACTION_TO_ID[name] for name in action_names_from_sequence(action_sequence))


def segment_trajectory(value: Any) -> torch.Tensor:
    """Convert ``[320,24,3]`` into four contiguous ``[80,72]`` segments."""

    array = trajectory_array(value)
    flattened = array.reshape(TRAJECTORY_SHAPE[0], INPUT_CHANNELS)
    segments = flattened.reshape(SEGMENT_COUNT, SEGMENT_LENGTH, INPUT_CHANNELS)
    # Copy so tensors decoded from Arrow never retain read-only or mmapped buffers.
    return torch.from_numpy(segments.copy())


def _column(rows: Any, name: str) -> Sequence[Any]:
    """Read a small column without decoding every large trajectory when possible."""

    if not isinstance(rows, list | tuple):
        try:
            column = rows[name]
        except (KeyError, TypeError):
            pass
        else:
            return column
    return [rows[index][name] for index in range(len(rows))]


class ActionSegmentDataset(Dataset[dict[str, torch.Tensor]]):
    """A lazy view turning each QuAnTS QA row into four classification rows.

    By default the index preserves per-question-row weighting: if a
    trajectory has multiple QA rows, its four segments occur once per QA row.
    ``deduplicate_samples=True`` instead returns exactly four rows per unique
    ``sample_id`` and is the required mode for prediction CSV generation.

    Duplicate rows point at the first row for their sample, avoiding repeated
    trajectory storage while retaining the same sampling multiplicity.
    """

    def __init__(
        self,
        rows: Any,
        *,
        split: str,
        deduplicate_samples: bool = False,
        cache_size: int = 8,
    ) -> None:
        if cache_size < 0:
            raise ValueError("cache_size must be non-negative")
        self.rows = rows
        self.split = normalize_split(split)
        self.deduplicate_samples = deduplicate_samples
        self.qa_row_count = len(rows)

        row_indices: list[int] = []
        sample_ids: list[int] = []
        canonical_by_sample: dict[int, int] = {}
        raw_sample_ids = _column(rows, "sample_id")
        raw_question_ids = _column(rows, "question_id")
        if len(raw_sample_ids) != len(raw_question_ids) or len(raw_sample_ids) != len(rows):
            raise ValueError("sample_id/question_id columns do not match the row count")
        for row_index, (raw_sample_id, raw_question_id) in enumerate(
            zip(raw_sample_ids, raw_question_ids, strict=True)
        ):
            sample_id = validate_sample_id(raw_sample_id, self.split)
            try:
                question_id = int(raw_question_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"question_id must be an integer, got {raw_question_id!r}"
                ) from exc
            if not 0 <= question_id <= 4:
                raise ValueError(f"question_id {question_id} is outside [0, 4]")
            canonical_index = canonical_by_sample.setdefault(sample_id, row_index)
            if deduplicate_samples and canonical_index != row_index:
                continue
            row_indices.append(canonical_index)
            sample_ids.append(sample_id)

        if not row_indices:
            raise ValueError(f"the {self.split} dataset is empty")
        self._row_indices = tuple(row_indices)
        self._sample_ids = tuple(sample_ids)
        self.unique_sample_count = len(canonical_by_sample)

        self._cache_size = cache_size
        self._cache: OrderedDict[int, tuple[torch.Tensor, tuple[int, ...]]] = OrderedDict()

    @property
    def sample_ids(self) -> tuple[int, ...]:
        """Sample IDs in dataset-index order (repeated in fidelity mode)."""

        return self._sample_ids

    def __len__(self) -> int:
        return len(self._row_indices) * SEGMENT_COUNT

    def _decode_row_uncached(self, row_index: int) -> tuple[torch.Tensor, tuple[int, ...]]:
        row = self.rows[row_index]
        if not isinstance(row, Mapping):
            raise ValueError(f"dataset row {row_index} is not a mapping")
        segments = segment_trajectory(row.get("trajectory"))
        labels = action_ids_from_sequence(row.get("action_sequence"))
        return segments, labels

    def _decoded_row(self, row_index: int) -> tuple[torch.Tensor, tuple[int, ...]]:
        cached = self._cache.get(row_index)
        if cached is not None:
            self._cache.move_to_end(row_index)
            return cached
        decoded = self._decode_row_uncached(row_index)
        if self._cache_size:
            self._cache[row_index] = decoded
            self._cache.move_to_end(row_index)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return decoded

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        row_position, segment_id = divmod(index, SEGMENT_COUNT)
        row_index = self._row_indices[row_position]
        segments, labels = self._decoded_row(row_index)
        return {
            "past_values": segments[segment_id],
            "labels": torch.tensor(labels[segment_id], dtype=torch.long),
            "sample_id": torch.tensor(self._sample_ids[row_position], dtype=torch.long),
            "segment_id": torch.tensor(segment_id, dtype=torch.long),
        }

    def validate_all_samples(self) -> dict[int, tuple[str, tuple[int, ...]]]:
        """Validate all rows and prove duplicates have identical supervision.

        Fingerprints keep this check bounded to a few bytes per sample rather
        than retaining all 320-frame tensors in memory.
        """

        signatures: dict[int, tuple[str, tuple[int, ...]]] = {}
        for row_index, raw_sample_id in enumerate(_column(self.rows, "sample_id")):
            sample_id = validate_sample_id(raw_sample_id, self.split)
            row = self.rows[row_index]
            if not isinstance(row, Mapping):
                raise ValueError(f"dataset row {row_index} is not a mapping")
            trajectory = trajectory_array(row.get("trajectory"))
            signature = (
                hashlib.sha256(trajectory.tobytes(order="C")).hexdigest(),
                action_ids_from_sequence(row.get("action_sequence")),
            )
            previous = signatures.setdefault(sample_id, signature)
            if previous != signature:
                raise ValueError(
                    f"sample_id {sample_id} has inconsistent trajectory or action "
                    "labels across QA rows"
                )
        return signatures


def _local_parquet_files(data_root: str | Path, task: TaskName, split: SplitName) -> list[str]:
    root = Path(data_root).expanduser()
    task_root = root if root.name == task else root / task
    files = sorted(task_root.glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"no {task}/{split} Parquet shards found below {root.resolve()}"
        )
    return [str(path) for path in files]


def _file_sha256(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def data_source_manifest(
    task: str,
    splits: Sequence[str],
    *,
    dataset_id: str = DATASET_ID,
    revision: str = DATASET_REVISION,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    """Describe the exact Hub revision or hash every selected local shard.

    Absolute paths are intentionally excluded from the content identity so the
    same shards can move between clusters without changing their fingerprint.
    """

    if task not in TASKS:
        raise ValueError(f"unsupported task {task!r}; expected one of {TASKS}")
    normalized_splits = tuple(dict.fromkeys(normalize_split(split) for split in splits))
    if not normalized_splits:
        raise ValueError("at least one split is required for a data manifest")
    if data_root is None:
        return {
            "kind": "huggingface",
            "dataset_id": dataset_id,
            "revision": revision,
            "task": task,
            "splits": list(normalized_splits),
        }

    root = Path(data_root).expanduser()
    task_root = root if root.name == task else root / task
    shards: list[dict[str, Any]] = []
    for split in normalized_splits:
        for raw_path in _local_parquet_files(root, task, split):  # type: ignore[arg-type]
            path = Path(raw_path)
            try:
                name = str(path.relative_to(task_root))
            except ValueError:
                name = path.name
            shards.append(
                {
                    "split": split,
                    "name": name,
                    "size_bytes": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
            )
    return {
        "kind": "local_parquet",
        "dataset_id": dataset_id,
        "revision": revision,
        "task": task,
        "splits": list(normalized_splits),
        "shards": shards,
    }


def data_source_fingerprint(manifest: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 fingerprint for a data-source manifest."""

    payload = json.dumps(
        manifest,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_data_source_location(
    manifest: Mapping[str, Any],
    data_root: str | Path | None,
) -> None:
    """Check that an invocation selects the location described by a manifest.

    Local content hashes are established by the CPU preflight. This inexpensive
    check revalidates shard names and sizes before allocating a GPU.
    """

    kind = manifest.get("kind")
    if kind == "huggingface":
        if data_root is not None:
            raise ValueError("Hub preflight manifest cannot be used with --data-root")
        return
    if kind != "local_parquet":
        raise ValueError(f"unsupported data manifest kind {kind!r}")
    if data_root is None:
        raise ValueError("local preflight manifest requires --data-root")
    task = manifest.get("task")
    if task not in TASKS:
        raise ValueError("local preflight manifest has an invalid task")
    actual: set[tuple[str, str, int]] = set()
    for split in manifest.get("splits", []):
        normalized_split = normalize_split(split)
        root = Path(data_root).expanduser()
        task_root = root if root.name == task else root / task
        for raw_path in _local_parquet_files(root, task, normalized_split):
            path = Path(raw_path)
            actual.add((normalized_split, str(path.relative_to(task_root)), path.stat().st_size))
    expected = {
        (str(shard["split"]), str(shard["name"]), int(shard["size_bytes"]))
        for shard in manifest.get("shards", [])
    }
    if actual != expected:
        raise ValueError("local shard names or sizes differ from the CPU preflight manifest")


def load_question_rows(
    task: str,
    split: str,
    *,
    dataset_id: str = DATASET_ID,
    revision: str = DATASET_REVISION,
    data_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> Any:
    """Load one released answer-format split from local shards or the Hub.

    Hub loading always passes the pinned revision by default. ``data_root`` is
    useful for an already downloaded dataset and expects ``<root>/<task>/*.parquet``.
    """

    if task not in TASKS:
        raise ValueError(f"unsupported task {task!r}; expected one of {TASKS}")
    normalized_split = normalize_split(split)
    from datasets import load_dataset

    cache = str(Path(cache_dir).expanduser()) if cache_dir is not None else None
    if data_root is not None:
        files = _local_parquet_files(data_root, task, normalized_split)  # type: ignore[arg-type]
        return load_dataset(
            "parquet",
            data_files={normalized_split: files},
            split=normalized_split,
            cache_dir=cache,
        )
    return load_dataset(
        dataset_id,
        task,
        revision=revision,
        split=normalized_split,
        cache_dir=cache,
    )


def load_quants_segments(
    task: str,
    split: str,
    *,
    dataset_id: str = DATASET_ID,
    revision: str = DATASET_REVISION,
    data_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    deduplicate_samples: bool = False,
    validate_all: bool = False,
) -> ActionSegmentDataset:
    """Load a task split and expose its four labeled 80-frame segments."""

    rows = load_question_rows(
        task,
        split,
        dataset_id=dataset_id,
        revision=revision,
        data_root=data_root,
        cache_dir=cache_dir,
    )
    dataset = ActionSegmentDataset(
        rows,
        split=split,
        deduplicate_samples=deduplicate_samples,
    )
    if validate_all:
        dataset.validate_all_samples()
    return dataset


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def preflight_data(
    *,
    output: str | Path,
    tasks: Sequence[str] = TASKS,
    splits: Sequence[str] = ("train", "val", "test"),
    dataset_id: str = DATASET_ID,
    revision: str = DATASET_REVISION,
    data_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validate all selected rows and atomically emit their content manifest."""

    normalized_splits = tuple(dict.fromkeys(normalize_split(split) for split in splits))
    selected_tasks = tuple(dict.fromkeys(tasks))
    if not selected_tasks or any(task not in TASKS for task in selected_tasks):
        raise ValueError(f"tasks must be selected from {TASKS}")

    signatures_by_sample: dict[int, tuple[str, tuple[int, ...]]] = {}
    split_samples: dict[str, set[int]] = {split: set() for split in normalized_splits}
    task_reports: dict[str, Any] = {}
    for task in selected_tasks:
        source = data_source_manifest(
            task,
            normalized_splits,
            dataset_id=dataset_id,
            revision=revision,
            data_root=data_root,
        )
        if source["kind"] == "local_parquet":
            for split in normalized_splits:
                count = EXPECTED_PARQUET_SHARDS[task][split]
                expected_names = {
                    f"{split}-{index:05d}-of-{count:05d}.parquet"
                    for index in range(count)
                }
                actual_names = {
                    shard["name"]
                    for shard in source["shards"]
                    if shard["split"] == split
                }
                if actual_names != expected_names:
                    raise ValueError(
                        f"{task}/{split} local shards do not match the pinned set; "
                        f"missing={sorted(expected_names - actual_names)}, "
                        f"unexpected={sorted(actual_names - expected_names)}"
                    )
        split_reports: dict[str, Any] = {}
        for split in normalized_splits:
            print(f"validating task={task} split={split}", flush=True)
            rows = load_question_rows(
                task,
                split,
                dataset_id=dataset_id,
                revision=revision,
                data_root=data_root,
                cache_dir=cache_dir,
            )
            dataset = ActionSegmentDataset(rows, split=split)
            expected_rows = EXPECTED_ROWS[task][split]
            if dataset.qa_row_count != expected_rows:
                raise ValueError(
                    f"{task}/{split} has {dataset.qa_row_count} rows; "
                    f"expected {expected_rows}"
                )
            raw_sample_ids = _column(rows, "sample_id")
            raw_question_ids = _column(rows, "question_id")
            keys = [
                (validate_sample_id(sample_id, split), int(question_id))
                for sample_id, question_id in zip(
                    raw_sample_ids, raw_question_ids, strict=True
                )
            ]
            if len(keys) != len(set(keys)):
                raise ValueError(f"{task}/{split} contains duplicate sample/question keys")
            signatures = dataset.validate_all_samples()
            for sample_id, signature in signatures.items():
                previous = signatures_by_sample.setdefault(sample_id, signature)
                if previous != signature:
                    raise ValueError(
                        f"sample_id {sample_id} differs across answer-format datasets"
                    )
            split_samples[split].update(signatures)
            split_reports[split] = {
                "qa_rows": dataset.qa_row_count,
                "unique_samples": dataset.unique_sample_count,
                "weighted_segments": len(dataset),
            }
        task_reports[task] = {
            "source_manifest": source,
            "source_fingerprint": data_source_fingerprint(source),
            "splits": split_reports,
        }

    if set(selected_tasks) == set(TASKS):
        for split in normalized_splits:
            lower, upper = SPLIT_ID_RANGES[split]
            expected = set(range(lower, upper + 1))
            if split_samples[split] != expected:
                missing = sorted(expected - split_samples[split])
                unexpected = sorted(split_samples[split] - expected)
                raise ValueError(
                    f"task union for {split} has {len(missing)} missing and "
                    f"{len(unexpected)} unexpected sample IDs"
                )

    report: dict[str, Any] = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "dataset_revision": revision,
        "tasks": task_reports,
        "split_unique_samples": {
            split: len(sample_ids) for split, sample_ids in split_samples.items()
        },
    }
    report["manifest_fingerprint"] = data_source_fingerprint(report)
    _write_json_atomic(Path(output).expanduser(), report)
    return report


def preflight_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate and fingerprint QuAnTS shards")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--task", dest="tasks", action="append", choices=TASKS)
    parser.add_argument("--split", dest="splits", action="append", choices=SPLITS)
    args = parser.parse_args(argv)
    report = preflight_data(
        output=args.output,
        tasks=args.tasks or TASKS,
        splits=args.splits or SPLITS,
        data_root=args.data_root,
        cache_dir=args.cache_dir,
    )
    print(
        f"data preflight passed: {args.output} "
        f"fingerprint={report['manifest_fingerprint']}",
        flush=True,
    )
