"""CSV contracts and crash-safe writers for action-encoder experiments.

The QA prediction CSV deliberately contains the model's raw continuation.  Parsing is
post-hoc so malformed answers remain observable instead of becoming fabricated labels.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

PREDICTION_COLUMNS = ("sample_id", "question_id", "prediction_text")
ACTION_PREDICTION_COLUMNS = (
    "sample_id",
    "segment_id",
    "predicted_action_id",
    "predicted_action_name",
)
JUDGE_COLUMNS = (
    "config",
    "split",
    "sample_id",
    "question_id",
    "brief_rationale",
    "total_rating",
    "normalized_rating",
)


class CsvContractError(ValueError):
    """Raised when a result file violates its public CSV contract."""


def metadata_fingerprint(metadata: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 fingerprint for JSON-serializable metadata."""

    payload = json.dumps(
        metadata,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def path_sha256(path: str | Path) -> str:
    """Hash a file or directory tree using relative names and file contents."""

    source = Path(path)
    if source.is_file():
        return file_sha256(source)
    if not source.is_dir():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in source.rglob("*") if candidate.is_file())
    if not files:
        raise ValueError(f"cannot fingerprint empty directory: {source}")
    for candidate in files:
        relative = candidate.relative_to(source).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with candidate.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: str | Path, value: Mapping[str, Any]) -> None:
    """Write a UTF-8 JSON object and atomically publish it."""

    path = Path(path)
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


def write_csv_atomic(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
) -> None:
    """Write an RFC 4180-compatible UTF-8 CSV and atomically publish it."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = tuple(fieldnames)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            writer = csv.DictWriter(
                handle,
                fieldnames=list(expected),
                extrasaction="raise",
                lineterminator="\r\n",
            )
            writer.writeheader()
            for row in rows:
                if tuple(row.keys()) != expected and set(row) != set(expected):
                    raise CsvContractError(
                        f"expected columns {expected}, received {tuple(row.keys())}"
                    )
                writer.writerow({column: row[column] for column in expected})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def read_csv_rows(path: str | Path, *, fieldnames: Sequence[str]) -> list[dict[str, str]]:
    """Read a CSV while enforcing the exact header and rejecting duplicate headers."""

    expected = tuple(fieldnames)
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        actual = tuple(reader.fieldnames or ())
        if actual != expected:
            raise CsvContractError(f"expected header {expected}, received {actual}")
        rows = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise CsvContractError(f"row does not match the complete header in {path}")
            rows.append(dict(row))
    return rows


def validate_prediction_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_keys: Iterable[tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    """Validate, type, and sort canonical model predictions."""

    normalized: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for row in rows:
        if set(row) != set(PREDICTION_COLUMNS):
            raise CsvContractError(
                f"prediction row must contain exactly {PREDICTION_COLUMNS}: {tuple(row)}"
            )
        sample_id = int(row["sample_id"])
        question_id = int(row["question_id"])
        if not 0 <= sample_id <= 29_999:
            raise CsvContractError(f"sample_id {sample_id} outside [0, 29999]")
        if not 0 <= question_id <= 4:
            raise CsvContractError(f"question_id {question_id} outside [0, 4]")
        prediction_text = row["prediction_text"]
        if not isinstance(prediction_text, str):
            raise CsvContractError("prediction_text must be a Unicode string")
        key = (sample_id, question_id)
        if key in seen:
            raise CsvContractError(f"duplicate prediction key {key}")
        seen.add(key)
        normalized.append(
            {
                "sample_id": sample_id,
                "question_id": question_id,
                "prediction_text": prediction_text,
            }
        )

    if expected_keys is not None:
        expected = set(expected_keys)
        missing = sorted(expected - seen)
        unexpected = sorted(seen - expected)
        if missing or unexpected:
            raise CsvContractError(
                f"prediction keys differ: {len(missing)} missing, "
                f"{len(unexpected)} unexpected"
            )
    normalized.sort(key=lambda row: (row["sample_id"], row["question_id"]))
    return normalized


def validate_action_prediction_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    action_names: Sequence[str] | None = None,
    expected_sample_ids: Iterable[int] | None = None,
) -> list[dict[str, Any]]:
    """Validate the long-form four-segment action prediction contract."""

    normalized: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    segments_by_sample: dict[int, set[int]] = {}
    for row in rows:
        if set(row) != set(ACTION_PREDICTION_COLUMNS):
            raise CsvContractError(
                "action row must contain exactly "
                f"{ACTION_PREDICTION_COLUMNS}: {tuple(row)}"
            )
        sample_id = int(row["sample_id"])
        segment_id = int(row["segment_id"])
        action_id = int(row["predicted_action_id"])
        action_name = row["predicted_action_name"]
        if not 0 <= sample_id <= 29_999:
            raise CsvContractError(f"sample_id {sample_id} outside [0, 29999]")
        if not 0 <= segment_id <= 3:
            raise CsvContractError(f"segment_id {segment_id} outside [0, 3]")
        if not isinstance(action_name, str):
            raise CsvContractError("predicted_action_name must be a Unicode string")
        if action_names is not None:
            if not 0 <= action_id < len(action_names):
                raise CsvContractError(f"predicted_action_id {action_id} is invalid")
            expected_name = action_names[action_id]
            if action_name != expected_name:
                raise CsvContractError(
                    f"action {action_id} must be named {expected_name!r}, got {action_name!r}"
                )
        key = (sample_id, segment_id)
        if key in seen:
            raise CsvContractError(f"duplicate action prediction key {key}")
        seen.add(key)
        segments_by_sample.setdefault(sample_id, set()).add(segment_id)
        normalized.append(
            {
                "sample_id": sample_id,
                "segment_id": segment_id,
                "predicted_action_id": action_id,
                "predicted_action_name": action_name,
            }
        )

    malformed = {
        sample_id: sorted({0, 1, 2, 3} - segments)
        for sample_id, segments in segments_by_sample.items()
        if segments != {0, 1, 2, 3}
    }
    if malformed:
        first = next(iter(malformed.items()))
        raise CsvContractError(
            f"every sample requires segments 0..3; sample {first[0]} misses {first[1]}"
        )
    if expected_sample_ids is not None:
        expected = set(int(value) for value in expected_sample_ids)
        actual = set(segments_by_sample)
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing or unexpected:
            raise CsvContractError(
                f"action sample IDs differ: {len(missing)} missing, "
                f"{len(unexpected)} unexpected"
            )
    normalized.sort(key=lambda row: (row["sample_id"], row["segment_id"]))
    return normalized


def read_prediction_csv(path: str | Path) -> list[dict[str, Any]]:
    return validate_prediction_rows(read_csv_rows(path, fieldnames=PREDICTION_COLUMNS))


def read_action_prediction_csv(
    path: str | Path,
    *,
    action_names: Sequence[str] | None = None,
    expected_sample_ids: Iterable[int] | None = None,
) -> list[dict[str, Any]]:
    return validate_action_prediction_rows(
        read_csv_rows(path, fieldnames=ACTION_PREDICTION_COLUMNS),
        action_names=action_names,
        expected_sample_ids=expected_sample_ids,
    )


class ResumableCsvWriter:
    """Append to a fingerprinted partial CSV, then atomically publish a final CSV.

    A resume is accepted only if configuration, schema, and key columns match.  A
    completed final file is immutable and is validated before it is reused.
    """

    def __init__(
        self,
        output_path: str | Path,
        *,
        fieldnames: Sequence[str],
        key_fields: Sequence[str],
        metadata: Mapping[str, Any],
        resume: bool = True,
    ) -> None:
        self.output_path = Path(output_path)
        self.fieldnames = tuple(fieldnames)
        self.key_fields = tuple(key_fields)
        if not self.key_fields or not set(self.key_fields) <= set(self.fieldnames):
            raise ValueError("key_fields must be a non-empty subset of fieldnames")
        self.fingerprint = metadata_fingerprint(metadata)
        self.metadata = dict(metadata)
        self.partial_path = self.output_path.with_name(self.output_path.name + ".partial")
        self.partial_metadata_path = self.partial_path.with_name(
            self.partial_path.name + ".meta.json"
        )
        self.final_metadata_path = self.output_path.with_name(
            self.output_path.name + ".meta.json"
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._finalized = False
        self._completed_keys: set[tuple[str, ...]] = set()
        self._committed_size = 0
        self._row_count = 0

        if self.output_path.exists():
            self._open_final_or_resume()
            return
        if self.partial_path.exists() or self.partial_metadata_path.exists():
            if not resume:
                raise FileExistsError(
                    f"partial result exists and resume is disabled: {self.partial_path}"
                )
            self._load_partial()
            return
        self._initialize_partial()

    @property
    def completed_keys(self) -> set[tuple[str, ...]]:
        return set(self._completed_keys)

    @property
    def finalized(self) -> bool:
        return self._finalized

    def _manifest(
        self,
        *,
        complete: bool,
        row_count: int,
        committed_size: int | None = None,
    ) -> dict[str, Any]:
        manifest = {
            "schema_version": 2,
            "complete": complete,
            "fingerprint": self.fingerprint,
            "fieldnames": list(self.fieldnames),
            "key_fields": list(self.key_fields),
            "row_count": row_count,
            "configuration": self.metadata,
        }
        if not complete:
            if committed_size is None or committed_size < 0:
                raise ValueError("partial metadata requires a committed byte size")
            manifest["committed_size"] = committed_size
        return manifest

    def _read_manifest(self, path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CsvContractError(f"cannot read metadata {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise CsvContractError(f"metadata is not a JSON object: {path}")
        return value

    def _check_manifest(self, manifest: Mapping[str, Any], *, complete: bool) -> None:
        if manifest.get("fingerprint") != self.fingerprint:
            raise CsvContractError("resume metadata fingerprint does not match configuration")
        if tuple(manifest.get("fieldnames", ())) != self.fieldnames:
            raise CsvContractError("resume metadata has a different CSV schema")
        if tuple(manifest.get("key_fields", ())) != self.key_fields:
            raise CsvContractError("resume metadata has different key columns")
        if bool(manifest.get("complete")) is not complete:
            raise CsvContractError("resume metadata has an inconsistent completion state")
        if manifest.get("schema_version") != 2:
            raise CsvContractError("resume metadata uses an unsupported schema version")

    def _keys_from_rows(self, rows: Sequence[Mapping[str, str]]) -> set[tuple[str, ...]]:
        keys: set[tuple[str, ...]] = set()
        for row in rows:
            key = tuple(row[field] for field in self.key_fields)
            if key in keys:
                raise CsvContractError(f"duplicate key in resumable CSV: {key}")
            keys.add(key)
        return keys

    def _open_final_or_resume(self) -> None:
        rows = read_csv_rows(self.output_path, fieldnames=self.fieldnames)
        if self.final_metadata_path.exists():
            manifest = self._read_manifest(self.final_metadata_path)
            self._check_manifest(manifest, complete=True)
        elif self.partial_metadata_path.exists():
            # Resume a crash after the final CSV rename but before metadata publish.
            partial_manifest = self._read_manifest(self.partial_metadata_path)
            self._check_manifest(partial_manifest, complete=False)
            write_json_atomic(
                self.final_metadata_path,
                self._manifest(complete=True, row_count=len(rows)),
            )
        else:
            raise CsvContractError(f"final CSV has no fingerprint metadata: {self.output_path}")
        self._completed_keys = self._keys_from_rows(rows)
        self._row_count = len(rows)
        self._finalized = True

    def _initialize_partial(self) -> None:
        write_csv_atomic(self.partial_path, (), fieldnames=self.fieldnames)
        self._committed_size = self.partial_path.stat().st_size
        self._row_count = 0
        write_json_atomic(
            self.partial_metadata_path,
            self._manifest(
                complete=False,
                row_count=0,
                committed_size=self._committed_size,
            ),
        )

    def _load_partial(self) -> None:
        if not self.partial_path.exists() or not self.partial_metadata_path.exists():
            raise CsvContractError(
                "partial CSV and its metadata must either both exist or both be absent"
            )
        manifest = self._read_manifest(self.partial_metadata_path)
        self._check_manifest(manifest, complete=False)
        try:
            committed_size = int(manifest["committed_size"])
            row_count = int(manifest["row_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CsvContractError("partial metadata lacks a valid commit boundary") from exc
        actual_size = self.partial_path.stat().st_size
        if actual_size < committed_size:
            raise CsvContractError("partial CSV is shorter than its committed byte boundary")
        if actual_size > committed_size:
            with self.partial_path.open("r+b") as handle:
                handle.truncate(committed_size)
                handle.flush()
                os.fsync(handle.fileno())
        rows = read_csv_rows(self.partial_path, fieldnames=self.fieldnames)
        if len(rows) != row_count:
            raise CsvContractError(
                f"partial metadata commits {row_count} rows, but CSV contains {len(rows)}"
            )
        self._committed_size = committed_size
        self._row_count = row_count
        self._completed_keys = self._keys_from_rows(rows)

    def append_rows(self, rows: Iterable[Mapping[str, Any]]) -> int:
        """Durably append rows not already present in the partial result."""

        if self._finalized:
            raise RuntimeError("cannot append to a finalized CSV")
        pending: list[dict[str, Any]] = []
        new_keys: set[tuple[str, ...]] = set()
        for row in rows:
            if set(row) != set(self.fieldnames):
                raise CsvContractError(
                    f"expected columns {self.fieldnames}, received {tuple(row)}"
                )
            key = tuple(str(row[field]) for field in self.key_fields)
            if key in self._completed_keys or key in new_keys:
                raise CsvContractError(f"duplicate key while appending: {key}")
            new_keys.add(key)
            pending.append({field: row[field] for field in self.fieldnames})
        if not pending:
            return 0
        if self.partial_path.stat().st_size != self._committed_size:
            raise CsvContractError("partial CSV differs from its committed byte boundary")
        with self.partial_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(self.fieldnames),
                extrasaction="raise",
                lineterminator="\r\n",
            )
            writer.writerows(pending)
            handle.flush()
            os.fsync(handle.fileno())
        committed_size = self.partial_path.stat().st_size
        row_count = self._row_count + len(pending)
        write_json_atomic(
            self.partial_metadata_path,
            self._manifest(
                complete=False,
                row_count=row_count,
                committed_size=committed_size,
            ),
        )
        self._committed_size = committed_size
        self._row_count = row_count
        self._completed_keys.update(new_keys)
        return len(pending)

    def finalize(
        self,
        *,
        expected_keys: Iterable[tuple[Any, ...]] | None = None,
    ) -> None:
        """Validate keys, sort by key columns, and atomically publish the result."""

        if self._finalized:
            if expected_keys is not None:
                expected = {tuple(str(value) for value in key) for key in expected_keys}
                if expected != self._completed_keys:
                    raise CsvContractError("completed CSV differs from expected keys")
            return
        if self.partial_path.stat().st_size != self._committed_size:
            raise CsvContractError("partial CSV has uncommitted bytes; reopen it before finalizing")
        rows = read_csv_rows(self.partial_path, fieldnames=self.fieldnames)
        if len(rows) != self._row_count:
            raise CsvContractError("partial CSV row count differs from commit metadata")
        keys = self._keys_from_rows(rows)
        if expected_keys is not None:
            expected = {tuple(str(value) for value in key) for key in expected_keys}
            missing = expected - keys
            unexpected = keys - expected
            if missing or unexpected:
                raise CsvContractError(
                    f"cannot finalize: {len(missing)} keys missing and "
                    f"{len(unexpected)} unexpected"
                )
        rows.sort(key=lambda row: tuple(int(row[field]) for field in self.key_fields))
        write_csv_atomic(self.output_path, rows, fieldnames=self.fieldnames)
        write_json_atomic(
            self.final_metadata_path,
            self._manifest(complete=True, row_count=len(rows)),
        )
        self._completed_keys = keys
        self._finalized = True
        self.partial_path.unlink(missing_ok=True)
        self.partial_metadata_path.unlink(missing_ok=True)
