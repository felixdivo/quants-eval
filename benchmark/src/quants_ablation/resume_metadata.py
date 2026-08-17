from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable


METADATA_SCHEMA_VERSION = 1


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path, *, name: str | None = None) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": name if name is not None else path.name,
        "bytes": stat.st_size,
        "sha256": sha256_file(path),
    }


def file_metadata(path: Path, *, name: str | None = None) -> dict[str, object]:
    """Return cheap identity for a derived cache already tied to hashed inputs."""
    stat = path.stat()
    return {
        "path": name if name is not None else path.name,
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def source_files_identity(paths: Iterable[Path]) -> list[dict[str, object]]:
    return [file_identity(path, name=path.name) for path in sorted(paths)]


def directory_files_identity(
    root: Path, files: Iterable[Path]
) -> list[dict[str, object]]:
    return [
        file_identity(path, name=path.relative_to(root).as_posix())
        for path in sorted(files)
    ]


def dataset_source_identity(
    data_root: Path, task: str, split: str
) -> dict[str, object]:
    manifest = data_root / "manifest.json"
    if manifest.is_file():
        return {
            "task": task,
            "split": split,
            "manifest": file_identity(manifest, name="manifest.json"),
        }

    shards = sorted((data_root / task).glob(f"{split}-*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"no dataset manifest or {task}/{split} shards below {data_root}"
        )
    return {
        "task": task,
        "split": split,
        "shards": [
            file_metadata(path, name=path.relative_to(data_root).as_posix())
            for path in shards
        ],
    }


def fingerprint(identity: dict[str, Any]) -> str:
    encoded = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def metadata_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(checkpoint_path.name + ".metadata.json")


def expected_metadata(kind: str, identity: dict[str, Any]) -> dict[str, object]:
    return {
        "schema_version": METADATA_SCHEMA_VERSION,
        "kind": kind,
        "fingerprint": fingerprint(identity),
        "identity": identity,
    }


def atomic_json_dump(payload: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False, newline="\n"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def validate_resume_metadata(
    checkpoint_path: Path, kind: str, identity: dict[str, Any]
) -> None:
    sidecar = metadata_path(checkpoint_path)
    if not sidecar.is_file():
        raise ValueError(
            f"checkpoint exists without resume metadata: {checkpoint_path}"
        )
    try:
        with sidecar.open("r", encoding="utf-8") as handle:
            actual = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid resume metadata: {sidecar}") from error
    expected = expected_metadata(kind, identity)
    if actual != expected:
        raise ValueError(f"stale resume metadata for checkpoint: {checkpoint_path}")


def prepare_resume_metadata(
    checkpoint_path: Path, kind: str, identity: dict[str, Any]
) -> None:
    """Validate an existing checkpoint, or atomically prepare its sidecar."""
    if checkpoint_path.exists():
        validate_resume_metadata(checkpoint_path, kind, identity)
        return
    atomic_json_dump(expected_metadata(kind, identity), metadata_path(checkpoint_path))
