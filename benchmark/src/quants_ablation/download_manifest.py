"""Validate a downloaded QuAnTS snapshot and write a content manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

if __package__:
    from .constants import DATASET_REPO, DATASET_REVISION, SPLITS, TASKS
else:  # Support the bootstrap download job before the package is installed.
    from constants import DATASET_REPO, DATASET_REVISION, SPLITS, TASKS


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    root: Path, output: Path | None = None
) -> dict[str, object]:
    if not root.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {root}")

    excluded_output = (output or (root / "manifest.json")).resolve()
    files: list[dict[str, object]] = []
    parquet_counts: dict[str, dict[str, int]] = {}
    for config in TASKS:
        config_dir = root / config
        if not config_dir.is_dir():
            raise FileNotFoundError(f"missing dataset config directory: {config_dir}")
        parquet_counts[config] = {}
        for split in SPLITS:
            shards = sorted(config_dir.glob(f"{split}-*.parquet"))
            if not shards:
                raise FileNotFoundError(f"no {config}/{split} parquet shards found")
            parquet_counts[config][split] = len(shards)

    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or ".cache" in path.parts
            or path.resolve() == excluded_output
        ):
            continue
        stat = path.stat()
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": stat.st_size,
                "sha256": sha256(path),
            }
        )

    return {
        "dataset": DATASET_REPO,
        "revision": DATASET_REVISION,
        "root": str(root.resolve()),
        "parquet_shards": parquet_counts,
        "total_files": len(files),
        "total_bytes": sum(int(item["bytes"]) for item in files),
        "files": files,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.root, args.output)
    atomic_json_dump(manifest, args.output)
    print(
        f"validated {manifest['total_files']} files "
        f"({manifest['total_bytes']} bytes) at {args.root}"
    )


if __name__ == "__main__":
    main()
