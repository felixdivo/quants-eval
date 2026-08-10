from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_download_module():
    path = Path(__file__).parents[1] / "analysis" / "download_data.py"
    spec = importlib.util.spec_from_file_location("download_data", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_download_manifest_is_stable_and_excludes_huggingface_metadata(tmp_path):
    module = _load_download_module()
    for task, splits in module.SHARD_COUNTS.items():
        task_root = tmp_path / task
        task_root.mkdir()
        for split, count in splits.items():
            for index in range(count):
                path = task_root / f"{split}-{index:05d}-of-{count:05d}.parquet"
                path.write_bytes(f"{task}/{split}/{index}".encode())
    (tmp_path / "README.md").write_text("dataset\n", encoding="utf-8")
    (tmp_path / "LICENSE").write_text("license\n", encoding="utf-8")
    metadata = tmp_path / ".cache" / "huggingface" / "download.lock"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("environment-specific", encoding="utf-8")

    split_reports = {"binary/train": {"rows": 1, "unique_samples": 1}}
    first = module.build_download_manifest(tmp_path, split_reports)
    second = module.build_download_manifest(tmp_path, split_reports)
    assert first == second
    assert not any(name.startswith(".cache/") for name in first["files"])
