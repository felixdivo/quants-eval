from quants_ablation.constants import SPLITS, TASKS
from quants_ablation.download_manifest import atomic_json_dump, build_manifest


def test_manifest_is_idempotent_when_built_twice(tmp_path):
    for task in TASKS:
        task_root = tmp_path / task
        task_root.mkdir()
        for split in SPLITS:
            (task_root / f"{split}-00000-of-00001.parquet").write_bytes(
                f"{task}/{split}".encode()
            )

    destination = tmp_path / "manifest.json"
    first = build_manifest(tmp_path, destination)
    atomic_json_dump(first, destination)
    second = build_manifest(tmp_path, destination)

    assert second == first
    assert "manifest.json" not in {item["path"] for item in second["files"]}
