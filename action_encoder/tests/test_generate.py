from __future__ import annotations

import json
from pathlib import Path

import pytest

import quants_action_encoder.data as data_module
from quants_action_encoder.constants import DATASET_ID, DATASET_REVISION
from quants_action_encoder.generate import _data_identity


def _preflight_report() -> dict[str, object]:
    source = {
        "kind": "local_parquet",
        "dataset_id": DATASET_ID,
        "revision": DATASET_REVISION,
        "task": "binary",
        "splits": ["test"],
        "shards": [
            {
                "split": "test",
                "name": "test-00000-of-00001.parquet",
                "size_bytes": 123,
                "sha256": "a" * 64,
            }
        ],
    }
    report: dict[str, object] = {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "tasks": {
            "binary": {
                "source_manifest": source,
                "source_fingerprint": data_module.data_source_fingerprint(source),
            }
        },
    }
    report["manifest_fingerprint"] = data_module.data_source_fingerprint(report)
    return report


def test_data_identity_verifies_preflight_and_selected_location(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _preflight_report()
    manifest_path = tmp_path / "preflight.json"
    manifest_path.write_text(json.dumps(report), encoding="utf-8")
    calls: list[tuple[dict[str, object], Path | None]] = []

    def validate(manifest, data_root):
        calls.append((dict(manifest), data_root))

    monkeypatch.setattr(data_module, "validate_data_source_location", validate)
    result = _data_identity(
        preflight_manifest=manifest_path,
        data_root=tmp_path / "dataset",
        dataset_id=DATASET_ID,
        dataset_revision=DATASET_REVISION,
        task="binary",
        split="test",
    )

    assert result["preflight_fingerprint"] == report["manifest_fingerprint"]
    assert result["fingerprint"] == data_module.data_source_fingerprint(result["manifest"])
    assert calls == [(result["manifest"], tmp_path / "dataset")]


def test_data_identity_rejects_tampered_preflight(tmp_path: Path) -> None:
    report = _preflight_report()
    report["dataset_revision"] = "tampered"
    manifest_path = tmp_path / "preflight.json"
    manifest_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint is invalid"):
        _data_identity(
            preflight_manifest=manifest_path,
            data_root=tmp_path / "dataset",
            dataset_id=DATASET_ID,
            dataset_revision=DATASET_REVISION,
            task="binary",
            split="test",
        )


def test_data_identity_rejects_internally_consistent_wrong_source(tmp_path: Path) -> None:
    report = _preflight_report()
    task_report = report["tasks"]["binary"]
    source = task_report["source_manifest"]
    source["revision"] = "wrong-source-revision"
    task_report["source_fingerprint"] = data_module.data_source_fingerprint(source)
    del report["manifest_fingerprint"]
    report["manifest_fingerprint"] = data_module.data_source_fingerprint(report)
    manifest_path = tmp_path / "preflight.json"
    manifest_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="source identity differs"):
        _data_identity(
            preflight_manifest=manifest_path,
            data_root=tmp_path / "dataset",
            dataset_id=DATASET_ID,
            dataset_revision=DATASET_REVISION,
            task="binary",
            split="test",
        )


@pytest.mark.parametrize(
    "script_name",
    [
        "10_train.sbatch",
        "20_predict_actions.sbatch",
        "30_xqa_gt.sbatch",
        "31_xqa_predicted.sbatch",
        "45_judge_xqa_open.sbatch",
    ],
)
def test_manifest_bound_slurm_stages_pass_required_data_root(script_name: str) -> None:
    script = Path(__file__).parents[1] / "slurm" / script_name
    text = script.read_text(encoding="utf-8")

    assert '${DATA_ROOT:?Set DATA_ROOT to the verified local QuAnTS snapshot}' in text
    assert '--data-root "${DATA_ROOT}"' in text
    assert '--preflight-manifest "${ACTION_ROOT}/data/preflight_manifest.json"' in text
