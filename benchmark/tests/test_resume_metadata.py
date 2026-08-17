import json

import pytest

from quants_ablation.generate import generation_resume_identity
from quants_ablation.judge import judge_resume_identity
from quants_ablation.resume_metadata import (
    metadata_path,
    prepare_resume_metadata,
    validate_resume_metadata,
)


def test_generation_part_rejects_missing_and_stale_metadata(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (adapter / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "manifest.json").write_text(
        '{"revision":"fixed"}\n', encoding="utf-8"
    )

    part = tmp_path / "result.rank0.csv"
    part.write_text("sample_id,question_id,prediction_text\n", encoding="utf-8")
    identity = generation_resume_identity(
        "question_only", "binary", "test", 0, 1, adapter, data_root, None
    )
    with pytest.raises(ValueError, match="without resume metadata"):
        prepare_resume_metadata(part, "generation-part", identity)

    part.unlink()
    prepare_resume_metadata(part, "generation-part", identity)
    assert metadata_path(part).is_file()
    part.write_text("sample_id,question_id,prediction_text\n", encoding="utf-8")
    stale_identity = dict(identity)
    stale_identity["world_size"] = 2
    with pytest.raises(ValueError, match="stale resume metadata"):
        validate_resume_metadata(part, "generation-part", stale_identity)


def test_judge_checkpoint_rejects_changed_prediction_csv(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "manifest.json").write_text(
        '{"revision":"fixed"}\n', encoding="utf-8"
    )
    prediction = tmp_path / "question_only_open_test.csv"
    prediction.write_text(
        "sample_id,question_id,prediction_text\n27000,0,first\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "judge_question_only_open_test.csv"
    identity = judge_resume_identity(
        "question_only_open", "test", prediction, data_root
    )
    prepare_resume_metadata(checkpoint, "judge-checkpoint", identity)
    checkpoint.write_text("checkpoint\n", encoding="utf-8")

    prediction.write_text(
        "sample_id,question_id,prediction_text\n27000,0,changed\n",
        encoding="utf-8",
    )
    stale_identity = judge_resume_identity(
        "question_only_open", "test", prediction, data_root
    )
    with pytest.raises(ValueError, match="stale resume metadata"):
        prepare_resume_metadata(checkpoint, "judge-checkpoint", stale_identity)

    payload = json.loads(metadata_path(checkpoint).read_text(encoding="utf-8"))
    assert payload["kind"] == "judge-checkpoint"
    assert len(payload["fingerprint"]) == 64
