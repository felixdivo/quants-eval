from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import quants_action_encoder.data as data_module
from quants_action_encoder.constants import (
    ACTION_NAMES,
    ACTION_TO_ID,
    DATASET_ID,
    DATASET_REVISION,
)
from quants_action_encoder.data import (
    ActionSegmentDataset,
    action_ids_from_sequence,
    action_names_from_sequence,
    data_source_fingerprint,
    data_source_manifest,
    load_question_rows,
    preflight_data,
    segment_trajectory,
    validate_data_source_location,
    validate_sample_id,
)


def _trajectory(offset: float = 0.0) -> np.ndarray:
    values = np.arange(320 * 24 * 3, dtype=np.float32).reshape(320, 24, 3)
    return values + offset


def _action_sequence(names: tuple[str, ...] = ACTION_NAMES[:4]) -> dict[str, list[object]]:
    return {
        "start": [0.0, 4.0, 8.0, 12.0],
        "end": [4.0, 8.0, 12.0, 16.0],
        "action": list(names),
        "action_sentence": ["a", "b", "c", "d"],
    }


def _row(sample_id: int, question_id: int, *, offset: float = 0.0) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "question_id": question_id,
        "trajectory": _trajectory(offset),
        "action_sequence": _action_sequence(),
        "question": "What happens?",
    }


def test_segment_trajectory_preserves_time_and_channel_order() -> None:
    trajectory = _trajectory()
    segments = segment_trajectory(trajectory)
    assert segments.shape == (4, 80, 72)
    assert segments.dtype == torch.float32
    torch.testing.assert_close(
        segments[2],
        torch.from_numpy(trajectory[160:240].reshape(80, 72).copy()),
    )


@pytest.mark.parametrize(
    ("shape", "message"),
    [((320, 72), "expected trajectory shape"), ((319, 24, 3), "expected trajectory shape")],
)
def test_segment_trajectory_rejects_wrong_shape(shape: tuple[int, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        segment_trajectory(np.zeros(shape, dtype=np.float32))


def test_segment_trajectory_rejects_non_finite_values() -> None:
    trajectory = np.zeros((320, 24, 3), dtype=np.float32)
    trajectory[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        segment_trajectory(trajectory)


def test_action_label_mapping_is_stable_and_accepts_both_representations() -> None:
    assert action_ids_from_sequence(_action_sequence(ACTION_NAMES[-4:])) == (15, 16, 17, 18)
    records = [{"action": name} for name in ACTION_NAMES[:4]]
    assert action_names_from_sequence(records) == ACTION_NAMES[:4]


def test_stable_ids_match_the_action_decoder_contract() -> None:
    assert [ACTION_TO_ID[name] for name in ACTION_NAMES[12:]] == list(range(12, 19))
    assert ACTION_TO_ID["kicking a ball"] == 12
    assert ACTION_TO_ID["picking something up with both hands"] == 16
    assert ACTION_TO_ID["T-posing"] == 14


def test_action_labels_reject_unknown_and_wrong_count() -> None:
    with pytest.raises(ValueError, match="unknown QuAnTS action"):
        action_ids_from_sequence(_action_sequence(("unknown", *ACTION_NAMES[1:4])))
    with pytest.raises(ValueError, match="must contain 4"):
        action_ids_from_sequence({"action": list(ACTION_NAMES[:3])})


def test_default_preserves_qa_row_weighting_and_opt_in_deduplicates() -> None:
    rows = [_row(10, 0), _row(10, 1), _row(11, 0, offset=1.0)]
    weighted = ActionSegmentDataset(rows, split="train")
    unique = ActionSegmentDataset(rows, split="train", deduplicate_samples=True)
    weighted.validate_all_samples()

    assert weighted.qa_row_count == 3
    assert weighted.unique_sample_count == 2
    assert len(weighted) == 12
    assert weighted.sample_ids == (10, 10, 11)
    assert len(unique) == 8
    assert unique.sample_ids == (10, 11)
    assert unique[4]["sample_id"].item() == 11
    assert unique[4]["segment_id"].item() == 0
    assert unique[4]["labels"].item() == 0


def test_eager_validation_rejects_inconsistent_duplicate_qa_rows() -> None:
    rows = [_row(10, 0), _row(10, 1, offset=1.0)]
    dataset = ActionSegmentDataset(rows, split="train")
    with pytest.raises(ValueError, match="inconsistent trajectory or action labels"):
        dataset.validate_all_samples()


def test_dataset_validates_split_and_question_ids_before_trajectory_decode() -> None:
    with pytest.raises(ValueError, match="outside the val range"):
        ActionSegmentDataset([_row(23_999, 0)], split="val")
    with pytest.raises(ValueError, match=r"outside \[0, 4\]"):
        ActionSegmentDataset([_row(24_000, 5)], split="validation")
    assert validate_sample_id(24_000, "validation") == 24_000


def test_item_contract_has_model_inputs_and_source_keys() -> None:
    dataset = ActionSegmentDataset([_row(27_000, 2)], split="test")
    item = dataset[3]
    assert set(item) == {"past_values", "labels", "sample_id", "segment_id"}
    assert item["past_values"].shape == (80, 72)
    assert item["labels"].dtype == torch.long
    assert item["sample_id"].item() == 27_000
    assert item["segment_id"].item() == 3


def test_hub_loader_passes_the_pinned_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    sentinel = object()

    def fake_load_dataset(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    result = load_question_rows("binary", "validation")
    assert result is sentinel
    assert calls == [
        (
            (DATASET_ID, "binary"),
            {
                "revision": DATASET_REVISION,
                "split": "val",
                "cache_dir": None,
            },
        )
    ]


def test_local_data_manifest_hashes_shard_contents_not_absolute_root(tmp_path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    for root in (first_root, second_root):
        task_root = root / "binary"
        task_root.mkdir(parents=True)
        (task_root / "train-00000.parquet").write_bytes(b"same shard bytes")

    first = data_source_manifest("binary", ("train",), data_root=first_root)
    second = data_source_manifest("binary", ("train",), data_root=second_root)
    assert first == second
    validate_data_source_location(first, first_root)
    assert data_source_fingerprint(first) == data_source_fingerprint(second)
    assert first["shards"][0]["sha256"] == (
        "dbe82bfc336d29130795303f71b51fe83229f1dab29f4e5389fcfeb56cde5b34"
    )

    (second_root / "binary" / "train-00000.parquet").write_bytes(b"changed")
    changed = data_source_manifest("binary", ("train",), data_root=second_root)
    assert data_source_fingerprint(changed) != data_source_fingerprint(first)
    with pytest.raises(ValueError, match="names or sizes differ"):
        validate_data_source_location(first, second_root)


def test_cpu_preflight_validates_rows_and_writes_a_fingerprinted_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    rows = [_row(0, 0), _row(0, 1)]
    monkeypatch.setattr(data_module, "load_question_rows", lambda *args, **kwargs: rows)
    monkeypatch.setitem(data_module.EXPECTED_ROWS["binary"], "train", 2)
    output = tmp_path / "preflight.json"
    report = preflight_data(
        output=output,
        tasks=("binary",),
        splits=("train",),
    )
    unsigned = {key: value for key, value in report.items() if key != "manifest_fingerprint"}
    assert report["manifest_fingerprint"] == data_source_fingerprint(unsigned)
    assert report["tasks"]["binary"]["splits"]["train"] == {
        "qa_rows": 2,
        "unique_samples": 1,
        "weighted_segments": 8,
    }
    assert output.is_file()
