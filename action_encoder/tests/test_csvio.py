from __future__ import annotations

import csv

import pytest

from quants_action_encoder.csvio import (
    ACTION_PREDICTION_COLUMNS,
    PREDICTION_COLUMNS,
    CsvContractError,
    ResumableCsvWriter,
    read_action_prediction_csv,
    read_prediction_csv,
    validate_action_prediction_rows,
    write_csv_atomic,
)


def test_prediction_csv_round_trips_raw_unicode_and_multiline_text(tmp_path):
    path = tmp_path / "predictions.csv"
    rows = [
        {"sample_id": 8, "question_id": 4, "prediction_text": '"quoted", emoji 🧪\n東京'},
        {"sample_id": 3, "question_id": 0, "prediction_text": " café "},
    ]
    write_csv_atomic(path, rows, fieldnames=PREDICTION_COLUMNS)

    assert path.read_bytes().startswith(b"sample_id,question_id,prediction_text\r\n")
    assert read_prediction_csv(path) == [
        {"sample_id": 3, "question_id": 0, "prediction_text": " café "},
        {"sample_id": 8, "question_id": 4, "prediction_text": '"quoted", emoji 🧪\n東京'},
    ]


def test_resumable_writer_reuses_partial_and_publishes_sorted_file(tmp_path):
    path = tmp_path / "binary_test.csv"
    metadata = {"task": "binary", "split": "test", "seed": 42}
    writer = ResumableCsvWriter(
        path,
        fieldnames=PREDICTION_COLUMNS,
        key_fields=("sample_id", "question_id"),
        metadata=metadata,
    )
    writer.append_rows(
        [{"sample_id": 9, "question_id": 2, "prediction_text": "first\nraw"}]
    )

    resumed = ResumableCsvWriter(
        path,
        fieldnames=PREDICTION_COLUMNS,
        key_fields=("sample_id", "question_id"),
        metadata=metadata,
    )
    assert resumed.completed_keys == {("9", "2")}
    resumed.append_rows(
        [{"sample_id": 4, "question_id": 1, "prediction_text": "second"}]
    )
    resumed.finalize(expected_keys=[(4, 1), (9, 2)])

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["sample_id"], row["question_id"]) for row in rows] == [
        ("4", "1"),
        ("9", "2"),
    ]
    assert not path.with_name(path.name + ".partial").exists()
    assert path.with_name(path.name + ".meta.json").exists()


def test_resumable_writer_rejects_changed_configuration(tmp_path):
    path = tmp_path / "predictions.csv"
    ResumableCsvWriter(
        path,
        fieldnames=PREDICTION_COLUMNS,
        key_fields=("sample_id", "question_id"),
        metadata={"model": "one"},
    )
    with pytest.raises(CsvContractError, match="fingerprint"):
        ResumableCsvWriter(
            path,
            fieldnames=PREDICTION_COLUMNS,
            key_fields=("sample_id", "question_id"),
            metadata={"model": "two"},
        )


def test_resume_discards_bytes_after_last_committed_batch(tmp_path):
    path = tmp_path / "predictions.csv"
    metadata = {"task": "open", "split": "test"}
    writer = ResumableCsvWriter(
        path,
        fieldnames=PREDICTION_COLUMNS,
        key_fields=("sample_id", "question_id"),
        metadata=metadata,
    )
    writer.append_rows(
        [{"sample_id": 1, "question_id": 0, "prediction_text": "committed"}]
    )
    with writer.partial_path.open("ab") as handle:
        handle.write(b'2,1,"torn continuation')

    resumed = ResumableCsvWriter(
        path,
        fieldnames=PREDICTION_COLUMNS,
        key_fields=("sample_id", "question_id"),
        metadata=metadata,
    )
    assert resumed.completed_keys == {("1", "0")}
    resumed.append_rows(
        [{"sample_id": 2, "question_id": 1, "prediction_text": "retried"}]
    )
    resumed.finalize(expected_keys=[(1, 0), (2, 1)])
    assert [row["prediction_text"] for row in read_prediction_csv(path)] == [
        "committed",
        "retried",
    ]


def test_reader_rejects_row_missing_a_column(tmp_path):
    path = tmp_path / "short.csv"
    path.write_text("sample_id,question_id,prediction_text\r\n1,0\r\n", encoding="utf-8")
    with pytest.raises(CsvContractError, match="complete header"):
        read_prediction_csv(path)


def test_action_prediction_contract_requires_four_consistent_segments(tmp_path):
    names = ["run", "jump"]
    rows = [
        {
            "sample_id": 17,
            "segment_id": segment,
            "predicted_action_id": segment % 2,
            "predicted_action_name": names[segment % 2],
        }
        for segment in range(4)
    ]
    path = tmp_path / "actions.csv"
    write_csv_atomic(path, rows, fieldnames=ACTION_PREDICTION_COLUMNS)
    assert read_action_prediction_csv(path, action_names=names) == rows

    with pytest.raises(CsvContractError, match="segments 0..3"):
        validate_action_prediction_rows(rows[:-1], action_names=names)
