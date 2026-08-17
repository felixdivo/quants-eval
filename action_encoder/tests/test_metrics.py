from __future__ import annotations

import json

import pytest

from quants_action_encoder.csvio import read_csv_rows, write_csv_atomic
from quants_action_encoder.metrics import (
    XQA_METRIC_COLUMNS,
    XQA_PARSED_COLUMNS,
    action_rows_to_sequences,
    aggregate_xqa_metric_files,
    evaluate_action_sequences,
    evaluate_qa_row_weighted,
    evaluate_xqa_predictions,
    ground_truth_from_rows,
    write_metric_outputs,
)


def test_segment_position_exact_and_class_metrics(tmp_path):
    ground_truth = {1: [0, 1, 2, 3], 2: [0, 1, 2, 3], 3: [0, 0, 0, 0]}
    predictions = {1: [0, 1, 2, 3], 2: [0, 1, 1, 3], 99: [0, 0, 0, 0]}
    result = evaluate_action_sequences(
        ground_truth,
        predictions,
        action_names=("a", "b", "c", "d"),
    )

    summary = result["summary"]
    assert summary["matched_samples"] == 2
    assert summary["missing_samples"] == 1
    assert summary["unexpected_samples"] == 1
    assert summary["segment_accuracy"] == pytest.approx(7 / 8)
    assert summary["exact_sequence_accuracy"] == pytest.approx(1 / 2)
    assert [row["accuracy"] for row in result["by_position"]] == [1, 1, 0.5, 1]
    assert result["per_class"][2]["recall"] == pytest.approx(0.5)

    weighted = {
        **result,
        "by_question_type": [{"question_type": "count", **result["summary"]}],
    }
    write_metric_outputs(
        tmp_path,
        result,
        qa_row_weighted_result=weighted,
        metadata={"split": "test"},
    )
    payload = json.loads((tmp_path / "action_metrics.json").read_text())
    assert payload["metadata"]["split"] == "test"
    assert payload["views"]["unique_sample"]["summary"]["matched_samples"] == 2
    assert (tmp_path / "action_metrics_unique_sample_summary.csv").exists()
    assert (tmp_path / "action_metrics_qa_row_weighted_summary.csv").exists()
    assert (tmp_path / "action_metrics_qa_row_weighted_by_question_type.csv").exists()


def test_ground_truth_rows_are_deduplicated_and_checked():
    row = {
        "sample_id": 10,
        "action_sequence": {"action": ["a", "b", "c", "d"]},
    }
    import quants_action_encoder.metrics as metrics

    original = metrics.ACTION_NAMES
    metrics.ACTION_NAMES = ("a", "b", "c", "d")
    try:
        assert ground_truth_from_rows([row, dict(row)]) == {10: [0, 1, 2, 3]}
        changed = {
            "sample_id": 10,
            "action_sequence": {"action": ["a", "b", "d", "c"]},
        }
        with pytest.raises(ValueError, match="inconsistent"):
            ground_truth_from_rows([row, changed])
    finally:
        metrics.ACTION_NAMES = original


def test_action_rows_to_sequences_uses_segment_order():
    rows = [
        {
            "sample_id": 5,
            "segment_id": segment,
            "predicted_action_id": action,
            "predicted_action_name": "unused",
        }
        for segment, action in ((3, 9), (1, 7), (0, 6), (2, 8))
    ]
    assert action_rows_to_sequences(rows) == {5: [6, 7, 8, 9]}


def test_qa_row_weighted_view_counts_repeated_samples():
    action_names = [
        "holding a baby",
        "shaking hands",
        "running",
        "jumping once",
    ]
    rows = [
        {
            "sample_id": 1,
            "question_id": question_id,
            "question_type": question_type,
            "action_sequence": {"action": action_names},
        }
        for question_id, question_type in ((0, "count"), (1, "order"))
    ]
    result = evaluate_qa_row_weighted(rows, {1: [0, 1, 2, 0]})
    assert result["summary"]["ground_truth_qa_rows"] == 2
    assert result["summary"]["segment_total"] == 8
    assert result["summary"]["segment_correct"] == 6
    assert {row["question_type"] for row in result["by_question_type"]} == {
        "count",
        "order",
    }
    assert all(
        row["unexpected_prediction_samples"] == 0
        for row in result["by_question_type"]
    )


def _raw_answer(answer: str) -> str:
    return json.dumps(
        {
            "actions": ["running", "bowing", "waving", "playing guitar"],
            "steps": ["1. Evaluate the question."],
            "answer": answer,
        },
        ensure_ascii=False,
    )


def _reference(sample_id, question_id, *, answer=None, answer_text="", qtype="count"):
    row = {
        "sample_id": sample_id,
        "question_id": question_id,
        "question_type": qtype,
        "question": f"Question {sample_id}/{question_id}?",
        "answer_text": answer_text,
    }
    if answer is not None:
        row["answer"] = answer
    return row


def test_binary_xqa_metrics_count_invalid_as_incorrect():
    references = [_reference(1, 0, answer=1), _reference(2, 0, answer=0)]
    predictions = [
        {"sample_id": 1, "question_id": 0, "prediction_text": _raw_answer("true")},
        {"sample_id": 2, "question_id": 0, "prediction_text": "not JSON"},
    ]
    metrics, parsed = evaluate_xqa_predictions(
        references,
        predictions,
        task="binary",
        config="xqa_gt_binary",
        split="test",
    )
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["macro_f1"] == pytest.approx(0.5)
    assert metrics["invalid_output_rate"] == pytest.approx(0.5)
    assert parsed[1]["valid_output"] == 0
    assert parsed[1]["parsed_answer"] == ""


def test_multi_xqa_metrics_parse_a_b_c_and_invalid():
    references = [_reference(index, 0, answer=index) for index in range(3)]
    predictions = [
        {"sample_id": 0, "question_id": 0, "prediction_text": _raw_answer("A")},
        {"sample_id": 1, "question_id": 0, "prediction_text": _raw_answer("B")},
        {"sample_id": 2, "question_id": 0, "prediction_text": _raw_answer("D")},
    ]
    metrics, _ = evaluate_xqa_predictions(
        references,
        predictions,
        task="multi",
        config="xqa_predicted_multi",
        split="train",
    )
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["invalid_output_rate"] == pytest.approx(1 / 3)


def test_open_xqa_metrics_unicode_parsed_csv_and_exact_judge_keys(tmp_path):
    references = [
        _reference(1, 0, answer_text="Tokyo 東京 🧪", qtype="describe"),
        _reference(2, 1, answer_text="running", qtype="describe"),
    ]
    predictions = [
        {
            "sample_id": 1,
            "question_id": 0,
            "prediction_text": _raw_answer("Tokyo 東京 🧪"),
        },
        {"sample_id": 2, "question_id": 1, "prediction_text": "invalid"},
    ]
    judges = [
        {
            "config": "xqa_gt_open",
            "split": "test",
            "sample_id": 1,
            "question_id": 0,
            "brief_rationale": "correct",
            "total_rating": 3,
            "normalized_rating": 1.0,
        },
        {
            "config": "xqa_gt_open",
            "split": "test",
            "sample_id": 2,
            "question_id": 1,
            "brief_rationale": "invalid",
            "total_rating": 1,
            "normalized_rating": 0.0,
        },
    ]
    metrics, parsed = evaluate_xqa_predictions(
        references,
        predictions,
        task="open",
        config="xqa_gt_open",
        split="test",
        judge_rows=judges,
    )
    assert metrics["rouge_l_f1"] == pytest.approx(0.5)
    assert 0 < metrics["meteor"] <= 0.5
    assert metrics["llm_judge"] == pytest.approx(0.5)
    parsed_path = tmp_path / "parsed.csv"
    write_csv_atomic(parsed_path, parsed, fieldnames=XQA_PARSED_COLUMNS)
    assert read_csv_rows(parsed_path, fieldnames=XQA_PARSED_COLUMNS)[0][
        "parsed_answer"
    ] == "Tokyo 東京 🧪"

    with pytest.raises(ValueError, match="judge keys differ"):
        evaluate_xqa_predictions(
            references,
            predictions,
            task="open",
            config="xqa_gt_open",
            split="test",
            judge_rows=judges[:1],
        )


def test_xqa_metrics_reject_key_mismatch():
    with pytest.raises(ValueError, match="xQA keys differ"):
        evaluate_xqa_predictions(
            [_reference(1, 0, answer=1)],
            [],
            task="binary",
            config="xqa_gt_binary",
            split="test",
        )


def test_aggregate_six_xqa_metric_files(tmp_path):
    paths = []
    for index, (mode, task) in enumerate(
        (mode, task)
        for mode in ("gt", "predicted")
        for task in ("binary", "multi", "open")
    ):
        row = {column: "" for column in XQA_METRIC_COLUMNS}
        row.update(
            {
                "config": f"xqa_{mode}_{task}",
                "split": "test",
                "task": task,
                "num_examples": 1,
                "valid_examples": 1,
                "invalid_examples": 0,
                "invalid_output_rate": 0.0,
            }
        )
        path = tmp_path / f"part-{index}.csv"
        write_csv_atomic(path, [row], fieldnames=XQA_METRIC_COLUMNS)
        paths.append(path)
    output = tmp_path / "all.csv"
    rows = aggregate_xqa_metric_files(paths, output)
    assert len(rows) == 6
    assert len(read_csv_rows(output, fieldnames=XQA_METRIC_COLUMNS)) == 6
