import json

import pytest

from quants_ablation.aggregate import judge_mean
from quants_ablation.judge import write_judge_csv
from quants_ablation.judge_utils import JUDGE_SCHEMA, judge_prompt, parse_response


def test_judge_prompt_contains_all_fields():
    row = {
        "textual_description": "Scene {with braces}",
        "question": "What happened?",
        "answer_text": "Jumping",
    }
    prompt = judge_prompt(row, "Jumping 🦘")
    assert "scene_description: Scene {with braces}" in prompt
    assert "scene_question: What happened?" in prompt
    assert "reference_answer: Jumping" in prompt
    assert "system_answer: Jumping 🦘" in prompt


def test_judge_response_parser():
    text = json.dumps({"BriefRationale": "Correct.", "TotalRating": 3})
    assert parse_response(text) == ("Correct.", 3)
    assert JUDGE_SCHEMA["properties"]["BriefRationale"]["maxLength"] == 500


def test_aggregate_rejects_wrong_judge_key_with_same_row_count(tmp_path):
    path = tmp_path / "judge_question_only_open_test.csv"
    write_judge_csv(
        path,
        [
            {
                "config": "question_only_open",
                "split": "test",
                "sample_id": 27001,
                "question_id": 0,
                "brief_rationale": "Fine.",
                "total_rating": 2,
                "normalized_rating": 0.5,
            }
        ],
    )
    with pytest.raises(ValueError, match="judge/reference keys differ"):
        judge_mean(path, "question_only_open", "test", {(27000, 0)})


def test_aggregate_rejects_wrong_judge_config(tmp_path):
    path = tmp_path / "judge_question_only_open_test.csv"
    write_judge_csv(
        path,
        [
            {
                "config": "ts_only_open",
                "split": "test",
                "sample_id": 27000,
                "question_id": 0,
                "brief_rationale": "Fine.",
                "total_rating": 2,
                "normalized_rating": 0.5,
            }
        ],
    )
    with pytest.raises(ValueError, match="stale judge checkpoint row"):
        judge_mean(path, "question_only_open", "test", {(27000, 0)})
