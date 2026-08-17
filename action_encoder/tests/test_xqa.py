from __future__ import annotations

import asyncio
import hashlib

import pytest

from quants_action_encoder.xqa import (
    CompletionError,
    StructuredOutputError,
    build_judge_prompt,
    build_messages,
    complete_with_retries,
    parse_judge_response,
    parse_structured_response,
    structured_response_format,
)

ACTIONS = ["running", "bowing", "waving", "playing guitar"]


class FakeCompleter:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.temperatures = []

    async def complete(self, messages, *, temperature=None):
        self.temperatures.append(temperature)
        value = next(self.outputs)
        if isinstance(value, Exception):
            raise value
        return value


@pytest.mark.parametrize(
    ("task", "system_sha256", "suffix"),
    [
        ("binary", "4b993c31a8d904bd40ed097495bbd2907406f75e41ed5c37ad71fe4bce0b6617", ""),
        (
            "multi",
            "0b61ec38ea83f1eb5a332429384dc1d9352a556b1deb50f7d3299c712540d69b",
            "**Respond with exactly one JSON object and no additional explanation or "
            "narrative. If none of the options seem to be valid choose a random one.**",
        ),
        (
            "open",
            "cac36cbf84fbcb25b564141c7f0de968de00b8c4d7d8fe643c025d7e149a7897",
            "**Respond with exactly one JSON object and no additional explanation or "
            "narrative.**",
        ),
    ],
)
def test_prompt_is_golden_for_each_task(task, system_sha256, suffix):
    messages = build_messages(task=task, actions=ACTIONS, question="Which café? 🧪")
    assert [message["role"] for message in messages] == ["system", "user"]
    assert hashlib.sha256(messages[0]["content"].encode("utf-8")).hexdigest() == system_sha256
    assert messages[1]["content"] == (
        'TS: ["running", "bowing", "waving", "playing guitar"], QS: Which café? 🧪'
    )
    if suffix:
        assert messages[0]["content"].endswith(suffix)
    else:
        assert "Respond with exactly one JSON object" not in messages[0]["content"]
    assert "http://" not in str(messages)


def test_parse_structured_response_extracts_json_but_does_not_change_raw_text():
    raw = (
        "prefix\n"
        '{"actions":["running","bowing","waving","playing guitar"],'
        '"steps":["1. Count.","2. Select."],"answer":"B"}\ntrailer'
    )
    parsed = parse_structured_response(raw, task="multi")
    assert parsed.answer == "B"
    assert parsed.steps == ("1. Count.", "2. Select.")


def test_parse_accepts_nonempty_action_list_but_rejects_missing_steps():
    parsed = parse_structured_response(
        '{"actions":["running"],"steps":["1. Count."],"answer":"true"}',
        task="binary",
    )
    assert parsed.actions == ("running",)
    with pytest.raises(StructuredOutputError, match="steps"):
        parse_structured_response(
            '{"actions":["running","bowing","waving","playing guitar"],'
            '"answer":"true"}',
            task="binary",
        )
    with pytest.raises(StructuredOutputError, match="numbered step"):
        parse_structured_response(
            '{"actions":["running"],"steps":["Compare actions."],"answer":"true"}',
            task="binary",
        )


def test_retry_returns_valid_later_response():
    valid = (
        '{"actions":["running","bowing","waving","playing guitar"],'
        '"steps":["1. Compare."],"answer":"false"}'
    )
    client = FakeCompleter(["not json", valid])
    result = asyncio.run(
        complete_with_retries(
            client,
            task="binary",
            actions=ACTIONS,
            question="Same?",
            max_attempts=2,
            retry_delay_seconds=0,
        )
    )
    assert result.valid
    assert result.raw_text == valid
    assert result.attempt_count == 2
    assert client.temperatures == [None, 0.3]


def test_exhausted_validation_preserves_last_invalid_output():
    result = asyncio.run(
        complete_with_retries(
            FakeCompleter(["bad one", "bad two 🧪"]),
            task="open",
            actions=ACTIONS,
            question="Describe this.",
            max_attempts=2,
            retry_delay_seconds=0,
        )
    )
    assert not result.valid
    assert result.raw_text == "bad two 🧪"
    assert result.raw_attempts == ("bad one", "bad two 🧪")


def test_no_endpoint_output_raises_instead_of_fabricating_answer():
    with pytest.raises(CompletionError, match="no continuation"):
        asyncio.run(
            complete_with_retries(
                FakeCompleter([RuntimeError("offline")]),
                task="multi",
                actions=ACTIONS,
                question="Which?",
                max_attempts=1,
                retry_delay_seconds=0,
            )
        )


def test_strict_response_schema_has_all_fields():
    response_format = structured_response_format("binary")
    schema = response_format["json_schema"]["schema"]
    assert response_format["type"] == "json_schema"
    assert schema["required"] == ["actions", "steps", "answer"]
    assert schema["properties"]["answer"]["enum"] == ["true", "false"]
    assert schema["properties"]["steps"]["items"]["pattern"] == r"^\d+\.\s.+"


def test_judge_prompt_and_parser_match_rating_contract():
    prompt = build_judge_prompt(
        {
            "textual_description": "0-4 seconds: running",
            "question": "What happens?",
            "answer_text": "The person runs.",
        },
        "The person runs.",
    )
    assert "scene_description: 0-4 seconds: running" in prompt
    rationale, rating = parse_judge_response(
        'prefix {"BriefRationale":"Correct and concise.","TotalRating":3}'
    )
    assert rationale == "Correct and concise."
    assert rating == 3
    with pytest.raises(StructuredOutputError, match="TotalRating"):
        parse_judge_response('{"BriefRationale":"bad","TotalRating":4}')
