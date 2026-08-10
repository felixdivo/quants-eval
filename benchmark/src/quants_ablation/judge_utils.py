from __future__ import annotations

import json
from typing import Any


JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "BriefRationale": {"type": "string", "maxLength": 500},
        "TotalRating": {"type": "integer", "minimum": 1, "maximum": 3},
    },
    "required": ["BriefRationale", "TotalRating"],
    "additionalProperties": False,
}

JUDGE_TEMPLATE = """You will be given a scene_description with timestamps and a scene_question regarding that scene. You will also be given a reference_answer and system_answer couple.
Your task is to provide a TotalRating scoring how well the system_answer answers the user concerns expressed in the scene_question. The reference_answer is provided as reference for a very good answer.
Give your answer on a scale of 1 to 3, where 1 means that the system_answer is not helpful at all, and 3 means that the system_answer completely and helpfully addresses the scene_question.
You will also provide a brief rationale for your rating.

Here is the scale you should use to build your answer:
1: The system_answer is terrible: the answer is factually incorrect or corrupt in some way (e.g., empty or nonsensical)
2: The system_answer is bad: somewhat correctly answers the scene_question like the reference_answer, yet contains additional useless/ambiguous/wrong information or is incomplete
3: The system_answer is good: correctly answers the scene_question and is helpful in the context of the scene

Provide your assessment in the following JSON format:
{{
    "BriefRationale": "<your rationale, at most 3 sentences>",
    "TotalRating": <your rating>
}}

You MUST provide values for "BriefRationale" and "TotalRating" in your answer.
Do NOT respond with any other text in your answer.
Immediately, start with the JSON object, without any additional text or explanation.
Make sure to output complete and valid JSON.

Now here are the question and answer:
- scene_description: {scene_description}
- scene_question: {scene_question}
- reference_answer: {reference_answer}
- system_answer: {system_answer}

Provide your assessment."""


def judge_prompt(row: dict[str, Any], prediction: str) -> str:
    return JUDGE_TEMPLATE.format(
        scene_description=row["textual_description"],
        scene_question=row["question"],
        reference_answer=row["answer_text"],
        system_answer=prediction,
    )


def parse_response(text: str) -> tuple[str, int]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])
    rationale = payload["BriefRationale"]
    rating = payload["TotalRating"]
    if not isinstance(rationale, str) or len(rationale) > 500:
        raise ValueError("invalid BriefRationale")
    if type(rating) is not int or rating not in (1, 2, 3):
        raise ValueError("invalid TotalRating")
    return rationale, rating
