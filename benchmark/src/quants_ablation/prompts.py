from __future__ import annotations

import json
from typing import Any

import numpy as np


TS_INSTRUCTION = (
    "### Instruction: You are provided with a 3-dimensional dataset of shape "
    "[320, 24, 3], representing 3-dimensional spatial locations of 24 human-body "
    "joints over 320 time steps. Using this data, please analyze the movements and "
    "provide the answer associated with this human activity sequence."
)

NAIVE_INSTRUCTION = (
    "### Instruction: You are provided with a 3-dimensional dataset of shape "
    "[320, 24, 3], representing 3-dimensional spatial locations of 24 human-body "
    "joints over 320 time steps. Using this data, please analyze the movements and "
    "respond to the following question related to human activity recognition."
)

ANSWER_SUFFIX = "\n\n### Answer:"


def question_only_prompt(question: str) -> str:
    return f"### Question: {question}\n\n### Answer:"


def scale_trajectory(trajectory: Any) -> np.ndarray:
    values = np.asarray(trajectory, dtype=np.float32)
    if values.shape != (320, 24, 3):
        raise ValueError(f"expected trajectory shape (320, 24, 3), got {values.shape}")
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise ValueError("trajectory contains a non-finite value")
    if maximum == minimum:
        return np.zeros(values.shape, dtype=np.int16)
    scaled = (values.astype(np.float64) - minimum) * (
        1998.0 / (maximum - minimum)
    ) - 999.0
    return np.rint(scaled).clip(-999, 999).astype(np.int16)


def serialize_trajectory(trajectory: Any) -> str:
    scaled = scale_trajectory(trajectory)
    return json.dumps(scaled.tolist(), ensure_ascii=False, separators=(",", ":"))


def ts_only_prompt(trajectory: Any) -> str:
    data = serialize_trajectory(trajectory)
    return f"{TS_INSTRUCTION}\n\n### Data: {data}{ANSWER_SUFFIX}"


def naive_prompt(trajectory: Any, question: str) -> str:
    data = serialize_trajectory(trajectory)
    return (
        f"{NAIVE_INSTRUCTION}\n\n### Data: {data}"
        f"\n\n### Question: {question}{ANSWER_SUFFIX}"
    )


def naive_prompt_ids(
    ts_prompt_ids: list[int], question: str, tokenizer: Any
) -> list[int]:
    """Reuse the cached compact-JSON tokens and append the question exactly once.

    Q3 and Naive use different instruction prefixes, so the prefix is replaced at
    token level while the large serialized data portion is retained. Both replacement
    boundaries are newline-delimited.
    """
    # End the header marker at the colon so the cached token that may merge the
    # following space/opening brackets with the first number remains untouched.
    # Include the closing JSON brackets in the tail marker because the tokenizer
    # merges the last bracket with the following newlines.
    old_header = tokenizer(
        f"{TS_INSTRUCTION}\n\n### Data:", add_special_tokens=True
    )["input_ids"]
    new_header = tokenizer(
        f"{NAIVE_INSTRUCTION}\n\n### Data:", add_special_tokens=True
    )["input_ids"]
    old_tail = tokenizer(f"]]]{ANSWER_SUFFIX}", add_special_tokens=False)["input_ids"]
    if ts_prompt_ids[: len(old_header)] != old_header:
        raise ValueError("cached TS prompt does not start with the expected instruction")
    if ts_prompt_ids[-len(old_tail) :] != old_tail:
        raise ValueError("cached TS prompt does not end with the expected answer suffix")
    new_tail = tokenizer(
        f"]]]\n\n### Question: {question}{ANSWER_SUFFIX}",
        add_special_tokens=False,
    )["input_ids"]
    return [
        *new_header,
        *ts_prompt_ids[len(old_header) : -len(old_tail)],
        *new_tail,
    ]


def target_text(row: dict[str, Any], task: str) -> str:
    if task == "binary":
        answer = int(row["answer"])
        if answer not in (0, 1):
            raise ValueError(f"invalid binary answer {answer}")
        return "Yes" if answer == 1 else "No"
    if task == "multi":
        answer = int(row["answer"])
        if answer not in (0, 1, 2):
            raise ValueError(f"invalid multiple-choice answer {answer}")
        return ("A", "B", "C")[answer]
    if task == "open":
        return str(row["answer_text"])
    raise ValueError(f"unknown task: {task}")


def completion_example(
    prompt_ids: list[int], answer_ids: list[int], eos_token_id: int
) -> dict[str, list[int]]:
    completion_ids = [*answer_ids, eos_token_id]
    input_ids = [*prompt_ids, *completion_ids]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + completion_ids,
    }
