"""xQA prompting and a small OpenAI-compatible asynchronous HTTP client."""

# Prompt and judge templates use exact long lines for stable request fingerprints.
# ruff: noqa: E501

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

TaskName = Literal["binary", "multi", "open"]
PROMPT_VERSION = "quants-action-names-v1"
STEP_PATTERN = re.compile(r"^\d+\.\s.+")


class CompletionError(RuntimeError):
    """Raised when an endpoint never returns any model continuation."""


class StructuredOutputError(ValueError):
    """Raised when a continuation is not a valid xQA response object."""


@dataclass(frozen=True)
class StructuredResponse:
    actions: tuple[str, ...]
    steps: tuple[str, ...]
    answer: str


@dataclass(frozen=True)
class XQAResult:
    """One endpoint result, including raw invalid output when parsing failed."""

    raw_text: str
    parsed: StructuredResponse | None
    attempt_count: int
    validation_error: str | None
    raw_attempts: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return self.parsed is not None


class AsyncCompleter(Protocol):
    async def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
    ) -> str: ...


def _task_suffix(task: TaskName) -> str:
    if task == "binary":
        return ""
    if task == "multi":
        return (
            "**Respond with exactly one JSON object and no additional explanation or "
            "narrative. If none of the options seem to be valid choose a random one.**"
        )
    if task == "open":
        return (
            "**Respond with exactly one JSON object and no additional explanation or "
            "narrative.**"
        )
    raise ValueError(f"unsupported task: {task}")


def _few_shot_examples(task: TaskName) -> str:
    """Return task-specific few-shot examples as valid JSON."""

    if task == "binary":
        return """## Example 1
TS: ["running", "bowing", "jumping once", "playing guitar"], QS: Do the person's actions stay the same before and after they are jumping once?
{
  "actions": ["running", "bowing", "jumping once", "playing guitar"],
  "steps": [
    "1. Identify the index of \\\"jumping once\\\" which is 2.",
    "2. Now we need to look before and after index 2, thus index 1 and index 3.",
    "3. At actions[1] the action is bowing and actions[3] the action is playing guitar.",
    "4. Because these two actions are different, the answer is false."
  ],
  "answer": "false"
}
## Example 2
TS: ["catching a ball", "running", "catching a ball", "running"], QS: Is the person conducting catching a ball the same amount as running?
{
  "actions": ["catching a ball", "running", "catching a ball", "running"],
  "steps": [
    "1. Let's count catching a ball. Catching a ball occurs at index 0 and 2, thus 2 times.",
    "2. Let's count running. Running occurs at index 1 and 3, thus 2 times.",
    "3. Therefore, the answer is true."
  ],
  "answer": "true"
}"""
    if task == "multi":
        return """## Example 1
TS: ["running", "jumping once", "shaking hands", "jumping once"], QS: Which activity does the person carry out 1 time? A: jumping once, B: shaking hands, C: golfing (swinging a club)
{
  "actions": ["running", "jumping once", "shaking hands", "jumping once"],
  "steps": [
    "1. Identify the actions that are performed only once. From the timeseries we see that running and shaking hands occur only once.",
    "2. Among A,B,C, only shaking hands matches this (so B).",
    "3. Therefore, the answer is **B**."
  ],
  "answer": "B"
}
## Example 2
TS: ["golfing (swinging a club)", "golfing (swinging a club)", "shaking hands", "running"], QS: Which activity does the person carry out 0 times? A: golfing (swinging a club), B: shaking hands, C: punching
{
  "actions": ["golfing (swinging a club)", "golfing (swinging a club)", "shaking hands", "running"],
  "steps": [
    "1. Rule out actions performed > 0 times.",
    "2. golfing (swinging a club) appears (consecutive repeats count as 1), so rule out A.",
    "3. shaking hands appears once, so rule out B.",
    "4. punching does not appear, so answer is **C**."
  ],
  "answer": "C"
}"""
    if task == "open":
        return """## Example 1
TS: ["running", "punching", "punching", "playing guitar"], QS: Describe in plain English what the person is doing.
{
  "actions": ["running", "punching", "punching", "playing guitar"],
  "steps": [
    "1. Notice two consecutive occurrences of the same action count as one.",
    "2. Unique action sequence = [running, punching, playing guitar].",
    "3. Summarize in plain English."
  ],
  "answer": "The person running, then punching, then playing guitar."
}
## Example 2
TS: ["jumping once", "jumping once", "shaking hands", "golfing (swinging a club)"], QS: Explain step by step how many times each action occurs, and then summarize.
{
  "actions": ["jumping once", "jumping once", "shaking hands", "golfing (swinging a club)"],
  "steps": [
    "1. Count unique consecutive actions.",
    "2. jumping once appears consecutively, so count=1.",
    "3. shaking hands appears once, so count=1.",
    "4. golfing (swinging a club) appears once, so count=1.",
    "5. Totals: jumping once=1, shaking hands=1, golfing (swinging a club)=1.",
    "6. Summarize in one sentence."
  ],
  "answer": "The person jumps once, shakes hands once, and swings a golf club once."
}"""
    raise ValueError(f"unsupported task: {task}")


def build_messages(
    *,
    task: TaskName,
    actions: Sequence[str],
    question: str,
) -> list[dict[str, str]]:
    """Build the xQA prompt with deterministic JSON rendering."""

    if len(actions) != 4 or not all(isinstance(action, str) for action in actions):
        raise ValueError("xQA requires exactly four action names")
    if not isinstance(question, str) or not question:
        raise ValueError("question must be a non-empty string")
    system = (
        "You are a helpful timeseries question answering model.\n"
        "You are given a timeseries (TS) of **action names** and a question about them.\n"
        "If an action appears multiple times in a row in the TS, it counts as **one** "
        "occurrence (consecutive repeats collapse conceptually).\n"
        'Translate the timeseries into reasoning steps and then answer the question. '
        'Provide your thought process in "steps": [...].\n\n'
        + _few_shot_examples(task)
        + ("\n\n" + _task_suffix(task) if _task_suffix(task) else "")
    )
    action_json = json.dumps(list(actions), ensure_ascii=False)
    user = f"TS: {action_json}, QS: {question}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _first_json_object(raw_text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    cursor = 0
    while True:
        start = raw_text.find("{", cursor)
        if start < 0:
            raise StructuredOutputError("no JSON object found")
        try:
            value, _ = decoder.raw_decode(raw_text[start:])
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        if isinstance(value, dict):
            return value
        cursor = start + 1


def parse_structured_response(
    raw_text: str,
    *,
    task: TaskName,
) -> StructuredResponse:
    """Parse and validate the first JSON object without altering the raw text."""

    if not isinstance(raw_text, str):
        raise StructuredOutputError("model output is not text")
    value = _first_json_object(raw_text)
    actions = value.get("actions")
    steps = value.get("steps")
    answer = value.get("answer")
    if not isinstance(actions, list) or not actions:
        raise StructuredOutputError('"actions" must be a non-empty array of strings')
    if not all(isinstance(action, str) for action in actions):
        raise StructuredOutputError('every item in "actions" must be a string')
    if not isinstance(steps, list) or not steps or not all(
        isinstance(step, str) for step in steps
    ):
        raise StructuredOutputError('"steps" must be a non-empty array of strings')
    if not all(STEP_PATTERN.fullmatch(step) for step in steps):
        raise StructuredOutputError('every item in "steps" must be a numbered step')
    if not isinstance(answer, str):
        raise StructuredOutputError('"answer" must be a string')
    if task == "binary" and answer.casefold() not in {"yes", "no", "true", "false"}:
        raise StructuredOutputError("binary answer is not Yes/No or true/false")
    if task == "multi" and answer.upper() not in {"A", "B", "C"}:
        raise StructuredOutputError("multi answer is not A, B, or C")
    return StructuredResponse(actions=tuple(actions), steps=tuple(steps), answer=answer)


def structured_response_format(task: TaskName) -> dict[str, Any]:
    """Return the strict OpenAI JSON-schema constraint used by default."""

    answer_schema: dict[str, Any] = {"type": "string"}
    if task == "binary":
        answer_schema["enum"] = ["true", "false"]
    elif task == "multi":
        answer_schema["enum"] = ["A", "B", "C"]
    elif task == "open":
        pass
    else:
        raise ValueError(f"unsupported task: {task}")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "quants_xqa_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "actions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "steps": {
                        "type": "array",
                        "items": {"type": "string", "pattern": r"^\d+\.\s.+"},
                        "minItems": 1,
                    },
                    "answer": answer_schema,
                },
                "required": ["actions", "steps", "answer"],
                "additionalProperties": False,
            },
        },
    }


async def complete_with_retries(
    client: AsyncCompleter,
    *,
    task: TaskName,
    actions: Sequence[str],
    question: str,
    max_attempts: int = 3,
    retry_delay_seconds: float = 1.0,
    retry_temperature: float = 0.3,
) -> XQAResult:
    """Retry network/validation failures while retaining the final real output.

    If at least one continuation was received, exhausted validation retries return the
    last continuation verbatim with ``parsed=None``.  When the endpoint never returns a
    continuation, the function raises so a resumable job does not publish a fake row.
    """

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")
    messages = build_messages(task=task, actions=actions, question=question)
    raw_attempts: list[str] = []
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            raw_text = await client.complete(
                messages,
                temperature=None if attempt == 1 else retry_temperature,
            )
            raw_attempts.append(raw_text)
            parsed = parse_structured_response(
                raw_text,
                task=task,
            )
            return XQAResult(
                raw_text=raw_text,
                parsed=parsed,
                attempt_count=attempt,
                validation_error=None,
                raw_attempts=tuple(raw_attempts),
            )
        except (StructuredOutputError, CompletionError) as exc:
            last_error = exc
        except Exception as exc:  # transport implementations may expose library errors
            last_error = exc
        if attempt < max_attempts and retry_delay_seconds:
            await asyncio.sleep(retry_delay_seconds)

    if raw_attempts:
        return XQAResult(
            raw_text=raw_attempts[-1],
            parsed=None,
            attempt_count=max_attempts,
            validation_error=str(last_error),
            raw_attempts=tuple(raw_attempts),
        )
    raise CompletionError(
        f"endpoint returned no continuation after {max_attempts} attempts: {last_error}"
    ) from last_error


class OpenAIChatClient:
    """Minimal aiohttp client for an OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2_048,
        timeout_seconds: float = 300.0,
        response_format: dict[str, Any] | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required")
        if not model:
            raise ValueError("model is required")
        self.url = self._completion_url(base_url)
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.response_format = response_format
        self._session: Any = None

    @staticmethod
    def _completion_url(base_url: str) -> str:
        base = base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return base + "/chat/completions"
        return base + "/v1/chat/completions"

    async def __aenter__(self) -> OpenAIChatClient:
        import aiohttp

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        self._session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
    ) -> str:
        if self._session is None:
            raise RuntimeError("OpenAIChatClient must be used as an async context manager")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens,
        }
        if self.response_format is not None:
            payload["response_format"] = self.response_format
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        async with self._session.post(self.url, json=payload, headers=headers) as response:
            body = await response.text()
            if response.status < 200 or response.status >= 300:
                raise CompletionError(f"HTTP {response.status}: {body[:1_000]}")
            try:
                value = json.loads(body)
                content = value["choices"][0]["message"]["content"]
            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
                raise CompletionError("endpoint response lacks choices[0].message.content") from exc
            if not isinstance(content, str):
                raise CompletionError("endpoint continuation is not a string")
            return cast(str, content)


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


def build_judge_prompt(reference: Mapping[str, Any], prediction: str) -> str:
    """Build the open-answer judge prompt used by the QuAnTS evaluation."""

    return JUDGE_TEMPLATE.format(
        scene_description=reference["textual_description"],
        scene_question=reference["question"],
        reference_answer=reference["answer_text"],
        system_answer=prediction,
    )


def parse_judge_response(text: str) -> tuple[str, int]:
    """Parse one constrained judge response."""

    payload = _first_json_object(text)
    rationale = payload.get("BriefRationale")
    rating = payload.get("TotalRating")
    if not isinstance(rationale, str) or len(rationale) > 500:
        raise StructuredOutputError("invalid BriefRationale")
    if type(rating) is not int or rating not in {1, 2, 3}:
        raise StructuredOutputError("invalid TotalRating")
    return rationale, rating


class OpenAIJudgeClient:
    """OpenAI-compatible client with the fixed QuAnTS judge decoding settings."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 600.0,
    ) -> None:
        self.url = OpenAIChatClient._completion_url(base_url)
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._session: Any = None

    async def __aenter__(self) -> OpenAIJudgeClient:
        import aiohttp

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        self._session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def rate(self, prompt: str, *, seed: int) -> tuple[str, int]:
        if self._session is None:
            raise RuntimeError("OpenAIJudgeClient must be used as an async context manager")
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "repetition_penalty": 1.5,
            "max_completion_tokens": 8_192,
            "seed": seed,
            "separate_reasoning": True,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "quants_rating",
                    "schema": JUDGE_SCHEMA,
                    "strict": True,
                },
            },
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        async with self._session.post(self.url, json=payload, headers=headers) as response:
            body = await response.text()
            if response.status < 200 or response.status >= 300:
                raise CompletionError(f"judge HTTP {response.status}: {body[:1_000]}")
            try:
                value = json.loads(body)
                content = value["choices"][0]["message"]["content"]
            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
                raise CompletionError("judge response lacks choices[0].message.content") from exc
            if not isinstance(content, str):
                raise CompletionError("judge continuation is not a string")
            return parse_judge_response(content)
