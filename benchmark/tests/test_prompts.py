import json
import os
from pathlib import Path

import numpy as np
import pytest
from transformers import AutoTokenizer

from quants_ablation.constants import LLAMA_REVISION, MODEL_ROOT
from quants_ablation.prompts import (
    completion_example,
    naive_prompt,
    naive_prompt_ids,
    question_only_prompt,
    scale_trajectory,
    serialize_trajectory,
    target_text,
    ts_only_prompt,
)


def test_question_only_prompt_is_paper_prompt():
    assert question_only_prompt("What happened?") == (
        "### Question: What happened?\n\n### Answer:"
    )


def test_scaling_and_shape():
    trajectory = np.linspace(-2, 3, 320 * 24 * 3, dtype=np.float32).reshape(320, 24, 3)
    scaled = scale_trajectory(trajectory)
    assert scaled.shape == (320, 24, 3)
    assert scaled.dtype == np.int16
    assert int(scaled.min()) == -999
    assert int(scaled.max()) == 999


def test_constant_trajectory_maps_to_zero():
    trajectory = np.full((320, 24, 3), 4.2, dtype=np.float32)
    assert np.count_nonzero(scale_trajectory(trajectory)) == 0


def test_serialization_retains_three_axes_and_prompt_shape():
    trajectory = np.zeros((320, 24, 3), dtype=np.float32)
    serialized = serialize_trajectory(trajectory)
    decoded = json.loads(serialized)
    assert len(decoded) == 320
    assert len(decoded[0]) == 24
    assert len(decoded[0][0]) == 3
    prompt = ts_only_prompt(trajectory)
    assert "shape [320, 24, 3]" in prompt
    assert "24 human-body joints over 320 time steps" in prompt
    assert prompt.endswith("\n\n### Answer:")


def test_naive_prompt_uses_full_trajectory_shape():
    trajectory = np.zeros((320, 24, 3), dtype=np.float32)
    prompt = naive_prompt(trajectory, "Did the person run?")
    assert "shape [320, 24, 3]" in prompt
    assert "24 human-body joints over 320 time steps" in prompt
    assert prompt.endswith(
        "\n\n### Question: Did the person run?\n\n### Answer:"
    )


def test_invalid_shape_is_rejected():
    with pytest.raises(ValueError, match="expected trajectory shape"):
        scale_trajectory(np.zeros((320, 72), dtype=np.float32))


def test_target_mappings():
    assert target_text({"answer": 0}, "binary") == "No"
    assert target_text({"answer": 1}, "binary") == "Yes"
    assert target_text({"answer": 0}, "multi") == "A"
    assert target_text({"answer": 2}, "multi") == "C"
    assert target_text({"answer_text": "Dansé 🕺"}, "open") == "Dansé 🕺"


def test_only_answer_and_eos_are_supervised():
    example = completion_example([1, 2, 3], [8, 9], 99)
    assert example == {
        "input_ids": [1, 2, 3, 8, 9, 99],
        "attention_mask": [1, 1, 1, 1, 1, 1],
        "labels": [-100, -100, -100, 8, 9, 99],
    }


def _pinned_tokenizer_snapshot() -> Path | None:
    candidates = []
    if override := os.environ.get("QUANTS_TEST_LLAMA_TOKENIZER"):
        candidates.append(Path(override))
    candidates.extend([
        MODEL_ROOT / "llama-3.1-8b",
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--meta-llama--Llama-3.1-8B"
        / "snapshots"
        / LLAMA_REVISION,
    ])
    for candidate in candidates:
        if (candidate / "tokenizer.json").is_file():
            return candidate
    return None


def test_naive_token_splicing_matches_direct_pinned_tokenization():
    snapshot = _pinned_tokenizer_snapshot()
    if snapshot is None:
        pytest.skip("pinned Llama 3.1 tokenizer snapshot is not available locally")
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, revision=LLAMA_REVISION, use_fast=True
    )
    trajectory = np.zeros((320, 24, 3), dtype=np.float32)
    question = "Did the person turn left, then jump? 🧪"
    ts_ids = tokenizer(
        ts_only_prompt(trajectory), add_special_tokens=True
    )["input_ids"]
    direct_ids = tokenizer(
        naive_prompt(trajectory, question), add_special_tokens=True
    )["input_ids"]
    assert naive_prompt_ids(ts_ids, question, tokenizer) == direct_ids
