from __future__ import annotations

import argparse
import asyncio
import csv
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import aiohttp

from .constants import (
    ABLATIONS,
    DATA_ROOT,
    JUDGE_REPO,
    JUDGE_REVISION,
    PROJECT_ROOT,
    RESULTS_ROOT,
    SEED,
    SPLITS,
    config_name,
)
from .csvio import read_prediction_csv
from .data import load_rows
from .judge_utils import JUDGE_SCHEMA, JUDGE_TEMPLATE, judge_prompt, parse_response
from .resume_metadata import (
    dataset_source_identity,
    file_identity,
    prepare_resume_metadata,
    source_files_identity,
)


JUDGE_COLUMNS = (
    "config",
    "split",
    "sample_id",
    "question_id",
    "brief_rationale",
    "total_rating",
    "normalized_rating",
)

JUDGE_MODEL_ALIAS = "qwen3-judge"
JUDGE_SAMPLING_SETTINGS = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "repetition_penalty": 1.5,
    "max_completion_tokens": 8192,
    "separate_reasoning": True,
}


def judge_resume_identity(
    config: str,
    split: str,
    prediction_path: Path,
    data_root: Path,
) -> dict[str, object]:
    source_paths = [
        Path(__file__),
        Path(__file__).with_name("constants.py"),
        Path(__file__).with_name("csvio.py"),
        Path(__file__).with_name("data.py"),
        Path(__file__).with_name("judge_utils.py"),
        Path(__file__).with_name("resume_metadata.py"),
        PROJECT_ROOT / "sglang-requirements.txt",
    ]
    return {
        "config": config,
        "split": split,
        "prediction_csv": file_identity(prediction_path, name=prediction_path.name),
        "reference_data": dataset_source_identity(data_root, "open", split),
        "judge_model": {
            "repo": JUDGE_REPO,
            "revision": JUDGE_REVISION,
            "served_model_name": JUDGE_MODEL_ALIAS,
        },
        "prompt_template": JUDGE_TEMPLATE,
        "response_schema": JUDGE_SCHEMA,
        "sampling_settings": JUDGE_SAMPLING_SETTINGS,
        "request_seed": "SEED + sample_id * 5 + question_id",
        "seed": SEED,
        "prompt_and_source_files": source_files_identity(source_paths),
    }


async def request_rating(
    session: aiohttp.ClientSession,
    endpoint: str,
    prompt: str,
    request_seed: int,
    semaphore: asyncio.Semaphore,
) -> tuple[str, int]:
    payload = {
        "model": JUDGE_MODEL_ALIAS,
        "messages": [{"role": "user", "content": prompt}],
        **JUDGE_SAMPLING_SETTINGS,
        "seed": request_seed,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "quants_rating",
                "schema": JUDGE_SCHEMA,
                "strict": True,
            },
        },
    }
    last_error: Exception | None = None
    async with semaphore:
        for attempt in range(3):
            try:
                async with session.post(endpoint, json=payload) as response:
                    response.raise_for_status()
                    body = await response.json()
                generated = body["choices"][0]["message"]["content"]
                return parse_response(str(generated))
            except Exception as error:  # retry transient server or parsing failures
                last_error = error
                await asyncio.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def write_judge_csv(path: Path, rows: list[dict[str, object]]) -> None:
    rows.sort(key=lambda row: (int(row["sample_id"]), int(row["question_id"])))
    keys = [(int(row["sample_id"]), int(row["question_id"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate judge keys in {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(JUDGE_COLUMNS), lineterminator="\r\n"
        )
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def read_judge_csv(path: Path, config: str, split: str) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != JUDGE_COLUMNS:
            raise ValueError(f"invalid judge columns in {path}: {reader.fieldnames}")
        rows: list[dict[str, object]] = []
        for row in reader:
            if row["config"] != config or row["split"] != split:
                raise ValueError(f"stale judge checkpoint row in {path}")
            rating = int(row["total_rating"])
            normalized = float(row["normalized_rating"])
            if rating not in (1, 2, 3) or normalized != (rating - 1) / 2:
                raise ValueError(f"invalid judge rating in {path}")
            rows.append(
                {
                    "config": config,
                    "split": split,
                    "sample_id": int(row["sample_id"]),
                    "question_id": int(row["question_id"]),
                    "brief_rationale": row["brief_rationale"],
                    "total_rating": rating,
                    "normalized_rating": normalized,
                }
            )
    keys = [(int(row["sample_id"]), int(row["question_id"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate judge keys in {path}")
    return rows


async def judge_file(
    ablation: str,
    split: str,
    endpoint: str,
    concurrency: int,
    data_root: Path,
    results_root: Path,
) -> None:
    config = config_name(ablation, "open")
    prediction_path = results_root / f"{config}_{split}.csv"
    predictions = read_prediction_csv(prediction_path)
    prediction_map = {
        (int(row["sample_id"]), int(row["question_id"])): str(row["prediction_text"])
        for row in predictions
    }
    references = load_rows("open", split, data_root=data_root, include_description=True)
    reference_keys = {
        (int(references[index]["sample_id"]), int(references[index]["question_id"]))
        for index in range(len(references))
    }
    if prediction_map.keys() != reference_keys:
        raise ValueError(f"prediction/reference keys differ for {config}/{split}")

    destination = results_root / f"judge_{config}_{split}.csv"
    resume_identity = judge_resume_identity(
        config, split, prediction_path, data_root
    )
    prepare_resume_metadata(destination, "judge-checkpoint", resume_identity)
    rows = read_judge_csv(destination, config, split) if destination.exists() else []
    completed_keys = {(int(row["sample_id"]), int(row["question_id"])) for row in rows}
    if not completed_keys <= reference_keys:
        raise ValueError(f"stale judge checkpoint keys in {destination}")

    timeout = aiohttp.ClientTimeout(total=None, connect=120, sock_read=None)
    semaphore = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        pending = []
        for index in range(len(references)):
            reference = references[index]
            key = (int(reference["sample_id"]), int(reference["question_id"]))
            if key in completed_keys:
                continue
            prediction = prediction_map[key]
            coroutine = request_rating(
                session,
                endpoint,
                judge_prompt(reference, prediction),
                SEED + key[0] * 5 + key[1],
                semaphore,
            )
            pending.append((key, coroutine))
        for start in range(0, len(pending), 256):
            chunk = pending[start : start + 256]
            ratings = await asyncio.gather(*(coroutine for _, coroutine in chunk))
            for (sample_id, question_id), (rationale, rating) in zip(
                (key for key, _ in chunk), ratings, strict=True
            ):
                rows.append(
                    {
                        "config": config,
                        "split": split,
                        "sample_id": sample_id,
                        "question_id": question_id,
                        "brief_rationale": rationale,
                        "total_rating": rating,
                        "normalized_rating": (rating - 1) / 2,
                    }
                )
            write_judge_csv(destination, rows)
    final_keys = {(int(row["sample_id"]), int(row["question_id"])) for row in rows}
    if final_keys != reference_keys:
        raise ValueError(
            f"judge key mismatch for {config}/{split}: "
            f"expected {len(reference_keys)}, got {len(final_keys)}"
        )
    write_judge_csv(destination, rows)
    print(f"wrote {len(rows)} judge ratings to {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint", default="http://127.0.0.1:30000/v1/chat/completions"
    )
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument(
        "--ablations", nargs="+", choices=ABLATIONS, default=list(ABLATIONS)
    )
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    args = parser.parse_args()

    async def run_all() -> None:
        for ablation in args.ablations:
            for split in args.splits:
                await judge_file(
                    ablation,
                    split,
                    args.endpoint,
                    args.concurrency,
                    args.data_root,
                    args.results_root,
                )

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
