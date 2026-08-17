"""Generate action predictions and xQA answers with resumable CSV outputs."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .constants import ACTION_NAMES, DATASET_ID, DATASET_REVISION
from .csvio import (
    ACTION_PREDICTION_COLUMNS,
    JUDGE_COLUMNS,
    PREDICTION_COLUMNS,
    ResumableCsvWriter,
    file_sha256,
    path_sha256,
    read_action_prediction_csv,
    read_prediction_csv,
)
from .metrics import action_rows_to_sequences, ground_truth_from_rows
from .xqa import (
    PROMPT_VERSION,
    CompletionError,
    OpenAIChatClient,
    OpenAIJudgeClient,
    StructuredOutputError,
    build_judge_prompt,
    complete_with_retries,
    parse_structured_response,
    structured_response_format,
)

TASKS = ("binary", "multi", "open")
SPLITS = ("train", "validation", "val", "test")


def _normal_split(split: str) -> str:
    return "validation" if split == "val" else split


def _scalar_int(value: Any) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def _checkpoint_sha256(path: Path) -> str:
    return path_sha256(path)


def _data_identity(
    *,
    preflight_manifest: Path | None,
    data_root: Path | None,
    dataset_id: str,
    dataset_revision: str,
    task: str,
    split: str,
) -> dict[str, Any]:
    """Select one split's verified content identity."""

    from .data import (
        data_source_fingerprint,
        data_source_manifest,
        normalize_split,
        validate_data_source_location,
    )

    normalized_split = normalize_split(split)
    if preflight_manifest is None:
        source = data_source_manifest(
            task,
            (normalized_split,),
            dataset_id=dataset_id,
            revision=dataset_revision,
            data_root=data_root,
        )
        return {"manifest": source, "fingerprint": data_source_fingerprint(source)}

    report = json.loads(preflight_manifest.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("preflight manifest must contain a JSON object")
    report_fingerprint = report.get("manifest_fingerprint")
    unsigned = {key: value for key, value in report.items() if key != "manifest_fingerprint"}
    if report_fingerprint != data_source_fingerprint(unsigned):
        raise ValueError("preflight manifest fingerprint is invalid")
    if report.get("dataset_id") != dataset_id or report.get("dataset_revision") != dataset_revision:
        raise ValueError("preflight manifest dataset identity differs from this invocation")
    task_reports = report.get("tasks")
    if not isinstance(task_reports, Mapping):
        raise ValueError("preflight manifest has no task reports")
    task_report = task_reports.get(task)
    if not isinstance(task_report, Mapping):
        raise ValueError(f"preflight manifest does not contain task {task!r}")
    source = task_report.get("source_manifest")
    if not isinstance(source, dict):
        raise ValueError(f"preflight manifest has no source identity for task {task!r}")
    if task_report.get("source_fingerprint") != data_source_fingerprint(source):
        raise ValueError(f"preflight source fingerprint is invalid for task {task!r}")
    if (
        source.get("dataset_id") != dataset_id
        or source.get("revision") != dataset_revision
        or source.get("task") != task
    ):
        raise ValueError(f"preflight source identity differs for task {task!r}")
    source_splits = source.get("splits")
    if not isinstance(source_splits, list) or normalized_split not in source_splits:
        raise ValueError(f"preflight manifest does not cover {task}/{normalized_split}")
    selected = {**source, "splits": [normalized_split]}
    if source.get("kind") == "local_parquet":
        shards = source.get("shards")
        if not isinstance(shards, list):
            raise ValueError("preflight local source has no shard list")
        selected["shards"] = [
            shard
            for shard in shards
            if isinstance(shard, Mapping) and shard.get("split") == normalized_split
        ]
        if not selected["shards"]:
            raise ValueError(f"preflight manifest has no shards for {task}/{normalized_split}")
    validate_data_source_location(selected, data_root)
    return {
        "manifest": selected,
        "fingerprint": data_source_fingerprint(selected),
        "preflight_fingerprint": report_fingerprint,
    }


def _action_prediction_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the xLSTMMixer action encoder and write one row per 80-frame segment."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--preflight-manifest", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def predict_actions(args: argparse.Namespace) -> None:
    """Run four-segment inference and publish the separate action CSV."""

    import torch
    from torch.utils.data import DataLoader

    from .data import load_quants_segments
    from .model import load_action_encoder, predict_action_ids

    if args.batch_size < 1:
        raise ValueError("batch-size must be at least one")
    split = _normal_split(args.split)
    dataset = load_quants_segments(
        args.task,
        split,
        dataset_id=args.dataset_id,
        revision=args.dataset_revision,
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        deduplicate_samples=True,
    )
    # Training preserves per-QA-row weighting. Prediction is
    # intentionally deduplicated so this public file has four rows per sample.
    expected_keys: list[tuple[int, int]] = []
    for index in range(len(dataset)):
        item = dataset[index]
        expected_keys.append((_scalar_int(item["sample_id"]), _scalar_int(item["segment_id"])))
    if len(expected_keys) != len(set(expected_keys)):
        raise ValueError("deduplicated prediction dataset still has duplicate sample/segment keys")

    metadata = {
        "kind": "action_predictions",
        "task": args.task,
        "split": split,
        "dataset_id": args.dataset_id,
        "dataset_revision": args.dataset_revision,
        "data_source": _data_identity(
            preflight_manifest=args.preflight_manifest,
            data_root=args.data_root,
            dataset_id=args.dataset_id,
            dataset_revision=args.dataset_revision,
            task=args.task,
            split=split,
        ),
        "checkpoint_sha256": _checkpoint_sha256(args.checkpoint),
        "backend": "xlstm_mixer",
        "action_names": list(ACTION_NAMES),
    }
    writer = ResumableCsvWriter(
        args.output,
        fieldnames=ACTION_PREDICTION_COLUMNS,
        key_fields=("sample_id", "segment_id"),
        metadata=metadata,
        resume=not args.no_resume,
    )
    if writer.finalized:
        writer.finalize(expected_keys=expected_keys)
        return

    completed = {(int(sample), int(segment)) for sample, segment in writer.completed_keys}
    device = torch.device(args.device)
    model = load_action_encoder(args.checkpoint, device=device)
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    with torch.inference_mode():
        for batch in loader:
            sample_ids = [_scalar_int(value) for value in batch["sample_id"]]
            segment_ids = [_scalar_int(value) for value in batch["segment_id"]]
            keys = list(zip(sample_ids, segment_ids, strict=True))
            missing_indices = [index for index, key in enumerate(keys) if key not in completed]
            if not missing_indices:
                continue
            positions = torch.tensor(missing_indices, dtype=torch.long)
            past_values = batch["past_values"].index_select(0, positions).to(device)
            action_ids = predict_action_ids(model, past_values)
            if hasattr(action_ids, "detach"):
                action_ids = action_ids.detach().cpu().tolist()
            action_ids = [int(value) for value in action_ids]
            if len(action_ids) != len(missing_indices):
                raise RuntimeError("model returned a different number of predictions than inputs")
            output_rows = []
            for position, action_id in zip(missing_indices, action_ids, strict=True):
                if not 0 <= action_id < len(ACTION_NAMES):
                    raise RuntimeError(f"model predicted invalid action ID {action_id}")
                sample_id, segment_id = keys[position]
                output_rows.append(
                    {
                        "sample_id": sample_id,
                        "segment_id": segment_id,
                        "predicted_action_id": action_id,
                        "predicted_action_name": ACTION_NAMES[action_id],
                    }
                )
            writer.append_rows(output_rows)
            completed.update(keys[position] for position in missing_indices)
    writer.finalize(expected_keys=expected_keys)


def predict_main() -> None:
    predict_actions(_action_prediction_parser().parse_args())


def _xqa_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Answer QuAnTS questions from ground-truth or predicted action names."
    )
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--mode", choices=("gt", "predicted"), required=True)
    parser.add_argument("--action-predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--preflight-manifest", type=Path)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"))
    parser.add_argument(
        "--model-revision",
        default=os.environ.get("OPENAI_MODEL_REVISION"),
        help="Immutable revision of the weights served by --model (recorded in metadata).",
    )
    parser.add_argument(
        "--deployment-fingerprint",
        default=os.environ.get("OPENAI_DEPLOYMENT_FINGERPRINT"),
        help="Immutable server/container manifest digest recorded in result metadata.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Name of an environment variable containing the endpoint token.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2_048)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=1.0)
    parser.add_argument("--retry-temperature", type=float, default=0.3)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--checkpoint-every", type=int, default=128)
    parser.add_argument(
        "--response-format",
        choices=("json-schema", "json-object", "none"),
        default="json-schema",
        help="Use json-object only when the endpoint lacks strict JSON-schema support.",
    )
    parser.add_argument("--no-resume", action="store_true")
    return parser


def _actions_for_xqa(
    rows: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    action_predictions: Path | None,
) -> tuple[dict[int, list[str]], str | None]:
    expected_sample_ids = {int(row["sample_id"]) for row in rows}
    if mode == "gt":
        ids_by_sample = ground_truth_from_rows(rows)
        return (
            {
                sample_id: [ACTION_NAMES[action_id] for action_id in sequence]
                for sample_id, sequence in ids_by_sample.items()
            },
            None,
        )
    if action_predictions is None:
        raise ValueError("--action-predictions is required in predicted mode")
    action_rows = read_action_prediction_csv(
        action_predictions,
        action_names=ACTION_NAMES,
        expected_sample_ids=expected_sample_ids,
    )
    ids_by_sample = action_rows_to_sequences(action_rows)
    missing = expected_sample_ids - set(ids_by_sample)
    if missing:
        raise ValueError(
            f"action prediction CSV is missing {len(missing)} required samples; "
            f"first missing sample is {min(missing)}"
        )
    return (
        {
            sample_id: [ACTION_NAMES[action_id] for action_id in sequence]
            for sample_id, sequence in ids_by_sample.items()
        },
        file_sha256(action_predictions),
    )


async def generate_xqa(args: argparse.Namespace) -> None:
    from .data import load_question_rows

    if args.concurrency < 1 or args.checkpoint_every < 1:
        raise ValueError("concurrency and checkpoint-every must be at least one")
    if (
        not args.base_url
        or not args.model
        or not args.model_revision
        or not args.deployment_fingerprint
    ):
        raise ValueError(
            "endpoint URL, model ID, weights revision, and deployment fingerprint are "
            "required; set the corresponding OPENAI_* variables or CLI arguments"
        )
    split = _normal_split(args.split)
    rows = list(
        load_question_rows(
            task=args.task,
            split=split,
            dataset_id=args.dataset_id,
            revision=args.dataset_revision,
            data_root=args.data_root,
            cache_dir=args.cache_dir,
        )
    )
    rows.sort(key=lambda row: (int(row["sample_id"]), int(row["question_id"])))
    expected_keys = [(int(row["sample_id"]), int(row["question_id"])) for row in rows]
    if len(expected_keys) != len(set(expected_keys)):
        raise ValueError("QuAnTS split contains duplicate sample_id/question_id keys")
    actions_by_sample, action_csv_sha256 = _actions_for_xqa(
        rows,
        mode=args.mode,
        action_predictions=args.action_predictions,
    )
    metadata = {
        "kind": "xqa_predictions",
        "task": args.task,
        "split": split,
        "mode": args.mode,
        "dataset_id": args.dataset_id,
        "dataset_revision": args.dataset_revision,
        "data_source": _data_identity(
            preflight_manifest=args.preflight_manifest,
            data_root=args.data_root,
            dataset_id=args.dataset_id,
            dataset_revision=args.dataset_revision,
            task=args.task,
            split=split,
        ),
        "action_prediction_sha256": action_csv_sha256,
        "prompt_version": PROMPT_VERSION,
        "endpoint": args.base_url,
        "model": args.model,
        "model_revision": args.model_revision,
        "deployment_fingerprint": args.deployment_fingerprint,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "response_format": args.response_format,
        "max_attempts": args.max_attempts,
        "retry_temperature": args.retry_temperature,
        "retry_delay_seconds": args.retry_delay_seconds,
        "timeout_seconds": args.timeout_seconds,
        "concurrency": args.concurrency,
        "checkpoint_every": args.checkpoint_every,
        "api_key_env": args.api_key_env,
    }
    writer = ResumableCsvWriter(
        args.output,
        fieldnames=PREDICTION_COLUMNS,
        key_fields=("sample_id", "question_id"),
        metadata=metadata,
        resume=not args.no_resume,
    )
    if writer.finalized:
        writer.finalize(expected_keys=expected_keys)
        return
    completed = {(int(sample), int(question)) for sample, question in writer.completed_keys}
    pending = [
        row
        for row in rows
        if (int(row["sample_id"]), int(row["question_id"])) not in completed
    ]
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    semaphore = asyncio.Semaphore(args.concurrency)
    invalid_count = 0

    if args.response_format == "json-schema":
        response_format = structured_response_format(args.task)
    elif args.response_format == "json-object":
        response_format = {"type": "json_object"}
    else:
        response_format = None
    async with OpenAIChatClient(
        base_url=args.base_url,
        model=args.model,
        api_key=api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
        response_format=response_format,
    ) as client:

        async def generate_one(row: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
            sample_id = int(row["sample_id"])
            question_id = int(row["question_id"])
            question = row["question"]
            if not isinstance(question, str):
                raise ValueError(f"question {(sample_id, question_id)} is not text")
            async with semaphore:
                result = await complete_with_retries(
                    client,
                    task=args.task,
                    actions=actions_by_sample[sample_id],
                    question=question,
                    max_attempts=args.max_attempts,
                    retry_delay_seconds=args.retry_delay_seconds,
                    retry_temperature=args.retry_temperature,
                )
            return (
                {
                    "sample_id": sample_id,
                    "question_id": question_id,
                    "prediction_text": result.raw_text,
                },
                result.valid,
            )

        for start in range(0, len(pending), args.checkpoint_every):
            batch = pending[start : start + args.checkpoint_every]
            results = await asyncio.gather(*(generate_one(row) for row in batch))
            writer.append_rows(row for row, _ in results)
            invalid_count += sum(not valid for _, valid in results)
    writer.finalize(expected_keys=expected_keys)
    print(
        f"wrote {len(expected_keys)} xQA predictions to {args.output}; "
        f"invalid structured outputs retained in this run: {invalid_count}"
    )


def xqa_main() -> None:
    asyncio.run(generate_xqa(_xqa_parser().parse_args()))


def _judge_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Judge one canonical open xQA CSV.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--preflight-manifest", type=Path)
    parser.add_argument("--base-url", default=os.environ.get("JUDGE_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("JUDGE_MODEL"))
    parser.add_argument("--model-revision", default=os.environ.get("JUDGE_MODEL_REVISION"))
    parser.add_argument(
        "--deployment-fingerprint",
        default=os.environ.get("JUDGE_DEPLOYMENT_FINGERPRINT"),
    )
    parser.add_argument("--api-key-env", default="JUDGE_API_KEY")
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=1.0)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--checkpoint-every", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-resume", action="store_true")
    return parser


async def judge_xqa(args: argparse.Namespace) -> None:
    from .data import load_question_rows

    if not all(
        (args.base_url, args.model, args.model_revision, args.deployment_fingerprint)
    ):
        raise ValueError(
            "judge endpoint, model ID, weights revision, and deployment fingerprint "
            "are required"
        )
    if args.max_attempts < 1 or args.concurrency < 1 or args.checkpoint_every < 1:
        raise ValueError("judge attempts, concurrency, and checkpoint size must be positive")
    split = _normal_split(args.split)
    references = list(
        load_question_rows(
            task="open",
            split=split,
            dataset_id=args.dataset_id,
            revision=args.dataset_revision,
            data_root=args.data_root,
            cache_dir=args.cache_dir,
        )
    )
    reference_map = {
        (int(row["sample_id"]), int(row["question_id"])): row for row in references
    }
    if len(reference_map) != len(references):
        raise ValueError("open references contain duplicate keys")
    prediction_rows = read_prediction_csv(args.predictions)
    prediction_map = {
        (int(row["sample_id"]), int(row["question_id"])): str(row["prediction_text"])
        for row in prediction_rows
    }
    if prediction_map.keys() != reference_map.keys():
        raise ValueError("open prediction/reference keys differ")
    expected_keys = sorted(reference_map)
    metadata = {
        "kind": "xqa_open_judge",
        "config": args.config,
        "split": split,
        "prediction_sha256": file_sha256(args.predictions),
        "data_source": _data_identity(
            preflight_manifest=args.preflight_manifest,
            data_root=args.data_root,
            dataset_id=args.dataset_id,
            dataset_revision=args.dataset_revision,
            task="open",
            split=split,
        ),
        "judge_model": args.model,
        "judge_model_revision": args.model_revision,
        "judge_deployment_fingerprint": args.deployment_fingerprint,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "repetition_penalty": 1.5,
        "max_completion_tokens": 8_192,
        "seed": args.seed,
        "timeout_seconds": args.timeout_seconds,
        "max_attempts": args.max_attempts,
        "retry_delay_seconds": args.retry_delay_seconds,
        "concurrency": args.concurrency,
        "checkpoint_every": args.checkpoint_every,
        "api_key_env": args.api_key_env,
    }
    writer = ResumableCsvWriter(
        args.output,
        fieldnames=JUDGE_COLUMNS,
        key_fields=("sample_id", "question_id"),
        metadata=metadata,
        resume=not args.no_resume,
    )
    if writer.finalized:
        writer.finalize(expected_keys=expected_keys)
        return
    completed = {(int(sample), int(question)) for sample, question in writer.completed_keys}
    pending = [key for key in expected_keys if key not in completed]
    semaphore = asyncio.Semaphore(args.concurrency)
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None

    async with OpenAIJudgeClient(
        base_url=args.base_url,
        model=args.model,
        api_key=api_key,
        timeout_seconds=args.timeout_seconds,
    ) as client:

        async def judge_one(key: tuple[int, int]) -> dict[str, Any]:
            raw_prediction = prediction_map[key]
            try:
                system_answer = parse_structured_response(
                    raw_prediction,
                    task="open",
                ).answer
            except StructuredOutputError:
                system_answer = raw_prediction
            prompt = build_judge_prompt(reference_map[key], system_answer)
            last_error: Exception | None = None
            async with semaphore:
                for attempt in range(args.max_attempts):
                    try:
                        rationale, rating = await client.rate(
                            prompt,
                            seed=args.seed + key[0] * 5 + key[1],
                        )
                        return {
                            "config": args.config,
                            "split": split,
                            "sample_id": key[0],
                            "question_id": key[1],
                            "brief_rationale": rationale,
                            "total_rating": rating,
                            "normalized_rating": (rating - 1) / 2,
                        }
                    except Exception as exc:
                        last_error = exc
                        if attempt + 1 < args.max_attempts:
                            await asyncio.sleep(args.retry_delay_seconds * (2**attempt))
            raise CompletionError(
                f"judge failed for key {key} after {args.max_attempts} attempts: {last_error}"
            ) from last_error

        for start in range(0, len(pending), args.checkpoint_every):
            chunk = pending[start : start + args.checkpoint_every]
            writer.append_rows(await asyncio.gather(*(judge_one(key) for key in chunk)))
    writer.finalize(expected_keys=expected_keys)


def judge_main() -> None:
    asyncio.run(judge_xqa(_judge_parser().parse_args()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("actions", "xqa", "judge"))
    args, remaining = parser.parse_known_args()
    if args.command == "actions":
        predict_actions(_action_prediction_parser().parse_args(remaining))
    elif args.command == "xqa":
        asyncio.run(generate_xqa(_xqa_parser().parse_args(remaining)))
    else:
        asyncio.run(judge_xqa(_judge_parser().parse_args(remaining)))


if __name__ == "__main__":
    main()
