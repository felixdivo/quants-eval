from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .constants import (
    ABLATIONS,
    ARTIFACTS_ROOT,
    CACHE_ROOT,
    DATA_ROOT,
    LLAMA_REPO,
    LLAMA_REVISION,
    MAX_CONTEXT_TOKENS,
    MODEL_ROOT,
    RESULTS_ROOT,
    SPLITS,
    TASKS,
    config_name,
)
from .csvio import read_prediction_csv, write_prediction_csv
from .data import load_rows, load_ts_prompt_cache
from .prompts import naive_prompt_ids, question_only_prompt
from .resume_metadata import (
    dataset_source_identity,
    directory_files_identity,
    file_metadata,
    prepare_resume_metadata,
    source_files_identity,
    validate_resume_metadata,
)


TOKENIZER_FILENAMES = {
    "added_tokens.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "tokenizer.json",
    "tokenizer_config.json",
}


def distributed_context() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def load_model(
    model_path: Path, adapter_path: Path, local_rank: int
) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path, local_files_only=True, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": local_rank},
    )
    model = PeftModel.from_pretrained(base, adapter_path, local_files_only=True)
    model.eval()
    model.config.use_cache = True
    # The base checkpoint carries sampling-only defaults. They are irrelevant
    # for greedy decoding and otherwise trigger one warning per batch.
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    return model, tokenizer


def left_pad(
    sequences: list[list[int]], pad_token_id: int, device: torch.device
) -> dict[str, torch.Tensor]:
    width = max(len(sequence) for sequence in sequences)
    ids = [
        [pad_token_id] * (width - len(sequence)) + sequence for sequence in sequences
    ]
    masks = [
        [0] * (width - len(sequence)) + [1] * len(sequence) for sequence in sequences
    ]
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(masks, dtype=torch.long, device=device),
    }


def prompt_ids_for_row(
    row: dict[str, Any],
    ablation: str,
    tokenizer: Any,
    ts_cache: Any | None,
    split_start: int,
) -> list[int]:
    if ablation == "question_only":
        return tokenizer(
            question_only_prompt(str(row["question"])), add_special_tokens=True
        )["input_ids"]
    assert ts_cache is not None
    sample_id = int(row["sample_id"])
    cached = ts_cache[sample_id - split_start]
    if int(cached["sample_id"]) != sample_id:
        raise ValueError(f"missing cached prompt for sample_id={sample_id}")
    prompt_ids = list(cached["input_ids"])
    if ablation == "naive":
        return naive_prompt_ids(prompt_ids, str(row["question"]), tokenizer)
    return prompt_ids


def deterministic_generation_units(rows: Any, ablation: str) -> list[list[int]]:
    if ablation != "ts_only":
        return [[index] for index in range(len(rows))]
    groups_by_sample: dict[int, list[int]] = {}
    for index in range(len(rows)):
        sample_id = int(rows[index]["sample_id"])
        groups_by_sample.setdefault(sample_id, []).append(index)
    return list(groups_by_sample.values())


def generation_settings(ablation: str, task: str) -> dict[str, object]:
    return {
        "batch_size": 1 if ablation in ("ts_only", "naive") else 32,
        "checkpoint_every": 16 if ablation in ("ts_only", "naive") else 256,
        "clean_up_tokenization_spaces": False,
        "do_sample": False,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "max_new_tokens": 128 if task == "open" else 32,
        "skip_special_tokens": True,
        "truncation": False,
        "use_cache": True,
    }


def generation_resume_identity(
    ablation: str,
    task: str,
    split: str,
    rank: int,
    world_size: int,
    adapter_path: Path,
    data_root: Path,
    ts_cache_path: Path | None,
) -> dict[str, object]:
    adapter_files = sorted(path for path in adapter_path.rglob("*") if path.is_file())
    tokenizer_files = [
        path
        for path in adapter_files
        if path.name in TOKENIZER_FILENAMES or path.name.startswith("tokenizer.")
    ]
    lora_files = [path for path in adapter_files if path not in tokenizer_files]
    if not lora_files:
        raise FileNotFoundError(f"no adapter files found below {adapter_path}")
    if not tokenizer_files:
        raise FileNotFoundError(f"no tokenizer files found below {adapter_path}")

    source_paths = [
        Path(__file__),
        Path(__file__).with_name("constants.py"),
        Path(__file__).with_name("csvio.py"),
        Path(__file__).with_name("data.py"),
        Path(__file__).with_name("prompts.py"),
        Path(__file__).with_name("resume_metadata.py"),
    ]
    identity: dict[str, object] = {
        "ablation": ablation,
        "task": task,
        "split": split,
        "rank": rank,
        "world_size": world_size,
        "base_model": {"repo": LLAMA_REPO, "revision": LLAMA_REVISION},
        "adapter_files": directory_files_identity(adapter_path, lora_files),
        "tokenizer_files": directory_files_identity(adapter_path, tokenizer_files),
        "dataset": dataset_source_identity(data_root, task, split),
        "prompt_and_source_files": source_files_identity(source_paths),
        "generation_settings": generation_settings(ablation, task),
    }
    if ts_cache_path is not None:
        identity["serialized_prompt_cache"] = file_metadata(
            ts_cache_path, name=f"ts_prompts/{split}.parquet"
        )
    return identity


def generate_split(
    model: Any,
    tokenizer: Any,
    ablation: str,
    task: str,
    split: str,
    rank: int,
    world_size: int,
    data_root: Path,
    cache_root: Path,
    results_root: Path,
    adapter_path: Path,
) -> None:
    rows = load_rows(task, split, data_root=data_root)
    ts_cache = None
    split_start = 0
    ts_cache_path = None
    if ablation in ("ts_only", "naive"):
        ts_cache_path = cache_root / "ts_prompts" / f"{split}.parquet"
        ts_cache = load_ts_prompt_cache(ts_cache_path)
        split_start = int(ts_cache[0]["sample_id"])
    all_units = deterministic_generation_units(rows, ablation)
    local_units = all_units[rank::world_size]
    settings = generation_settings(ablation, task)
    batch_size = int(settings["batch_size"])
    checkpoint_every = int(settings["checkpoint_every"])
    max_new_tokens = int(settings["max_new_tokens"])
    device = next(model.parameters()).device
    parts_dir = results_root / ".parts"
    part_path = parts_dir / f"{config_name(ablation, task)}_{split}.rank{rank}.csv"
    resume_identity = generation_resume_identity(
        ablation,
        task,
        split,
        rank,
        world_size,
        adapter_path,
        data_root,
        ts_cache_path,
    )
    prepare_resume_metadata(part_path, "generation-part", resume_identity)
    assigned_keys = {
        (int(rows[index]["sample_id"]), int(rows[index]["question_id"]))
        for unit in local_units
        for index in unit
    }
    predictions: list[dict[str, object]] = []
    if part_path.exists():
        predictions = read_prediction_csv(part_path)
        existing_keys = {
            (int(row["sample_id"]), int(row["question_id"])) for row in predictions
        }
        if not existing_keys <= assigned_keys:
            raise ValueError(f"stale or invalid rank checkpoint: {part_path}")
    completed_keys = {
        (int(row["sample_id"]), int(row["question_id"])) for row in predictions
    }
    pending_units: list[list[int]] = []
    for unit in local_units:
        unit_keys = {
            (int(rows[index]["sample_id"]), int(rows[index]["question_id"]))
            for index in unit
        }
        completed_in_unit = unit_keys & completed_keys
        if completed_in_unit and completed_in_unit != unit_keys:
            raise ValueError(f"partially completed deterministic unit in {part_path}")
        if not completed_in_unit:
            pending_units.append(unit)

    for offset in range(0, len(pending_units), batch_size):
        units = pending_units[offset : offset + batch_size]
        batch_rows = [rows[unit[0]] for unit in units]
        sequences = [
            prompt_ids_for_row(row, ablation, tokenizer, ts_cache, split_start)
            for row in batch_rows
        ]
        if any(
            len(sequence) + max_new_tokens > MAX_CONTEXT_TOKENS
            for sequence in sequences
        ):
            raise ValueError(
                "generation prompt plus requested continuation exceeds the model "
                "context; "
                "truncation is forbidden"
            )
        encoded = left_pad(sequences, tokenizer.pad_token_id, device)
        input_width = encoded["input_ids"].shape[1]
        with torch.inference_mode():
            output_ids = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        added = 0
        for unit, generated in zip(units, output_ids, strict=True):
            continuation = generated[input_width:].tolist()
            prediction = tokenizer.decode(
                continuation,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for index in unit:
                row = rows[index]
                predictions.append(
                    {
                        "sample_id": int(row["sample_id"]),
                        "question_id": int(row["question_id"]),
                        "prediction_text": prediction,
                    }
                )
                added += 1
        if len(predictions) % checkpoint_every < added:
            write_prediction_csv(part_path, predictions, split)

    write_prediction_csv(part_path, predictions, split)
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        merged: list[dict[str, object]] = []
        for part_rank in range(world_size):
            rank_part_path = (
                parts_dir
                / f"{config_name(ablation, task)}_{split}.rank{part_rank}.csv"
            )
            rank_identity = dict(resume_identity)
            rank_identity["rank"] = part_rank
            validate_resume_metadata(
                rank_part_path, "generation-part", rank_identity
            )
            merged.extend(
                read_prediction_csv(rank_part_path)
            )
        expected = {
            (int(sample_id), int(question_id))
            for sample_id, question_id in zip(
                rows["sample_id"], rows["question_id"], strict=True
            )
        }
        actual = {(int(row["sample_id"]), int(row["question_id"])) for row in merged}
        if actual != expected or len(merged) != len(rows):
            raise ValueError(
                f"prediction key mismatch for {ablation}/{task}/{split}: "
                f"expected {len(expected)}, got {len(actual)}"
            )
        destination = results_root / f"{config_name(ablation, task)}_{split}.csv"
        write_prediction_csv(destination, merged, split)
        print(f"wrote {len(merged)} predictions to {destination}")
    if world_size > 1:
        dist.barrier()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", required=True, choices=ABLATIONS)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--model", type=Path, default=MODEL_ROOT / "llama-3.1-8b")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    args = parser.parse_args()
    rank, world_size, local_rank = distributed_context()
    adapter = args.adapter or (
        ARTIFACTS_ROOT
        / config_name(args.ablation, args.task)
        / "seed42"
        / "final_adapter"
    )
    model, tokenizer = load_model(args.model, adapter, local_rank)
    for split in args.splits:
        generate_split(
            model,
            tokenizer,
            args.ablation,
            args.task,
            split,
            rank,
            world_size,
            args.data_root,
            args.cache_root,
            args.results_root,
            adapter,
        )
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
