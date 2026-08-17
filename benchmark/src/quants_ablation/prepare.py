from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from .constants import (
    CACHE_ROOT,
    DATA_ROOT,
    MAX_CONTEXT_TOKENS,
    MODEL_ROOT,
    SPLIT_ID_RANGES,
    SPLITS,
    TASKS,
)
from .data import load_rows, parquet_shards, trajectory_array, trajectory_sha256
from .prompts import (
    ANSWER_SUFFIX,
    NAIVE_INSTRUCTION,
    TS_INSTRUCTION,
    question_only_prompt,
    target_text,
    ts_only_prompt,
)


CACHE_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.int32()),
        pa.field("input_ids", pa.list_(pa.int32())),
        pa.field("trajectory_sha256", pa.string()),
        pa.field("token_count", pa.int32()),
    ]
)


def source_rows_for_split(
    split: str, data_root: Path
) -> dict[int, tuple[np.ndarray, str]]:
    sources: dict[int, tuple[np.ndarray, str]] = {}
    origins: dict[int, str] = {}
    for task in TASKS:
        for path in parquet_shards(task, split, data_root):
            parquet = pq.ParquetFile(path)
            for batch_index, batch in enumerate(
                parquet.iter_batches(
                    batch_size=64,
                    columns=["sample_id", "trajectory"],
                    use_threads=False,
                )
            ):
                sample_ids = batch.column(0).to_pylist()
                trajectories = batch.column(1).to_pylist()
                for row_index, (sample_id, trajectory) in enumerate(
                    zip(sample_ids, trajectories, strict=True)
                ):
                    sample_id = int(sample_id)
                    array = trajectory_array(trajectory)
                    digest = trajectory_sha256(array)
                    origin = f"{task}/{path.name}:batch{batch_index}:row{row_index}"
                    if sample_id in sources and sources[sample_id][1] != digest:
                        raise ValueError(
                            f"trajectory mismatch for sample_id={sample_id}: "
                            f"{origins[sample_id]}={sources[sample_id][1]}, "
                            f"{origin}={digest}"
                        )
                    if sample_id not in sources:
                        sources[sample_id] = (array.copy(), digest)
                        origins[sample_id] = origin
    lower, upper = SPLIT_ID_RANGES[split]
    expected_ids = set(range(lower, upper + 1))
    if sources.keys() != expected_ids:
        missing = sorted(expected_ids - sources.keys())[:10]
        extra = sorted(sources.keys() - expected_ids)[:10]
        raise ValueError(
            f"sample coverage mismatch for {split}: missing={missing}, extra={extra}"
        )
    return sources


def prepare_split(
    split: str, tokenizer: object, data_root: Path, cache_root: Path
) -> dict[str, object]:
    sources = source_rows_for_split(split, data_root)
    destination = cache_root / "ts_prompts" / f"{split}.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    lengths: list[int] = []
    with NamedTemporaryFile(dir=destination.parent, delete=False) as tmp:
        temporary = Path(tmp.name)
    writer = pq.ParquetWriter(temporary, CACHE_SCHEMA, compression="zstd")
    try:
        buffer: list[dict[str, object]] = []
        for sample_id in sorted(sources):
            trajectory, digest = sources[sample_id]
            prompt = ts_only_prompt(trajectory)
            input_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            length = len(input_ids)
            if length >= MAX_CONTEXT_TOKENS:
                raise ValueError(
                    f"sample_id={sample_id} prompt has {length} tokens, exceeding context limit"
                )
            lengths.append(length)
            buffer.append(
                {
                    "sample_id": sample_id,
                    "input_ids": input_ids,
                    "trajectory_sha256": digest,
                    "token_count": length,
                }
            )
            if len(buffer) == 32:
                writer.write_table(pa.Table.from_pylist(buffer, schema=CACHE_SCHEMA))
                buffer.clear()
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=CACHE_SCHEMA))
    finally:
        writer.close()
    os.replace(temporary, destination)
    values = np.asarray(lengths, dtype=np.int64)
    return {
        "split": split,
        "samples": len(lengths),
        "minimum": int(values.min()),
        "p50": int(np.percentile(values, 50)),
        "p95": int(np.percentile(values, 95)),
        "p99": int(np.percentile(values, 99)),
        "maximum": int(values.max()),
        "length_counts": dict(Counter(lengths).most_common(20)),
        "cache": str(destination),
    }


def validate_full_example_lengths(
    tokenizer: object, data_root: Path, cache_root: Path
) -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []
    old_header_length = len(
        tokenizer(
            f"{TS_INSTRUCTION}\n\n### Data:", add_special_tokens=True
        )["input_ids"]
    )
    new_header_length = len(
        tokenizer(
            f"{NAIVE_INSTRUCTION}\n\n### Data:", add_special_tokens=True
        )["input_ids"]
    )
    old_tail_length = len(
        tokenizer(f"]]]{ANSWER_SUFFIX}", add_special_tokens=False)["input_ids"]
    )
    for split in SPLITS:
        cache_table = pq.read_table(
            cache_root / "ts_prompts" / f"{split}.parquet",
            columns=["sample_id", "token_count"],
        )
        ts_lengths = dict(
            zip(
                cache_table["sample_id"].to_pylist(),
                cache_table["token_count"].to_pylist(),
                strict=True,
            )
        )
        for task in TASKS:
            rows = load_rows(task, split, data_root=data_root)
            for ablation in ("question_only", "ts_only", "naive"):
                lengths: list[int] = []
                for index in range(len(rows)):
                    row = rows[index]
                    if ablation == "question_only":
                        prompt_length = len(
                            tokenizer(
                                question_only_prompt(str(row["question"])),
                                add_special_tokens=True,
                            )["input_ids"]
                        )
                    elif ablation == "ts_only":
                        prompt_length = int(ts_lengths[int(row["sample_id"])])
                    else:
                        new_tail_length = len(
                            tokenizer(
                                f"]]]\n\n### Question: {row['question']}{ANSWER_SUFFIX}",
                                add_special_tokens=False,
                            )["input_ids"]
                        )
                        prompt_length = (
                            int(ts_lengths[int(row["sample_id"])])
                            - old_header_length
                            - old_tail_length
                            + new_header_length
                            + new_tail_length
                        )
                    answer_length = (
                        len(
                            tokenizer(
                                " " + target_text(row, task), add_special_tokens=False
                            )["input_ids"]
                        )
                        + 1
                    )
                    length = prompt_length + answer_length
                    if length > MAX_CONTEXT_TOKENS:
                        raise ValueError(
                            f"{ablation}/{task}/{split} sample_id={row['sample_id']} "
                            f"question_id={row['question_id']} has {length} tokens"
                        )
                    lengths.append(length)
                values = np.asarray(lengths, dtype=np.int64)
                reports.append(
                    {
                        "ablation": ablation,
                        "task": task,
                        "split": split,
                        "rows": len(lengths),
                        "minimum": int(values.min()),
                        "p50": int(np.percentile(values, 50)),
                        "p95": int(np.percentile(values, 95)),
                        "p99": int(np.percentile(values, 99)),
                        "maximum": int(values.max()),
                    }
                )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--model", type=Path, default=MODEL_ROOT / "llama-3.1-8b")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, use_fast=True
    )
    report_path = args.cache_root / "context_lengths.json"
    if args.validate_only:
        if not report_path.exists():
            raise FileNotFoundError(
                f"cannot validate existing cache without {report_path}"
            )
        with report_path.open("r", encoding="utf-8") as handle:
            prompt_reports = json.load(handle)["ts_prompt_cache"]
    else:
        prompt_reports = [
            prepare_split(split, tokenizer, args.data_root, args.cache_root)
            for split in SPLITS
        ]
    full_reports = validate_full_example_lengths(
        tokenizer, args.data_root, args.cache_root
    )
    reports = {"ts_prompt_cache": prompt_reports, "full_examples": full_reports}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=report_path.parent, delete=False, newline="\n"
    ) as handle:
        json.dump(reports, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, report_path)
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
