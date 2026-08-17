from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable

import numpy as np
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .constants import (
    ABLATIONS,
    ARTIFACTS_ROOT,
    DATA_ROOT,
    DATASET_REPO,
    DATASET_REVISION,
    JUDGE_REPO,
    JUDGE_REVISION,
    LLAMA_REPO,
    LLAMA_REVISION,
    PROJECT_ROOT,
    RESULTS_ROOT,
    SEED,
    SPLITS,
    TASKS,
    config_name,
)
from .csvio import read_prediction_csv, validate_prediction_csv
from .data import load_rows
from .judge import read_judge_csv
from .metrics_utils import parse_binary, parse_multi


METRIC_COLUMNS = (
    "config",
    "split",
    "n",
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "invalid_rate",
    "rouge_l_f1",
    "meteor",
    "llm_judge",
)


def atomic_dict_csv(
    path: Path, columns: tuple[str, ...], rows: list[dict[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_files(paths: list[Path]) -> dict[str, dict[str, object]]:
    return {
        str(path.relative_to(PROJECT_ROOT)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(paths)
    }


def write_run_manifest(results_root: Path) -> None:
    source_paths = [
        path
        for root in (
            PROJECT_ROOT / "src",
            PROJECT_ROOT / "slurm",
            PROJECT_ROOT / "tests",
        )
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    source_paths.extend(
        path
        for path in (
            PROJECT_ROOT / "pyproject.toml",
            PROJECT_ROOT / "sglang-requirements.txt",
            PROJECT_ROOT / "uv.lock",
            PROJECT_ROOT / "README.md",
        )
        if path.exists()
    )
    result_paths = [path for path in results_root.glob("*.csv") if path.is_file()]
    adapter_paths = [
        path
        for ablation in ABLATIONS
        for task in TASKS
        for path in (
            ARTIFACTS_ROOT / config_name(ablation, task) / "seed42" / "final_adapter"
        ).rglob("*")
        if path.is_file()
    ]
    package_names = (
        "accelerate",
        "bitsandbytes",
        "datasets",
        "huggingface-hub",
        "nltk",
        "numpy",
        "peft",
        "pyarrow",
        "scikit-learn",
        "torch",
        "transformers",
    )
    packages = {name: importlib.metadata.version(name) for name in package_names}
    data_manifest = DATA_ROOT / "manifest.json"
    payload = {
        "dataset": {
            "repo": DATASET_REPO,
            "revision": DATASET_REVISION,
            "manifest": str(data_manifest.relative_to(PROJECT_ROOT)),
            "manifest_sha256": sha256_file(data_manifest),
        },
        "base_model": {"repo": LLAMA_REPO, "revision": LLAMA_REVISION},
        "judge_model": {"repo": JUDGE_REPO, "revision": JUDGE_REVISION},
        "seed": SEED,
        "python": platform.python_version(),
        "packages": packages,
        "source_files": manifest_files(source_paths),
        "adapter_files": manifest_files(adapter_paths),
        "result_csv_files": manifest_files(result_paths),
    }
    destination = ARTIFACTS_ROOT / "run_manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False, newline="\n"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def aligned_predictions(task: str, split: str, path: Path, data_root: Path):
    references = load_rows(task, split, data_root=data_root)
    validate_prediction_csv(path, split, expected_count=len(references))
    predictions = read_prediction_csv(path)
    reference_map = {
        (int(row["sample_id"]), int(row["question_id"])): row
        for row in (references[index] for index in range(len(references)))
    }
    prediction_map = {
        (int(row["sample_id"]), int(row["question_id"])): str(row["prediction_text"])
        for row in predictions
    }
    if prediction_map.keys() != reference_map.keys():
        raise ValueError(f"prediction/reference keys differ for {task}/{split}")
    keys = sorted(reference_map)
    return [reference_map[key] for key in keys], [prediction_map[key] for key in keys]


def closed_metrics(
    references: list[dict[str, object]],
    predictions: list[str],
    parser: Callable[[str], int | None],
    labels: list[int],
) -> dict[str, float]:
    truth = [int(row["answer"]) for row in references]
    parsed = [parser(text) for text in predictions]
    numeric = [-1 if value is None else value for value in parsed]
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, numeric, labels=labels, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(truth, numeric)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "invalid_rate": float(sum(value is None for value in parsed) / len(parsed)),
    }


def judge_mean(
    path: Path,
    config: str,
    split: str,
    expected_keys: set[tuple[int, int]],
) -> float:
    rows = read_judge_csv(path, config, split)
    keys = {
        (int(row["sample_id"]), int(row["question_id"])) for row in rows
    }
    if keys != expected_keys or len(rows) != len(expected_keys):
        raise ValueError(
            f"judge/reference keys differ for {config}/{split}: "
            f"expected {len(expected_keys)}, found {len(keys)}"
        )
    values = [float(row["normalized_rating"]) for row in rows]
    return float(np.mean(values))


def open_metrics(
    references: list[dict[str, object]],
    predictions: list[str],
    judge_path: Path,
    config: str,
    split: str,
) -> dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge = [
        scorer.score(str(row["answer_text"]), prediction)["rougeL"].fmeasure
        for row, prediction in zip(references, predictions, strict=True)
    ]
    meteor = [
        meteor_score([str(row["answer_text"]).split()], prediction.split())
        for row, prediction in zip(references, predictions, strict=True)
    ]
    expected_keys = {
        (int(row["sample_id"]), int(row["question_id"])) for row in references
    }
    return {
        "rouge_l_f1": float(np.mean(rouge)),
        "meteor": float(np.mean(meteor)),
        "llm_judge": judge_mean(judge_path, config, split, expected_keys),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument(
        "--ablations", nargs="+", choices=ABLATIONS, default=list(ABLATIONS)
    )
    args = parser.parse_args()
    metrics: list[dict[str, object]] = []
    for ablation in args.ablations:
        for task in TASKS:
            config = config_name(ablation, task)
            for split in SPLITS:
                prediction_path = args.results_root / f"{config}_{split}.csv"
                references, predictions = aligned_predictions(
                    task, split, prediction_path, args.data_root
                )
                row: dict[str, object] = {
                    "config": config,
                    "split": split,
                    "n": len(references),
                    "accuracy": "",
                    "precision_macro": "",
                    "recall_macro": "",
                    "f1_macro": "",
                    "invalid_rate": "",
                    "rouge_l_f1": "",
                    "meteor": "",
                    "llm_judge": "",
                }
                if task == "binary":
                    row.update(
                        closed_metrics(references, predictions, parse_binary, [0, 1])
                    )
                elif task == "multi":
                    row.update(
                        closed_metrics(references, predictions, parse_multi, [0, 1, 2])
                    )
                else:
                    row.update(
                        open_metrics(
                            references,
                            predictions,
                            args.results_root / f"judge_{config}_{split}.csv",
                            config,
                            split,
                        )
                    )
                metrics.append(row)
    metrics.sort(key=lambda row: (str(row["config"]), SPLITS.index(str(row["split"]))))
    atomic_dict_csv(
        args.results_root / "metrics_by_config_and_split.csv", METRIC_COLUMNS, metrics
    )
    test_rows = [row for row in metrics if row["split"] == "test"]
    atomic_dict_csv(
        args.results_root / "rebuttal_test_metrics.csv", METRIC_COLUMNS, test_rows
    )
    write_run_manifest(args.results_root)
    print(f"wrote {len(metrics)} metric rows")


if __name__ == "__main__":
    main()
