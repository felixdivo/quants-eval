"""Post-hoc action-encoder metrics for four-segment QuAnTS trajectories."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .constants import ACTION_NAMES
from .csvio import (
    JUDGE_COLUMNS,
    read_action_prediction_csv,
    read_csv_rows,
    read_prediction_csv,
    write_csv_atomic,
    write_json_atomic,
)
from .xqa import StructuredOutputError, parse_structured_response

XQA_METRIC_COLUMNS = (
    "config",
    "split",
    "task",
    "num_examples",
    "valid_examples",
    "invalid_examples",
    "invalid_output_rate",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "rouge_l_f1",
    "meteor",
    "llm_judge",
)
XQA_QUESTION_TYPE_METRIC_COLUMNS = (
    "config",
    "split",
    "task",
    "question_type",
    *XQA_METRIC_COLUMNS[3:],
)
XQA_PARSED_COLUMNS = (
    "config",
    "split",
    "task",
    "sample_id",
    "question_id",
    "question_type",
    "question",
    "valid_output",
    "parsed_answer",
    "prediction_label",
    "reference_label",
    "reference_text",
)


def evaluate_action_sequences(
    ground_truth: Mapping[int, Sequence[int]],
    predictions: Mapping[int, Sequence[int]],
    *,
    action_names: Sequence[str] = ACTION_NAMES,
) -> dict[str, Any]:
    """Compute segment, sequence, position, and per-class classification metrics."""

    expected_segments = 4
    for source_name, values in (("ground truth", ground_truth), ("predictions", predictions)):
        for sample_id, sequence in values.items():
            if len(sequence) != expected_segments:
                raise ValueError(
                    f"{source_name} sample {sample_id} has {len(sequence)} segments; expected 4"
                )
            for action_id in sequence:
                if not 0 <= int(action_id) < len(action_names):
                    raise ValueError(
                        f"{source_name} sample {sample_id} has invalid action ID {action_id}"
                    )

    truth_ids = set(ground_truth)
    prediction_ids = set(predictions)
    matched_ids = sorted(truth_ids & prediction_ids)
    missing_ids = sorted(truth_ids - prediction_ids)
    unexpected_ids = sorted(prediction_ids - truth_ids)

    position_correct = [0] * expected_segments
    truth_count = [0] * len(action_names)
    predicted_count = [0] * len(action_names)
    true_positive = [0] * len(action_names)
    segment_correct = 0
    exact_sequences = 0

    for sample_id in matched_ids:
        target = [int(value) for value in ground_truth[sample_id]]
        predicted = [int(value) for value in predictions[sample_id]]
        exact = True
        for position, (predicted_id, target_id) in enumerate(zip(predicted, target, strict=True)):
            is_correct = predicted_id == target_id
            segment_correct += int(is_correct)
            position_correct[position] += int(is_correct)
            truth_count[target_id] += 1
            predicted_count[predicted_id] += 1
            true_positive[target_id] += int(is_correct)
            exact &= is_correct
        exact_sequences += int(exact)

    segment_total = len(matched_ids) * expected_segments
    by_position = [
        {
            "segment_id": position,
            "correct": correct,
            "total": len(matched_ids),
            "accuracy": correct / len(matched_ids) if matched_ids else None,
        }
        for position, correct in enumerate(position_correct)
    ]
    per_class: list[dict[str, Any]] = []
    class_f1_values: list[float] = []
    for action_id, action_name in enumerate(action_names):
        support = truth_count[action_id]
        predicted = predicted_count[action_id]
        tp = true_positive[action_id]
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if support:
            class_f1_values.append(f1)
        per_class.append(
            {
                "action_id": action_id,
                "action_name": action_name,
                "true_positive": tp,
                "predicted_count": predicted,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    summary = {
        "ground_truth_samples": len(truth_ids),
        "prediction_samples": len(prediction_ids),
        "matched_samples": len(matched_ids),
        "missing_samples": len(missing_ids),
        "unexpected_samples": len(unexpected_ids),
        "segment_correct": segment_correct,
        "segment_total": segment_total,
        "segment_accuracy": segment_correct / segment_total if segment_total else None,
        "exact_sequences": exact_sequences,
        "exact_sequence_accuracy": (
            exact_sequences / len(matched_ids) if matched_ids else None
        ),
        "macro_f1_present_classes": (
            sum(class_f1_values) / len(class_f1_values) if class_f1_values else None
        ),
    }
    return {
        "summary": summary,
        "by_position": by_position,
        "per_class": per_class,
        "missing_sample_ids": missing_ids,
        "unexpected_sample_ids": unexpected_ids,
    }


def action_rows_to_sequences(rows: Sequence[Mapping[str, Any]]) -> dict[int, list[int]]:
    sequences: dict[int, list[int | None]] = {}
    for row in rows:
        sample_id = int(row["sample_id"])
        segment_id = int(row["segment_id"])
        slots = sequences.setdefault(sample_id, [None, None, None, None])
        if slots[segment_id] is not None:
            raise ValueError(f"duplicate action row {(sample_id, segment_id)}")
        slots[segment_id] = int(row["predicted_action_id"])
    result: dict[int, list[int]] = {}
    for sample_id, slots in sequences.items():
        if any(value is None for value in slots):
            raise ValueError(f"sample {sample_id} does not contain four segments")
        result[sample_id] = [int(value) for value in slots if value is not None]
    return result


def ground_truth_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[int, list[int]]:
    """Deduplicate QuAnTS QA rows into one ground-truth action sequence per sample."""

    action_to_id = {name: index for index, name in enumerate(ACTION_NAMES)}
    result: dict[int, list[int]] = {}
    for row in rows:
        sample_id = int(row["sample_id"])
        action_sequence = row.get("action_sequence")
        if not isinstance(action_sequence, Mapping):
            raise ValueError(f"sample {sample_id} lacks action_sequence")
        names = action_sequence.get("action")
        if not isinstance(names, list | tuple) or len(names) != 4:
            raise ValueError(f"sample {sample_id} does not have four action labels")
        try:
            sequence = [action_to_id[str(name)] for name in names]
        except KeyError as exc:
            raise ValueError(f"sample {sample_id} has unknown action {exc.args[0]!r}") from exc
        previous = result.setdefault(sample_id, sequence)
        if previous != sequence:
            raise ValueError(f"sample {sample_id} has inconsistent action labels across QA rows")
    return result


def evaluate_qa_row_weighted(
    rows: Sequence[Mapping[str, Any]],
    predictions: Mapping[int, Sequence[int]],
) -> dict[str, Any]:
    """Evaluate once per QA row, matching the source benchmark's weighting."""

    action_to_id = {name: index for index, name in enumerate(ACTION_NAMES)}
    records: list[tuple[int, int, str, list[int]]] = []
    seen_keys: set[tuple[int, int]] = set()
    for row in rows:
        sample_id = int(row["sample_id"])
        question_id = int(row["question_id"])
        key = (sample_id, question_id)
        if key in seen_keys:
            raise ValueError(f"duplicate QA key {key}")
        seen_keys.add(key)
        action_sequence = row.get("action_sequence")
        if not isinstance(action_sequence, Mapping):
            raise ValueError(f"QA row {key} lacks action_sequence")
        names = action_sequence.get("action")
        if not isinstance(names, list | tuple) or len(names) != 4:
            raise ValueError(f"QA row {key} does not have four action labels")
        try:
            target = [action_to_id[str(name)] for name in names]
        except KeyError as exc:
            raise ValueError(f"QA row {key} has unknown action {exc.args[0]!r}") from exc
        records.append((sample_id, question_id, str(row.get("question_type", "unknown")), target))

    def evaluate_records(
        selected: Sequence[tuple[int, int, str, list[int]]],
        *,
        report_predictions_outside_scope: bool,
    ) -> dict[str, Any]:
        synthetic_truth = {index: record[3] for index, record in enumerate(selected)}
        synthetic_predictions = {
            index: predictions[record[0]]
            for index, record in enumerate(selected)
            if record[0] in predictions
        }
        evaluated = evaluate_action_sequences(synthetic_truth, synthetic_predictions)
        base_summary = evaluated["summary"]
        missing_keys = [
            [sample_id, question_id]
            for sample_id, question_id, _, _ in selected
            if sample_id not in predictions
        ]
        sample_ids = {record[0] for record in selected}
        summary = {
            "ground_truth_qa_rows": len(selected),
            "matched_qa_rows": int(base_summary["matched_samples"]),
            "missing_qa_rows": len(missing_keys),
            "unique_ground_truth_samples": len(sample_ids),
            "unexpected_prediction_samples": (
                len(set(predictions) - sample_ids) if report_predictions_outside_scope else 0
            ),
            "segment_correct": int(base_summary["segment_correct"]),
            "segment_total": int(base_summary["segment_total"]),
            "segment_accuracy": base_summary["segment_accuracy"],
            "exact_sequences": int(base_summary["exact_sequences"]),
            "exact_sequence_accuracy": base_summary["exact_sequence_accuracy"],
            "macro_f1_present_classes": base_summary["macro_f1_present_classes"],
        }
        return {
            "summary": summary,
            "by_position": evaluated["by_position"],
            "per_class": evaluated["per_class"],
            "missing_qa_keys": missing_keys,
        }

    overall = evaluate_records(records, report_predictions_outside_scope=True)
    grouped_rows = []
    for question_type in sorted({record[2] for record in records}):
        group = evaluate_records(
            [record for record in records if record[2] == question_type],
            report_predictions_outside_scope=False,
        )
        grouped_rows.append({"question_type": question_type, **group["summary"]})
    overall["by_question_type"] = grouped_rows
    return overall


def write_metric_outputs(
    output_dir: str | Path,
    unique_sample_result: Mapping[str, Any],
    *,
    qa_row_weighted_result: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write clearly labeled unique-sample and QA-row-weighted metric views."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": dict(metadata or {}),
        "views": {
            "unique_sample": dict(unique_sample_result),
            "qa_row_weighted": dict(qa_row_weighted_result),
        },
    }

    write_json_atomic(destination / "action_metrics.json", payload)

    for view_name, result in (
        ("unique_sample", unique_sample_result),
        ("qa_row_weighted", qa_row_weighted_result),
    ):
        summary = dict(result["summary"])
        write_csv_atomic(
            destination / f"action_metrics_{view_name}_summary.csv",
            [summary],
            fieldnames=tuple(summary),
        )
        by_position = list(result["by_position"])
        write_csv_atomic(
            destination / f"action_metrics_{view_name}_by_position.csv",
            by_position,
            fieldnames=tuple(by_position[0]) if by_position else ("segment_id",),
        )
        per_class = list(result["per_class"])
        write_csv_atomic(
            destination / f"action_metrics_{view_name}_by_class.csv",
            per_class,
            fieldnames=tuple(per_class[0]) if per_class else ("action_id",),
        )
    question_types = list(qa_row_weighted_result.get("by_question_type", []))
    write_csv_atomic(
        destination / "action_metrics_qa_row_weighted_by_question_type.csv",
        question_types,
        fieldnames=(
            tuple(question_types[0])
            if question_types
            else ("question_type", "ground_truth_qa_rows")
        ),
    )


def _closed_metrics(
    references: Sequence[int],
    predictions: Sequence[int | None],
    *,
    num_classes: int,
) -> dict[str, float]:
    if len(references) != len(predictions):
        raise ValueError("reference/prediction lengths differ")
    total = len(references)
    correct = sum(
        prediction is not None and prediction == reference
        for reference, prediction in zip(references, predictions, strict=True)
    )
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for class_id in range(num_classes):
        tp = sum(
            prediction == class_id and reference == class_id
            for reference, prediction in zip(references, predictions, strict=True)
        )
        fp = sum(
            prediction == class_id and reference != class_id
            for reference, prediction in zip(references, predictions, strict=True)
        )
        fn = sum(
            reference == class_id and prediction != class_id
            for reference, prediction in zip(references, predictions, strict=True)
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return {
        "accuracy": correct / total if total else 0.0,
        "macro_precision": sum(precisions) / num_classes,
        "macro_recall": sum(recalls) / num_classes,
        "macro_f1": sum(f1s) / num_classes,
    }


def _answer_label(answer: str, task: str) -> int:
    if task == "binary":
        normalized = answer.casefold()
        if normalized in {"true", "yes"}:
            return 1
        if normalized in {"false", "no"}:
            return 0
        raise ValueError(f"invalid binary answer {answer!r}")
    if task == "multi":
        normalized = answer.upper()
        if normalized in {"A", "B", "C"}:
            return ord(normalized) - ord("A")
        raise ValueError(f"invalid multi answer {answer!r}")
    raise ValueError(f"task {task!r} has no closed label mapping")


def evaluate_xqa_predictions(
    reference_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    task: str,
    config: str,
    split: str,
    judge_rows: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Parse canonical raw xQA output and compute task metrics post-hoc."""

    if task not in {"binary", "multi", "open"}:
        raise ValueError(f"unsupported task {task!r}")
    references: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in reference_rows:
        key = (int(row["sample_id"]), int(row["question_id"]))
        if key in references:
            raise ValueError(f"duplicate reference key {key}")
        references[key] = row
    predictions: dict[tuple[int, int], str] = {}
    for row in prediction_rows:
        key = (int(row["sample_id"]), int(row["question_id"]))
        if key in predictions:
            raise ValueError(f"duplicate prediction key {key}")
        raw_text = row["prediction_text"]
        if not isinstance(raw_text, str):
            raise ValueError(f"prediction {key} is not text")
        predictions[key] = raw_text
    if references.keys() != predictions.keys():
        missing = references.keys() - predictions.keys()
        unexpected = predictions.keys() - references.keys()
        raise ValueError(
            f"xQA keys differ: {len(missing)} missing and {len(unexpected)} unexpected"
        )

    judge_values: dict[tuple[int, int], float] | None = None
    if judge_rows is not None:
        judge_values = {}
        for row in judge_rows:
            if row["config"] != config or row["split"] != split:
                raise ValueError("judge row config/split does not match evaluated predictions")
            key = (int(row["sample_id"]), int(row["question_id"]))
            if key in judge_values:
                raise ValueError(f"duplicate judge key {key}")
            rating = int(row["total_rating"])
            normalized = float(row["normalized_rating"])
            if rating not in {1, 2, 3} or normalized != (rating - 1) / 2:
                raise ValueError(f"invalid judge rating for key {key}")
            judge_values[key] = normalized
        if judge_values.keys() != references.keys():
            missing = references.keys() - judge_values.keys()
            unexpected = judge_values.keys() - references.keys()
            raise ValueError(
                f"judge keys differ: {len(missing)} missing and {len(unexpected)} unexpected"
            )

    parsed_rows: list[dict[str, Any]] = []
    reference_labels: list[int] = []
    prediction_labels: list[int | None] = []
    rouge_scores: list[float] = []
    meteor_scores: list[float] = []
    invalid_examples = 0
    if task == "open":
        from nltk.translate.meteor_score import single_meteor_score
        from rouge_score.rouge_scorer import RougeScorer

        rouge_scorer = RougeScorer(["rougeL"], use_stemmer=True)

    for key in sorted(references):
        reference = references[key]
        raw_text = predictions[key]
        valid = True
        answer = ""
        predicted_label: int | None = None
        try:
            parsed = parse_structured_response(raw_text, task=task)  # type: ignore[arg-type]
            answer = parsed.answer
            if task != "open":
                predicted_label = _answer_label(answer, task)
        except (StructuredOutputError, ValueError):
            valid = False
            invalid_examples += 1

        reference_text = str(reference.get("answer_text", ""))
        reference_label: int | None = None
        if task == "open":
            if valid:
                rouge_scores.append(rouge_scorer.score(reference_text, answer)["rougeL"].fmeasure)
                try:
                    meteor_scores.append(
                        single_meteor_score(
                            reference_text.split(),
                            answer.split(),
                        )
                    )
                except LookupError as exc:
                    raise RuntimeError(
                        "NLTK WordNet data is required for METEOR; run the setup job or "
                        "download wordnet and omw-1.4 into NLTK_DATA"
                    ) from exc
            else:
                rouge_scores.append(0.0)
                meteor_scores.append(0.0)
        else:
            reference_label = int(reference["answer"])
            reference_labels.append(reference_label)
            prediction_labels.append(predicted_label)

        question = reference.get("question")
        if not isinstance(question, str):
            raise ValueError(f"reference {key} has no text question")
        parsed_rows.append(
            {
                "config": config,
                "split": split,
                "task": task,
                "sample_id": key[0],
                "question_id": key[1],
                "question_type": str(reference.get("question_type", "unknown")),
                "question": question,
                "valid_output": int(valid),
                "parsed_answer": answer,
                "prediction_label": "" if predicted_label is None else predicted_label,
                "reference_label": "" if reference_label is None else reference_label,
                "reference_text": reference_text,
            }
        )

    num_examples = len(references)
    metric_row: dict[str, Any] = {
        "config": config,
        "split": split,
        "task": task,
        "num_examples": num_examples,
        "valid_examples": num_examples - invalid_examples,
        "invalid_examples": invalid_examples,
        "invalid_output_rate": invalid_examples / num_examples if num_examples else 0.0,
        "accuracy": None,
        "macro_precision": None,
        "macro_recall": None,
        "macro_f1": None,
        "rouge_l_f1": None,
        "meteor": None,
        "llm_judge": None,
    }
    if task == "open":
        metric_row["rouge_l_f1"] = sum(rouge_scores) / num_examples if num_examples else 0.0
        metric_row["meteor"] = sum(meteor_scores) / num_examples if num_examples else 0.0
        if judge_values is not None:
            metric_row["llm_judge"] = (
                sum(judge_values.values()) / num_examples if num_examples else 0.0
            )
    else:
        metric_row.update(
            _closed_metrics(
                reference_labels,
                prediction_labels,
                num_classes=2 if task == "binary" else 3,
            )
        )
    return metric_row, parsed_rows


def evaluate_xqa_by_question_type(
    reference_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    task: str,
    config: str,
    split: str,
    judge_rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Compute the same metrics independently for each released question type."""

    prediction_map = {
        (int(row["sample_id"]), int(row["question_id"])): row for row in prediction_rows
    }
    judge_map = (
        {(int(row["sample_id"]), int(row["question_id"])): row for row in judge_rows}
        if judge_rows is not None
        else None
    )
    output: list[dict[str, Any]] = []
    question_types = sorted({str(row.get("question_type", "unknown")) for row in reference_rows})
    for question_type in question_types:
        selected_references = [
            row
            for row in reference_rows
            if str(row.get("question_type", "unknown")) == question_type
        ]
        keys = {(int(row["sample_id"]), int(row["question_id"])) for row in selected_references}
        selected_predictions = [prediction_map[key] for key in sorted(keys)]
        selected_judges = (
            [judge_map[key] for key in sorted(keys)] if judge_map is not None else None
        )
        metric_row, _ = evaluate_xqa_predictions(
            selected_references,
            selected_predictions,
            task=task,
            config=config,
            split=split,
            judge_rows=selected_judges,
        )
        output.append(
            {
                "config": metric_row.pop("config"),
                "split": metric_row.pop("split"),
                "task": metric_row.pop("task"),
                "question_type": question_type,
                **metric_row,
            }
        )
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--task", choices=("binary", "multi", "open"), required=True)
    parser.add_argument("--split", choices=("train", "validation", "val", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--dataset-id", default="dasyd/quants")
    parser.add_argument("--dataset-revision", default=None)
    return parser.parse_args()


def _xqa_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a canonical raw xQA CSV without modifying its continuations."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--task", choices=("binary", "multi", "open"), required=True)
    parser.add_argument("--split", choices=("train", "validation", "val", "test"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parsed-output", type=Path)
    parser.add_argument("--question-type-output", type=Path)
    parser.add_argument(
        "--judge-ratings",
        type=Path,
        help="Optional exact-key rating CSV with total_rating and normalized_rating.",
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--dataset-id", default="dasyd/quants")
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--cache-dir", type=Path)
    return parser.parse_args()


def xqa_metrics_main() -> None:
    args = _xqa_parse_args()
    from .constants import DATASET_REVISION
    from .data import load_question_rows

    revision = args.dataset_revision or DATASET_REVISION
    split = "validation" if args.split == "val" else args.split
    reference_rows = list(
        load_question_rows(
            task=args.task,
            split=split,
            dataset_id=args.dataset_id,
            revision=revision,
            data_root=args.data_root,
            cache_dir=args.cache_dir,
        )
    )
    prediction_rows = read_prediction_csv(args.predictions)
    judge_rows = (
        read_csv_rows(args.judge_ratings, fieldnames=JUDGE_COLUMNS)
        if args.judge_ratings is not None
        else None
    )
    metric_row, parsed_rows = evaluate_xqa_predictions(
        reference_rows,
        prediction_rows,
        task=args.task,
        config=args.config,
        split=split,
        judge_rows=judge_rows,
    )
    question_type_rows = evaluate_xqa_by_question_type(
        reference_rows,
        prediction_rows,
        task=args.task,
        config=args.config,
        split=split,
        judge_rows=judge_rows,
    )
    write_csv_atomic(args.output, [metric_row], fieldnames=XQA_METRIC_COLUMNS)
    parsed_output = args.parsed_output or args.output.with_name(
        args.output.stem + "_parsed.csv"
    )
    write_csv_atomic(parsed_output, parsed_rows, fieldnames=XQA_PARSED_COLUMNS)
    question_type_output = args.question_type_output or args.output.with_name(
        args.output.stem + "_by_question_type.csv"
    )
    write_csv_atomic(
        question_type_output,
        question_type_rows,
        fieldnames=XQA_QUESTION_TYPE_METRIC_COLUMNS,
    )


def _aggregate_xqa_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine validated one-row xQA metric CSVs.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def aggregate_xqa_metrics_main() -> None:
    args = _aggregate_xqa_parse_args()
    aggregate_xqa_metric_files(args.inputs, args.output)


def aggregate_xqa_metric_files(
    inputs: Sequence[str | Path],
    output: str | Path,
) -> list[dict[str, Any]]:
    """Validate and combine per-configuration xQA metric files."""

    rows: list[dict[str, Any]] = []
    keys: set[tuple[str, str, str]] = set()
    for path in inputs:
        for row in read_csv_rows(path, fieldnames=XQA_METRIC_COLUMNS):
            key = (row["config"], row["split"], row["task"])
            if key in keys:
                raise ValueError(f"duplicate xQA metric key {key}")
            keys.add(key)
            rows.append(row)
    rows.sort(key=lambda row: (row["config"], row["split"], row["task"]))
    write_csv_atomic(output, rows, fieldnames=XQA_METRIC_COLUMNS)
    return rows


def main() -> None:
    args = _parse_args()
    from .constants import DATASET_REVISION
    from .data import load_question_rows

    revision = args.dataset_revision or DATASET_REVISION
    split = "validation" if args.split == "val" else args.split
    rows = load_question_rows(
        task=args.task,
        split=split,
        dataset_id=args.dataset_id,
        revision=revision,
        data_root=args.data_root,
    )
    ground_truth = ground_truth_from_rows(rows)
    action_rows = read_action_prediction_csv(
        args.predictions,
        action_names=ACTION_NAMES,
        expected_sample_ids=ground_truth.keys(),
    )
    predictions = action_rows_to_sequences(action_rows)
    unique_sample_result = evaluate_action_sequences(ground_truth, predictions)
    qa_row_weighted_result = evaluate_qa_row_weighted(rows, predictions)
    write_metric_outputs(
        args.output_dir,
        unique_sample_result,
        qa_row_weighted_result=qa_row_weighted_result,
        metadata={
            "task": args.task,
            "split": split,
            "dataset_id": args.dataset_id,
            "dataset_revision": revision,
            "prediction_csv": str(args.predictions),
        },
    )


if __name__ == "__main__":
    main()
