"""Portable, resumable training CLI for the QuAnTS xLSTMMixer action encoder."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import random
import sys
import tempfile
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .constants import ACTION_NAMES, DATASET_ID, DATASET_REVISION, TRANSFORMERS_REVISION
from .data import (
    data_source_fingerprint,
    data_source_manifest,
    load_quants_segments,
    validate_data_source_location,
)
from .model import (
    build_action_encoder,
    extract_logits,
    extract_loss,
    forward_action_encoder,
    load_action_encoder,
)

CHECKPOINT_FORMAT_VERSION = 2
RUN_MANIFEST_SCHEMA_VERSION = 1


def set_seed(seed: int) -> None:
    """Seed the randomness used by model initialization and training."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto`` without assuming that a CUDA device is present."""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("a CUDA device was requested but CUDA is unavailable")
    return device


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def build_run_manifest(
    args: argparse.Namespace,
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Capture every setting that can affect training or data identity."""

    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "task": args.task,
        "data": dict(source_manifest),
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "max_grad_norm": args.max_grad_norm,
            "max_train_batches": args.max_train_batches,
            "max_val_batches": args.max_val_batches,
            "deduplicate_samples": args.deduplicate_samples,
            "skip_data_validation": args.skip_data_validation,
            "optimizer": "torch.optim.RAdam",
        },
        "model": {
            "backend": "xLSTMMixerForTimeSeriesClassification",
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "dropout": args.dropout,
            "head_dropout": args.head_dropout,
            "backcast": not args.no_backcast,
            "prediction_length": args.prediction_length,
            "context_length": 80,
            "num_input_channels": 72,
            "num_labels": 19,
            "scaling": "mean",
        },
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "numpy": np.__version__,
            "datasets": _distribution_version("datasets"),
            "transformers": _distribution_version("transformers"),
            "transformers_revision": TRANSFORMERS_REVISION,
            "xlstm": _distribution_version("xlstm"),
        },
    }


def validate_resume_manifest(
    state: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    run_fingerprint: str,
) -> None:
    """Reject a checkpoint if any data, model, optimizer, or runtime input changed."""

    stored_manifest = state.get("run_manifest")
    stored_fingerprint = state.get("run_fingerprint")
    if not isinstance(stored_manifest, Mapping) or not isinstance(stored_fingerprint, str):
        raise ValueError("checkpoint does not contain a complete run manifest")
    if data_source_fingerprint(stored_manifest) != stored_fingerprint:
        raise ValueError("checkpoint run manifest does not match its stored fingerprint")
    if stored_fingerprint != run_fingerprint or stored_manifest != run_manifest:
        changed = sorted(
            key
            for key in set(stored_manifest) | set(run_manifest)
            if stored_manifest.get(key) != run_manifest.get(key)
        )
        raise ValueError(
            "checkpoint run configuration differs from this invocation; changed "
            f"sections: {', '.join(changed) or 'unknown'}"
        )


def load_preflight_source_manifest(path: str | Path, task: str) -> dict[str, Any]:
    """Load and validate one task's content identity from a CPU preflight report."""

    report_path = Path(path).expanduser()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"preflight report is not an object: {report_path}")
    stored_fingerprint = report.get("manifest_fingerprint")
    unsigned_report = {key: value for key, value in report.items() if key != "manifest_fingerprint"}
    if stored_fingerprint != data_source_fingerprint(unsigned_report):
        raise ValueError(f"preflight report fingerprint is invalid: {report_path}")
    if report.get("dataset_id") != DATASET_ID or report.get("dataset_revision") != DATASET_REVISION:
        raise ValueError("preflight report does not describe the pinned QuAnTS dataset")
    task_report = report.get("tasks", {}).get(task)
    if not isinstance(task_report, Mapping):
        raise ValueError(f"preflight report does not contain task {task!r}")
    source = task_report.get("source_manifest")
    source_fingerprint = task_report.get("source_fingerprint")
    if not isinstance(source, dict) or source_fingerprint != data_source_fingerprint(source):
        raise ValueError(f"preflight data identity is invalid for task {task!r}")
    if not {"train", "val"}.issubset(set(source.get("splits", []))):
        raise ValueError("preflight data identity must include train and val splits")
    selected = {**source, "splits": ["train", "val"]}
    if source.get("kind") == "local_parquet":
        shards = source.get("shards")
        if not isinstance(shards, list) or any(
            not isinstance(shard, Mapping)
            or shard.get("split") not in {"train", "val", "test"}
            for shard in shards
        ):
            raise ValueError("preflight local shard records require a valid split")
        selected["shards"] = [
            shard for shard in shards if shard.get("split") in {"train", "val"}
        ]
    return selected


def _seed_worker(worker_id: int, *, base_seed: int) -> None:
    worker_seed = (base_seed + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def make_dataloader(
    dataset: Dataset[Any],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader[Any]:
    """Create a deterministic loader; callers vary the seed by epoch."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=partial(_seed_worker, base_seed=seed),
        generator=generator,
    )


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None = None,
    max_grad_norm: float | None = None,
    log_every: int = 0,
) -> dict[str, float | int]:
    """Run one train or validation epoch and return loss and accuracy."""

    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    completed_batches = 0

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            past_values = batch["past_values"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            labels = batch["labels"].to(
                device=device,
                dtype=torch.long,
                non_blocking=True,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = forward_action_encoder(model, past_values, labels)
            loss = extract_loss(output)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at batch {batch_index}: {loss}")
            if training:
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            logits = extract_logits(output)
            batch_size = labels.numel()
            total_loss += float(loss.detach()) * batch_size
            total_correct += int((logits.argmax(dim=-1) == labels).sum().item())
            total_examples += batch_size
            completed_batches += 1
            if log_every and completed_batches % log_every == 0:
                mode = "train" if training else "val"
                print(
                    f"{mode} batch={completed_batches} "
                    f"loss={total_loss / total_examples:.6f} "
                    f"accuracy={total_correct / total_examples:.6f}",
                    flush=True,
                )

    if not total_examples:
        raise RuntimeError("epoch processed no examples")
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "examples": total_examples,
        "batches": completed_batches,
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def _checkpoint_name(completed_epoch: int) -> str:
    return f"checkpoint-epoch-{completed_epoch:04d}"


def save_checkpoint(
    output_dir: str | Path,
    *,
    completed_epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    history: Sequence[Mapping[str, Any]],
    task: str,
    deduplicate_samples: bool,
    run_manifest: Mapping[str, Any],
    run_fingerprint: str,
) -> Path:
    """Atomically publish a model plus optimizer/RNG state for exact resumption."""

    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / _checkpoint_name(completed_epoch)
    if destination.exists():
        raise FileExistsError(f"checkpoint already exists: {destination}")
    temporary = Path(tempfile.mkdtemp(dir=root, prefix=f".{destination.name}.tmp-"))
    try:
        model.save_pretrained(temporary / "model", safe_serialization=True)  # type: ignore[attr-defined]
        state: dict[str, Any] = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "completed_epoch": completed_epoch,
            "optimizer": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
            "history": [dict(item) for item in history],
            "task": task,
            "deduplicate_samples": deduplicate_samples,
            "run_manifest": dict(run_manifest),
            "run_fingerprint": run_fingerprint,
        }
        torch.save(state, temporary / "training_state.pt")
        _atomic_json(
            temporary / "manifest.json",
            {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "completed_epoch": completed_epoch,
                "task": task,
                "deduplicate_samples": deduplicate_samples,
                "dataset_id": DATASET_ID,
                "dataset_revision": DATASET_REVISION,
                "transformers_revision": TRANSFORMERS_REVISION,
                "run_manifest": dict(run_manifest),
                "run_fingerprint": run_fingerprint,
            },
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            # This branch only contains files if publication failed. Avoid a broad
            # recursive delete; leave it for diagnosis and report its exact path.
            print(f"incomplete checkpoint retained at {temporary}", flush=True)
    return destination


def latest_checkpoint(output_dir: str | Path) -> Path:
    """Return the most recent complete epoch checkpoint."""

    root = Path(output_dir).expanduser()
    candidates = sorted(
        path
        for path in root.glob("checkpoint-epoch-*")
        if (path / "training_state.pt").is_file() and (path / "model" / "config.json").is_file()
    )
    if not candidates:
        raise FileNotFoundError(f"no complete checkpoints found below {root}")
    return candidates[-1]


def resolve_resume_checkpoint(value: str | Path, output_dir: str | Path) -> Path:
    if str(value) == "latest":
        return latest_checkpoint(output_dir)
    path = Path(value).expanduser()
    if not (path / "training_state.pt").is_file():
        raise FileNotFoundError(f"missing training_state.pt in checkpoint {path}")
    return path


def load_training_state(path: str | Path) -> dict[str, Any]:
    """Load only tensors and primitive state from a package-created checkpoint."""

    state_path = Path(path) / "training_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"invalid training state in {state_path}")
    if state.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format {state.get('format_version')!r} in {state_path}"
        )
    return state


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                state[name] = value.to(device)


def restore_training_state(
    state: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
) -> tuple[int, list[dict[str, Any]]]:
    optimizer.load_state_dict(state["optimizer"])
    _optimizer_to(optimizer, device)
    torch.set_rng_state(state["torch_rng_state"])
    cuda_states = state.get("cuda_rng_state_all", [])
    if torch.cuda.is_available() and cuda_states:
        torch.cuda.set_rng_state_all(cuda_states)
    completed_epoch = int(state["completed_epoch"])
    history = [dict(item) for item in state.get("history", [])]
    return completed_epoch, history


def _publish_model(
    output_dir: Path,
    model: torch.nn.Module,
    summary: Mapping[str, Any],
    *,
    name: str,
) -> Path:
    destination = output_dir / name
    if destination.exists():
        config_path = destination / "config.json"
        summary_path = destination / "training_summary.json"
        weight_files = list(destination.glob("*.safetensors")) + list(
            destination.glob("pytorch_model*.bin")
        )
        if not config_path.is_file() or not summary_path.is_file() or not weight_files:
            raise FileExistsError(f"published model is incomplete: {destination}")
        existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        identity_keys = (
            "task",
            "epochs",
            "dataset_revision",
            "best_checkpoint",
            "run_fingerprint",
        )
        if any(existing_summary.get(key) != summary.get(key) for key in identity_keys):
            raise FileExistsError(
                f"published model metadata does not match this run: {destination}"
            )
        return destination
    temporary = Path(tempfile.mkdtemp(dir=output_dir, prefix=f".{name}.tmp-"))
    try:
        model.save_pretrained(temporary, safe_serialization=True)  # type: ignore[attr-defined]
        _atomic_json(temporary / "training_summary.json", summary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            print(f"incomplete {name} model retained at {temporary}", flush=True)
    return destination


def run_backend_smoke(args: argparse.Namespace) -> Path:
    """Exercise forward, loss, backward, RAdam, save, and reload on the real backend."""

    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = build_action_encoder(
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        backcast=not args.no_backcast,
        prediction_length=args.prediction_length,
    ).to(device)
    optimizer = torch.optim.RAdam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    past_values = torch.randn(2, 80, 72, device=device)
    labels = torch.tensor([0, 18], dtype=torch.long, device=device)
    optimizer.zero_grad(set_to_none=True)
    output = forward_action_encoder(model, past_values, labels)
    loss = extract_loss(output)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"smoke loss is non-finite: {loss}")
    loss.backward()
    optimizer.step()

    smoke_manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "kind": "real_backend_smoke",
        "task": args.task,
        "seed": args.seed,
        "model": {
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "dropout": args.dropout,
            "head_dropout": args.head_dropout,
            "backcast": not args.no_backcast,
            "prediction_length": args.prediction_length,
        },
        "software": {
            "torch": torch.__version__,
            "transformers": _distribution_version("transformers"),
            "transformers_revision": TRANSFORMERS_REVISION,
            "xlstm": _distribution_version("xlstm"),
        },
    }
    smoke_fingerprint = data_source_fingerprint(smoke_manifest)
    summary = {
        "task": args.task,
        "epochs": 0,
        "dataset_revision": DATASET_REVISION,
        "best_checkpoint": None,
        "run_fingerprint": smoke_fingerprint,
        "run_manifest": smoke_manifest,
    }
    model.eval()
    with torch.inference_mode():
        expected_logits = extract_logits(forward_action_encoder(model, past_values)).detach()
    model_path = _publish_model(args.output_dir, model, summary, name="smoke_model")
    reloaded = load_action_encoder(model_path, device=device)
    reloaded.eval()
    with torch.inference_mode():
        actual_logits = extract_logits(forward_action_encoder(reloaded, past_values))
    if not torch.allclose(expected_logits, actual_logits, rtol=1e-5, atol=1e-6):
        difference = float((expected_logits - actual_logits).abs().max())
        raise RuntimeError(f"reloaded smoke logits differ; max_abs_difference={difference}")
    _atomic_json(
        args.output_dir / "smoke_test.json",
        {
            "status": "passed",
            "loss": float(loss.detach()),
            "device": str(device),
            "run_fingerprint": smoke_fingerprint,
            "model_path": str(model_path),
        },
    )
    return model_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("binary", "multi", "open"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--preflight-manifest",
        type=Path,
        help="reuse a successful CPU data-preflight manifest instead of hashing shards again",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument(
        "--deduplicate-samples",
        action="store_true",
        help=(
            "train each unique sample once instead of retaining per-question-row "
            "weighting"
        ),
    )
    parser.add_argument(
        "--skip-data-validation",
        action="store_true",
        help="skip the eager shape/action-label preflight (rows remain validated lazily)",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        nargs="?",
        const="latest",
        help="resume from a checkpoint directory; omit the value to select the latest",
    )
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--head-dropout", type=float, default=0.0)
    parser.add_argument("--prediction-length", type=int, default=96)
    parser.add_argument("--no-backcast", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="run a synthetic real-backend train/save/reload check without loading QuAnTS",
    )
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    return args


def train(args: argparse.Namespace) -> Path:
    """Execute training and return the published best-validation model directory."""

    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.preflight_manifest is not None:
        source_manifest = load_preflight_source_manifest(args.preflight_manifest, args.task)
        validate_data_source_location(source_manifest, args.data_root)
    else:
        source_manifest = data_source_manifest(
            args.task,
            ("train", "val"),
            data_root=args.data_root,
        )
    run_manifest = build_run_manifest(args, source_manifest)
    run_fingerprint = data_source_fingerprint(run_manifest)

    print(
        f"loading pinned {DATASET_ID}@{DATASET_REVISION} task={args.task}; "
        f"deduplicate_samples={args.deduplicate_samples}; "
        f"run_fingerprint={run_fingerprint}",
        flush=True,
    )
    train_dataset = load_quants_segments(
        args.task,
        "train",
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        deduplicate_samples=args.deduplicate_samples,
    )
    val_dataset = load_quants_segments(
        args.task,
        "val",
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        deduplicate_samples=args.deduplicate_samples,
    )
    if not args.skip_data_validation:
        train_dataset.validate_all_samples()
        val_dataset.validate_all_samples()

    resume_path: Path | None = None
    if args.resume_from_checkpoint is not None:
        resume_path = resolve_resume_checkpoint(args.resume_from_checkpoint, args.output_dir)
        state = load_training_state(resume_path)
        validate_resume_manifest(state, run_manifest, run_fingerprint)
        if state.get("task") != args.task:
            raise ValueError(
                f"checkpoint task {state.get('task')!r} does not match {args.task!r}"
            )
        if bool(state.get("deduplicate_samples")) != args.deduplicate_samples:
            raise ValueError("checkpoint and CLI disagree on --deduplicate-samples")
        model = load_action_encoder(resume_path, device=device)
    else:
        state = None
        model = build_action_encoder(
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            head_dropout=args.head_dropout,
            backcast=not args.no_backcast,
            prediction_length=args.prediction_length,
        ).to(device)

    optimizer = torch.optim.RAdam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    start_epoch = 0
    history: list[dict[str, Any]] = []
    if state is not None:
        start_epoch, history = restore_training_state(state, optimizer, device=device)
    if start_epoch > args.epochs:
        raise ValueError(
            f"checkpoint completed epoch {start_epoch}, but --epochs is {args.epochs}"
        )

    for epoch_index in range(start_epoch, args.epochs):
        epoch_number = epoch_index + 1
        train_loader = make_dataloader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            seed=args.seed + epoch_index,
            pin_memory=device.type == "cuda",
        )
        val_loader = make_dataloader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            seed=args.seed,
            pin_memory=device.type == "cuda",
        )
        train_metrics = run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
            max_grad_norm=args.max_grad_norm,
            log_every=args.log_every,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device=device,
            optimizer=None,
            max_batches=args.max_val_batches,
            log_every=0,
        )
        record = {
            "epoch": epoch_number,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(record)
        save_checkpoint(
            args.output_dir,
            completed_epoch=epoch_number,
            model=model,
            optimizer=optimizer,
            history=history,
            task=args.task,
            deduplicate_samples=args.deduplicate_samples,
            run_manifest=run_manifest,
            run_fingerprint=run_fingerprint,
        )
        print(json.dumps(record, sort_keys=True), flush=True)

    best_record = min(history, key=lambda item: float(item["val_loss"]))
    best_val_loss = float(best_record["val_loss"])
    best_checkpoint = args.output_dir / _checkpoint_name(int(best_record["epoch"]))
    summary = {
        "task": args.task,
        "epochs": args.epochs,
        "seed": args.seed,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "transformers_revision": TRANSFORMERS_REVISION,
        "action_names": list(ACTION_NAMES),
        "deduplicate_samples": args.deduplicate_samples,
        "train_qa_rows": train_dataset.qa_row_count,
        "train_unique_samples": train_dataset.unique_sample_count,
        "val_qa_rows": val_dataset.qa_row_count,
        "val_unique_samples": val_dataset.unique_sample_count,
        "best_val_loss": best_val_loss,
        "best_checkpoint": str(best_checkpoint),
        "run_manifest": run_manifest,
        "run_fingerprint": run_fingerprint,
        "history": history,
    }
    _publish_model(args.output_dir, model, summary, name="final")
    best_model = load_action_encoder(best_checkpoint, device="cpu")
    return _publish_model(args.output_dir, best_model, summary, name="best")


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.smoke_test:
        smoke_model = run_backend_smoke(args)
        print(f"backend smoke model: {smoke_model}", flush=True)
        return
    best_model = train(args)
    print(f"best validation model: {best_model}", flush=True)


if __name__ == "__main__":
    main()
