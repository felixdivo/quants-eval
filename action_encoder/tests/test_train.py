from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as functional
from torch.utils.data import Dataset

import quants_action_encoder.train as train_module
from quants_action_encoder.data import data_source_fingerprint
from quants_action_encoder.train import (
    _publish_model,
    latest_checkpoint,
    load_preflight_source_manifest,
    load_training_state,
    make_dataloader,
    restore_training_state,
    run_backend_smoke,
    run_epoch,
    save_checkpoint,
    set_seed,
    validate_resume_manifest,
)

RUN_MANIFEST = {
    "schema_version": 1,
    "task": "binary",
    "data": {"kind": "huggingface", "revision": "revision"},
    "training": {"batch_size": 512, "seed": 42},
    "model": {"d_model": 256},
    "software": {"torch": "2.4.0"},
}
RUN_FINGERPRINT = data_source_fingerprint(RUN_MANIFEST)


class TinyDataset(Dataset[dict[str, torch.Tensor]]):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        values = torch.zeros(80, 72)
        values[:, 0] = float(index % 2)
        return {
            "past_values": values,
            "labels": torch.tensor(index % 2),
            "sample_id": torch.tensor(index // 4),
            "segment_id": torch.tensor(index % 4),
        }


class TinyClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = torch.nn.Linear(1, 19)

    def forward(
        self,
        *,
        past_values: torch.Tensor,
        target_values: torch.Tensor | None,
        return_loss: bool,
        return_dict: bool,
    ) -> SimpleNamespace:
        features = past_values[:, :, :1].mean(dim=1)
        logits = self.classifier(features)
        loss = (
            functional.cross_entropy(logits, target_values)
            if return_loss and target_values is not None
            else None
        )
        return SimpleNamespace(
            loss=loss,
            logits=logits,
            prediction_outputs=logits,
        )

    def save_pretrained(self, path, *, safe_serialization: bool) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(
            json.dumps(
                {"context_length": 80, "num_input_channels": 72, "num_labels": 19}
            ),
            encoding="utf-8",
        )
        torch.save(self.state_dict(), path / "model.safetensors")


def test_run_epoch_updates_weights_and_reports_counts() -> None:
    set_seed(42)
    model = TinyClassifier()
    before = model.classifier.weight.detach().clone()
    loader = make_dataloader(
        TinyDataset(),
        batch_size=4,
        shuffle=True,
        num_workers=0,
        seed=42,
        pin_memory=False,
    )
    optimizer = torch.optim.RAdam(model.parameters(), lr=1e-2)
    metrics = run_epoch(
        model,
        loader,
        device=torch.device("cpu"),
        optimizer=optimizer,
    )
    assert metrics["examples"] == 8
    assert metrics["batches"] == 2
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["loss"] > 0.0
    assert not torch.equal(before, model.classifier.weight)


def test_epoch_checkpoint_contains_resume_state_and_is_discoverable(tmp_path) -> None:
    set_seed(7)
    model = TinyClassifier()
    optimizer = torch.optim.RAdam(model.parameters(), lr=1e-3)
    checkpoint = save_checkpoint(
        tmp_path,
        completed_epoch=1,
        model=model,
        optimizer=optimizer,
        history=[{"epoch": 1, "val_loss": 2.0}],
        task="binary",
        deduplicate_samples=False,
        run_manifest=RUN_MANIFEST,
        run_fingerprint=RUN_FINGERPRINT,
    )
    assert latest_checkpoint(tmp_path) == checkpoint
    assert (checkpoint / "model" / "config.json").is_file()
    state = load_training_state(checkpoint)
    assert state["completed_epoch"] == 1
    assert state["task"] == "binary"
    assert state["deduplicate_samples"] is False
    validate_resume_manifest(state, RUN_MANIFEST, RUN_FINGERPRINT)

    restored_optimizer = torch.optim.RAdam(model.parameters(), lr=9e-3)
    completed_epoch, history = restore_training_state(
        state,
        restored_optimizer,
        device=torch.device("cpu"),
    )
    assert completed_epoch == 1
    assert history == [{"epoch": 1, "val_loss": 2.0}]
    assert restored_optimizer.param_groups[0]["lr"] == 1e-3


def test_model_publication_is_idempotent_after_a_finalization_crash(tmp_path) -> None:
    model = TinyClassifier()
    summary = {
        "task": "binary",
        "epochs": 5,
        "dataset_revision": "revision",
        "best_checkpoint": "checkpoint-epoch-0003",
        "run_fingerprint": RUN_FINGERPRINT,
    }
    first = _publish_model(tmp_path, model, summary, name="final")
    second = _publish_model(tmp_path, model, summary, name="final")
    assert first == second == tmp_path / "final"


def test_resume_rejects_changed_training_or_data_identity() -> None:
    state = {
        "run_manifest": RUN_MANIFEST,
        "run_fingerprint": RUN_FINGERPRINT,
    }
    changed = {**RUN_MANIFEST, "training": {"batch_size": 256, "seed": 42}}
    try:
        validate_resume_manifest(state, changed, data_source_fingerprint(changed))
    except ValueError as exc:
        assert "changed sections: training" in str(exc)
    else:
        raise AssertionError("changed run configuration was accepted")


def test_preflight_manifest_is_verified_before_gpu_training(tmp_path) -> None:
    source = {
        "kind": "local_parquet",
        "dataset_id": "dasyd/quants",
        "revision": "0e78849313d3b043a0b08697dc004fb6cba15df9",
        "task": "binary",
        "splits": ["train", "val"],
        "shards": [
            {
                "split": "train",
                "name": "train.parquet",
                "sha256": "a",
                "size_bytes": 1,
            }
        ],
    }
    report = {
        "schema_version": 1,
        "dataset_id": "dasyd/quants",
        "dataset_revision": "0e78849313d3b043a0b08697dc004fb6cba15df9",
        "tasks": {
            "binary": {
                "source_manifest": source,
                "source_fingerprint": data_source_fingerprint(source),
            }
        },
        "split_unique_samples": {"train": 1, "val": 1},
    }
    report["manifest_fingerprint"] = data_source_fingerprint(report)
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert load_preflight_source_manifest(path, "binary") == source

    source["shards"][0].pop("split")
    report["tasks"]["binary"]["source_manifest"] = source
    report["tasks"]["binary"]["source_fingerprint"] = data_source_fingerprint(source)
    unsigned = {key: value for key, value in report.items() if key != "manifest_fingerprint"}
    report["manifest_fingerprint"] = data_source_fingerprint(unsigned)
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="require a valid split"):
        load_preflight_source_manifest(path, "binary")


def test_real_backend_smoke_mode_exercises_save_and_reload(monkeypatch, tmp_path) -> None:
    model = TinyClassifier()
    monkeypatch.setattr(train_module, "build_action_encoder", lambda **kwargs: model)
    monkeypatch.setattr(train_module, "load_action_encoder", lambda *args, **kwargs: model)
    args = argparse.Namespace(
        output_dir=tmp_path,
        task="binary",
        seed=42,
        device="cpu",
        d_model=256,
        num_layers=2,
        num_heads=8,
        dropout=0.1,
        head_dropout=0.0,
        no_backcast=False,
        prediction_length=96,
        learning_rate=1e-4,
        weight_decay=0.0,
    )
    model_path = run_backend_smoke(args)
    assert model_path == tmp_path / "smoke_model"
    assert json.loads((tmp_path / "smoke_test.json").read_text())["status"] == "passed"
