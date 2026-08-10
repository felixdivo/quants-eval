from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as functional

import quants_action_encoder.model as model_module
from quants_action_encoder.constants import NUM_ACTIONS
from quants_action_encoder.model import (
    build_action_encoder,
    extract_logits,
    forward_action_encoder,
    load_action_encoder,
    predict_action_ids,
)


class FakeConfig:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        for name, value in kwargs.items():
            setattr(self, name, value)


class FakeClassifier(torch.nn.Module):
    loaded_path: str | None = None

    def __init__(self, config: FakeConfig) -> None:
        super().__init__()
        self.config = config
        self.bias = torch.nn.Parameter(torch.arange(NUM_ACTIONS, dtype=torch.float32))
        self.last_call: dict[str, object] | None = None

    @classmethod
    def from_pretrained(cls, path: str) -> FakeClassifier:
        cls.loaded_path = path
        return cls(
            FakeConfig(context_length=80, num_input_channels=72, num_labels=19)
        )

    def forward(
        self,
        *,
        past_values: torch.Tensor,
        target_values: torch.Tensor | None,
        return_loss: bool,
        return_dict: bool,
    ) -> SimpleNamespace:
        self.last_call = {
            "target_values": target_values,
            "return_loss": return_loss,
            "return_dict": return_dict,
        }
        logits = self.bias.unsqueeze(0).expand(past_values.shape[0], -1)
        loss = (
            functional.cross_entropy(logits, target_values)
            if target_values is not None
            else None
        )
        return SimpleNamespace(
            loss=loss,
            logits=logits,
            prediction_outputs=logits,
        )


@pytest.fixture
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        model_module,
        "_xlstm_classes",
        lambda: (FakeConfig, FakeClassifier),
    )


def test_xlstm_experiment_configuration(fake_backend: None) -> None:
    model = build_action_encoder()
    config = model.config.kwargs
    assert config["context_length"] == 80
    assert config["num_input_channels"] == 72
    assert config["num_labels"] == 19
    assert config["d_model"] == 256
    assert config["num_layers"] == 2
    assert config["num_heads"] == 8
    assert config["backcast"] is True
    assert config["prediction_length"] == 96


def test_forward_adapts_labels_to_pinned_target_values(fake_backend: None) -> None:
    model = build_action_encoder()
    inputs = torch.zeros(2, 80, 72)
    labels = torch.tensor([0, 1])
    output = forward_action_encoder(model, inputs, labels)
    assert output.loss.ndim == 0
    assert model.last_call is not None
    assert model.last_call["target_values"] is labels
    assert model.last_call["return_loss"] is True
    assert model.last_call["return_dict"] is True


def test_forward_rejects_shape_mismatches(fake_backend: None) -> None:
    model = build_action_encoder()
    with pytest.raises(ValueError, match=r"\[batch, 80, 72\]"):
        forward_action_encoder(model, torch.zeros(2, 320, 72))
    with pytest.raises(ValueError, match="labels must have shape"):
        forward_action_encoder(model, torch.zeros(2, 80, 72), torch.zeros(2, 1))


def test_extract_logits_supports_both_output_names() -> None:
    logits = torch.randn(2, 19)
    assert extract_logits(SimpleNamespace(logits=logits)) is logits
    assert extract_logits({"prediction_outputs": logits}) is logits
    with pytest.raises(TypeError, match="neither tensor logits"):
        extract_logits(SimpleNamespace(loss=torch.tensor(1.0)))


def test_predict_action_ids_is_greedy_and_restores_mode(fake_backend: None) -> None:
    model = build_action_encoder()
    model.train()
    predictions = predict_action_ids(model, torch.zeros(3, 80, 72))
    assert predictions.tolist() == [18, 18, 18]
    assert model.training is True


def test_load_accepts_final_or_epoch_checkpoint_layout(
    fake_backend: None, tmp_path
) -> None:
    checkpoint = tmp_path / "checkpoint-epoch-0001"
    model_dir = checkpoint / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    loaded = load_action_encoder(checkpoint, device="cpu")
    assert isinstance(loaded, FakeClassifier)
    assert FakeClassifier.loaded_path == str(model_dir)
