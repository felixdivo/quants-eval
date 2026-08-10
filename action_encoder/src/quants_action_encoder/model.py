"""Thin, testable adapters around the pinned Transformers xLSTMMixer fork."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .constants import INPUT_CHANNELS, NUM_ACTIONS, SEGMENT_LENGTH


def _xlstm_classes() -> tuple[type[Any], type[torch.nn.Module]]:
    """Import the non-upstream model lazily and provide an actionable error."""

    try:
        from transformers import (
            xLSTMMixerConfig,
            xLSTMMixerForTimeSeriesClassification,
        )
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "xLSTMMixer is unavailable. Install mauricekraus/transformers at "
            "commit e97fc33f97155b79658af37fde29b39b5139e066 and its xlstm "
            "dependency."
        ) from exc
    return xLSTMMixerConfig, xLSTMMixerForTimeSeriesClassification


def build_action_encoder(
    *,
    d_model: int = 256,
    num_layers: int = 2,
    num_heads: int = 8,
    dropout: float = 0.1,
    head_dropout: float = 0.0,
    backcast: bool = True,
    prediction_length: int = 96,
    scaling: str = "mean",
) -> torch.nn.Module:
    """Instantiate the 19-class xLSTMMixer action encoder.

    The experiment uses 80 timesteps, 72 channels, 256 hidden dimensions,
    backcast enabled, and all other architecture settings inherited
    from the pinned fork's configuration defaults.
    """

    config_class, model_class = _xlstm_classes()
    config = config_class(
        context_length=SEGMENT_LENGTH,
        prediction_length=prediction_length,
        num_input_channels=INPUT_CHANNELS,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        head_dropout=head_dropout,
        num_labels=NUM_ACTIONS,
        backcast=backcast,
        scaling=scaling,
        return_loss=True,
    )
    return model_class(config)


def _pretrained_model_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if (candidate / "model" / "config.json").is_file():
        candidate = candidate / "model"
    if not (candidate / "config.json").is_file():
        raise FileNotFoundError(
            f"no xLSTMMixer config.json found at {candidate} or {candidate / 'model'}"
        )
    return candidate


def _validate_action_config(config: Any) -> None:
    expected = {
        "context_length": SEGMENT_LENGTH,
        "num_input_channels": INPUT_CHANNELS,
        "num_labels": NUM_ACTIONS,
    }
    for name, expected_value in expected.items():
        actual = getattr(config, name, None)
        if actual != expected_value:
            raise ValueError(
                f"checkpoint config {name}={actual!r}; expected {expected_value!r}"
            )


def load_action_encoder(
    path: str | Path,
    device: str | torch.device | None = None,
) -> torch.nn.Module:
    """Load a final model directory or this package's epoch checkpoint."""

    _, model_class = _xlstm_classes()
    model_path = _pretrained_model_path(path)
    model = model_class.from_pretrained(str(model_path))
    _validate_action_config(model.config)
    if device is not None:
        model.to(torch.device(device))
    return model


def extract_logits(output: Any) -> torch.Tensor:
    """Extract classifier scores from both fork output spellings.

    Commit ``e97fc33`` exposes both ``.logits`` and ``.prediction_outputs``.
    Supporting either name keeps checkpoint tools compatible with both wrappers.
    """

    if isinstance(output, torch.Tensor):
        return output
    for name in ("logits", "prediction_outputs"):
        value = output.get(name) if isinstance(output, Mapping) else getattr(output, name, None)
        if isinstance(value, torch.Tensor):
            return value
    raise TypeError("model output has neither tensor logits nor prediction_outputs")


def extract_loss(output: Any) -> torch.Tensor:
    """Return the scalar loss from a model output."""

    value = output.get("loss") if isinstance(output, Mapping) else getattr(output, "loss", None)
    if not isinstance(value, torch.Tensor):
        raise TypeError("model output does not contain a tensor loss")
    if value.numel() != 1:
        raise ValueError(f"expected a scalar loss, got shape {tuple(value.shape)}")
    return value


def forward_action_encoder(
    model: torch.nn.Module,
    past_values: torch.Tensor,
    labels: torch.Tensor | None = None,
) -> Any:
    """Call the exact classifier API at the pinned fork revision.

    The pinned classifier accepts ``target_values``. This wrapper presents a stable
    ``labels`` interface to training and inference callers.
    """

    if past_values.ndim != 3 or tuple(past_values.shape[1:]) != (
        SEGMENT_LENGTH,
        INPUT_CHANNELS,
    ):
        raise ValueError(
            "past_values must have shape [batch, 80, 72], got "
            f"{tuple(past_values.shape)}"
        )
    if labels is not None:
        if labels.ndim != 1 or labels.shape[0] != past_values.shape[0]:
            raise ValueError(
                f"labels must have shape [{past_values.shape[0]}], got {tuple(labels.shape)}"
            )
        labels = labels.to(dtype=torch.long)
    return model(
        past_values=past_values,
        target_values=labels,
        return_loss=labels is not None,
        return_dict=True,
    )


@torch.inference_mode()
def predict_action_ids(
    model: torch.nn.Module,
    past_values: torch.Tensor,
) -> torch.Tensor:
    """Greedily predict one of the 19 action IDs for each ``[80,72]`` input."""

    if past_values.ndim == 2:
        past_values = past_values.unsqueeze(0)
    was_training = model.training
    model.eval()
    try:
        logits = extract_logits(forward_action_encoder(model, past_values))
    finally:
        model.train(was_training)
    if logits.ndim != 2 or logits.shape != (past_values.shape[0], NUM_ACTIONS):
        raise ValueError(
            f"expected logits shape [{past_values.shape[0]}, {NUM_ACTIONS}], "
            f"got {tuple(logits.shape)}"
        )
    return logits.argmax(dim=-1)
