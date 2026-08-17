import torch
from transformers.loss.loss_utils import ForCausalLMLoss

from quants_ablation.generate import deterministic_generation_units
from quants_ablation.train import model_vocab_size, supervised_logit_indices


class _ConfiguredModel(torch.nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.config = type("Config", (), {"vocab_size": vocab_size})()


class _DistributedWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module


def test_vocab_size_is_read_through_distributed_wrapper():
    wrapped = _DistributedWrapper(_ConfiguredModel(128256))
    assert model_vocab_size(wrapped) == 128256


def test_sparse_completion_loss_matches_full_causal_loss():
    torch.manual_seed(42)
    logits = torch.randn(2, 9, 17, dtype=torch.float32)
    labels = torch.tensor(
        [
            [-100, -100, -100, -100, -100, 4, 5, 6, -100],
            [-100, -100, -100, 7, 8, 9, -100, -100, -100],
        ]
    )

    full_loss = ForCausalLMLoss(logits, labels, vocab_size=17)
    indices = supervised_logit_indices(labels)
    sparse_logits = logits.index_select(1, indices)
    shift_labels = labels.index_select(1, indices + 1)
    sparse_loss = ForCausalLMLoss(
        sparse_logits,
        labels,
        vocab_size=17,
        shift_labels=shift_labels,
    )

    assert indices.tolist() == [2, 3, 4, 5, 6]
    torch.testing.assert_close(sparse_loss, full_loss)


def test_ts_generation_groups_identical_prompts_by_sample_id():
    rows = [
        {"sample_id": 4, "question_id": 0},
        {"sample_id": 4, "question_id": 2},
        {"sample_id": 9, "question_id": 1},
    ]
    assert deterministic_generation_units(rows, "ts_only") == [[0, 1], [2]]
    assert deterministic_generation_units(rows, "question_only") == [[0], [1], [2]]
