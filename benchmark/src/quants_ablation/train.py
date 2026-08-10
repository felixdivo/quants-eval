from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.loss.loss_utils import ForCausalLMLoss
from transformers.trainer_utils import get_last_checkpoint

from .constants import (
    ABLATIONS,
    ARTIFACTS_ROOT,
    CACHE_ROOT,
    DATA_ROOT,
    MAX_CONTEXT_TOKENS,
    MODEL_ROOT,
    SEED,
    TASKS,
    config_name,
    validate_choice,
)
from .data import load_rows, load_ts_prompt_cache
from .prompts import (
    completion_example,
    naive_prompt_ids,
    question_only_prompt,
    target_text,
)


class CompletionDataset(TorchDataset):
    def __init__(
        self,
        rows: Any,
        tokenizer: Any,
        task: str,
        ablation: str,
        ts_cache: Any | None,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.task = task
        self.ablation = ablation
        self.ts_cache = ts_cache
        self.split_start = 0
        if ts_cache is not None and len(ts_cache):
            self.split_start = int(ts_cache[0]["sample_id"])

    def __len__(self) -> int:
        return len(self.rows)

    def prompt_ids(self, row: dict[str, Any]) -> list[int]:
        if self.ablation == "question_only":
            prompt = question_only_prompt(str(row["question"]))
            return self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        assert self.ts_cache is not None
        sample_id = int(row["sample_id"])
        cache_index = sample_id - self.split_start
        cached = self.ts_cache[cache_index]
        if int(cached["sample_id"]) != sample_id:
            raise ValueError(f"missing TS prompt cache for sample_id={sample_id}")
        prompt_ids = list(cached["input_ids"])
        if self.ablation == "naive":
            return naive_prompt_ids(prompt_ids, str(row["question"]), self.tokenizer)
        return prompt_ids

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        row = self.rows[index]
        prompt_ids = self.prompt_ids(row)
        answer = " " + target_text(row, self.task)
        answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
        example = completion_example(
            prompt_ids, answer_ids, self.tokenizer.eos_token_id
        )
        if len(example["input_ids"]) > MAX_CONTEXT_TOKENS:
            raise ValueError(
                f"sample_id={row['sample_id']} question_id={row['question_id']} has "
                f"{len(example['input_ids'])} tokens; truncation is forbidden"
            )
        return example


class CompletionCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def supervised_logit_indices(labels: torch.Tensor) -> torch.Tensor:
    """Return the minimal shared logit slice needed by completion labels.

    A causal label at position ``i`` is predicted by the logit at ``i - 1``.
    Prompt labels are all -100, so projecting the 128k-token vocabulary for the
    complete ~47k-token TS prompt is both unnecessary and very memory hungry.
    """
    supervised = labels.ne(-100)
    if not bool(supervised.any()):
        raise ValueError("batch contains no supervised answer tokens")
    positions = supervised.nonzero(as_tuple=False)[:, 1]
    first = int(positions.min().item())
    last = int(positions.max().item())
    if first == 0:
        raise ValueError("a completion label at position zero cannot be shifted")
    return torch.arange(first - 1, last, device=labels.device)


def model_vocab_size(model: torch.nn.Module) -> int:
    """Read vocab size through wrappers such as DistributedDataParallel."""
    unwrapped = model
    while not hasattr(unwrapped, "config") and hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module
    config = getattr(unwrapped, "config", None)
    vocab_size = getattr(config, "vocab_size", None)
    if vocab_size is None:
        raise AttributeError("model or wrapped model has no config.vocab_size")
    return int(vocab_size)


class CompletionTrainer(Trainer):
    """Trainer that computes exactly the required completion logits only."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Match the original model-loss behavior: each microbatch loss is the
        # mean over its non-masked target tokens before gradient accumulation.
        self.model_accepts_loss_kwargs = False

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> Any:
        del num_items_in_batch
        labels = inputs.pop("labels")
        logit_indices = supervised_logit_indices(labels)
        outputs = model(
            **inputs,
            labels=None,
            logits_to_keep=logit_indices,
        )
        shift_labels = labels.index_select(1, logit_indices + 1)
        loss = ForCausalLMLoss(
            logits=outputs.logits,
            labels=labels,
            shift_labels=shift_labels,
            vocab_size=model_vocab_size(model),
        )
        return (loss, outputs) if return_outputs else loss


def atomic_json(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False, newline="\n"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def build_model(model_path: Path, local_rank: int) -> Any:
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": local_rank},
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora)
    trainable = [
        name
        for name, parameter in peft_model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise ValueError("QLoRA setup produced no trainable parameters")
    unexpected = [name for name in trainable if "lora_" not in name]
    if unexpected:
        raise ValueError(
            "base model is not fully frozen; unexpected trainable parameters: "
            + ", ".join(unexpected[:10])
        )
    return peft_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", required=True, choices=ABLATIONS)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--model", type=Path, default=MODEL_ROOT / "llama-3.1-8b")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--micro-batch-size", type=int, required=True)
    parser.add_argument("--gradient-accumulation", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--save-steps", type=int, default=100)
    args = parser.parse_args()
    validate_choice(args.ablation, ABLATIONS, "ablation")
    validate_choice(args.task, TASKS, "task")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    effective_batch = world_size * args.micro_batch_size * args.gradient_accumulation
    if effective_batch != 16:
        raise ValueError(
            f"effective batch must be 16, got {world_size} × {args.micro_batch_size} × "
            f"{args.gradient_accumulation} = {effective_batch}"
        )

    set_seed(SEED)
    output = (
        args.output or ARTIFACTS_ROOT / config_name(args.ablation, args.task) / "seed42"
    )
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, use_fast=True
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("Llama tokenizer has no EOS token")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    rows = load_rows(args.task, "train", data_root=args.data_root)
    ts_cache = None
    if args.ablation in ("ts_only", "naive"):
        ts_cache = load_ts_prompt_cache(
            args.cache_root / "ts_prompts" / "train.parquet"
        )
    train_dataset = CompletionDataset(
        rows, tokenizer, args.task, args.ablation, ts_cache
    )

    model = build_model(args.model, local_rank)
    if local_rank == 0:
        model.print_trainable_parameters()
    training_args = TrainingArguments(
        output_dir=str(output / "checkpoints"),
        overwrite_output_dir=False,
        do_train=True,
        num_train_epochs=5,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        warmup_steps=0,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        max_grad_norm=1.0,
        optim="adamw_torch",
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_strategy="steps",
        logging_steps=1 if args.max_steps > 0 else 10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
        seed=SEED,
        data_seed=SEED,
    )
    trainer = CompletionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=CompletionCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    checkpoint_dir = output / "checkpoints"
    last_checkpoint = (
        get_last_checkpoint(str(checkpoint_dir)) if checkpoint_dir.exists() else None
    )
    trainer.train(resume_from_checkpoint=last_checkpoint)
    resume_verification_checkpoint = None
    if args.max_steps > 0:
        resume_verification_checkpoint = get_last_checkpoint(str(checkpoint_dir))
        if resume_verification_checkpoint is None:
            raise ValueError("smoke test did not create a resumable checkpoint")
        trainer.train(resume_from_checkpoint=resume_verification_checkpoint)
    if trainer.is_world_process_zero():
        final_dir = output / "final_adapter"
        trainer.model.save_pretrained(final_dir, safe_serialization=True)
        tokenizer.save_pretrained(final_dir)
        atomic_json(
            {
                "ablation": args.ablation,
                "task": args.task,
                "seed": SEED,
                "world_size": world_size,
                "micro_batch_size": args.micro_batch_size,
                "gradient_accumulation": args.gradient_accumulation,
                "effective_batch_size": effective_batch,
                "epochs": 5,
                "max_steps_override": args.max_steps,
                "learning_rate": 2e-4,
                "sparse_supervised_logits": True,
                "sparse_supervised_logits_exact_loss": True,
                "last_checkpoint": last_checkpoint,
                "resume_verification_checkpoint": resume_verification_checkpoint,
            },
            output / "run_manifest.json",
        )


if __name__ == "__main__":
    main()
