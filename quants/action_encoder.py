from pathlib import Path
from typing import Literal, cast
from einops import rearrange
from lightning import LightningModule
from omegaconf import DictConfig
import torch
from transformers import PatchTSMixerConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.models.patchtsmixer.modeling_patchtsmixer import (
    PatchTSMixerForPreTrainingOutput,
    PatchTSMixerForTimeSeriesClassification,
)
from outlines.models import transformers
import outlines.generate as gen
from outlines.samplers import multinomial
import lightning as L
import torchmetrics as tm
import json
import re
from pydantic import BaseModel


class Task(BaseModel):
    steps: list[str]
    timeseries_actions: list[str]


class OutTask(Task):
    answer: int
    answer_str: str


class BinaryTask(Task):
    answer: Literal["Yes", "No"]


class MulticlassTask(Task):
    answer: Literal["A", "B", "C"]


class OpenTask(Task):
    answer: str


def convert_label(answer: Literal["A", "B", "C", "Yes", "No"]) -> int:
    if answer == "Yes":
        return 1
    elif answer == "No":
        return 0
    # Assuming the answer is a letter like "A", "B", "C"
    return ord(answer) - ord("A")


def get_task(task: str) -> type[Task]:
    if task == "binary":
        return BinaryTask
    elif task == "multi":
        return MulticlassTask
    else:
        return OpenTask


class QuantsBaseline(LightningModule):
    def __init__(
        self,
        num_classes: int,
        task: str = "binary",
        ts_length: int = 80,
        num_vars: int = 72,
        patch_length: int = 8,
        patch_stride: int = 8,
        d_model: int = 40,
        lr: float = 5e-3,
        config: DictConfig = DictConfig({}),
        gt_annotations: bool = False,
    ):
        super().__init__()
        self.config = PatchTSMixerConfig(
            context_length=ts_length,
            num_input_channels=num_vars,
            patch_length=patch_length,
            patch_stride=patch_stride,
            return_loss=True,
            mode="mix_channel",
            d_model=d_model,
            use_positional_encoding=True,
            positional_encoding_type="sincos",
            num_targets=num_classes,
        )
        self.ts_length = ts_length
        self.train_acc = tm.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = tm.Accuracy(task="multiclass", num_classes=num_classes)
        if task == "binary":
            self.test_metrics = tm.MetricCollection(
                [tm.Accuracy(task="binary"), tm.F1Score(task="binary")], prefix="test/"
            )
        else:
            self.test_metrics = tm.MetricCollection(
                [
                    tm.Accuracy(task="multiclass", num_classes=num_classes),
                    tm.F1Score(task="multiclass", num_classes=num_classes),
                ],
                prefix="test/",
            )
        self.action_predictor = PatchTSMixerForTimeSeriesClassification(self.config)
        self.lr = lr
        self.skipped_questions = 0
        self.task = task
        self.config = config
        self.gt_annotations = gt_annotations
        self.save_hyperparameters()

    def configure_model(self) -> None:
        if self.trainer.state.fn == "test":
            model_path = "/common-repos/LMs/Llama3_converted/Meta-Llama-3-8B-Instruct"

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.llm = transformers(
                model_path, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16}
            )  # todo hardocded cuda

    def on_test_start(self) -> None:
        fn = self.config.fn

        db_content = "\n".join(
            [
                fn.action.format(action=action) + fn.timeseries_id.format(id=id)
                for action, id in self.trainer.datamodule.action_2_idx.items()  # type: ignore
            ]
        )

        # replay_data = self.config.examples

        system_content = self.config.system.prompt
        if self.config.system.db_in_system:
            system_content = system_content.format(mapping=db_content, answer="...")
            print("SYSTEM PROMPT:", system_content)
        # else:
        #     db = {
        #     "role": "user",
        #     "content": db_content,
        #     }
        #     replay_data = [db] + replay_data

        self.make_db_template = lambda question, ids: [
            {"role": "system", "content": system_content},  # + replay_data,
            # *replay_data,
            {
                "role": "user",
                "content": fn.timeseries.format(timeseries=",".join(map(str, ids)))
                + fn.question.format(question=question),
            },
        ]

        self.generate_answer = lambda messages: gen.json(self.llm, get_task(self.task), sampler=multinomial(top_k=50,top_p=0.95,temperature=0.1))(
            cast(
                list[str], self.tokenizer.apply_chat_template(messages, tokenize=False)
            )
        )  # , add_generation_prompt=True)))
        # self.generate_answer = lambda messages: self.llm.generate(
        #     messages,
        #     max_new_tokens=256,
        #     # attention_mask=attention_mask,
        #     eos_token_id=self.terminators,
        #     do_sample=True,
        #     temperature=0.1,
        #     pad_token_id=self.tokenizer.pad_token_id,
        #     top_p=0.95,
        #     top_k=50,
        # )

    def query_and_update(self, messages, pred_answers, attempt=1, max_attempts=3):
        outputs: list[Task] = self.generate_answer(messages)  # type: ignore
        resched_indices = []

        for i, output in enumerate(outputs):
            if getattr(output, "steps", None) is not None and len(output.steps) == 0:
                # No reasoning found, schedule for re-querying
                print("No reasoning found. Skipping...")
                resched_indices.append(i)
                continue

            if isinstance(output, MulticlassTask) or isinstance(output, BinaryTask):
                label = convert_label(output.answer)
            else:
                label = 0
                raise ValueError(f"Open task not supported: {output}")

            pred_answers[i] = OutTask(
                answer=label,
                answer_str=output.answer,
                steps=output.steps,
                timeseries_actions=output.timeseries_actions,
            )

        if resched_indices and attempt < max_attempts:
            requery_messages = [messages[i] for i in resched_indices]
            self.query_and_update(
                requery_messages, pred_answers, attempt + 1, max_attempts
            )

    def test_step(self, batch):
        batch_size = batch["trajectory"].shape[0]

        if self.gt_annotations:
            messages = []
            for i in range(batch_size):
                res = []
                for j in range(len(batch["action_sequence"]["action"])):
                    id = self.trainer.datamodule.action_2_idx[batch["action_sequence"]["action"][j][i]]  # type: ignore
                    res.append(id)
                messages.append(self.make_db_template(batch["question"][i], res))

        else:
            traj = batch["trajectory"]
            traj = rearrange(
                traj,
                "b (actcnt length) channels -> (b actcnt) length channels",
                length=self.ts_length,
            )

            outs = self.action_predictor(traj).prediction_outputs
            outs = rearrange(outs, "(b actcnt) c -> b actcnt c", b=batch_size)
            messages = [
                self.make_db_template(
                    batch["question"][i], outs[i].argmax(dim=1).tolist()
                )
                for i in range(batch_size)
            ]

        pred_answers: list[OutTask | None] = [
            None
        ] * batch_size  # Initialize with None for all positions

        self.query_and_update(messages, pred_answers)

        # Filter out None values before converting to tensor
        pred_answers_filtered: list[OutTask] = [
            p for p in pred_answers if p is not None
        ]
        valid_indices = [i for i, p in enumerate(pred_answers) if p is not None]

        self.skipped_questions += batch_size - len(pred_answers_filtered)

        # Calculate accuracy using torchmetrics
        pred_answers_tensor = torch.tensor(
            list(map(lambda p: p.answer, pred_answers_filtered)),
            dtype=torch.float32,
            device=self.device,
        )

        # Get valid batch answers corresponding to valid indices
        valid_batch_answers = batch["answer"][valid_indices]
        valid_batch_messages = [messages[i] for i in valid_indices]

        # Find indices where the similarity is 0
        incorrect_indices = torch.where(pred_answers_tensor != valid_batch_answers)[0]
        print(
            json.dumps(
                [
                    {
                        "question": valid_batch_messages[i][-1]["content"],
                        "pred": pred_answers_filtered[i].answer_str,
                        "true": valid_batch_answers[i].cpu().item(),
                        "steps": pred_answers_filtered[i].steps,
                    }
                    for i in incorrect_indices.tolist()
                ],
                indent=4,
            )
        )

        metrics = self.test_metrics(pred_answers_tensor, valid_batch_answers)
        print(f"Test metrics: {metrics}")

        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)

    def training_step(self, batch, batch_idx):
        inputs = batch["trajectory"]  # [:, :seq_len, :]
        labels = batch["action_ids"]
        # inputs = inputs.transpose(1, 2)  # batch x seq x feat
        outputs: PatchTSMixerForPreTrainingOutput = self.action_predictor(
            inputs,
            labels,
        )
        self.train_acc(outputs.prediction_outputs, labels)
        loss = outputs.loss
        self.log("train/loss", loss, prog_bar=True, on_epoch=True, on_step=True) #type: ignore
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch["trajectory"]  # [:, :seq_len, :]
        labels = batch["action_ids"]
        # inputs = inputs.transpose(1, 2)  # batch x seq x feat
        outputs: PatchTSMixerForPreTrainingOutput = self.action_predictor(
            inputs,
            labels,
        )
        self.val_acc(outputs.prediction_outputs, labels)
        loss = outputs.loss
        self.log("val/loss", loss, prog_bar=True, on_epoch=True) #type: ignore
        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train/acc", self.train_acc.compute(), prog_bar=True, on_epoch=True)
        self.train_acc.reset()

        self.log("val/acc", self.val_acc.compute(), prog_bar=True, on_epoch=True)
        self.val_acc.reset()

    def configure_optimizers(self):
        return torch.optim.RAdam(self.parameters(), lr=self.lr)
