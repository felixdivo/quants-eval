from enum import Enum
from typing import cast
from einops import rearrange
from lightning import LightningModule
import torch
from transformers import PatchTSMixerConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.models.patchtsmixer.modeling_patchtsmixer import (
    PatchTSMixerForPreTrainingOutput,
    PatchTSMixerForTimeSeriesClassification,
)
from pydantic_core import from_json
import lightning as L
import torchmetrics as tm
from torchmetrics.text import BERTScore, BLEUScore, SacreBLEUScore, ROUGEScore
import json
import re
from pydantic import BaseModel
import sglang as sgl
from sglang.lang.ir import SglConstantText
import wandb
import numpy as np


class ActionNames(Enum):
    HOLDING_A_BABY = "holding a baby"
    SHAKING_HANDS = "shaking hands"
    RUNNING = "running"
    JUMPING_ONCE = "jumping once"
    PUNCHING = "punching"
    GOLFING_SWINGING_A_CLUB = "golfing (swinging a club)"
    DRINKING_WITH_THE_LEFT_HAND = "drinking with the left hand"
    SKIPPING_ROPE = "skipping rope"
    DANCING = "dancing"
    WAVING = "waving"
    PLAYING_GUITAR = "playing guitar"
    BOWING = "bowing"
    KICKING_A_BALL = "kicking a ball"
    THROWING_A_BALL = "throwing a ball"
    T_POSING = "T-posing"
    CATCHING_A_BALL = "catching a ball"
    PICKING_SOMETHING_UP_WITH_BOTH_HANDS = "picking something up with both hands"
    SITTING_DOWN = "sitting down"
    EATING_WITH_THE_RIGHT_HAND = "eating with the right hand"

    @classmethod
    def from_id(cls, id: int):
        return list(cls)[id].value


class QuantsSegmentationBaseline(LightningModule):
    def __init__(
        self,
        num_classes: int,
        ts_length: int = 80,
        num_vars: int = 72,
        patch_length: int = 8,
        patch_stride: int = 8,
        d_model: int = 40,
        lr: float = 5e-3,
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
        self.train_acc = tm.MetricCollection(
            [
                tm.Accuracy(
                    task="multiclass",
                    num_classes=num_classes,
                )
            ],
            prefix="train/",
        )
        self.val_acc = tm.MetricCollection(
            [
                tm.Accuracy(
                    task="multiclass",
                    num_classes=num_classes,
                )
            ],
            prefix="val/",
        )
        self.test_metrics = tm.MetricCollection(
            [
                tm.Accuracy(task="multiclass", num_classes=num_classes),
                tm.F1Score(task="multiclass", num_classes=num_classes),
            ],
            prefix="test/",
        )

        self.action_predictor = PatchTSMixerForTimeSeriesClassification(self.config)
        self.lr = lr
        self.save_hyperparameters()

        self.train_all_labels = []
        self.val_all_labels = []

    def test_step(self, batch):
        batch_size = batch["trajectory"].shape[0]

        gt_messages = []
        ress = []
        for i in range(batch_size):
            res = []
            acts = []
            for j in range(len(batch["action_sequence"]["action"])):
                id = self.trainer.datamodule.action_2_idx[batch["action_sequence"]["action"][j][i]]  # type: ignore
                res.append(id)
                acts.append(batch["action_sequence"]["action"][j][i])
            gt_messages.append({"timeseries": res, "acts": acts})
            ress.append(res)

        ress = torch.tensor(ress, dtype=torch.long, device=self.device)

        traj = batch["trajectory"]
        traj = rearrange(
            traj,
            "b (actcnt length) channels -> (b actcnt) length channels",
            length=self.ts_length,
        )

        outs = self.action_predictor(traj).prediction_outputs
        outs = rearrange(outs, "(b actcnt) c -> b c actcnt", b=batch_size)
        met = self.test_metrics(outs, ress)
        self.log_dict(met, on_step=True, on_epoch=True, prog_bar=True)

    def log_label_distribution(self, prefix: str):
        # Log label distribution to wandb as a histogram
        hist = np.histogram(
            getattr(self, f"{prefix}_all_labels"), bins=np.arange(19+1), density=True
        )
        df_hist = np.stack([hist[1][:-1], hist[0]], axis=1)
        table = wandb.Table(data=df_hist, columns=["label", "frequency"])
        wandb.log(
            {
                f"{prefix}/label_distribution_epoch": wandb.plot.histogram(
                    table, "label"
                )
            }
        )

    def training_step(self, batch, batch_idx):
        inputs = batch["trajectory"]  # [:, :seq_len, :]
        labels = batch["action_ids"]

        self.train_all_labels.extend(labels.cpu().numpy())

        # inputs = inputs.transpose(1, 2)  # batch x seq x feat
        outputs: PatchTSMixerForPreTrainingOutput = self.action_predictor(
            inputs,
            labels,
        )
        met = self.train_acc(outputs.prediction_outputs, labels)
        loss = outputs.loss
        self.log_dict(met, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss", loss, prog_bar=True, on_epoch=True, on_step=True)  # type: ignore
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch["trajectory"]  # [:, :seq_len, :]
        labels = batch["action_ids"]
        self.val_all_labels.extend(labels.cpu().numpy())

        # inputs = inputs.transpose(1, 2)  # batch x seq x feat
        outputs: PatchTSMixerForPreTrainingOutput = self.action_predictor(
            inputs,
            labels,
        )
        met = self.val_acc(outputs.prediction_outputs, labels)
        loss = outputs.loss
        self.log_dict(met, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)  # type: ignore
        return loss

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            self.val_all_labels.clear()
        else:
            self.log_label_distribution("val")
            self.val_all_labels.clear()

    def on_train_epoch_end(self):
        self.log_label_distribution("train")
        self.train_all_labels.clear()

    def configure_optimizers(self):
        return torch.optim.RAdam(self.parameters(), lr=self.lr)

class Task(BaseModel):
    steps: list[str]
    actions: list[str]
    answer: str

    @classmethod
    def regex(cls):
        return r""

class BinaryTask(Task):

    def parse_answer(self) -> int:
        if self.answer == "true":
            return 1
        elif self.answer == "false":
            return 0
        else:
            print(f"Unknown answer: {self.answer}")
            return 0

    @classmethod
    def regex(cls):
        acts = "|".join([re.escape(action.value) for action in ActionNames])
        return (
            r"\{\n"
            + r'\t"actions":\s*\[\s*"('
            + acts
            + r')"(?:,\s*"('
            + acts
            + r')"){3}\s*\]\,\n'
            + r'\t"steps":\s*\[\s*"\d+\.\s[\w\d\s]{1,}"(?:,\s*"\d+\.\s[\w\d\s]{1,}"){1,}\s*\]\,\n'
            + r'\t"answer":\s*"(true|false)"\s*'
            + r"\}"
        )


class MultiTask(Task):

    def parse_answer(self) -> int:
        return ord(self.answer) - ord("A")

    @classmethod
    def regex(cls):
        acts = "|".join([re.escape(action.value) for action in ActionNames])
        return (
            r"\{\n"
            + r'\t"actions":\s*\[\s*"('
            + acts
            + r')"(?:,\s*"('
            + acts
            + r')"){3}\s*\]\,\n'
            + r'\t"steps":\s*\[\s*"\d+\.\s[\w\d\s]{1,}"(?:,\s*"\d+\.\s[\w\d\s]{1,}"){1,}\s*\]\,\n'
            + r'\t"answer":\s*"(A|B|C)"\s*'
            + r"\}"
        )
    
class OpenTask(BaseModel):
    steps: list[str]
    actions: list[str]
    answer: str


class QuantsBaseline(QuantsSegmentationBaseline):

    def __init__(
        self,
        task: str = "binary",
        ts_length: int = 80,
        num_vars: int = 72,
        patch_length: int = 8,
        patch_stride: int = 8,
        d_model: int = 40,
        max_retries: int = 3,
        gt_annotations: bool = False,
    ):
        num_classes = 2 if task == "binary" else 3
        super().__init__(
            num_classes=num_classes,
            ts_length=ts_length,
            num_vars=num_vars,
            patch_length=patch_length,
            patch_stride=patch_stride,
            d_model=d_model,
        )
        if task == "binary":
            self.test_metrics = tm.MetricCollection(
                [tm.Accuracy(task="binary"), tm.F1Score(task="binary")], prefix="test/"
            )
        elif task == "multi":
            self.test_metrics = tm.MetricCollection(
                [
                    tm.Accuracy(task="multiclass", num_classes=num_classes),
                    tm.F1Score(task="multiclass", num_classes=num_classes),
                ],
                prefix="test/",
            )
        else:
            self.test_metrics = tm.MetricCollection(
                [
                    tm
                ],
                prefix="test/",
            )
        self.skipped_questions = 0
        self.task = task
        self.gt_annotations = gt_annotations
        self.max_retries = max_retries
        self.save_hyperparameters()

    def configure_model(self) -> None:
        if self.trainer.state.fn == "test":
            endpoint = sgl.RuntimeEndpoint("http://localhost:30000")

            sgl.set_default_backend(endpoint)

            base_text = (
                "You are a helpful timeseries question answering model.\n"
                + "You are given a timeseries of actions and a question about the actions.\n"
                + "## TS to Action Mapping\n"
                + "\n".join(
                    [f"{idx}: {action.value}" for idx, action in enumerate(ActionNames)]
                )
                + "\n\n"
                + "If an action appears multiple times in a row in the timeseries, it is only counted once, e.g. [2,2,2,15] would mean the person is running 1 time and also catching a ball 1 time.\n"
                + "The same is valid for action occurring in the middle, e.g. [9,2,2,15] would mean the person is waving then running 1 time and afterwards catching a ball 1 time.\n"
                + 'Please translate the timeseries (TS) first into a list of actions and afterward answer the question. Think step by step and provide your thought process in "steps: [...]"\n'
                + "\n\n"
            )

            if self.task == "binary":
                self.sys_prompt = SglConstantText(
                    base_text
                    + "## Example 1\n"
                    + "TS: [2, 11, 3, 10], QS: Do the person's actions stay the same before and after they are jumping once?\n"
                    + "{\n"
                    + '\t"actions": ["running", "bowing", "jumping once", "playing guitar"],\n'
                    + '\t"steps": [\n'
                    + '\t\t"1. First, we convert the timeseries to a list of actions.",\n'
                    + '\t\t"2. For each action in the list, we use the TS to Action Mapping to translate the action number to the corresponding action.",\n'
                    + '\t\t"3. We need to identify the position of the action which corresponds to jumping once. This is index 2.",\n'
                    + '\t\t"4. Now we need to look before and after index 2, thus index 1 and index 3.",\n'
                    + '\t\t"5. At actions[1] the action is bowing and actions[3] the action is playing guitar.",\n'
                    + "\t\t\"6. Because these two actions are different, we can conclude that the person's actions don't stay the same before and after they are jumping once, and the answer is false.\"\n"
                    + "\t],\n"
                    + '\t"answer": "false"\n'
                    + "}\n"
                    "## Example 2\n"
                    + "TS: [15,2,15,2], QS: Is the person conducting chatching a ball the same amount as running?\n"
                    + "{\n"
                    + '\t"actions": ["catching a ball", "running", "catching a ball", "running" ],\n'
                    + '\t"steps": [\n'
                    + '\t\t"1. First, we convert the timeseries to a list of actions.",\n'
                    + '\t\t"2. For each action in the list, we use the TS to Action Mapping to translate the action number to the corresponding action.",\n'
                    + '\t\t"3. Let\'s count catching a ball. Catching a ball occurs at index 0 and 2, thus 2 times.",\n'
                    + '\t\t"4. Let\'s count running. Running occurs at index 1 and 3, thus 2 times.",\n'
                    + '\t\t"5. Because both actions occur two times, we can conclude that the person\'s is conducting both actions the same amount. Therfore, the answer is true."\n'
                    + "\t],\n"
                    + '\t"answer": "true"\n'
                    + "}\n"
                )
            else:
                self.sys_prompt = SglConstantText(
                    base_text
                    + "## Example 1\n"
                    + f"TS: [2,3,1,3], QS: Which activity does the person carry out 1 times? A: {ActionNames.from_id(3)}, B: {ActionNames.from_id(1)}, C: {ActionNames.from_id(5)}\n"
                    + "{\n"
                    + f'\t"actions": ["{ActionNames.from_id(2)}", "{ActionNames.from_id(3)}", "{ActionNames.from_id(1)}", "{ActionNames.from_id(3)}"],\n'
                    + '\t"steps": [\n'
                    + '\t\t"1. Convert the timeseries into a list of actions.",\n'
                    + '\t\t"2. For each action in the list, we use the TS to Action Mapping to translate the action number to the corresponding action.",\n'
                    + f'\t\t"3. Identify the actions that are performed only once. From the timeseries we see that {ActionNames.from_id(2)} and {ActionNames.from_id(1)} occur only once.",\n'
                    + f'\t\t"4. Let\'s check the possible answers (A,B,C). From the possible found canditates, we see that {ActionNames.from_id(1)} is the only action that is performed only once and is in the list of possible answers, i.e., B.",\n'
                    + '\t\t"5. Therefore, the answer is **B**."\n'
                    + "\t],\n"
                    + '\t"answer": "B"\n'
                    + "}\n"
                    + "## Example 2\n"
                    + f"TS: [5,5,1,2], QS: Which activity does the person carry out 0 times? A: {ActionNames.from_id(5)}, B: {ActionNames.from_id(1)}, C: {ActionNames.from_id(4)}\n"
                    + "{\n"
                    + f'\t"actions": ["{ActionNames.from_id(5)}", "{ActionNames.from_id(5)}", "{ActionNames.from_id(1)}", "{ActionNames.from_id(2)}"],\n'
                    + '\t"steps": [\n'
                    + '\t\t"1. Let\'s convert the timeseries into a list of actions.",\n'
                    + '\t\t"2. For each action in the list, we use the TS to Action Mapping to translate the action number to the corresponding action.",\n'
                    + '\t\t"3. Let\'s go through the timeseries and answers step by step and rule out the actions that are performed greater than 0 times.",\n'
                    + f'\t\t"4. The first action 5 ({ActionNames.from_id(5)}) occurs 1 time (the same action following each other is counted as 1), thus we can rule out answer option A: {ActionNames.from_id(5)}.\n'
                    + f'\t\t"5. The next answer option is 1 ({ActionNames.from_id(1)}), which occurs at index 2, thus 1 time. We can rule out answer option B: {ActionNames.from_id(1)}.\n'
                    + f'\t\t"6. The last answer option is 4 ({ActionNames.from_id(4)}), which does not occur in the timeseries. Therefore, the answer is **C**."\n'
                    + "\t],\n"
                    + '\t"answer": "C"\n'
                    + "}\n"
                )

    def on_test_start(self) -> None:
        def _qa_time(s, timeseries: list[int], question: str):

            s += sgl.system(self.sys_prompt)  # type: ignore
            s += (
                sgl.user_begin()
                + "TS: "
                + f'"{timeseries}"'
                + ", QS: "
                + question
                + "\n"
                + sgl.user_end()
            )
            s += sgl.assistant(
                sgl.gen(
                    "json_output",
                    max_tokens=1024,
                    temperature=0,
                    regex=(
                        MultiTask.regex()
                        if self.task == "multi"
                        else BinaryTask.regex()
                    ),
                )
            )

        self.qa_time = sgl.function(_qa_time)

        res = self.qa_time.run(
            timeseries="[2,3,4,3]", question="Is the person running?"
        )
        if self.task == "binary":

            res = BinaryTask.model_validate(
                from_json(res["json_output"], allow_partial=True)
            )
        else:
            res = MultiTask.model_validate(
                from_json(res["json_output"], allow_partial=True)
            )

    def validate_with_retries(self, message, original_res):
        for attempt in range(self.max_retries):
            try:
                # Attempt to validate the original response
                if self.task == "multi":
                    return MultiTask.model_validate(
                        from_json(original_res["json_output"])
                    )
                else:
                    return BinaryTask.model_validate(
                        from_json(original_res["json_output"])
                    )
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(
                        f"Validation failed for message {message}. Retrying... (Attempt {attempt + 1}/{self.max_retries})"
                    )
                    # Retry the single message
                    original_res = self.qa_time.run_batch([message])[0]
                else:
                    print(
                        f"Validation failed for message {message}. Reached maximum retries. Error: {e}\nMessage: {message}"
                    )
                    self.log("test/invalid_response", 1)
                    if self.task == "multi":
                        return MultiTask(
                            steps=["1. Broken"],
                            actions=["running", "running", "running", "running"],
                            answer="A",
                        )
                    else:
                        return BinaryTask(
                            steps=["1. Broken"],
                            actions=["running", "running", "running", "running"],
                            answer="false",
                        )  # should never reach this
                    # raise e# Re-raise the last exception if maximum retries reached
        if self.task == "multi":
            return MultiTask(
                steps=["1. Broken"],
                actions=["running", "running", "running", "running"],
                answer="A",
            )
        return BinaryTask(
            steps=[], actions=[], answer="false"
        )  # should never reach this

    def test_step(self, batch):
        batch_size = batch["trajectory"].shape[0]

        gt_messages = []
        for i in range(batch_size):
            res = []
            acts = []
            for j in range(len(batch["action_sequence"]["action"])):
                id = self.trainer.datamodule.action_2_idx[batch["action_sequence"]["action"][j][i]]  # type: ignore
                res.append(id)
                acts.append(batch["action_sequence"]["action"][j][i])
            gt_messages.append(
                {"question": batch["question"][i], "timeseries": res, "acts": acts}
            )

        if self.gt_annotations:
            messages = [
                {"question": m["question"], "timeseries": m["timeseries"]}
                for m in gt_messages
            ]
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
                {
                    "question": batch["question"][i],
                    "timeseries": outs[i].argmax(dim=1).tolist(),
                }
                for i in range(batch_size)
            ]

        # Run batch for all messages initially
        initial_responses = self.qa_time.run_batch(messages)

        # Validate responses and retry failed ones
        pred_answers = [
            self.validate_with_retries(message, res)
            for message, res in zip(messages, initial_responses)
        ]

        pred_answers_tensor = torch.tensor(
            list(map(lambda p: p.parse_answer(), pred_answers)),
            dtype=torch.float32,
            device=self.device,
        )

        # Find indices where the similarity is 0
        incorrect_indices = torch.where(pred_answers_tensor != batch["answer"])[0]

        def get_answer_str(answer):
            if self.task == "binary":
                return "true" if answer == 1 else "false"
            else:
                return chr(answer + ord("A"))

        print(
            json.dumps(
                [
                    {
                        "question": gt_messages[i]["question"],
                        "gt": {
                            "timeseries": gt_messages[i]["timeseries"],
                            "actions": gt_messages[i]["acts"],
                            "answer": (get_answer_str(batch["answer"][i].cpu().item())),
                        },
                        "pred": {
                            "timeseries": messages[i]["timeseries"],
                            "actions": pred_answers[i].actions,
                            "answer": pred_answers[i].answer,
                        },
                        "steps": pred_answers[i].steps,
                    }
                    for i in incorrect_indices.tolist()
                ],
                indent=4,
            )
        )

        metrics = self.test_metrics(pred_answers_tensor, batch["answer"])
        print(f"Test metrics: {metrics}")

        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)
