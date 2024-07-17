from pathlib import Path
from typing import cast
from einops import rearrange
from lightning import LightningModule
from omegaconf import DictConfig
import torch
from transformers import PatchTSMixerConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.models.patchtsmixer.modeling_patchtsmixer import (
    PatchTSMixerForPreTrainingOutput,
    PatchTSMixerForTimeSeriesClassification,
)
import lightning as L
import torchmetrics as tm
import json
import re


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
        config: DictConfig =DictConfig({}),
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
            self.test_metrics = tm.MetricCollection([tm.Accuracy(task="binary"), tm.F1Score(task="binary")],prefix="test/")
        else:
            self.test_metrics = tm.MetricCollection([tm.Accuracy(task="multiclass", num_classes=num_classes), tm.F1Score(task="multiclass", num_classes=num_classes)],prefix="test/")
        self.action_predictor = PatchTSMixerForTimeSeriesClassification(self.config)
        self.lr = lr
        self.skipped_questions = 0
        self.task = task
        self.config = config
        self.save_hyperparameters()

    def configure_model(self) -> None:
        if self.trainer.state.fn == "test":
            model_path = "/common-repos/LMs/Llama3_converted/Meta-Llama-3-8B-Instruct"

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

            self.tokenizer.padding_side = "left"

            self.terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ]
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def on_test_start(self) -> None:
        fn = self.config.fn

        db_content = "\n".join([
                    fn.action.format(action=action) + fn.timeseries_id.format(id=id)
                    for action, id in self.trainer.datamodule.action_2_idx.items()  # type: ignore
                ])

        replay_data = self.config.examples
        
        system_content = self.config.system.prompt + self.config.system[self.task].addition
        if self.config.system.db_in_system:
            system_content = system_content.format(mapping=db_content, answer="...")
            print("SYSTEM PROMPT:\n", system_content)
        else:
            db = {
            "role": "user",
            "content": db_content,
            }
            replay_data = [db] + replay_data

        self.make_db_template = lambda question, ids: [
            {
                "role": "system",
                "content": system_content,
            },
            *replay_data,
            {
                "role": "user",
                "content": fn.timeseries.format(timeseries=','.join(map(str, ids))) + fn.question.format(question=question)
            },
        ]


        self.generate_answer = lambda messages: self.llm.generate(
            messages,
            max_new_tokens=256,
            # attention_mask=attention_mask,
            eos_token_id=self.terminators,
            do_sample=True,
            temperature=0.1,
            pad_token_id=self.tokenizer.pad_token_id,
            top_p=0.95,
            top_k=50,
        )

   
    def test_step(self, batch):
        batch_size = batch["trajectory"].shape[0]
        traj = batch["trajectory"]
        traj = rearrange(
            traj,
            "b (actcnt length) channels -> (b actcnt) length channels",
            length=self.ts_length,
        )

        outs = self.action_predictor(traj).prediction_outputs
        outs = rearrange(outs, "(b actcnt) c -> b actcnt c", b=batch_size)

        all_explanations = []
        pred_answers = []

        messages = [
            self.make_db_template(batch["question"][i], outs[i].argmax(dim=1).tolist())
            for i in range(batch_size)
        ]
        template_messages = cast(
            list[str],
            self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            ),
        )
        input_ids = self.tokenizer(
            template_messages, padding=True, return_tensors="pt"
        )["input_ids"]
        input_ids = input_ids.to(self.device)  # type: ignore

        outputs = self.generate_answer(input_ids)


        def construct_answer_pattern(template):
            # Escape special regex characters in the template string
            escaped_template = re.escape(template)
            # Replace the escaped placeholder with a regex pattern to capture content
            pattern = escaped_template.replace(r"\{answer\}", r"(.*?)")
            return pattern

        # Construct the pattern from the template string
        pattern = construct_answer_pattern(self.config.fn.answer)

        

        valid_indices = []
        valid_messages = []
        for i in range(batch_size):
            response = outputs[i][input_ids.shape[-1] :]
            resp_str = self.tokenizer.decode(response, skip_special_tokens=True)

            # Search for the pattern in the input string
            match = re.search(pattern, resp_str)

            if match:
                # Extract the matched content
                pred_answer_str = match.group(1)
            else:
                print(resp_str)
                print("No answer found in response. Skipping...")
                self.skipped_questions += 1
                continue 

            # pred_answer_str = resp_str.split(self.config.tags.answer.start)[1].split(self.config.tags.answer.end)[0]
            # if "[" in pred_answer_str:
            #     pred_answer_str = pred_answer_str.split("[")[1].split("]")[0]
            
            if pred_answer_str in ["0","1","2"]:
                pred_answer_str = pred_answer_str
            elif "A" in pred_answer_str:
                pred_answer_str = "0"
            elif "B" in pred_answer_str:
                pred_answer_str = "1"
            elif "C" in pred_answer_str:
                pred_answer_str = "2"
            else:
                print(resp_str)
                print("No answer found in response. Skipping...")
                self.skipped_questions += 1
                continue

            pred_answer = int(pred_answer_str)

            pred_answers.append(pred_answer)
            valid_indices.append(i)
            valid_messages.append(messages[i])

        # Calculate accuracy using torchmetrics
        pred_answers_tensor = torch.tensor(
            pred_answers, dtype=torch.float32, device=self.device
        )

        valid_batch_answers = batch["answer"][valid_indices]


        # find indices where the similarity is 0
        incorrect_indices = torch.where(pred_answers_tensor != valid_batch_answers)[0]
        print(
            json.dumps([{"question": valid_messages[i][-1]["content"], "pred": pred_answers_tensor[i].cpu().item(), "true": valid_batch_answers[i].cpu().item()} for i in incorrect_indices.tolist()], indent=4)
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
        self.log("train/loss", loss, prog_bar=True, on_epoch=True, on_step=True)
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
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train/acc", self.train_acc.compute(), prog_bar=True, on_epoch=True)
        self.train_acc.reset()

        self.log("val/acc", self.val_acc.compute(), prog_bar=True, on_epoch=True)
        self.val_acc.reset()

    def configure_optimizers(self):
        return torch.optim.RAdam(self.parameters(), lr=self.lr)
