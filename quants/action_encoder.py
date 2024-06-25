from einops import rearrange
from lightning import LightningModule
import torch
from transformers import PatchTSMixerConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.models.patchtsmixer.modeling_patchtsmixer import (
    PatchTSMixerForPreTrainingOutput,
    PatchTSMixerForTimeSeriesClassification,
)
import lightning as L
from aim.pytorch_lightning import AimLogger
import torchmetrics as tm


class QuantsBaseline(LightningModule):
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
        self.train_acc = tm.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = tm.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = tm.Accuracy(task="binary")
        self.action_predictor = PatchTSMixerForTimeSeriesClassification(self.config)
        self.lr = lr
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

            self.terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ]
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def on_test_start(self) -> None:

        db = {
            "role": "user",
            "content": "\n".join(
                [
                    f"[Action]{action}[\\Action][TimeseriesID]{id}[\\TimeseriesID]"
                    for action, id in self.trainer.datamodule.action_2_idx.items()  # type: ignore
                ]
            ),
        }

        example_query = {
            "role": "user",
            "content": f"[Timeseries]9,8,10,11[\\Timeseries][Question]Is the person waving before dancing?[\\Question]",
        }
        example_answer = {"role": "assistant", "content": "[Answer]Yes[\\Answer]"}

        self.make_db_template = lambda question, ids: [
            {
                "role": "system",
                "content": (
                    "You are a multimodal time series question answering model"
                    " designed to analyze sequences of actions and respond to queries"
                    " about them. Your input will be provided in two specific tags: "
                    "[Action] and [TimeseriesID]. The [Action] tags describe different"
                    " activities, each associated with a unique [TimeseriesID]."
                    " When answering questions, refer to these IDs to understand the"
                    " sequence and nature of the actions in the [Timeseries]...[\\Timeseries]"
                    " tags. Responses to questions should be enclosed in [Answer]...[\\Answer] tags."
                    " If an explanation is necessary, enclose it in [Explanation]...[\\Explanation]"
                    " tags. Your responses should focus on interpreting the action"
                    " sequences accurately and providing clear answers or explanations as required."
                    "Please either return '0' for False or '1' for True"
                    if self.trainer.datamodule.task == "binary" # type: ignore
                    else ""
                ),
            },
            db,
            example_query,
            example_answer,
            {
                "role": "user",
                "content": f"[Timeseries]{ids}[\\Timeseries][Question]{question}[\\Question]",
            },
        ]


        # # Ensure pad_token_id is set
        # if self.llm.config.pad_token_id is None:
        #     self.llm.config.pad_token_id = self.tokenizer.eos_token_id


        



        self.generate_answer = lambda messages: self.llm.generate(
            messages,
            max_new_tokens=64,
            # attention_mask=attention_mask,
            eos_token_id=self.terminators,
            # do_sample=True,
            temperature=0.6,
            pad_token_id=self.tokenizer.pad_token_id
            # top_p=0.9,
        )

    def eval_text(self, pred, reference):
        make_eval_template = lambda pred, gt: [
            {
                "role": "system",
                "content": "You are an evaluation model tasked with assessing the similarity of predicted answers compared to reference answers. For each evaluation, present the predicted answer text within the [Prediction] tag and the reference answer text within the [Reference] tag. Determine if the predicted and reference answers are equivalent. Return '1' for equivalent answers and '0' for non-equivalent answers, enclosed in the [Answer] tag. Additionally, provide your assessment of similarity for each pair in the [Similarity] tag, expressed as a percentage. Only the [Answer] and [Similarity] tags should be returned; no further explanation is required.",
            },
            {
                "role": "user",
                "content": f"[Prediction]{pred}[\\Prediction][Reference]{gt}[\\Reference]",
            },
        ]

        eval_messages = make_eval_template(pred, reference)

        input_ids = self.tokenizer.apply_chat_template(
            eval_messages, add_generation_prompt=True, return_tensors="pt"
        ).to(  # type: ignore
            self.device
        )

        outputs = self.generate_answer(input_ids)

        response = outputs[0][input_ids.shape[-1] :]
        resp_str = self.tokenizer.decode(response, skip_special_tokens=True)

        # parse the response, the answer is between the [Answer]...[\Answer] tags and a potential explanation is between the [Explanation]...[\Explanation] tags
        pred_answer = int(resp_str.split("[Answer]")[1].split("[\\Answer]")[0])

        return pred_answer

    def test_step(self, batch):
        batch_size = batch["trajectory"].shape[0]
        traj = batch["trajectory"]
        traj = rearrange(
            traj,
            "b (actcnt length) channels -> (b actcnt) length (channels)",
            length=self.ts_length,
        )

        outs = self.action_predictor(traj).prediction_outputs
        outs = rearrange(outs, "(b actcnt) c -> b actcnt c", b=batch_size)

        all_explanations = []
        all_similarities = []

        for i in range(batch_size):
            messages = self.make_db_template(
                batch["question"][i], outs[i].argmax(dim=1).tolist()
            )
            input_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(  # type: ignore
                self.device
            )

            # Create attention mask
            # attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
            outputs = self.generate_answer(input_ids)

            response = outputs[0][input_ids.shape[-1] :]
            resp_str = self.tokenizer.decode(response, skip_special_tokens=True)
            pred_answer = resp_str.split("[Answer]")[1].split("[\\Answer]")[0]

            if "[Explanation]" in resp_str:
                explanation = resp_str.split("[Explanation]")[1].split(
                    "[\\Explanation]"
                )[0]
            else:
                explanation = None

            similar = int(pred_answer)
            # similar = self.eval_text(pred_answer, batch["answer"][i])

            # all_explanations.append(explanation)
            all_similarities.append(similar)

        # Calculate accuracy using torchmetrics
        similarities_tensor = torch.tensor(all_similarities, dtype=torch.float32)
        targets_tensor = torch.ones_like(
            similarities_tensor
        )  # Assuming targets are always 1 for correct answers

        self.test_acc(similarities_tensor, targets_tensor)

        self.log("test/acc", self.test_acc, on_step=True, on_epoch=True, prog_bar=True)

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
