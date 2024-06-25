import os
from lightning import Trainer
import numpy as np
import torch
from ts_qa.ofa_system import OFA
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from ts_qa.linearprobing_callback import LinearProbing2Fine
from ts_qa.qa_data import TimeQADataModule, OppQADataModule
from transformers import (
    PatchTSMixerConfig,
    PatchTSMixerForPretraining,
    PatchTSMixerForTimeSeriesClassification,
    BertForSequenceClassification,
    AutoTokenizer,
    BertModel,
    BertTokenizer,
)
from transformers.models.patchtsmixer.modeling_patchtsmixer import (
    PatchTSMixerForTimeSeriesClassificationOutput,
    PatchTSMixerForPreTrainingOutput,
)
import lightning as L
from torchmetrics import MetricCollection, Accuracy


def load_wandb_secret():
    try:
        # Open the file containing your secret key
        with open("/.wandb_secret") as f:
            secret_key = f.read().split("=")[1].strip()

        # Set the environment variable
        os.environ["WANDB_API_KEY"] = secret_key
    except FileNotFoundError:
        print("The .wandb_secret file was not found.")


Item = tuple[list[str], torch.FloatTensor, torch.LongTensor]


class PreTrainModel(L.LightningModule):
    def __init__(
        self,
        num_channels: int = 1,
        patch_length: int = 2,
        context_length: int = 16,
        patch_stride: int = 1,
    ):
        super(PreTrainModel, self).__init__()
        config = PatchTSMixerConfig(
            num_input_channels=num_channels,
            patch_length=patch_length,
            context_length=context_length,
            patch_stride=patch_stride,
        )
        self.model = PatchTSMixerForPretraining(config)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: Item, batch_idx):
        _, inputs, labels = batch
        inputs = inputs.transpose(1, 2)  # batch x seq x feat
        outputs: PatchTSMixerForPreTrainingOutput = self.model(inputs, inputs.clone())
        loss = outputs.loss
        return loss

    def validation_step(self, batch: Item, batch_idx):
        _, inputs, labels = batch
        inputs = inputs.transpose(1, 2)
        outputs: PatchTSMixerForPreTrainingOutput = self.model(inputs, inputs.clone())
        loss = outputs.loss
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class FineTuneModel(L.LightningModule):
    def __init__(
        self,
        num_channels: int = 1,
        patch_length: int = 2,
        context_length: int = 16,
        num_labels: int = 2,
        patch_stride: int = 1,
    ):
        super(FineTuneModel, self).__init__()
        config = PatchTSMixerConfig(
            num_input_channels=num_channels,
            patch_length=patch_length,
            context_length=context_length,
            num_targets=num_labels,
            patch_stride=patch_stride,
        )
        self.model = PatchTSMixerForTimeSeriesClassification(config)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.text_model = BertModel.from_pretrained("bert-base-uncased")

        self.fusion_model = torch.nn.Sequential(
            torch.nn.Linear(38, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.LayerNorm(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2),
        )

        metrics = MetricCollection(
            [Accuracy(task="multiclass", num_classes=num_labels)]
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(
        self, batch: tuple[list[str], torch.FloatTensor, torch.LongTensor], batch_idx
    ):
        question, trajectory, y = batch
        trajectory = trajectory.transpose(1, 2)
        outputs: PatchTSMixerForTimeSeriesClassificationOutput = self.model(
            trajectory, y
        )

        # encoded_input = self.tokenizer(
        #     question,
        #     return_tensors="pt",
        #     padding="max_length",
        #     truncation=True,
        #     max_length=30,
        # ).to(self.device)
        output = self.text_model(**encoded_input)
        text_embedding = output.last_hidden_state.mean(dim=-1)
        time_embedding = outputs.last_hidden_state.mean(dim=(1, 2))
        time_embedding = torch.zeros_like(time_embedding)
        x = torch.cat([time_embedding, text_embedding], dim=1)

        outputs = self.fusion_model(x)

        loss = self.criterion(outputs, y)

        # loss = outputs.loss
        self.train_metrics(outputs, y)
        return loss

    def on_train_epoch_end(self):
        # log epoch metric
        self.log_dict(self.train_metrics, prog_bar=True)

    def validation_step(
        self, batch: tuple[list[str], torch.FloatTensor, torch.LongTensor], batch_idx
    ):
        question, inputs, labels = batch
        inputs = inputs.transpose(1, 2)
        outputs = self.model(inputs, labels)
        # loss = outputs.loss

        # encoded_input = self.tokenizer(
        #     question,
        #     return_tensors="pt",
        #     padding="max_length",
        #     truncation=True,
        #     max_length=30,
        # ).to(self.device)
        output = self.text_model(**encoded_input)
        text_embedding = output.last_hidden_state.mean(dim=-1)
        time_embedding = outputs.last_hidden_state.mean(dim=(1, 2))
        time_embedding = torch.zeros_like(time_embedding)

        x = torch.cat([time_embedding, text_embedding], dim=1)

        outputs = self.fusion_model(x)

        loss = self.criterion(outputs, labels)

        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.val_metrics(outputs, labels)
        return loss

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=1e-4
        )

    def load_pretrained(self, path: str):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict, strict=False)


def main():
    load_wandb_secret()
    datamodule = TimeQADataModule(name="time-qa-simple", task="binary", batch_size=64)
    datamodule.prepare_data()
    num_classes = 2  # len(datamodule.dataset["val"].features["options"])
    max_len = datamodule.max_len
    feat_dim = datamodule.feat_dim
    d_model = 768
    max_token_length = 40  # 28
    patch_size = 16
    stride = 8

    # system = PreTrainModel(num_channels=feat_dim, context_length=max_len, patch_length=16, patch_stride=8)

    logger = WandbLogger(project="time-qa", tags=["mixer", "alignment", "simple"])

    # LightningCLI(datamodule_class=TSRegressionDataModule, model_class=OFA)
    trainer = Trainer(
        max_epochs=100,
        callbacks=[
            RichProgressBar(),
            # LinearProbing2Fine(5)
            ModelCheckpoint(monitor="val/loss", save_top_k=1, mode="min"),
        ],
        logger=logger,
        # overfit_batches=10,
        # train
        # limit_val_batches=0,
        # enable_checkpointing=False,
    )

    system = FineTuneModel(
        num_channels=feat_dim,
        context_length=max_len,
        num_labels=2,
        patch_length=16,
        patch_stride=8,
    )
    # system.load_pretrained("/workspaces/ts-qa/time-qa/nqwilcdw/checkpoints/epoch=91-step=644.ckpt")
    for param in system.model.model.parameters():
        param.requires_grad = False
    for param in system.text_model.embeddings.parameters():
        param.requires_grad = False
    # for param in system.text_model.encoder.parameters():
    #     param.requires_grad = False

    # for param in system.tokenizer.parameters():
    # param.requires_grad = False

    trainer.fit(system, datamodule=datamodule)


if __name__ == "__main__":
    main()
