import os
from typing import Literal, cast
from lightning import Trainer, LightningDataModule
from ts_qa.ofa_system import OFA
from lightning.pytorch.cli import LightningCLI
from ts_qa.ofa_data import TSRegressionDataModule
from lightning.pytorch.callbacks import RichProgressBar, RichModelSummary
from datasets import load_dataset, VerificationMode, DatasetDict, Dataset as HFDataset
from torch.utils.data import DataLoader
from json import loads as jloads
from torch.utils.data import TensorDataset, StackDataset, Dataset
import torch
from einops import rearrange
from lightning.pytorch.loggers import WandbLogger

def load_wandb_secret():
    try:
        # Open the file containing your secret key
        with open("/.wandb_secret") as f:
            secret_key = f.read().split("=")[1].strip()

        # Set the environment variable
        os.environ["WANDB_API_KEY"] = secret_key
    except FileNotFoundError:
        print("The .wandb_secret file was not found.")


class TimeQADataset(Dataset):

    def __init__(self, hf_dataset: HFDataset):
        self.hf_dataset = hf_dataset

    
    def __len__(self):
        return len(self.hf_dataset)
    

    def __getitem__(self, idx):
        elem = self.hf_dataset[idx]
        question,trajectory,label = elem["question"],elem["trajectory"],  elem["answer"] #type: ignore

        trajectory =  rearrange(trajectory, "joint xyz len -> (joint xyz) len")

        return question,trajectory,label

        




class TimeQADataModule(LightningDataModule):
    KEY = "dasyd/time-qa"

    def __init__(
        self,
        batch_size: int = 32,
        task: Literal["binary", "multi", "open"] = "binary",
    ):
        super().__init__()

        self.batch_size = batch_size
        self.task = task

    def _load_dataset_split(self, splits: list[str]):
        """Workaround to overcome the missing hf implementation of only dowloading the split shards"""
        files = {}
        for split in splits:
            files[split] = f"{split}-*"
        return load_dataset(
            TimeQADataModule.KEY,
            self.task,
            data_dir=self.task,
            data_files=files,
            verification_mode=VerificationMode.NO_CHECKS,
            num_proc=len(splits),
        )

    def prepare_data(self) -> None:
        self.dataset = self._load_dataset_split(["val", "train"])

    @property
    def __trajectory_shape(self) -> torch.Size:
        assert self.dataset is not None, "You need to run prepare first"

        self.dataset = cast(DatasetDict, self.dataset.with_format("torch"))
        trajectory = cast(torch.Tensor, self.dataset["train"]["trajectory"])

        return trajectory.shape

    @property
    def max_len(self) -> int:

        return self.__trajectory_shape[-1]

    @property
    def feat_dim(self) -> int:
        return int(torch.mul(*self.__trajectory_shape[1:3]))

    def setup(self, stage: str) -> None:
        if stage == "fit":
            dataset = self._load_dataset_split(["train", "val"])
            dataset = cast(DatasetDict, dataset.with_format("torch"))
            # label_dict = jloads(dataset["train"][0]["options"])
            self.train_dataset = TimeQADataset(dataset["train"])
            self.val_dataset = TimeQADataset(dataset["val"])

            
        elif stage == "test":
            dataset = self._load_dataset_split(["test"])
            dataset = cast(DatasetDict, dataset.with_format("torch"))
            self.test_dataset = TimeQADataset(dataset["test"])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=40
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=40)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


def main():
    load_wandb_secret()
    datamodule = TimeQADataModule(task="binary", batch_size=500)
    datamodule.prepare_data()
    # num_classes = len(datamodule.dataset["val"].features["options"])
    max_len = datamodule.max_len
    feat_dim = datamodule.feat_dim
    d_model = 768
    max_token_length = 28

    model = OFA(max_token_length=max_token_length, num_classes=2 if datamodule.task == "binary" else 3, max_seq_len=max_len, patch_size=16, stride=8, dropout=0.1, d_model=d_model, feat_dim=feat_dim)
    logger = WandbLogger(project="time-qa", tags=["ofa"])
    # LightningCLI(datamodule_class=TSRegressionDataModule, model_class=OFA)
    trainer = Trainer(max_epochs=50, callbacks=[RichProgressBar(), RichModelSummary()], logger=logger)

    # datamodule = TSRegressionDataModule(name="Heartbeat", batch_size=64)
    # datamodule.prepare_data()
    # model = OFA(num_classes=len(datamodule.meta["class_values"]), max_seq_len=datamodule.max_seq_len, patch_size=16, stride=8, dropout=0.1, d_model=768, feat_dim=datamodule.feat_dim)

    # trainer.test(model, datamodule)
    trainer.fit(model, datamodule)
    # trainer.test(model, datamodule)


if __name__ == "__main__":
    main()
