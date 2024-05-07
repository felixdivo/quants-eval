import os
from typing import Literal, cast
from lightning import Trainer, LightningDataModule
import numpy as np
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
from sklearn.preprocessing import MinMaxScaler
from ts_qa.linearprobing_callback import LinearProbing2Fine


channels = [
    "pelvis_x",
    "left_hip_x", 
    "right_hip_x",
    "spine1",  
    "left_knee_x",
    "right_knee_x" ,
    "spine2_x", 
    "left_ankle",
    "right_ankle",
    "spine3_x",
    "left_foot_x",
    "right_foot_x",
    "neck_x",
    "left_collar_x",
    "right_collar_x",
    "head",
    "left_shoulder_x",
    "right_shoulder_x",
    "left_elbow", 
    "right_elbow", 
    "left_wrist",
    "right_wrist",
]

indices = [i for i, channel in enumerate(channels) if "_x" not in channel]





class Scaler:

    def __init__(self) -> None:
        self.__scaler = MinMaxScaler()

    def fit(self, data: torch.Tensor) -> None:
        _data = rearrange(data, "b s f -> (b s) f")
        self.__scaler.fit(_data.numpy())

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        _data = rearrange(data, "b s f -> (b s) f")
        _normalized = self.__scaler.transform(_data.numpy())
        return torch.from_numpy(
            rearrange(_normalized, "(b s) f -> b s f", b=data.shape[0], s=data.shape[1])
        )

    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        self.fit(data)
        return self.transform(data)
    


class DummyDataModule(Dataset):

    def __init__(self) -> None:
        super().__init__()



class TimeQADataset(Dataset):

    def __init__(self, hf_dataset: HFDataset, scaler: Scaler):
        self.hf_dataset = hf_dataset
        self.scaler = scaler

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        elem = self.hf_dataset[idx]
        question, trajectory, label = elem["question"], elem["trajectory"], elem["answer"]  # type: ignore

        joint, xyz = trajectory.shape[0:2]

        trajectory = rearrange(trajectory[indices], "joint xyz len -> 1 len (joint xyz)")
        # trajectory = rearrange(
        #     self.scaler.transform(trajectory),
        #     "1 len (joint xyz) -> joint xyz len",
        #     joint=joint,
        #     xyz=xyz,
        # )
        trajectory = rearrange(self.scaler.transform(trajectory), "1 len f -> f len")

        return question, trajectory, label


class TimeQADataModule(LightningDataModule):
    KEY = "dasyd/time-qa-simple"

    def __init__(
        self,
        batch_size: int = 32,
        task: Literal["binary", "multi", "open"] = "binary",
    ):
        super().__init__()

        self.batch_size = batch_size
        self.task = task
        self.scaler = Scaler()

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
        # min max normalization

    @property
    def __trajectory_shape(self) -> torch.Size:
        assert self.dataset is not None, "You need to run prepare first"

        self.dataset = cast(DatasetDict, self.dataset.with_format("torch"))
        trajectory = cast(torch.Tensor, self.dataset["train"]["trajectory"])

        # return torch.Size((trajectory.shape[0], 1,1,300)) # trajectory.shape
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
            traj = dataset["train"]["trajectory"]

            self.scaler.fit(
                rearrange(
                    traj,
                    "batch joint xyz len -> batch len (joint xyz)",
                )
            )
            self.train_dataset = TimeQADataset(dataset["train"], self.scaler)
            self.val_dataset = TimeQADataset(dataset["val"], self.scaler)

        elif stage == "test":
            dataset = self._load_dataset_split(["test", "train"])

            dataset = cast(DatasetDict, dataset.with_format("torch"))
            traj = dataset["train"]["trajectory"]

            self.scaler.fit(
                rearrange(
                    traj,
                    "batch joint xyz len -> batch len (joint xyz)",
                )
            )
            self.test_dataset = TimeQADataset(dataset["test"], self.scaler)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


class OppQADataset(Dataset):

    def __init__(self, hf_dataset: HFDataset, scaler: Scaler):
        self.hf_dataset = hf_dataset
        self.scaler = scaler

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        elem = self.hf_dataset[idx]
        question, trajectory, label = elem["question"], elem["trajectory"], elem["answer"]  # type: ignore

        # joint, xyz = trajectory.shape[0:2]

        # trajectory = rearrange(trajectory, "joint xyz len -> 1 len (joint xyz)")
        # trajectory = rearrange(
        #     self.scaler.transform(trajectory),
        #     "1 len (joint xyz) -> joint xyz len",
        #     joint=joint,
        #     xyz=xyz,
        # )
        # trajectory = rearrange(self.scaler.transform(trajectory), "len f -> f len") #[:1]
        trajectory = rearrange(trajectory, "len f -> f len") #[:1]

        return question, trajectory, label


    
class OppQADataModule(LightningDataModule):
    KEY = "dasyd/OppQA"

    def __init__(
        self,
        batch_size: int = 32,
        task: Literal["binary", "multi", "open"] = "multi",
    ):
        super().__init__()

        self.batch_size = batch_size
        self.task = task

    def _load_dataset_split(self, splits: list[str]):
        """Workaround to overcome the missing hf implementation of only dowloading the split shards"""

        return load_dataset(
            OppQADataModule.KEY,
            self.task,
            data_dir=self.task,
            data_files={split: f"{split}-*" for split in splits},
            verification_mode=VerificationMode.NO_CHECKS,
            num_proc=len(splits),
        )

    def prepare_data(self) -> None:
        self._load_dataset_split(["val", "train"])

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.dataset = self._load_dataset_split(["train", "val"])
        elif stage == "test":
            self.dataset = self._load_dataset_split(["test"])

        self.dataset = self.dataset.with_format("torch")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            OppQADataset(self.dataset["train"], scaler=Scaler()),
             batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(OppQADataset(self.dataset["val"], scaler=Scaler()), batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(OppQADataset(self.dataset["test"], scaler=Scaler()), batch_size=self.batch_size)
    

