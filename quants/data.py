from pathlib import Path
import pickle
from einops import rearrange
from lightning import LightningDataModule
from typing import Literal
from datasets import (
    load_dataset,
    load_from_disk,
    DatasetDict,
    VerificationMode,
    Dataset,
)
import os
from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm


class TSQADataModule(LightningDataModule):
    KEY = "dasyd/time-qa"

    def __init__(
        self,
        batch_size: int = 32,
        task: Literal["binary", "multi", "open", "count"] = "multi",
    ):
        super().__init__()

        self.batch_size = batch_size
        self.task = task
        self.root_cache = Path(f"/workspaces/ts-qa/preprocess_data/ts_qa")
        self.root_cache.mkdir(parents=True, exist_ok=True)

        self.cache_path_action_split = self.root_cache / f"{task}_action_split"
        self.cache_path_normal = self.root_cache / f"{task}_normal"

        self.action_2_idx = {}

    def load_data(self, splits=["val", "train", "test"]) -> DatasetDict:
        ds = load_dataset(
            TSQADataModule.KEY,
            self.task,
            data_dir=self.task,
            data_files={split: f"{split}-*" for split in splits},
            verification_mode=VerificationMode.NO_CHECKS,
            num_proc=len(splits),
        )
        return ds  # type: ignore

    def process(self, action_split: bool):
        ds = self.load_data()

        ds = ds.with_format("torch")

        action_2_idx = {}

        def convert_action(action_dict):
            action_list = action_dict["action"]
            for action in action_list:
                if action not in action_2_idx:
                    action_2_idx[action] = len(action_2_idx)
            return [action_2_idx[action] for action in action_list]

        def make_action_split(x):
            traj = x["trajectory"].view(-1, 24 * 3)
            traj = rearrange(traj, "(act cnt) channels -> act cnt channels", cnt=80)
            action_ids = convert_action(x["action_sequence"])
            return action_ids, traj

        for sp in ["train", "val", "test"]:
            ds_split = ds[sp]

            if action_split:
                trajs = []
                action_ids_lst = []

                for i in tqdm(range(len(ds_split))):
                    action_ids, traj = make_action_split(ds_split[i])
                    trajs.append(traj)
                    action_ids_lst.append(torch.tensor(action_ids, dtype=torch.long))

                trajs = torch.concat(trajs)
                action_ids = torch.concat(action_ids_lst)

                new_ds = Dataset.from_dict(
                    {"action_ids": action_ids, "trajectory": trajs}
                )

            else:
                new_ds = ds_split.map(
                    lambda x: {"trajectory": x["trajectory"].view(-1, 24 * 3)}
                )

            ds[sp] = new_ds

        if action_split:
            ds.save_to_disk(str(self.cache_path_action_split))
            pickle.dump(action_2_idx, open(self.root_cache / "action2idx.pkl", "wb"))
        else:
            ds.save_to_disk(str(self.cache_path_normal))

    def prepare_data(self) -> None:
        if not os.path.exists(self.cache_path_action_split):
            self.process(action_split=True)
        if not os.path.exists(self.cache_path_normal):
            self.process(action_split=False)

    def setup(self, stage: str) -> None:
        if stage in ["fit", "validate"]:
            self.dataset: DatasetDict = load_from_disk(str(self.cache_path_action_split))  # type: ignore
        else:
            self.dataset: DatasetDict = load_from_disk(str(self.cache_path_normal))  # type: ignore

        self.dataset: DatasetDict = self.dataset.with_format("torch")
        self.action_2_idx = pickle.load(open(self.root_cache / "action2idx.pkl", "rb"))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset["train"],  # type: ignore
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=10,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset["val"],  # type: ignore
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=10,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset["test"],  # type: ignore
            batch_size=20 if self.task == "binary" else 15,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset["test"],  # type: ignore
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=10,
        )
