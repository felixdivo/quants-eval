import os
from typing import Literal, cast
from lightning import Trainer, LightningDataModule
import numpy as np
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import RichProgressBar, RichModelSummary
from datasets import load_dataset, VerificationMode, DatasetDict, Dataset as HFDataset
from torch.utils.data import DataLoader
from json import loads as jloads
from torch.utils.data import TensorDataset, StackDataset, Dataset
import torch
from einops import rearrange
from lightning.pytorch.loggers import WandbLogger
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
import opp_qa.embedding as ebd
import operator
import sys
import scipy as sc
from collections import defaultdict
from nltk import word_tokenize
from datasets import load_from_disk


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


class OppQADataModule(LightningDataModule):
    KEY = "dasyd/OppQA"

    def __init__(
        self,
        batch_size: int = 32,
        short: bool = False,
        task: Literal["binary", "multi", "open", "count"] = "multi",
    ):
        super().__init__()

        self.batch_size = batch_size
        self.task = task
        self.short = short
        self.cache = f"/workspaces/ts-qa/preprocess_data/opp_qa_{self.task}{'_short'if short else ''}_glove-300d"

    def load_data(self,splits=["val", "train", "test"]) -> DatasetDict:
        ds = load_dataset(
            f"dasyd/OppQA{'-500' if self.short else ''}",
            self.task,
            data_dir=self.task,
            data_files={split: f"{split}-*" for split in splits},
            verification_mode=VerificationMode.NO_CHECKS,
            num_proc=len(splits),
        )
        return ds # type: ignore



    def process(self):
    
        word_idx = ebd.load_idx()
        ds = self.load_data()

        for sp in ["train", "val", "test"]:
            ds_split = ds[sp]

        
            int_to_answer = [
                    "no",
                    "yes",
                    "0",
                    "1",
                    "2",
                    "open the front door",
                    "clean the table",
                    "open the third drawer",
                    "close the front door",
                    "toggle the switch",
                    "close the third drawer",
                    "open the second drawer",
                    "close the first drawer",
                    "close the second drawer",
                    "open the first drawer",
                    "close the back door",
                    "open the back door",
                    "close the fridge",
                    "open the fridge",
                    "close the dishwasher",
                    "drink from the cup",
                    "open the dishwasher",
                    "3",
                    "6",
                    "4",
                    "7",
                ]


            def tokenize_and_pad(examples, word_idx, maxlen=31):
                # Tokenize the question
                tokens = word_tokenize(examples["question"])
                
                # Convert tokens to their corresponding indices
                tok_indices = [word_idx.get(token.lower(), 0) for token in tokens]
                
                # Pad the token indices
                padded_tok = tok_indices[:maxlen] + [0] * max(0, maxlen - len(tok_indices))
                
                return {'text_idx': padded_tok, "answer_text": int_to_answer[examples["answer"]]}
            
            # Apply the transformation to the dataset
            tok_data = ds_split.map(lambda x: tokenize_and_pad(x, word_idx=word_idx), batched=False)
            ds[sp] = tok_data
        
        ds.save_to_disk(self.cache)
        # return ds.with_format("torch")

    def prepare_data(self) -> None:
        # self._load_dataset_split(["val", "train"])
        if not os.path.exists(self.cache):
            self.process()

    def setup(self, stage: str) -> None:
        self.dataset: DatasetDict = load_from_disk(self.cache) # type: ignore
        self.dataset: DatasetDict = self.dataset.with_format("torch")
        # self.dataset = self.dataset.map(lambda x: self.tokenizer(x['question'], padding="max_length", max_length=30, truncation=True), batched=True)

        # self.dataset = self.dataset.with_format(type='torch', columns=['input_ids', 'attention_mask', 'trajectory', 'answer'])

        # self.dataset = self.dataset.with_format("torch")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            # OppQADataset(self.dataset["train"], scaler=Scaler(), tokenizer=self.tokenizer),
            self.dataset["train"], # type: ignore
            batch_size=self.batch_size,
            shuffle=True,
            # num_workers=100
            num_workers=10
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset["val"], # type: ignore
            batch_size=self.batch_size,
            # num_workers=50
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset["test"], # type: ignore
            batch_size=self.batch_size,
        )
