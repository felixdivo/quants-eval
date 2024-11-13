import os
import glob
import re
from typing import Literal, Union
import numpy as np
import pandas as pd
from lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset
import torch
import logging
from sktime.datasets import load_from_tsfile_to_dataframe
# from aeon import Pipeline
from aeon.datasets import load_classification
from aeon.transformations.impute import Imputer
from sklearn.pipeline import Pipeline
from aeon.transformations.collection.interpolate import TSInterpolator
from sklearn.model_selection import train_test_split

# from aeon.contrib.timeseries import TimeSeriesSampler, load_classification

def subsample(y, limit=256, factor=2):
    """
    If a given Series is longer than `limit`, returns subsampled sequence by the specified integer factor
    """
    if len(y) > limit:
        return y[::factor].reset_index(drop=True)
    return y

def interpolate_missing(y):
    """
    Replaces NaN values in pd.Series `y` using linear interpolation
    """
    if y.isna().any():
        y = y.interpolate(method='linear', limit_direction='both')
    return y


class TSRegressionDataModule(LightningDataModule):
    def __init__(self,name:str, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
        self.task = "classification"
        self.name = name

    
    
    def prepare_data(self) -> None:
        x,_, meta = load_classification(self.name, return_metadata=True) # type: ignore
        self.meta = meta
        self.max_seq_len = x.shape[2] #type: ignore
        self.feat_dim = x.shape[1] # type: ignore

    # def preprocess_data(self, x):
    #     pipeline = Pipeline([
    #         ('imputer', Imputer(method="linear")),
    #         ('interpolator', TSInterpolator())
    #         # ('sampler', TimeSeriesSampler(window_size=self.window_size))
    #     ])
    #     x = pipeline.fit_transform(x)
    #     return x

    def setup(self, stage=None):
         # Load dataset here (idempotent function)

        if stage == 'fit' or stage is None:
            x, y, meta = load_classification(self.name, split='train', return_metadata=True) # type: ignore
            # x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
            x_train = x_val = x
            y_train = y_val = y
            # map class labels to integers, meta["class_values"] contains the original class labels
            y_train = list(map(meta["class_values"].index, y_train))
            y_val = list(map(meta["class_values"].index, y_val))

            x_tensor = torch.tensor(x_train).float()
            y_tensor = torch.tensor(y_train)
            self.train_dataset = TensorDataset(x_tensor, y_tensor)

            x_tensor = torch.tensor(x_val).float()
            y_tensor = torch.tensor(y_val)
            # The following line is just an example and should be replaced with actual validation dataset creation
            self.val_dataset = TensorDataset(x_tensor, y_tensor)

        if stage == 'test' or stage is None:
            x, y, meta = load_classification(self.name, split='test', return_metadata=True) # type: ignore

            y = list(map(meta["class_values"].index, y))

            x_tensor = torch.tensor(x).float()
            y_tensor = torch.tensor(y)
            self.test_dataset = TensorDataset(x_tensor, y_tensor)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    
