import logging
import pdb
import numpy as np
import torch.utils.data as data
from PIL import Image
import torch


class hmnist(data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        X = Image.open(self.df["path"][index])
        y = torch.tensor(int(self.df["cell_type_idx"][index]))

        if self.transform:
            X = self.transform(X)
        return X, y


class hmnist_truncated(data.Dataset):
    def __init__(self, df, dataidxs=None, transform=None, cache_data_set=None):

        self.dataidxs = dataidxs
        self.transform = transform
        self.df = self.__build_truncated_dataset__(df, cache_data_set)

    def __build_truncated_dataset__(self, df, cache_data_set=None):
        # print("download = " + str(self.download))
        if cache_data_set is None:
            cache_data_set = hmnist(self.df, self.transform)

        if self.dataidxs is not None:
            df = df.iloc[self.dataidxs]

        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        X = Image.open(self.df["path"][index])
        y = torch.tensor(int(self.df["cell_type_idx"][index]))

        if self.transform:
            X = self.transform(X)
        return X, y
