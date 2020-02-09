"""
Match Dataset
==========================================================
PyTorch dataset construct to train MatchNN. Returns a pair of users and a binary y value for that pair when requested.

Author: Bento Gonçalves
License: MIT
Copyright: 2020-2021
"""

__all__ = ['MatchDataset']

import numpy as np
from sklearn.preprocessing import minmax_scale
from torch.utils.data import Dataset


class MatchDataset(Dataset):
    def __init__(self, x1, x2, y):
        """

        :param x1:
        :param x2:
        :param y:
        """

        # get length
        self.n = x1.shape[0]

        # reformat labels
        self.y = y.astype(np.float32).reshape(-1, 1)

        # get data and reformat
        self.X1 = minmax_scale(x1.astype(np.float32).values)
        self.X2 = minmax_scale(x2.astype(np.float32).values)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return [self.X1[idx], self.X2[idx], self.y[idx]]
