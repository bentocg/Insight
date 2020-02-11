"""
Encode user
==========================================================
User encoder class and preprocessing helpers. MimMaxScaler mainly to replicate training preprocessing. User encoder
interfaces MatchNN to arrays of raw user data to produce user encodings.

Author: Bento Gonçalves
License: MIT
Copyright: 2020-2021
"""

__all__ = ['UserEncoder', 'MinMaxScaler']

import torch


class MinMaxScaler:
    """
    Helper to scale input users to according to a reference dataset
    """

    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, reference_db):
        self.min = [min(reference_db.iloc[:, idx]) for idx in range(reference_db.shape[1])]
        self.max = [max(reference_db.iloc[:, idx]) for idx in range(reference_db.shape[1])]

    def rescale(self, input_vec, tensor=True):
        scaled_vec = []
        for idx, ele in enumerate(input_vec):
            ele_std = (ele - self.min[idx]) / (self.max[idx] - self.min[idx])
            ele_scaled = ele_std * (self.max[idx] - self.min[idx]) + self.min[idx]
            scaled_vec.append(ele_scaled)
        if tensor:
            return torch.tensor(scaled_vec)
        else:
            return scaled_vec


class UserEncoder:
    """
    Class to apply neural net encoding to user parameter vectors
    """

    def __init__(self, preprocessing, encoder):
        self.encoder = encoder
        self.preprocessing = preprocessing

    def encode_user(self, user):
        user = self.preprocessing.rescale(user)
        encoding = self.encoder.forward_arm(user)
        return encoding.detach().numpy().reshape([1, -1])
