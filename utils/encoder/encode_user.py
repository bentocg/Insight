__all__ = ['encode_user', 'MinMaxScaler']

from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize
import torch
from utils.encoder.match_nn import MatchNN


class MinMaxScaler():
    """
    Helper to scale input users to the
    """

    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, reference_db):
        self.min = [min(reference_db.iloc[:, idx]) for idx in range(reference_db.shape[1])]
        self.max = [max(reference_db.iloc[:, idx]) for idx in range(reference_db.shape[1])]

    def rescale(self, input_vec):
        assert len(input_vec) == len(self.min)
        scaled_vec = []
        for idx, ele in enumerate(input_vec):
            ele_std = (ele - self.min[idx]) / (self.max[idx] - self.min[idx])
            ele_scaled = ele_std * (self.max[idx] - self.min[idx]) + self.min[idx]
            scaled_vec.append(ele_scaled)
        return torch.tensor(scaled_vec)


class UserEncoder():
    def __init__(self, preprocessing, encoder):
        self.encoder = encoder
        self.preprocessing = preprocessing

    def encode_user(self, user):
        user = self.preprocessing.rescale(user)
        encoding = self.encoder.forward_arm(user)
        return encoding
