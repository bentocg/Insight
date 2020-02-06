__all__ = ['encode_user']

from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize
import torch
from utils.neural_net.match_nn import MatchNN


def encode_user(user):
    assert len(user) == 23, 'missing parameters, could not generate encoding!'
    # load neural network
    model = MatchNN()
    model.load_state_dict('Model_weights/match_net.tar')

    # get user encoding
    return model.forward_arm(torch.tensor(user))

