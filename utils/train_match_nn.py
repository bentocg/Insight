import operator

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.neural_net.match_dataset import MatchDataset
from utils.neural_net.match_nn import MatchNN
from utils.neural_net.training_loop import train_loop

# set seed and deterministic for reproducibility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# flag to use gpu
if torch.cuda.is_available():
    use_gpu = True

# get pairs and user metadata for training
training_pairs = pd.read_csv('../Datasets/pairs_NA_2005-2019.csv')
users_train = pd.read_csv('../Datasets/users_NA_relDec-2010.csv', index_col=0)

# only keep numerical columns
unused = ['sample_checklist', 'geometry']
users_train = users_train.iloc[:, [idx for idx, ele in enumerate(users_train.columns) if ele not in unused]]
users_train = users_train.fillna(0)

# find good and bad matches
good_matches = training_pairs.loc[(training_pairs.count_percent > 0.15) & (training_pairs['count'] >= 3)]
bad_matches = training_pairs.loc[training_pairs.count_percent < 0.001]

# get match data and split between train and validation
x1 = users_train.loc[good_matches['user_1']].append(users_train.loc[bad_matches['user_1']])
x2 = users_train.loc[good_matches['user_2']].append(users_train.loc[bad_matches['user_2']])
y_vec = np.array([1] * len(good_matches) + [0] * len(bad_matches))
valid_idcs = np.random.choice(len(y_vec), int(len(y_vec) * 0.1), replace=False)
train_idcs = [ele for ele in range(len(y_vec)) if ele not in valid_idcs]

# get weights for getting training samples
y_train = y_vec[train_idcs]
ratio = sum(y_train) / len(y_train)

weights = torch.DoubleTensor([ratio if ele == 0 else 1 - ratio for ele in y_train])
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

# create training and validation datasets
train_ds = MatchDataset(x1.iloc[train_idcs], x2.iloc[train_idcs], y_vec[train_idcs])
valid_ds = MatchDataset(x1.iloc[valid_idcs], x2.iloc[valid_idcs], y_vec[valid_idcs])

batch_size = 1024
train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

epochs = 250
hidden_size_pool = [30, 50, 70]
hidden_layers_pool = [1, 2, 3, 4]
dropout_pool = [0., 0.25, 0.50]
learning_rate_pool = [1E-3, 5E-4, 1E-4, 5E-5]
out_size_pool = [10, 25, 50]

combinations = {}
comb = {}

loss_fn = nn.MSELoss()

for i in range(100):
    hidden_size = np.random.choice(hidden_size_pool)
    hidden_layers = int(np.random.choice(hidden_layers_pool))
    dropout = np.random.choice(dropout_pool)
    learning_rate = np.random.choice(learning_rate_pool)
    out_size = np.random.choice(out_size_pool)
    key = f"{hidden_size}_{out_size}_{hidden_layers}_{dropout}_{learning_rate}"

    model = MatchNN(23, [hidden_size] * hidden_layers, out_size, [dropout] * hidden_layers)
    if use_gpu:
        model = nn.DataParallel(model.cuda())

    combinations[key] = train_loop(model, epochs, loss_fn, train_dl, valid_dl, use_gpu=use_gpu, lr=learning_rate)

comb = max(combinations.iteritems(), key=operator.itemgetter(1))[0]
print('max f1 :', comb, combinations[comb])
