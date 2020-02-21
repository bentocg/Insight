"""
Train Encoder NN
===========================================================
Training / Hyperparameter search for MatchNN. Tries random combinations of hyperparameters keeping the one with the
highest f1 score.

Author: Bento Gonçalves
License: MIT
Copyright: 2020-2021
"""

import os
import sys
import time
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils.encoder.match_dataset import MatchDataset
from utils.encoder.match_nn import MatchNN
from utils.encoder.training_loop import train_loop


def parse_args():
    parser = ArgumentParser('Training script for match encoder neural network')
    parser.add_argument('--input_pairs', '-p', type=str, default='Datasets/pairs_NA_2005-2019.csv',
                        help='path to csv with user pair data')
    parser.add_argument('--input_users', '-u', type=str, default='Datasets/users_NA_2005-2019.csv',
                        help='path to csv with training users')
    parser.add_argument('--out', '-o', type=str, default='Saved_models',
                        help='path to folder where model weights will be saved')

    return parser.parse_args()


def main():
    # read arguments
    args = parse_args()

    # set seed and deterministic for reproducibility
    np.random.seed(42)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # flag to use gpu
    use_gpu = False
    if torch.cuda.is_available():
        use_gpu = True

    # get pairs and user metadata for training
    training_pairs = pd.read_csv(args.input_pairs)
    users_train = pd.read_csv(args.input_users, index_col=0)

    # find good and bad matches
    good_matches = training_pairs.loc[(training_pairs.count_percent > 0.15) & (training_pairs['count'] >= 3)]
    bad_matches = training_pairs.loc[training_pairs.count_percent < 0.001]
    train_ids = set(list(good_matches['user_1']) + list(good_matches['user_2']) + list(bad_matches['user_1']) + list(
        bad_matches['user_2']))

    # write training users
    with open('Datasets/users_train.txt', 'w') as src:
        for checklist in users_train.loc[train_ids]['sample_checklist']:
            src.write(checklist + ' ')

    # drop categorical columns
    unused = ['sample_checklist', 'geometry']
    users_train = users_train.drop(unused, axis=1)
    users_train = users_train.fillna(0)
    n_features = users_train.shape[1]

    # get match data and split between train and validation
    x1 = users_train.loc[good_matches['user_1']].append(users_train.loc[bad_matches['user_1']])
    x2 = users_train.loc[good_matches['user_2']].append(users_train.loc[bad_matches['user_2']])
    y_vec = np.array([1] * len(good_matches) + [0] * len(bad_matches))
    valid_idcs = np.random.choice(len(y_vec), int(len(y_vec) * 0.1), replace=False)
    train_idcs = [ele for ele in range(len(y_vec)) if ele not in valid_idcs]

    # write validation users
    with open('Datasets/users_validation.txt', 'w') as src:
        for idx in valid_idcs:
            src.write(f"{idx}_{y_vec[idx]} ")

    # get weights for getting training samples
    y_train = y_vec[train_idcs]
    ratio = sum(y_train) / len(y_train)

    weights = torch.DoubleTensor([ratio if ele == 0 else 1 - ratio for ele in y_train])
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    # create training and validation datasets
    train_ds = MatchDataset(x1.iloc[train_idcs], x2.iloc[train_idcs], y_vec[train_idcs])
    valid_ds = MatchDataset(x1.iloc[valid_idcs], x2.iloc[valid_idcs], y_vec[valid_idcs])

    # create training and validation dataloaders
    batch_size = 1024
    train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    # hyperparameter search

    # pool of hyperparameter values
    n_combs = 5
    epochs = 10
    hidden_size_pool = [100, 200, 500]
    hidden_layers_pool = [1, 2, 3]
    dropout_pool = [0.]
    learning_rate_pool = [1E-3, 5E-4, 1E-4, 5E-5]
    out_size_pool = [100, 200, 500]
    loss_pool = ["L1", "SmoothL1"]
    loss_functions = {"SmoothL1": nn.SmoothL1Loss(),
                      "L1": nn.L1Loss()}

    # store combinations
    combinations = {}

    # get n combinations at random from pool
    start = time.time()
    for i in range(n_combs):
        hidden_size = np.random.choice(hidden_size_pool)
        hidden_layers = int(np.random.choice(hidden_layers_pool))
        dropout = np.random.choice(dropout_pool)
        learning_rate = np.random.choice(learning_rate_pool)
        out_size = np.random.choice(out_size_pool)
        loss = np.random.choice(loss_pool)
        loss_fn = loss_functions[loss]

        # create an identifier for combination
        key = f"{hidden_size}_{out_size}_{hidden_layers}_{dropout}_{learning_rate}_{loss}"

        # instantiate model using combination of parameters
        model = MatchNN(n_features, [hidden_size] * hidden_layers, out_size, [dropout] * hidden_layers)
        if use_gpu:
            model = nn.DataParallel(model.cuda())

        # get combination, f1 score and state dict from training
        combinations[key] = train_loop(model, epochs, loss_fn, train_dl, valid_dl, use_gpu=use_gpu, lr=learning_rate)

    # get the combination with the highest f1 score
    max_val = 0
    max_key = ''
    max_params = None
    for key, val in combinations.items():
        if val[0] > max_val:
            max_val = val[0]
            max_params = val[1]
            max_key = key

    print(f"Evaluated {n_combs} hyperparameter combinations in "
          f"{time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")
    print(f"\nBest combination {max_key}: best F1 score: {max_val}")

    # save best model weights
    torch.save(max_params, f"{args.out}/{max_key}.pth")


if __name__ == "__main__":
    main()
