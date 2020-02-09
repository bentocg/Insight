"""
Validation
===========================================================
Validates baselines for Match NN. Current baselines: cosine distance using raw input vectors and at random.

Author: Bento Gonçalves
License: MIT
Copyright: 2020-2021
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from utils.encoder.encode_user import MinMaxScaler


# helper function to drop columns not used by the model
def drop_unused(dataframe, unused=['geometry', 'sample_checklist']):
    dataframe = dataframe.drop(unused, axis=1)
    dataframe = dataframe.fillna(0)
    return dataframe


# get validation indices
with open('../Datasets/users_validation.txt', 'r') as src:
    pairs = src.readline()
    pairs = pairs.strip().split(' ')
    val_idcs = [int(pair.split('_')[0]) for pair in pairs]
    val_labels = [int(pair.split('_')[1]) for pair in pairs]

# get validation pairs
training_pairs = pd.read_csv('../Datasets/pairs_NA_2005-2019.csv')
val_pairs = training_pairs.iloc[val_idcs]

# get input data
user_data = pd.read_csv('../Datasets/users_NA_2005-2019.csv', index_col=0)
val_users = set(list(val_pairs['user_1']) + list(val_pairs['user_2']))
user_data = user_data.loc[val_users]

# get min max scaling
with open('../utils/users_train.txt', 'r') as src:
    check_train = src.readline().strip().split(' ')

# get preprocessing scaling from training users
users_train = user_data.loc[user_data.sample_checklist.isin(check_train)]
users_train = drop_unused(users_train)
preprocessing = MinMaxScaler()
preprocessing.fit(users_train)

# drop categorical columns and fill missing values
user_data = drop_unused(user_data)
user_data = user_data.fillna(0)

# get pair covariates
x1 = user_data.loc[val_pairs['user_1']].astype(np.float32).values
x2 = user_data.loc[val_pairs['user_2']].astype(np.float32).values
y = np.array(val_labels)


# cosine distance between covariates
def cosine(x1, x2):
    if cosine_similarity(x1, x2) >= 0.5:
        return 1
    else:
        return 0


# euclidean distance between covariates
def euclidean(x1, x2, threshold=999999):
    if euclidean_distances(x1, x2) <= threshold:
        return 1
    else:
        return 0


# validate models
metrics = ['euclidean', 'cosine', 'random']
baselines = {metric: [] for metric in metrics}
for idx in range(len(y)):
    # get output for baselines
    euc_out = euclidean(preprocessing.rescale(x1[idx, :].reshape([1, -1]), tensor=False),
                        preprocessing.rescale(x2[idx, :].reshape([1, -1]), tensor=False))
    cos_out = cosine(x1[idx, :].reshape([1, -1]),
                     x2[idx, :].reshape([1, -1]))
    random_out = np.random.randint(2)

    baselines['euclidean'].append(euc_out)
    baselines['cosine'].append(cos_out)
    baselines['random'].append(random_out)

# convert to numpy array
for base in baselines:
    baselines[base] = np.array(baselines[base])

# get metrics
true_positives = {metric: 0 for metric in metrics}
false_positives = {metric: 0 for metric in metrics}
false_negatives = {metric: 0 for metric in metrics}
recall = {metric: 0 for metric in metrics}
precision = {metric: 0 for metric in metrics}

for base in baselines:
    true_positives[base] = np.sum(baselines[base][y == 1])
    false_positives[base] = np.sum(baselines[base][y == 0])
    false_negatives[base] = np.sum(y) - true_positives[base]
    recall[base] = true_positives[base] / (true_positives[base] + false_negatives[base])
    precision[base] = true_positives[base] / (true_positives[base] + false_positives[base])

    # print precision and recall
    print(f"{base}: precision = {precision[base]}, recall = {recall[base]}")
