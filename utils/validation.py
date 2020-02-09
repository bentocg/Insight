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
from sklearn.metrics.pairwise import cosine_similarity


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

# drop categorical columns and fill missing values
user_data = drop_unused(user_data)
user_data = user_data.fillna(0)

# get pair covariates
x1 = user_data.loc[val_pairs['user_1']].astype(np.float32).values
x2 = user_data.loc[val_pairs['user_2']].astype(np.float32).values
y = np.array(val_labels)


# cosine distance between covariates
def raw_cosine(x1, x2):
    if cosine_similarity(x1, x2) >= 0.5:
        return 1
    else:
        return 0


# validate models
baselines = {'cosine': [],
             'random': []}
for idx in range(len(y)):
    # get baselines for baselines
    cos_out = raw_cosine(x1[idx, :].reshape([1, -1]),
                         x2[idx, :].reshape([1, -1]))
    random_out = np.random.randint(2)

    baselines['cosine'].append(cos_out)
    baselines['random'].append(random_out)

# convert to numpy array
for base in baselines:
    baselines[base] = np.array(baselines[base])

# get metrics
true_positives = {'cosine': 0,
                  'random': 0}
false_positives = {'cosine': 0,
                   'random': 0}
false_negatives = {'cosine': 0,
                   'random': 0}
recall = {'cosine': 0,
          'random': 0}
precision = {'cosine': 0,
             'random': 0}

for base in baselines:
    true_positives[base] = np.sum(baselines[base][y == 1])
    false_positives[base] = np.sum(baselines[base][y == 0])
    false_negatives[base] = np.sum(y) - true_positives[base]
    recall[base] = true_positives[base] / (true_positives[base] + false_negatives[base])
    precision[base] = true_positives[base] / (true_positives[base] + false_positives[base])

    # print precision and recall
    print(f"{base}: precision = {precision[base]}, recall = {recall[base]}")
