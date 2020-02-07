import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


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
scaler = MinMaxScaler()
scaler.fit(user_data)

# get pair covariates
x1 = user_data.loc[val_pairs['user_1']].astype(np.float32).values
x2 = user_data.loc[val_pairs['user_2']].astype(np.float32).values
y = val_labels

# rescale between and 1


# cosine distance between covariates
def raw_cosine(x1, x2):
    if cosine_similarity(x1, x2) >= 0.9999:
        return 1
    else:
        return 0


# validate models
true_positives = {'random': 0, 'cosine': 0}
false_positives = {'random': 0, 'cosine': 0}
false_negatives = {'random': 0, 'cosine': 0}

for idx in range(len(y)):
    cos_out = raw_cosine(x1[idx, :].reshape([1, -1]),
                         x2[idx, :].reshape([1, -1]))
    random_out = np.random.randint(2)

    # compute correct
    if random_out == y[idx]:
        true_positives['random'] += 1

    # get mistakes
    else:
        if y[idx] == 0:
            false_positives['random'] += 1
        else:
            false_negatives['random'] += 1

    if cos_out == y[idx]:
        true_positives['cosine'] += 1

    else:
        if y[idx] == 0:
            false_positives['cosine'] += 1
        else:
            false_negatives['cosine'] += 1

# get metrics        
recall_cosine = true_positives['cosine'] / max(1, true_positives['cosine'] + false_negatives['cosine'])
precision_cosine = true_positives['cosine'] / max(1, true_positives['cosine'] + false_positives['cosine'])

recall_random = true_positives['random'] / max(1, true_positives['random'] + false_negatives['random'])
precision_random = true_positives['random'] / max(1, true_positives['random'] + false_positives['random'])

print('cosine: ', precision_cosine, recall_cosine)
print('random: ', precision_random, recall_random)