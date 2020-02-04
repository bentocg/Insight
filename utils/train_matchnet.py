import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from utils.matchNet import matchNet
from sklearn.preprocessing import normalize

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
total_good = len(good_matches)
total_bad = len(bad_matches)

# draw metadata from users

users_good = [torch.FloatTensor(normalize(users_train.loc[good_matches['user_1']].values)),
              torch.FloatTensor(normalize(users_train.loc[good_matches['user_2']].values))]
users_bad = [torch.FloatTensor(normalize(users_train.loc[bad_matches['user_1']].values)),
             torch.FloatTensor(normalize(users_train.loc[bad_matches['user_2']].values))]
# define model
emb_size = 5
params = len(users_train.columns)
model = matchNet(params, 100, emb_size)
model = model.float()
model.train()

# initiate optimizer and loss function
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1E-5)

# training loop
exp_avg_loss = 0
for i in range(100000):
    # zero gradients
    optimizer.zero_grad()

    # draw a good pair and a bad pair at random
    choice_good = np.random.choice(total_good)
    choice_bad = np.random.choice(total_bad)
    g1, g2 = users_good[0][choice_good], users_good[1][choice_good]
    b1, b2 = users_bad[0][choice_bad], users_bad[1][choice_bad]

    # pass them through matchNet and get the difference between good and bad samples
    diff = model.forward(g1, g2, b1, b2)

    # get loss
    loss = loss_fn(diff, torch.tensor([1.]))
    exp_avg_loss = (exp_avg_loss * 0.999 + loss.item() * 0.001)

    if i % 1000 == 0:
        print(exp_avg_loss)
    loss.backward()
    optimizer.step()
