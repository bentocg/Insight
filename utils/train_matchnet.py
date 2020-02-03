import torch
import torch.nn as nn
import pandas as pd
from utils.matchNet import matchNet

# get pairs and user metadata for training
training_pairs = pd.read_csv('../Datasets/pairs_NA_2005-2019.csv')
users_train = pd.read_csv('../Datasets/users_NA_relDec-2010.csv')

# find good matches


# find bad matches


# draw metadata from users

# define model
model = matchNet(25, 5)
model.train()

# initiate optimizer and loss function
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1E-3)

# training loop
for i in range(500):
    # zero gradients
    optimizer.zero_grad()

    # draw a good pair and a bad pair at random
    g1, g2 = [1, 1]
    b1, b2 = [1, 1]

    # pass them through matchNet and get the difference between good and bad samples
    diff = model.forward(g1, g2, b1, b2)

    # get loss
    loss = loss_fn(diff, torch.tensor([1.]))
    loss.backward()
    optimizer.step()