import torch.nn as nn


class matchNet(nn.Module):

    def __init__(self, n_inputs, hidden_size, n_out):
        super(matchNet, self).__init__()
        self.fc = nn.Sequential(nn.Linear(n_inputs, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, n_out))
        self.cosine = nn.modules.distance.CosineSimilarity()
        self.n_out = n_out
        self.relu = nn.ReLU()

    def forward_arm(self, x):
        x = self.fc(x)
        return x

    def forward(self, good1, good2, bad1, bad2):
        out_g1 = self.forward_arm(good1).reshape(1, -1)
        out_g2 = self.forward_arm(good2).reshape(1, -1)
        out_b1 = self.forward_arm(bad1).reshape(1, -1)
        out_b2 = self.forward_arm(bad2).reshape(1, -1)
        similarity_g = self.cosine(out_g1, out_g2)
        similarity_b = self.cosine(out_b1, out_b2)
        return similarity_g - self.relu(similarity_b)
