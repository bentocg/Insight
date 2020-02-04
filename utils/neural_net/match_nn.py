__all__ = ['MatchNN']

import torch.nn as nn
import torch.nn.functional as F


class MatchNN(nn.Module):
    def __init__(
            self,
            input_size,
            lin_layer_sizes,
            output_size,
            lin_layer_dropouts,
    ):

        """
        Parameters
        ----------
        This list will contain a two element tuple for each
        categorical feature. The first element of a tuple will
        denote the number of unique values of the categorical
        feature. The second element will denote the embedding
        dimension to be used for that feature.
        The number of continuous features in the data.
        lin_layer_sizes: List of integers.
        The size of each linear layer. The length will be equal
        to the total number
        of linear layers in the network.
        output_size: Integer
        The size of the final output.
        The dropout to be used after the embedding layers.
        lin_layer_dropouts: List of floats
        The dropouts to be used after each linear layer.
        """

        super().__init__()

        # helper to get cosine distance
        self.cosine = nn.modules.distance.CosineSimilarity()

        # input size
        self.input_size = input_size
        self.relu = nn.Hardtanh(0., 1.)

        # Linear Layers
        first_lin_layer = nn.Linear(self.input_size, lin_layer_sizes[0])

        self.lin_layers = nn.ModuleList(
            [first_lin_layer]
            + [
                nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i + 1])
                for i in range(len(lin_layer_sizes) - 1)
            ]
        )

        for lin_layer in self.lin_layers:
            nn.init.kaiming_normal_(lin_layer.weight.data)

        # Output Layer
        self.output_layer = nn.Linear(lin_layer_sizes[-1], output_size)
        nn.init.kaiming_normal_(self.output_layer.weight.data)

        # Batch Norm Layers
        self.first_bn_layer = nn.BatchNorm1d(self.input_size)
        self.bn_layers = nn.ModuleList(
            [nn.BatchNorm1d(size) for size in lin_layer_sizes]
        )

        # Dropout Layers
        self.droput_layers = nn.ModuleList(
            [nn.Dropout(prob) for prob in lin_layer_dropouts]
        )

    def forward_arm(self, x):

        x = self.first_bn_layer(x)

        for lin_layer, dropout_layer, bn_layer in zip(
                self.lin_layers, self.droput_layers, self.bn_layers
        ):
            x = F.relu(lin_layer(x))
            x = bn_layer(x)
            x = dropout_layer(x)

        x = self.output_layer(x)
        return x

    def forward(self, x1, x2):
        x1 = self.forward_arm(x1)
        x2 = self.forward_arm(x2)
        out = self.cosine(x1, x2).reshape([-1, 1])
        out = self.relu(out)
        return out


class MatchNN(nn.Module):
    def __init__(
            self,
            input_size,
            lin_layer_sizes,
            output_size,
            lin_layer_dropouts,
    ):

        """
        Parameters
        ----------
        This list will contain a two element tuple for each
        categorical feature. The first element of a tuple will
        denote the number of unique values of the categorical
        feature. The second element will denote the embedding
        dimension to be used for that feature.
        The number of continuous features in the data.
        lin_layer_sizes: List of integers.
        The size of each linear layer. The length will be equal
        to the total number
        of linear layers in the network.
        output_size: Integer
        The size of the final output.
        The dropout to be used after the embedding layers.
        lin_layer_dropouts: List of floats
        The dropouts to be used after each linear layer.
        """

        super().__init__()

        # helper to get cosine distance
        self.cosine = nn.modules.distance.CosineSimilarity()

        # input size
        self.input_size = input_size
        self.relu = nn.Hardtanh(0., 1.)

        # Linear Layers
        first_lin_layer = nn.Linear(self.input_size, lin_layer_sizes[0])

        self.lin_layers = nn.ModuleList(
            [first_lin_layer]
            + [
                nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i + 1])
                for i in range(len(lin_layer_sizes) - 1)
            ]
        )

        for lin_layer in self.lin_layers:
            nn.init.kaiming_normal_(lin_layer.weight.data)

        # Output Layer
        self.output_layer = nn.Linear(lin_layer_sizes[-1], output_size)
        nn.init.kaiming_normal_(self.output_layer.weight.data)

        # Batch Norm Layers
        self.first_bn_layer = nn.BatchNorm1d(self.input_size)
        self.bn_layers = nn.ModuleList(
            [nn.BatchNorm1d(size) for size in lin_layer_sizes]
        )

        # Dropout Layers
        self.droput_layers = nn.ModuleList(
            [nn.Dropout(prob) for prob in lin_layer_dropouts]
        )

    def forward_arm(self, x):

        x = self.first_bn_layer(x)

        for lin_layer, dropout_layer, bn_layer in zip(
                self.lin_layers, self.droput_layers, self.bn_layers
        ):
            x = F.relu(lin_layer(x))
            x = bn_layer(x)
            x = dropout_layer(x)

        x = self.output_layer(x)
        return x

    def forward(self, x1, x2):
        x1 = self.forward_arm(x1)
        x2 = self.forward_arm(x2)
        out = self.cosine(x1, x2).reshape([-1, 1])
        out = self.relu(out)
        return out


