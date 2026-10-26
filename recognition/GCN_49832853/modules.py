
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, num_features, num_classes,
                 hidden_channels=64, dropout=0.5):
        super().__init__()

        self.conv1 = GCNConv(
            num_features, hidden_channels
        )

        self.conv2 = GCNConv(
            hidden_channels, num_classes
        )

        self.dropout = dropout

    def forward(self, x, edge_index):
        # First graph convolution
        x = self.conv1(x, edge_index)

        # Nonlinear activation
        x = F.relu(x)

        # Dropout during training
        x = F.dropout(
            x, p=self.dropout, training=self.training
        )

        # Second graph convolution
        x = self.conv2(x, edge_index)

        # Return raw classification scores
        return x
