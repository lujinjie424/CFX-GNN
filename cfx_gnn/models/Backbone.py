"""GNN encoders used by the paper experiments."""

import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import APPNPConv, GraphConv


class GCNEncoder(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers=2,
                 activation=F.relu, dropout=0.5):
        super().__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([GraphConv(in_feats, hidden_feats)])
        self.layers.extend(
            GraphConv(hidden_feats, hidden_feats) for _ in range(num_layers - 2)
        )
        self.layers.append(GraphConv(hidden_feats, out_feats))

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, graph, features, edge_weight=None):
        hidden = features
        for index, layer in enumerate(self.layers):
            hidden = layer(graph, hidden, edge_weight=edge_weight)
            if index != len(self.layers) - 1:
                hidden = self.dropout(self.activation(hidden))
        return hidden


class APPNPEncoder(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers=2,
                 k=10, alpha=0.1, dropout=0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([nn.Linear(in_feats, hidden_feats)])
        self.layers.extend(
            nn.Linear(hidden_feats, hidden_feats) for _ in range(num_layers - 2)
        )
        self.layers.append(nn.Linear(hidden_feats, out_feats))
        self.propagate = APPNPConv(k=k, alpha=alpha)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, graph, features, edge_weight=None):
        hidden = features
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if index != len(self.layers) - 1:
                hidden = self.dropout(F.relu(hidden))
        return self.propagate(graph, hidden, edge_weight=edge_weight)
