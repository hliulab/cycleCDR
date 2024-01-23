import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool as gmp


class GATNet(torch.nn.Module):
    def __init__(self, num_features=78, output_dim=128, dropout=0.2, dtype=torch.float32):
        super().__init__()

        self.gat1 = GATConv(num_features, output_dim, heads=10, dropout=dropout)
        self.gat2 = GATConv(output_dim * 10, output_dim, dropout=dropout)
        self.gat_fc = nn.Linear(output_dim, output_dim, dtype=dtype)

        self.dropout = dropout

    def forward(self, data):
        x1, edge_index, batch = data.x, data.edge_index, data.batch

        x1, _ = self.gat1(x1, edge_index, return_attention_weights=True)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=self.dropout)

        x1, _ = self.gat2(x1, edge_index, return_attention_weights=True)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=self.dropout)
        x1 = gmp(x1, batch)

        x1 = self.gat_fc(x1)
        x1 = F.relu(x1)

        return x1
