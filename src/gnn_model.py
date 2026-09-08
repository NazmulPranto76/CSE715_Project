"""
gnn_model.py
------------
Task 2: a GraphSAGE (or GAT) encoder over segment graphs.

How it works:
    1. Each node starts with its own segment features (chroma + MFCC).
    2. Each GNN layer lets a node update its features by looking at its
       neighbors' features ("message passing").
    3. After the last layer, mean-pool ALL node embeddings into one vector
       per graph ("readout") -- this is the whole track's embedding.
    4. A linear layer turns that embedding into class logits.

Uses PyTorch Geometric's built-in SAGEConv / GATConv layers rather than
writing message passing by hand.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool


class GNNEncoder(nn.Module):
    """Stack of GraphSAGE or GAT layers. Outputs one embedding per NODE."""

    def __init__(self, in_channels, hidden_channels=64, out_channels=64,
                 num_layers=2, arch="graphsage", heads=4, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()

        # keep track of the current feature size, since GAT with
        # multiple heads changes the output size layer by layer
        current_dim = in_channels
        target_dims = [hidden_channels] * (num_layers - 1) + [out_channels]

        for i, target_dim in enumerate(target_dims):
            is_last = i == num_layers - 1
            if arch == "graphsage":
                self.convs.append(SAGEConv(current_dim, target_dim))
                current_dim = target_dim
            elif arch == "gat":
                concat = not is_last
                self.convs.append(GATConv(current_dim, target_dim, heads=heads, concat=concat))
                current_dim = target_dim * heads if concat else target_dim
            else:
                raise ValueError(f"Unknown GNN arch: {arch}")

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x  # (n_nodes, out_channels)

    def readout(self, node_embeddings, batch):
        """Mean-pool every graph's nodes into one vector per graph."""
        return global_mean_pool(node_embeddings, batch)


class GNNGenreClassifier(nn.Module):
    """Task 2 end-to-end model: GNNEncoder + mean-pool readout + linear head."""

    def __init__(self, in_channels, num_classes, hidden_channels=64,
                 out_channels=64, num_layers=2, arch="graphsage",
                 heads=4, dropout=0.3):
        super().__init__()
        self.gnn = GNNEncoder(in_channels, hidden_channels, out_channels, num_layers, arch,
                               heads=heads, dropout=dropout)
        self.head = nn.Linear(out_channels, num_classes)

    def forward(self, x, edge_index, batch):
        h = self.gnn(x, edge_index)
        g = self.gnn.readout(h, batch)
        return self.head(g)  # raw logits, (batch_size, num_classes)


if __name__ == "__main__":
    # Smoke test with two tiny random graphs batched together
    from torch_geometric.data import Data, Batch

    g1 = Data(x=torch.randn(5, 32), edge_index=torch.randint(0, 5, (2, 8)))
    g2 = Data(x=torch.randn(7, 32), edge_index=torch.randint(0, 7, (2, 10)))
    batch = Batch.from_data_list([g1, g2])

    model = GNNGenreClassifier(in_channels=32, num_classes=10)
    logits = model(batch.x, batch.edge_index, batch.batch)
    print("output shape:", logits.shape)  # expect (2, 10)
