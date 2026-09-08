"""
graph_builder.py
-----------------
Turns a track's per-segment audio features into a graph that PyTorch
Geometric can train on.

Nodes  = time segments (e.g. one node per 2 seconds of audio)
Edges  = (a) temporal adjacency: segment i connects to segment i+1
         (b) similarity: segment i connects to segment j if their features
             are cosine-similar above a threshold (captures repeated
             sections, e.g. a chorus that comes back later in the song)
"""

import numpy as np
import torch
from torch_geometric.data import Data


def cosine_similarity_matrix(features):
    """features: (n_nodes, feat_dim) -> (n_nodes, n_nodes) cosine similarity matrix."""
    norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    return norm @ norm.T


def build_segment_graph(segment_features, similarity_threshold=0.8):
    """
    segment_features: (n_segments, feat_dim), one row per time segment.

    Returns a PyG Data object with:
      x          : (n_segments, feat_dim) node features
      edge_index : (2, n_edges)
      edge_attr  : (n_edges, 1) edge weight (1.0 for temporal edges, cosine
                   similarity value for similarity edges)
    """
    n = segment_features.shape[0]
    sim = cosine_similarity_matrix(segment_features)

    edges = []
    weights = []

    # (a) temporal edges: connect every segment to the next one, both directions
    for i in range(n - 1):
        edges += [(i, i + 1), (i + 1, i)]
        weights += [1.0, 1.0]

    # (b) similarity edges: connect segments that "sound alike", skipping
    # pairs already connected above so we don't add a duplicate edge
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1:
                continue  # already added as a temporal edge
            if sim[i, j] > similarity_threshold:
                edges += [(i, j), (j, i)]
                weights += [float(sim[i, j]), float(sim[i, j])]

    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(weights, dtype=torch.float).unsqueeze(-1)

    x = torch.tensor(segment_features, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


if __name__ == "__main__":
    # Smoke test with random segment features (10 nodes, 12-dim features)
    fake_segments = np.random.randn(10, 12)
    graph = build_segment_graph(fake_segments, similarity_threshold=0.8)
    print(graph)
    print("nodes:", graph.num_nodes, "edges:", graph.num_edges)
