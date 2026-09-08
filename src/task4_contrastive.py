"""
task4_contrastive.py
----------------------
Task 4: a dual encoder that learns to match a caption with its audio
graph (and vice versa), without ever seeing tag labels.

Both the graph and the caption get turned into a vector in the same
128-dim space. Training pulls a matching (graph, caption) pair's vectors
closer together and pushes every other pair in the batch further apart --
this is the InfoNCE loss below.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from bert_encoder import BertTextEncoder
from gnn_model import GNNEncoder


class DualEncoder(nn.Module):
    """Projects graph and text embeddings into one shared, L2-normalized space."""

    def __init__(self, bert_model_name, gnn_in_channels, gnn_hidden=64,
                 gnn_out=64, gnn_layers=2, embed_dim=128, freeze_bert=False):
        super().__init__()
        self.bert = BertTextEncoder(bert_model_name, freeze_backbone=freeze_bert)
        self.gnn = GNNEncoder(gnn_in_channels, gnn_hidden, gnn_out, gnn_layers)

        self.graph_proj = nn.Linear(gnn_out, embed_dim)
        self.text_proj = nn.Linear(self.bert.hidden_dim, embed_dim)

    def encode_graph(self, node_x, edge_index, graph_batch):
        node_h = self.gnn(node_x, edge_index)
        g = self.gnn.readout(node_h, graph_batch)
        g = self.graph_proj(g)
        return F.normalize(g, dim=-1)

    def encode_text(self, input_ids, attention_mask):
        cls_vec, _ = self.bert(input_ids, attention_mask)
        t = self.text_proj(cls_vec)
        return F.normalize(t, dim=-1)

    def forward(self, node_x, edge_index, graph_batch, input_ids, attention_mask):
        g = self.encode_graph(node_x, edge_index, graph_batch)
        t = self.encode_text(input_ids, attention_mask)
        return g, t


def info_nce_loss(g, t, temperature=0.07):
    """
    Symmetric InfoNCE over a batch of N (graph, caption) pairs, with the
    correct match assumed to be on the diagonal (g[i] pairs with t[i]).
    """
    logits = g @ t.t() / temperature
    labels = torch.arange(g.size(0), device=g.device)

    loss_graph_to_text = F.cross_entropy(logits, labels)
    loss_text_to_graph = F.cross_entropy(logits.t(), labels)
    return (loss_graph_to_text + loss_text_to_graph) / 2.0


def retrieval_recall_at_k(g, t, ks=(1, 5, 10)):
    """
    Computes Audio->Caption and Caption->Audio Recall@K on a full set of
    embeddings (encode everything first, then call this once).
    """
    with torch.no_grad():
        n = g.size(0)
        sim = g @ t.t()
        results = {}

        for k in ks:
            k_eff = min(k, n)  # can't ask for more neighbors than there are examples

            topk_audio_to_text = sim.topk(k_eff, dim=1).indices
            correct_a2t = (topk_audio_to_text == torch.arange(n, device=g.device).unsqueeze(1)).any(dim=1)
            results[f"audio_to_text_R@{k}"] = correct_a2t.float().mean().item()

            topk_text_to_audio = sim.t().topk(k_eff, dim=1).indices
            correct_t2a = (topk_text_to_audio == torch.arange(n, device=g.device).unsqueeze(1)).any(dim=1)
            results[f"text_to_audio_R@{k}"] = correct_t2a.float().mean().item()

    return results


if __name__ == "__main__":
    # Smoke test with random embeddings (no real encoders needed)
    g = F.normalize(torch.randn(8, 128), dim=-1)
    t = F.normalize(torch.randn(8, 128), dim=-1)
    print("InfoNCE loss:", info_nce_loss(g, t).item())
    print("Retrieval metrics:", retrieval_recall_at_k(g, t))
