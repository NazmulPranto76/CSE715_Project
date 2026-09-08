"""
task3_fusion.py
----------------
Task 3: combine the BERT text tower and the GNN graph tower into one model
that predicts tags from both the caption and the audio graph.

Four modes to compare:
  - "bert_only"       : ignore the graph, use only the text vector t
  - "gnn_only"        : ignore the text, use only the graph vector g
  - "concat"          : simplest fusion -- prediction = Linear(CONCAT(g, t))
  - "cross_attention" : graph vector g "looks up" the most relevant words
                        in the caption before combining with g

concat is the simple version that should work first; cross_attention is a
fancier model built on top of the same two encoders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from bert_encoder import BertTextEncoder
from gnn_model import GNNEncoder


class CrossAttentionFusion(nn.Module):
    """
    Lets the graph embedding "look up" which words in the caption matter
    most, instead of just averaging all of them together. This is the
    same attention idea transformers use, just applied here between the
    graph and the text instead of between words.
    """

    def __init__(self, graph_dim, text_dim, attn_dim=128):
        super().__init__()
        self.attn_dim = attn_dim
        self.W_Q = nn.Linear(graph_dim, attn_dim)
        self.W_K = nn.Linear(text_dim, attn_dim)
        self.W_V = nn.Linear(text_dim, attn_dim)

    def forward(self, g, H_text, attention_mask=None):
        Q = self.W_Q(g).unsqueeze(1)             # (batch, 1, attn_dim)
        K = self.W_K(H_text)                      # (batch, seq_len, attn_dim)
        V = self.W_V(H_text)                      # (batch, seq_len, attn_dim)

        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.attn_dim ** 0.5)  # (batch, 1, seq_len)
        if attention_mask is not None:
            mask = (1.0 - attention_mask.unsqueeze(1)) * -1e9
            scores = scores + mask

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, V).squeeze(1)  # (batch, attn_dim)
        return context


class GNNBertFusion(nn.Module):
    """End-to-end Task 3 model. `mode` picks which of the 4 versions to run."""

    def __init__(self, bert_model_name, gnn_in_channels, num_labels,
                 gnn_hidden=64, gnn_out=64, gnn_layers=2,
                 attn_dim=128, mode="concat", freeze_bert=False):
        super().__init__()
        assert mode in {"bert_only", "gnn_only", "concat", "cross_attention"}
        self.mode = mode

        self.bert = BertTextEncoder(bert_model_name, freeze_backbone=freeze_bert)
        self.gnn = GNNEncoder(gnn_in_channels, gnn_hidden, gnn_out, gnn_layers)

        text_dim = self.bert.hidden_dim
        graph_dim = gnn_out

        if mode == "cross_attention":
            self.cross_attn = CrossAttentionFusion(graph_dim, text_dim, attn_dim)
            fused_dim = graph_dim + attn_dim
        elif mode == "concat":
            fused_dim = graph_dim + text_dim
        elif mode == "bert_only":
            fused_dim = text_dim
        else:  # gnn_only
            fused_dim = graph_dim

        self.tag_head = nn.Linear(fused_dim, num_labels)

    def forward(self, input_ids, attention_mask, node_x, edge_index, graph_batch):
        cls_vec, H_text = self.bert(input_ids, attention_mask)
        node_h = self.gnn(node_x, edge_index)
        g = self.gnn.readout(node_h, graph_batch)

        if self.mode == "bert_only":
            z = cls_vec
        elif self.mode == "gnn_only":
            z = g
        elif self.mode == "concat":
            z = torch.cat([g, cls_vec], dim=-1)
        else:  # cross_attention
            context = self.cross_attn(g, H_text, attention_mask)
            z = torch.cat([g, context], dim=-1)

        return self.tag_head(z), z  # z is also returned for the t-SNE plot


if __name__ == "__main__":
    print("Import-only smoke test -- run task3_train.py for a real forward pass.")
