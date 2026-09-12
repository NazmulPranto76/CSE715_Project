"""
task4_ablation_nograph.py
---------------------------
Extends the same graph-necessity question the audit paper asked of Tasks 2/3
(materials/paper_latex/) to Task 4's contrastive retrieval: does the GNN
encoder's structure actually help retrieval, or would a structure-free
encoder over the same node features do just as well?

Builds a no-graph variant of the Task 4 dual encoder -- same per-node
feature width, same depth, same mean-pool readout, but a per-node MLP
instead of GraphSAGE (no edges, no cross-node computation) -- trains it
identically to task4_train.py, and reports test-set Recall@K side by side
with the real-graph run's numbers (read from results/task4_retrieval.json,
already produced by task4_train.py + task4_evaluate.py).

Does not modify task4_contrastive.py, task4_train.py, or any existing
result file.
"""

import copy
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from common import set_seed, get_device, save_json
from bert_encoder import BertTextEncoder
from task4_contrastive import info_nce_loss, retrieval_recall_at_k
from datasets import FusionPairDataset, make_fusion_collate_fn
from torch_geometric.nn import global_mean_pool


class NoGraphEncoder(nn.Module):
    """Same width/depth as GNNEncoder, but a per-node MLP -- no edges, no
    cross-node computation -- matching the audit paper's structure-free
    baseline definition exactly (Section 4.3, 04_setup.tex)."""

    def __init__(self, in_channels, hidden_channels=64, out_channels=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

    def forward(self, x, edge_index):  # edge_index accepted but ignored -- the point of the ablation
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def readout(self, node_embeddings, batch):
        return global_mean_pool(node_embeddings, batch)


class NoGraphDualEncoder(nn.Module):
    def __init__(self, bert_model_name, gnn_in_channels, gnn_hidden=64,
                 gnn_out=64, gnn_layers=2, embed_dim=128, freeze_bert=False):
        super().__init__()
        self.bert = BertTextEncoder(bert_model_name, freeze_backbone=freeze_bert)
        self.gnn = NoGraphEncoder(gnn_in_channels, gnn_hidden, gnn_out, gnn_layers)
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


def evaluate_split(model, loader, device):
    model.eval()
    all_g, all_t = [], []
    with torch.no_grad():
        for graph_batch, input_ids, attention_mask, _ in loader:
            graph_batch = graph_batch.to(device)
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            g, t = model(graph_batch.x, graph_batch.edge_index, graph_batch.batch, input_ids, attention_mask)
            all_g.append(g.cpu())
            all_t.append(t.cpu())
    return retrieval_recall_at_k(torch.cat(all_g), torch.cat(all_t))


def main():
    set_seed(config.SEED)
    device = get_device()
    print(f"Using device: {device}")

    train_ds = FusionPairDataset(config.SPLITS_DIR / "task34_train.json", config.SPLITS_DIR / "tag_vocab_task34.json")
    val_ds = FusionPairDataset(config.SPLITS_DIR / "task34_val.json", config.SPLITS_DIR / "tag_vocab_task34.json")
    test_ds = FusionPairDataset(config.SPLITS_DIR / "task34_test.json", config.SPLITS_DIR / "tag_vocab_task34.json")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    model = NoGraphDualEncoder(
        bert_model_name=config.TASK1_MODEL_NAME, gnn_in_channels=config.GRAPH_IN_CHANNELS,
        embed_dim=config.TASK4_EMBED_DIM,
    ).to(device)

    collate_fn = make_fusion_collate_fn(model.bert, max_length=config.TASK1_MAX_TEXT_LEN)
    train_loader = DataLoader(train_ds, batch_size=config.TASK4_BATCH_SIZE, shuffle=True,
                               collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.TASK4_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=config.TASK4_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.TASK4_LR)

    best_recall, best_state = -1.0, None
    start_time = time.time()
    for epoch in range(1, config.TASK4_EPOCHS + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for graph_batch, input_ids, attention_mask, _ in train_loader:
            graph_batch = graph_batch.to(device)
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            optimizer.zero_grad()
            g, t = model(graph_batch.x, graph_batch.edge_index, graph_batch.batch, input_ids, attention_mask)
            loss = info_nce_loss(g, t, temperature=config.TASK4_TEMPERATURE)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        epoch_loss /= max(1, n_batches)

        val_metrics = evaluate_split(model, val_loader, device)
        val_r10 = (val_metrics["audio_to_text_R@10"] + val_metrics["text_to_audio_R@10"]) / 2
        print(f"[no-graph] Epoch {epoch}/{config.TASK4_EPOCHS}  loss={epoch_loss:.4f}  val_avg_R@10={val_r10:.4f}")
        if val_r10 > best_recall:
            best_recall = val_r10
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    test_metrics = evaluate_split(model, test_loader, device)
    elapsed_min = (time.time() - start_time) / 60
    print(f"\n[no-graph] Training finished in {elapsed_min:.1f} min. Best val R@10: {best_recall:.4f}")
    print("[no-graph] TEST metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    out = {"condition": "no_graph_mlp_ablation", "best_val_avg_r10": best_recall, "test_metrics": test_metrics}
    save_json(out, config.RESULTS_DIR / "task4_ablation_nograph.json")
    print(f"\nSaved: {config.RESULTS_DIR / 'task4_ablation_nograph.json'}")

    real_graph_path = config.RESULTS_DIR / "task4_retrieval.json"
    if real_graph_path.exists():
        real = json.loads(real_graph_path.read_text())
        print("\n=== Real-graph vs no-graph, side by side ===")
        for k in test_metrics:
            rv = real.get(k)
            if rv is not None:
                print(f"  {k}: real_graph={rv:.4f}  no_graph={test_metrics[k]:.4f}  diff={rv - test_metrics[k]:+.4f}")


if __name__ == "__main__":
    main()
