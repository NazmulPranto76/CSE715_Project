"""
_control_10epoch_check.py
----------------------------
One-off diagnostic (not part of the pipeline): isolates whether today's
Task 3 branch-zeroing/complementarity swing is caused by the extra epochs
(10 -> 25) or by something that was already true at the old 10-epoch
budget (e.g. this repo's uniform positive-class weighting, which the other
codebase did not use for its gnn_only complementarity checkpoint).

Trains bert_only, gnn_only, and concat for exactly 10 epochs (the OLD
budget) to separate "_ctrl10" checkpoint files, without touching the real
20/25-epoch checkpoints, then re-runs branch-zeroing and complementarity
against these control checkpoints.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from common import set_seed, get_device
from task3_fusion import GNNBertFusion
from datasets import FusionPairDataset, make_fusion_collate_fn, compute_pos_weight

CTRL_EPOCHS = 10  # the OLD budget, for comparison


def evaluate_split(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for graph_batch, input_ids, attention_mask, labels in loader:
            graph_batch = graph_batch.to(device)
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            logits, _ = model(input_ids, attention_mask, graph_batch.x, graph_batch.edge_index, graph_batch.batch)
            preds = (torch.sigmoid(logits) > 0.5).float().cpu()
            all_preds.append(preds)
            all_targets.append(labels)
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    return f1_score(targets, preds, average="macro", zero_division=0)


def train_control(mode, train_ds, val_ds, device):
    model = GNNBertFusion(
        bert_model_name=config.TASK1_MODEL_NAME, gnn_in_channels=config.GRAPH_IN_CHANNELS,
        num_labels=len(train_ds.vocab), attn_dim=config.TASK3_ATTN_DIM, mode=mode,
    ).to(device)
    collate_fn = make_fusion_collate_fn(model.bert, max_length=config.TASK1_MAX_TEXT_LEN)
    train_loader = DataLoader(train_ds, batch_size=config.TASK3_BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config.TASK3_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    pos_weight = compute_pos_weight(train_ds.records, train_ds.vocab).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.TASK3_LR)

    best_f1, best_state = -1.0, None
    t0 = time.time()
    for epoch in range(1, CTRL_EPOCHS + 1):
        model.train()
        for graph_batch, input_ids, attention_mask, labels in train_loader:
            graph_batch = graph_batch.to(device)
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(input_ids, attention_mask, graph_batch.x, graph_batch.edge_index, graph_batch.batch)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
        val_f1 = evaluate_split(model, val_loader, device)
        if val_f1 > best_f1:
            best_f1, best_state = val_f1, {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  [{mode}] epoch {epoch}/{CTRL_EPOCHS}  val_macro_f1={val_f1:.4f}")
    print(f"  [{mode}] control run done in {(time.time()-t0)/60:.1f} min, best val={best_f1:.4f}")
    ckpt_path = config.CHECKPOINT_DIR / f"task3_fusion_{mode}_ctrl10.pt"
    torch.save(best_state, ckpt_path)
    return ckpt_path


def main():
    device = get_device()
    train_ds = FusionPairDataset(config.SPLITS_DIR / "task34_train.json", config.SPLITS_DIR / "tag_vocab_task34.json")
    val_ds = FusionPairDataset(config.SPLITS_DIR / "task34_val.json", config.SPLITS_DIR / "tag_vocab_task34.json")
    test_ds = FusionPairDataset(config.SPLITS_DIR / "task34_test.json", config.SPLITS_DIR / "tag_vocab_task34.json")

    ckpts = {}
    for mode in ["bert_only", "gnn_only", "concat"]:
        set_seed(config.SEED)
        ckpts[mode] = train_control(mode, train_ds, val_ds, device)

    # --- branch-zeroing on the 10-epoch concat control checkpoint ---
    model = GNNBertFusion(
        bert_model_name=config.TASK1_MODEL_NAME, gnn_in_channels=config.GRAPH_IN_CHANNELS,
        num_labels=len(test_ds.vocab), attn_dim=config.TASK3_ATTN_DIM, mode="concat",
    ).to(device)
    model.load_state_dict(torch.load(ckpts["concat"], map_location=device))
    model.eval()
    collate_fn = make_fusion_collate_fn(model.bert, max_length=config.TASK1_MAX_TEXT_LEN)
    loader = DataLoader(test_ds, batch_size=config.TASK3_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    all_probs = {"correct": [], "null_graph": []}
    all_labels = []
    with torch.no_grad():
        for graph_batch, input_ids, attention_mask, labels in loader:
            graph_batch = graph_batch.to(device)
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            cls_vec, _ = model.bert(input_ids, attention_mask)
            node_h = model.gnn(graph_batch.x, graph_batch.edge_index)
            g = model.gnn.readout(node_h, graph_batch.batch)
            zero_g = torch.zeros_like(g)
            for cond, z in [("correct", torch.cat([g, cls_vec], -1)), ("null_graph", torch.cat([zero_g, cls_vec], -1))]:
                logits = model.tag_head(z)
                all_probs[cond].append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.numpy())
    labels_arr = np.concatenate(all_labels)
    f1s = {}
    preds_by_cond = {}
    for cond in ["correct", "null_graph"]:
        probs = np.concatenate(all_probs[cond])
        preds = (probs > 0.5).astype(int)
        preds_by_cond[cond] = preds
        f1s[cond] = f1_score(labels_arr, preds, average="macro", zero_division=0)
    pct_changed = float(np.mean(preds_by_cond["correct"] != preds_by_cond["null_graph"]) * 100)
    print("\n=== CONTROL (10-epoch) branch-zeroing ===")
    print(f"correct={f1s['correct']:.4f}  null_graph={f1s['null_graph']:.4f}  "
          f"graph_contribution={f1s['correct']-f1s['null_graph']:.4f}  pct_changed={pct_changed:.2f}%")

    # --- complementarity on the 10-epoch bert_only / gnn_only control checkpoints ---
    def get_preds(mode, ckpt):
        m = GNNBertFusion(
            bert_model_name=config.TASK1_MODEL_NAME, gnn_in_channels=config.GRAPH_IN_CHANNELS,
            num_labels=len(test_ds.vocab), attn_dim=config.TASK3_ATTN_DIM, mode=mode,
        ).to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device))
        m.eval()
        cfn = make_fusion_collate_fn(m.bert, max_length=config.TASK1_MAX_TEXT_LEN)
        ld = DataLoader(test_ds, batch_size=config.TASK3_BATCH_SIZE, shuffle=False, collate_fn=cfn)
        logits_all, labels_all = [], []
        with torch.no_grad():
            for graph_batch, input_ids, attention_mask, labels in ld:
                graph_batch = graph_batch.to(device)
                input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                logits, _ = m(input_ids, attention_mask, graph_batch.x, graph_batch.edge_index, graph_batch.batch)
                logits_all.append(logits.cpu().numpy())
                labels_all.append(labels.numpy())
        return np.concatenate(logits_all), np.concatenate(labels_all)

    bert_logits, labels_b = get_preds("bert_only", ckpts["bert_only"])
    gnn_logits, labels_g = get_preds("gnn_only", ckpts["gnn_only"])
    assert np.array_equal(labels_b, labels_g)
    bert_pred = (1 / (1 + np.exp(-bert_logits)) > 0.5).astype(int)
    gnn_pred = (1 / (1 + np.exp(-gnn_logits)) > 0.5).astype(int)
    bert_correct = bert_pred == labels_b
    gnn_correct = gnn_pred == labels_b
    rescued = ~bert_correct & gnn_correct
    is_pos = labels_b == 1
    genuine = int(np.sum(rescued & is_pos))
    trivial = int(np.sum(rescued & ~is_pos))
    total = genuine + trivial
    print("\n=== CONTROL (10-epoch) complementarity ===")
    print(f"total_rescued={total}  genuine={genuine}  trivial={trivial}  "
          f"genuine_fraction={genuine/total if total else 0:.4f}")
    print(f"gnn_only overall accuracy={np.mean(gnn_correct):.4f}")


if __name__ == "__main__":
    main()
