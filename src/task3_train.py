"""
task3_train.py
----------------
Task 3: train the GNN-BERT fusion model to predict tags from BOTH a
caption and its audio graph.

Trains all four required ablation variants, one after another:
    bert_only, gnn_only, concat, cross_attention
so they can be compared fairly (same data, same training setup).

Usage:
    python src/prepare_task34_graphs.py     (run once first)
    python src/task3_train.py
"""

import sys
import time
from pathlib import Path

import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from common import set_seed, get_device, save_json
from task3_fusion import GNNBertFusion
from datasets import FusionPairDataset, make_fusion_collate_fn, compute_pos_weight

MODES = ["bert_only", "gnn_only", "concat", "cross_attention"]


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
    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(targets, preds, average="micro", zero_division=0)
    return macro_f1, micro_f1


def train_one_mode(mode, train_ds, val_ds, device):
    print(f"\n=== Training Task 3 fusion, mode = {mode} ===")
    model = GNNBertFusion(
        bert_model_name=config.TASK1_MODEL_NAME, gnn_in_channels=config.GRAPH_IN_CHANNELS,
        num_labels=len(train_ds.vocab), attn_dim=config.TASK3_ATTN_DIM, mode=mode,
    ).to(device)

    collate_fn = make_fusion_collate_fn(model.bert, max_length=config.TASK1_MAX_TEXT_LEN)
    train_loader = DataLoader(train_ds, batch_size=config.TASK3_BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config.TASK3_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    pos_weight = compute_pos_weight(train_ds.records, train_ds.vocab).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.TASK3_LR)

    history = {"train_loss": [], "val_macro_f1": [], "val_micro_f1": []}
    best_macro_f1 = -1.0
    best_path = config.CHECKPOINT_DIR / f"task3_fusion_{mode}_best.pt"

    start_time = time.time()
    for epoch in range(1, config.TASK3_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for graph_batch, input_ids, attention_mask, labels in train_loader:
            graph_batch = graph_batch.to(device)
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            optimizer.zero_grad()
            logits, _ = model(input_ids, attention_mask, graph_batch.x, graph_batch.edge_index, graph_batch.batch)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(labels)

        epoch_loss /= len(train_ds)
        val_macro_f1, val_micro_f1 = evaluate_split(model, val_loader, device)
        history["train_loss"].append(epoch_loss)
        history["val_macro_f1"].append(val_macro_f1)
        history["val_micro_f1"].append(val_micro_f1)
        print(f"  Epoch {epoch}/{config.TASK3_EPOCHS}  loss={epoch_loss:.4f}  "
              f"val_macro_f1={val_macro_f1:.4f}  val_micro_f1={val_micro_f1:.4f}")

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            torch.save(model.state_dict(), best_path)

    elapsed_min = (time.time() - start_time) / 60
    print(f"  [{mode}] finished in {elapsed_min:.1f} min. Best val Macro-F1: {best_macro_f1:.4f}")
    save_json(history, config.RESULTS_DIR / f"task3_fusion_{mode}_train_history.json")


def main():
    set_seed(config.SEED)
    device = get_device()
    print(f"Using device: {device}")
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = FusionPairDataset(config.SPLITS_DIR / "task34_train.json", config.SPLITS_DIR / "tag_vocab_task34.json")
    val_ds = FusionPairDataset(config.SPLITS_DIR / "task34_val.json", config.SPLITS_DIR / "tag_vocab_task34.json")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Tags: {len(train_ds.vocab)}")

    for mode in MODES:
        set_seed(config.SEED)  # same starting point for every ablation variant, for a fair comparison
        train_one_mode(mode, train_ds, val_ds, device)


if __name__ == "__main__":
    main()
