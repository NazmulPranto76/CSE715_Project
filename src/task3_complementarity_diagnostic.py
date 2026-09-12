"""
task3_complementarity_diagnostic.py
--------------------------------------
Cross-tabulates the retrained (20/25-epoch) bert_only and gnn_only
checkpoints' per-(example, tag) predictions on the test set: is the
graph-only model right on a distinct subset of cases the text-only model
gets wrong, or is it just noise around its own overall accuracy?

Read-only. Does not retrain or modify anything.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from common import get_device
from task3_fusion import GNNBertFusion
from datasets import FusionPairDataset, make_fusion_collate_fn


def get_predictions(mode, test_ds, device):
    model = GNNBertFusion(
        bert_model_name=config.TASK1_MODEL_NAME, gnn_in_channels=config.GRAPH_IN_CHANNELS,
        num_labels=len(test_ds.vocab), attn_dim=config.TASK3_ATTN_DIM, mode=mode,
    ).to(device)
    model.load_state_dict(torch.load(config.CHECKPOINT_DIR / f"task3_fusion_{mode}_best.pt", map_location=device))
    model.eval()

    collate_fn = make_fusion_collate_fn(model.bert, max_length=config.TASK1_MAX_TEXT_LEN)
    loader = DataLoader(test_ds, batch_size=config.TASK3_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    all_logits, all_labels = [], []
    with torch.no_grad():
        for graph_batch, input_ids, attention_mask, labels in loader:
            graph_batch = graph_batch.to(device)
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            logits, _ = model(input_ids, attention_mask, graph_batch.x, graph_batch.edge_index, graph_batch.batch)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


def main():
    device = get_device()
    test_ds = FusionPairDataset(config.SPLITS_DIR / "task34_test.json", config.SPLITS_DIR / "tag_vocab_task34.json")

    bert_logits, labels = get_predictions("bert_only", test_ds, device)
    gnn_logits, labels2 = get_predictions("gnn_only", test_ds, device)
    assert np.array_equal(labels, labels2), "label arrays differ between the two eval passes"

    bert_pred = (1 / (1 + np.exp(-bert_logits)) > 0.5).astype(int)
    gnn_pred = (1 / (1 + np.exp(-gnn_logits)) > 0.5).astype(int)

    bert_correct = (bert_pred == labels)
    gnn_correct = (gnn_pred == labels)

    both_right = int(np.sum(bert_correct & gnn_correct))
    bert_right_gnn_wrong = int(np.sum(bert_correct & ~gnn_correct))
    bert_wrong_gnn_right = int(np.sum(~bert_correct & gnn_correct))
    both_wrong = int(np.sum(~bert_correct & ~gnn_correct))
    total = bert_correct.size

    bert_wrong_total = bert_wrong_gnn_right + both_wrong
    rescue_rate = bert_wrong_gnn_right / bert_wrong_total if bert_wrong_total else 0.0
    gnn_overall_acc = float(np.mean(gnn_correct))

    out = {
        "n_test_examples": int(labels.shape[0]), "n_tags": int(labels.shape[1]),
        "both_right": both_right, "bert_right_gnn_wrong": bert_right_gnn_wrong,
        "bert_wrong_gnn_right": bert_wrong_gnn_right, "both_wrong": both_wrong,
        "total_cells": total,
        "gnn_rescue_rate_overall": rescue_rate,
        "gnn_overall_cell_accuracy": gnn_overall_acc,
        "checkpoints": {"bert_only": "task3_fusion_bert_only_best.pt", "gnn_only": "task3_fusion_gnn_only_best.pt"},
    }
    print(json.dumps(out, indent=2))
    out_path = config.RESULTS_DIR / "task3_complementarity_diagnostic.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
