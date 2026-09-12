"""
task3_branch_zeroing_v2.py
----------------------------
Branch-zeroing diagnostic for the retrained (20/25-epoch) checkpoints in
this repo. Loads task3_fusion_concat_best.pt and, for each test example,
computes tag logits under four conditions by directly manipulating the
fused vector z = concat(g, cls_vec), with no retraining:
  correct     : z = concat(g, cls_vec)          -- both branches real
  null_graph  : z = concat(0, cls_vec)           -- graph branch zeroed
  null_text   : z = concat(g, 0)                 -- text branch zeroed
  both_null   : z = concat(0, 0)                 -- prior-only floor

Read-only. Does not modify the checkpoint or retrain anything.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from common import get_device
from task3_fusion import GNNBertFusion
from datasets import FusionPairDataset, make_fusion_collate_fn


def main():
    device = get_device()
    test_ds = FusionPairDataset(config.SPLITS_DIR / "task34_test.json", config.SPLITS_DIR / "tag_vocab_task34.json")

    model = GNNBertFusion(
        bert_model_name=config.TASK1_MODEL_NAME, gnn_in_channels=config.GRAPH_IN_CHANNELS,
        num_labels=len(test_ds.vocab), attn_dim=config.TASK3_ATTN_DIM, mode="concat",
    ).to(device)
    model.load_state_dict(torch.load(config.CHECKPOINT_DIR / "task3_fusion_concat_best.pt", map_location=device))
    model.eval()

    collate_fn = make_fusion_collate_fn(model.bert, max_length=config.TASK1_MAX_TEXT_LEN)
    loader = DataLoader(test_ds, batch_size=config.TASK3_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    conditions = ["correct", "null_graph", "null_text", "both_null"]
    all_probs = {c: [] for c in conditions}
    all_labels = []

    with torch.no_grad():
        for graph_batch, input_ids, attention_mask, labels in loader:
            graph_batch = graph_batch.to(device)
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

            cls_vec, _ = model.bert(input_ids, attention_mask)
            node_h = model.gnn(graph_batch.x, graph_batch.edge_index)
            g = model.gnn.readout(node_h, graph_batch.batch)

            zero_g = torch.zeros_like(g)
            zero_t = torch.zeros_like(cls_vec)

            z_by_cond = {
                "correct": torch.cat([g, cls_vec], dim=-1),
                "null_graph": torch.cat([zero_g, cls_vec], dim=-1),
                "null_text": torch.cat([g, zero_t], dim=-1),
                "both_null": torch.cat([zero_g, zero_t], dim=-1),
            }
            for c in conditions:
                logits = model.tag_head(z_by_cond[c])
                all_probs[c].append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.numpy())

    labels = np.concatenate(all_labels)
    results = {}
    preds_by_cond = {}
    for c in conditions:
        probs = np.concatenate(all_probs[c])
        preds = (probs > 0.5).astype(int)
        preds_by_cond[c] = preds
        results[c] = {"macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0))}

    pct_changed = float(np.mean(preds_by_cond["correct"] != preds_by_cond["null_graph"]) * 100)

    out = {
        "n_test": len(labels),
        "n_tags": labels.shape[1],
        "checkpoint": "task3_fusion_concat_best.pt",
        "results": results,
        "graph_contribution_macro_f1": results["correct"]["macro_f1"] - results["null_graph"]["macro_f1"],
        "text_contribution_macro_f1": results["correct"]["macro_f1"] - results["null_text"]["macro_f1"],
        "pct_individual_predictions_changed_when_graph_zeroed": pct_changed,
    }
    print(json.dumps(out, indent=2))
    out_path = config.RESULTS_DIR / "task3_branch_zeroing.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
