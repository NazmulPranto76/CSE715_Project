"""
task3_complementarity_by_polarity.py
--------------------------------------
Follow-up to task3_complementarity_diagnostic.py: splits the
bert_wrong_gnn_right cells by true-label polarity, to tell apart:
  - true label = 1 (bert false negative, gnn true positive): gnn actually
    detected something bert missed -- a genuine rescue.
  - true label = 0 (bert false positive, gnn true negative): gnn just
    predicted "absent" and happened to be right because bert wrongly said
    "present" -- not evidence of audio signal detection.

Read-only. Reuses the same evaluation path as
task3_complementarity_diagnostic.py.
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
    assert np.array_equal(labels, labels2)

    bert_pred = (1 / (1 + np.exp(-bert_logits)) > 0.5).astype(int)
    gnn_pred = (1 / (1 + np.exp(-gnn_logits)) > 0.5).astype(int)

    bert_correct = (bert_pred == labels)
    gnn_correct = (gnn_pred == labels)
    rescued = ~bert_correct & gnn_correct

    is_positive = (labels == 1)
    is_negative = (labels == 0)

    genuine_rescue = int(np.sum(rescued & is_positive))
    trivial_rescue = int(np.sum(rescued & is_negative))
    total_rescue = genuine_rescue + trivial_rescue

    bert_false_negatives = int(np.sum(~bert_correct & is_positive))
    bert_false_positives = int(np.sum(~bert_correct & is_negative))

    out = {
        "total_bert_wrong_gnn_right": total_rescue,
        "genuine_rescue_true_positive": genuine_rescue,
        "trivial_rescue_true_negative": trivial_rescue,
        "genuine_rescue_fraction_of_total_rescues": genuine_rescue / total_rescue if total_rescue else None,
        "bert_false_negatives_total": bert_false_negatives,
        "bert_false_positives_total": bert_false_positives,
        "genuine_rescue_rate_among_bert_false_negatives": genuine_rescue / bert_false_negatives if bert_false_negatives else None,
        "trivial_rescue_rate_among_bert_false_positives": trivial_rescue / bert_false_positives if bert_false_positives else None,
        "note": "genuine_rescue = true label 1, bert false negative, gnn true positive. "
                "trivial_rescue = true label 0, bert false positive, gnn true negative.",
    }
    print(json.dumps(out, indent=2))
    out_path = config.RESULTS_DIR / "task3_complementarity_by_polarity.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
