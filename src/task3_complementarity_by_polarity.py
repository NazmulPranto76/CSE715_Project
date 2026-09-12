"""
task3_complementarity_by_polarity.py
--------------------------------------
Follow-up to task3_complementarity_diagnostic.py, prompted by a round-6
audit finding: the gnn_only checkpoint used there has 45 of 50 tags at
precision=recall=0 (metrics.json), i.e. it never predicts those tags
present at all. A cell where GNN is "right" and BERT is "wrong" could mean
two very different things:

  - true label = 1 (BERT false negative, GNN true positive): GNN actually
    detected something BERT missed -- a genuine rescue.
  - true label = 0 (BERT false positive, GNN true negative): GNN just
    predicted "absent" (its default behavior for most tags) and happened
    to be right because BERT wrongly said "present" -- not evidence of
    audio signal detection.

This script splits the existing bert_wrong_gnn_right count by true-label
polarity to tell these apart. Read-only, reuses the same evaluation path
as task3_complementarity_diagnostic.py.
"""

import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from evaluate import get_outputs_for_fusion_split  # noqa: E402


def main():
    cfg = yaml.safe_load((REPO / "config.yaml").read_text())

    bert_logits, _, labels, vocab, n = get_outputs_for_fusion_split(
        str(REPO / "results/checkpoints/fusion_bert_only_best.pt"), cfg, "test", mode="bert_only")
    gnn_logits, _, labels2, vocab2, n2 = get_outputs_for_fusion_split(
        str(REPO / "results/checkpoints/fusion_gnn_only_best.pt"), cfg, "test", mode="gnn_only")

    assert n == n2 and vocab == vocab2
    assert np.array_equal(labels, labels2)

    bert_pred = (1 / (1 + np.exp(-bert_logits)) > 0.5).astype(int)
    gnn_pred = (1 / (1 + np.exp(-gnn_logits)) > 0.5).astype(int)

    bert_correct = (bert_pred == labels)
    gnn_correct = (gnn_pred == labels)
    rescued = ~bert_correct & gnn_correct  # BERT wrong, GNN right

    is_positive = (labels == 1)
    is_negative = (labels == 0)

    # Genuine rescue: true label is 1, BERT missed it (false negative), GNN caught it (true positive)
    genuine_rescue = int(np.sum(rescued & is_positive))
    # Trivial rescue: true label is 0, BERT wrongly said present (false positive), GNN said absent (true negative)
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
        "note": "genuine_rescue = true label 1, BERT false negative, GNN true positive (GNN actually "
                "detected the tag). trivial_rescue = true label 0, BERT false positive, GNN true "
                "negative (GNN just predicted absent, its default behavior for 45/50 tags per "
                "metrics.json's fusion_gnn_only per-tag precision/recall, both 0.0).",
    }
    print(json.dumps(out, indent=2))
    out_path = REPO / "results" / "graph_repair" / "task3_complementarity_by_polarity.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
