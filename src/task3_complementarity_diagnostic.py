"""
task3_complementarity_diagnostic.py
------------------------------------
Four-task paper diagnostic: does the GNN branch solve any (example, tag)
predictions that the BERT-only branch gets wrong?

Reuses evaluate.py's existing get_outputs_for_fusion_split() unchanged --
loads the already-trained fusion_bert_only_best.pt and fusion_gnn_only_best.pt
checkpoints (mode="bert_only" and mode="gnn_only"), scores both on the same
MusicCaps test split in the same order, thresholds at 0.5, and cross-tabulates
per-(example, tag) correctness into the four cells the paper needs:
BERT-right/GNN-right, BERT-right/GNN-wrong, BERT-wrong/GNN-right,
BERT-wrong/GNN-wrong.

Does not retrain or modify anything. Read-only diagnostic.
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

    assert n == n2 and vocab == vocab2, "checkpoints scored on different splits/vocab -- not comparable"
    assert np.array_equal(labels, labels2), "label arrays differ between the two eval passes -- ordering mismatch"

    bert_pred = (1 / (1 + np.exp(-bert_logits)) > 0.5).astype(int)
    gnn_pred = (1 / (1 + np.exp(-gnn_logits)) > 0.5).astype(int)

    bert_correct = (bert_pred == labels)
    gnn_correct = (gnn_pred == labels)

    both_right = int(np.sum(bert_correct & gnn_correct))
    bert_right_gnn_wrong = int(np.sum(bert_correct & ~gnn_correct))
    bert_wrong_gnn_right = int(np.sum(~bert_correct & gnn_correct))
    both_wrong = int(np.sum(~bert_correct & ~gnn_correct))
    total = bert_correct.size

    print(f"Test examples: {n}, tags: {len(vocab)}, total (example,tag) cells: {total}")
    print(f"BERT-right / GNN-right: {both_right} ({100*both_right/total:.2f}%)")
    print(f"BERT-right / GNN-wrong: {bert_right_gnn_wrong} ({100*bert_right_gnn_wrong/total:.2f}%)")
    print(f"BERT-wrong / GNN-right: {bert_wrong_gnn_right} ({100*bert_wrong_gnn_right/total:.2f}%)")
    print(f"BERT-wrong / GNN-wrong: {both_wrong} ({100*both_wrong/total:.2f}%)")

    # Among cells BERT got wrong, what fraction does GNN rescue?
    bert_wrong_total = bert_wrong_gnn_right + both_wrong
    rescue_rate = bert_wrong_gnn_right / bert_wrong_total if bert_wrong_total else 0.0
    print(f"\nOf {bert_wrong_total} cells BERT gets wrong, GNN is right on {bert_wrong_gnn_right} "
          f"({100*rescue_rate:.2f}%) -- 'GNN rescue rate'.")
    # Baseline: fraction of ALL cells GNN gets right, for comparison (is the
    # rescue rate above or below GNN's overall accuracy?)
    gnn_overall_acc = float(np.mean(gnn_correct))
    print(f"GNN's overall per-cell accuracy (for comparison): {100*gnn_overall_acc:.2f}%")
    print("If the rescue rate is close to GNN's overall accuracy, GNN is not")
    print("preferentially right exactly where BERT is wrong -- no evidence of")
    print("complementary specialization, just ordinary background accuracy.")

    # Per-tag breakdown
    per_tag_rows = []
    for i, tag in enumerate(vocab):
        b_correct_t = bert_correct[:, i]
        g_correct_t = gnn_correct[:, i]
        bert_wrong_t = int(np.sum(~b_correct_t))
        rescued_t = int(np.sum(~b_correct_t & g_correct_t))
        rate_t = rescued_t / bert_wrong_t if bert_wrong_t else None
        per_tag_rows.append({
            "tag": tag, "bert_wrong_count": bert_wrong_t, "gnn_rescued_count": rescued_t,
            "gnn_rescue_rate": rate_t, "n_test": n,
        })

    out = {
        "n_test_examples": n, "n_tags": len(vocab),
        "both_right": both_right, "bert_right_gnn_wrong": bert_right_gnn_wrong,
        "bert_wrong_gnn_right": bert_wrong_gnn_right, "both_wrong": both_wrong,
        "total_cells": total, "gnn_rescue_rate_overall": rescue_rate,
        "gnn_overall_cell_accuracy": gnn_overall_acc,
        "per_tag": per_tag_rows,
    }
    out_dir = REPO / "results" / "graph_repair"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "task3_complementarity_diagnostic.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
