"""
task3_branch_zeroing.py
--------------------------
Reproduces, from this canonical codebase, the branch-zeroing diagnostic the
paper cites -- previously only available as
project/triguard-lp/src/branch_contribution_diagnostic.py, a terminated
research direction's directory (flagged by a round-6 audit as a
reproducibility gap: the number was real, but not reproducible from either
codebase named as this paper's evidence base).

Loads the trained fusion_concat_best.pt checkpoint (mode="concat"), and for
each test example computes tag logits under four conditions by directly
manipulating the fused vector z = concat(g, cls_vec), with no retraining:
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
import yaml
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from fusion_model import GNNBertFusion  # noqa: E402
from fusion_dataset import FusionPairDataset, make_fusion_collate_fn  # noqa: E402


def main():
    cfg = yaml.safe_load((REPO / "config.yaml").read_text())
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    fcfg, bcfg, gcfg, dcfg = cfg["fusion"], cfg["bert"], cfg["gnn"], cfg["data"]

    splits_dir = Path(dcfg["splits_dir"])
    vocab_path = splits_dir / f"tag_vocab_top{dcfg['top_k_tags']}.json"
    split_path = splits_dir / "musiccaps_audio_test.json"
    ds = FusionPairDataset(split_path, vocab_path)
    num_labels = len(ds.vocab)

    model = GNNBertFusion(
        bert_model_name=bcfg["model_name"], gnn_in_channels=gcfg["in_channels"],
        num_labels=num_labels, gnn_hidden=fcfg["gnn_hidden"], gnn_out=fcfg["gnn_out"],
        gnn_layers=fcfg["gnn_layers"], gnn_arch=fcfg["gnn_arch"], attn_dim=fcfg["attn_dim"],
        mode="concat", use_emotion_head=fcfg.get("use_emotion_head", False),
        freeze_bert=fcfg.get("freeze_bert", False),
    ).to(device)
    model.load_state_dict(torch.load(REPO / "results/checkpoints/fusion_concat_best.pt", map_location=device))
    model.eval()

    collate_fn = make_fusion_collate_fn(model.bert, max_length=dcfg["max_text_len"])
    loader = torch.utils.data.DataLoader(ds, batch_size=fcfg["batch_size"], shuffle=False, collate_fn=collate_fn)

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
        results[c] = {
            "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        }

    pct_predictions_changed = float(np.mean(preds_by_cond["correct"] != preds_by_cond["null_graph"]) * 100)

    out = {
        "n_test": len(labels),
        "n_tags": labels.shape[1],
        "checkpoint": "fusion_concat_best.pt",
        "results": results,
        "graph_contribution_macro_f1": results["correct"]["macro_f1"] - results["null_graph"]["macro_f1"],
        "text_contribution_macro_f1": results["correct"]["macro_f1"] - results["null_text"]["macro_f1"],
        "pct_individual_predictions_changed_when_graph_zeroed": pct_predictions_changed,
    }
    print(json.dumps(out, indent=2))
    out_path = REPO / "results" / "graph_repair" / "task3_branch_zeroing_reproduced.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
