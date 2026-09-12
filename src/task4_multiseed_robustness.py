"""
task4_multiseed_robustness.py
-------------------------------
Four-task paper diagnostic: is the Task 4 graph vs. no-graph comparison
stable across training seeds, or is it a one-seed accident?

Trains both the real-graph dual encoder (task4_contrastive.DualEncoder) and
the no-graph dual encoder (task4_ablation_nograph.NoGraphDualEncoder) across
several seeds, evaluates each on the same held-out test split, and reports
mean +/- std Recall@K per condition.

Read-only with respect to the canonical task4_train.py / task4_evaluate.py
checkpoints -- this script keeps its own separate checkpoints and result
file, per the project's convention of not silently overwriting an existing
result when adding a new diagnostic.
"""

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from common import set_seed, get_device, save_json
from task4_contrastive import DualEncoder, info_nce_loss, retrieval_recall_at_k
from task4_ablation_nograph import NoGraphDualEncoder
from datasets import FusionPairDataset, make_fusion_collate_fn

SEEDS = [0, 1, 2]


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


def train_one(model_cls, seed, train_ds, val_ds, test_ds, device):
    set_seed(seed)
    model = model_cls(
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
    for epoch in range(1, config.TASK4_EPOCHS + 1):
        model.train()
        for graph_batch, input_ids, attention_mask, _ in train_loader:
            graph_batch = graph_batch.to(device)
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            optimizer.zero_grad()
            g, t = model(graph_batch.x, graph_batch.edge_index, graph_batch.batch, input_ids, attention_mask)
            loss = info_nce_loss(g, t, temperature=config.TASK4_TEMPERATURE)
            loss.backward()
            optimizer.step()
        val_metrics = evaluate_split(model, val_loader, device)
        val_r10 = (val_metrics["audio_to_text_R@10"] + val_metrics["text_to_audio_R@10"]) / 2
        if val_r10 > best_recall:
            best_recall = val_r10
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return evaluate_split(model, test_loader, device)


def main():
    device = get_device()
    print(f"Using device: {device}")
    train_ds = FusionPairDataset(config.SPLITS_DIR / "task34_train.json", config.SPLITS_DIR / "tag_vocab_task34.json")
    val_ds = FusionPairDataset(config.SPLITS_DIR / "task34_val.json", config.SPLITS_DIR / "tag_vocab_task34.json")
    test_ds = FusionPairDataset(config.SPLITS_DIR / "task34_test.json", config.SPLITS_DIR / "tag_vocab_task34.json")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}  Seeds: {SEEDS}")

    results = {"real_graph": [], "no_graph": []}
    start = time.time()
    for seed in SEEDS:
        print(f"\n--- seed {seed}, real graph ---")
        m = train_one(DualEncoder, seed, train_ds, val_ds, test_ds, device)
        print(m)
        results["real_graph"].append(m)

        print(f"--- seed {seed}, no graph ---")
        m = train_one(NoGraphDualEncoder, seed, train_ds, val_ds, test_ds, device)
        print(m)
        results["no_graph"].append(m)

    summary = {}
    for cond in ["real_graph", "no_graph"]:
        summary[cond] = {}
        for k in results[cond][0]:
            vals = [r[k] for r in results[cond]]
            summary[cond][k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": vals}

    print(f"\nFinished in {(time.time()-start)/60:.1f} min.")
    print("\n=== Summary (mean +/- std across seeds) ===")
    for cond in ["real_graph", "no_graph"]:
        print(f"\n{cond}:")
        for k, v in summary[cond].items():
            print(f"  {k}: {v['mean']:.4f} +/- {v['std']:.4f}  (seeds: {[round(x,4) for x in v['values']]})")

    out = {"seeds": SEEDS, "per_seed_results": results, "summary": summary}
    out_path = config.RESULTS_DIR / "task4_multiseed_robustness.json"
    save_json(out, out_path)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
