"""
task3_evaluate.py
-------------------
Evaluates all four Task 3 ablation variants (bert_only, gnn_only, concat,
cross_attention) on the held-out test set and writes the required
comparison table + plot.

Outputs:
    results/task3_ablation.csv
    results/plots/task3_ablation.png
    results/plots/task3_tsne_concat.png   (bonus: t-SNE of the fused embeddings)

Usage:
    python src/task3_train.py     (train all 4 modes first)
    python src/task3_evaluate.py
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, average_precision_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from common import get_device
from task3_fusion import GNNBertFusion
from datasets import FusionPairDataset, make_fusion_collate_fn
from task3_train import MODES


def evaluate_mode(mode, test_ds, device):
    model = GNNBertFusion(
        bert_model_name=config.TASK1_MODEL_NAME, gnn_in_channels=config.GRAPH_IN_CHANNELS,
        num_labels=len(test_ds.vocab), attn_dim=config.TASK3_ATTN_DIM, mode=mode,
    ).to(device)
    ckpt_path = config.CHECKPOINT_DIR / f"task3_fusion_{mode}_best.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    collate_fn = make_fusion_collate_fn(model.bert, max_length=config.TASK1_MAX_TEXT_LEN)
    loader = DataLoader(test_ds, batch_size=config.TASK3_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    all_probs, all_targets, all_z = [], [], []
    with torch.no_grad():
        for graph_batch, input_ids, attention_mask, labels in loader:
            graph_batch = graph_batch.to(device)
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            logits, z = model(input_ids, attention_mask, graph_batch.x, graph_batch.edge_index, graph_batch.batch)
            all_probs.append(torch.sigmoid(logits).cpu())
            all_targets.append(labels)
            all_z.append(z.cpu())

    probs = torch.cat(all_probs).numpy()
    targets = torch.cat(all_targets).numpy()
    z = torch.cat(all_z).numpy()
    preds = (probs > 0.5).astype(float)

    try:
        auc_pr = average_precision_score(targets, probs, average="macro")
    except ValueError:
        auc_pr = None

    metrics = {
        "mode": mode,
        "macro_f1": f1_score(targets, preds, average="macro", zero_division=0),
        "micro_f1": f1_score(targets, preds, average="micro", zero_division=0),
        "auc_pr_macro": auc_pr,
    }
    return metrics, z, targets


def plot_ablation(all_metrics):
    fig, ax = plt.subplots(figsize=(7, 4))
    modes = [m["mode"] for m in all_metrics]
    macro_f1s = [m["macro_f1"] for m in all_metrics]
    micro_f1s = [m["micro_f1"] for m in all_metrics]

    x = range(len(modes))
    width = 0.35
    ax.bar([i - width / 2 for i in x], macro_f1s, width, label="Macro-F1")
    ax.bar([i + width / 2 for i in x], micro_f1s, width, label="Micro-F1")
    ax.set_xticks(list(x))
    ax.set_xticklabels(modes, rotation=15)
    ax.set_ylim(0, 1)
    ax.set_title("Task 3: fusion ablation (MusicCaps test set)")
    ax.legend()
    fig.tight_layout()

    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.PLOTS_DIR / "task3_ablation.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_tsne(z, targets, vocab):
    """Bonus: 2D t-SNE of the fused embeddings, colored by the most common tag present."""
    if len(z) < 10:
        return  # too few points for a meaningful t-SNE plot
    tsne = TSNE(n_components=2, random_state=config.SEED, perplexity=min(30, len(z) - 1))
    z_2d = tsne.fit_transform(z)

    # Color each point by its first true tag, just for a readable legend.
    top_tag_idx = targets.argmax(axis=1)
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=top_tag_idx, cmap="tab20", s=15)
    ax.set_title("Task 3 (concat mode): t-SNE of fused embeddings")
    fig.tight_layout()

    out_path = config.PLOTS_DIR / "task3_tsne_concat.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    device = get_device()
    test_ds = FusionPairDataset(config.SPLITS_DIR / "task34_test.json", config.SPLITS_DIR / "tag_vocab_task34.json")
    print(f"Test examples: {len(test_ds)}  Tags: {len(test_ds.vocab)}")

    all_metrics = []
    concat_z, concat_targets = None, None
    for mode in MODES:
        metrics, z, targets = evaluate_mode(mode, test_ds, device)
        print(f"  {mode}: macro_f1={metrics['macro_f1']:.4f}  micro_f1={metrics['micro_f1']:.4f}  "
              f"auc_pr={metrics['auc_pr_macro']}")
        all_metrics.append(metrics)
        if mode == "concat":
            concat_z, concat_targets = z, targets

    csv_path = config.RESULTS_DIR / "task3_ablation.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "macro_f1", "micro_f1", "auc_pr_macro"])
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"Saved {csv_path}")

    plot_ablation(all_metrics)
    if concat_z is not None:
        plot_tsne(concat_z, concat_targets, test_ds.vocab)


if __name__ == "__main__":
    main()
