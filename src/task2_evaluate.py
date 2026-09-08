"""
task2_evaluate.py
-------------------
Evaluates BOTH the GNN (task2_gnn.py) and the CNN baseline (task2_cnn.py)
on the GTZAN test set, and produces the required GNN-vs-CNN comparison.

Outputs:
    results/task2_metrics.json
    results/plots/task2_comparison.png

Usage:
    python src/task2_gnn.py    (train first)
    python src/task2_cnn.py    (train first)
    python src/task2_evaluate.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from common import get_device, save_json
from gnn_model import GNNGenreClassifier
from cnn_baseline import MelSpectrogramCNN
from datasets import GraphDataset, MelSpectrogramDataset
from audio_features import AudioConfig


def evaluate_gnn(device):
    test_ds = GraphDataset(config.SPLITS_DIR / "gtzan_test.json", config.SPLITS_DIR / "gtzan_genre_vocab.json")
    loader = PyGDataLoader(test_ds, batch_size=config.TASK2_GNN_BATCH_SIZE, shuffle=False)

    model = GNNGenreClassifier(
        in_channels=config.GRAPH_IN_CHANNELS, num_classes=len(test_ds.genre_vocab),
        hidden_channels=config.TASK2_GNN_HIDDEN, out_channels=config.TASK2_GNN_OUT,
        num_layers=config.TASK2_GNN_LAYERS, arch=config.TASK2_GNN_ARCH,
    ).to(device)
    model.load_state_dict(torch.load(config.CHECKPOINT_DIR / "task2_gnn_best.pt", map_location=device))
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds = model(batch.x, batch.edge_index, batch.batch).argmax(dim=1).cpu()
            all_preds.append(preds)
            all_targets.append(batch.y.cpu())
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    return {
        "accuracy": float((preds == targets).mean()),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(targets, preds, average="micro", zero_division=0)),
        "num_test_examples": len(test_ds),
    }


def evaluate_cnn(device):
    audio_cfg = AudioConfig(sample_rate=config.SAMPLE_RATE, n_mels=config.N_MELS)
    test_ds = MelSpectrogramDataset(config.SPLITS_DIR / "gtzan_test.json", audio_cfg, config.TASK2_CNN_TARGET_FRAMES)
    loader = DataLoader(test_ds, batch_size=config.TASK2_CNN_BATCH_SIZE, shuffle=False, num_workers=0)

    model = MelSpectrogramCNN(num_classes=len(config.GTZAN_GENRES)).to(device)
    model.load_state_dict(torch.load(config.CHECKPOINT_DIR / "task2_cnn_best.pt", map_location=device))
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1).cpu()
            all_preds.append(preds)
            all_targets.append(y)
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    return {
        "accuracy": float((preds == targets).mean()),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(targets, preds, average="micro", zero_division=0)),
        "num_test_examples": len(test_ds),
    }


def plot_comparison(gnn_metrics, cnn_metrics):
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["Accuracy", "Macro-F1"]
    gnn_values = [gnn_metrics["accuracy"], gnn_metrics["macro_f1"]]
    cnn_values = [cnn_metrics["accuracy"], cnn_metrics["macro_f1"]]

    x = range(len(labels))
    width = 0.35
    ax.bar([i - width / 2 for i in x], gnn_values, width, label="GNN (GraphSAGE)")
    ax.bar([i + width / 2 for i in x], cnn_values, width, label="CNN (mel-spectrogram)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("Task 2: GNN vs. CNN baseline (GTZAN test set)")
    ax.legend()
    fig.tight_layout()

    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.PLOTS_DIR / "task2_comparison.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    device = get_device()
    print("Evaluating GNN...")
    gnn_metrics = evaluate_gnn(device)
    print(f"  GNN accuracy={gnn_metrics['accuracy']:.4f}  macro_f1={gnn_metrics['macro_f1']:.4f}")

    print("Evaluating CNN baseline...")
    cnn_metrics = evaluate_cnn(device)
    print(f"  CNN accuracy={cnn_metrics['accuracy']:.4f}  macro_f1={cnn_metrics['macro_f1']:.4f}")

    metrics = {"dataset": "GTZAN", "gnn": gnn_metrics, "cnn_baseline": cnn_metrics}
    save_json(metrics, config.RESULTS_DIR / "task2_metrics.json")
    plot_comparison(gnn_metrics, cnn_metrics)

    winner = "GNN" if gnn_metrics["macro_f1"] > cnn_metrics["macro_f1"] else "CNN baseline"
    print(f"Higher Macro-F1 on this run: {winner} (reported honestly either way)")


if __name__ == "__main__":
    main()
