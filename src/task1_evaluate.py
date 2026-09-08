"""
task1_evaluate.py
-------------------
Loads the best Task 1 checkpoint, evaluates it on the held-out test set,
and saves:
    results/task1_metrics.json
    results/task1_predictions.csv     (every test example + prediction)
    results/task1_examples.json       (5 example predictions, human-readable)

Usage:
    python src/task1_evaluate.py
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score, average_precision_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from common import get_device, save_json
from bert_encoder import BertTagClassifier
from datasets import TagDataset, make_text_collate_fn


def run_inference(model, loader, device):
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():  # no need to track gradients, we're not training here
        for input_ids, attention_mask, labels in loader:
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            probs = torch.sigmoid(model(input_ids, attention_mask)).cpu()
            all_probs.append(probs)
            all_targets.append(labels)
    return torch.cat(all_probs).numpy(), torch.cat(all_targets).numpy()


def main():
    device = get_device()
    test_ds = TagDataset(config.SPLITS_DIR / "task1_test.json", config.SPLITS_DIR / "tag_vocab_task1.json")
    print(f"Test examples: {len(test_ds)}  Tags: {len(test_ds.vocab)}")

    model = BertTagClassifier(config.TASK1_MODEL_NAME, num_labels=len(test_ds.vocab)).to(device)
    ckpt_path = config.CHECKPOINT_DIR / "task1_bert_best.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded checkpoint {ckpt_path}")

    collate_fn = make_text_collate_fn(model.encoder, max_length=config.TASK1_MAX_TEXT_LEN)
    test_loader = DataLoader(test_ds, batch_size=config.TASK1_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    probs, targets = run_inference(model, test_loader, device)
    preds = (probs > 0.5).astype(float)

    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(targets, preds, average="micro", zero_division=0)
    try:
        auc_pr = average_precision_score(targets, probs, average="macro")
    except ValueError:
        auc_pr = None  # can happen if a tag has zero positives in the test subset

    metrics = {
        "task": "task1_bert_tag_classifier",
        "dataset": "MusicCaps captions",
        "num_test_examples": len(test_ds),
        "num_tags": len(test_ds.vocab),
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "auc_pr_macro": auc_pr,
    }
    save_json(metrics, config.RESULTS_DIR / "task1_metrics.json")
    print(f"Macro-F1: {macro_f1:.4f}  Micro-F1: {micro_f1:.4f}  AUC-PR: {auc_pr}")

    # --- predictions.csv: one row per test example ---
    pred_csv_path = config.RESULTS_DIR / "task1_predictions.csv"
    with open(pred_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "text", "true_tags", "predicted_tags"])
        for i, record in enumerate(test_ds.records):
            true_tags = [t for t in record["tags"] if t in test_ds.tag_to_idx]
            pred_tags = [test_ds.vocab[j] for j in np.nonzero(preds[i])[0]]
            writer.writerow([record["track_id"], record["text"], "; ".join(true_tags), "; ".join(pred_tags)])
    print(f"Saved {pred_csv_path}")

    # --- 5 example predictions, human-readable ---
    examples = []
    for i in range(min(5, len(test_ds))):
        record = test_ds.records[i]
        true_tags = [t for t in record["tags"] if t in test_ds.tag_to_idx]
        pred_tags = [test_ds.vocab[j] for j in np.nonzero(preds[i])[0]]
        examples.append({"text": record["text"], "true_tags": true_tags, "predicted_tags": pred_tags})
    save_json(examples, config.RESULTS_DIR / "task1_examples.json")

    plot_f1_curve(macro_f1, micro_f1)


def plot_f1_curve(test_macro_f1, test_micro_f1):
    """Plots the validation F1 curve across training epochs, plus the final test scores."""
    history_path = config.RESULTS_DIR / "task1_train_history.json"
    if not history_path.exists():
        return
    history = json.loads(history_path.read_text())

    fig, ax = plt.subplots(figsize=(6, 4))
    epochs = range(1, len(history["val_macro_f1"]) + 1)
    ax.plot(epochs, history["val_macro_f1"], marker="o", label="Val Macro-F1")
    ax.plot(epochs, history["val_micro_f1"], marker="o", label="Val Micro-F1")
    ax.axhline(test_macro_f1, color="tab:blue", linestyle="--", alpha=0.6, label="Test Macro-F1")
    ax.axhline(test_micro_f1, color="tab:orange", linestyle="--", alpha=0.6, label="Test Micro-F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 score")
    ax.set_title("Task 1: BERT Tag Classifier -- F1 over training")
    ax.legend()
    fig.tight_layout()

    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.PLOTS_DIR / "task1_f1.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
