"""
task4_evaluate.py
-------------------
Evaluates the Task 4 dual encoder on the test set: retrieval Recall@{1,5,10}
in both directions (audio->caption and caption->audio), plus qualitative
"top-3 retrieved clips per caption" examples.

Also attempts the assignment's zero-shot tag experiment: embed each TAG
NAME as text (no fine-tuning for this), and see how well cosine similarity
to the audio embedding predicts that tag -- without ever training a
tag classifier. A low score here is fine; the point is just to check it
runs correctly.

Outputs:
    results/task4_retrieval.json
    results/task4_examples.csv
    results/plots/task4_recall.png

Usage:
    python src/task4_train.py     (train first)
    python src/task4_evaluate.py
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from common import get_device, save_json
from task4_contrastive import DualEncoder, retrieval_recall_at_k
from datasets import FusionPairDataset, make_fusion_collate_fn


def encode_test_set(model, test_ds, device):
    collate_fn = make_fusion_collate_fn(model.bert, max_length=config.TASK1_MAX_TEXT_LEN)
    loader = DataLoader(test_ds, batch_size=config.TASK4_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    all_g, all_t = [], []
    with torch.no_grad():
        for graph_batch, input_ids, attention_mask, _ in loader:
            graph_batch = graph_batch.to(device)
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            g, t = model(graph_batch.x, graph_batch.edge_index, graph_batch.batch, input_ids, attention_mask)
            all_g.append(g.cpu())
            all_t.append(t.cpu())
    return torch.cat(all_g), torch.cat(all_t)


def zero_shot_tag_experiment(model, test_ds, g_embeds, device):
    """
    Embeds each tag NAME as if it were a caption, then predicts a clip's
    tags by thresholding cosine similarity between the clip's audio
    embedding and each tag's text embedding. No supervised tag training at
    all -- purely reusing the contrastive space Task 4 already learned.
    """
    with torch.no_grad():
        tokenized = model.bert.tokenize(test_ds.vocab, max_length=8)
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        tag_embeds = model.encode_text(tokenized["input_ids"], tokenized["attention_mask"]).cpu()  # (num_tags, dim)

    similarity = g_embeds @ tag_embeds.t()  # (num_test, num_tags)
    preds = (similarity > similarity.mean()).float().numpy()  # simple per-matrix threshold

    targets = torch.zeros(len(test_ds), len(test_ds.vocab))
    for i, record in enumerate(test_ds.records):
        for tag in record["tags"]:
            if tag in test_ds.tag_to_idx:
                targets[i, test_ds.tag_to_idx[tag]] = 1.0
    targets = targets.numpy()

    return {
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(targets, preds, average="micro", zero_division=0)),
        "note": "Zero-shot: tag names embedded as text, no supervised tag training.",
    }


def plot_recall(metrics):
    fig, ax = plt.subplots(figsize=(6, 4))
    ks = [1, 5, 10]
    a2t = [metrics[f"audio_to_text_R@{k}"] for k in ks]
    t2a = [metrics[f"text_to_audio_R@{k}"] for k in ks]

    ax.plot(ks, a2t, marker="o", label="Audio -> Caption")
    ax.plot(ks, t2a, marker="o", label="Caption -> Audio")
    ax.set_xlabel("K")
    ax.set_xticks(ks)
    ax.set_ylabel("Recall@K")
    ax.set_title("Task 4: contrastive retrieval (MusicCaps test set)")
    ax.legend()
    fig.tight_layout()

    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.PLOTS_DIR / "task4_recall.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved {out_path}")


def save_qualitative_examples(test_ds, g, t, n=10):
    """For n random captions, save the top-3 audio clips retrieved by similarity."""
    sim = t @ g.t()  # (num_captions, num_clips)
    csv_path = config.RESULTS_DIR / "task4_examples.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["caption", "true_track_id", "top1_track_id", "top2_track_id", "top3_track_id"])
        for i in range(min(n, len(test_ds))):
            top3 = sim[i].topk(3).indices.tolist()
            row = [test_ds.records[i]["text"], test_ds.records[i]["track_id"]]
            row += [test_ds.records[j]["track_id"] for j in top3]
            writer.writerow(row)
    print(f"Saved {csv_path}")


def main():
    device = get_device()
    test_ds = FusionPairDataset(config.SPLITS_DIR / "task34_test.json", config.SPLITS_DIR / "tag_vocab_task34.json")
    print(f"Test examples: {len(test_ds)}")

    model = DualEncoder(
        bert_model_name=config.TASK1_MODEL_NAME, gnn_in_channels=config.GRAPH_IN_CHANNELS,
        embed_dim=config.TASK4_EMBED_DIM,
    ).to(device)
    model.load_state_dict(torch.load(config.CHECKPOINT_DIR / "task4_contrastive_best.pt", map_location=device))
    model.eval()

    g, t = encode_test_set(model, test_ds, device)
    metrics = retrieval_recall_at_k(g, t)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    zero_shot = zero_shot_tag_experiment(model, test_ds, g, device)
    print(f"  Zero-shot tag prediction: macro_f1={zero_shot['macro_f1']:.4f}  micro_f1={zero_shot['micro_f1']:.4f}")

    metrics["num_test_examples"] = len(test_ds)
    metrics["zero_shot_tag_prediction"] = zero_shot
    save_json(metrics, config.RESULTS_DIR / "task4_retrieval.json")

    plot_recall(metrics)
    save_qualitative_examples(test_ds, g, t)


if __name__ == "__main__":
    main()
