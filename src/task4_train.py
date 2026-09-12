"""
task4_train.py
----------------
Task 4: train a CLIP-style dual encoder to match captions with their audio
graphs, using a symmetric InfoNCE loss (no tag labels involved at all --
the model only ever sees which graph goes with which caption).

Usage:
    python src/prepare_task34_graphs.py     (run once first, shared with Task 3)
    python src/task4_train.py
"""

import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from common import set_seed, get_device, save_json
from task4_contrastive import DualEncoder, info_nce_loss, retrieval_recall_at_k
from datasets import FusionPairDataset, make_fusion_collate_fn


def evaluate_split(model, loader, device):
    """Encodes an entire split and computes retrieval Recall@K."""
    model.eval()
    all_g, all_t = [], []
    with torch.no_grad():
        for graph_batch, input_ids, attention_mask, _ in loader:
            graph_batch = graph_batch.to(device)
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            g, t = model(graph_batch.x, graph_batch.edge_index, graph_batch.batch, input_ids, attention_mask)
            all_g.append(g.cpu())
            all_t.append(t.cpu())
    g = torch.cat(all_g)
    t = torch.cat(all_t)
    return retrieval_recall_at_k(g, t)


def main(seed=None):
    seed = config.SEED if seed is None else seed
    set_seed(seed)
    device = get_device()
    print(f"Using device: {device}  seed={seed}")

    train_ds = FusionPairDataset(config.SPLITS_DIR / "task34_train.json", config.SPLITS_DIR / "tag_vocab_task34.json")
    val_ds = FusionPairDataset(config.SPLITS_DIR / "task34_val.json", config.SPLITS_DIR / "tag_vocab_task34.json")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    model = DualEncoder(
        bert_model_name=config.TASK1_MODEL_NAME, gnn_in_channels=config.GRAPH_IN_CHANNELS,
        embed_dim=config.TASK4_EMBED_DIM,
    ).to(device)

    collate_fn = make_fusion_collate_fn(model.bert, max_length=config.TASK1_MAX_TEXT_LEN)
    # drop_last=True keeps every training batch full-size, which matters here:
    # InfoNCE treats every OTHER item in the batch as a negative, so a
    # same-size batch keeps the "difficulty" of the loss consistent epoch to epoch.
    train_loader = DataLoader(train_ds, batch_size=config.TASK4_BATCH_SIZE, shuffle=True,
                               collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.TASK4_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.TASK4_LR)

    history = {"train_loss": [], "val_recall_metrics": []}
    best_recall = -1.0
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_path = config.CHECKPOINT_DIR / "task4_contrastive_best.pt"

    start_time = time.time()
    for epoch in range(1, config.TASK4_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for graph_batch, input_ids, attention_mask, _ in train_loader:
            graph_batch = graph_batch.to(device)
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

            optimizer.zero_grad()
            g, t = model(graph_batch.x, graph_batch.edge_index, graph_batch.batch, input_ids, attention_mask)
            loss = info_nce_loss(g, t, temperature=config.TASK4_TEMPERATURE)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        epoch_loss /= max(1, n_batches)
        val_metrics = evaluate_split(model, val_loader, device)
        val_recall_at_10 = (val_metrics["audio_to_text_R@10"] + val_metrics["text_to_audio_R@10"]) / 2
        history["train_loss"].append(epoch_loss)
        history["val_recall_metrics"].append(val_metrics)
        print(f"Epoch {epoch}/{config.TASK4_EPOCHS}  loss={epoch_loss:.4f}  "
              f"val_avg_R@10={val_recall_at_10:.4f}")

        if val_recall_at_10 > best_recall:
            best_recall = val_recall_at_10
            torch.save(model.state_dict(), best_path)
            print(f"  -> new best model saved to {best_path}")

    elapsed_min = (time.time() - start_time) / 60
    print(f"Training finished in {elapsed_min:.1f} minutes. Best val avg R@10: {best_recall:.4f}")
    save_json(history, config.RESULTS_DIR / "task4_train_history.json")


if __name__ == "__main__":
    main()
