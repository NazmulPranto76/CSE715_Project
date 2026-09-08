"""
task2_gnn.py
--------------
Task 2 (GNN): train GraphSAGE on GTZAN segment graphs to classify genre.

Pipeline:  audio -> segments -> graph (graph_builder.py, already cached by
prepare_task2_graphs.py) -> GraphSAGE -> mean-pool -> linear head -> genre.

Usage:
    python src/prepare_task2_graphs.py     (run once first)
    python src/task2_gnn.py
"""

import sys
import time
from pathlib import Path

import torch
from sklearn.metrics import f1_score
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from common import set_seed, get_device, save_json
from gnn_model import GNNGenreClassifier
from datasets import GraphDataset


def evaluate_split(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            preds = logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_targets.append(batch.y.cpu())
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    accuracy = (preds == targets).mean()
    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
    return accuracy, macro_f1


def main():
    set_seed(config.SEED)
    device = get_device()
    print(f"Using device: {device}")

    train_ds = GraphDataset(config.SPLITS_DIR / "gtzan_train.json", config.SPLITS_DIR / "gtzan_genre_vocab.json")
    val_ds = GraphDataset(config.SPLITS_DIR / "gtzan_val.json", config.SPLITS_DIR / "gtzan_genre_vocab.json")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Genres: {len(train_ds.genre_vocab)}")

    train_loader = DataLoader(train_ds, batch_size=config.TASK2_GNN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.TASK2_GNN_BATCH_SIZE, shuffle=False)

    model = GNNGenreClassifier(
        in_channels=config.GRAPH_IN_CHANNELS, num_classes=len(train_ds.genre_vocab),
        hidden_channels=config.TASK2_GNN_HIDDEN, out_channels=config.TASK2_GNN_OUT,
        num_layers=config.TASK2_GNN_LAYERS, arch=config.TASK2_GNN_ARCH, dropout=config.TASK2_GNN_DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.TASK2_GNN_LR, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_accuracy": [], "val_macro_f1": []}
    best_macro_f1 = -1.0
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_path = config.CHECKPOINT_DIR / "task2_gnn_best.pt"

    start_time = time.time()
    for epoch in range(1, config.TASK2_GNN_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(logits, batch.y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.num_graphs

        epoch_loss /= len(train_ds)
        val_acc, val_macro_f1 = evaluate_split(model, val_loader, device)
        history["train_loss"].append(epoch_loss)
        history["val_accuracy"].append(val_acc)
        history["val_macro_f1"].append(val_macro_f1)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{config.TASK2_GNN_EPOCHS}  loss={epoch_loss:.4f}  "
                  f"val_acc={val_acc:.4f}  val_macro_f1={val_macro_f1:.4f}")

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            torch.save(model.state_dict(), best_path)

    elapsed_min = (time.time() - start_time) / 60
    print(f"Training finished in {elapsed_min:.1f} minutes. Best val Macro-F1: {best_macro_f1:.4f}")
    save_json(history, config.RESULTS_DIR / "task2_gnn_train_history.json")


if __name__ == "__main__":
    main()
