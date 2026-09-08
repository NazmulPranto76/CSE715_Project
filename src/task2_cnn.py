"""
task2_cnn.py
-------------
Task 2's required baseline: a plain CNN on the mel-spectrogram, with no
graph and no text. Trained on the same GTZAN split as task2_gnn.py, so
its accuracy is directly comparable.

Pipeline:  audio -> log-mel spectrogram -> 2D CNN -> genre.

Usage:
    python src/prepare_task2_graphs.py     (run once first -- also builds the audio_path split)
    python src/task2_cnn.py
"""

import sys
import time
from pathlib import Path

import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from common import set_seed, get_device, save_json
from cnn_baseline import MelSpectrogramCNN
from datasets import MelSpectrogramDataset
from audio_features import AudioConfig


def evaluate_split(model, loader, device):
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
    accuracy = (preds == targets).mean()
    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
    return accuracy, macro_f1


def main():
    set_seed(config.SEED)
    device = get_device()
    print(f"Using device: {device}")

    audio_cfg = AudioConfig(sample_rate=config.SAMPLE_RATE, n_mels=config.N_MELS)
    train_ds = MelSpectrogramDataset(config.SPLITS_DIR / "gtzan_train.json", audio_cfg, config.TASK2_CNN_TARGET_FRAMES)
    val_ds = MelSpectrogramDataset(config.SPLITS_DIR / "gtzan_val.json", audio_cfg, config.TASK2_CNN_TARGET_FRAMES)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    # num_workers=0 keeps this simple and avoids platform-specific multiprocessing quirks.
    train_loader = DataLoader(train_ds, batch_size=config.TASK2_CNN_BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.TASK2_CNN_BATCH_SIZE, shuffle=False, num_workers=0)

    model = MelSpectrogramCNN(num_classes=len(config.GTZAN_GENRES), dropout=config.TASK2_CNN_DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.TASK2_CNN_LR)
    loss_fn = torch.nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_accuracy": [], "val_macro_f1": []}
    best_macro_f1 = -1.0
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_path = config.CHECKPOINT_DIR / "task2_cnn_best.pt"

    start_time = time.time()
    for epoch in range(1, config.TASK2_CNN_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)

        epoch_loss /= len(train_ds)
        val_acc, val_macro_f1 = evaluate_split(model, val_loader, device)
        history["train_loss"].append(epoch_loss)
        history["val_accuracy"].append(val_acc)
        history["val_macro_f1"].append(val_macro_f1)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{config.TASK2_CNN_EPOCHS}  loss={epoch_loss:.4f}  "
                  f"val_acc={val_acc:.4f}  val_macro_f1={val_macro_f1:.4f}")

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            torch.save(model.state_dict(), best_path)

    elapsed_min = (time.time() - start_time) / 60
    print(f"Training finished in {elapsed_min:.1f} minutes. Best val Macro-F1: {best_macro_f1:.4f}")
    save_json(history, config.RESULTS_DIR / "task2_cnn_train_history.json")


if __name__ == "__main__":
    main()
