"""
task1_train.py
----------------
Task 1: fine-tune DistilBERT to predict music tags from a caption.

Pipeline:  caption -> tokenizer -> DistilBERT -> CLS vector -> linear head
           -> sigmoid -> one probability per tag (multi-label).

Usage:
    python src/prepare_task1.py     (run once first)
    python src/task1_train.py
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
from common import set_seed, get_device, save_json, load_json
from bert_encoder import BertTagClassifier, bce_multilabel_loss
from datasets import TagDataset, make_text_collate_fn, compute_pos_weight


def evaluate_split(model, loader, device):
    """Runs the model on a whole split and returns Macro-F1 + Micro-F1 (0.5 threshold)."""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():  # no need to track gradients, we're not training here
        for input_ids, attention_mask, labels in loader:
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            logits = model(input_ids, attention_mask)
            preds = (torch.sigmoid(logits) > 0.5).float().cpu()
            all_preds.append(preds)
            all_targets.append(labels)
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(targets, preds, average="micro", zero_division=0)
    return macro_f1, micro_f1


def main():
    set_seed(config.SEED)
    device = get_device()
    print(f"Using device: {device}")

    train_ds = TagDataset(config.SPLITS_DIR / "task1_train.json", config.SPLITS_DIR / "tag_vocab_task1.json")
    val_ds = TagDataset(config.SPLITS_DIR / "task1_val.json", config.SPLITS_DIR / "tag_vocab_task1.json")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Tags: {len(train_ds.vocab)}")

    model = BertTagClassifier(config.TASK1_MODEL_NAME, num_labels=len(train_ds.vocab)).to(device)
    collate_fn = make_text_collate_fn(model.encoder, max_length=config.TASK1_MAX_TEXT_LEN)

    train_loader = DataLoader(train_ds, batch_size=config.TASK1_BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config.TASK1_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Rare tags are positive in only a small fraction of clips -- pos_weight
    # makes missing a positive "cost" more, so the model doesn't just learn
    # to always predict "no tag".
    pos_weight = compute_pos_weight(train_ds.records, train_ds.vocab).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.TASK1_LR)

    history = {"train_loss": [], "val_macro_f1": [], "val_micro_f1": []}
    best_macro_f1 = -1.0
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_path = config.CHECKPOINT_DIR / "task1_bert_best.pt"

    start_time = time.time()
    for epoch in range(1, config.TASK1_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for input_ids, attention_mask, labels in train_loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = bce_multilabel_loss(logits, labels, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * input_ids.size(0)

        epoch_loss /= len(train_ds)
        val_macro_f1, val_micro_f1 = evaluate_split(model, val_loader, device)
        history["train_loss"].append(epoch_loss)
        history["val_macro_f1"].append(val_macro_f1)
        history["val_micro_f1"].append(val_micro_f1)
        print(f"Epoch {epoch}/{config.TASK1_EPOCHS}  loss={epoch_loss:.4f}  "
              f"val_macro_f1={val_macro_f1:.4f}  val_micro_f1={val_micro_f1:.4f}")

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            torch.save(model.state_dict(), best_path)
            print(f"  -> new best model saved to {best_path}")

    elapsed_min = (time.time() - start_time) / 60
    print(f"Training finished in {elapsed_min:.1f} minutes. Best val Macro-F1: {best_macro_f1:.4f}")

    save_json(history, config.RESULTS_DIR / "task1_train_history.json")


if __name__ == "__main__":
    main()
