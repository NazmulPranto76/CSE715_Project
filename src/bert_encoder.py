"""
bert_encoder.py
----------------
Task 1: a DistilBERT-based multi-label tag classifier over MusicCaps
captions. No audio, no graphs -- just text in, tags out.

Pipeline:  caption text -> tokenizer -> DistilBERT -> CLS vector -> linear
layer -> sigmoid -> one probability per tag.

`BertTextEncoder` is reused by Task 3 (fusion) and Task 4 (contrastive) as
the text tower, so it's kept separate from the Task 1 classifier head.
"""

import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class BertTextEncoder(nn.Module):
    """Wraps a HuggingFace text model. Returns the CLS vector and full token sequence."""

    def __init__(self, model_name="distilbert-base-uncased", freeze_backbone=False):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden_dim = self.backbone.config.hidden_size

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def tokenize(self, texts, max_length=64):
        return self.tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=max_length, return_tensors="pt",
        )

    def forward(self, input_ids, attention_mask):
        # cls_vec: one summary vector per caption. full_tokens: every token's
        # vector, needed later by Task 3's cross-attention.
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        full_tokens = out.last_hidden_state
        cls_vec = full_tokens[:, 0, :]  # the [CLS] token summarizes the whole caption
        return cls_vec, full_tokens


class BertTagClassifier(nn.Module):
    """Task 1 model: BertTextEncoder + a linear multi-label classification head."""

    def __init__(self, model_name, num_labels, dropout=0.1, freeze_backbone=False):
        super().__init__()
        self.encoder = BertTextEncoder(model_name, freeze_backbone)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.hidden_dim, num_labels)

    def forward(self, input_ids, attention_mask):
        cls_vec, _ = self.encoder(input_ids, attention_mask)
        logits = self.head(self.dropout(cls_vec))
        return logits  # raw logits -- apply sigmoid only at evaluation/prediction time


def bce_multilabel_loss(logits, targets, pos_weight=None):
    """
    Standard multi-label loss: one binary cross-entropy per tag, averaged.
    `pos_weight[k]` (optional) upweights tag k's positive examples -- useful
    because most tags are only positive in a small fraction of clips, so
    without reweighting the model can get away with "always predict no".
    """
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return loss_fn(logits, targets.float())


if __name__ == "__main__":
    # Smoke test with two dummy captions
    model = BertTagClassifier("distilbert-base-uncased", num_labels=30)
    batch = model.encoder.tokenize(["melancholic piano ballad", "upbeat 80s synth pop"])
    logits = model(batch["input_ids"], batch["attention_mask"])
    print("output shape:", logits.shape)  # expect (2, 30)
