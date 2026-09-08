"""
datasets.py
-----------
Simple PyTorch Dataset classes for all four tasks, plus their collate_fn
helpers (a collate_fn decides how a list of individual examples gets
stacked into one batch tensor).

  TagDataset          - Task 1: caption text -> multi-hot tag vector
  GraphDataset        - Task 2 (GNN): cached segment graph -> genre label
  MelSpectrogramDataset - Task 2 (CNN baseline): raw audio -> log-mel spectrogram -> genre label
  FusionPairDataset   - Task 3 & 4: cached segment graph + caption text -> multi-hot tag vector
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Dataset as PyGDataset, Batch

from audio_features import load_audio, extract_log_mel
from cnn_baseline import pad_or_truncate


# ---------------------------------------------------------------------
# Task 1: text -> tags
# ---------------------------------------------------------------------
class TagDataset(Dataset):
    def __init__(self, split_path, vocab_path):
        self.records = json.loads(Path(split_path).read_text())
        self.vocab = json.loads(Path(vocab_path).read_text())
        self.tag_to_idx = {tag: i for i, tag in enumerate(self.vocab)}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        text = record["text"]
        label = torch.zeros(len(self.vocab), dtype=torch.float)
        for tag in record["tags"]:
            if tag in self.tag_to_idx:
                label[self.tag_to_idx[tag]] = 1.0
        return text, label


def make_text_collate_fn(text_encoder, max_length=64):
    def collate_fn(batch):
        texts, labels = zip(*batch)
        tokenized = text_encoder.tokenize(list(texts), max_length=max_length)
        labels = torch.stack(labels)
        return tokenized["input_ids"], tokenized["attention_mask"], labels
    return collate_fn


def compute_pos_weight(records, vocab, cap=20.0):
    """pos_weight[k] = n_negative_k / n_positive_k, capped -- upweights rare tags."""
    tag_to_idx = {tag: i for i, tag in enumerate(vocab)}
    pos_counts = torch.zeros(len(vocab))
    for record in records:
        for tag in record["tags"]:
            if tag in tag_to_idx:
                pos_counts[tag_to_idx[tag]] += 1
    n = len(records)
    neg_counts = n - pos_counts
    pos_counts = pos_counts.clamp(min=1.0)
    return (neg_counts / pos_counts).clamp(max=cap)


# ---------------------------------------------------------------------
# Task 2 (GNN): cached graph -> genre
# ---------------------------------------------------------------------
class GraphDataset(PyGDataset):
    def __init__(self, split_path, genre_vocab_path):
        super().__init__()
        self.records = json.loads(Path(split_path).read_text())
        self.genre_vocab = json.loads(Path(genre_vocab_path).read_text())

    def len(self):
        return len(self.records)

    def get(self, idx):
        record = self.records[idx]
        graph = torch.load(record["graph_path"], weights_only=False)
        if not hasattr(graph, "y") or graph.y is None:
            graph.y = torch.tensor([record["genre_idx"]], dtype=torch.long)
        return graph


# ---------------------------------------------------------------------
# Task 2 (CNN baseline): raw audio -> log-mel spectrogram -> genre
# ---------------------------------------------------------------------
class MelSpectrogramDataset(Dataset):
    def __init__(self, split_path, audio_cfg, target_frames, hop_length=512):
        self.records = json.loads(Path(split_path).read_text())
        self.audio_cfg = audio_cfg
        self.target_frames = target_frames
        self.hop_length = hop_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        waveform = load_audio(rec["audio_path"], self.audio_cfg.sample_rate)
        log_mel = extract_log_mel(waveform, self.audio_cfg.sample_rate, self.audio_cfg.n_mels, self.hop_length)
        log_mel = pad_or_truncate(log_mel, self.target_frames).astype(np.float32)
        x = torch.from_numpy(log_mel).unsqueeze(0)  # (1, n_mels, target_frames)
        y = torch.tensor(rec["genre_idx"], dtype=torch.long)
        return x, y


# ---------------------------------------------------------------------
# Task 3 & 4: paired (graph, caption) -> tags
# ---------------------------------------------------------------------
class FusionPairDataset(Dataset):
    def __init__(self, split_path, vocab_path):
        self.records = json.loads(Path(split_path).read_text())
        self.vocab = json.loads(Path(vocab_path).read_text())
        self.tag_to_idx = {tag: i for i, tag in enumerate(self.vocab)}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        graph = torch.load(record["graph_path"], weights_only=False)
        text = record["text"]
        label = torch.zeros(len(self.vocab), dtype=torch.float)
        for tag in record["tags"]:
            if tag in self.tag_to_idx:
                label[self.tag_to_idx[tag]] = 1.0
        return graph, text, label


def make_fusion_collate_fn(text_encoder, max_length=64):
    def collate_fn(items):
        graphs, texts, labels = zip(*items)
        graph_batch = Batch.from_data_list(list(graphs))
        tokenized = text_encoder.tokenize(list(texts), max_length=max_length)
        labels = torch.stack(labels)
        return graph_batch, tokenized["input_ids"], tokenized["attention_mask"], labels
    return collate_fn
