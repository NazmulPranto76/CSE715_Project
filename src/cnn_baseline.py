"""
cnn_baseline.py
------------------
Task 2's required baseline: a plain 2D CNN over the log-mel spectrogram,
with NO graph and NO text. This is the "simple baseline" that the GNN
(gnn_model.py) is compared against.

Architecture: 4 conv blocks (conv -> batchnorm -> ReLU -> maxpool), then
global average pooling and one linear layer. Nothing fancy -- its whole
purpose is to be the honest, unstructured comparison point for the GNN.
"""

import numpy as np
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
        )

    def forward(self, x):
        return self.block(x)


class MelSpectrogramCNN(nn.Module):
    """Input: (batch, 1, n_mels, target_frames) log-mel spectrogram. Output: class logits."""

    def __init__(self, num_classes=10, base_channels=32, dropout=0.3):
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            ConvBlock(1, c),
            ConvBlock(c, c * 2),
            ConvBlock(c * 2, c * 4),
            ConvBlock(c * 4, c * 4),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(c * 4, num_classes)

    def forward(self, x):
        h = self.features(x)
        h = self.pool(h).flatten(1)
        h = self.dropout(h)
        return self.head(h)


def pad_or_truncate(log_mel, target_frames):
    """Fixed-width log-mel: truncate if longer than target, zero-pad if shorter."""
    n_mels, cur_frames = log_mel.shape
    if cur_frames >= target_frames:
        return log_mel[:, :target_frames]
    pad = np.zeros((n_mels, target_frames - cur_frames), dtype=log_mel.dtype)
    return np.concatenate([log_mel, pad], axis=1)


if __name__ == "__main__":
    model = MelSpectrogramCNN(num_classes=10)
    dummy = torch.randn(4, 1, 128, 1300)
    out = model(dummy)
    print("output shape:", out.shape)  # expect (4, 10)
