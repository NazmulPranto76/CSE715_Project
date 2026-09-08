"""
audio_features.py
------------------
Loads an audio file and turns it into the features we need:
  - log-mel spectrogram   (used by the Task 2 CNN baseline)
  - chroma + MFCC         (used as GNN node features, Task 2/3/4)

Then splits a track into fixed-length time segments (e.g. one segment per
2 seconds) -- each segment becomes one NODE in the graph built by
graph_builder.py.
"""

import numpy as np
import librosa


class AudioConfig:
    """Just holds the few numbers every audio function below needs."""

    def __init__(self, sample_rate=22050, n_mels=128, n_chroma=12, segment_seconds=2.0):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_chroma = n_chroma
        self.segment_seconds = segment_seconds


def load_audio(path, sample_rate=22050):
    """Load an audio file and resample it to `sample_rate`. Returns a 1D waveform."""
    waveform, _ = librosa.load(path, sr=sample_rate, mono=True)
    return waveform


def extract_log_mel(waveform, sr, n_mels=128, hop_length=512):
    """Log-mel spectrogram, shape (n_mels, n_frames), normalized to zero mean / unit variance."""
    mel = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=n_mels, hop_length=hop_length)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
    return log_mel


def extract_chroma(waveform, sr, n_chroma=12, hop_length=512):
    """Chromagram (which of the 12 pitch classes are active), shape (n_chroma, n_frames)."""
    return librosa.feature.chroma_cqt(y=waveform, sr=sr, n_chroma=n_chroma, hop_length=hop_length)


def extract_mfcc(waveform, sr, n_mfcc=20):
    """MFCC features (rough timbre/texture), shape (n_mfcc, n_frames)."""
    return librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc)


def segment_fixed_windows(feature, sr, hop_length, segment_seconds=2.0):
    """
    Splits a (n_features, n_frames) array into fixed-length time windows.

    `feature` has one column per analysis frame, spaced `hop_length` samples
    apart at sample rate `sr`. A window of `segment_seconds` therefore spans
        window_frames = round(segment_seconds * sr / hop_length)
    frames, and windows do not overlap (stride == window length).

    A trailing partial window (shorter than window_frames) is dropped rather
    than zero-padded, since a padded/silent segment would just be a noisy
    node in the graph.

    Returns a list of (n_features, window_frames) arrays, in time order.
    """
    n_frames = feature.shape[1]
    window_frames = max(1, round(segment_seconds * sr / hop_length))

    segments = []
    start = 0
    while start + window_frames <= n_frames:
        segments.append(feature[:, start:start + window_frames])
        start += window_frames

    if not segments:
        # Track shorter than one full window -- use the whole thing as a
        # single segment instead of returning nothing.
        segments.append(feature)

    return segments


def preprocess_track(path, cfg, hop_length=512, n_mfcc=20):
    """
    End-to-end: load -> extract mel + chroma + MFCC -> segment -> mean-pool
    each segment to one feature vector per segment. That per-segment
    feature vector (chroma + MFCC concatenated) is what becomes a GNN
    node's features in graph_builder.build_segment_graph.

    Returns a dict with keys:
      "mel"              : full-track log-mel spectrogram, for the CNN baseline
      "segment_features" : (n_segments, n_chroma + n_mfcc) -- one row per node
    """
    waveform = load_audio(path, cfg.sample_rate)
    mel = extract_log_mel(waveform, cfg.sample_rate, cfg.n_mels, hop_length)
    chroma = extract_chroma(waveform, cfg.sample_rate, cfg.n_chroma, hop_length)
    mfcc = extract_mfcc(waveform, cfg.sample_rate, n_mfcc)

    chroma_segments = segment_fixed_windows(chroma, cfg.sample_rate, hop_length, cfg.segment_seconds)
    mfcc_segments = segment_fixed_windows(mfcc, cfg.sample_rate, hop_length, cfg.segment_seconds)
    n_segments = min(len(chroma_segments), len(mfcc_segments))

    # one row per segment: that segment's chroma average + MFCC average, stuck together
    segment_features = []
    for i in range(n_segments):
        chroma_avg = chroma_segments[i].mean(axis=1)
        mfcc_avg = mfcc_segments[i].mean(axis=1)
        segment_features.append(np.concatenate([chroma_avg, mfcc_avg]))
    segment_features = np.stack(segment_features)

    return {"mel": mel, "segment_features": segment_features}


if __name__ == "__main__":
    # Quick smoke test with a fake waveform (no audio file needed).
    cfg = AudioConfig()
    fake_wave = np.random.randn(cfg.sample_rate * 10)  # 10 seconds of noise
    mel = extract_log_mel(fake_wave, cfg.sample_rate, cfg.n_mels)
    chroma = extract_chroma(fake_wave, cfg.sample_rate, cfg.n_chroma)
    print("log-mel shape:", mel.shape)
    print("chroma shape:", chroma.shape)
    segments = segment_fixed_windows(chroma, cfg.sample_rate, 512, cfg.segment_seconds)
    print(f"{len(segments)} segments of shape {segments[0].shape}")
