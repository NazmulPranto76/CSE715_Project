"""
prepare_task2_graphs.py
------------------------
Task 2 data prep: builds one segment graph per GTZAN track (one node per
2 seconds of audio, edges from time order + audio similarity -- see
graph_builder.py) and writes genre-labeled train/val/test splits.

This repo already ships the graphs it needs (data/processed/graphs/) plus
the train/val/test split files, so most people won't need to run this at
all. It's here in case you want to rebuild everything, or add more genres.

If you do want to rebuild: download GTZAN from Kaggle (only the
genres_original folder is needed):
    https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
and place it at data/raw/gtzan/genres_original/<genre>/<file>.wav

Usage:
    python src/prepare_task2_graphs.py
"""

import json
import shutil
import sys
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from common import set_seed
from audio_features import AudioConfig, preprocess_track
from graph_builder import build_segment_graph


def find_gtzan_folder():
    """Kaggle's GTZAN zip can unpack with a couple of different folder layouts."""
    candidates = [
        config.RAW_DIR / "gtzan" / "genres_original",
        config.RAW_DIR / "gtzan" / "Data" / "genres_original",
        config.RAW_DIR / "genres_original",
    ]
    for folder in candidates:
        if folder.exists() and any(folder.iterdir()):
            return folder
    return None


def build_graphs(gtzan_folder, graphs_dir, audio_cfg):
    """Walks every genre folder's .wav files and builds one graph per track."""
    graphs_dir.mkdir(parents=True, exist_ok=True)
    records = []

    wav_paths = sorted(gtzan_folder.glob("*/*.wav"))
    for wav_path in wav_paths:
        genre = wav_path.parent.name
        if genre not in config.GTZAN_GENRES:
            continue
        track_id = wav_path.stem

        try:
            features = preprocess_track(str(wav_path), audio_cfg)
            graph = build_segment_graph(features["segment_features"], config.SIMILARITY_THRESHOLD)
        except Exception as e:
            print(f"  skipping {track_id}: {e}")
            continue

        genre_idx = config.GTZAN_GENRES.index(genre)
        graph.y = torch.tensor([genre_idx], dtype=torch.long)

        graph_path = graphs_dir / f"{track_id}.pt"
        torch.save(graph, graph_path)

        records.append({
            "track_id": track_id, "genre": genre, "genre_idx": genre_idx,
            "graph_path": str(graph_path), "audio_path": str(wav_path),
        })

    return records


def main():
    set_seed(config.SEED)

    if (config.SPLITS_DIR / "gtzan_train.json").exists():
        print("GTZAN splits already exist -- nothing to do.")
        print("(delete data/splits/gtzan_*.json first if you want to rebuild them)")
        return

    gtzan_folder = find_gtzan_folder()
    if gtzan_folder is None:
        raise FileNotFoundError(
            "Could not find GTZAN audio under data/raw/gtzan/. "
            "See this script's docstring for the download link."
        )

    audio_cfg = AudioConfig(sample_rate=config.SAMPLE_RATE, n_mels=config.N_MELS,
                             n_chroma=config.N_CHROMA, segment_seconds=config.SEGMENT_SECONDS)
    records = build_graphs(gtzan_folder, config.PROCESSED_DIR / "graphs", audio_cfg)
    print(f"Built {len(records)} graphs")

    config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    (config.SPLITS_DIR / "gtzan_genre_vocab.json").write_text(json.dumps(config.GTZAN_GENRES, indent=2))

    # Split by genre so each genre stays roughly evenly spread across train/val/test.
    genre_labels = [r["genre_idx"] for r in records]
    train_val, test = train_test_split(records, test_size=0.15, random_state=config.SEED, stratify=genre_labels)
    train_val_labels = [r["genre_idx"] for r in train_val]
    train, val = train_test_split(train_val, test_size=0.15 / 0.85, random_state=config.SEED, stratify=train_val_labels)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        out_path = config.SPLITS_DIR / f"gtzan_{name}.json"
        out_path.write_text(json.dumps(split, indent=2))
        print(f"  {name}: {len(split)} tracks -> {out_path}")

    # Save a handful of example graphs, as a representative sample.
    config.GRAPH_EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for rec in train[:25]:
        shutil.copyfile(rec["graph_path"], config.GRAPH_EXAMPLES_DIR / f"gtzan_{rec['track_id']}.pt")

    print("Done. Next: python src/task2_gnn.py")


if __name__ == "__main__":
    main()
