"""
prepare_task1.py
-----------------
Task 1 data prep: read the MusicCaps captions CSV, build a small top-K tag
vocabulary from the `aspect_list` column, and write small train/val/test
subsets in JSON. Text only -- no audio needed for Task 1.

Usage:
    python src/prepare_task1.py

Outputs (under data/splits/):
    tag_vocab_task1.json
    task1_train.json, task1_val.json, task1_test.json
"""

import ast
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from common import set_seed


def build_tag_vocab(df, top_k):
    """Count every tag across all clips and keep the top_k most frequent ones."""
    counter = Counter()
    for tags in df["aspect_list_parsed"]:
        counter.update(tags)
    return [tag for tag, _ in counter.most_common(top_k)]


def build_records(df, vocab):
    """One record per clip: {track_id, text, tags}. Drops clips with none of the top-K tags."""
    vocab_set = set(vocab)
    records = []
    for _, row in df.iterrows():
        clip_tags = [t for t in row["aspect_list_parsed"] if t in vocab_set]
        if not clip_tags:
            continue
        records.append({
            "track_id": row["ytid"],
            "text": row["caption"],
            "tags": clip_tags,
        })
    return records


def main():
    set_seed(config.SEED)

    csv_path = config.RAW_DIR / "musiccaps" / "musiccaps-public.csv"
    print(f"Reading {csv_path}")
    df = pd.read_csv(csv_path)
    df["aspect_list_parsed"] = df["aspect_list"].apply(ast.literal_eval)
    print(f"Loaded {len(df)} MusicCaps clips (captions only, no audio needed for Task 1)")

    vocab = build_tag_vocab(df, config.TASK1_TOP_K_TAGS)
    records = build_records(df, vocab)
    print(f"{len(records)} clips have at least one of the top-{len(vocab)} tags")

    # Shuffle once, then take a small subset so training finishes fast
    # (see config.py TASK1_TRAIN_SIZE / TASK1_VAL_SIZE / TASK1_TEST_SIZE).
    train_val, test = train_test_split(records, test_size=0.15, random_state=config.SEED)
    train, val = train_test_split(train_val, test_size=0.15, random_state=config.SEED)

    train = train[:config.TASK1_TRAIN_SIZE]
    val = val[:config.TASK1_VAL_SIZE]
    test = test[:config.TASK1_TEST_SIZE]

    config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    (config.SPLITS_DIR / "tag_vocab_task1.json").write_text(json.dumps(vocab, indent=2))
    for name, split in [("train", train), ("val", val), ("test", test)]:
        (config.SPLITS_DIR / f"task1_{name}.json").write_text(json.dumps(split, indent=2))
        print(f"  {name}: {len(split)} examples")

    print("Done. Next: python src/task1_train.py")


if __name__ == "__main__":
    main()
