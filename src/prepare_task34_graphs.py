"""
prepare_task34_graphs.py
--------------------------
Task 3 (fusion) and Task 4 (contrastive) share the same data: MusicCaps
clips that have BOTH a caption and an audio graph. Building the audio
graphs requires downloading each clip's audio from YouTube first, which
is slow and not always available -- so this repo ships a pre-built set of
graphs for a few thousand clips in data/processed/musiccaps_graphs/.

This script just matches those already-built graphs up with their
captions and tags (read from the MusicCaps CSV) and writes train/val/test
splits.

Usage:
    python src/prepare_task34_graphs.py
"""

import ast
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from common import set_seed


def main():
    set_seed(config.SEED)

    if (config.SPLITS_DIR / "task34_train.json").exists():
        print("Task 3/4 splits already exist -- nothing to do.")
        print("(delete data/splits/task34_*.json first if you want to rebuild them)")
        return

    graphs_dir = config.PROCESSED_DIR / "musiccaps_graphs"
    graph_ids = sorted(p.stem for p in graphs_dir.glob("*.pt"))
    if not graph_ids:
        raise FileNotFoundError(f"No graph files found in {graphs_dir}.")
    print(f"Found {len(graph_ids)} audio graphs already built")

    csv_path = config.RAW_DIR / "musiccaps" / "musiccaps-public.csv"
    df = pd.read_csv(csv_path)
    df["aspect_list_parsed"] = df["aspect_list"].apply(ast.literal_eval)
    caption_by_id = {row["ytid"]: row["caption"] for _, row in df.iterrows()}
    tags_by_id = {row["ytid"]: row["aspect_list_parsed"] for _, row in df.iterrows()}

    usable_ids = [gid for gid in graph_ids if gid in caption_by_id]
    print(f"{len(usable_ids)} of those also have a caption in the CSV")

    # Build a small tag vocabulary from just these clips.
    counter = Counter()
    for gid in usable_ids:
        counter.update(tags_by_id[gid])
    vocab = [tag for tag, _ in counter.most_common(config.TASK1_TOP_K_TAGS)]
    vocab_set = set(vocab)

    records = []
    for gid in usable_ids:
        clip_tags = [t for t in tags_by_id[gid] if t in vocab_set]
        if not clip_tags:
            continue
        records.append({
            "track_id": gid,
            "text": caption_by_id[gid],
            "tags": clip_tags,
            "graph_path": f"data/processed/musiccaps_graphs/{gid}.pt",
        })

    train_val, test = train_test_split(records, test_size=config.TASK3_TEST_FRACTION, random_state=config.SEED)
    val_fraction = config.TASK3_VAL_FRACTION / (1.0 - config.TASK3_TEST_FRACTION)
    train, val = train_test_split(train_val, test_size=val_fraction, random_state=config.SEED)

    config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    (config.SPLITS_DIR / "tag_vocab_task34.json").write_text(json.dumps(vocab, indent=2))
    for name, split in [("train", train), ("val", val), ("test", test)]:
        out_path = config.SPLITS_DIR / f"task34_{name}.json"
        out_path.write_text(json.dumps(split, indent=2))
        print(f"  {name}: {len(split)} paired examples -> {out_path}")

    # Save a few more example graphs, on top of Task 2's, as a representative sample.
    config.GRAPH_EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for rec in train[:10]:
        shutil.copyfile(graphs_dir / f"{rec['track_id']}.pt", config.GRAPH_EXAMPLES_DIR / f"musiccaps_{rec['track_id']}.pt")

    print("Done. Next: python src/task3_train.py  and  python src/task4_train.py")


if __name__ == "__main__":
    main()
