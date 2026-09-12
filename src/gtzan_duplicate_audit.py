"""
gtzan_duplicate_audit.py
--------------------------
Four-task paper diagnostic: exact-duplicate and split-leakage audit for
GTZAN, referenced in the paper's data-integrity section. Hashes every raw
GTZAN audio file, finds byte-identical duplicate pairs, and checks whether
any duplicate pair crosses the train/val/test split boundary used
throughout this project.

Read-only. Does not modify the dataset or the splits.
"""

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main():
    audio_dir = REPO / "data" / "raw" / "gtzan" / "genres_original"
    files = sorted(audio_dir.rglob("*.wav")) + sorted(audio_dir.rglob("*.au"))
    print(f"GTZAN audio files found: {len(files)}")

    hashes = {}
    dupe_pairs = []
    for f in files:
        h = hashlib.md5(f.read_bytes()).hexdigest()
        if h in hashes:
            dupe_pairs.append((hashes[h].stem, f.stem))
        else:
            hashes[h] = f
    print(f"Exact byte-identical duplicate pairs: {len(dupe_pairs)}")

    splits_dir = REPO / "data" / "splits"
    track_split = {}
    for split in ("train", "val", "test"):
        for rec in json.loads((splits_dir / f"gtzan_{split}.json").read_text()):
            tid = rec.get("track_id", rec.get("id"))
            track_split[tid] = split

    rows = []
    n_cross_split = 0
    n_cross_genre = 0
    for a, b in dupe_pairs:
        genre_a, genre_b = a.split(".")[0], b.split(".")[0]
        split_a, split_b = track_split.get(a, "UNKNOWN"), track_split.get(b, "UNKNOWN")
        cross_split = split_a != split_b and "UNKNOWN" not in (split_a, split_b)
        cross_genre = genre_a != genre_b
        if cross_split:
            n_cross_split += 1
        if cross_genre:
            n_cross_genre += 1
        rows.append({
            "track_a": a, "track_b": b, "split_a": split_a, "split_b": split_b,
            "cross_split_leakage": cross_split, "cross_genre_label": cross_genre,
        })
        flag = " <-- CROSS-SPLIT" if cross_split else ""
        flag += " <-- CROSS-GENRE" if cross_genre else ""
        print(f"  {a} ({split_a}) <-> {b} ({split_b}){flag}")

    # How many distinct test tracks have an exact duplicate sitting in train?
    test_tracks_with_train_duplicate = set()
    for r in rows:
        if r["cross_split_leakage"]:
            for x, sx in [(r["track_a"], r["split_a"]), (r["track_b"], r["split_b"])]:
                if sx == "test":
                    test_tracks_with_train_duplicate.add(x)

    out = {
        "n_files_hashed": len(files),
        "n_exact_duplicate_pairs": len(dupe_pairs),
        "n_cross_split_leakage_pairs": n_cross_split,
        "n_cross_genre_label_pairs": n_cross_genre,
        "n_test_tracks_with_a_train_duplicate": len(test_tracks_with_train_duplicate),
        "test_tracks_with_a_train_duplicate": sorted(test_tracks_with_train_duplicate),
        "pairs": rows,
    }
    out_path = REPO / "results" / "gtzan_duplicate_audit.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n{len(test_tracks_with_train_duplicate)} of 150 test tracks have an exact "
          f"duplicate in the training set.")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
