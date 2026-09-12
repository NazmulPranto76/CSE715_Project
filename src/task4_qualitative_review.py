"""
task4_qualitative_review.py
------------------------------
Four-task paper diagnostic: for retrieval failures, is the top-1 retrieved
clip a genuine mismatch, or a clip that is a semantically reasonable answer
to the caption but happens not to be the one official paired clip (a
"false negative" of exact-pair evaluation, not a real retrieval error)?

Picks 20 successes (true track ranked in the top 3) and 20 failures (true
track ranked outside the top 10), and for each, saves the query caption,
the true track's own caption, and the top-1 retrieved track's caption side
by side, so a human reader can judge semantic reasonableness directly.
This script does not judge automatically -- it prepares the material for a
human judgment pass, which is recorded separately in the paper/report text.
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from common import get_device
from task4_contrastive import DualEncoder
from task4_evaluate import encode_test_set
from datasets import FusionPairDataset


def main():
    device = get_device()
    test_ds = FusionPairDataset(config.SPLITS_DIR / "task34_test.json", config.SPLITS_DIR / "tag_vocab_task34.json")

    model = DualEncoder(
        bert_model_name=config.TASK1_MODEL_NAME, gnn_in_channels=config.GRAPH_IN_CHANNELS,
        embed_dim=config.TASK4_EMBED_DIM,
    ).to(device)
    model.load_state_dict(torch.load(config.CHECKPOINT_DIR / "task4_contrastive_best.pt", map_location=device))
    model.eval()

    g, t = encode_test_set(model, test_ds, device)
    sim = t @ g.t()  # (num_captions, num_clips), audio->caption direction uses sim.t()

    n = len(test_ds)
    ranks = []
    for i in range(n):
        order = sim[i].argsort(descending=True).tolist()
        rank = order.index(i) + 1  # 1-indexed rank of the true clip for caption i
        ranks.append(rank)

    successes = [i for i in range(n) if ranks[i] <= 3][:20]
    failures = [i for i in range(n) if ranks[i] > 10][:20]

    def row(i, label):
        top1_idx = sim[i].argmax().item()
        return {
            "label": label, "index": i, "rank_of_true": ranks[i],
            "query_caption": test_ds.records[i]["text"],
            "true_track_id": test_ds.records[i]["track_id"],
            "true_track_own_caption": test_ds.records[i]["text"],  # caption IS the true track's caption
            "top1_retrieved_track_id": test_ds.records[top1_idx]["track_id"],
            "top1_retrieved_caption": test_ds.records[top1_idx]["text"],
        }

    out = {
        "n_test": n,
        "successes": [row(i, "success") for i in successes],
        "failures": [row(i, "failure") for i in failures],
        "note": "true_track_own_caption == query_caption by construction (the query caption IS "
                "the true track's caption); shown for clarity when comparing against top1_retrieved_caption.",
    }
    out_path = config.RESULTS_DIR / "task4_qualitative_review.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Successes: {len(successes)}, Failures: {len(failures)} (of {n} test examples)")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
