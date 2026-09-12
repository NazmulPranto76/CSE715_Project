"""
task4_retrieval_examples.py
------------------------------
Produces the qualitative retrieval artifact required for Task 4: for a
sample of query captions, the top-3 retrieved audio clips (by caption, not
just the top-1 clip). Complements task4_qualitative_review.py, which looks
at top-1 only for the success/failure semantic-reasonableness check.

Selects a spread of 10 examples across the rank distribution (some easy,
some hard), not just successes, so the artifact shows real model behavior
rather than only cherry-picked wins.
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
    sim = t @ g.t()  # (num_captions, num_clips): row i = caption i's similarity to every clip

    n = len(test_ds)
    ranks = []
    for i in range(n):
        order = sim[i].argsort(descending=True).tolist()
        ranks.append(order.index(i) + 1)

    # Spread the 10 examples across the rank distribution: not just easy
    # successes -- pick every (n // 10)-th example by sorted true-rank so
    # the sample includes near-misses and clear failures too.
    sorted_by_rank = sorted(range(n), key=lambda i: ranks[i])
    step = max(1, n // 10)
    selected = sorted_by_rank[::step][:10]

    examples = []
    for i in selected:
        top3_idx = sim[i].argsort(descending=True)[:3].tolist()
        examples.append({
            "index": i,
            "rank_of_true_clip": ranks[i],
            "query_caption": test_ds.records[i]["text"],
            "true_track_id": test_ds.records[i]["track_id"],
            "top3_matches": [
                {
                    "rank": rank + 1,
                    "track_id": test_ds.records[idx]["track_id"],
                    "caption": test_ds.records[idx]["text"],
                    "is_true_clip": idx == i,
                }
                for rank, idx in enumerate(top3_idx)
            ],
        })

    out = {
        "n_test": n,
        "description": "10 query captions spread across the true-clip rank distribution "
                        "(not cherry-picked successes), each with its top-3 retrieved clips.",
        "examples": examples,
    }
    out_path = config.RESULTS_DIR / "task4_retrieval_examples_top3.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved {len(examples)} examples to {out_path}")


if __name__ == "__main__":
    main()
