"""
task3_case_studies.py
------------------------
Produces the "3 case studies showing graph paths + caption/lyric alignment"
artifact required for Task 3. For a small set of test examples, shows:
  - the caption text and ground-truth tags
  - the segment graph's structure: how many segments, and which pairs of
    non-adjacent segments the similarity threshold connects (a "path"
    through the graph beyond the trivial temporal chain)
  - concat-fusion vs. bert-only predicted tags, so a reader can see
    directly whether the graph changed anything for this specific example

Selection is not cherry-picked for a flattering story: we take the first 3
test examples (in dataset order) where concat and bert_only actually
disagree on at least one tag, so there is something concrete to compare,
then report what we find plainly -- including if the graph's extra edges
don't obviously explain the disagreement.
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from common import get_device
from task3_fusion import GNNBertFusion
from datasets import FusionPairDataset, make_fusion_collate_fn


def load_model(mode, num_labels, device):
    model = GNNBertFusion(
        bert_model_name=config.TASK1_MODEL_NAME, gnn_in_channels=config.GRAPH_IN_CHANNELS,
        num_labels=num_labels, attn_dim=config.TASK3_ATTN_DIM, mode=mode,
    ).to(device)
    ckpt = config.CHECKPOINT_DIR / f"task3_fusion_{mode}_best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


def describe_graph(graph):
    """Non-adjacent (i, j) pairs connected by a similarity edge, i.e. the
    part of the graph that is not just the trivial temporal chain."""
    n_nodes = graph.x.shape[0]
    edge_index = graph.edge_index.t().tolist()
    similarity_pairs = set()
    for i, j in edge_index:
        if abs(i - j) > 1 and i < j:
            similarity_pairs.add((i, j))
    return {
        "n_nodes": n_nodes,
        "n_temporal_edges": sum(1 for i, j in edge_index if abs(i - j) == 1),
        "n_similarity_edges_nonadjacent": len(similarity_pairs),
        "similarity_pairs": sorted(similarity_pairs),
    }


def predict_tags(model, mode, record, vocab, tokenizer_bert, device):
    graph = torch.load(record["graph_path"], weights_only=False).to(device)
    tokenized = tokenizer_bert.tokenize([record["text"]], max_length=config.TASK1_MAX_TEXT_LEN)
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    batch = torch.zeros(graph.x.shape[0], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _ = model(input_ids, attention_mask, graph.x, graph.edge_index, batch)
    probs = torch.sigmoid(logits).cpu().squeeze(0)
    pred_idx = (probs > 0.5).nonzero(as_tuple=True)[0].tolist()
    return sorted(vocab[i] for i in pred_idx)


def main():
    device = get_device()
    test_ds = FusionPairDataset(config.SPLITS_DIR / "task34_test.json", config.SPLITS_DIR / "tag_vocab_task34.json")
    vocab = test_ds.vocab

    concat_model = load_model("concat", len(vocab), device)
    bert_model = load_model("bert_only", len(vocab), device)

    candidates = []
    for i, record in enumerate(test_ds.records):
        concat_tags = set(predict_tags(concat_model, "concat", record, vocab, concat_model.bert, device))
        bert_tags = set(predict_tags(bert_model, "bert_only", record, vocab, bert_model.bert, device))
        if concat_tags != bert_tags:
            candidates.append((i, record, concat_tags, bert_tags))
        if len(candidates) >= 3:
            break

    case_studies = []
    for i, record, concat_tags, bert_tags in candidates:
        graph = torch.load(record["graph_path"], weights_only=False)
        graph_desc = describe_graph(graph)
        case_studies.append({
            "index": i,
            "track_id": record.get("track_id", record.get("ytid", "unknown")),
            "caption": record["text"],
            "ground_truth_tags": sorted(record["tags"]),
            "graph_structure": graph_desc,
            "concat_predicted_tags": sorted(concat_tags),
            "bert_only_predicted_tags": sorted(bert_tags),
            "tags_only_concat_gets": sorted(concat_tags - bert_tags),
            "tags_only_bert_gets": sorted(bert_tags - concat_tags),
        })

    out = {
        "description": "3 case studies for Task 3, selected as the first 3 test examples "
                        "(dataset order) where concat and bert_only disagree on at least one "
                        "predicted tag. 'graph_structure' lists non-adjacent segment pairs "
                        "connected by a similarity edge (beyond the trivial temporal chain), "
                        "so a reader can check directly whether the graph's extra edges have "
                        "any visible relationship to the prediction difference.",
        "case_studies": case_studies,
    }
    out_path = config.RESULTS_DIR / "task3_case_studies.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved {len(case_studies)} case studies to {out_path}")


if __name__ == "__main__":
    main()
