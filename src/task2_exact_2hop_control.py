"""
task2_exact_2hop_control.py
------------------------------
Reproduces, from this canonical codebase, the exact 2-hop neighbor
aggregation control the four-task paper cites for Task 2 -- previously
only available as project/caps-gi/src/test_1a_exact_precompute.py (flagged
by a round-7 audit as a reproducibility gap: the number was real, but not
reproducible from either codebase named as this paper's evidence base).
Ported here unchanged in method, ordered here to make the control
independently re-derivable from this repository.

Question: does GTZAN genre classification need learned message passing, or
does an EXACT (non-learned, non-sketched) precomputed neighborhood
aggregate already capture most of the signal?

Three graph-level readouts, same downstream MLP head, same splits, one seed:
  B0 (raw MLP)    : mean_v x_v                                   (32-d)
  B2 (1-hop exact): mean_v [x_v || mean_{u in N(v)} x_u]         (64-d)
  B3 (2-hop exact): mean_v [x_v || 1-hop_v || 2-hop_v]           (96-d)
where 2-hop propagates the exact 1-hop aggregate through edges once more.

Training recipe (disclosed precisely because it differs from GraphSAGE's,
Section VIII of the paper): full-batch Adam, weight_decay=1e-4, fixed 200
epochs, no dropout, best-checkpoint-by-validation-macro-F1 selection.
GraphSAGE (train.py) uses AdamW, weight_decay=1e-4, mini-batches of 32,
dropout (see task2_exact_2hop_control.py's own header note on the
dropout-mismatch bug: train.py does not pass GraphSAGE its configured
dropout, so it trains at the class default 0.2, not the config's 0.4),
and early stopping over up to 100 epochs.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
SPLITS_DIR = REPO / "data" / "splits"
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def exact_hop_aggregate(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Exact mean over each node's direct neighbors (self excluded); nodes
    with no neighbors get a zero vector. No learning, no approximation."""
    n = x.shape[0]
    agg = torch.zeros_like(x)
    deg = torch.zeros(n)
    if edge_index.numel() > 0:
        src, dst = edge_index[0], edge_index[1]
        agg.index_add_(0, dst, x[src])
        deg.index_add_(0, dst, torch.ones(src.shape[0]))
    deg = deg.clamp(min=1).unsqueeze(-1)
    return agg / deg


def graph_features(path: Path):
    g = torch.load(REPO / path, weights_only=False)
    x, edge_index = g.x, g.edge_index
    hop1 = exact_hop_aggregate(x, edge_index)
    hop2 = exact_hop_aggregate(hop1, edge_index)
    raw = x.mean(dim=0)
    f_b0 = raw
    f_b2 = torch.cat([raw, hop1.mean(dim=0)])
    f_b3 = torch.cat([raw, hop1.mean(dim=0), hop2.mean(dim=0)])
    return f_b0, f_b2, f_b3


def load_split(name):
    records = json.loads((SPLITS_DIR / f"gtzan_{name}.json").read_text())
    b0, b2, b3, y = [], [], [], []
    for r in records:
        f0, f2, f3 = graph_features(Path(r["graph_path"]))
        b0.append(f0); b2.append(f2); b3.append(f3)
        y.append(r["genre_idx"])
    return (torch.stack(b0), torch.stack(b2), torch.stack(b3), torch.tensor(y, dtype=torch.long))


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=10):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

    def forward(self, x):
        return self.net(x)


def macro_f1(preds, labels, n_classes=10):
    f1s = []
    for c in range(n_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s)


def train_eval(name, Xtr, ytr, Xval, yval, Xte, yte, epochs=200, lr=1e-3):
    set_seed(SEED)
    model = MLP(Xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best_val, best_state = -1, None
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xtr), ytr)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(Xval).argmax(-1)
            val_f1 = macro_f1(val_pred, yval)
        if val_f1 > best_val:
            best_val, best_state = val_f1, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        te_pred = model(Xte).argmax(-1)
    acc = (te_pred == yte).float().mean().item()
    f1 = macro_f1(te_pred, yte)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{name}] dim={Xtr.shape[1]:>3}  params={n_params}  best_val_f1={best_val:.4f}  test_acc={acc:.4f}  test_macro_f1={f1:.4f}")
    return {"dim": Xtr.shape[1], "params": n_params, "test_acc": acc, "test_macro_f1": f1}


def main():
    print("Loading GTZAN graphs (train/val/test) and computing exact features...")
    tr = load_split("train")
    val = load_split("val")
    te = load_split("test")

    results = {}
    for i, name in [(0, "B0_raw_mlp_nograph"), (1, "B2_1hop_exact"), (2, "B3_2hop_exact")]:
        results[name] = train_eval(name, tr[i], tr[3], val[i], val[3], te[i], te[3])

    results["reference_GraphSAGE_existing"] = {
        "test_acc": 0.607, "test_macro_f1": 0.598,
        "note": "cited from results/metrics.json's gnn entry, not retrained here",
    }
    results["reference_CNN_baseline_existing"] = {
        "test_acc": 0.780, "test_macro_f1": 0.765,
        "note": "cited from results/metrics.json's cnn_baseline entry, not retrained here",
    }

    out_path = REPO / "results" / "graph_repair" / "task2_exact_2hop_control.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
