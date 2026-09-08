# Results Summary

Plain-language summary of what was built and found, for use in the project
report/writeup. All numbers below come directly from `results/*.json` and
`results/*.csv` in this folder -- see `STATUS.md` for the exact files.

## Task 1: BERT tag classification

A DistilBERT model was fine-tuned to predict up to 30 music tags (e.g.
"instrumental", "low quality", "energetic") from a MusicCaps caption. Trained
on 1,200 captions, evaluated on a held-out 250-caption test set.

**Result:** Macro-F1 0.566, Micro-F1 0.594, AUC-PR 0.756. The model learns
common tags well (see `results/task1_predictions.csv` for full per-example
output, `results/task1_examples.json` for 5 human-readable examples) but,
like any multi-label classifier trained on real-world imbalanced tags, does
worse on rare tags than common ones.

## Task 2: GNN music structure vs. CNN baseline

Each GTZAN track was turned into a graph (~15 nodes, one per 2-second
segment, edges from temporal order + audio similarity) and classified by
genre using a GraphSAGE GNN. A plain CNN on the raw mel-spectrogram was
trained as the required baseline, with no graph and no text.

**Result:** the CNN baseline (Macro-F1 0.762) outperforms the GNN
(Macro-F1 0.616) on this run. A plausible reason: GTZAN genres correlate
strongly with raw timbre/spectral texture, which a CNN sees directly at
full time resolution, while the graph's 2-second segments compress that
detail down in exchange for explicit temporal/similarity structure.

## Task 3: GNN + BERT fusion (4-way ablation)

Four ways of combining the caption and the audio graph were compared,
trained on identical data (858 paired MusicCaps examples):

| Mode | Macro-F1 |
|---|---|
| bert_only (text alone) | 0.666 |
| gnn_only (graph alone) | 0.167 |
| concat (simplest fusion) | 0.629 |
| cross_attention (fancier fusion) | 0.500 |

**Finding:** text alone already does most of the work for MusicCaps' tags,
and the graph-alone mode collapses badly. A likely reason: MusicCaps' most
common tags (e.g. "low quality", "instrumental", "medium tempo") describe
recording/production style more than they describe the sound itself, so a
model that only sees the audio graph has little to go on. Cross-attention
also underperforms plain concatenation here -- its extra parameters
probably need more training data/epochs than this run gives it to pay off.
Both results are kept as-is rather than hidden.

## Task 4: Contrastive graph/text retrieval

A dual encoder (same BERT + GNN towers as Task 3, projected into a shared
128-dim space) was trained with a symmetric InfoNCE loss to match captions
with their audio graphs, with no tag labels at all.

**Result (142-clip test pool):** Audio->Text R@1/5/10 = 0.077/0.155/0.239;
Text->Audio R@1/5/10 = 0.035/0.134/0.218. A zero-shot tag-prediction
experiment (embedding tag names as text and thresholding cosine similarity to
the audio embedding, with no supervised tag training at all) reaches
Macro-F1 0.144 -- well below Task 3's supervised fusion models, as expected,
since it never sees a single tag label during training.
