# Project Status

| Task | Status |
|---|---|
| Task 1 -- BERT tag classification | **DONE** |
| Task 2 -- GNN music structure + CNN baseline | **DONE** |
| Task 3 -- GNN + BERT fusion (4-way comparison) | **DONE** |
| Task 4 -- Contrastive graph/text retrieval | **DONE** |

All four tasks were run end-to-end on real data -- see `results/` for the
raw metrics files these numbers come from, and `results/plots/` for the
matching charts.

## Datasets used

| Task | Dataset | Samples used (train / val / test) |
|---|---|---|
| 1 | MusicCaps captions (text only, no audio) | 1,200 / 250 / 250 (of 4,786 usable clips out of 5,521 total) |
| 2 | GTZAN (10 genres x ~100 clips) | 699 / 150 / 150 (all of GTZAN) |
| 3 | MusicCaps paired (caption + audio graph) | 858 / 146 / 142 (subset of ~1,146 available paired clips) |
| 4 | Same MusicCaps pairs as Task 3 | 858 / 146 / 142 |

These are small subsets on purpose so the whole pipeline trains quickly.
`config.py` has the exact sizes -- change them and rerun the matching
`prepare_*.py` script to try more data.

## Best results

| Task | Model | Metric | Result |
|---|---|---|---|
| 1 | DistilBERT | Macro-F1 / Micro-F1 / AUC-PR | 0.566 / 0.594 / 0.756 |
| 2 | GraphSAGE (GNN) | Accuracy / Macro-F1 | 0.620 / 0.616 |
| 2 | CNN (mel-spectrogram baseline) | Accuracy / Macro-F1 | 0.780 / 0.762 |
| 3 | Fusion -- bert_only | Macro-F1 | 0.666 |
| 3 | Fusion -- gnn_only | Macro-F1 | 0.167 |
| 3 | Fusion -- concat | Macro-F1 | 0.629 |
| 3 | Fusion -- cross_attention | Macro-F1 | 0.500 |
| 4 | Contrastive (InfoNCE) | Audio->Text R@10 / Text->Audio R@10 | 0.239 / 0.218 |

Full per-tag/per-genre breakdowns and training curves are in `results/*.json`
and `results/plots/*.png`.

## Notes on the results (not hidden, explained)

- **Task 2:** the CNN baseline beats the GNN (0.762 vs. 0.616 Macro-F1).
  Reported honestly instead of picking hyperparameters until the GNN wins --
  GTZAN genres correlate a lot with raw timbre/texture, which the CNN sees
  at full time resolution, while the graph's 2-second segments compress
  that detail in exchange for explicit temporal/similarity structure.
- **Task 3:** `gnn_only` (graph structure with zero text) collapses to
  Macro-F1 0.167, far below every mode that includes text. MusicCaps' most
  common tags ("low quality", "instrumental", "medium tempo", ...) describe
  recording quality and production style more than they describe sound, so
  a model that only sees the audio graph has little to go on for these
  particular labels. `cross_attention` also scores below plain `concat`
  here (0.500 vs. 0.629) -- the extra attention parameters likely need more
  training data than this small subset provides to pay off.
- **Task 4:** retrieval recall is modest (R@10 around 0.22-0.24), which is
  expected for a ~140-clip test set and a short training run.

## Main outputs / checkpoints

- `results/checkpoints/task1_bert_best.pt` -- Task 1 DistilBERT classifier
- `results/checkpoints/task2_gnn_best.pt` -- Task 2 GraphSAGE classifier
- `results/checkpoints/task2_cnn_best.pt` -- Task 2 CNN baseline
- `results/checkpoints/task3_fusion_{bert_only,gnn_only,concat,cross_attention}_best.pt` -- Task 3 comparison
- `results/checkpoints/task4_contrastive_best.pt` -- Task 4 dual encoder
- `data/graph_examples/` -- 35 example segment graphs (25 GTZAN + 10 MusicCaps)

Checkpoints that include a fine-tuned DistilBERT are large (~250 MB each,
since the whole model gets saved, not just the head) -- not included in
this repo's git history, see `requirements.txt`/README for how to
regenerate them by rerunning the `task*_train.py` scripts.
