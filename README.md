# GNN-Based BERT for Understanding Context from Music

The goal is to understand the **context** of a piece of
music -- genre, mood, instruments -- using two different kinds of input:

1. **Text** -- a caption describing the music (via BERT/DistilBERT)
2. **Audio structure** -- the music turned into a graph, one node per short
   time segment (via a Graph Neural Network)

There are four tasks:

- **Task 1** -- classify tags from a caption (text only)
- **Task 2** -- classify genre from an audio graph, compared against a plain
  CNN baseline
- **Task 3** -- combine text + audio graph (fusion) to predict tags
- **Task 4** -- learn to match a caption to its audio clip with no labels at
  all (contrastive retrieval)

## 1. Folder structure

```
project_submission_simple/
  README.md
  requirements.txt
  config.py                 <- every setting lives here (paths, sizes, hyperparameters)

  data/
    raw/musiccaps/           <- MusicCaps captions CSV (ships with this repo)
    raw/gtzan/                <- GTZAN audio (you download this, see Task 2 below)
    processed/graphs/           <- GTZAN segment graphs (Task 2)
    processed/musiccaps_graphs/  <- MusicCaps segment graphs (Task 3/4, ships with this repo)
    splits/                        <- train/val/test lists + tag vocabularies (JSON)
    graph_examples/                 <- example graph files (a representative sample)

  src/
    common.py                 <- seed + device helpers
    audio_features.py          <- audio -> mel/chroma/MFCC -> time segments
    graph_builder.py             <- segments -> graph (nodes + edges)
    gnn_model.py                   <- Task 2 GraphSAGE/GAT model
    bert_encoder.py                 <- Task 1 DistilBERT model (also used by Task 3/4)
    cnn_baseline.py                  <- Task 2's CNN-on-mel-spectrogram baseline
    task3_fusion.py                   <- Task 3 GNN+BERT fusion model (4 modes)
    task4_contrastive.py               <- Task 4 dual encoder + loss
    datasets.py                         <- PyTorch Dataset classes for all 4 tasks
    prepare_task1.py                     <- Task 1 data prep
    prepare_task2_graphs.py               <- Task 2 data prep
    prepare_task34_graphs.py               <- Task 3/4 data prep (shared)
    task1_train.py / task1_evaluate.py
    task2_gnn.py / task2_cnn.py / task2_evaluate.py
    task3_train.py / task3_evaluate.py
    task4_train.py / task4_evaluate.py
    task4_ablation_nograph.py / task4_multiseed_robustness.py / task4_qualitative_review.py
    task3_case_studies.py / task3_complementarity_diagnostic.py / task3_complementarity_by_polarity.py
    task2_exact_2hop_control.py / task3_branch_zeroing.py
    gtzan_duplicate_audit.py             <- follow-up diagnostics behind report/final_report.pdf

  scripts/
    run_task1.py .. run_task4.py    <- one command per task (prepare -> train -> evaluate)

  results/
    metrics.json, task*_metrics.json / *.csv   <- final numbers
    checkpoints/                      <- best model weights per task
    plots/                            <- required charts

  notebooks/demo_context.ipynb    <- one short end-to-end demo
  report/                         <- the final report (LaTeX source + final_report.pdf)
```

**A note on the diagnostic scripts above the `scripts/` line.** All of them run end-to-end
against this repo's own data and, where relevant, its own checkpoints: `task3_branch_zeroing.py`,
`task3_complementarity_diagnostic.py`, `task3_complementarity_by_polarity.py`, and
`task3_control_epoch_check.py` compute this repo's *own* Task 3 numbers, which
`report/final_report.pdf`'s Discussion section reports as a follow-up check that disagreed
with its primary Task 3 finding (see that section for why). `task2_exact_2hop_control.py`
and `gtzan_duplicate_audit.py` reproduce the Task 2 exact-2-hop control and the GTZAN
duplicate/leakage audit that `report/final_report.pdf` cites, using this repo's own GTZAN
splits directly; `gtzan_duplicate_audit.py` additionally needs the raw GTZAN audio described
under Datasets below (not shipped, same as Task 2 training). Note that
`report/final_report.pdf`'s own *headline* Task 1-3 numbers come from a separate, larger-scale
run at a bigger data size than this repo trains at (Section 2 of the report explains that
distinction) -- these two scripts reproduce the specific cited diagnostic results, not that
larger-scale run's main table.

## 2. Datasets

| Task | Dataset | Why |
|---|---|---|
| 1 | MusicCaps captions | Caption + short tag phrases per clip -- text only, no audio needed |
| 2 | GTZAN | 10 genres x ~100 30-second clips -- standard genre-classification set |
| 3 & 4 | MusicCaps (paired) | Clips that have both a caption AND an audio graph, so text and audio can be combined |

The MusicCaps captions CSV and a pre-built set of MusicCaps audio graphs
already ship with this repo (both are small). GTZAN audio does **not** ship
with this repo (1.3 GB) -- download it yourself:

1. Get it from [Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
   (only the `genres_original/` folder is needed).
2. Place it at `data/raw/gtzan/genres_original/<genre>/<file>.wav`.
3. Run `python src/prepare_task2_graphs.py` to build the graphs.

If you skip steps 1-3, Task 2's GNN still works using the graphs already
included in `data/processed/graphs/` -- only the CNN baseline (which reads
raw audio directly) needs the actual GTZAN files.

Sample counts used for training:
Task 1 uses 1,200/250/250 captions (a deliberately small subset -- see
`config.py` to use more of the 4,786 available), Task 2 uses all 999 GTZAN
tracks (699/150/150), and Task 3/4 use all currently-cached paired
examples, 3,136/673/673.

## 3. Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# Install PyTorch first (pick the command for your GPU/CPU from pytorch.org)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt
```

An NVIDIA GPU is recommended for Tasks 1/3/4 (BERT fine-tuning) but
everything also runs on CPU, just slower.

## 4. How to run each task

```bash
python scripts/run_task1.py     # BERT tag classifier
python scripts/run_task2.py     # GNN + CNN baseline on GTZAN
python scripts/run_task3.py     # GNN+BERT fusion (4-way comparison)
python scripts/run_task4.py     # Contrastive retrieval
```

Each `run_task*.py` calls that task's `prepare -> train -> evaluate` scripts
in order. Run from the project root. If you'd rather run a step by hand:

```bash
python src/task1_train.py
python src/task1_evaluate.py
```

## 5. Expected outputs

After running all four:

```
results/task1_metrics.json, task1_predictions.csv, task1_examples.json
results/task2_metrics.json
results/task3_ablation.csv
results/task4_retrieval.json, task4_examples.csv
results/plots/task1_f1.png, task2_comparison.png, task3_ablation.png,
              task3_tsne_concat.png, task4_recall.png
results/checkpoints/*.pt
```

## 6. Best results

| Task | Model | Dataset | Test samples | Main metric | Result |
|---|---|---|---|---|---|
| 1 | DistilBERT | MusicCaps | 250 | Macro-F1 | 0.727 |
| 2 | GraphSAGE (GNN) | GTZAN | 150 | Macro-F1 | 0.616 |
| 2 | CNN (mel-spectrogram) | GTZAN | 150 | Macro-F1 | 0.762 |
| 3 | BERT only | MusicCaps | 673 | Macro-F1 | 0.787 |
| 3 | GNN only | MusicCaps | 673 | Macro-F1 | 0.187 |
| 3 | BERT+GNN concat | MusicCaps | 673 | Macro-F1 | 0.777 |
| 3 | BERT+GNN cross-attention | MusicCaps | 673 | Macro-F1 | 0.760 |
| 4 | GNN+BERT contrastive | MusicCaps | 673 | Audio->Text R@10 | 0.086 |

Task 1 and Task 3 were retrained with a larger epoch budget after checking their validation
curves showed they were still improving, not plateaued, at the original cutoffs (6 and 10
epochs respectively -- see `config.py`'s `TASK1_EPOCHS`/`TASK3_EPOCHS` comments). Task 2 and
Task 4 were left as-is; their validation curves had already plateaued. Note that even at this
larger budget, BERT alone (0.787) still beats both fusion modes (0.777, 0.760) here, and the
graph-only model badly underperforms (0.187) -- the same qualitative pattern
`report/final_report.pdf` reports at its larger data scale, now also visible in this repo's
own, independently-trained numbers.

This table is this repo's own Task 1-4 pipeline, at its own (smaller) data scale --
`report/final_report.pdf` reports different, larger-scale numbers for Tasks 1-3 (see the note
above), and is the source to cite for the actual findings,
including the direct test of whether the graph branch matters at all in Tasks 2/3/4: not
detectably for Task 2 or Task 4, but Task 3 is unresolved -- this repo's own branch-zeroing
and complementarity checks (`task3_branch_zeroing.py`,
`task3_complementarity_by_polarity.py`) find a real graph contribution here, disagreeing with
`report/final_report.pdf`'s primary Task 3 evidence; the Discussion section explains what we
do and do not know about why.

## 7. Demo notebook

`notebooks/demo_context.ipynb` loads one trained checkpoint per task and
shows a single example prediction end-to-end (caption -> predicted tags,
audio -> genre, caption -> top-3 retrieved clips).
