# GNN-Based BERT for Understanding Context from Music

A course project. The goal is to understand the **context** of a piece of
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
  STATUS.md                 <- what's done, dataset sizes, best results
  requirements.txt
  config.py                 <- every setting lives here (paths, sizes, hyperparameters)

  data/
    raw/musiccaps/           <- MusicCaps captions CSV (ships with this repo)
    raw/gtzan/                <- GTZAN audio (you download this, see Task 2 below)
    processed/graphs/           <- GTZAN segment graphs (Task 2)
    processed/musiccaps_graphs/  <- MusicCaps segment graphs (Task 3/4, ships with this repo)
    splits/                        <- train/val/test lists + tag vocabularies (JSON)
    graph_examples/                 <- example graph files for the submission requirement

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

  scripts/
    run_task1.py .. run_task4.py    <- one command per task (prepare -> train -> evaluate)

  results/
    task*_metrics.json / *.csv       <- final numbers
    checkpoints/                      <- best model weights per task
    plots/                            <- required charts

  notebooks/demo_context.ipynb    <- one short end-to-end demo
  report_material/                <- PROJECT_REPORT.md (full writeup), results_summary.md, tables.md
```

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

Sample counts used for training (see `STATUS.md` for the full breakdown):
Task 1 uses 1,200/250/250 captions, Task 2 uses all 999 GTZAN tracks
(699/150/150), and Task 3/4 use 858/146/142 paired examples. These are
small subsets on purpose so the whole project runs in well under 30
minutes -- see `config.py` to change the sizes.

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
| 1 | DistilBERT | MusicCaps | 250 | Macro-F1 | 0.566 |
| 2 | GraphSAGE (GNN) | GTZAN | 150 | Macro-F1 | 0.616 |
| 2 | CNN (mel-spectrogram) | GTZAN | 150 | Macro-F1 | 0.762 |
| 3 | BERT+GNN concat | MusicCaps | 142 | Macro-F1 | 0.629 |
| 3 | BERT+GNN cross-attention | MusicCaps | 142 | Macro-F1 | 0.500 |
| 4 | GNN+BERT contrastive | MusicCaps | 142 | Audio->Text R@10 | 0.239 |

See `STATUS.md` for the full breakdown, including two results that came out
lower than you might expect (the CNN beating the GNN in Task 2, and the
graph-only mode collapsing in Task 3) -- both are real, both are explained
there.

## 7. Demo notebook

`notebooks/demo_context.ipynb` loads one trained checkpoint per task and
shows a single example prediction end-to-end (caption -> predicted tags,
audio -> genre, caption -> top-3 retrieved clips).
