"""
config.py
---------
All the settings for this project in one simple file. No YAML, no argparse
magic -- just plain Python variables that every script imports.

Paths are relative to the project root (the folder that contains this
file), so the code works no matter where the repo is checked out.
"""

from pathlib import Path

# ---------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
GRAPH_EXAMPLES_DIR = DATA_DIR / "graph_examples"

RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"

# ---------------------------------------------------------------------
# General
# ---------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------
# Audio / graph settings (Task 2, Task 3, Task 4)
# ---------------------------------------------------------------------
SAMPLE_RATE = 22050
N_MELS = 128
N_CHROMA = 12
N_MFCC = 20
SEGMENT_SECONDS = 2.0          # length of one graph node / mel window, in seconds
SIMILARITY_THRESHOLD = 0.8     # cosine similarity above this -> extra graph edge
GRAPH_IN_CHANNELS = N_CHROMA + N_MFCC  # 32 -- node feature size (chroma + MFCC)

# ---------------------------------------------------------------------
# Task 1: BERT tag classifier (MusicCaps captions)
# ---------------------------------------------------------------------
TASK1_MODEL_NAME = "distilbert-base-uncased"
TASK1_TOP_K_TAGS = 30           # small tag set so training doesn't take forever
TASK1_MAX_TEXT_LEN = 64
TASK1_TRAIN_SIZE = 1200         # small subset so training finishes quickly
TASK1_VAL_SIZE = 250
TASK1_TEST_SIZE = 250
TASK1_BATCH_SIZE = 16
TASK1_EPOCHS = 20               # was 6 -- validation macro-F1 was still rising at epoch 6
TASK1_LR = 2e-5

# ---------------------------------------------------------------------
# Task 2: GNN genre classifier + CNN baseline (GTZAN)
# ---------------------------------------------------------------------
GTZAN_GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]
TASK2_GNN_HIDDEN = 64
TASK2_GNN_OUT = 64
TASK2_GNN_LAYERS = 2
TASK2_GNN_ARCH = "graphsage"   # "graphsage" or "gat"
TASK2_GNN_DROPOUT = 0.3
TASK2_GNN_LR = 5e-4
TASK2_GNN_BATCH_SIZE = 32
TASK2_GNN_EPOCHS = 60

TASK2_CNN_TARGET_FRAMES = 1300
TASK2_CNN_DROPOUT = 0.3
TASK2_CNN_LR = 5e-4
TASK2_CNN_BATCH_SIZE = 16
TASK2_CNN_EPOCHS = 30

# ---------------------------------------------------------------------
# Task 3: GNN + BERT fusion (MusicCaps audio + captions, paired)
# ---------------------------------------------------------------------
TASK3_VAL_FRACTION = 0.15
TASK3_TEST_FRACTION = 0.15
TASK3_ATTN_DIM = 128
TASK3_BATCH_SIZE = 16
TASK3_EPOCHS = 25               # was 10 -- validation macro-F1 was still rising at epoch 10
TASK3_LR = 5e-5

# ---------------------------------------------------------------------
# Task 4: contrastive graph/text retrieval (MusicCaps, same pairs as Task 3)
# ---------------------------------------------------------------------
TASK4_EMBED_DIM = 128
TASK4_TEMPERATURE = 0.07
TASK4_BATCH_SIZE = 128          # more negatives per InfoNCE anchor -- the main lever tried after the 4.7x dataset merge
TASK4_EPOCHS = 20
TASK4_LR = 1e-4
