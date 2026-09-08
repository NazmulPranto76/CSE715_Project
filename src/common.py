"""
common.py
---------
Small helper functions shared by every task: setting random seeds, picking
a device (GPU if available, else CPU), and saving/loading JSON metrics.
Nothing fancy on purpose -- this is the only "shared infrastructure" file
in the whole project.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed=42):
    """Make results reproducible across numpy / torch / python's random module."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    """Use the GPU automatically if one is available, otherwise fall back to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"Saved {path}")


def load_json(path):
    with open(path) as f:
        return json.load(f)
