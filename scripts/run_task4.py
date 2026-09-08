"""
run_task4.py
------------
Runs all of Task 4 in order: prepare paired data (shared with Task 3) ->
train the contrastive dual encoder -> evaluate retrieval.

Usage:
    python scripts/run_task4.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"


def run(script_name):
    print(f"\n=== Running {script_name} ===")
    subprocess.run([sys.executable, str(SRC / script_name)], check=True, cwd=ROOT)


if __name__ == "__main__":
    run("prepare_task34_graphs.py")
    run("task4_train.py")
    run("task4_evaluate.py")
