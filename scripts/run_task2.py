"""
run_task2.py
------------
Runs all of Task 2 in order: prepare graphs -> train GNN -> train CNN baseline -> evaluate both.

Usage:
    python scripts/run_task2.py
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
    run("prepare_task2_graphs.py")
    run("task2_gnn.py")
    run("task2_cnn.py")
    run("task2_evaluate.py")
