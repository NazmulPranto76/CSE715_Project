"""
run_task3.py
------------
Runs all of Task 3 in order: prepare paired data -> train all 4 ablation
modes -> evaluate all 4.

Usage:
    python scripts/run_task3.py
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
    run("task3_train.py")
    run("task3_evaluate.py")
