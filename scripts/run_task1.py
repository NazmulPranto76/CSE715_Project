"""
run_task1.py
------------
Runs all of Task 1 in order: prepare data -> train -> evaluate.

Usage:
    python scripts/run_task1.py
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
    run("prepare_task1.py")
    run("task1_train.py")
    run("task1_evaluate.py")
