#!/usr/bin/env python
"""
Runs every training / plot script in the correct order so the marker
(or you) can regenerate the entire pipeline in one command:

    python src/retrain_all.py
"""
from pathlib import Path
import subprocess, sys

root     = Path(__file__).resolve().parents[1]
scripts  = [
    "build_priority_labels.py",
    "train_priority_model.py",
    "build_labels_enron.py",
    "train_priority_model.py",          # ensure ML‑priority ready
    "main.py",                          # category model only
    ("main.py", "--minilm"),            # MiniLM baseline numbers
    "error_analysis.py",
    "make_charts.py"
]

for item in scripts:
    if isinstance(item, tuple):
        script, *extra = item
        cmd = [sys.executable, str(root / "src" / script), *extra]
        name = f"{script} {' '.join(extra)}"
    else:
        script = item
        cmd    = [sys.executable, str(root / "src" / script)]
        name   = script

    print(f"🟢  Running {name}")
    ret = subprocess.call(cmd)
    if ret != 0:
        raise RuntimeError(f"{name} failed!")

print("\n All stages completed without error.")
