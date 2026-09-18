# -*- coding: utf-8 -*-

import os
import subprocess
import sys
from pathlib import Path


# ============================================================
# Project paths
# ============================================================
ROOT = Path(__file__).resolve().parents[1]

BDE_SCRIPT = ROOT / "07_BDE" / "BDE.py"
BDE_INPUT = ROOT / "07_BDE" / "data" / "BDE.csv"
BDE_RESULTS = ROOT / "results" / "07_BDE" / "BDE"


# ============================================================
# Environment
# ============================================================
env = os.environ.copy()

env["DOPO_BDE_INPUT"] = str(BDE_INPUT)
env["DOPO_BDE_RESULTS"] = str(BDE_RESULTS)


# ============================================================
# Run standalone BDE model
#
# IMPORTANT:
# The main DOPO+EP BDE-enriched dataset is frozen.
# Running the standalone BDE model must NOT automatically
# overwrite data/DOPO_EP_new_with_BDE.csv.
# ============================================================
cmd = [
    sys.executable,
    str(BDE_SCRIPT),
    "--no-update-main-bde",
]

print("=" * 78)
print("[BDE] Standalone model evaluation")
print("=" * 78)
print(f"[INFO] Input   : {BDE_INPUT}")
print(f"[INFO] Results : {BDE_RESULTS}")
print("[INFO] Main BDE dataset update: DISABLED")
print("[RUN]", " ".join(cmd))

subprocess.run(
    cmd,
    check=True,
    cwd=str(ROOT),
    env=env,
)