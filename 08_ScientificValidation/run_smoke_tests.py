# -*- coding: utf-8 -*-
"""Quick end-to-end checks for regression and UL-94 classification."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results" / "scientific_validation" / "_smoke_tests"


def run(task: str, model: str) -> None:
    subprocess.run([
        sys.executable, str(SCRIPT_DIR / "run_scientific_evaluation.py"),
        "--tasks", task,
        "--results", str(RESULTS / task),
        "--split-strategies", "molecule",
        "--screening-modes", "formulation",
        "--bde-modes", "without",
        "--selection-scope", "fixed",
        "--row-policy", "baseline_inclusive",
        "--outer-splits", "2",
        "--inner-splits", "2",
        "--models", model,
    ], check=True)


def main() -> None:
    if RESULTS.exists():
        shutil.rmtree(RESULTS)
    run("THR", "Ridge")
    run("UL94_V0", "GBDT")
    print(f"[PASS] Smoke-test results: {RESULTS}")


if __name__ == "__main__":
    main()
