# -*- coding: utf-8 -*-
"""Nested-CV TabPFN challenge comparison on the current fixed view/K settings.

TabPFN is optional and never replaces the classic model pool automatically.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="Char_yield,FS_MPa,TS_MPa,Tg,LOI,PHRR,THR,UL94_V0")
    parser.add_argument("--split-strategies", default="molecule")
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "scientific_validation" / "TabPFN_nested_comparison")
    parser.add_argument("--classic-models", default="ExtraTrees,XGB,LGBM,SVR,SVC,SoftVote")
    args = parser.parse_args()

    models = args.classic_models + ",TabPFN"
    subprocess.run([
        sys.executable, str(SCRIPT_DIR / "run_scientific_evaluation.py"),
        "--tasks", args.tasks,
        "--results", str(args.results),
        "--split-strategies", args.split_strategies,
        "--screening-modes", "formulation",
        "--bde-modes", "without",
        "--selection-scope", "fixed",
        "--outer-splits", str(args.outer_splits),
        "--inner-splits", str(args.inner_splits),
        "--models", models,
        "--include-tabpfn",
    ], check=True)


if __name__ == "__main__":
    main()
