# -*- coding: utf-8 -*-
"""Run the three paper validation scenarios: molecule, scaffold and reference."""
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
    parser.add_argument("--tasks", default="LOI,PHRR,THR,UL94_V0")
    parser.add_argument("--selection-scope", choices=["fixed", "curated", "full"], default="fixed")
    parser.add_argument("--row-policy", choices=["modified_only", "baseline_inclusive"], default="baseline_inclusive")
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "scientific_validation" / "FINAL_grouping_sensitivity_5x5")
    parser.add_argument("--models", default="")
    args = parser.parse_args()

    command = [
        sys.executable, str(SCRIPT_DIR / "run_scientific_evaluation.py"),
        "--tasks", args.tasks,
        "--results", str(args.results),
        "--split-strategies", "molecule,scaffold,reference",
        "--screening-modes", "formulation",
        "--bde-modes", "without",
        "--selection-scope", args.selection_scope,
        "--row-policy", args.row_policy,
        "--outer-splits", str(args.outer_splits),
        "--inner-splits", str(args.inner_splits),
    ]
    if args.models:
        command.extend(["--models", args.models])
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
