# -*- coding: utf-8 -*-
"""Rebuild the four frozen FINAL paper-ready 5x5 result groups.

This is intentionally expensive.  Do not run it merely to regenerate paper
figures when the frozen upstream results already exist and pass validation.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
EVALUATOR = SCRIPT_DIR / "run_scientific_evaluation.py"


def run(tasks: str, bde: str, result_dir: str, *, outer: int, inner: int, dry_run: bool) -> None:
    command = [
        sys.executable, "-u", str(EVALUATOR),
        "--input", str(ROOT / "data" / "DOPO_EP_new_with_BDE.csv"),
        "--results", str(ROOT / "results" / "scientific_validation" / result_dir),
        "--tasks", tasks,
        "--split-strategies", "molecule",
        "--screening-modes", "formulation",
        "--bde-modes", bde,
        "--selection-scope", "fixed",
        "--row-policy", "baseline_inclusive",
        "--outer-splits", str(outer),
        "--inner-splits", str(inner),
    ]
    print("\n[RUN]", " ".join(command))
    if not dry_run:
        subprocess.run(command, cwd=str(ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer", type=int, default=5)
    parser.add_argument("--inner", type=int, default=5)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the four canonical commands without executing them.",
    )
    args = parser.parse_args()

    jobs = [
        ("LOI,PHRR,THR,UL94_V0", "without", "FINAL_core_fixed_baseline_inclusive_5x5"),
        ("Tg,TS_MPa", "without", "FINAL_aux_fixed_baseline_inclusive_5x5"),
        ("Char_yield,FS_MPa", "without", "FINAL_exploratory_fixed_baseline_inclusive_5x5"),
        ("Delta_LOI,Delta_PHRR,Delta_THR,Delta_CY", "without", "FINAL_Delta_fixed_5x5"),
    ]
    for tasks, bde, result_dir in jobs:
        run(
            tasks,
            bde,
            result_dir,
            outer=args.outer,
            inner=args.inner,
            dry_run=args.dry_run,
        )

    print(
        "\n[DONE] Canonical strict result commands printed."
        if args.dry_run
        else "\n[DONE] Canonical strict result groups rebuilt."
    )


if __name__ == "__main__":
    main()
