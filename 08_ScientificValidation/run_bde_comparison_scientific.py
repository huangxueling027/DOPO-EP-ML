# -*- coding: utf-8 -*-
"""Paper-ready with-BDE / without-BDE nested ablation."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="LOI,PHRR,THR,UL94_V0,Tg,TS_MPa")
    parser.add_argument("--split-strategies", default="molecule")
    parser.add_argument("--screening-modes", default="formulation")
    parser.add_argument("--selection-scope", choices=["fixed", "curated", "full"], default="fixed")
    parser.add_argument("--row-policy", choices=["modified_only", "baseline_inclusive"], default="baseline_inclusive")
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "scientific_validation" / "FINAL_BDE_paired_ablation_5x5")
    parser.add_argument("--models", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = [
        sys.executable, str(SCRIPT_DIR / "run_scientific_evaluation.py"),
        "--input", str(ROOT / "data" / "DOPO_EP_new_with_BDE.csv"),
        "--results", str(args.results),
        "--tasks", args.tasks,
        "--split-strategies", args.split_strategies,
        "--screening-modes", args.screening_modes,
        "--bde-modes", "both",
        "--selection-scope", args.selection_scope,
        "--row-policy", args.row_policy,
        "--outer-splits", str(args.outer_splits),
        "--inner-splits", str(args.inner_splits),
    ]
    if args.models:
        command.extend(["--models", args.models])
    subprocess.run(command, check=True)

    summary_path = args.results / "scientific_nested_summary_all.csv"
    if not summary_path.exists():
        return
    df = pd.read_csv(summary_path)
    value_cols = [
        column for column in df.columns
        if column.endswith("_mean") and column.startswith("outer_")
    ]
    key_cols = ["task", "split_strategy", "screening_mode"]
    without = df.loc[df["use_BDE"] == 0, key_cols + value_cols].copy()
    with_bde = df.loc[df["use_BDE"] == 1, key_cols + value_cols].copy()
    merged = without.merge(with_bde, on=key_cols, suffixes=("_without_BDE", "_with_BDE"))
    for metric in value_cols:
        merged[f"delta_{metric}_with_minus_without"] = (
            merged[f"{metric}_with_BDE"] - merged[f"{metric}_without_BDE"]
        )
    merged.to_csv(args.results / "BDE_ablation_comparison.csv", index=False, encoding="utf-8-sig")
    print(merged.to_string(index=False))


if __name__ == "__main__":
    main()
