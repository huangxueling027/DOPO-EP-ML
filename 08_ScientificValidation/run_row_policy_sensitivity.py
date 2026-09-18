# -*- coding: utf-8 -*-
"""Compare corrected Model A and Model B row policies without result overwrites.

The baseline-inclusive model is the FINAL primary manuscript evaluation for
absolute properties. A separate modified-only model is retained only as a
row-policy sensitivity analysis. The FINAL baseline-inclusive run also reports
metrics for the modified-formulation test subset within the same outer folds.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", default="LOI,PHRR,THR,UL94_V0")
    p.add_argument("--selection-scope", choices=["fixed", "curated", "full"], default="fixed")
    p.add_argument("--outer-splits", type=int, default=5)
    p.add_argument("--inner-splits", type=int, default=5)
    p.add_argument("--results", type=Path, default=ROOT / "results" / "scientific_validation" / "FINAL_row_policy_sensitivity_fixed_5x5")
    p.add_argument("--models", default="")
    return p.parse_args()


def run_policy(args: argparse.Namespace, policy: str) -> Path:
    out = args.results / policy
    command = [
        sys.executable, "-u", str(SCRIPT_DIR / "run_scientific_evaluation.py"),
        "--tasks", args.tasks,
        "--results", str(out),
        "--split-strategies", "molecule",
        "--screening-modes", "formulation",
        "--bde-modes", "without",
        "--row-policy", policy,
        "--selection-scope", args.selection_scope,
        "--outer-splits", str(args.outer_splits),
        "--inner-splits", str(args.inner_splits),
    ]
    if args.models.strip():
        command.extend(["--models", args.models])
    print("[RUN]", " ".join(command))
    subprocess.run(command, cwd=str(ROOT), check=True)
    return out / "scientific_nested_summary_all.csv"


def comparison_table(model_a: pd.DataFrame, model_b: pd.DataFrame) -> pd.DataFrame:
    rows = []
    tasks = list(dict.fromkeys(model_a.get("task", pd.Series(dtype=str)).astype(str).tolist()))
    for task in tasks:
        a = model_a.loc[model_a["task"].astype(str) == task]
        b = model_b.loc[model_b["task"].astype(str) == task]
        if a.empty or b.empty:
            continue
        a = a.iloc[0]
        b = b.iloc[0]
        is_cls = str(a.get("task_type", "")) == "classification"
        metric = "Macro_F1" if is_cls else "R2"
        main_col = f"outer_{metric}_mean"
        modified_col = f"outer_modified_{metric}_mean"
        a_score = pd.to_numeric(pd.Series([a.get(main_col)]), errors="coerce").iloc[0]
        b_overall = pd.to_numeric(pd.Series([b.get(main_col)]), errors="coerce").iloc[0]
        b_modified = pd.to_numeric(pd.Series([b.get(modified_col)]), errors="coerce").iloc[0]
        rows.append({
            "task": task,
            "primary_metric": metric,
            "Modified_only_sensitivity_model": a_score,
            "FINAL_baseline_inclusive_overall": b_overall,
            "FINAL_modified_test_subset": b_modified,
            "Delta_FINAL_subset_minus_modified_only_model": (b_modified - a_score) if np.isfinite(a_score) and np.isfinite(b_modified) else np.nan,
            "Modified_only_n_rows": a.get("n_valid_formal_rows"),
            "FINAL_n_rows": b.get("n_valid_formal_rows"),
            "FINAL_n_neat": b.get("n_neat_rows"),
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    a_path = run_policy(args, "modified_only")
    b_path = run_policy(args, "baseline_inclusive")
    model_a = pd.read_csv(a_path)
    model_b = pd.read_csv(b_path)
    combined = pd.concat([model_a, model_b], ignore_index=True)
    combined.to_csv(args.results / "row_policy_sensitivity_all_summaries.csv", index=False, encoding="utf-8-sig")
    comparison = comparison_table(model_a, model_b)
    comparison.to_csv(args.results / "row_policy_sensitivity_comparison.csv", index=False, encoding="utf-8-sig")
    print("\n[ROW POLICY SENSITIVITY]")
    print(comparison.to_string(index=False))
    print(f"\n[DONE] {args.results}")


if __name__ == "__main__":
    main()
