# -*- coding: utf-8 -*-
"""Run outer-fold SHAP stability analysis for validated tasks."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        default="LOI,PHRR,THR,UL94_V0",
        help="Comma-separated validated tasks. Default: the four core flame-retardancy tasks.",
    )
    parser.add_argument("--split-strategies", default="molecule")
    parser.add_argument("--selection-scope", choices=["fixed", "curated", "full"], default="fixed")
    parser.add_argument("--row-policy", choices=["modified_only", "baseline_inclusive"], default="baseline_inclusive")
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--max-shap-samples", type=int, default=60)
    parser.add_argument("--models", default="", help="Optional fixed model list, e.g. ExtraTrees or XGB")
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "results" / "05_Shap" / "FINAL_core_fixed_baseline_inclusive_5x5",
    )
    parser.add_argument(
        "--parent-results",
        type=Path,
        default=ROOT / "results" / "05_Shap" / "FINAL_core_features",
        help="Output directory for stable parent-variable tables.",
    )
    parser.add_argument(
        "--skip-parent-tables",
        action="store_true",
        help="Skip the High/Medium stable parent-variable aggregation step.",
    )
    args = parser.parse_args()

    command = [
        sys.executable, str(ROOT / "08_ScientificValidation" / "run_scientific_evaluation.py"),
        "--tasks", args.tasks,
        "--results", str(args.results),
        "--split-strategies", args.split_strategies,
        "--screening-modes", "formulation",
        "--bde-modes", "without",
        "--selection-scope", args.selection_scope,
        "--row-policy", args.row_policy,
        "--outer-splits", str(args.outer_splits),
        "--inner-splits", str(args.inner_splits),
        "--shap-stability",
        "--max-shap-samples", str(args.max_shap_samples),
    ]
    if args.models.strip():
        command.extend(["--models", args.models])
    subprocess.run(command, check=True, cwd=str(ROOT))

    figure_dir = args.results / "figures"
    if not args.no_plots:
        subprocess.run([
            sys.executable, str(ROOT / "05_Shap" / "plot_shap_stability.py"),
            "--results", str(args.results),
            "--tasks", args.tasks,
            "--top-n", str(args.top_n),
            "--output", str(figure_dir),
            "--dpi", str(args.dpi),
            "--formats", args.formats,
        ], check=True, cwd=str(ROOT))

    if not args.skip_parent_tables:
        subprocess.run([
            sys.executable, str(ROOT / "05_Shap" / "build_core_stable_feature_table.py"),
            "--results-root", str(args.results),
            "--figures-dir", str(figure_dir),
            "--output-dir", str(args.parent_results),
            "--tasks", args.tasks,
            "--include-levels", "High,Medium",
            "--top-parent-n", "12",
            "--output-prefix", "core",
            "--step-label", "STEP2",
            "--summary-title", "核心任务SHAP稳定父变量汇总",
        ], check=True, cwd=str(ROOT))


if __name__ == "__main__":
    main()
