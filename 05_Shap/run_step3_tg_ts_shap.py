#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Complete STEP 3: Tg and TS SHAP stability, plotting and parent-variable tables.

Run from the project root:

    python -u 05_Shap/run_step3_tg_ts_shap.py

By default this script:
1. Runs FINAL fixed baseline-inclusive 5x5 nested evaluation with SHAP stability for Tg and TS_MPa.
2. Converts the generated SHAP CSV files into publication-ready figures.
3. Merges transformed child features into interpretable parent variables.

Use ``--skip-evaluation`` when the Tg/TS SHAP CSV files already exist and only
figures/tables need to be regenerated.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("\n" + "=" * 96)
    print("[RUN]", " ".join(command))
    print("=" * 96)
    subprocess.run(command, check=True, cwd=str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="Tg,TS_MPa")
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--max-shap-samples", type=int, default=60)
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--top-parent-n", type=int, default=12)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--evaluation-results",
        type=Path,
        default=ROOT / "results" / "05_Shap" / "FINAL_aux_fixed_baseline_inclusive_5x5",
    )
    parser.add_argument(
        "--figure-results",
        type=Path,
        default=ROOT / "results" / "05_Shap" / "FINAL_aux_fixed_baseline_inclusive_5x5/figures",
    )
    parser.add_argument(
        "--parent-results",
        type=Path,
        default=ROOT / "results" / "05_Shap" / "FINAL_aux_features",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evaluator = ROOT / "08_ScientificValidation" / "run_scientific_evaluation.py"
    plotter = ROOT / "05_Shap" / "plot_shap_stability.py"
    parent_builder = ROOT / "05_Shap" / "build_core_stable_feature_table.py"

    for required in [plotter, parent_builder]:
        if not required.exists():
            raise FileNotFoundError(f"Required file not found: {required}")

    if not args.skip_evaluation:
        if not evaluator.exists():
            raise FileNotFoundError(f"Scientific evaluation entry not found: {evaluator}")
        run([
            sys.executable,
            "-u",
            str(evaluator),
            "--tasks",
            args.tasks,
            "--results",
            str(args.evaluation_results),
            "--split-strategies",
            "molecule",
            "--screening-modes",
            "formulation",
            "--bde-modes",
            "without",
            "--selection-scope",
            "fixed",
            "--row-policy",
            "baseline_inclusive",
            "--outer-splits",
            str(args.outer_splits),
            "--inner-splits",
            str(args.inner_splits),
            "--shap-stability",
            "--max-shap-samples",
            str(args.max_shap_samples),
        ])

    if not args.skip_plots:
        run([
            sys.executable,
            "-u",
            str(plotter),
            "--results",
            str(args.evaluation_results),
            "--tasks",
            args.tasks,
            "--top-n",
            str(args.top_n),
            "--output",
            str(args.figure_results),
            "--dpi",
            "600",
            "--formats",
            "png,pdf",
        ])

    run([
        sys.executable,
        "-u",
        str(parent_builder),
        "--results-root",
        str(args.evaluation_results),
        "--figures-dir",
        str(args.figure_results),
        "--output-dir",
        str(args.parent_results),
        "--tasks",
        args.tasks,
        "--include-levels",
        "High,Medium",
        "--top-parent-n",
        str(args.top_parent_n),
        "--output-prefix",
        "auxiliary",
        "--step-label",
        "STEP3",
        "--summary-title",
        "Tg与TS辅助任务SHAP稳定父变量汇总",
    ])

    print("\n[DONE] STEP 3 completed.")
    print("[DONE] SHAP raw results:", args.evaluation_results)
    print("[DONE] Figures:", args.figure_results)
    print("[DONE] Parent-variable tables:", args.parent_results)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] A subprocess failed with return code {exc.returncode}", file=sys.stderr)
        raise
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
