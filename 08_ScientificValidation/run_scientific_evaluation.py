# -*- coding: utf-8 -*-
"""Run leakage-controlled nested evaluation for DOPO+EP property models."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.scientific_evaluation import EvaluationOptions, TASK_CONFIGS, run_nested_project


def parse_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "DOPO_EP_new_with_BDE.csv")
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "scientific_validation" / "FINAL_all_fixed_baseline_inclusive_5x5")
    parser.add_argument(
        "--tasks",
        default="LOI,PHRR,THR,Tg,Char_yield,TS_MPa,FS_MPa,UL94_V0",
        help=f"Comma-separated tasks or ALL. Available: {','.join(TASK_CONFIGS)}",
    )
    parser.add_argument("--split-strategies", default="molecule", help="molecule,scaffold,reference,random")
    parser.add_argument("--screening-modes", default="formulation", help="formulation,molecular")
    parser.add_argument(
        "--feature-scopes",
        default="all",
        help="all,conditions_only,molecular_only; used for information-source ablation",
    )
    parser.add_argument("--bde-modes", default="without", help="without,with,both")
    parser.add_argument(
        "--row-policy",
        choices=["modified_only", "baseline_inclusive"],
        default="baseline_inclusive",
        help=(
            "baseline_inclusive: FINAL absolute-property policy; retain neat EP with FR-feature masking and matching-baseline leakage guard; "
            "modified_only: sensitivity/development-only positive-loading evaluation"
        ),
    )
    parser.add_argument("--selection-scope", choices=["fixed", "curated", "full"], default="fixed")
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--conformal-alpha", type=float, default=0.05)
    parser.add_argument("--models", default="", help="Optional comma-separated model names for a focused run")
    parser.add_argument("--include-tabpfn", action="store_true")
    parser.add_argument("--shap-stability", action="store_true")
    parser.add_argument("--max-shap-samples", type=int, default=80)
    return parser.parse_args()


def resolve_tasks(text: str) -> list[str]:
    if text.strip().upper() == "ALL":
        return list(TASK_CONFIGS)
    tasks = parse_csv(text)
    unknown = [task for task in tasks if task not in TASK_CONFIGS]
    if unknown:
        raise KeyError(f"Unknown tasks: {unknown}")
    return tasks


def resolve_bde_modes(text: str) -> list[bool]:
    value = text.strip().lower()
    if value == "without":
        return [False]
    if value == "with":
        return [True]
    if value == "both":
        return [False, True]
    raise ValueError("--bde-modes must be without, with or both")


def main() -> None:
    args = parse_args()
    options = EvaluationOptions(
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        random_state=args.random_state,
        selection_scope=args.selection_scope,
        row_policy=args.row_policy,
        conformal_alpha=args.conformal_alpha,
        include_tabpfn=args.include_tabpfn,
        model_names=tuple(parse_csv(args.models)) or None,
        shap_stability=args.shap_stability,
        max_shap_samples=args.max_shap_samples,
    )
    summary = run_nested_project(
        input_path=args.input,
        output_dir=args.results,
        tasks=resolve_tasks(args.tasks),
        split_strategies=parse_csv(args.split_strategies),
        screening_modes=parse_csv(args.screening_modes),
        feature_scopes=parse_csv(args.feature_scopes),
        bde_modes=resolve_bde_modes(args.bde_modes),
        options=options,
    )
    print("\n[SCIENTIFIC NESTED SUMMARY]")
    print(summary.to_string(index=False))
    print(f"\n[DONE] Results: {args.results}")


if __name__ == "__main__":
    main()
