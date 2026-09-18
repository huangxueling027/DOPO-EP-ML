# -*- coding: utf-8 -*-
"""Run strict information-source ablation for frozen DOPO+EP task settings.

Three information-source scenarios are compared under identical molecule groups and nested folds:
1. conditions_only: formulation/element/curing/test-condition features only;
2. molecular_only: SMILES-derived main/co/curing fingerprints/descriptors only;
3. all: all standard formulation+molecular features without BDE.\n\nBDE is tested separately in the paired BDE ablation, avoiding confounding two ablations.

The molecular feature bundle is built once, which avoids four expensive repeated
fingerprint-generation passes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.scientific_evaluation import (
    EvaluationOptions,
    TASK_CONFIGS,
    evaluate_task_nested,
    prepare_scientific_bundle,
)


def parse_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="LOI,PHRR,THR,UL94_V0")
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--selection-scope", choices=["fixed", "curated", "full"], default="fixed")
    parser.add_argument("--row-policy", choices=["modified_only", "baseline_inclusive"], default="baseline_inclusive")
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "results" / "scientific_validation" / "FINAL_information_source_ablation_5x5",
    )
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "DOPO_EP_new_with_BDE.csv")
    parser.add_argument("--models", default="", help="Optional focused model list, e.g. ExtraTrees")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = parse_csv(args.tasks)
    unknown = [task for task in tasks if task not in TASK_CONFIGS]
    if unknown:
        raise KeyError(f"Unknown tasks: {unknown}")
    args.results.mkdir(parents=True, exist_ok=True)
    bundle = prepare_scientific_bundle(args.input)

    audit_dir = args.results / "data_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    bundle.df.to_csv(audit_dir / "data_standardized_with_audit.csv", index=False, encoding="utf-8-sig")

    scenarios = [
        ("conditions_only", "conditions_only", False),
        ("molecular_only", "molecular_only", False),
        ("all", "all", False),
    ]
    summaries: list[dict[str, object]] = []
    model_names = tuple(parse_csv(args.models)) or None

    for scenario_name, feature_scope, use_bde in scenarios:
        scenario_dir = args.results / scenario_name
        options = EvaluationOptions(
            outer_splits=args.outer_splits,
            inner_splits=args.inner_splits,
            random_state=42,
            split_strategy="molecule",
            selection_scope=args.selection_scope,
            screening_mode="formulation",
            row_policy=args.row_policy,
            feature_scope=feature_scope,
            use_bde=use_bde,
            model_names=model_names,
        )
        for task in tasks:
            print("=" * 88)
            print(f"[SOURCE ABLATION] scenario={scenario_name} task={task}")
            _, _, summary = evaluate_task_nested(bundle, task, scenario_dir, options)
            summary = {"ablation_scenario": scenario_name, **summary}
            summaries.append(summary)

    combined = pd.DataFrame(summaries)
    combined.to_csv(
        args.results / "information_source_ablation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metric_candidates = [
        "outer_R2_mean",
        "outer_RMSE_mean",
        "outer_MAE_mean",
        "outer_Accuracy_mean",
        "outer_Macro_F1_mean",
        "outer_ROC_AUC_mean",
        "outer_Brier_mean",
    ]
    columns = [
        c for c in [
            "task", "ablation_scenario", "feature_scope", "use_BDE",
            "outer_splits_completed", *metric_candidates,
        ] if c in combined.columns
    ]
    compact = combined[columns].copy()
    compact.to_csv(
        args.results / "information_source_ablation_compact.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.results / "run_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print("\n[INFORMATION SOURCE ABLATION]")
    print(compact.to_string(index=False))
    print(f"\n[DONE] {args.results}")


if __name__ == "__main__":
    main()
