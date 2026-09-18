# -*- coding: utf-8 -*-
"""Run leakage-controlled dummy baselines and Y-scrambling sanity tests.

This is a diagnostic rather than a replacement for the nested model results.
The same frozen feature view/K, molecule groups and outer folds are used for the
observed-label and scrambled-label models. By default a single robust
ExtraTrees model is used so the null test remains computationally manageable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import pipeline_core as core
from common.literature_feature_views import get_groups, get_target
from common.scientific_evaluation import (
    FORMAL_ABSOLUTE_TASKS,
    MATCHING_EP_BASELINE,
    TASK_CONFIGS,
    _apply_baseline_inclusive_mask,
    _filter_task_matrix,
    build_pipeline,
    classification_metrics,
    prepare_scientific_bundle,
    regression_metrics,
)
from common.split_strategies import iter_cv_folds


def parse_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "DOPO_EP_new_with_BDE.csv")
    parser.add_argument("--tasks", default="LOI,PHRR,THR,UL94_V0")
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "scientific_validation" / "null_tests")
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--model", default="ExtraTrees")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--row-policy",
        choices=["baseline_inclusive", "modified_only"],
        default="baseline_inclusive",
        help=(
            "Row policy for absolute-property diagnostics. FINAL manuscript diagnostics "
            "must use baseline_inclusive so they match the frozen FINAL primary models."
        ),
    )
    parser.add_argument(
        "--progress-every", type=int, default=100,
        help="Print Y-scrambling progress every N permutations (0 disables progress prints).",
    )
    return parser.parse_args()


def _predict_probability(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    scores = np.asarray(model.decision_function(X), dtype=float)
    return 1.0 / (1.0 + np.exp(-scores))


def main() -> None:
    args = parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    bundle = prepare_scientific_bundle(args.input)
    fold_rows: list[dict[str, object]] = []
    permutation_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for task in parse_csv(args.tasks):
        if task not in TASK_CONFIGS:
            raise KeyError(f"Unknown task: {task}")
        config = TASK_CONFIGS[task]
        y_all = get_target(bundle.base, task)
        valid = core.build_task_valid_mask(
            bundle.df, bundle.base.colmap, task, target=y_all
        )
        if task in FORMAL_ABSOLUTE_TASKS:
            loading_column = bundle.base.colmap.get(
                "Loading_total_FR wt%", "Loading_total_FR wt%"
            )
            if loading_column not in bundle.df.columns:
                raise KeyError(
                    f"Missing loading column required by formal row policy: {loading_column}"
                )
            loading_all = pd.to_numeric(bundle.df[loading_column], errors="coerce")
            if args.row_policy == "modified_only":
                valid = valid & loading_all.gt(0)
            else:
                # FINAL primary protocol: retain confirmed neat EP (0) and modified (>0)
                # rows. Missing loading is not silently treated as neat EP.
                valid = valid & loading_all.ge(0)

        y = y_all.loc[valid].reset_index(drop=True)
        df_task = bundle.df.loc[valid].reset_index(drop=True)
        groups = get_groups(bundle.base, task).loc[valid].astype(str).reset_index(drop=True)
        X = _filter_task_matrix(
            bundle,
            task,
            config.current_view,
            use_bde=False,
            screening_mode="formulation",
            feature_scope="all",
        ).loc[valid].reset_index(drop=True)
        # Apply the exact same row-level neat-EP leakage guard used by the
        # frozen FINAL baseline-inclusive nested workflow.
        X = _apply_baseline_inclusive_mask(
            X, df_task, bundle.base.colmap, task, args.row_policy
        )

        if task in FORMAL_ABSOLUTE_TASKS:
            loading_column = bundle.base.colmap.get(
                "Loading_total_FR wt%", "Loading_total_FR wt%"
            )
            loading_task = pd.to_numeric(df_task[loading_column], errors="coerce")
            n_neat = int(loading_task.eq(0.0).sum())
            n_modified = int(loading_task.gt(0.0).sum())
        else:
            n_neat = 0
            n_modified = int(valid.sum())
        print(
            f"[PROTOCOL] task={task} row_policy={args.row_policy} "
            f"n_valid={int(valid.sum())} neat={n_neat} modified={n_modified} "
            f"view={config.current_view} K={config.current_k} model={args.model}",
            flush=True,
        )
        folds = list(
            iter_cv_folds(
                X,
                y,
                groups=groups,
                n_splits=args.outer_splits,
                random_state=args.random_state,
                classification=config.task_type == "classification",
            )
        )
        observed_primary: list[float] = []
        dummy_primary: list[float] = []
        null_primary_by_permutation = {
            permutation + 1: []
            for permutation in range(args.permutations)
        }

        for fold in folds:
            train_idx, test_idx = fold.train_idx, fold.test_idx
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            pipeline = build_pipeline(
                task_type=config.task_type,
                model_name=args.model,
                requested_k=config.current_k,
                seed=args.random_state + fold.fold_id,
            )
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            if config.task_type == "regression":
                obs = regression_metrics(y_test, y_pred)
                dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
                dummy_metrics = regression_metrics(y_test, dummy.predict(X_test))
                observed_value = obs["R2"]
                dummy_value = dummy_metrics["R2"]
                fold_rows.append({
                    "task": task,
                    "outer_fold": fold.fold_id,
                    "model": args.model,
                    "observed_R2": obs["R2"],
                    "observed_RMSE": obs["RMSE"],
                    "observed_MAE": obs["MAE"],
                    "dummy_R2": dummy_metrics["R2"],
                    "dummy_RMSE": dummy_metrics["RMSE"],
                    "dummy_MAE": dummy_metrics["MAE"],
                })
            else:
                probability = _predict_probability(pipeline, X_test)
                obs = classification_metrics(y_test, y_pred, probability)
                dummy = DummyClassifier(strategy="prior", random_state=args.random_state).fit(X_train, y_train)
                dummy_pred = dummy.predict(X_test)
                dummy_prob = dummy.predict_proba(X_test)[:, 1]
                dummy_metrics = classification_metrics(y_test, dummy_pred, dummy_prob)
                observed_value = obs["Macro_F1"]
                dummy_value = dummy_metrics["Macro_F1"]
                fold_rows.append({
                    "task": task,
                    "outer_fold": fold.fold_id,
                    "model": args.model,
                    "observed_Accuracy": obs["Accuracy"],
                    "observed_Macro_F1": obs["Macro_F1"],
                    "observed_ROC_AUC": obs["ROC_AUC"],
                    "dummy_Accuracy": dummy_metrics["Accuracy"],
                    "dummy_Macro_F1": dummy_metrics["Macro_F1"],
                    "dummy_ROC_AUC": dummy_metrics["ROC_AUC"],
                })

            observed_primary.append(float(observed_value))
            dummy_primary.append(float(dummy_value))

            for permutation in range(args.permutations):
                rng = np.random.default_rng(
                    args.random_state + 10000 * fold.fold_id + permutation
                )
                y_permuted = pd.Series(rng.permutation(y_train.to_numpy()), index=y_train.index)
                permuted_pipeline = build_pipeline(
                    task_type=config.task_type,
                    model_name=args.model,
                    requested_k=config.current_k,
                    seed=args.random_state + fold.fold_id + permutation + 1000,
                )
                permuted_pipeline.fit(X_train, y_permuted)
                perm_pred = permuted_pipeline.predict(X_test)
                if config.task_type == "regression":
                    perm_metrics = regression_metrics(y_test, perm_pred)
                    primary = perm_metrics["R2"]
                    row = {
                        "task": task,
                        "outer_fold": fold.fold_id,
                        "permutation": permutation + 1,
                        "R2": perm_metrics["R2"],
                        "RMSE": perm_metrics["RMSE"],
                        "MAE": perm_metrics["MAE"],
                    }
                else:
                    perm_prob = _predict_probability(permuted_pipeline, X_test)
                    perm_metrics = classification_metrics(y_test, perm_pred, perm_prob)
                    primary = perm_metrics["Macro_F1"]
                    row = {
                        "task": task,
                        "outer_fold": fold.fold_id,
                        "permutation": permutation + 1,
                        "Accuracy": perm_metrics["Accuracy"],
                        "Macro_F1": perm_metrics["Macro_F1"],
                        "ROC_AUC": perm_metrics["ROC_AUC"],
                    }
                permutation_rows.append(row)
                null_primary_by_permutation[permutation + 1].append(
                    float(primary)
                )
                if args.progress_every > 0 and (
                    permutation == 0
                    or (permutation + 1) % args.progress_every == 0
                    or (permutation + 1) == args.permutations
                ):
                    print(
                        f"[NULL] task={task} fold={fold.fold_id}/{len(folds)} "
                        f"permutation={permutation + 1}/{args.permutations}",
                        flush=True,
                    )

        observed_mean = float(np.mean(observed_primary))
        dummy_mean = float(np.mean(dummy_primary))
        # One null value represents one complete permutation replicate,
        # aggregated across all outer folds.
        null_arr = np.asarray([
            np.mean(values)
            for _, values in sorted(
                null_primary_by_permutation.items()
            )
            if values
        ], dtype=float)
        empirical_p = float((1 + np.sum(null_arr >= observed_mean)) / (1 + len(null_arr)))
        summary_rows.append({
            "task": task,
            "task_type": config.task_type,
            "model": args.model,
            "primary_metric": "R2" if config.task_type == "regression" else "Macro_F1",
            "observed_mean": observed_mean,
            "observed_std": float(np.std(observed_primary, ddof=1)),
            "dummy_mean": dummy_mean,
            "null_mean": float(np.mean(null_arr)),
            "null_std": float(np.std(null_arr, ddof=1)),
            "null_95_percentile": float(np.quantile(null_arr, 0.95)),
            "empirical_p_value": empirical_p,
            "n_outer_folds": len(folds),
            "n_permutations_per_fold": args.permutations,
            "n_independent_permutation_replicates": int(len(null_arr)),
            "p_value_unit": "complete permutation replicate aggregated across outer folds",
            "row_policy": args.row_policy,
            "n_valid_formal_rows": int(valid.sum()),
            "n_neat_rows": n_neat,
            "n_modified_rows": n_modified,
            "neat_fr_feature_masking": bool(
                task in FORMAL_ABSOLUTE_TASKS and args.row_policy == "baseline_inclusive"
            ),
            "matching_baseline_leakage_guard": bool(
                task in MATCHING_EP_BASELINE and args.row_policy == "baseline_inclusive"
            ),
            "formal_row_policy": (
                "Baseline-inclusive absolute-property diagnostic: valid Loading_total_FR >= 0; "
                "neat-EP FR-derived features masked; task-matching EP_matrix feature masked "
                "on neat-EP rows"
                if task in FORMAL_ABSOLUTE_TASKS and args.row_policy == "baseline_inclusive"
                else (
                    "Loading_total_FR > 0 for absolute-property tasks; "
                    "Neat EP excluded from this sensitivity diagnostic"
                    if task in FORMAL_ABSOLUTE_TASKS
                    else "task-specific valid target mask"
                )
            ),
        })

        # Task-level checkpoint: the audit checks task completeness, so partial
        # files cannot be mistaken for a finished 4-task FINAL null test.
        pd.DataFrame(fold_rows).to_csv(
            args.results / "null_test_observed_and_dummy_by_fold.csv",
            index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(permutation_rows).to_csv(
            args.results / "y_scrambling_all_results.csv",
            index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(summary_rows).to_csv(
            args.results / "null_test_summary.csv",
            index=False, encoding="utf-8-sig"
        )
        print(f"[CHECKPOINT] completed task={task}; partial outputs saved", flush=True)

    pd.DataFrame(fold_rows).to_csv(args.results / "null_test_observed_and_dummy_by_fold.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(permutation_rows).to_csv(args.results / "y_scrambling_all_results.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.results / "null_test_summary.csv", index=False, encoding="utf-8-sig")
    (args.results / "run_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print("\n[NULL TEST SUMMARY]")
    print(summary.to_string(index=False))
    print(f"\n[DONE] {args.results}")


if __name__ == "__main__":
    main()
