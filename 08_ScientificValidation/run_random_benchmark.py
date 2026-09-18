# -*- coding: utf-8 -*-
"""Repeated 80/20 random-split benchmark for the corrected DOPO+EP V5 project.

Purpose
-------
This script is a SECONDARY interpolation benchmark. It does not replace the
primary molecule-grouped nested evaluation. For each requested seed it:

1. applies the same formal row policy as V5 (Loading_total_FR > 0 for absolute
   property tasks),
2. makes one independent 80/20 random train/test split,
3. selects feature view + K + model using inner CV on the training portion only,
4. selects/calibrates the UL-94 threshold using training data only,
5. evaluates the untouched 20% test portion once,
6. reports mean +/- SD across seeds.

Default seeds: 42,52,62,72,82.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import pipeline_core as core
from common.literature_feature_views import get_target
from common.split_strategies import build_group_labels
from common.scientific_evaluation import (
    DEFAULT_CLASSIFICATION_MODELS,
    DEFAULT_REGRESSION_MODELS,
    FORMAL_ABSOLUTE_TASKS,
    TASK_CONFIGS,
    _apply_platt,
    _candidate_space,
    _deduplicate_requested_k,
    _filter_task_matrix,
    _fit_platt_calibrator,
    _minimum_inner_available_features,
    _probability_oof,
    _inner_splits,
    _selected_feature_info,
    build_pipeline,
    classification_metrics,
    evaluate_candidate_inner,
    optimise_threshold,
    prepare_scientific_bundle,
    regression_metrics,
)


def parse_csv(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def parse_int_csv(text: str) -> list[int]:
    values = [int(item) for item in parse_csv(text)]
    if not values:
        raise ValueError("At least one seed is required.")
    if len(values) != len(set(values)):
        raise ValueError("Seeds must be unique.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "DOPO_EP_new_with_BDE.csv",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "results" / "scientific_validation" / "V5_random_benchmark",
    )
    parser.add_argument(
        "--tasks",
        default="LOI,PHRR,THR,UL94_V0",
        help="Comma-separated tasks. Default: LOI,PHRR,THR,UL94_V0",
    )
    parser.add_argument(
        "--screening-modes",
        default="formulation",
        help="formulation or formulation,molecular. Default: formulation",
    )
    parser.add_argument(
        "--seeds",
        default="42,52,62,72,82",
        help="Independent random-split seeds.",
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument(
        "--selection-scope",
        choices=["fixed", "curated", "full"],
        default="fixed",
    )
    parser.add_argument(
        "--models",
        default="",
        help="Optional comma-separated model names for a focused diagnostic run.",
    )
    parser.add_argument(
        "--grouped-summary",
        type=Path,
        default=ROOT
        / "results"
        / "scientific_validation"
        / "FINAL_core_fixed_baseline_inclusive_5x5"
        / "scientific_nested_summary_all.csv",
        help="Optional V5 molecule-grouped summary used to make a comparison table.",
    )
    return parser.parse_args()


def _resolve_tasks(text: str) -> list[str]:
    tasks = parse_csv(text)
    unknown = [task for task in tasks if task not in TASK_CONFIGS]
    if unknown:
        raise KeyError(f"Unknown tasks: {unknown}")
    return tasks


def _formal_valid_mask(bundle, task: str, y: pd.Series) -> pd.Series:
    valid = core.build_task_valid_mask(bundle.df, bundle.base.colmap, task, target=y)
    if task in FORMAL_ABSOLUTE_TASKS:
        loading_col = bundle.base.colmap.get(
            "Loading_total_FR wt%", "Loading_total_FR wt%"
        )
        if loading_col not in bundle.df.columns:
            raise KeyError(
                "Missing loading column required by the formal V5 row policy: "
                f"{loading_col}"
            )
        loading = pd.to_numeric(bundle.df[loading_col], errors="coerce")
        valid = valid & loading.gt(0)
    return valid


def _safe_json_frequency(values: list[Any]) -> str:
    normalized = ["None" if value is None or pd.isna(value) else str(value) for value in values]
    return json.dumps(Counter(normalized).most_common(), ensure_ascii=False)


def _outer_split_indices(
    y: pd.Series,
    *,
    task_type: str,
    seed: int,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    stratify = y if task_type == "classification" else None
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return np.asarray(train_idx), np.asarray(test_idx)


def _evaluate_one_task_mode(
    *,
    bundle,
    task: str,
    screening_mode: str,
    seeds: list[int],
    test_size: float,
    inner_splits: int,
    selection_scope: str,
    model_names_override: list[str] | None,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    config = TASK_CONFIGS[task]
    y = get_target(bundle.base, task)
    valid = _formal_valid_mask(bundle, task, y)
    y_task = y.loc[valid].reset_index(drop=True)
    df_task = bundle.df.loc[valid].reset_index(drop=True)

    views, requested_ks = _candidate_space(config, bundle, selection_scope)
    matrices = {
        view: _filter_task_matrix(
            bundle,
            task,
            view,
            use_bde=False,
            screening_mode=screening_mode,
            feature_scope="all",
        )
        .loc[valid]
        .reset_index(drop=True)
        for view in views
    }
    if not matrices:
        raise ValueError(f"No feature views available for task={task}")

    if model_names_override:
        model_names = model_names_override
    else:
        model_names = list(
            DEFAULT_REGRESSION_MODELS
            if config.task_type == "regression"
            else DEFAULT_CLASSIFICATION_MODELS
        )

    seed_rows: list[dict[str, Any]] = []
    pred_tables: list[pd.DataFrame] = []
    candidate_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []

    reference_col = bundle.base.colmap.get("Reference", "Reference")
    meta_cols = [
        "Record_ID",
        bundle.base.colmap.get("FR_main"),
        bundle.base.colmap.get("FR_co"),
        bundle.base.colmap.get("SMILES_main"),
        bundle.base.colmap.get("SMILES_co"),
        reference_col,
        "Canonical_SMILES_main",
        "Murcko_scaffold_main",
    ]
    meta_cols = [c for c in meta_cols if c and c in df_task.columns]
    meta_cols = list(dict.fromkeys(meta_cols))

    for seed in seeds:
        print("=" * 88)
        print(
            f"[RANDOM BENCHMARK] task={task} mode={screening_mode} "
            f"seed={seed} test_size={test_size:.2f}"
        )
        train_idx, test_idx = _outer_split_indices(
            y_task,
            task_type=config.task_type,
            seed=seed,
            test_size=test_size,
        )
        y_train = y_task.iloc[train_idx].reset_index(drop=True)
        y_test = y_task.iloc[test_idx].reset_index(drop=True)

        if config.task_type == "classification":
            if y_train.nunique() < 2 or y_test.nunique() < 2:
                raise RuntimeError(
                    f"seed={seed}: classification split contains a single class."
                )

        if "Record_ID" in df_task.columns:
            for subset, indices in (("train", train_idx), ("test", test_idx)):
                for rid in df_task.iloc[indices]["Record_ID"].astype(str):
                    split_rows.append(
                        {
                            "task": task,
                            "screening_mode": screening_mode,
                            "seed": seed,
                            "Record_ID": rid,
                            "subset": subset,
                        }
                    )

        candidates = []
        for view, X_all in matrices.items():
            X_train = X_all.iloc[train_idx].reset_index(drop=True)
            min_available = _minimum_inner_available_features(
                X_train,
                y_train,
                None,
                inner_splits=inner_splits,
                seed=seed,
                classification=config.task_type == "classification",
            )
            ks = _deduplicate_requested_k(requested_ks, min_available)
            for requested_k in ks:
                for model_name in model_names:
                    result = evaluate_candidate_inner(
                        X_train=X_train,
                        y_train=y_train,
                        groups_train=None,
                        task_type=config.task_type,
                        view=view,
                        requested_k=requested_k,
                        model_name=model_name,
                        inner_splits=inner_splits,
                        seed=seed,
                        include_tabpfn=False,
                    )
                    candidates.append(result)
                    candidate_rows.append(
                        {
                            "task": task,
                            "screening_mode": screening_mode,
                            "seed": seed,
                            "view": view,
                            "requested_k": requested_k,
                            "model": model_name,
                            "inner_primary": result.inner_primary,
                            "inner_secondary": result.inner_secondary,
                            "inner_threshold": result.threshold,
                            "error": result.error,
                        }
                    )

        valid_candidates = [
            candidate for candidate in candidates if np.isfinite(candidate.inner_primary)
        ]
        if not valid_candidates:
            raise RuntimeError(f"No valid inner-CV candidate for task={task}, seed={seed}")

        selected = max(
            valid_candidates,
            key=lambda candidate: (candidate.inner_primary, candidate.inner_secondary),
        )
        X_all = matrices[selected.view]
        X_train = X_all.iloc[train_idx].reset_index(drop=True)
        X_test = X_all.iloc[test_idx].reset_index(drop=True)

        fitted = build_pipeline(
            task_type=config.task_type,
            model_name=selected.model_name,
            requested_k=selected.requested_k,
            seed=seed,
            include_tabpfn=False,
        ).fit(X_train, y_train)
        n_after_variance, effective_k = _selected_feature_info(fitted)

        # Quantify how much ordinary random splitting reuses molecular/reference
        # identities across train and test. This is intentionally diagnostic: a
        # high overlap explains why random-split interpolation is easier than the
        # primary unseen-molecule grouped evaluation.
        molecule_labels = build_group_labels(
            df_task, strategy="molecule", reference_col=reference_col
        )
        reference_labels = build_group_labels(
            df_task, strategy="reference", reference_col=reference_col
        )

        train_molecules = set(molecule_labels.iloc[train_idx].astype(str))
        test_molecule_series = molecule_labels.iloc[test_idx].astype(str)
        test_molecules = set(test_molecule_series)
        molecule_overlap = train_molecules & test_molecules
        seen_molecule_fraction = float(test_molecule_series.isin(train_molecules).mean())

        train_references = set(reference_labels.iloc[train_idx].astype(str))
        test_reference_series = reference_labels.iloc[test_idx].astype(str)
        test_references = set(test_reference_series)
        reference_overlap = train_references & test_references
        seen_reference_fraction = float(test_reference_series.isin(train_references).mean())

        row: dict[str, Any] = {
            "task": task,
            "task_type": config.task_type,
            "screening_mode": screening_mode,
            "seed": seed,
            "split_strategy": "repeated_random_80_20",
            "test_size": test_size,
            "selection_scope": selection_scope,
            "inner_splits": inner_splits,
            "n_valid_formal_rows": int(valid.sum()),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_train_molecule_groups": int(len(train_molecules)),
            "n_test_molecule_groups": int(len(test_molecules)),
            "molecule_group_overlap_count": int(len(molecule_overlap)),
            "test_rows_with_seen_molecule_fraction": seen_molecule_fraction,
            "n_train_reference_groups": int(len(train_references)),
            "n_test_reference_groups": int(len(test_references)),
            "reference_group_overlap_count": int(len(reference_overlap)),
            "test_rows_with_seen_reference_fraction": seen_reference_fraction,
            "selected_view": selected.view,
            "selected_requested_k": selected.requested_k,
            "selected_effective_k": int(effective_k),
            "n_features_after_variance": int(n_after_variance),
            "selected_model": selected.model_name,
            "inner_primary": float(selected.inner_primary),
            "inner_secondary": float(selected.inner_secondary),
        }

        pred = df_task.iloc[test_idx].loc[:, meta_cols].reset_index(drop=True).copy()
        pred.insert(0, "task", task)
        pred.insert(1, "screening_mode", screening_mode)
        pred.insert(2, "seed", seed)
        pred["y_true"] = y_test.to_numpy()

        if config.task_type == "regression":
            y_pred = np.asarray(fitted.predict(X_test), dtype=float)
            row.update(regression_metrics(y_test, y_pred, prefix="test_"))
            pred["y_pred"] = y_pred
        else:
            raw_probability = np.asarray(fitted.predict_proba(X_test)[:, 1], dtype=float)

            if selected.oof_probability is None:
                inner_cv = _inner_splits(
                    X_train,
                    y_train,
                    None,
                    n_splits=inner_splits,
                    seed=seed,
                    classification=True,
                )
                oof_probability = _probability_oof(
                    build_pipeline(
                        task_type="classification",
                        model_name=selected.model_name,
                        requested_k=selected.requested_k,
                        seed=seed,
                        include_tabpfn=False,
                    ),
                    X_train,
                    y_train,
                    inner_cv,
                )
            else:
                oof_probability = selected.oof_probability

            calibrator = _fit_platt_calibrator(y_train, oof_probability)
            calibrated_oof = _apply_platt(calibrator, oof_probability)
            threshold, _, _ = optimise_threshold(y_train, calibrated_oof)
            probability = _apply_platt(calibrator, raw_probability)
            y_pred = (probability >= threshold).astype(int)

            row["selected_threshold"] = float(threshold)
            row.update(
                classification_metrics(y_test, y_pred, probability, prefix="test_")
            )
            pred["raw_V0_probability"] = raw_probability
            pred["calibrated_V0_probability"] = probability
            pred["threshold"] = threshold
            pred["y_pred"] = y_pred

        print(
            f"[SELECT] view={selected.view} K={selected.requested_k} "
            f"model={selected.model_name}"
        )
        if config.task_type == "regression":
            print(
                f"[TEST] R2={row['test_R2']:.4f} RMSE={row['test_RMSE']:.4f} "
                f"MAE={row['test_MAE']:.4f}"
            )
        else:
            print(
                f"[TEST] Macro-F1={row['test_Macro_F1']:.4f} "
                f"Accuracy={row['test_Accuracy']:.4f} "
                f"ROC-AUC={row['test_ROC_AUC']:.4f}"
            )

        seed_rows.append(row)
        pred_tables.append(pred)

    seed_df = pd.DataFrame(seed_rows)
    predictions_df = pd.concat(pred_tables, ignore_index=True) if pred_tables else pd.DataFrame()
    candidates_df = pd.DataFrame(candidate_rows)
    split_df = pd.DataFrame(split_rows)

    summary: dict[str, Any] = {
        "task": task,
        "task_type": config.task_type,
        "screening_mode": screening_mode,
        "benchmark_role": "secondary_interpolation_benchmark",
        "split_strategy": "repeated_random_80_20",
        "test_size": float(test_size),
        "seeds": json.dumps(seeds),
        "n_seeds_completed": int(len(seed_df)),
        "inner_splits": int(inner_splits),
        "selection_scope": selection_scope,
        "selection_rule": (
            "feature view, K and model selected by inner random CV within each "
            "80% training split; UL94 calibration and threshold use training data only; "
            "20% test split used once per seed"
        ),
        "formal_row_policy": (
            "Loading_total_FR > 0 for absolute-property tasks; Neat EP retained in "
            "the frozen database but excluded from this formal benchmark"
            if task in FORMAL_ABSOLUTE_TASKS
            else "task-specific valid target mask"
        ),
        "n_valid_formal_rows": int(valid.sum()),
        "selected_view_frequency": _safe_json_frequency(seed_df["selected_view"].tolist()),
        "selected_model_frequency": _safe_json_frequency(seed_df["selected_model"].tolist()),
        "selected_k_frequency": _safe_json_frequency(seed_df["selected_requested_k"].tolist()),
        "mean_test_rows_with_seen_molecule_fraction": float(
            pd.to_numeric(seed_df["test_rows_with_seen_molecule_fraction"], errors="coerce").mean()
        ),
        "mean_test_rows_with_seen_reference_fraction": float(
            pd.to_numeric(seed_df["test_rows_with_seen_reference_fraction"], errors="coerce").mean()
        ),
    }

    if config.task_type == "regression":
        for metric in ("R2", "RMSE", "MAE"):
            values = pd.to_numeric(seed_df[f"test_{metric}"], errors="coerce")
            summary[f"random_{metric}_mean"] = float(values.mean())
            summary[f"random_{metric}_std"] = float(values.std(ddof=1))
    else:
        for metric in (
            "Accuracy",
            "Balanced_Accuracy",
            "Macro_F1",
            "Weighted_F1",
            "Brier",
            "ECE",
            "ROC_AUC",
            "PR_AUC",
        ):
            values = pd.to_numeric(seed_df[f"test_{metric}"], errors="coerce")
            summary[f"random_{metric}_mean"] = float(values.mean())
            summary[f"random_{metric}_std"] = float(values.std(ddof=1))

    task_dir = output_dir / task / screening_mode
    task_dir.mkdir(parents=True, exist_ok=True)
    seed_df.to_csv(task_dir / f"{task}_random_seed_metrics.csv", index=False, encoding="utf-8-sig")
    predictions_df.to_csv(task_dir / f"{task}_random_test_predictions.csv", index=False, encoding="utf-8-sig")
    candidates_df.to_csv(task_dir / f"{task}_random_inner_candidates.csv", index=False, encoding="utf-8-sig")
    if not split_df.empty:
        split_df.to_csv(task_dir / f"{task}_random_split_manifest.csv", index=False, encoding="utf-8-sig")

    return seed_df, predictions_df, candidates_df, split_df, summary


def _make_grouped_comparison(
    random_summary: pd.DataFrame,
    grouped_summary_path: Path,
) -> pd.DataFrame:
    if not grouped_summary_path.exists():
        print(f"[INFO] Grouped summary not found; comparison skipped: {grouped_summary_path}")
        return pd.DataFrame()

    grouped = pd.read_csv(grouped_summary_path, encoding="utf-8-sig")
    required = {"task", "screening_mode", "split_strategy"}
    if not required.issubset(grouped.columns):
        print("[WARN] Existing grouped summary lacks required columns; comparison skipped.")
        return pd.DataFrame()

    grouped = grouped.loc[
        grouped["split_strategy"].astype(str).str.lower().eq("molecule")
        & grouped["screening_mode"].isin(random_summary["screening_mode"])
    ].copy()

    rows: list[dict[str, Any]] = []
    for _, random_row in random_summary.iterrows():
        task = random_row["task"]
        mode = random_row["screening_mode"]
        match = grouped.loc[(grouped["task"] == task) & (grouped["screening_mode"] == mode)]
        if match.empty:
            continue
        grouped_row = match.iloc[0]
        task_type = random_row["task_type"]
        if task_type == "regression":
            grouped_score = pd.to_numeric(pd.Series([grouped_row.get("outer_R2_mean")]), errors="coerce").iloc[0]
            random_score = random_row.get("random_R2_mean")
            score_name = "R2"
        else:
            grouped_score = pd.to_numeric(pd.Series([grouped_row.get("outer_Macro_F1_mean")]), errors="coerce").iloc[0]
            random_score = random_row.get("random_Macro_F1_mean")
            score_name = "Macro_F1"
        rows.append(
            {
                "task": task,
                "screening_mode": mode,
                "primary_metric": score_name,
                "molecule_grouped_nested_mean": grouped_score,
                "repeated_random_80_20_mean": random_score,
                "random_minus_grouped": (
                    float(random_score - grouped_score)
                    if pd.notna(random_score) and pd.notna(grouped_score)
                    else np.nan
                ),
                "interpretation": (
                    "Positive gap indicates easier interpolation under random splitting; "
                    "the molecule-grouped result remains the primary unseen-molecule estimate."
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if not 0.05 <= args.test_size <= 0.50:
        raise ValueError("--test-size must be between 0.05 and 0.50")
    if args.inner_splits < 2:
        raise ValueError("--inner-splits must be >= 2")

    tasks = _resolve_tasks(args.tasks)
    modes = parse_csv(args.screening_modes)
    invalid_modes = [mode for mode in modes if mode not in {"formulation", "molecular"}]
    if invalid_modes:
        raise ValueError(f"Invalid screening modes: {invalid_modes}")
    seeds = parse_int_csv(args.seeds)
    models = parse_csv(args.models) or None

    output_dir = args.results
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[LOAD] {args.input}")
    bundle = prepare_scientific_bundle(args.input)

    summaries: list[dict[str, Any]] = []
    for mode in modes:
        for task in tasks:
            _, _, _, _, summary = _evaluate_one_task_mode(
                bundle=bundle,
                task=task,
                screening_mode=mode,
                seeds=seeds,
                test_size=args.test_size,
                inner_splits=args.inner_splits,
                selection_scope=args.selection_scope,
                model_names_override=models,
                output_dir=output_dir,
            )
            summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_path = output_dir / "random_benchmark_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    comparison_df = _make_grouped_comparison(summary_df, args.grouped_summary)
    if not comparison_df.empty:
        comparison_df.to_csv(
            output_dir / "random_vs_molecule_comparison.csv",
            index=False,
            encoding="utf-8-sig",
        )

    config = {
        "benchmark_role": "secondary_interpolation_benchmark",
        "primary_evaluation_remains": "V5 molecule-grouped nested 5x5",
        "input": str(args.input),
        "results": str(args.results),
        "tasks": tasks,
        "screening_modes": modes,
        "seeds": seeds,
        "test_size": args.test_size,
        "inner_splits": args.inner_splits,
        "selection_scope": args.selection_scope,
        "models_override": models,
        "use_BDE": False,
        "formal_absolute_row_policy": "Loading_total_FR > 0",
        "grouped_summary_for_comparison": str(args.grouped_summary),
        "important_note": (
            "Do not replace the primary molecule-grouped nested result with this benchmark. "
            "Use it only to quantify the interpolation-vs-unseen-molecule generalization gap."
        ),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n[RANDOM BENCHMARK SUMMARY]")
    print(summary_df.to_string(index=False))
    if not comparison_df.empty:
        print("\n[RANDOM VS MOLECULE-GROUPED]")
        print(comparison_df.to_string(index=False))
    print(f"\n[DONE] Results: {output_dir}")


if __name__ == "__main__":
    main()
