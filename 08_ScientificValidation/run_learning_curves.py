# -*- coding: utf-8 -*-
"""Generate group-aware learning curves for selected DOPO+EP tasks.

For each repeat, molecule groups are split once into development and held-out
sets. Increasing fractions of the development groups are then sampled while the
held-out groups remain fixed. This estimates whether additional independent
molecules are likely to improve performance.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

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


def parse_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_fractions(text: str) -> list[float]:
    values = sorted({float(item) for item in parse_csv(text)})
    if not values or values[0] <= 0 or values[-1] > 1:
        raise ValueError("Fractions must be in (0, 1].")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "DOPO_EP_new_with_BDE.csv")
    parser.add_argument("--tasks", default="LOI,PHRR,THR,UL94_V0,Tg,TS_MPa")
    parser.add_argument("--fractions", default="0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--model", default="ExtraTrees")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "scientific_validation" / "learning_curves")
    parser.add_argument(
        "--row-policy",
        choices=["baseline_inclusive", "modified_only"],
        default="baseline_inclusive",
        help=(
            "Row policy for absolute-property learning curves. FINAL manuscript curves "
            "must use baseline_inclusive to match the frozen primary models."
        ),
    )
    return parser.parse_args()


def _probability(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    scores = np.asarray(model.decision_function(X), dtype=float)
    return 1.0 / (1.0 + np.exp(-scores))


def _sample_training_groups(
    train_groups: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
) -> set[str]:
    unique = np.unique(train_groups.astype(str))
    n_select = max(2, int(np.ceil(len(unique) * fraction)))
    n_select = min(n_select, len(unique))
    return set(rng.choice(unique, size=n_select, replace=False).tolist())


def main() -> None:
    args = parse_args()
    fractions = parse_fractions(args.fractions)
    args.results.mkdir(parents=True, exist_ok=True)
    bundle = prepare_scientific_bundle(args.input)
    rows: list[dict[str, object]] = []

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

        for repeat in range(args.repeats):
            seed = args.random_state + repeat * 10
            splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=seed)
            train_idx, test_idx = next(splitter.split(X, y, groups))
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]
            train_groups = groups.iloc[train_idx].to_numpy()

            if config.task_type == "classification" and y_test.nunique() < 2:
                raise RuntimeError(
                    f"{task} repeat {repeat + 1}: held-out set has one class. Change random-state."
                )

            for fraction in fractions:
                rng = np.random.default_rng(seed + int(fraction * 1000))
                selected_groups: set[str] | None = None
                selected_idx: np.ndarray | None = None
                for _ in range(100):
                    candidate_groups = _sample_training_groups(train_groups, fraction, rng)
                    mask = groups.iloc[train_idx].isin(candidate_groups).to_numpy()
                    candidate_idx = train_idx[mask]
                    if len(candidate_idx) < 5:
                        continue
                    if config.task_type == "classification" and y.iloc[candidate_idx].nunique() < 2:
                        continue
                    selected_groups = candidate_groups
                    selected_idx = candidate_idx
                    break
                if selected_idx is None or selected_groups is None:
                    raise RuntimeError(
                        f"Could not create a valid subset for {task}, repeat={repeat + 1}, fraction={fraction}."
                    )

                pipeline = build_pipeline(
                    task_type=config.task_type,
                    model_name=args.model,
                    requested_k=config.current_k,
                    seed=seed + int(fraction * 100),
                )
                pipeline.fit(X.iloc[selected_idx], y.iloc[selected_idx])
                pred = pipeline.predict(X_test)
                base_row: dict[str, object] = {
                    "task": task,
                    "task_type": config.task_type,
                    "repeat": repeat + 1,
                    "seed": seed,
                    "train_fraction": fraction,
                    "n_train_rows": len(selected_idx),
                    "n_train_groups": len(selected_groups),
                    "n_test_rows": len(test_idx),
                    "n_test_groups": groups.iloc[test_idx].nunique(),
                    "model": args.model,
                    "view": config.current_view,
                    "configured_k": config.current_k,
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
                        "Baseline-inclusive absolute-property learning curve: valid Loading_total_FR >= 0; "
                        "neat-EP FR-derived features masked; task-matching EP_matrix feature masked "
                        "on neat-EP rows"
                        if task in FORMAL_ABSOLUTE_TASKS and args.row_policy == "baseline_inclusive"
                        else (
                            "Loading_total_FR > 0 for absolute-property tasks; "
                            "Neat EP excluded from this sensitivity learning curve"
                            if task in FORMAL_ABSOLUTE_TASKS
                            else "task-specific valid target mask"
                        )
                    ),
                }
                if config.task_type == "regression":
                    base_row.update(regression_metrics(y_test, pred))
                else:
                    base_row.update(classification_metrics(y_test, pred, _probability(pipeline, X_test)))
                rows.append(base_row)

    all_results = pd.DataFrame(rows)
    all_results.to_csv(args.results / "learning_curve_all_results.csv", index=False, encoding="utf-8-sig")
    metric_columns = [
        c for c in ["R2", "RMSE", "MAE", "Accuracy", "Macro_F1", "ROC_AUC", "Brier"]
        if c in all_results.columns
    ]
    agg_spec = {metric: ["mean", "std"] for metric in metric_columns}
    summary = all_results.groupby(["task", "task_type", "train_fraction"], as_index=False).agg(agg_spec)
    summary.columns = [
        "_".join([str(part) for part in col if str(part)]) if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]
    size_summary = all_results.groupby(["task", "train_fraction"], as_index=False).agg(
        n_train_rows_mean=("n_train_rows", "mean"),
        n_train_groups_mean=("n_train_groups", "mean"),
        n_valid_formal_rows=("n_valid_formal_rows", "first"),
        n_neat_rows=("n_neat_rows", "first"),
        n_modified_rows=("n_modified_rows", "first"),
        row_policy=("row_policy", "first"),
        formal_row_policy=("formal_row_policy", "first"),
        neat_fr_feature_masking=("neat_fr_feature_masking", "first"),
        matching_baseline_leakage_guard=("matching_baseline_leakage_guard", "first"),
    )
    summary = summary.merge(size_summary, on=["task", "train_fraction"], how="left")
    summary.to_csv(args.results / "learning_curve_summary.csv", index=False, encoding="utf-8-sig")
    (args.results / "run_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print("\n[LEARNING CURVE SUMMARY]")
    print(summary.to_string(index=False))
    print(f"\n[DONE] {args.results}")


if __name__ == "__main__":
    main()
