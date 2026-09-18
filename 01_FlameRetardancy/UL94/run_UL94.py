# -*- coding: utf-8 -*-
"""Run the UL-94 V-0 development-stage repeated-holdout model.

This script keeps UL-94 output organization consistent with the regression
development tasks while using classification metrics (Macro-F1, Accuracy,
Balanced Accuracy, ROC-AUC where available).

Important:
- feature view / K / candidate model settings come from the frozen task config;
- this entry point is for development diagnostics and reproducibility;
- paper-ready performance must come from the strict 5x5 nested grouped
  validation in 08_ScientificValidation, not from the best development seed
  or a historical five-seed mean.
"""

import argparse
import json
import os
import sys
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "common" / "pipeline_core.py"

CONFIG = json.loads((ROOT / "config" / "task_config.json").read_text(encoding="utf-8"))
TASK_CFG = CONFIG["tasks"]["UL94_V0"]
SEEDS = list(CONFIG["evaluation"]["seeds"])
CONFIGURED_UL94_VIEW = str(TASK_CFG["view"])
CONFIGURED_UL94_K = str(TASK_CFG["k"])
CONFIGURED_UL94_MODELS = str(TASK_CFG["model"])
LOCAL_DIR = Path(__file__).resolve().parent
BASE_OUTDIR = ROOT / "results" / "main" / "01_FlameRetardancy" / "UL94_V0" / "without_BDE"
CURRENT_USE_BDE = False



RESULT_FILE_CANDIDATES = [
    "UL94_V0_view_search_results.csv",
    "UL94_view_search_results.csv",
    "UL94_V0_result_summary.csv",
]

METRIC_ALIASES = {
    "cv_Accuracy": ["cv_Accuracy", "cv_accuracy", "cv_acc", "CV_Accuracy", "CV_ACC"],
    "cv_Macro_F1": ["cv_Macro_F1", "cv_macro_f1", "CV_Macro_F1", "cv_f1_macro"],
    "cv_Weighted_F1": ["cv_Weighted_F1", "cv_weighted_f1", "CV_Weighted_F1", "cv_f1_weighted"],
    "test_Accuracy": ["test_Accuracy", "test_accuracy", "test_acc", "Test_Accuracy", "Test_ACC"],
    "test_Macro_F1": ["test_Macro_F1", "test_macro_f1", "Test_Macro_F1", "test_f1_macro"],
    "test_Weighted_F1": ["test_Weighted_F1", "test_weighted_f1", "Test_Weighted_F1", "test_f1_weighted"],
    "balanced_score": ["balanced_score", "Balanced_score", "balancedScore", "selection_score"],
    "best_threshold": ["best_threshold", "Best_threshold", "threshold", "best_thr"],
}

GROUP_ALIASES = {
    "view": ["view", "feature_view", "fingerprint", "fp", "features"],
    "k": ["k", "K", "select_k", "SelectK", "k_value", "K_value"],
    "model": ["model", "Model", "model_name", "estimator"],
}


def _first_existing_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    lowered = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        found = lowered.get(c.lower())
        if found is not None:
            return found
    return None


def _find_result_file(seed_dir):
    for name in RESULT_FILE_CANDIDATES:
        path = seed_dir / name
        if path.exists():
            return path

    patterns = [
        "*UL94*view*search*results*.csv",
        "*UL94*result*summary*.csv",
        "*UL94*.csv",
    ]
    matches = []
    for pat in patterns:
        matches.extend(seed_dir.rglob(pat))
    matches = sorted(set(matches), key=lambda p: (len(str(p)), str(p)))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Missing UL94 result CSV under {seed_dir}. Expected one of: "
        + ", ".join(RESULT_FILE_CANDIDATES)
    )


def _normalize_ul94_results(
    df: pd.DataFrame,
    *,
    default_view: str,
    default_k: str,
    default_model: str = "unknown",
) -> pd.DataFrame:
    """Normalize pipeline_core classification output to LOI-style summary input.

    UL94 is a classification task, so it should not contain R2/RMSE/MAE.
    The comparable columns are Accuracy, Macro_F1, Weighted_F1,
    balanced_score, and best_threshold when available.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    for canonical, aliases in GROUP_ALIASES.items():
        src = _first_existing_column(df, aliases)
        if src is not None and src != canonical:
            df[canonical] = df[src]

    if "view" not in df.columns:
        df["view"] = default_view
    if "k" not in df.columns:
        df["k"] = default_k
    if "model" not in df.columns:
        df["model"] = default_model

    for canonical, aliases in METRIC_ALIASES.items():
        src = _first_existing_column(df, aliases)
        if src is not None and src != canonical:
            df[canonical] = df[src]

    # If pipeline_core did not save balanced_score, build a stable classification
    # selection score so that UL94 can be ranked like LOI/PHRR/THR summaries.
    if "balanced_score" not in df.columns:
        if {"test_Macro_F1", "test_Accuracy"}.issubset(df.columns):
            df["balanced_score"] = (
                pd.to_numeric(df["test_Macro_F1"], errors="coerce")
                + pd.to_numeric(df["test_Accuracy"], errors="coerce")
            ) / 2.0
        elif "test_Macro_F1" in df.columns:
            df["balanced_score"] = pd.to_numeric(df["test_Macro_F1"], errors="coerce")
        elif "test_Accuracy" in df.columns:
            df["balanced_score"] = pd.to_numeric(df["test_Accuracy"], errors="coerce")

    metric_cols = [c for c in METRIC_ALIASES if c in df.columns]
    for c in metric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _build_repeat_summary(all_df, group_cols):
    metric_cols = [c for c in METRIC_ALIASES if c in all_df.columns]
    metric_cols = [c for c in metric_cols if all_df[c].notna().any()]
    if not metric_cols:
        raise ValueError(
            "No UL94 classification metric columns were found. Please check the "
            "pipeline_core.py output CSV columns."
        )

    agg = all_df.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).rstrip("_")
        if isinstance(col, tuple) else str(col)
        for col in agg.columns
    ]

    if "seed" in all_df.columns:
        n_seeds = all_df.groupby(group_cols, dropna=False)["seed"].nunique().reset_index(name="n_seeds")
        agg = agg.merge(n_seeds, on=group_cols, how="left")

    sort_cols = [
        "balanced_score_mean",
        "test_Macro_F1_mean",
        "test_Accuracy_mean",
        "cv_Macro_F1_mean",
        "cv_Accuracy_mean",
    ]
    sort_cols = [c for c in sort_cols if c in agg.columns]
    if sort_cols:
        agg = agg.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)

    # Put identifiers first, then n_seeds, then metrics.
    front = [c for c in group_cols + ["n_seeds"] if c in agg.columns]
    others = [c for c in agg.columns if c not in front]
    return agg[front + others]


def _write_outputs(
    all_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    outdir: Path,
    local_dir: Path,
    prefix: str,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    all_path = outdir / f"{prefix}_all_seed_results.csv"
    summary_path = outdir / f"{prefix}_repeat_summary.csv"
    all_df.to_csv(all_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("=" * 78)
    print("[SAVED] UL94 all-seed results:")
    print(all_path)
    print("[SAVED] UL94 repeat summary:")
    print(summary_path)
    print("=" * 78)
    print(summary_df.head(30).to_string(index=False))



def run_seed(seed: int) -> None:
    env = os.environ.copy()

    env["DOPO_TASKS"] = "UL94_V0"
    # Use the same external repeated-seed logic as the UL94 view/K scripts.
    env["DOPO_REPEAT_EVAL"] = "0"
    env["DOPO_REPEAT_SEEDS"] = str(seed)
    env["DOPO_DESCRIPTOR_MODE"] = "advanced"
    env["DOPO_USE_V7_FEATURES"] = "1"

    # Main UL94 model remains no-BDE unless ablation proves a stable gain.
    env["DOPO_USE_BDE_FEATURES"] = "1" if CURRENT_USE_BDE else "0"

    env["DOPO_UL94_RANDOM_STATE"] = str(seed)
    env["DOPO_UL94_VIEWS"] = CONFIGURED_UL94_VIEW
    env["DOPO_UL94_K_LIST"] = CONFIGURED_UL94_K
    env["DOPO_UL94_MODELS"] = CONFIGURED_UL94_MODELS

    bde_data = ROOT / "data" / "DOPO_EP_new_with_BDE.csv"
    base_data = ROOT / "data" / "DOPO_EP_new.csv"
    env["DOPO_INPUT_PATH"] = str(bde_data if bde_data.exists() else base_data)

    outdir = BASE_OUTDIR / f"seed_{seed}"
    env["DOPO_RESULTS"] = str(outdir)

    print("=" * 78)
    print(f"[UL94_V0] seed={seed}, view={CONFIGURED_UL94_VIEW}, k={CONFIGURED_UL94_K}, BDE={CURRENT_USE_BDE}")
    print(f"[INFO] DATA={env['DOPO_INPUT_PATH']}")
    print(f"[INFO] OUTDIR={outdir}")
    print("=" * 78)

    subprocess.run([sys.executable, str(CORE)], check=True, env=env)


def collect_results() -> pd.DataFrame:
    rows = []

    for seed in SEEDS:
        seed_dir = BASE_OUTDIR / f"seed_{seed}"
        path = _find_result_file(seed_dir)
        df = pd.read_csv(path)
        df = _normalize_ul94_results(df, default_view=CONFIGURED_UL94_VIEW, default_k=CONFIGURED_UL94_K)
        # pipeline_core already writes a task column. Replace metadata safely
        # instead of inserting duplicate column names during collection.
        for col in ("task", "seed", "source_file"):
            if col in df.columns:
                df = df.drop(columns=col)
        df.insert(0, "task", "UL94_V0")
        df.insert(1, "seed", seed)
        df.insert(2, "source_file", str(path))
        rows.append(df)

    all_df = pd.concat(rows, ignore_index=True)
    summary_df = _build_repeat_summary(all_df, ["view", "k", "model"])

    _write_outputs(
        all_df,
        summary_df,
        outdir=BASE_OUTDIR,
        local_dir=LOCAL_DIR,
        prefix="UL94_development",
    )
    return summary_df


def main() -> None:
    global BASE_OUTDIR, CURRENT_USE_BDE
    parser = argparse.ArgumentParser(description="Run UL94_V0 with the unified project rules.")
    parser.add_argument("--mode", choices=["main", "bde_ablation"], default="main")
    parser.add_argument("--bde", choices=["main", "without", "with"], default="main")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    CURRENT_USE_BDE = bool(TASK_CFG["main_bde"]) if args.bde == "main" else args.bde == "with"
    label = "with_BDE" if CURRENT_USE_BDE else "without_BDE"
    if args.mode == "main":
        BASE_OUTDIR = ROOT / "results" / "main" / "01_FlameRetardancy" / "UL94_V0" / label
    else:
        BASE_OUTDIR = ROOT / "results" / "bde_ablation" / "UL94_V0" / label

    if not args.collect_only:
        for seed in SEEDS:
            seed_dir = BASE_OUTDIR / f"seed_{seed}"
            if args.skip_existing:
                try:
                    _find_result_file(seed_dir)
                    print(f"[SKIP] existing result found for seed={seed}: {seed_dir}")
                    continue
                except FileNotFoundError:
                    pass
            run_seed(seed)
    collect_results()


if __name__ == "__main__":
    main()
