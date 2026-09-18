#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate structural applicability domains using outer-fold predictions.

This script reconstructs each outer-fold training pool from the rows assigned to
all other outer folds. It then computes Morgan-fingerprint Tanimoto similarity,
exact-component novelty, and Murcko-scaffold novelty for every outer test row.

The script is intended for the FINAL fixed baseline-inclusive 5x5 scientific-validation/SHAP outputs of:
LOI, PHRR, THR and UL94_V0. No model is retrained.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    auc,
    r2_score,
    roc_auc_score,
)

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "RDKit is required. Install/activate the same environment used by the ML project. "
        f"Original error: {exc}"
    )

TASK_TYPE = {
    "LOI": "regression",
    "PHRR": "regression",
    "THR": "regression",
    "UL94_V0": "classification",
}


@dataclass(frozen=True)
class Config:
    results_root: str
    output: str
    tasks: list[str]
    radius: int
    nbits: int
    reliable_threshold: float
    caution_threshold: float
    bootstrap_iterations: int
    random_state: int
    formats: list[str]
    dpi: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Outer-fold applicability-domain validation")
    p.add_argument("--results-root", default="results/05_Shap/FINAL_core_fixed_baseline_inclusive_5x5")
    p.add_argument("--output", default="results/06_ApplicabilityDomain/FINAL_fixed_5x5")
    p.add_argument("--tasks", default="LOI,PHRR,THR,UL94_V0")
    p.add_argument("--radius", type=int, default=3)
    p.add_argument("--nbits", type=int, default=2048)
    p.add_argument("--reliable-threshold", type=float, default=0.70)
    p.add_argument("--caution-threshold", type=float, default=0.50)
    p.add_argument("--bootstrap-iterations", type=int, default=1000)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--formats", default="png,pdf")
    p.add_argument("--dpi", type=int, default=600)
    return p.parse_args()


def canonicalize(smiles: object) -> tuple[Optional[str], Optional[str]]:
    if smiles is None or (isinstance(smiles, float) and math.isnan(smiles)):
        return None, None
    text = str(smiles).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None, None
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None, None
    canonical = Chem.MolToSmiles(mol, canonical=True)
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        scaffold = ""
    return canonical, scaffold or ""


def find_prediction_file(results_root: Path, task: str) -> Path:
    """Locate the FINAL SHAP outer predictions only; never fall back to old runs."""
    expected = (
        results_root / task / "molecule" / "formulation" / "without_BDE"
        / f"{task}_outer_predictions.csv"
    )
    if not expected.exists():
        raise FileNotFoundError(
            f"FINAL outer predictions missing for {task}: {expected}. "
            "Run FINAL fixed baseline-inclusive SHAP first."
        )
    return expected


def prepare_predictions(path: Path, task: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"outer_fold", "SMILES_main", "y_true"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} lacks required columns: {sorted(missing)}")

    main_can, main_scaf, co_can, co_scaf = [], [], [], []
    for row in df.itertuples(index=False):
        existing_main = getattr(row, "Canonical_SMILES_main", None)
        existing_scaf = getattr(row, "Murcko_scaffold_main", None)
        can, scaf = canonicalize(existing_main if pd.notna(existing_main) else getattr(row, "SMILES_main"))
        if existing_scaf is not None and pd.notna(existing_scaf):
            scaf = str(existing_scaf)
        main_can.append(can)
        main_scaf.append(scaf)

        co_raw = getattr(row, "SMILES_co", None)
        ccan, cscaf = canonicalize(co_raw)
        co_can.append(ccan)
        co_scaf.append(cscaf)

    df = df.copy()
    df["AD_Canonical_SMILES_main"] = main_can
    df["AD_Murcko_scaffold_main"] = main_scaf
    df["AD_Canonical_SMILES_co"] = co_can
    df["AD_Murcko_scaffold_co"] = co_scaf
    df["AD_has_co"] = df["AD_Canonical_SMILES_co"].notna().astype(int)
    df["AD_pair_key"] = (
        df["AD_Canonical_SMILES_main"].fillna("<INVALID_MAIN>")
        + "||"
        + df["AD_Canonical_SMILES_co"].fillna("<NO_CO>")
    )
    df["task"] = task

    if TASK_TYPE[task] == "regression":
        if "y_pred" not in df.columns:
            raise ValueError(f"{path} lacks y_pred")
        df["absolute_error"] = (df["y_true"] - df["y_pred"]).abs()
        df["squared_error"] = (df["y_true"] - df["y_pred"]) ** 2
    else:
        if "y_pred" not in df.columns:
            raise ValueError(f"{path} lacks y_pred")
        prob_col = "calibrated_V0_probability" if "calibrated_V0_probability" in df.columns else "raw_V0_probability"
        if prob_col not in df.columns:
            raise ValueError(f"{path} lacks a V0 probability column")
        df["V0_probability_for_AD"] = pd.to_numeric(df[prob_col], errors="coerce")
        df["correct"] = (df["y_true"].astype(int) == df["y_pred"].astype(int)).astype(int)
    return df


def unique_valid(values: Iterable[object]) -> list[str]:
    return sorted({str(v) for v in values if v is not None and pd.notna(v) and str(v)})


def build_fp_map(smiles_values: Iterable[object], generator) -> dict[str, object]:
    out: dict[str, object] = {}
    for smi in unique_valid(smiles_values):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            out[smi] = generator.GetFingerprint(mol)
    return out


def max_similarity(query: Optional[str], pool: dict[str, object], generator) -> float:
    if query is None or not pool:
        return float("nan")
    mol = Chem.MolFromSmiles(query)
    if mol is None:
        return float("nan")
    qfp = generator.GetFingerprint(mol)
    sims = DataStructs.BulkTanimotoSimilarity(qfp, list(pool.values()))
    return float(max(sims)) if sims else float("nan")


def classify_zone(similarity: float, reliable: float, caution: float) -> str:
    if pd.isna(similarity):
        return "Invalid_or_missing"
    if similarity >= reliable:
        return "In_domain"
    if similarity >= caution:
        return "Caution"
    return "Extrapolation"


def calculate_fold_ad(
    df: pd.DataFrame,
    radius: int,
    nbits: int,
    reliable: float,
    caution: float,
) -> pd.DataFrame:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    outputs: list[pd.DataFrame] = []

    for fold in sorted(df["outer_fold"].dropna().unique()):
        train = df[df["outer_fold"] != fold].copy()
        test = df[df["outer_fold"] == fold].copy()

        main_pool = build_fp_map(train["AD_Canonical_SMILES_main"], generator)
        co_pool = build_fp_map(train["AD_Canonical_SMILES_co"], generator)
        any_pool = build_fp_map(
            pd.concat([train["AD_Canonical_SMILES_main"], train["AD_Canonical_SMILES_co"]], ignore_index=True),
            generator,
        )

        train_main_set = set(unique_valid(train["AD_Canonical_SMILES_main"]))
        train_co_set = set(unique_valid(train["AD_Canonical_SMILES_co"]))
        train_any_set = train_main_set | train_co_set
        train_pair_set = set(unique_valid(train["AD_pair_key"]))
        train_main_scaf = set(unique_valid(train["AD_Murcko_scaffold_main"]))
        train_co_scaf = set(unique_valid(train["AD_Murcko_scaffold_co"]))
        train_any_scaf = train_main_scaf | train_co_scaf

        records = []
        for row in test.itertuples(index=False):
            main = getattr(row, "AD_Canonical_SMILES_main")
            co = getattr(row, "AD_Canonical_SMILES_co")
            main_scaf = getattr(row, "AD_Murcko_scaffold_main")
            co_scaf = getattr(row, "AD_Murcko_scaffold_co")
            has_co = bool(getattr(row, "AD_has_co"))

            main_role = max_similarity(main, main_pool, generator)
            main_any = max_similarity(main, any_pool, generator)
            if has_co:
                co_role_pool = co_pool if co_pool else any_pool
                co_role = max_similarity(co, co_role_pool, generator)
                co_any = max_similarity(co, any_pool, generator)
                role_min = float(np.nanmin([main_role, co_role])) if not (pd.isna(main_role) and pd.isna(co_role)) else float("nan")
                any_min = float(np.nanmin([main_any, co_any])) if not (pd.isna(main_any) and pd.isna(co_any)) else float("nan")
                co_scaf_seen_role = bool(co_scaf in train_co_scaf) if co_scaf else False
                co_scaf_seen_any = bool(co_scaf in train_any_scaf) if co_scaf else False
            else:
                co_role = float("nan")
                co_any = float("nan")
                role_min = main_role
                any_min = main_any
                co_scaf_seen_role = True
                co_scaf_seen_any = True

            main_scaf_seen_role = bool(main_scaf in train_main_scaf) if main_scaf else False
            main_scaf_seen_any = bool(main_scaf in train_any_scaf) if main_scaf else False
            pair_key = getattr(row, "AD_pair_key")

            records.append(
                {
                    "main_role_max_similarity": main_role,
                    "main_any_component_max_similarity": main_any,
                    "co_role_max_similarity": co_role,
                    "co_any_component_max_similarity": co_any,
                    "AD_similarity": role_min,
                    "AD_similarity_any_component": any_min,
                    "main_exact_seen_role": int(main in train_main_set if main else False),
                    "main_exact_seen_any": int(main in train_any_set if main else False),
                    "co_exact_seen_role": int(co in train_co_set if has_co and co else True),
                    "co_exact_seen_any": int(co in train_any_set if has_co and co else True),
                    "pair_exact_seen": int(pair_key in train_pair_set),
                    "main_scaffold_seen_role": int(main_scaf_seen_role),
                    "main_scaffold_seen_any": int(main_scaf_seen_any),
                    "co_scaffold_seen_role": int(co_scaf_seen_role),
                    "co_scaffold_seen_any": int(co_scaf_seen_any),
                    "formulation_scaffold_seen_role": int(main_scaf_seen_role and co_scaf_seen_role),
                    "formulation_scaffold_seen_any": int(main_scaf_seen_any and co_scaf_seen_any),
                    "n_train_rows": int(len(train)),
                    "n_train_unique_main": int(len(main_pool)),
                    "n_train_unique_co": int(len(co_pool)),
                }
            )
        ad = pd.DataFrame(records, index=test.index)
        test = pd.concat([test, ad], axis=1)
        test["AD_zone"] = test["AD_similarity"].apply(lambda x: classify_zone(x, reliable, caution))
        test["scaffold_status"] = np.where(
            test["formulation_scaffold_seen_role"].eq(1), "Seen_scaffold", "New_scaffold"
        )
        outputs.append(test)

    result = pd.concat(outputs, ignore_index=True)
    return result.sort_values(["outer_fold"]).reset_index(drop=True)


def safe_r2(y_true, y_pred) -> float:
    if len(y_true) < 2 or np.nanstd(y_true) == 0:
        return float("nan")
    return float(r2_score(y_true, y_pred))


def safe_roc(y_true, probability) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, probability))


def safe_pr_auc(y_true, probability) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    precision, recall, _ = precision_recall_curve(y_true, probability)
    return float(auc(recall, precision))


def expected_calibration_error(y_true, probability, n_bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probability, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    if len(y) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.digitize(p, edges[1:-1], right=False)
    ece = 0.0
    for b in range(n_bins):
        mask = ids == b
        if not mask.any():
            continue
        ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def bootstrap_ci(values: np.ndarray, metric, n_iter: int, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values)
    if len(values) < 2 or n_iter <= 0:
        return float("nan"), float("nan")
    stats = []
    for _ in range(n_iter):
        sample = values[rng.integers(0, len(values), len(values))]
        try:
            stats.append(float(metric(sample)))
        except Exception:
            continue
    if not stats:
        return float("nan"), float("nan")
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def metrics_for_group(group: pd.DataFrame, task: str, n_boot: int, seed: int) -> dict:
    out: dict[str, object] = {"n": int(len(group))}
    rng = np.random.default_rng(seed)
    if len(group) == 0:
        return out
    if TASK_TYPE[task] == "regression":
        y = group["y_true"].to_numpy(float)
        pred = group["y_pred"].to_numpy(float)
        ae = np.abs(y - pred)
        out.update(
            {
                "R2": safe_r2(y, pred),
                "RMSE": float(mean_squared_error(y, pred) ** 0.5),
                "MAE": float(mean_absolute_error(y, pred)),
                "Median_AE": float(np.median(ae)),
            }
        )
        lo, hi = bootstrap_ci(ae, np.mean, n_boot, rng)
        out["MAE_bootstrap_CI_low"] = lo
        out["MAE_bootstrap_CI_high"] = hi
    else:
        y = group["y_true"].astype(int).to_numpy()
        pred = group["y_pred"].astype(int).to_numpy()
        prob = group["V0_probability_for_AD"].to_numpy(float)
        correct = (y == pred).astype(float)
        out.update(
            {
                "Accuracy": float(accuracy_score(y, pred)),
                "Balanced_Accuracy": float(balanced_accuracy_score(y, pred)),
                "Macro_F1": float(f1_score(y, pred, average="macro", zero_division=0)),
                "ROC_AUC": safe_roc(y, prob),
                "PR_AUC": safe_pr_auc(y, prob),
                "Brier": float(brier_score_loss(y, prob)),
                "ECE": expected_calibration_error(y, prob),
            }
        )
        lo, hi = bootstrap_ci(correct, np.mean, n_boot, rng)
        out["Accuracy_bootstrap_CI_low"] = lo
        out["Accuracy_bootstrap_CI_high"] = hi
    return out


def grouped_metrics(df: pd.DataFrame, task: str, group_col: str, n_boot: int, seed: int) -> pd.DataFrame:
    order = ["In_domain", "Caution", "Extrapolation", "Invalid_or_missing"] if group_col == "AD_zone" else sorted(df[group_col].dropna().unique())
    rows = []
    for i, label in enumerate(order):
        g = df[df[group_col] == label]
        if len(g) == 0:
            continue
        row = {"task": task, "grouping": group_col, "group": label}
        row.update(metrics_for_group(g, task, n_boot, seed + i))
        rows.append(row)
    return pd.DataFrame(rows)


def fold_metrics(df: pd.DataFrame, task: str, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for fold in sorted(df["outer_fold"].unique()):
        fold_df = df[df["outer_fold"] == fold]
        for zone in ["In_domain", "Caution", "Extrapolation", "Invalid_or_missing"]:
            g = fold_df[fold_df["AD_zone"] == zone]
            if len(g) == 0:
                continue
            row = {"task": task, "outer_fold": int(fold), "AD_zone": zone}
            row.update(metrics_for_group(g, task, 0, seed))
            rows.append(row)
    return pd.DataFrame(rows)


def threshold_sensitivity(df: pd.DataFrame, task: str, reliable_values: list[float], caution_values: list[float]) -> pd.DataFrame:
    rows = []
    for reliable in reliable_values:
        for caution in caution_values:
            if caution >= reliable:
                continue
            temp = df.copy()
            temp["zone_temp"] = temp["AD_similarity"].apply(lambda x: classify_zone(x, reliable, caution))
            metrics = grouped_metrics(temp.rename(columns={"zone_temp": "AD_zone_temp"}), task, "AD_zone_temp", 0, 42)
            for _, m in metrics.iterrows():
                rec = m.to_dict()
                rec["reliable_threshold"] = reliable
                rec["caution_threshold"] = caution
                rows.append(rec)
    return pd.DataFrame(rows)


def assess_threshold_support(zone_metrics: pd.DataFrame, task: str) -> tuple[str, str]:
    if zone_metrics.empty:
        return "Not_evaluable", "No zone metrics"
    metric = "MAE" if TASK_TYPE[task] == "regression" else "Accuracy"
    vals = {r["group"]: r.get(metric, np.nan) for _, r in zone_metrics.iterrows()}
    needed = ["In_domain", "Caution", "Extrapolation"]
    if not all(k in vals and pd.notna(vals[k]) for k in needed):
        return "Partly_evaluable", "At least one AD zone is empty or lacks a valid metric"
    a, b, c = (float(vals[k]) for k in needed)
    if TASK_TYPE[task] == "regression":
        if a <= b <= c:
            return "Supported", "MAE increases monotonically from in-domain to extrapolation"
        if a < c:
            return "Partly_supported", "Extrapolation MAE exceeds in-domain MAE, but the caution zone is non-monotonic"
        return "Not_supported", "Extrapolation MAE does not exceed in-domain MAE"
    if a >= b >= c:
        return "Supported", "Accuracy decreases monotonically from in-domain to extrapolation"
    if a > c:
        return "Partly_supported", "Extrapolation accuracy is lower than in-domain accuracy, but the caution zone is non-monotonic"
    return "Not_supported", "Extrapolation accuracy is not lower than in-domain accuracy"


def save_figure(fig, base: Path, formats: list[str], dpi: int) -> list[str]:
    paths = []
    for fmt in formats:
        path = base.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=dpi if fmt.lower() == "png" else None, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def plot_similarity_distribution(df: pd.DataFrame, task: str, outdir: Path, cfg: Config) -> list[str]:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    vals = df["AD_similarity"].dropna()
    ax.hist(vals, bins=np.linspace(0, 1, 21), edgecolor="black", alpha=0.8)
    ax.axvline(cfg.caution_threshold, linestyle="--", label=f"Caution threshold={cfg.caution_threshold:.2f}")
    ax.axvline(cfg.reliable_threshold, linestyle="--", label=f"Reliable threshold={cfg.reliable_threshold:.2f}")
    ax.set_xlabel("Maximum role-specific Morgan Tanimoto similarity")
    ax.set_ylabel("Number of outer-test formulations")
    ax.set_title(f"{task}: structural-similarity distribution")
    ax.legend(frameon=False)
    return save_figure(fig, outdir / f"{task}_AD_similarity_distribution", cfg.formats, cfg.dpi)


def plot_error_or_accuracy(df: pd.DataFrame, task: str, outdir: Path, cfg: Config) -> list[str]:
    if TASK_TYPE[task] == "regression":
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        valid = df[["AD_similarity", "absolute_error"]].dropna().sort_values("AD_similarity")
        ax.scatter(valid["AD_similarity"], valid["absolute_error"], alpha=0.45, s=20)
        if len(valid) >= 20:
            window = max(15, len(valid) // 10)
            rolling = valid["absolute_error"].rolling(window=window, center=True, min_periods=max(5, window // 3)).median()
            ax.plot(valid["AD_similarity"], rolling, linewidth=2, label="Rolling median absolute error")
            ax.legend(frameon=False)
        ax.axvline(cfg.caution_threshold, linestyle="--")
        ax.axvline(cfg.reliable_threshold, linestyle="--")
        ax.set_xlabel("Maximum role-specific Morgan Tanimoto similarity")
        ax.set_ylabel("Absolute outer-fold prediction error")
        ax.set_title(f"{task}: error versus structural similarity")
        return save_figure(fig, outdir / f"{task}_AD_error_vs_similarity", cfg.formats, cfg.dpi)

    temp = df[["AD_similarity", "correct"]].dropna().copy()
    temp["similarity_bin"] = pd.cut(temp["AD_similarity"], bins=np.linspace(0, 1, 11), include_lowest=True)
    binned = temp.groupby("similarity_bin", observed=False).agg(
        similarity_mean=("AD_similarity", "mean"), accuracy=("correct", "mean"), n=("correct", "size")
    ).dropna(subset=["similarity_mean"])
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(binned["similarity_mean"], binned["accuracy"], marker="o")
    for row in binned.itertuples(index=False):
        ax.annotate(f"n={int(row.n)}", (row.similarity_mean, row.accuracy), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
    ax.axvline(cfg.caution_threshold, linestyle="--")
    ax.axvline(cfg.reliable_threshold, linestyle="--")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Maximum role-specific Morgan Tanimoto similarity")
    ax.set_ylabel("Outer-fold classification accuracy")
    ax.set_title(f"{task}: accuracy versus structural similarity")
    return save_figure(fig, outdir / f"{task}_AD_accuracy_vs_similarity", cfg.formats, cfg.dpi)


def plot_zone_metric(zone_metrics: pd.DataFrame, task: str, outdir: Path, cfg: Config) -> list[str]:
    metric = "MAE" if TASK_TYPE[task] == "regression" else "Accuracy"
    order = ["In_domain", "Caution", "Extrapolation", "Invalid_or_missing"]
    d = zone_metrics[zone_metrics["group"].isin(order)].copy()
    d["order"] = d["group"].map({k: i for i, k in enumerate(order)})
    d = d.sort_values("order")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bars = ax.bar(d["group"], d[metric])
    for bar, n in zip(bars, d["n"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"n={int(n)}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(metric)
    ax.set_title(f"{task}: performance by applicability-domain zone")
    ax.tick_params(axis="x", rotation=15)
    return save_figure(fig, outdir / f"{task}_AD_zone_{metric}", cfg.formats, cfg.dpi)


def plot_scaffold_metric(scaffold_metrics: pd.DataFrame, task: str, outdir: Path, cfg: Config) -> list[str]:
    metric = "MAE" if TASK_TYPE[task] == "regression" else "Accuracy"
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    bars = ax.bar(scaffold_metrics["group"], scaffold_metrics[metric])
    for bar, n in zip(bars, scaffold_metrics["n"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"n={int(n)}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(metric)
    ax.set_title(f"{task}: seen versus new Murcko scaffold")
    ax.tick_params(axis="x", rotation=10)
    return save_figure(fig, outdir / f"{task}_AD_scaffold_{metric}", cfg.formats, cfg.dpi)


def summarize_task(df: pd.DataFrame, task: str, zone_metrics: pd.DataFrame, support: str, reason: str) -> dict:
    valid = df.dropna(subset=["AD_similarity"])
    if TASK_TYPE[task] == "regression":
        corr, pval = spearmanr(valid["AD_similarity"], valid["absolute_error"], nan_policy="omit")
        corr_name = "Spearman_similarity_vs_absolute_error"
    else:
        corr, pval = spearmanr(valid["AD_similarity"], valid["correct"], nan_policy="omit")
        corr_name = "Spearman_similarity_vs_correctness"
    counts = df["AD_zone"].value_counts().to_dict()
    row = {
        "task": task,
        "n": int(len(df)),
        "n_unique_main": int(df["AD_Canonical_SMILES_main"].nunique(dropna=True)),
        "n_unique_pairs": int(df["AD_pair_key"].nunique(dropna=True)),
        "AD_similarity_mean": float(valid["AD_similarity"].mean()) if len(valid) else float("nan"),
        "AD_similarity_median": float(valid["AD_similarity"].median()) if len(valid) else float("nan"),
        "In_domain_n": int(counts.get("In_domain", 0)),
        "Caution_n": int(counts.get("Caution", 0)),
        "Extrapolation_n": int(counts.get("Extrapolation", 0)),
        "Invalid_or_missing_n": int(counts.get("Invalid_or_missing", 0)),
        "new_scaffold_fraction": float((df["scaffold_status"] == "New_scaffold").mean()),
        corr_name: float(corr) if np.isfinite(corr) else float("nan"),
        f"{corr_name}_p": float(pval) if np.isfinite(pval) else float("nan"),
        "threshold_support": support,
        "threshold_support_reason": reason,
    }
    for _, r in zone_metrics.iterrows():
        zone = r["group"]
        metric = "MAE" if TASK_TYPE[task] == "regression" else "Accuracy"
        if metric in r and pd.notna(r[metric]):
            row[f"{zone}_{metric}"] = float(r[metric])
    return row


def main() -> int:
    args = parse_args()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in TASK_TYPE]
    if unknown:
        raise SystemExit(f"Unsupported tasks: {unknown}. Supported: {sorted(TASK_TYPE)}")
    if not (0 <= args.caution_threshold < args.reliable_threshold <= 1):
        raise SystemExit("Thresholds must satisfy 0 <= caution < reliable <= 1")

    cfg = Config(
        results_root=args.results_root,
        output=args.output,
        tasks=tasks,
        radius=args.radius,
        nbits=args.nbits,
        reliable_threshold=args.reliable_threshold,
        caution_threshold=args.caution_threshold,
        bootstrap_iterations=args.bootstrap_iterations,
        random_state=args.random_state,
        formats=[x.strip().lower() for x in args.formats.split(",") if x.strip()],
        dpi=args.dpi,
    )
    results_root = Path(cfg.results_root).resolve()
    output = Path(cfg.output).resolve()
    tables_dir = output / "tables"
    figures_dir = output / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    (output / "STEP4_run_config.json").write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")

    summary_rows = []
    manifest_rows = []
    all_zone_metrics = []
    all_scaffold_metrics = []

    for task in tasks:
        print("=" * 88)
        print(f"[STEP4 AD] task={task}")
        path = find_prediction_file(results_root, task)
        print(f"[INFO] predictions={path}")
        raw = prepare_predictions(path, task)
        ad = calculate_fold_ad(raw, cfg.radius, cfg.nbits, cfg.reliable_threshold, cfg.caution_threshold)

        task_table_dir = tables_dir / task
        task_figure_dir = figures_dir / task
        task_table_dir.mkdir(parents=True, exist_ok=True)
        task_figure_dir.mkdir(parents=True, exist_ok=True)

        pred_path = task_table_dir / f"{task}_outer_predictions_with_AD.csv"
        ad.to_csv(pred_path, index=False, encoding="utf-8-sig")

        zone_metrics = grouped_metrics(ad, task, "AD_zone", cfg.bootstrap_iterations, cfg.random_state)
        scaffold_metrics = grouped_metrics(ad, task, "scaffold_status", cfg.bootstrap_iterations, cfg.random_state + 100)
        fold_metric = fold_metrics(ad, task, 0, cfg.random_state)
        sens = threshold_sensitivity(
            ad,
            task,
            reliable_values=[0.60, 0.65, 0.70, 0.75, 0.80],
            caution_values=[0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
        )

        zone_path = task_table_dir / f"{task}_AD_zone_metrics.csv"
        scaffold_path = task_table_dir / f"{task}_AD_scaffold_metrics.csv"
        fold_path = task_table_dir / f"{task}_AD_fold_metrics.csv"
        sens_path = task_table_dir / f"{task}_AD_threshold_sensitivity.csv"
        zone_metrics.to_csv(zone_path, index=False, encoding="utf-8-sig")
        scaffold_metrics.to_csv(scaffold_path, index=False, encoding="utf-8-sig")
        fold_metric.to_csv(fold_path, index=False, encoding="utf-8-sig")
        sens.to_csv(sens_path, index=False, encoding="utf-8-sig")

        support, reason = assess_threshold_support(zone_metrics, task)
        summary = summarize_task(ad, task, zone_metrics, support, reason)
        summary["prediction_source"] = str(path)
        summary_rows.append(summary)
        all_zone_metrics.append(zone_metrics)
        all_scaffold_metrics.append(scaffold_metrics)

        fig_paths = []
        fig_paths += plot_similarity_distribution(ad, task, task_figure_dir, cfg)
        fig_paths += plot_error_or_accuracy(ad, task, task_figure_dir, cfg)
        fig_paths += plot_zone_metric(zone_metrics, task, task_figure_dir, cfg)
        fig_paths += plot_scaffold_metric(scaffold_metrics, task, task_figure_dir, cfg)

        for p in [pred_path, zone_path, scaffold_path, fold_path, sens_path, *map(Path, fig_paths)]:
            manifest_rows.append({"task": task, "artifact": str(p), "kind": "figure" if p.suffix.lower() in {".png", ".pdf", ".svg"} else "table"})

        print(f"[OK] {task}: zones={ad['AD_zone'].value_counts().to_dict()} support={support}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = tables_dir / "core_AD_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.concat(all_zone_metrics, ignore_index=True).to_csv(tables_dir / "core_AD_zone_metrics_all.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_scaffold_metrics, ignore_index=True).to_csv(tables_dir / "core_AD_scaffold_metrics_all.csv", index=False, encoding="utf-8-sig")

    manifest_path = output / "STEP4_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False, encoding="utf-8-sig")

    lines = [
        "# STEP 4 Applicability-domain summary",
        "",
        f"Morgan radius: {cfg.radius}; bits: {cfg.nbits}",
        f"Prespecified zones: In-domain >= {cfg.reliable_threshold:.2f}; Caution >= {cfg.caution_threshold:.2f}; otherwise Extrapolation.",
        "",
        "The threshold-support label is descriptive. Final paper wording should also inspect sample counts, fold-level metrics, and threshold-sensitivity tables.",
        "",
    ]
    for row in summary_rows:
        lines += [
            f"## {row['task']}",
            f"- In-domain / caution / extrapolation: {row['In_domain_n']} / {row['Caution_n']} / {row['Extrapolation_n']}",
            f"- New-scaffold fraction: {row['new_scaffold_fraction']:.3f}",
            f"- Threshold assessment: {row['threshold_support']} — {row['threshold_support_reason']}",
            "",
        ]
    (output / "STEP4_applicability_domain_summary.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n[DONE] STEP 4 applicability-domain validation completed.")
    print(f"[DONE] Summary: {summary_path}")
    print(f"[DONE] Figures: {figures_dir}")
    print(f"[DONE] Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
