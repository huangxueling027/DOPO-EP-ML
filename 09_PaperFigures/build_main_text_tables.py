# -*- coding: utf-8 -*-
"""Generate the three manuscript-facing tables used in the frozen paper.

The tables are presentation-layer outputs only.  They read the frozen FINAL
results and never refit a model, reselect a feature view/K, or choose a best
outer fold.

Frozen manuscript table set
---------------------------
Table 1  Outer-test performance under strict 5x5 nested grouped CV.
Table 2  Grouping robustness, BDE ablation, and cross-fold SHAP stability.
Table 3  Top-10 cross-scenario priority candidates from the final 19.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_utils import read_csv_auto  # noqa: E402

MAIN_TASKS = ["LOI", "PHRR", "THR", "UL94_V0", "Tg", "TS_MPa"]
CORE_TASKS = ["LOI", "PHRR", "THR", "UL94_V0"]
DISPLAY = {
    "LOI": "LOI",
    "PHRR": "PHRR",
    "THR": "THR",
    "UL94_V0": "UL-94",
    "Tg": "Tg",
    "TS_MPa": "TS",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tables-root", type=Path, default=ROOT / "results" / "09_PaperTables")
    p.add_argument("--output", type=Path, default=ROOT / "results" / "09_MainTextTables")
    return p.parse_args()


def _num(value: object) -> float:
    return pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]


def _fmt(mean: object, std: object, digits: int = 3) -> str:
    m, s = _num(mean), _num(std)
    if pd.isna(m):
        return ""
    if pd.isna(s):
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def _mean(value: object, digits: int = 2) -> str:
    x = _num(value)
    return "" if pd.isna(x) else f"{x:.{digits}f}"


def _preferred_summary_row(all_metrics: pd.DataFrame, task: str) -> pd.Series | None:
    """Select the canonical frozen molecule/formulation/without-BDE summary."""
    g = all_metrics.copy()
    g = g[g["task"].astype(str).eq(task)]
    if "use_BDE" in g:
        g = g[pd.to_numeric(g["use_BDE"], errors="coerce").fillna(0).eq(0)]
    if "split_strategy" in g:
        g = g[g["split_strategy"].astype(str).str.lower().eq("molecule")]
    if "screening_mode" in g:
        g = g[g["screening_mode"].astype(str).str.lower().eq("formulation")]
    if "selection_scope" in g:
        g = g[g["selection_scope"].astype(str).str.lower().eq("fixed")]
    if "feature_scope" in g:
        g = g[~g["feature_scope"].astype(str).str.lower().isin(["conditions_only", "molecular_only"])]
    if g.empty:
        return None

    src = g.get("source_file", pd.Series("", index=g.index)).astype(str).str.replace("\\", "/").str.lower()
    desired = (
        "final_aux_fixed_baseline_inclusive_5x5"
        if task in {"Tg", "TS_MPa"}
        else "final_core_fixed_baseline_inclusive_5x5"
    )
    score = pd.Series(0.0, index=g.index)
    score += src.str.contains(desired, regex=False) * 100
    score += src.str.endswith(f"/{task.lower()}_nested_summary.csv") * 20
    score += (~src.str.endswith("scientific_nested_summary_all.csv")) * 5
    score -= src.str.contains("smoke|2x2|conditions_only|molecular_only", regex=True) * 500
    g = g.assign(_priority=score).sort_values("_priority", ascending=False, kind="stable")
    return g.iloc[0]


def table1_performance(tables_root: Path) -> pd.DataFrame:
    """Match manuscript Table 1 exactly (six reported tasks, five columns)."""
    path = tables_root / "Data_complete_5x5_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run paper-tables first; missing {path}")
    metrics = read_csv_auto(path)
    rows: list[dict[str, object]] = []
    for task in MAIN_TASKS:
        r = _preferred_summary_row(metrics, task)
        if r is None:
            raise RuntimeError(f"Missing frozen FINAL summary for {task}")
        n = _num(r.get("n_valid_formal_rows", r.get("n_valid_target_rows", np.nan)))
        if pd.isna(n):
            # Frozen manuscript sample counts; used only if a legacy summary lacks n.
            n = {"LOI": 599, "PHRR": 406, "THR": 397, "UL94_V0": 599, "Tg": 357, "TS_MPa": 256}[task]

        if task == "UL94_V0":
            performance = _fmt(r.get("outer_Macro_F1_mean"), r.get("outer_Macro_F1_std"), 3)
            secondary = (
                f"Balanced Accuracy = {_mean(r.get('outer_Balanced_Accuracy_mean'), 3)}; "
                f"ROC-AUC = {_mean(r.get('outer_ROC_AUC_mean'), 3)}"
            )
            metric = "Macro-F1"
        else:
            performance = _fmt(r.get("outer_R2_mean"), r.get("outer_R2_std"), 3)
            rmse = _mean(r.get("outer_RMSE_mean"), 2)
            mae = _mean(r.get("outer_MAE_mean"), 2)
            unit = {
                "LOI": "",
                "PHRR": " kW·m^-2",
                "THR": " MJ·m^-2",
                "Tg": " °C",
                "TS_MPa": " MPa",
            }[task]
            secondary = f"RMSE = {rmse}{unit}; MAE = {mae}{unit}"
            metric = "R²"

        rows.append({
            "Task": DISPLAY[task],
            "n": int(n),
            "Primary metric": metric,
            "Performance": performance,
            "Secondary metrics": secondary,
        })
    return pd.DataFrame(rows)


def _pick_metric_row(frame: pd.DataFrame, task: str, split_name: str, use_bde: int) -> pd.Series | None:
    g = frame[frame["task"].astype(str).eq(task)].copy()
    if "split_strategy" in g:
        g = g[g["split_strategy"].astype(str).str.lower().eq(split_name)]
    if "use_BDE" in g:
        g = g[pd.to_numeric(g["use_BDE"], errors="coerce").fillna(0).eq(use_bde)]
    if "screening_mode" in g:
        g = g[g["screening_mode"].astype(str).str.lower().eq("formulation")]
    if "selection_scope" in g:
        g = g[g["selection_scope"].astype(str).str.lower().eq("fixed")]
    return None if g.empty else g.iloc[0]


def _shap_rank_mean(results_root: Path, task: str) -> str:
    path = (
        results_root / "05_Shap" / "FINAL_core_fixed_baseline_inclusive_5x5" /
        task / "molecule" / "formulation" / "without_BDE" /
        f"{task}_SHAP_rank_spearman.csv"
    )
    if not path.exists():
        return ""
    corr = read_csv_auto(path)
    matrix = (
        corr.drop(columns=["outer_fold"], errors="ignore")
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return ""
    upper = matrix[np.triu_indices_from(matrix, k=1)]
    return f"{float(np.nanmean(upper)):.3f}"


def table2_robustness(tables_root: Path) -> pd.DataFrame:
    """Match manuscript Table 2 exactly; molecule column is the without-BDE formal result."""
    results_root = tables_root.parent
    split_path = results_root / "scientific_validation" / "FINAL_grouping_sensitivity_5x5" / "scientific_nested_summary_all.csv"
    bde_path = results_root / "scientific_validation" / "FINAL_BDE_paired_ablation_5x5" / "scientific_nested_summary_all.csv"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing grouping-sensitivity summary: {split_path}")
    if not bde_path.exists():
        raise FileNotFoundError(f"Missing paired-BDE summary: {bde_path}")
    split, bde = read_csv_auto(split_path), read_csv_auto(bde_path)

    rows = []
    for task in CORE_TASKS:
        metric_mean = "outer_Macro_F1_mean" if task == "UL94_V0" else "outer_R2_mean"
        metric_std = "outer_Macro_F1_std" if task == "UL94_V0" else "outer_R2_std"

        vals: dict[str, str] = {}
        for split_name, label in [("molecule", "Molecule"), ("scaffold", "Scaffold"), ("reference", "Reference")]:
            r = _pick_metric_row(split, task, split_name, 0)
            vals[label] = "" if r is None else _fmt(r.get(metric_mean), r.get(metric_std), 3)

        r0 = _pick_metric_row(bde, task, "molecule", 0)
        r1 = _pick_metric_row(bde, task, "molecule", 1)
        v0 = np.nan if r0 is None else _num(r0.get(metric_mean))
        v1 = np.nan if r1 is None else _num(r1.get(metric_mean))
        with_bde = "" if r1 is None else _fmt(r1.get(metric_mean), r1.get(metric_std), 3)
        if np.isfinite(v0) and np.isfinite(v1):
            d = float(v1 - v0)
            delta = "0" if abs(d) < 0.0005 else f"{d:+.3f}"
        else:
            delta = ""

        rows.append({
            "Task": DISPLAY[task],
            "Metric": "Macro-F1" if task == "UL94_V0" else "R²",
            "Molecule": vals["Molecule"],
            "Scaffold": vals["Scaffold"],
            "Reference": vals["Reference"],
            "With BDE": with_bde,
            "ΔMetric after BDE": delta,
            "SHAP ρ": _shap_rank_mean(results_root, task),
        })
    return pd.DataFrame(rows)


def table3_virtual_screening(tables_root: Path) -> pd.DataFrame:
    """Match manuscript Table 3 exactly (50 kW/m² displayed values; cross-flux rank)."""
    path = (
        tables_root.parent / "06_ReverseDesign" / "FINAL_fixed" /
        "final_priority_combined" / "final_priority_candidates_combined.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing final combined priority table: {path}")
    frame = read_csv_auto(path).copy()
    if "Final_priority_rank" in frame:
        frame = frame.sort_values("Final_priority_rank", kind="stable")
    frame = frame.head(10).copy()

    out = pd.DataFrame({
        "Rank": frame.get("Final_priority_rank", pd.Series(range(1, len(frame) + 1), index=frame.index)),
        "Candidate": frame.get("Candidate_Name", ""),
        "Design family": frame.get("Design_Family", ""),
        "Loading (wt%)": frame.get("Loading_total_FR wt%_50", np.nan),
        "Pred. LOI (%)": frame.get("LOI_pred_50", np.nan),
        "Pred. PHRR (kW·m^-2)": frame.get("PHRR_pred_50", np.nan),
        "Pred. THR (MJ·m^-2)": frame.get("THR_pred_50", np.nan),
        "Pred. V-0 probability": frame.get("V0_probability_50", np.nan),
        "Pred. Tg (°C)": frame.get("Tg_pred_50", np.nan),
        "Pred. TS (MPa)": frame.get("TS_pred_50", np.nan),
    })
    for col, digits in {
        "Loading (wt%)": 2,
        "Pred. LOI (%)": 2,
        "Pred. PHRR (kW·m^-2)": 2,
        "Pred. THR (MJ·m^-2)": 2,
        "Pred. V-0 probability": 3,
        "Pred. Tg (°C)": 2,
        "Pred. TS (MPa)": 2,
    }.items():
        out[col] = pd.to_numeric(out[col], errors="coerce").round(digits)
    out["Rank"] = pd.to_numeric(out["Rank"], errors="coerce").astype("Int64")
    return out.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # Remove legacy five-table outputs so the directory exposes only the three
    # tables that actually appear in the final manuscript.
    for old in args.output.glob("*.csv"):
        try:
            old.unlink()
        except OSError:
            pass

    tables = [
        (
            "Table 1. Outer-test performance under strict 5x5 nested grouped cross-validation.csv",
            table1_performance(args.tables_root),
            "Table 1. Outer-test performance under strict 5×5 nested grouped cross-validation.",
        ),
        (
            "Table 2. Grouping robustness, BDE ablation, and cross-fold SHAP stability of the core flame-retardancy models.csv",
            table2_robustness(args.tables_root),
            "Table 2. Grouping robustness, BDE ablation, and cross-fold SHAP stability of the core flame-retardancy models.",
        ),
        (
            "Table 3. Top 10 cross-scenario priority DOPO-derived candidates identified by multi-objective virtual screening.csv",
            table3_virtual_screening(args.tables_root),
            "Table 3. Top 10 cross-scenario priority DOPO-derived candidates identified by multi-objective virtual screening.",
        ),
    ]

    manifest = []
    for i, (filename, frame, caption) in enumerate(tables, start=1):
        frame.to_csv(args.output / filename, index=False, encoding="utf-8-sig")
        manifest.append({"table": f"Table {i}", "file": filename, "rows": len(frame), "caption": caption})
        print(f"[OK] Table {i}: {filename} ({len(frame)} rows)")

    pd.DataFrame(manifest).to_csv(
        args.output / "main_text_table_manifest.csv", index=False, encoding="utf-8-sig"
    )
    print(f"\n[DONE] Frozen manuscript tables 1-3: {args.output}")


if __name__ == "__main__":
    main()
