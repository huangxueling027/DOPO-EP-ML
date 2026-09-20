# -*- coding: utf-8 -*-
"""Build frozen Supplementary Tables S1(A-B)-S7(A-B) plus source data.

The numbered CSV outputs mirror the current Supplementary Information table
content, while detailed audit rows remain as unnumbered machine-readable
Data_*.csv files. No model is refit and no scientific result is changed here.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from paper_utils import (
    ALL_REGRESSION_TASKS,
    ROOT,
    TARGET_COLUMNS,
    ResultIndex,
    canonical_smiles,
    murcko_scaffold,
    locate_task_file,
    normalise_ul94,
    numeric,
    probability_column,
    read_csv_auto,
    regression_metrics,
    task_label,
    write_json,
)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TASK_ORDER = [
    "LOI", "PHRR", "THR", "UL94_V0", "Tg", "Char_yield", "TS_MPa", "FS_MPa",
    "Delta_LOI", "Delta_PHRR", "Delta_THR", "Delta_CY",
]

TASK_DISPLAY = {
    "LOI": "LOI", "PHRR": "PHRR", "THR": "THR", "UL94_V0": "UL-94 V-0",
    "Tg": "Tg", "Char_yield": "Char yield", "TS_MPa": "TS", "FS_MPa": "FS",
    "Delta_LOI": "ΔLOI", "Delta_PHRR": "ΔPHRR", "Delta_THR": "ΔTHR", "Delta_CY": "ΔCY",
}
GROUP_DISPLAY = {
    "MAIN_CO": "Main FR + synergist",
    "MAIN_CO_CURING": "Main FR + synergist + curing agent",
}
FINAL_REFERENCE_COUNTS = {
    "LOI": 130, "PHRR": 116, "THR": 113, "UL94_V0": 130, "Tg": 90, "TS_MPa": 66,
}
EXTERNAL_REFERENCE_MAP = {
    "MFD": "[43]", "MBFAP": "[44]", "SPDO": "[45]",
    "VH-DOPO": "[46]", "VPAA-DOPO": "[47]", "DMM": "[48]",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "DOPO_EP_new_with_BDE.csv")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--task-config", type=Path, default=ROOT / "config" / "task_config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "09_PaperTables")
    parser.add_argument("--top-errors", type=int, default=20)
    return parser.parse_args()


def _unit(column: str) -> str:
    """Return manuscript-facing units/encodings with explicit target/baseline handling."""
    c = column.strip()
    cl = c.lower()
    exact = {
        "Record_ID": "identifier",
        "LOI": "%", "EP_matrix_LOI": "%", "Delta_LOI": "percentage points",
        "UL94": "class", "UL94_num": "code",
        "PHRR_kw_㎡": "kW m^-2", "EP_matrix_PHRR": "kW m^-2", "Delta_PHRR": "kW m^-2",
        "THR_MJ_㎡": "MJ m^-2", "EP_matrix_THR": "MJ m^-2", "Delta_THR": "MJ m^-2",
        "Tg_℃": "°C", "EP_matrix_Tg": "°C", "Delta_Tg": "°C",
        "Char_yield_％_700C": "%", "EP_matrix_CY": "%", "Delta_CY": "percentage points",
        "TS_MPa": "MPa", "EP_matrix_TS": "MPa", "Delta_TS": "MPa",
        "FS_MPa": "MPa", "EP_matrix_FS": "MPa", "Delta_FS": "MPa",
        "FR_main_BDE_Kcal/mol": "kcal mol^-1",
        "FR_main_BDE_KJ/mol": "kJ mol^-1",
        "FR_main_BDE_final_kJ_mol": "kJ mol^-1",
        "FR_main_BDE_available": "binary",
        "FR_main_BDE_source": "category",
        "FR_main_BDE_low_confidence": "binary",
        "Preparation_Method_num": "code",
        "Synergy_flag(single=0&synergy=1)": "binary",
        "BDE_Type": "category",
    }
    if c in exact:
        return exact[c]
    if re.search(r"(?:^|_)(?:model|mode|source|note|task|name)(?:$|_)", cl) and "bde" in cl:
        return "text / category"
    if "smiles" in cl:
        return "SMILES"
    if "thickness" in cl:
        return "mm"
    if "flux" in cl:
        return "kW m^-2"
    if "temp" in cl or "℃" in c:
        return "°C"
    if "_mpa" in cl:
        return "MPa"
    if "phrr" in cl:
        return "kW m^-2"
    if "thr" in cl:
        return "MJ m^-2"
    if "bde" in cl and any(t in cl for t in ["pred", "std", "diff", "kj_mol", "kj/mol"]):
        return "kJ mol^-1"
    if any(t in cl for t in ["_content wt%", "loading_total_fr wt%"]):
        return "wt%"
    if cl.endswith("_fraction") or "fraction" in cl or cl.endswith(" ratio") or cl.endswith("_ratio"):
        return "fraction / ratio"
    if cl.startswith("has_") or cl.startswith("curingagent_has_"):
        return "binary"
    if any(t in cl for t in ["reference", "journal", "title", "doi"]):
        return "text / identifier"
    return "dimensionless / text"

def _description(column: str) -> str:
    descriptions = {
        "FR_main": "Main DOPO-derived flame retardant name",
        "FR_co": "Co-flame-retardant or synergist name",
        "Curing_Agent": "Epoxy curing-agent name",
        "Preparation_Method": "Standardized nominal preparation method: additive, reactive, co-curing, or additive plus secondary crosslinking",
        "Preparation_Method_num": "Record-management code only: 0 additive; 1 reactive; 2 co-curing; 3 additive plus secondary crosslinking; not used as an ordinal predictive feature",
        "P_content wt%": "Formulation-level phosphorus content",
        "N_content wt%": "Formulation-level nitrogen content",
        "S_content wt%": "Formulation-level sulfur content",
        "B_content wt%": "Formulation-level boron content",
        "Si_content wt%": "Formulation-level silicon content",
        "Synergy_type": "Elemental or structural synergistic category",
        "Synergy_flag": "Whether a defined synergistic component/mechanism is present",
        "SMILES_main": "SMILES of main flame retardant",
        "SMILES_co": "SMILES of co-flame-retardant/synergist",
        "SMILES_Curing_Agent": "SMILES of curing agent",
        "Loading_total_FR wt%": "Total flame-retardant loading relative to the formulation",
        "LOI": "Limiting oxygen index",
        "UL94": "UL-94 vertical burning class",
        "PHRR_kw_㎡": "Peak heat-release rate",
        "THR_MJ_㎡": "Total heat release",
        "Tg_℃": "Glass-transition temperature measured by dynamic mechanical analysis (DMA)",
        "Char_yield_％_700C": "Char yield at 700 °C",
        "TS_MPa": "Tensile strength",
        "FS_MPa": "Flexural strength",
        "Delta_LOI": "LOI difference relative to matched neat-EP baseline",
        "Delta_PHRR": "PHRR difference relative to matched neat-EP baseline",
        "Delta_THR": "THR difference relative to matched neat-EP baseline",
        "Delta_CY": "Char-yield difference relative to matched neat-EP baseline",
        "Cure_Temp_Max": "Highest reported curing temperature",
        "Cone_flux_kW_m2": "Cone-calorimeter external heat flux; not inferred from PHRR/THR",
        "BDE": "Predicted or assigned bond-dissociation energy feature",
        "BDE_Type": "Bond type associated with the BDE value",
    }
    if column in descriptions:
        return descriptions[column]
    text = column.replace("_", " ").replace("％", "%").replace("℃", "°C").strip()
    return text[0].upper() + text[1:] if text else column


def table_s1(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, col in enumerate(df.columns, start=1):
        series = df[col]
        rows.append({
            "field_order": i,
            "field": col,
            "description": _description(col),
            "unit_or_encoding": _unit(col),
            "pandas_dtype": str(series.dtype),
            "non_missing_n": int(series.notna().sum()),
            "missing_n": int(series.isna().sum()),
            "missing_rate_percent": round(float(series.isna().mean() * 100), 3),
            "unique_non_missing": int(series.nunique(dropna=True)),
        })
    return pd.DataFrame(rows)


def table_s2(config_path: Path, df: pd.DataFrame) -> pd.DataFrame:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    strict_configs = None
    try:
        from common.scientific_evaluation import TASK_CONFIGS  # local project import
        strict_configs = TASK_CONFIGS
    except Exception:
        strict_configs = {}
    rows = []
    for task in payload.get("workflow_order", list(payload.get("tasks", {}))):
        cfg = payload["tasks"][task]
        strict = strict_configs.get(task) if strict_configs else None
        target_col = TARGET_COLUMNS.get(task, task)
        target = df[target_col] if target_col in df else pd.Series(dtype=float)

        # Formal Delta models use modified formulations only.  Neat-EP rows may
        # contain stored Delta fields for pairing/audit purposes, so a raw
        # non-missing count would overstate the actual modelling sample size.
        # Keep absolute-property tasks baseline-inclusive, but require
        # Loading_total_FR wt% > 0 for Delta tasks.
        valid_mask = target.notna()
        if str(task).startswith("Delta_") and "Loading_total_FR wt%" in df.columns:
            loading = pd.to_numeric(df["Loading_total_FR wt%"], errors="coerce")
            valid_mask = valid_mask & loading.gt(0)
        n = int(valid_mask.sum())
        rows.append({
            "task": task,
            "display_name": task_label(task),
            "target_column": target_col,
            "task_type": cfg.get("type"),
            "category": cfg.get("category"),
            "descriptor_mode": cfg.get("descriptor_mode"),
            "use_v7_features": cfg.get("use_v7"),
            "frozen_view": cfg.get("view"),
            "frozen_k": cfg.get("k"),
            "main_with_BDE": cfg.get("main_bde"),
            "group_rule": cfg.get("group"),
            "valid_target_n": n,
            "strict_curated_views": ";".join(strict.curated_views) if strict else "",
            "strict_curated_k": ";".join("ALL" if x is None else str(x) for x in strict.curated_k) if strict else "",
            "strict_full_k": ";".join("ALL" if x is None else str(x) for x in strict.full_k) if strict else "",
        })
    return pd.DataFrame(rows)


def _compact_params(model: object) -> str:
    params = getattr(model, "get_params", lambda: {})()
    keep = [
        "alpha", "C", "epsilon", "kernel", "n_estimators", "max_depth", "learning_rate",
        "min_samples_leaf", "subsample", "colsample_bytree", "num_leaves", "min_child_samples",
        "reg_alpha", "reg_lambda", "class_weight", "probability", "voting",
    ]
    pieces = []
    for key in keep:
        if key in params and params[key] is not None:
            pieces.append(f"{key}={params[key]}")
    return "; ".join(pieces)


def table_s3() -> pd.DataFrame:
    rows = []
    try:
        from common.scientific_evaluation import (
            DEFAULT_CLASSIFICATION_MODELS,
            DEFAULT_REGRESSION_MODELS,
            _classification_model,
            _regression_model,
        )
        for task_type, names, factory in [
            ("regression", DEFAULT_REGRESSION_MODELS, _regression_model),
            ("classification", DEFAULT_CLASSIFICATION_MODELS, _classification_model),
        ]:
            for name in names:
                model = factory(name, seed=42)
                rows.append({
                    "task_type": task_type,
                    "model": name,
                    "implementation": model.__class__.__name__,
                    "frozen_parameters": _compact_params(model),
                    "selection_level": "selected within each outer-training set by grouped inner CV",
                })
    except Exception as exc:
        rows.append({
            "task_type": "ERROR", "model": "", "implementation": "", "frozen_parameters": str(exc),
            "selection_level": "Inspect common/scientific_evaluation.py",
        })
    preprocessing = [
        ("all", "Non-empty column selector", "Fit on each training fold only"),
        ("all", "Median imputation", "Fit on each training fold only"),
        ("all", "Duplicate-column removal", "Fit on each training fold only"),
        ("all", "VarianceThreshold", "threshold=1e-8; fit on each training fold only"),
        ("all", "SelectKBest", "f_regression for regression; f_classif for classification"),
        ("linear/kernel", "StandardScaler", "Ridge, SVR, Logistic and SVC only"),
    ]
    for task_type, model, params in preprocessing:
        rows.append({
            "task_type": task_type, "model": model, "implementation": "preprocessing",
            "frozen_parameters": params, "selection_level": "inside nested CV; no test-fold fitting",
        })
    return pd.DataFrame(rows)


def _all_result_files(index: ResultIndex, suffix: str) -> list[Path]:
    return [p for p in index.csv_files if p.name.endswith(suffix) and "smoke" not in p.as_posix().lower() and "2x2" not in p.as_posix().lower()]


def _concat_with_source(paths: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        try:
            frame = read_csv_auto(path)
        except Exception:
            continue
        frame.insert(0, "source_file", str(path))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def table_s4(index: ResultIndex) -> pd.DataFrame:
    paths = _all_result_files(index, "_outer_fold_metrics.csv")
    frame = _concat_with_source(paths)
    if frame.empty:
        return frame
    sort_cols = [c for c in ["task", "use_BDE", "split_strategy", "outer_fold"] if c in frame]
    return frame.sort_values(sort_cols, kind="stable") if sort_cols else frame


def _summary_source_priority(path: str) -> tuple[int, int, str]:
    """Prefer canonical scientific-validation summaries over copied SHAP summaries."""
    norm = str(path).replace("\\", "/").lower()
    if any(token in norm for token in [
        "/scientific_validation/final_core_fixed_baseline_inclusive_5x5/",
        "/scientific_validation/final_aux_fixed_baseline_inclusive_5x5/",
        "/scientific_validation/final_exploratory_fixed_baseline_inclusive_5x5/",
        "/scientific_validation/final_delta_fixed_5x5/",
    ]):
        root_priority = 0
    elif "/scientific_validation/final_grouping_sensitivity_5x5/" in norm:
        root_priority = 1
    elif "/scientific_validation/final_information_source_ablation_5x5/" in norm:
        root_priority = 2
    elif "/scientific_validation/final_bde_paired_ablation_5x5/" in norm:
        root_priority = 2
    elif "/scientific_validation/" in norm:
        root_priority = 3
    elif "/05_shap/" in norm:
        root_priority = 4
    else:
        root_priority = 5
    # Prefer the task-level summary over the directory-wide convenience summary.
    aggregate_penalty = 1 if norm.endswith("scientific_nested_summary_all.csv") else 0
    return root_priority, aggregate_penalty, norm


def table_s5(index: ResultIndex) -> pd.DataFrame:
    summaries = _all_result_files(index, "_nested_summary.csv") + index.all_named("scientific_nested_summary_all.csv")
    frame = _concat_with_source(dict.fromkeys(summaries))
    if frame.empty:
        return table_s4(index)

    if "selection_rule" in frame.columns:
        frame["selection_rule"] = frame["selection_rule"].replace(
            {
                (
                    "view/K/model/UL94 threshold selected by inner CV only; "
                    "outer fold used once for evaluation"
                ): (
                    "view/K frozen from development; "
                    "model and UL94 threshold selected by inner CV only; "
                    "outer fold used once for evaluation"
                )
            }
        )

    # The same scientific result may exist both as a task-level summary and in
    # scientific_nested_summary_all.csv, and SHAP folders may contain copied
    # nested summaries.  source_file therefore MUST NOT be part of the
    # scientific identity key.
    identity_cols = [
        "task", "task_type", "use_BDE", "split_strategy", "screening_mode",
        "feature_scope", "descriptor_mode", "use_V7_formula_features",
        "selection_scope", "configured_current_view", "configured_current_k",
        "outer_splits_requested", "outer_splits_completed", "inner_splits",
    ]
    subset = [c for c in identity_cols if c in frame.columns]

    # Make the retained source deterministic and traceable: prefer canonical
    # scientific_validation paths, then task-level files, then copied summaries.
    if "source_file" in frame.columns:
        priorities = frame["source_file"].map(_summary_source_priority)
        frame = frame.assign(
            _source_root_priority=[x[0] for x in priorities],
            _source_aggregate_penalty=[x[1] for x in priorities],
            _source_path_key=[x[2] for x in priorities],
        ).sort_values(
            ["_source_root_priority", "_source_aggregate_penalty", "_source_path_key"],
            kind="stable",
        )

    if subset:
        frame = frame.drop_duplicates(subset=subset, keep="first")

    drop_helper = [c for c in ["_source_root_priority", "_source_aggregate_penalty", "_source_path_key"] if c in frame]
    if drop_helper:
        frame = frame.drop(columns=drop_helper)

    sort_cols = [c for c in ["task", "use_BDE", "split_strategy", "screening_mode", "feature_scope"] if c in frame]
    return frame.sort_values(sort_cols, kind="stable") if sort_cols else frame

def table_s6(index: ResultIndex) -> pd.DataFrame:
    path = index.locate_any(
        ["BDE_selected_config_all_predictions.csv", "BDE_all_predictions.csv"],
        prefer=["07_bde", "selected_config"], avoid=["smoke"], optional=True,
        label="S6 BDE predictions",
    )
    if not path:
        return pd.DataFrame()
    frame = read_csv_auto(path)
    frame.insert(0, "source_file", str(path))
    true_col = next((c for c in ["BDE_true_kJ_mol", "y_true", "BDE_true"] if c in frame), None)
    pred_col = next((c for c in ["BDE_pred_kJ_mol", "y_pred", "BDE_pred"] if c in frame), None)
    if true_col and pred_col:
        frame["residual_kJ_mol"] = numeric(frame[pred_col]) - numeric(frame[true_col])
        frame["absolute_error_kJ_mol"] = frame["residual_kJ_mol"].abs()
    sort_col = next((c for c in ["absolute_error_kJ_mol", "abs_error_kJ_mol"] if c in frame), None)
    return frame.sort_values(sort_col, ascending=False) if sort_col else frame


def table_s7(index: ResultIndex) -> pd.DataFrame:
    """Collect only High/Medium cross-fold stable SHAP features.

    Prefer the explicit *_SHAP_stable_features_filtered.csv outputs generated by
    the SHAP workflow.  If they are unavailable, use a stability table only when
    it already contains an explicit stability_level column; never invent a
    stability class from an arbitrary threshold here.
    """
    frames = []
    task_rank = {task: i for i, task in enumerate(TASK_ORDER)}

    for task in TASK_ORDER:
        exact_name = f"{task}_SHAP_stable_features_filtered.csv"
        candidates = [
            p for p in index.csv_files
            if p.name == exact_name and "05_shap" in p.as_posix().lower()
        ]

        # Backward-compatible fallback for a project where the filtered export
        # was saved under another SHAP directory, but only accept files with an
        # explicit stability_level column below.
        if not candidates:
            candidates = [
                p for p in index.csv_files
                if task.lower() in p.as_posix().lower()
                and p.name in {
                    f"{task}_SHAP_stability_summary.csv",
                    f"{task}_SHAP_aggregate.csv",
                    f"{task}_SHAP_feature_stability.csv",
                }
            ]

        if not candidates:
            continue

        def _candidate_priority(path: Path) -> tuple[int, int, float]:
            norm = path.as_posix().lower()
            explicit_filtered = 0 if path.name == exact_name else 1
            canonical_area = 0 if ("final_core_fixed_baseline_inclusive_5x5/figures" in norm or "final_aux_fixed_baseline_inclusive_5x5/figures" in norm) else 1
            return explicit_filtered, canonical_area, -path.stat().st_mtime

        path = sorted(candidates, key=_candidate_priority)[0]
        frame = read_csv_auto(path)

        if "stability_level" not in frame.columns:
            # A raw stability summary without explicit classes is not sufficient
            # for a table named "stable SHAP features".
            continue

        level = frame["stability_level"].astype(str).str.strip().str.title()
        frame = frame[level.isin({"High", "Medium"})].copy()
        if frame.empty:
            continue
        frame["stability_level"] = level.loc[frame.index]

        if "task" not in frame.columns:
            frame.insert(0, "task", task)
        else:
            frame["task"] = task
        frame.insert(0, "source_file", str(path))
        frame.insert(1, "task_order", task_rank.get(task, 999))
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True, sort=False)
    sort_cols = [c for c in ["task_order", "stability_level", "mean_abs_shap_all_folds"] if c in out]
    if sort_cols:
        ascending = [True, True, False][:len(sort_cols)]
        out = out.sort_values(sort_cols, ascending=ascending, kind="stable")
    if "task_order" in out:
        out = out.drop(columns=["task_order"])
    return out.reset_index(drop=True)

def _true_pred_columns(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    true_col = next((c for c in ["y_true", "true_value", "observed", "target", "actual"] if c in frame), None)
    pred_col = next((c for c in ["y_pred", "predicted_value", "prediction", "predicted"] if c in frame), None)
    return true_col, pred_col


def table_s8(index: ResultIndex, top_n: int) -> pd.DataFrame:
    frames = []
    for task in TASK_ORDER:
        path = locate_task_file(index, task, "outer_predictions", use_bde=False, optional=True)
        if not path:
            continue
        frame = read_csv_auto(path)
        true_col, pred_col = _true_pred_columns(frame)
        if not true_col or not pred_col:
            continue
        out = frame.copy()
        out.insert(0, "source_file", str(path))
        if "task" not in out.columns:
            out.insert(1, "task", task)
        else:
            out["task"] = task
        if task == "UL94_V0":
            y = normalise_ul94(out[true_col])
            pred_raw = out[pred_col]
            yhat = normalise_ul94(pred_raw) if not pd.api.types.is_numeric_dtype(pred_raw) else numeric(pred_raw).round()
            out["error_indicator"] = (y != yhat).astype(float)
            prob_col = probability_column(out)
            if prob_col:
                probs = numeric(out[prob_col]).clip(1e-8, 1 - 1e-8)
                out["single_sample_log_loss"] = -(y * np.log(probs) + (1-y) * np.log(1-probs))
                sort_col = "single_sample_log_loss"
            else:
                sort_col = "error_indicator"
        else:
            out["residual"] = numeric(out[pred_col]) - numeric(out[true_col])
            out["absolute_error"] = out["residual"].abs()
            sort_col = "absolute_error"
        frames.append(out.sort_values(sort_col, ascending=False).head(top_n))
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def table_s9(index: ResultIndex) -> pd.DataFrame:
    """All 50 kW/m² formulation predictions from the final combined screen."""
    path = index.locate(
        "candidate_predictions_all_formulations.csv",
        prefer=["06_reversedesign", "combined_flux50"],
        avoid=["final_flux50", "sensitivity_flux35", "smoke", "2x2"],
        optional=True,
        label="S9 combined candidates flux50",
    )
    if not path:
        return pd.DataFrame()
    frame = read_csv_auto(path)
    frame.insert(0, "source_file", str(path))
    return frame


def table_s11(index: ResultIndex) -> pd.DataFrame:
    """Final candidates stable at both 35 and 50 kW/m² under the frozen rule."""
    path = index.locate(
        "final_priority_candidates_combined.csv",
        prefer=["06_reversedesign", "final_priority_combined"],
        avoid=["final_priority/", "smoke", "2x2"],
        optional=True,
        label="S11 final combined priority candidates",
    )
    if not path:
        return pd.DataFrame()
    frame = read_csv_auto(path)
    preferred = [
        "Final_priority_rank", "Candidate_ID", "Candidate_Name", "Source_Type", "Design_Family", "Synthesis_Level",
        "Candidate_rank_50", "Candidate_rank_35", "Pareto_rank_50", "Pareto_rank_35",
        "Loading_total_FR wt%_50", "Loading_total_FR wt%_35",
        "LOI_pred_50", "PHRR_pred_50", "PHRR_pred_35", "THR_pred_50",
        "V0_probability_50", "Tg_pred_50", "TS_pred_50",
        "max_similarity_50", "max_similarity_35",
    ]
    cols = [c for c in preferred if c in frame.columns]
    out = frame[cols].copy() if cols else frame.copy()
    out.insert(0, "source_file", str(path))
    return out

def table_s12_external_validation(index: ResultIndex) -> pd.DataFrame:
    """Return manuscript-facing external-validation metrics in a fixed schema.

    ExternalValidation has used a few column-name variants across project
    revisions (for example ``n`` vs ``n_external`` and
    ``Balanced_Accuracy`` vs ``balanced_accuracy``).  This normalises those
    aliases so Table S7A is complete and reproducible from code.
    """
    path = index.locate(
        "external_validation_metrics.csv",
        prefer=["10_externalvalidation", "final_fixed"],
        avoid=["smoke", "archive"],
        optional=True,
    )
    if path is None:
        return pd.DataFrame()
    frame = read_csv_auto(path).copy()

    def first_existing(*names: str) -> pd.Series:
        for name in names:
            if name in frame.columns:
                return frame[name]
        return pd.Series(np.nan, index=frame.index)

    task = first_existing("task", "Task").astype(str)
    out = pd.DataFrame(index=frame.index)
    out["task"] = task
    out["task_type"] = task.map(
        lambda x: "classification"
        if re.sub(r"[^A-Z0-9]", "", str(x).upper()) in {"UL94", "UL94V0"}
        else "regression"
    )
    out["n"] = pd.to_numeric(
        first_existing("n", "n_external", "n_samples", "sample_n"), errors="coerce"
    ).round().astype("Int64")
    out["MAE"] = pd.to_numeric(first_existing("MAE", "mae"), errors="coerce")
    out["RMSE"] = pd.to_numeric(first_existing("RMSE", "rmse"), errors="coerce")
    out["R2"] = pd.to_numeric(first_existing("R2", "r2", "R_squared"), errors="coerce")
    out["Spearman_rho"] = pd.to_numeric(first_existing(
        "Spearman_rho", "spearman_rho", "Spearman", "spearman", "rank_spearman", "spearman_corr"
    ), errors="coerce")
    out["PICP"] = pd.to_numeric(first_existing(
        "PICP", "picp", "prediction_interval_coverage", "interval_coverage", "coverage_probability"
    ), errors="coerce")
    out["Accuracy"] = pd.to_numeric(first_existing("Accuracy", "accuracy"), errors="coerce")
    out["Balanced_accuracy"] = pd.to_numeric(first_existing(
        "Balanced_Accuracy", "balanced_accuracy", "Balanced accuracy", "balanced_acc"
    ), errors="coerce")
    out["Macro_F1"] = pd.to_numeric(first_existing("Macro_F1", "macro_f1", "Macro-F1"), errors="coerce")
    out["ROC_AUC"] = pd.to_numeric(first_existing("ROC_AUC", "roc_auc", "ROC-AUC"), errors="coerce")

    # Preserve source provenance for the machine-readable audit path.
    out.insert(0, "source_file", str(path))
    return out.reset_index(drop=True)

def table_s13_external_structure_audit(index: ResultIndex) -> pd.DataFrame:
    path = index.locate(
        "external_molecule_identity_audit.csv",
        prefer=["10_externalvalidation", "final_fixed"],
        avoid=["smoke", "archive"],
        optional=True,
    )
    if path is None:
        return pd.DataFrame()
    frame = read_csv_auto(path).copy()
    out = frame.copy()
    out.insert(0, "source_file", str(path))
    return out


COMPACT_DATABASE_FIELDS = [
    "Record_ID", "FR_main", "FR_co", "Curing_Agent", "Preparation_Method", "Synergy_type",
    "SMILES_main", "SMILES_co", "SMILES_Curing_Agent", "Loading_total_FR wt%",
    "P_content wt%", "N_content wt%", "S_content wt%", "B_content wt%", "Si_content wt%",
    "Cure_Temp_Max", "LOI_Thickness_mm", "LOI", "EP_matrix_LOI", "Delta_LOI",
    "UL94_Thickness_mm", "UL94", "Cone_Thickness_mm", "Cone_flux_kW_m2",
    "PHRR_kw_㎡", "EP_matrix_PHRR", "Delta_PHRR", "THR_MJ_㎡", "EP_matrix_THR", "Delta_THR",
    "Tg_℃", "EP_matrix_Tg", "Char_yield_％_700C", "EP_matrix_CY", "Delta_CY",
    "TS_MPa", "EP_matrix_TS", "FS_MPa", "EP_matrix_FS",
    "FR_main_BDE_final_kJ_mol", "FR_main_BDE_available", "FR_main_BDE_source", "FR_main_BDE_low_confidence",
]


def table_s1_compact(df: pd.DataFrame) -> pd.DataFrame:
    full = table_s1(df)
    keep = [c for c in COMPACT_DATABASE_FIELDS if c in set(full["field"].astype(str))]
    order = {name: i + 1 for i, name in enumerate(keep)}
    out = full[full["field"].isin(keep)].copy()
    out["field_order"] = out["field"].map(order)
    return out.sort_values("field_order").reset_index(drop=True)


def table_s1a_dataset_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Manuscript/SI Table S1(A): task size, structural coverage, and targets."""
    rows = []
    targets = {
        "LOI": "LOI", "PHRR": "PHRR_kw_㎡", "THR": "THR_MJ_㎡",
        "UL94_V0": "UL94", "Tg": "Tg_℃", "TS_MPa": "TS_MPa",
    }
    for task in ["LOI", "PHRR", "THR", "UL94_V0", "Tg", "TS_MPa"]:
        col = targets[task]
        if col not in df.columns:
            continue
        if task == "UL94_V0":
            y = normalise_ul94(df[col])
            valid = y.notna()
            counts = y[valid].astype(int).value_counts().to_dict()
            summary = f"V-0={counts.get(1, 0)}; non-V-0={counts.get(0, 0)}"
            task_type = "Binary classification"
        else:
            values = numeric(df[col])
            valid = values.notna()
            v = values[valid]
            summary = f"{v.min():.2f}–{v.max():.2f}; median={v.median():.2f}"
            task_type = "Regression"
        sub = df.loc[valid].copy()
        main = sub.get("SMILES_main", pd.Series(index=sub.index, dtype=object)).map(canonical_smiles)
        co = sub.get("SMILES_co", pd.Series(index=sub.index, dtype=object)).map(canonical_smiles).fillna("NONE")
        pairs = (main.fillna("INVALID") + "||" + co).nunique()
        scaffolds = main.dropna().map(murcko_scaffold).dropna().nunique()
        rows.append({
            "Task": TASK_DISPLAY[task].replace(" V-0", ""),
            "Task type": task_type,
            "Valid records (n)": int(valid.sum()),
            "Unique main FRs": int(main.dropna().nunique()),
            "Unique main FR/synergist combinations": int(pairs),
            "Unique Murcko scaffolds": int(scaffolds),
            "References (n)": FINAL_REFERENCE_COUNTS[task],
            "Target range / median": summary,
        })
    return pd.DataFrame(rows)


def table_s1b_database_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Manuscript/SI Table S1(B), limited to the 43 paper-facing database fields."""
    out = table_s1_compact(df).copy()
    out = out.rename(columns={
        "field_order": "No.", "field": "Database field", "description": "Description",
        "unit_or_encoding": "Unit/encoding", "pandas_dtype": "Data type",
        "non_missing_n": "non missing n", "missing_n": "missing n",
        "missing_rate_percent": "missing rate", "unique_non_missing": "unique non missing",
    })
    return out

def table_s2a_configurations(config_path: Path, df: pd.DataFrame) -> pd.DataFrame:
    cfg = table_s2(config_path, df).copy()
    cfg = cfg[cfg["task"].astype(str).isin(TASK_ORDER)].copy()
    out = pd.DataFrame({
        "Task": cfg["task"].map(TASK_DISPLAY),
        "Task type": cfg.get("task_type", "").astype(str).str.lower(),
        "Frozen view": cfg.get("frozen_view", "").astype(str).str.replace("_", " ", regex=False),
        "K": cfg.get("frozen_k", np.nan),
        "group rule": cfg.get("group_rule", "").map(GROUP_DISPLAY).fillna(cfg.get("group_rule", "")),
        "Formal n": pd.to_numeric(cfg.get("valid_target_n", np.nan), errors="coerce").astype("Int64"),
        "main metric": cfg["task"].map(lambda x: "Macro-F1" if x == "UL94_V0" else "R²"),
    })
    out["K"] = out["K"].where(out["K"].notna(), "all").replace({"ALL": "all", "All": "all"})
    return out.reset_index(drop=True)

def table_s2b_model_space() -> pd.DataFrame:
    full = table_s3().copy()
    cols = [c for c in ["task_type", "model", "implementation", "frozen_parameters"] if c in full]
    out = full[cols].copy()
    out = out.rename(columns={
        "task_type": "Task type", "model": "Model", "implementation": "Implementation",
        "frozen_parameters": "Fixed settings",
    })
    out["Task type"] = out["Task type"].astype(str).str.lower()
    out["Fixed settings"] = (out["Fixed settings"].fillna("").astype(str)
        .str.replace("_", " ", regex=False)
        .replace("", "—"))
    return out.reset_index(drop=True)

def _canonical_metric_row_priority(frame: pd.DataFrame) -> pd.Series:
    src = frame.get("source_file", pd.Series("", index=frame.index)).astype(str).str.replace("\\", "/").str.lower()
    score = pd.Series(0.0, index=frame.index)
    score += src.str.contains("/scientific_validation/final_core_fixed_baseline_inclusive_5x5/", regex=False) * 100
    score += src.str.contains("/scientific_validation/final_aux_fixed_baseline_inclusive_5x5/", regex=False) * 100
    score += src.str.contains("/scientific_validation/final_exploratory_fixed_baseline_inclusive_5x5/", regex=False) * 100
    score += src.str.contains("/scientific_validation/final_delta_fixed_5x5/", regex=False) * 100
    score -= src.str.contains("conditions_only|molecular_only|smoke|2x2", regex=True) * 500
    return score


def _format_model_frequency(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip().replace("；", ";").replace("（", "(").replace("）", ")")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return "; ".join(f"{name}({int(count)})" for name, count in parsed)
    except Exception:
        pass
    text = re.sub(r"\s*;\s*", "; ", text)
    return text


def table_s3_compact_validation(index: ResultIndex) -> pd.DataFrame:
    full = table_s5(index).copy()
    if full.empty:
        return full
    g = full[full.get("task", pd.Series(index=full.index, dtype=object)).astype(str).isin(TASK_ORDER)].copy()
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
    g = g.assign(_priority=_canonical_metric_row_priority(g))
    order = {t: i for i, t in enumerate(TASK_ORDER)}
    g["_task_order"] = g["task"].map(order)
    g = g.sort_values(["_task_order", "_priority"], ascending=[True, False], kind="stable").drop_duplicates("task", keep="first")

    rows = []
    for _, r in g.iterrows():
        task = str(r["task"])
        is_cls = task == "UL94_V0"
        n = pd.to_numeric(pd.Series([r.get("n_valid_formal_rows")]), errors="coerce").iloc[0]
        if pd.isna(n):
            # fallback to frozen SI counts
            n = {"LOI":599,"PHRR":406,"THR":397,"UL94_V0":599,"Tg":357,"Char_yield":196,"TS_MPa":256,"FS_MPa":228,"Delta_LOI":468,"Delta_PHRR":289,"Delta_THR":283,"Delta_CY":148}.get(task, np.nan)
        k = r.get("configured_current_k", np.nan)
        k = "ALL" if pd.isna(k) else int(float(k))
        rows.append({
            "Task": TASK_DISPLAY.get(task, task),
            "n": int(n) if pd.notna(n) else "",
            "Frozen view": str(r.get("configured_current_view", "")).replace("_", " "),
            "Frozen K": k,
            "R²": "--" if is_cls else (f"{float(r.get('outer_R2_mean')):.3f}" if pd.notna(r.get('outer_R2_mean')) else "--"),
            "SD": "--" if is_cls else (f"{float(r.get('outer_R2_std')):.3f}" if pd.notna(r.get('outer_R2_std')) else "--"),
            "RMSE": "--" if is_cls else (f"{float(r.get('outer_RMSE_mean')):.3f}" if pd.notna(r.get('outer_RMSE_mean')) else "--"),
            "MAE": "--" if is_cls else (f"{float(r.get('outer_MAE_mean')):.3f}" if pd.notna(r.get('outer_MAE_mean')) else "--"),
            "Macro-F1": f"{float(r.get('outer_Macro_F1_mean')):.3f}" if is_cls and pd.notna(r.get('outer_Macro_F1_mean')) else "--",
            "Accuracy": f"{float(r.get('outer_Accuracy_mean')):.3f}" if is_cls and pd.notna(r.get('outer_Accuracy_mean')) else "--",
            "Balanced accuracy": f"{float(r.get('outer_Balanced_Accuracy_mean')):.3f}" if is_cls and pd.notna(r.get('outer_Balanced_Accuracy_mean')) else "--",
            "ROC-AUC": f"{float(r.get('outer_ROC_AUC_mean')):.3f}" if is_cls and pd.notna(r.get('outer_ROC_AUC_mean')) else "--",
            "Selected model (fold count)": _format_model_frequency(r.get("selected_model_frequency", "")),
        })
    return pd.DataFrame(rows)

def _normalise_bond_type(series: pd.Series) -> pd.Series:
    return (series.astype(str).str.strip().str.upper()
            .str.replace("–", "-", regex=False).str.replace("—", "-", regex=False)
            .str.replace("−", "-", regex=False))


def table_s4_bde_summary(index: ResultIndex) -> pd.DataFrame:
    full = table_s6(index).copy()
    if full.empty:
        return full
    bond_col = next((c for c in ["Bond_Type", "bond_type", "pred_Bond_Type"] if c in full.columns), None)
    if bond_col is None:
        return pd.DataFrame()
    full["_bond"] = _normalise_bond_type(full[bond_col])
    full = full[full["_bond"].isin(["P-C", "P-N"])].copy()
    true_col = next((c for c in ["BDE_true_kJ_mol", "y_true", "BDE_true"] if c in full), None)
    pred_col = next((c for c in ["BDE_pred_kJ_mol", "y_pred", "BDE_pred"] if c in full), None)
    err_col = next((c for c in ["absolute_error_kJ_mol", "abs_error_kJ_mol"] if c in full), None)
    id_col = next((c for c in ["ID", "BDE_ID", "Record_ID", "Canonical_SMILES"] if c in full), None)
    rows = []
    for bond in ["P-C", "P-N"]:
        d = full[full["_bond"].eq(bond)].copy()
        err = numeric(d[err_col]).dropna() if err_col else pd.Series(dtype=float)
        rows.append({
            "Bond type": bond,
            "Prediction rows": int(len(d)),
            "Unique samples": int(d[id_col].nunique(dropna=True)) if id_col else np.nan,
            "Mean observed BDE (kJ mol^-1)": round(float(numeric(d[true_col]).mean()), 3) if true_col else np.nan,
            "Mean predicted BDE (kJ mol^-1)": round(float(numeric(d[pred_col]).mean()), 3) if pred_col else np.nan,
            "MAE (kJ mol^-1)": round(float(err.mean()), 3) if len(err) else np.nan,
            "Median AE (kJ mol^-1)": round(float(err.median()), 3) if len(err) else np.nan,
            "95th-percentile AE (kJ mol^-1)": round(float(err.quantile(.95)), 3) if len(err) else np.nan,
        })
    return pd.DataFrame(rows)

def table_s5_compact_shap(index: ResultIndex) -> pd.DataFrame:
    full = table_s7(index).copy()
    if full.empty:
        return full
    limits = {"LOI": 8, "PHRR": 8, "THR": 8, "UL94_V0": 8, "Tg": 5, "TS_MPa": 5}
    blocks = []
    for task, n in limits.items():
        d = full[full["task"].astype(str).eq(task)].copy()
        if d.empty:
            continue
        level_rank = d.get("stability_level", pd.Series("Medium", index=d.index)).map({"High": 0, "Medium": 1}).fillna(2)
        d = d.assign(_level_rank=level_rank)
        sort_cols, ascending = ["_level_rank"], [True]
        if "mean_abs_shap_all_folds" in d:
            sort_cols.append("mean_abs_shap_all_folds"); ascending.append(False)
        d = d.sort_values(sort_cols, ascending=ascending, kind="stable").head(n)
        blocks.append(d)
    if not blocks:
        return pd.DataFrame()
    d = pd.concat(blocks, ignore_index=True, sort=False)
    out = pd.DataFrame({
        "Task": d["task"].map(TASK_DISPLAY).fillna(d["task"]),
        "Feature": d.get("feature", ""),
        "Feature group": d.get("feature_group", ""),
        "Mean |SHAP|": pd.to_numeric(d.get("mean_abs_shap_all_folds", np.nan), errors="coerce").round(3),
        "Top-20 frequency": pd.to_numeric(d.get("top20_frequency", np.nan), errors="coerce").round(3),
        "Direction consistency": pd.to_numeric(d.get("direction_consistency", np.nan), errors="coerce").round(3),
        "Mean rank": pd.to_numeric(d.get("mean_rank_all_folds", np.nan), errors="coerce").round(1),
        "Stability": d.get("stability_level", ""),
    })
    return out

def _count_unique_candidates(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    col = next((c for c in ["Candidate_ID", "candidate_id", "Canonical_SMILES", "SMILES_main"] if c in frame), None)
    return int(frame[col].nunique(dropna=True)) if col else int(len(frame))


def table_s6a_screening_summary(index: ResultIndex) -> pd.DataFrame:
    all_pred = table_s9(index)
    formal_n = _count_unique_candidates(all_pred)
    if formal_n == 0:
        p_formal = index.locate(
            "candidate_best_per_molecule.csv", prefer=["06_reversedesign", "combined_flux50"],
            avoid=["final_flux50", "sensitivity_flux35", "smoke", "2x2", "designed_only"],
            optional=True, label="compact SI formal screening pool",
        )
        if p_formal:
            formal_n = _count_unique_candidates(read_csv_auto(p_formal))
    stability_path = index.locate(
        "flux35_50_rank_stability_combined.csv", prefer=["06_reversedesign", "final_priority_combined"],
        avoid=["smoke", "2x2"], optional=True, label="compact SI screening stability",
    )
    stability = read_csv_auto(stability_path) if stability_path else pd.DataFrame()
    eligible_both = _count_unique_candidates(stability)

    def _best(flux_token: str) -> pd.DataFrame:
        p = index.locate(
            "candidate_best_per_molecule.csv", prefer=["06_reversedesign", flux_token],
            avoid=["smoke", "2x2", "designed_only"], optional=True,
            label=f"compact SI {flux_token} best-per-molecule",
        )
        return read_csv_auto(p) if p else pd.DataFrame()

    f50, f35 = _best("combined_flux50"), _best("sensitivity_flux35")
    pareto50 = int(numeric(f50["pareto_flag"]).fillna(0).astype(bool).sum()) if "pareto_flag" in f50 else np.nan
    pareto35 = int(numeric(f35["pareto_flag"]).fillna(0).astype(bool).sum()) if "pareto_flag" in f35 else np.nan
    common_pareto = int(stability["Pareto_both"].fillna(False).astype(bool).sum()) if "Pareto_both" in stability else np.nan
    final_n = _count_unique_candidates(table_s11(index))
    return pd.DataFrame([
        ("Formal screening candidates", formal_n),
        ("Eligible in both heat-flux scenarios", eligible_both),
        ("Pareto-optimal at 50 kW m^-2", pareto50),
        ("Pareto-optimal at 35 kW m^-2", pareto35),
        ("Common cross-scenario Pareto candidates", common_pareto),
        ("Final priority candidates", final_n),
    ], columns=["Screening stage", "Candidates (n)"])

def table_s6b_final_candidates(index: ResultIndex) -> pd.DataFrame:
    d = table_s11(index).copy()
    if d.empty:
        return d
    out = pd.DataFrame({
        "Rank": d.get("Final_priority_rank", np.nan),
        "Candidate ID": d.get("Candidate_ID", ""),
        "Candidate name": d.get("Candidate_Name", ""),
        "Design family": d.get("Design_Family", ""),
        "Loading (wt%)": d.get("Loading_total_FR wt%_50", np.nan),
        "Pred. LOI (%)": d.get("LOI_pred_50", np.nan),
        "Pred. PHRR (kW m^-2)": d.get("PHRR_pred_50", np.nan),
        "Pred. THR (MJ m^-2)": d.get("THR_pred_50", np.nan),
        "Pred. V-0 probability": d.get("V0_probability_50", np.nan),
        "Pred. Tg (°C)": d.get("Tg_pred_50", np.nan),
        "Pred. TS (MPa)": d.get("TS_pred_50", np.nan),
        "Smax": d.get("max_similarity_50", np.nan),
    })
    out["Rank"] = pd.to_numeric(out["Rank"], errors="coerce").astype("Int64")
    for col in ["Loading (wt%)", "Pred. LOI (%)", "Pred. PHRR (kW m^-2)", "Pred. THR (MJ m^-2)", "Pred. Tg (°C)", "Pred. TS (MPa)", "Smax"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").round(3)
    out["Pred. V-0 probability"] = pd.to_numeric(out["Pred. V-0 probability"], errors="coerce").round(3)
    return out.sort_values("Rank", kind="stable").reset_index(drop=True)

def table_s7a_external_performance(index: ResultIndex) -> pd.DataFrame:
    d = table_s12_external_validation(index).copy()
    if d.empty:
        return d
    d = d[d["task"].astype(str).isin(["LOI", "PHRR", "THR", "UL94_V0", "TS_MPa"])].copy()
    order = {"LOI":0,"PHRR":1,"THR":2,"UL94_V0":3,"TS_MPa":4}
    d["_order"] = d["task"].map(order)
    d = d.sort_values("_order", kind="stable")
    out = pd.DataFrame({
        "Task": d["task"].map(TASK_DISPLAY),
        "Task type": d["task_type"],
        "n": d["n"],
        "MAE": d["MAE"], "RMSE": d["RMSE"], "R²": d["R2"],
        "Spearman ρ": d["Spearman_rho"], "PICP": d["PICP"],
        "Accuracy": d["Accuracy"], "Balanced accuracy": d["Balanced_accuracy"],
        "Macro-F1": d["Macro_F1"], "ROC-AUC": d["ROC_AUC"],
    })
    for c in ["MAE","RMSE","R²","Spearman ρ","PICP","Accuracy","Balanced accuracy","Macro-F1","ROC-AUC"]:
        vals = pd.to_numeric(out[c], errors="coerce")
        out[c] = vals.map(lambda x: "--" if pd.isna(x) else f"{x:.3f}")
    return out.reset_index(drop=True)

def table_s7b_external_structure(index: ResultIndex) -> pd.DataFrame:
    d = table_s13_external_structure_audit(index).copy()
    if d.empty:
        return d
    name_col = next((c for c in ["FR_main", "External_FR", "External FR"] if c in d.columns), None)
    if name_col is None:
        return pd.DataFrame()
    out = pd.DataFrame({
        "External FR": d[name_col],
        "Molecular formula": d.get("Molecular_formula", d.get("Molecular formula", "")),
        "Exact training match": d.get("V4_exact_canonical_match", d.get("Exact training match", "")),
        "Scaffold seen": d.get("V4_scaffold_seen", d.get("Scaffold seen", "")),
        "Smax": pd.to_numeric(d.get("Max_Tanimoto_to_V4", d.get("Smax", np.nan)), errors="coerce").round(3),
        "Nearest FR": d.get("Nearest_V4_FR", d.get("Nearest FR", "")),
        "AD zone": d.get("AD_category", d.get("AD zone", "")).astype(str).str.strip().str.lower().map({"in-domain": "In-domain", "caution": "Caution", "extrapolation": "Extrapolation"}).fillna(d.get("AD_category", d.get("AD zone", ""))),
        "Identity audit level": d.get("External_identity_level", d.get("Identity audit level", "")),
    })
    out["Reference"] = out["External FR"].map(EXTERNAL_REFERENCE_MAP).fillna("")
    order = {k:i for i,k in enumerate(EXTERNAL_REFERENCE_MAP)}
    out["_order"] = out["External FR"].map(order).fillna(999)
    return out.sort_values("_order", kind="stable").drop(columns="_order").reset_index(drop=True)

def _save(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # Remove legacy numbered/full-SI outputs so the visible numbered set is
    # exactly S1(A-B) through S7(A-B), matching the frozen SI manuscript.
    for pattern in ["TableS*.csv", "Table S*.csv", "Data_*.csv", "Table_main_*.csv", "paper_table_manifest.csv", "selected_input_files.csv", "run_config.json"]:
        for old in args.output.glob(pattern):
            try:
                old.unlink()
            except OSError:
                pass

    df = read_csv_auto(args.data)
    index = ResultIndex(args.results_root)
    manifest: list[dict[str, object]] = []

    machine_jobs = [
        ("Data_full_database_dictionary.csv", lambda: table_s1(df), "complete database dictionary"),
        ("Data_outer_fold_model_selection.csv", lambda: table_s4(index), "complete fold-level model selections"),
        ("Data_complete_5x5_metrics.csv", lambda: table_s5(index), "complete nested-validation summaries"),
        ("Data_all_BDE_prediction_errors.csv", lambda: table_s6(index), "complete repeated-OOF BDE prediction rows"),
        ("Data_all_stable_SHAP_features.csv", lambda: table_s7(index), "all High/Medium stable SHAP features"),
        ("Data_outer_test_error_cases.csv", lambda: table_s8(index, args.top_errors), "representative outer-test error rows"),
        ("Data_all_candidate_predictions.csv", lambda: table_s9(index), "all 50 kW m^-2 formulation predictions"),
    ]
    for filename, factory, note in machine_jobs:
        try:
            frame = factory()
            if frame.empty:
                print(f"[SKIP] DATA: {filename}: {note}")
                continue
            _save(frame, args.output / filename)
            print(f"[OK] DATA: {filename} ({len(frame)} rows)")
        except Exception as exc:
            print(f"[WARN] DATA: {filename}: {type(exc).__name__}: {exc}")

    jobs = [
        ("S1A", "TableS1A_dataset_size_structural_coverage_and_target_distributions.csv", lambda: table_s1a_dataset_overview(df), "Dataset size, structural coverage, and target distributions"),
        ("S1B", "TableS1B_database_variables_and_raw_field_availability.csv", lambda: table_s1b_database_fields(df), "Database variables and raw-field availability"),
        ("S2A", "TableS2A_frozen_task_configurations.csv", lambda: table_s2a_configurations(args.task_config, df), "Frozen task-specific configurations"),
        ("S2B", "TableS2B_candidate_model_space.csv", table_s2b_model_space, "Candidate models and preprocessing"),
        ("S3", "TableS3_strict_5x5_validation_performance.csv", lambda: table_s3_compact_validation(index), "Strict 5x5 grouped nested-validation performance"),
        ("S4", "TableS4_BDE_PC_PN_summary.csv", lambda: table_s4_bde_summary(index), "P-C/P-N BDE summary; complete 1,195-row table in Data_all_BDE_prediction_errors.csv"),
        ("S5", "TableS5_stable_SHAP_features_compact.csv", lambda: table_s5_compact_shap(index), "Stable SHAP features (High/Medium)"),
        ("S6A", "TableS6A_screening_summary.csv", lambda: table_s6a_screening_summary(index), "Screening-stage summary"),
        ("S6B", "TableS6B_final_19_priority_candidates.csv", lambda: table_s6b_final_candidates(index), "Final 19 cross-flux priority candidates"),
        ("S7A", "TableS7A_external_validation_performance.csv", lambda: table_s7a_external_performance(index), "Independent external-validation performance"),
        ("S7B", "TableS7B_external_structure_AD_audit.csv", lambda: table_s7b_external_structure(index), "External structure/applicability-domain audit; manuscript references [43]-[48]"),
    ]

    for table, filename, factory, note in jobs:
        try:
            frame = factory()
            if frame.empty:
                manifest.append({"table": table, "file": filename, "status": "skipped", "rows": 0, "note": note})
                print(f"[SKIP] {table}: {note}")
                continue
            _save(frame, args.output / filename)
            manifest.append({"table": table, "file": filename, "status": "generated", "rows": len(frame), "note": note})
            print(f"[OK] {table}: {filename} ({len(frame)} rows)")
        except Exception as exc:
            manifest.append({"table": table, "file": filename, "status": "error", "rows": 0, "note": f"{type(exc).__name__}: {exc}"})
            print(f"[ERROR] {table}: {exc}")

    pd.DataFrame(manifest).to_csv(args.output / "paper_table_manifest.csv", index=False, encoding="utf-8-sig")
    index.save_manifest(args.output / "selected_input_files.csv")
    write_json(args.output / "run_config.json", vars(args))
    print(f"\n[DONE] Frozen Supplementary Tables S1(A-B)-S7(A-B): {args.output}")



if __name__ == "__main__":
    main()
