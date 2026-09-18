# -*- coding: utf-8 -*-
"""Audit whether the corrected V5 results required by manuscript figures/tables exist.

This audit checks files and outer-test prediction integrity.  It does not run
models and does not treat missing future virtual-screening results as data.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from paper_utils import (
    CORE_REGRESSION_TASKS,
    ROOT,
    ResultIndex,
    locate_task_file,
    read_csv_auto,
    write_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "data" / "DOPO_EP_new_with_BDE.csv")
    p.add_argument("--results-root", type=Path, default=ROOT / "results")
    p.add_argument("--output", type=Path, default=ROOT / "results" / "09_PaperFigures" / "audit")
    p.add_argument("--strict", action="store_true", help="Return non-zero when any main-text prerequisite is missing")
    return p.parse_args()


def _add(rows: list[dict[str, object]], item: str, status: str, path: Path | None = None, note: str = "", command: str = "") -> None:
    rows.append({"item": item, "status": status, "path": str(path) if path else "", "note": note, "recommended_command": command})


def _locate_any(index: ResultIndex, names: list[str], prefer: list[str]) -> Path | None:
    return index.locate_any(names, prefer=prefer, avoid=["smoke", "2x2"], optional=True)


def _normalize_protocol_value(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.lower()


def _protocol_audit(path: Path, task: str, frame: pd.DataFrame) -> dict[str, object]:
    """Verify that a manuscript-facing prediction file uses the frozen FINAL protocol.

    Prefer protocol columns embedded in the prediction CSV.  For legacy files,
    use the sibling nested-summary CSV when available.  Missing protocol
    metadata is treated as unverified rather than inferred from a directory name.
    """
    wanted = ["split_strategy", "screening_mode", "row_policy", "selection_scope", "use_BDE"]
    protocol: dict[str, str] = {}

    for column in wanted:
        if column in frame.columns:
            values = [
                _normalize_protocol_value(v)
                for v in frame[column].dropna().unique().tolist()
                if _normalize_protocol_value(v)
            ]
            if len(set(values)) == 1:
                protocol[column] = values[0]
            elif len(set(values)) > 1:
                protocol[column] = "MULTIPLE:" + ",".join(sorted(set(values)))

    summary_path = path.with_name(f"{task}_nested_summary.csv")
    if summary_path.exists():
        try:
            summary = read_csv_auto(summary_path)
            if not summary.empty:
                first = summary.iloc[0]
                for column in wanted:
                    if not protocol.get(column) and column in summary.columns:
                        protocol[column] = _normalize_protocol_value(first[column])
        except Exception:
            pass

    expected = {
        "split_strategy": "molecule",
        "screening_mode": "formulation",
        "row_policy": "baseline_inclusive",
        "selection_scope": "fixed",
        "use_BDE": "0",
    }
    missing = [key for key in expected if not protocol.get(key)]
    mismatches = [
        f"{key}={protocol.get(key)!r} (expected {value!r})"
        for key, value in expected.items()
        if protocol.get(key) and protocol.get(key) != value
    ]
    if missing:
        status = "FAIL_UNVERIFIED_PROTOCOL"
        note = "missing protocol metadata: " + ", ".join(missing)
    elif mismatches:
        status = "FAIL_PROTOCOL_MISMATCH"
        note = "; ".join(mismatches)
    else:
        status = "PASS"
        note = "frozen FINAL fixed baseline-inclusive protocol verified"

    return {
        "protocol_status": status,
        "protocol_note": note,
        **{f"protocol_{key}": protocol.get(key, "") for key in wanted},
    }


def _prediction_audit(path: Path, task: str) -> dict[str, object]:
    frame = read_csv_auto(path)
    fold_col = next((c for c in ["outer_fold", "fold", "test_fold"] if c in frame), None)
    id_col = next((c for c in ["Record_ID", "sample_index", "row_index", "original_index", "sample_id"] if c in frame), None)
    folds = int(frame[fold_col].nunique()) if fold_col else np.nan
    duplicates = int(frame[id_col].duplicated().sum()) if id_col else np.nan
    missing_ids = int(frame[id_col].isna().sum()) if id_col else np.nan
    true_col = next((c for c in ["y_true", "true_value", "observed", "target", "actual"] if c in frame), None)
    pred_col = next((c for c in ["y_pred", "predicted_value", "prediction", "predicted"] if c in frame), None)
    valid_pairs = int((pd.to_numeric(frame[true_col], errors="coerce").notna() & pd.to_numeric(frame[pred_col], errors="coerce").notna()).sum()) if true_col and pred_col else 0

    if id_col is None:
        integrity_status = "FAIL_NO_STABLE_ID"
    elif missing_ids > 0:
        integrity_status = "FAIL_MISSING_ID"
    elif duplicates > 0:
        integrity_status = "FAIL_DUPLICATE_ID"
    elif valid_pairs != len(frame):
        integrity_status = "FAIL_MISSING_PREDICTION"
    elif folds != 5:
        integrity_status = "CHECK_OUTER_FOLDS"
    else:
        integrity_status = "PASS"

    protocol = _protocol_audit(path, task, frame)
    return {
        "task": task,
        "file": str(path),
        "rows": len(frame),
        "outer_fold_count": folds,
        "sample_id_column": id_col or "",
        "missing_sample_ids": missing_ids,
        "duplicate_sample_ids": duplicates,
        "valid_prediction_pairs": valid_pairs,
        "integrity_status": integrity_status,
        **protocol,
    }



def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _expected_rows_from_predictions(pred_rows: list[dict[str, object]]) -> dict[str, int]:
    expected: dict[str, int] = {}
    for row in pred_rows:
        task = str(row.get("task", ""))
        try:
            n = int(row.get("rows", 0))
        except Exception:
            continue
        if task and n > 0:
            expected[task] = n
    return expected


def _audit_null_protocol(path: Path, expected_rows: dict[str, int]) -> tuple[str, str]:
    required = ["LOI", "PHRR", "THR", "UL94_V0"]
    if not path.exists():
        return "MISSING", "FINAL null-test summary not found"
    try:
        frame = read_csv_auto(path)
    except Exception as exc:
        return "PROTOCOL_MISMATCH", f"Could not read null-test summary: {exc}"

    problems: list[str] = []
    tasks = set(frame.get("task", pd.Series(dtype=str)).astype(str))
    missing = [task for task in required if task not in tasks]
    if missing:
        problems.append("missing tasks=" + ",".join(missing))

    for task in required:
        row = frame[frame.get("task", pd.Series(dtype=str)).astype(str).eq(task)]
        if row.empty:
            continue
        first = row.iloc[0]
        policy = str(first.get("row_policy", "")).strip().lower()
        if policy != "baseline_inclusive":
            problems.append(f"{task}: row_policy={policy or 'missing'}")
        try:
            n_valid = int(first.get("n_valid_formal_rows"))
        except Exception:
            n_valid = -1
        expected = expected_rows.get(task)
        if expected is not None and n_valid != expected:
            problems.append(f"{task}: n_valid={n_valid}, expected={expected}")
        try:
            n_perm = int(first.get("n_independent_permutation_replicates"))
        except Exception:
            n_perm = -1
        if n_perm != 999:
            problems.append(f"{task}: permutation_replicates={n_perm}, expected=999")
        if not _as_bool(first.get("neat_fr_feature_masking", False)):
            problems.append(f"{task}: neat-EP FR-feature masking not verified")
        if task in {"LOI", "PHRR", "THR"} and not _as_bool(first.get("matching_baseline_leakage_guard", False)):
            problems.append(f"{task}: matching EP baseline leakage guard not verified")
        p_value_unit = str(first.get("p_value_unit", "")).strip().lower()
        if "complete permutation replicate aggregated across outer folds" not in p_value_unit:
            problems.append(f"{task}: p-value unit not verified")

    if problems:
        return "PROTOCOL_MISMATCH", "; ".join(problems[:12])
    counts = ", ".join(f"{t}={expected_rows.get(t, '?')}" for t in required)
    return "READY", f"FINAL baseline-inclusive null test verified; 999 complete permutations; rows: {counts}"


def _audit_learning_curve_protocol(
    summary_path: Path,
    all_results_path: Path,
    expected_rows: dict[str, int],
) -> tuple[str, str]:
    required = ["LOI", "PHRR", "THR", "UL94_V0", "Tg", "TS_MPa"]
    if not summary_path.exists() or not all_results_path.exists():
        return "MISSING", "FINAL learning-curve summary/all-results file not found"
    try:
        summary = read_csv_auto(summary_path)
        all_results = read_csv_auto(all_results_path)
    except Exception as exc:
        return "PROTOCOL_MISMATCH", f"Could not read learning-curve outputs: {exc}"

    problems: list[str] = []
    tasks = set(all_results.get("task", pd.Series(dtype=str)).astype(str))
    missing = [task for task in required if task not in tasks]
    if missing:
        problems.append("missing tasks=" + ",".join(missing))

    expected_fractions = {0.2, 0.4, 0.6, 0.8, 1.0}
    for task in required:
        sub = all_results[all_results.get("task", pd.Series(dtype=str)).astype(str).eq(task)]
        if sub.empty:
            continue
        policies = {str(v).strip().lower() for v in sub.get("row_policy", pd.Series(dtype=str)).dropna().unique()}
        if policies != {"baseline_inclusive"}:
            problems.append(f"{task}: row_policy={sorted(policies) if policies else 'missing'}")
        vals = pd.to_numeric(sub.get("n_valid_formal_rows", pd.Series(dtype=float)), errors="coerce").dropna().unique()
        n_valid = int(vals[0]) if len(vals) == 1 else -1
        expected = expected_rows.get(task)
        if expected is not None and n_valid != expected:
            problems.append(f"{task}: n_valid={n_valid}, expected={expected}")
        mask_values = {_as_bool(v) for v in sub.get("neat_fr_feature_masking", pd.Series(dtype=object)).dropna().unique()}
        if mask_values != {True}:
            problems.append(f"{task}: neat-EP FR-feature masking not verified")
        if task in {"LOI", "PHRR", "THR", "Tg", "TS_MPa"}:
            # All listed regression tasks have task-matching EP_matrix baselines.
            guard_values = {_as_bool(v) for v in sub.get("matching_baseline_leakage_guard", pd.Series(dtype=object)).dropna().unique()}
            if guard_values != {True}:
                problems.append(f"{task}: matching EP baseline leakage guard not verified")
        fractions = set(pd.to_numeric(sub.get("train_fraction", pd.Series(dtype=float)), errors="coerce").dropna().round(6).tolist())
        if fractions != expected_fractions:
            problems.append(f"{task}: fractions={sorted(fractions)}")
        repeats = pd.to_numeric(sub.get("repeat", pd.Series(dtype=float)), errors="coerce").dropna().nunique()
        if int(repeats) != 5:
            problems.append(f"{task}: repeats={int(repeats)}, expected=5")

    summary_tasks = set(summary.get("task", pd.Series(dtype=str)).astype(str))
    if not set(required).issubset(summary_tasks):
        problems.append("learning_curve_summary missing required tasks")

    if problems:
        return "PROTOCOL_MISMATCH", "; ".join(problems[:12])
    counts = ", ".join(f"{t}={expected_rows.get(t, '?')}" for t in required)
    return "READY", f"FINAL baseline-inclusive learning curves verified; 5 repeats × 5 fractions; rows: {counts}"

def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    index = ResultIndex(args.results_root)
    rows: list[dict[str, object]] = []
    pred_rows: list[dict[str, object]] = []

    if args.data.exists():
        df = read_csv_auto(args.data)
        _add(rows, "Dataset", "READY", args.data, f"rows={len(df)}, columns={len(df.columns)}")
    else:
        _add(rows, "Dataset", "MISSING", args.data, "Primary database not found")

    # Main Figure 2 is database-only.
    _add(rows, "Fig2 dataset composition and chemical space", "READY" if args.data.exists() else "MISSING", args.data)

    missing_fig3 = []
    incompatible_fig3 = []
    for task in CORE_REGRESSION_TASKS:
        pred = locate_task_file(index, task, "outer_predictions", use_bde=False, optional=True)
        metrics = locate_task_file(index, task, "outer_fold_metrics", use_bde=False, optional=True)
        if pred:
            audit = _prediction_audit(pred, task)
            pred_rows.append(audit)
            if str(audit.get("protocol_status", "")).startswith("FAIL"):
                incompatible_fig3.append(task)
        if not pred or not metrics:
            missing_fig3.append(task)
    fig3_status = "MISSING" if missing_fig3 else "PROTOCOL_MISMATCH" if incompatible_fig3 else "READY"
    fig3_note = (
        "missing: " + ", ".join(missing_fig3) if missing_fig3
        else "incompatible/unverified protocol: " + ", ".join(incompatible_fig3) if incompatible_fig3
        else "five pooled outer-test folds available; frozen FINAL fixed baseline-inclusive protocol verified"
    )
    _add(
        rows, "Fig3 outer-test predicted vs observed", fig3_status,
        note=fig3_note,
        command="python -u run.py nested --tasks LOI,PHRR,THR,Tg,TS_MPa --bde without --outer 5 --inner 5 --results results/scientific_validation/FINAL_core_fixed_baseline_inclusive_5x5",
    )

    split = args.results_root / "scientific_validation" / "FINAL_grouping_sensitivity_5x5" / "scientific_nested_summary_all.csv"
    split_ready = split.exists()
    if split_ready:
        try:
            split_df = read_csv_auto(split)
            required_pairs = {(task, strategy) for task in ["LOI", "PHRR", "THR", "UL94_V0"] for strategy in ["molecule", "scaffold", "reference"]}
            observed_pairs = set(zip(split_df.get("task", pd.Series(dtype=str)).astype(str), split_df.get("split_strategy", pd.Series(dtype=str)).astype(str)))
            split_ready = required_pairs.issubset(observed_pairs)
        except Exception:
            split_ready = False
    core_metrics_ready = all(locate_task_file(index, t, "outer_fold_metrics", use_bde=False, optional=True) for t in ["LOI", "PHRR", "THR", "UL94_V0"])
    _add(
        rows, "Fig4 fold stability and grouping sensitivity", "READY" if core_metrics_ready and split_ready else "MISSING", split if split.exists() else None,
        note="FINAL_grouping_sensitivity_5x5 contains molecule/scaffold/reference for all four core tasks" if split_ready else "FINAL grouping-sensitivity summary is missing or incomplete",
        command="python -u run.py final-robustness",
    )

    ul_pred = locate_task_file(index, "UL94_V0", "outer_predictions", use_bde=False, optional=True)
    ul_protocol_bad = False
    if ul_pred:
        ul_audit = _prediction_audit(ul_pred, "UL94_V0")
        pred_rows.append(ul_audit)
        ul_protocol_bad = str(ul_audit.get("protocol_status", "")).startswith("FAIL")
    _add(rows, "Fig5 UL-94 diagnostics", "MISSING" if not ul_pred else "PROTOCOL_MISMATCH" if ul_protocol_bad else "READY", ul_pred,
         command="python -u run.py nested --tasks UL94_V0 --bde without --results results/scientific_validation/FINAL_core_fixed_baseline_inclusive_5x5")

    source = args.results_root / "scientific_validation" / "FINAL_information_source_ablation_5x5" / "scientific_nested_summary_all.csv"
    source_ready = source.exists()
    source_note = ""
    if source_ready:
        try:
            src = read_csv_auto(source)
            required_pairs = {(task, scope) for task in ["LOI", "PHRR", "THR", "UL94_V0"] for scope in ["all", "conditions_only", "molecular_only"]}
            observed_pairs = set(zip(src.get("task", pd.Series(dtype=str)).astype(str), src.get("feature_scope", pd.Series(dtype=str)).astype(str)))
            source_ready = required_pairs.issubset(observed_pairs)
            source_note = f"FINAL information-source rows={len(src)}; required 4 tasks × 3 scopes verified" if source_ready else "FINAL information-source summary exists but required task/scope combinations are incomplete"
        except Exception as exc:
            source_ready = False
            source_note = f"Could not read FINAL information-source summary: {exc}"
    _add(rows, "Fig6 information-source ablation", "READY" if source_ready else "MISSING", source if source.exists() else None,
         note=source_note, command="python -u run.py final-robustness")

    paired_bde = args.results_root / "scientific_validation" / "FINAL_BDE_paired_ablation_5x5" / "scientific_nested_summary_all.csv"
    paired_ready = paired_bde.exists()
    bde_pred = _locate_any(index, ["BDE_selected_config_all_predictions.csv", "BDE_all_predictions.csv"], ["07_bde", "selected_config"])
    if bde_pred and paired_ready:
        bde_status = "READY"
    elif bde_pred or paired_ready:
        bde_status = "PARTIAL"
    else:
        bde_status = "MISSING"
    _add(rows, "Fig7 BDE model and ablation", bde_status, bde_pred or (paired_bde if paired_ready else None),
         note=f"independent_BDE={'yes' if bde_pred else 'no'}; FINAL paired ablation={'yes' if paired_ready else 'no'}",
         command="python -u run.py bde-model" if not bde_pred else "")

    shap_missing = []
    shap_long_missing = []
    for task in ["LOI", "PHRR", "THR", "UL94_V0"]:
        summary = _locate_any(index, [f"{task}_SHAP_stability_summary.csv", f"{task}_SHAP_aggregate.csv"], ["05_shap", task])
        long_files = [p for p in index.csv_files if p.name.startswith(f"{task}_outer_fold_") and "SHAP_values_long" in p.name]
        if not summary:
            shap_missing.append(task)
        if not long_files:
            shap_long_missing.append(task)
    _add(rows, "Fig8 SHAP structure-property relationships", "READY" if not shap_missing and not shap_long_missing else "PARTIAL" if not shap_missing else "MISSING",
         note=f"missing summary={shap_missing}; missing sample-level SHAP={shap_long_missing}",
         command="python -u run.py shap --tasks LOI,PHRR,THR,UL94_V0 --results results/05_Shap/FINAL_core_fixed_baseline_inclusive_5x5")

    ad_files = [p for p in index.csv_files if "06_applicabilitydomain" in p.as_posix().lower() or "applicability" in p.as_posix().lower()]
    _add(rows, "Fig9 applicability domain and error analysis", "READY" if ad_files else "MISSING",
         path=ad_files[0] if ad_files else None, note=f"AD CSV files={len(ad_files)}",
         command="python -u run.py ad")

    candidate = index.locate(
        "candidate_best_per_molecule.csv",
        prefer=["06_reversedesign", "combined_flux50"],
        avoid=["final_flux50", "sensitivity_flux35", "smoke", "2x2"],
        optional=True,
        label="audit combined virtual screening",
    )
    final_priority = index.locate(
        "final_priority_candidates_combined.csv",
        prefer=["06_reversedesign", "final_priority_combined"],
        avoid=["final_priority/", "smoke", "2x2"],
        optional=True,
        label="audit combined final priority",
    )
    candidate_status = "READY" if candidate and final_priority else "PARTIAL" if candidate or final_priority else "FUTURE"
    _add(rows, "Fig10 virtual screening", candidate_status, candidate or final_priority,
         note="Requires combined_flux50 eligible candidates and final_priority_combined final candidates")

    # Supplementary groups. FigS7 no longer forces rerunning historical K/view scans.
    dev_matches = [p for p in index.csv_files if "_model_comparison.csv" in p.name.lower() and "/results/main/" in ("/" + p.as_posix().lower())]
    final_fold_files = [locate_task_file(index, t, "outer_fold_metrics", use_bde=False, optional=True) for t in ["LOI", "PHRR", "THR", "UL94_V0", "Tg", "TS_MPa"]]
    frozen_fallback_ready = all(final_fold_files)
    s7_path = dev_matches[0] if dev_matches else next((p for p in final_fold_files if p), None)
    _add(rows, "FigS4 development configuration comparison", "READY" if (dev_matches or frozen_fallback_ready) else "MISSING", s7_path,
         note="uses historical five-seed development comparisons when present; otherwise uses frozen view/K plus FINAL outer-fold model-selection frequencies (no K/view rerun)",
         command="")

    expected_rows = _expected_rows_from_predictions(pred_rows)

    null_summary = args.results_root / "scientific_validation" / "FINAL_null_tests" / "null_test_summary.csv"
    null_status, null_note = _audit_null_protocol(null_summary, expected_rows)
    _add(
        rows, "FigS6 null and Y-scrambling tests", null_status,
        null_summary if null_summary.exists() else None, note=null_note,
        command="python -u run.py null-tests --tasks LOI,PHRR,THR,UL94_V0 --permutations 999",
    )

    learning_summary = args.results_root / "scientific_validation" / "FINAL_learning_curves" / "learning_curve_summary.csv"
    learning_all = args.results_root / "scientific_validation" / "FINAL_learning_curves" / "learning_curve_all_results.csv"
    learning_status, learning_note = _audit_learning_curve_protocol(
        learning_summary, learning_all, expected_rows
    )
    _add(
        rows, "FigS7 learning curves", learning_status,
        learning_summary if learning_summary.exists() else None, note=learning_note,
        command="python -u run.py learning-curves --tasks LOI,PHRR,THR,UL94_V0,Tg,TS_MPa",
    )

    error_matches = [
        p for p in index.csv_files
        if "_top_20_error_cases.csv" in p.name.lower()
        or "_top_20_error_cases.csv" in p.as_posix().lower()
    ]
    _add(
        rows, "TableS8 outer error cases", "READY" if error_matches else "MISSING",
        error_matches[0] if error_matches else None, command="python -u run.py error-analysis"
    )

    status = pd.DataFrame(rows)
    status.to_csv(args.output / "paper_input_audit.csv", index=False, encoding="utf-8-sig")
    predictions = pd.DataFrame(pred_rows).drop_duplicates(subset=["task", "file"] if pred_rows else None)
    predictions.to_csv(args.output / "outer_prediction_integrity.csv", index=False, encoding="utf-8-sig")
    index.save_manifest(args.output / "selected_input_files.csv")
    write_json(args.output / "run_config.json", vars(args))

    ready = int(status.status.eq("READY").sum())
    total = len(status)
    print(status[["item", "status", "note"]].to_string(index=False))
    print(f"\n[SUMMARY] READY={ready}/{total}; audit saved to {args.output}")
    if not predictions.empty:
        print("\n[OUTER PREDICTION INTEGRITY + PROTOCOL]")
        print(predictions[[
            "task", "rows", "outer_fold_count", "duplicate_sample_ids",
            "valid_prediction_pairs", "integrity_status", "protocol_status",
        ]].to_string(index=False))

        hard_fail = predictions[
            predictions["integrity_status"].astype(str).str.startswith("FAIL")
            | predictions["protocol_status"].astype(str).str.startswith("FAIL")
        ]
        if not hard_fail.empty:
            print("\n[FAIL] Manuscript-facing outer predictions are not safe to use. "
                  "Rerun/synchronize the frozen FINAL fixed baseline-inclusive workflow; do not fall back to curated/modified-only/V4 outputs.")
            raise SystemExit(3)

    diagnostic_mismatch = status[
        status["item"].isin([
            "FigS6 null and Y-scrambling tests",
            "FigS7 learning curves",
        ])
        & status["status"].eq("PROTOCOL_MISMATCH")
    ]
    if not diagnostic_mismatch.empty:
        print(
            "\n[FAIL] FINAL diagnostic outputs exist but do not match the frozen "
            "baseline-inclusive manuscript protocol. Regenerate only the flagged diagnostics; "
            "do not rerun the frozen FINAL primary models."
        )
        raise SystemExit(4)

    main_missing = status[status.item.str.startswith("Fig") & status.status.isin(["MISSING"])]
    if args.strict and not main_missing.empty:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
