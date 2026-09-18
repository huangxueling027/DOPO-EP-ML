# -*- coding: utf-8 -*-
"""Validate neat-EP feature masking and Delta-task row definitions.

This check is intentionally lightweight.  It does not train a model.

Checks
------
1. Rows with an explicit total FR loading of zero have zeroed main/co
   molecular features and zeroed FR-derived structured flags.
2. Curing-agent features remain available on neat-EP rows.
3. Delta tasks exclude explicit loading=0 reference rows while retaining rows
   whose loading is missing but whose Delta target is available.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import pipeline_core as core


DELTA_TASKS = ("Delta_LOI", "Delta_PHRR", "Delta_THR", "Delta_CY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "DOPO_EP_new_with_BDE.csv",
    )
    return parser.parse_args()


def max_abs(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    values = frame.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    return float(np.max(np.abs(values))) if values.size else 0.0


def main() -> None:
    args = parse_args()
    raw = core.read_csv_auto(str(args.input))
    colmap = core.resolve_columns(raw)
    df = core.clean_dataframe(raw, colmap)

    old_bde = core.USE_BDE_FEATURES
    core.USE_BDE_FEATURES = True
    try:
        X, _ = core.build_feature_matrix(
            df,
            colmap,
            fp_bits=128,
            use_p_interactions=False,
            fp_type="morgan",
            radius=2,
        )
    finally:
        core.USE_BDE_FEATURES = old_bde

    loading = core.get_loading_series(df, colmap)
    zero_mask = loading.eq(0.0)
    if not zero_mask.any():
        raise AssertionError("No explicit loading=0 rows were found; validation cannot run.")

    main_cols = [column for column in X.columns if str(column).startswith("main_")]
    co_cols = [column for column in X.columns if str(column).startswith("co_")]
    prep_cols = [
        column for column in X.columns
        if str(column).startswith("Preparation_Method_")
        or str(column).startswith("FR_class_")
        or str(column).startswith("Synergy_type_")
    ]
    fr_flag_names = {
        colmap[key]
        for key in (
            "Preparation_Method_num", "Main_FR_fraction", "Co_FR_fraction",
            "Synergy_flag", "Has_P", "Has_N", "Has_Si", "Has_B", "Has_S",
            "Has_Al",
        )
        if key in colmap
    }
    fr_flag_cols = [column for column in X.columns if column in fr_flag_names]

    checks = {
        "main_molecular_max_abs": max_abs(X.loc[zero_mask, main_cols]),
        "co_molecular_max_abs": max_abs(X.loc[zero_mask, co_cols]),
        "preparation_dummy_max_abs": max_abs(X.loc[zero_mask, prep_cols]),
        "fr_structured_flag_max_abs": max_abs(X.loc[zero_mask, fr_flag_cols]),
    }
    for name, value in checks.items():
        if value > 1e-12:
            raise AssertionError(f"{name} is not zero on neat-EP rows: {value}")

    _, _, co_present = core._flame_retardant_presence_masks(df, colmap)
    absent_co = co_present.eq(0.0)
    if X.loc[absent_co, co_cols].isna().any(axis=None):
        raise AssertionError(
            "Absent co-FR rows still contain NaN molecular descriptors; "
            "they would be median-imputed instead of represented by zeros."
        )

    if "FR_present" not in X.columns or not X.loc[zero_mask, "FR_present"].eq(0).all():
        raise AssertionError("FR_present is missing or not zero on neat-EP rows.")

    curing_cols = [column for column in X.columns if str(column).startswith("curing_")]
    curing_nonzero_rows = 0
    if curing_cols:
        curing_values = X.loc[zero_mask, curing_cols].apply(
            pd.to_numeric, errors="coerce"
        ).fillna(0.0)
        curing_nonzero_rows = int(curing_values.abs().sum(axis=1).gt(0).sum())

    delta_rows: list[dict[str, object]] = []
    for task in DELTA_TASKS:
        target_col = colmap.get(task)
        if target_col is None:
            continue
        target = pd.to_numeric(df[target_col], errors="coerce")
        before = int(target.notna().sum())
        valid = core.build_task_valid_mask(df, colmap, task, target=target)
        after = int(valid.sum())
        zero_removed = int((target.notna() & loading.eq(0.0)).sum())
        unknown_loading_retained = int((valid & loading.isna()).sum())
        if before - after != zero_removed:
            raise AssertionError(
                f"{task}: expected removal={zero_removed}, observed={before-after}"
            )
        delta_rows.append({
            "task": task,
            "target_non_null_before": before,
            "valid_after_baseline_filter": after,
            "explicit_zero_baselines_removed": zero_removed,
            "unknown_loading_rows_retained": unknown_loading_retained,
        })

    print("[PASS] Neat-EP FR molecular and structured features are zeroed.")
    print(f"[INFO] explicit loading=0 rows: {int(zero_mask.sum())}")
    print(f"[INFO] neat-EP rows retaining curing features: {curing_nonzero_rows}")
    print(pd.DataFrame(delta_rows).to_string(index=False))


if __name__ == "__main__":
    main()
