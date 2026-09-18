# -*- coding: utf-8 -*-
"""Shared data/feature preparation for literature-driven optional modules.

This module deliberately does not change the original model pool.  It rebuilds the
same feature views used by ``pipeline_core.py`` so that TabPFN and related optional
analyses can be compared under the current project settings.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from common import pipeline_core as core


TASK_TARGET_KEYS: Dict[str, str] = {
    "LOI": "LOI",
    "PHRR": "PHRR",
    "THR": "THR",
    "Tg": "Tg",
    "Char_yield": "Char_yield",
    "TS_MPa": "TS_MPa",
    "FS_MPa": "FS_MPa",
    "Delta_LOI": "Delta_LOI",
    "Delta_PHRR": "Delta_PHRR",
    "Delta_THR": "Delta_THR",
    "Delta_CY": "Delta_CY",
    "UL94_V0": "UL94_V0",
}

# Tasks whose generalization depends strongly on the curing network use
# MAIN+CO+CURING groups. Other tasks use MAIN+CO groups.
CURING_GROUP_TASKS: set[str] = {"Tg", "TS_MPa", "FS_MPa"}


@dataclass
class FeatureBundle:
    df: pd.DataFrame
    colmap: Dict[str, str]
    views: Dict[str, pd.DataFrame]
    molecule_groups: pd.Series
    molecule_groups_with_curing: pd.Series


def prepare_feature_bundle(input_path: str | Path) -> FeatureBundle:
    """Read the project dataset and reconstruct all current feature views."""
    input_path = Path(input_path)
    df_raw = core.read_csv_auto(str(input_path))
    colmap = core.resolve_columns(df_raw)
    df = core.clean_dataframe(df_raw, colmap)

    if "P_content wt%" in colmap:
        df["P_loading"] = pd.to_numeric(df[colmap["P_content wt%"]], errors="coerce")
    else:
        df["P_loading"] = np.nan
    colmap["P_loading"] = "P_loading"

    if "UL94" in colmap:
        def _ul94_v0_label(value):
            if pd.isna(value) or str(value).strip() == "":
                return np.nan
            return 1 if str(value).strip().upper() == "V-0" else 0
        df["UL94_V0"] = df[colmap["UL94"]].apply(_ul94_v0_label)
        colmap["UL94_V0"] = "UL94_V0"

    molecule_groups = core.build_molecule_group_labels(df, colmap)
    molecule_groups_with_curing = core.build_molecule_group_labels_with_curing(df, colmap)

    _old_bde_flag = core.USE_BDE_FEATURES
    core.USE_BDE_FEATURES = True
    try:
        full, _ = core.build_feature_matrix(
            df, colmap, fp_bits=512, use_p_interactions=False, fp_type="morgan", radius=2
        )
        compact, _ = core.build_feature_matrix(
            df, colmap, fp_bits=128, use_p_interactions=False, fp_type="morgan", radius=2
        )
        full_interaction, _ = core.build_feature_matrix(
            df, colmap, fp_bits=512, use_p_interactions=True, fp_type="morgan", radius=2
        )
        compact_interaction, _ = core.build_feature_matrix(
            df, colmap, fp_bits=128, use_p_interactions=True, fp_type="morgan", radius=2
        )
        morgan_r3, _ = core.build_feature_matrix(
            df, colmap, fp_bits=512, use_p_interactions=False, fp_type="morgan", radius=3
        )
        morgan_r3_1024, _ = core.build_feature_matrix(
            df, colmap, fp_bits=1024, use_p_interactions=False, fp_type="morgan", radius=3
        )
        maccs, _ = core.build_feature_matrix(
            df, colmap, fp_bits=166, use_p_interactions=False, fp_type="maccs", radius=0
        )
        descriptors, _ = core.build_feature_matrix(
            df, colmap, fp_bits=0, use_p_interactions=False, fp_type="descriptors", radius=0
        )
        maccs_descriptors = pd.concat([maccs, descriptors], axis=1)
        maccs_descriptors = maccs_descriptors.loc[
            :, ~maccs_descriptors.columns.duplicated()
        ].copy()

    finally:
        core.USE_BDE_FEATURES = _old_bde_flag

    views = {
        "full": full,
        "compact": compact,
        "full_interaction": full_interaction,
        "compact_interaction": compact_interaction,
        "morgan_r3": morgan_r3,
        "morgan_r3_1024": morgan_r3_1024,
        "maccs": maccs,
        "descriptors": descriptors,
        "maccs_descriptors": maccs_descriptors,
    }

    return FeatureBundle(
        df=df,
        colmap=colmap,
        views=views,
        molecule_groups=molecule_groups,
        molecule_groups_with_curing=molecule_groups_with_curing,
    )


def get_target(bundle: FeatureBundle, task_name: str) -> pd.Series:
    """Return the numeric target for a supported task."""
    if task_name == "UL94_V0":
        if "UL94_V0" not in bundle.df.columns:
            raise KeyError("UL94 column was not found; UL94_V0 cannot be generated.")
        return pd.to_numeric(bundle.df["UL94_V0"], errors="coerce")

    target_key = TASK_TARGET_KEYS.get(task_name)
    if target_key is None:
        raise KeyError(f"Unsupported task: {task_name}")
    if target_key not in bundle.colmap:
        raise KeyError(f"Target column for {task_name} was not found in the dataset.")
    return pd.to_numeric(bundle.df[bundle.colmap[target_key]], errors="coerce")


def get_groups(bundle: FeatureBundle, task_name: str) -> pd.Series:
    """Return the task-specific primary leakage-control groups."""
    if task_name in CURING_GROUP_TASKS:
        return bundle.molecule_groups_with_curing
    return bundle.molecule_groups


def get_task_matrix(
    bundle: FeatureBundle,
    task_name: str,
    view_name: str,
    *,
    use_bde: bool = False,
) -> pd.DataFrame:
    """Return a task-filtered feature matrix from a named current feature view."""
    view_key = str(view_name).strip().lower()
    if view_key not in bundle.views:
        raise KeyError(
            f"Unknown feature view '{view_name}'. Available: {sorted(bundle.views)}"
        )

    matrix = core.filter_features_for_task(bundle.views[view_key], task_name)

    # The main checked property models are no-BDE.  Remove every BDE-derived
    # feature defensively even when pipeline_core was imported with its legacy
    # default DOPO_USE_BDE_FEATURES=1.
    if not use_bde:
        bde_columns = [
            column for column in matrix.columns
            if "bde" in str(column).lower()
        ]
        if bde_columns:
            matrix = matrix.drop(columns=bde_columns, errors="ignore")

    return matrix.loc[:, ~matrix.columns.duplicated()].copy()
