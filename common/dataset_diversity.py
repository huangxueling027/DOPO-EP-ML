# -*- coding: utf-8 -*-
"""Chemical-structure diversity analysis for the DOPO+EP project."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from common import pipeline_core as core
from common.chem_standardization import standardize_dataset


TASK_KEYS = {
    "ALL": None,
    "LOI": "LOI",
    "PHRR": "PHRR",
    "THR": "THR",
    "UL94_V0": "UL94",
    "Tg": "Tg",
    "Char_yield": "Char_yield",
    "TS_MPa": "TS_MPa",
    "FS_MPa": "FS_MPa",
    "Delta_LOI": "Delta_LOI",
    "Delta_PHRR": "Delta_PHRR",
    "Delta_THR": "Delta_THR",
    "Delta_CY": "Delta_CY",
}


def _normalized_entropy(counts: Iterable[int]) -> float:
    values = np.asarray(list(counts), dtype=float)
    total = values.sum()
    if total <= 0 or len(values) <= 1:
        return 0.0
    p = values / total
    return float(-np.sum(p * np.log(p + 1e-15)) / math.log(len(values)))


def _csfp_diversity(counts: Iterable[int]) -> tuple[float, float]:
    values = np.asarray(sorted(list(counts), reverse=True), dtype=float)
    if values.size == 0 or values.sum() <= 0:
        return float("nan"), float("nan")
    cumulative = np.cumsum(values) / values.sum()
    # Right-step normalized area matching the project implementation.
    auc = float(np.mean(cumulative))
    diversity = float(np.clip(2.0 * (1.0 - auc), 0.0, 1.0))
    return auc, diversity


def _task_mask(df: pd.DataFrame, colmap: dict[str, str], task: str) -> pd.Series:
    key = TASK_KEYS[task]
    if key is None:
        return pd.Series(True, index=df.index)
    if key not in colmap:
        return pd.Series(False, index=df.index)
    column = colmap[key]
    if task == "UL94_V0":
        return df[column].notna() & df[column].astype(str).str.strip().ne("")
    target = pd.to_numeric(df[column], errors="coerce")
    return core.build_task_valid_mask(df, colmap, task, target=target)


def analyse_dataset(
    input_path: str | Path,
    output_dir: str | Path,
    tasks: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = core.read_csv_auto(str(input_path))
    colmap = core.resolve_columns(raw)
    cleaned = core.clean_dataframe(raw, colmap)
    df = standardize_dataset(cleaned, colmap)

    requested_tasks = list(tasks) if tasks is not None else list(TASK_KEYS)
    unknown = [task for task in requested_tasks if task not in TASK_KEYS]
    if unknown:
        raise KeyError(f"Unsupported diversity tasks: {unknown}")

    summaries: list[dict[str, object]] = []
    scaffold_rows: list[dict[str, object]] = []

    for task in requested_tasks:
        subset = df.loc[_task_mask(df, colmap, task)].copy()
        valid_main = subset.loc[subset["SMILES_main_valid"]].copy()
        scaffold_counts = valid_main.loc[
            valid_main["Murcko_scaffold_main"].astype(str).str.len() > 0,
            "Murcko_scaffold_main",
        ].value_counts()
        csfp_auc, csfp_diversity = _csfp_diversity(scaffold_counts.values)
        n_valid = int(valid_main.shape[0])
        n_scaffolds = int(scaffold_counts.shape[0])

        summaries.append({
            "task": task,
            "n_valid_formulation_rows": int(subset.shape[0]),
            "n_valid_main_smiles_rows": n_valid,
            "n_invalid_main_smiles_rows": int((~subset["SMILES_main_valid"]).sum()),
            "main_smiles_valid_rate": float(n_valid / len(subset)) if len(subset) else float("nan"),
            "n_unique_main_molecules": int(valid_main["Canonical_SMILES_main"].nunique()),
            "n_unique_main_co_pairs": int(subset["Canonical_main_co_pair"].nunique()),
            "n_unique_main_scaffolds": n_scaffolds,
            "scaffold_to_valid_row_ratio": float(n_scaffolds / n_valid) if n_valid else float("nan"),
            "scaffold_entropy_normalized": _normalized_entropy(scaffold_counts.values),
            "scaffold_simpson_diversity": float(1.0 - np.sum((scaffold_counts.values / scaffold_counts.values.sum()) ** 2)) if scaffold_counts.sum() else float("nan"),
            "csfp_auc_right_step": csfp_auc,
            "scaffold_diversity_score": csfp_diversity,
            "n_potential_duplicate_rows": int(subset["Potential_parallel_or_duplicate"].sum()),
            "n_potential_duplicate_groups": int(subset.loc[subset["Potential_parallel_or_duplicate"], "Duplicate_group_id"].nunique()),
            "n_multicomponent_rows": int(subset["Multicomponent_flag"].sum()),
            "n_references": int(subset[colmap.get("Reference", "Reference")].nunique()) if colmap.get("Reference", "Reference") in subset.columns else np.nan,
        })

        total = scaffold_counts.sum()
        running = 0
        for rank, (scaffold, count) in enumerate(scaffold_counts.items(), start=1):
            running += count
            scaffold_rows.append({
                "task": task,
                "rank": rank,
                "Murcko_scaffold_main": scaffold,
                "count": int(count),
                "fraction": float(count / total),
                "cumulative_fraction": float(running / total),
            })

    molecule_cols = [c for c in [
        colmap.get("FR_main"), colmap.get("FR_co"), colmap.get("SMILES_main"), colmap.get("SMILES_co"),
        "Canonical_SMILES_main", "Canonical_SMILES_co", "InChIKey_main", "InChIKey_co",
        "Murcko_scaffold_main", "Murcko_scaffold_co", "SMILES_valid_flag", "Multicomponent_flag",
        "Canonical_main_co_pair", "Duplicate_group_id", "Duplicate_group_size", "Potential_parallel_or_duplicate",
        colmap.get("Reference", "Reference"),
    ] if c and c in df.columns]
    molecule_table = df.loc[:, list(dict.fromkeys(molecule_cols))].copy()
    summary_table = pd.DataFrame(summaries)
    scaffold_table = pd.DataFrame(scaffold_rows)

    molecule_table.to_csv(output_dir / "canonicalized_molecule_table.csv", index=False, encoding="utf-8-sig")
    summary_table.to_csv(output_dir / "task_diversity_summary.csv", index=False, encoding="utf-8-sig")
    scaffold_table.to_csv(output_dir / "task_scaffold_counts.csv", index=False, encoding="utf-8-sig")
    return molecule_table, summary_table, scaffold_table
