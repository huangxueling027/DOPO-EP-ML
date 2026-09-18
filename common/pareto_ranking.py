# -*- coding: utf-8 -*-
"""Multi-objective candidate ranking with uncertainty and AD penalties."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Objective:
    column: str
    direction: str  # "max" or "min"
    weight: float = 1.0


def robust_minmax(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    finite = values[np.isfinite(values)]
    if finite.empty:
        return pd.Series(np.nan, index=series.index)
    low, high = finite.quantile([0.05, 0.95])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = finite.min(), finite.max()
    if high <= low:
        return pd.Series(0.5, index=series.index)
    return ((values.clip(low, high) - low) / (high - low)).clip(0.0, 1.0)


def pareto_front(values: np.ndarray) -> np.ndarray:
    """Return True for non-dominated rows; all columns are larger-is-better."""
    n = values.shape[0]
    front = np.ones(n, dtype=bool)
    for i in range(n):
        if not front[i] or not np.all(np.isfinite(values[i])):
            front[i] = False
            continue
        dominates_i = np.all(values >= values[i], axis=1) & np.any(values > values[i], axis=1)
        if np.any(dominates_i):
            front[i] = False
    return front


def pareto_layers(values: np.ndarray) -> np.ndarray:
    ranks = np.full(values.shape[0], np.nan)
    remaining = np.arange(values.shape[0])
    rank = 1
    while len(remaining):
        mask = pareto_front(values[remaining])
        if not mask.any():
            ranks[remaining] = rank
            break
        ranks[remaining[mask]] = rank
        remaining = remaining[~mask]
        rank += 1
    return ranks


def rank_candidates(
    frame: pd.DataFrame,
    objectives: Iterable[Objective],
    *,
    uncertainty_columns: Iterable[str] = (),
    loading_column: str | None = None,
    loading_reference: float = 10.0,
    uncertainty_penalty_weight: float = 0.15,
    ad_penalty_weight: float = 0.20,
    loading_penalty_weight: float = 0.10,
) -> pd.DataFrame:
    out = frame.copy()
    objective_list = [obj for obj in objectives if obj.column in out.columns]
    if not objective_list:
        raise ValueError("None of the requested objective columns exist in the candidate table.")

    normalized_columns: list[str] = []
    weighted_scores = []
    for objective in objective_list:
        normalized = robust_minmax(out[objective.column])
        if objective.direction.lower() == "min":
            normalized = 1.0 - normalized
        elif objective.direction.lower() != "max":
            raise ValueError(f"Invalid direction for {objective.column}: {objective.direction}")
        name = f"Normalized_{objective.column}"
        out[name] = normalized
        normalized_columns.append(name)
        weighted_scores.append(normalized * float(objective.weight))

    total_weight = sum(abs(float(obj.weight)) for obj in objective_list) or 1.0
    out["Performance_score"] = sum(weighted_scores) / total_weight

    uncertainty_parts = []
    for column in uncertainty_columns:
        if column in out.columns:
            uncertainty_parts.append(robust_minmax(out[column]).fillna(1.0))
    out["Uncertainty_penalty"] = (
        pd.concat(uncertainty_parts, axis=1).mean(axis=1)
        if uncertainty_parts else 0.0
    )

    ad_map = {
        "In_domain": 0.0,
        "Caution": 0.5,
        "Caution_new_scaffold": 0.65,
        "Extrapolation": 1.0,
        "Invalid_SMILES": 1.0,
    }
    if "Applicability_domain" in out.columns:
        out["AD_penalty"] = out["Applicability_domain"].map(ad_map).fillna(0.75)
    elif "Max_Tanimoto_to_training" in out.columns:
        out["AD_penalty"] = 1.0 - pd.to_numeric(out["Max_Tanimoto_to_training"], errors="coerce").clip(0, 1).fillna(0)
    else:
        out["AD_penalty"] = 0.0

    if loading_column and loading_column in out.columns:
        loading = pd.to_numeric(out[loading_column], errors="coerce")
        out["Loading_penalty"] = ((loading - loading_reference).clip(lower=0) / max(loading_reference, 1e-6)).clip(0, 1).fillna(0)
    else:
        out["Loading_penalty"] = 0.0

    out["Final_ranking_score"] = (
        out["Performance_score"]
        - uncertainty_penalty_weight * out["Uncertainty_penalty"]
        - ad_penalty_weight * out["AD_penalty"]
        - loading_penalty_weight * out["Loading_penalty"]
    )

    pareto_values = out[normalized_columns].to_numpy(dtype=float)
    out["Pareto_rank"] = pareto_layers(pareto_values)
    out["Pareto_front"] = out["Pareto_rank"].eq(1)
    out = out.sort_values(["Pareto_rank", "Final_ranking_score"], ascending=[True, False]).reset_index(drop=True)
    out.insert(0, "Overall_rank", np.arange(1, len(out) + 1))
    return out
