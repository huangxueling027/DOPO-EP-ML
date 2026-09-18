# -*- coding: utf-8 -*-
"""Group-label and cross-validation helpers for scientific evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover
    StratifiedGroupKFold = None


SUPPORTED_SPLITS = ("molecule", "molecule_curing", "scaffold", "reference", "random")


def _safe_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def build_group_labels(
    df: pd.DataFrame,
    *,
    strategy: str,
    reference_col: str | None = None,
) -> pd.Series | None:
    strategy = strategy.strip().lower()
    if strategy not in SUPPORTED_SPLITS:
        raise ValueError(f"Unsupported split strategy: {strategy}. Available: {SUPPORTED_SPLITS}")
    if strategy == "random":
        return None

    if strategy == "molecule":
        if "Canonical_main_co_pair" in df.columns:
            labels = _safe_text(df["Canonical_main_co_pair"])
        else:
            main = _safe_text(df.get("Canonical_SMILES_main", pd.Series("", index=df.index)))
            co = _safe_text(df.get("Canonical_SMILES_co", pd.Series("", index=df.index)))
            labels = "MAIN=" + main + "|CO=" + co
        fallback = pd.Series([f"ROW={i}" for i in df.index], index=df.index)
        return labels.where(labels.ne("MAIN=|CO="), fallback)

    if strategy == "molecule_curing":
        main = _safe_text(df.get("Canonical_SMILES_main", pd.Series("", index=df.index)))
        co = _safe_text(df.get("Canonical_SMILES_co", pd.Series("", index=df.index)))
        curing = _safe_text(
            df.get(
                "Canonical_SMILES_curing",
                df.get("SMILES_Curing_Agent", pd.Series("", index=df.index)),
            )
        )
        labels = "MAIN=" + main + "|CO=" + co + "|CURING=" + curing
        fallback = pd.Series([f"ROW={i}" for i in df.index], index=df.index)
        return labels.where(labels.ne("MAIN=|CO=|CURING="), fallback)

    if strategy == "scaffold":
        scaffold = _safe_text(df.get("Murcko_scaffold_main", pd.Series("", index=df.index)))
        molecule = _safe_text(df.get("Canonical_SMILES_main", pd.Series("", index=df.index)))
        fallback = pd.Series([f"MOL={value}" if value else f"ROW={i}" for i, value in zip(df.index, molecule)], index=df.index)
        return scaffold.where(scaffold.ne(""), fallback)

    if strategy == "reference":
        if reference_col and reference_col in df.columns:
            reference = _safe_text(df[reference_col])
        elif "Reference" in df.columns:
            reference = _safe_text(df["Reference"])
        else:
            reference = pd.Series("", index=df.index)
        fallback = pd.Series([f"ROW={i}" for i in df.index], index=df.index)
        return reference.where(reference.ne(""), fallback)

    raise AssertionError("unreachable")


@dataclass(frozen=True)
class Fold:
    train_idx: np.ndarray
    test_idx: np.ndarray
    fold_id: int


def _n_splits_for_groups(groups: pd.Series | None, requested: int, n_samples: int) -> int:
    if groups is None:
        return max(2, min(requested, n_samples))
    return max(2, min(requested, int(groups.nunique(dropna=False))))


def iter_cv_folds(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    groups: pd.Series | None,
    n_splits: int,
    random_state: int,
    classification: bool,
) -> Iterator[Fold]:
    """Yield folds, preferring group-aware and stratified group-aware CV."""
    n_splits_use = _n_splits_for_groups(groups, n_splits, len(y))
    if groups is not None:
        groups = pd.Series(groups).reset_index(drop=True)
        if classification and StratifiedGroupKFold is not None:
            splitter = StratifiedGroupKFold(
                n_splits=n_splits_use,
                shuffle=True,
                random_state=random_state,
            )
            iterator = splitter.split(X, y, groups)
        else:
            # GroupKFold has no random_state. Stable order is intentional.
            splitter = GroupKFold(n_splits=n_splits_use)
            iterator = splitter.split(X, y, groups)
    else:
        if classification:
            min_class = int(pd.Series(y).value_counts().min())
            n_splits_use = max(2, min(n_splits_use, min_class))
            splitter = StratifiedKFold(n_splits=n_splits_use, shuffle=True, random_state=random_state)
            iterator = splitter.split(X, y)
        else:
            splitter = KFold(n_splits=n_splits_use, shuffle=True, random_state=random_state)
            iterator = splitter.split(X, y)

    for fold_id, (train_idx, test_idx) in enumerate(iterator, start=1):
        yield Fold(np.asarray(train_idx), np.asarray(test_idx), fold_id)


def group_overlap(groups: pd.Series | None, train_idx: np.ndarray, test_idx: np.ndarray) -> int:
    if groups is None:
        return 0
    values = pd.Series(groups).reset_index(drop=True)
    return len(set(values.iloc[train_idx]) & set(values.iloc[test_idx]))
