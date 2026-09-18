# -*- coding: utf-8 -*-
"""Optional TabPFN helpers.

TabPFN is intentionally kept outside the original model pool.  The helper imports
it only when the comparison script is run, so the existing project remains usable
without the optional dependency.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, f_regression
from sklearn.impute import SimpleImputer

TaskType = Literal["regression", "classification"]


@dataclass
class SelectedData:
    X_train: np.ndarray
    X_test: np.ndarray
    selected_features: list[str]
    requested_k: int | str
    effective_k: int


def import_tabpfn_classes():
    """Import TabPFN lazily and provide a clear installation error."""
    try:
        from tabpfn import TabPFNClassifier, TabPFNRegressor
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "TabPFN is not installed or could not be imported. Run: "
            "pip install -r requirements_optional.txt"
        ) from exc
    return TabPFNRegressor, TabPFNClassifier


def _supported_kwargs(callable_obj: Any, candidates: dict[str, Any]) -> dict[str, Any]:
    """Keep only constructor arguments supported by the installed TabPFN version."""
    try:
        parameters = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return {}
    return {key: value for key, value in candidates.items() if key in parameters}


def create_tabpfn_model(
    task_type: TaskType,
    *,
    random_state: int,
    device: str = "auto",
):
    """Create a version-tolerant TabPFN regressor or classifier."""
    TabPFNRegressor, TabPFNClassifier = import_tabpfn_classes()
    model_class = TabPFNRegressor if task_type == "regression" else TabPFNClassifier

    candidates: dict[str, Any] = {"random_state": random_state}
    if device and str(device).lower() != "auto":
        candidates["device"] = device

    return model_class(**_supported_kwargs(model_class, candidates))


def select_features_for_tabpfn(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    *,
    k: int | str,
    task_type: TaskType,
) -> SelectedData:
    """Fit imputation, variance filtering and K-selection on the training split only."""
    non_empty = X_train.columns[~X_train.isna().all(axis=0)]
    if len(non_empty) == 0:
        raise ValueError("No usable feature columns remain after removing all-NaN columns.")

    X_train_use = X_train.loc[:, non_empty].copy()
    X_test_use = X_test.loc[:, non_empty].copy()

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train_use)
    X_test_imp = imputer.transform(X_test_use)

    variance = VarianceThreshold(threshold=1e-8)
    X_train_var = variance.fit_transform(X_train_imp)
    X_test_var = variance.transform(X_test_imp)
    variance_features = np.asarray(non_empty)[variance.get_support()]

    if X_train_var.shape[1] == 0:
        raise ValueError("No feature remains after VarianceThreshold.")

    if isinstance(k, str) and k.strip().lower() in {"all", "none", "full"}:
        effective_k = X_train_var.shape[1]
    else:
        effective_k = min(int(k), X_train_var.shape[1])

    score_func = f_regression if task_type == "regression" else f_classif
    selector = SelectKBest(score_func=score_func, k=effective_k)
    X_train_sel = selector.fit_transform(X_train_var, y_train)
    X_test_sel = selector.transform(X_test_var)
    selected_features = variance_features[selector.get_support()].tolist()

    # TabPFN accepts numeric arrays.  Defensive conversion avoids object dtypes
    # caused by mixed CSV columns.
    X_train_sel = np.asarray(X_train_sel, dtype=np.float32)
    X_test_sel = np.asarray(X_test_sel, dtype=np.float32)

    return SelectedData(
        X_train=X_train_sel,
        X_test=X_test_sel,
        selected_features=selected_features,
        requested_k=k,
        effective_k=effective_k,
    )
