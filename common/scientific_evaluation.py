# -*- coding: utf-8 -*-
"""Leakage-controlled nested evaluation for the DOPO+EP project.

Key design rules
----------------
1. Outer folds are used only for final evaluation.
2. Under curated/full selection, feature view, K and model are selected only
   from each outer-training set through inner cross-validation.  Under fixed
   selection, the configured view/K are fixed while model selection remains inner-CV-only.
   The UL-94 decision threshold also uses outer-training data only.
3. Imputation, variance filtering and SelectKBest are fitted inside every CV
   fold through a scikit-learn Pipeline.
4. Molecule, Murcko-scaffold, reference and random split scenarios are
   supported without changing the feature-generation code.
5. Regression prediction intervals use split conformal calibration; UL-94
   probabilities use outer-training-only Platt calibration.
"""
from __future__ import annotations

import json
import math
import os
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.feature_selection import VarianceThreshold

from common import pipeline_core as core
from common.chem_standardization import standardize_dataset
from common.literature_feature_views import FeatureBundle, get_groups, get_target, prepare_feature_bundle
from common.split_strategies import build_group_labels, group_overlap, iter_cv_folds


TaskType = Literal["regression", "classification"]


@dataclass(frozen=True)
class TaskConfig:
    task_type: TaskType
    descriptor_mode: Literal["basic", "advanced"]
    use_v7_features: bool
    current_view: str
    current_k: int | None
    curated_views: tuple[str, ...]
    curated_k: tuple[int | None, ...]
    full_k: tuple[int | None, ...]


# These settings mirror the task-specific final runners.  Keeping them here is
# essential: the same view name can contain a different number of columns under
# basic/advanced descriptor mode or with/without the V7 formula features.
TASK_CONFIGS: dict[str, TaskConfig] = {
    "LOI": TaskConfig("regression", "basic", True, "compact", 260, ("compact", "full", "descriptors", "morgan_r3"), (180, 220, 260, 300), (80, 120, 180, 220, 260, 300, 360, 440)),
    "PHRR": TaskConfig("regression", "basic", False, "compact", 330, ("compact", "full", "morgan_r3", "descriptors"), (220, 280, 330, 380), (80, 140, 220, 280, 330, 380, 440)),
    "THR": TaskConfig("regression", "advanced", True, "descriptors", 30, ("descriptors", "compact", "maccs"), (20, 30, 50, 80), (10, 20, 30, 50, 80, 120)),
    "Tg": TaskConfig("regression", "advanced", True, "morgan_r3", 270, ("morgan_r3", "compact", "descriptors", "full_interaction"), (180, 220, 270, 320), (80, 140, 180, 220, 270, 320, 400)),
    "Char_yield": TaskConfig("regression", "advanced", True, "full_interaction", 100, ("full_interaction", "compact_interaction", "descriptors", "maccs"), (50, 80, 100, 140), (30, 50, 80, 100, 140, 180)),
    "TS_MPa": TaskConfig("regression", "advanced", True, "compact", 140, ("compact", "morgan_r3", "descriptors", "full_interaction"), (80, 110, 140, 180), (50, 80, 110, 140, 180, 240)),
    "FS_MPa": TaskConfig("regression", "advanced", True, "morgan_r3", 240, ("morgan_r3", "compact", "descriptors", "full_interaction"), (140, 180, 240, 300), (80, 140, 180, 240, 300, 360)),
    "Delta_LOI": TaskConfig("regression", "advanced", False, "descriptors", 30, ("descriptors", "compact", "maccs", "morgan_r3"), (20, 30, 50, 80), (10, 20, 30, 50, 80, 120)),
    "Delta_PHRR": TaskConfig("regression", "advanced", False, "maccs", 240, ("maccs", "descriptors", "compact", "morgan_r3"), (160, 200, 240, 280), (80, 120, 160, 200, 240, 280, 360)),
    "Delta_THR": TaskConfig("regression", "advanced", False, "descriptors", 120, ("descriptors", "compact", "maccs", "morgan_r3"), (80, 100, 120, 160), (30, 50, 80, 100, 120, 160, 220)),
    "Delta_CY": TaskConfig("regression", "advanced", False, "descriptors", 180, ("descriptors", "full_interaction", "compact_interaction", "maccs"), (100, 140, 180, 220), (50, 80, 100, 140, 180, 220, 280)),
    "UL94_V0": TaskConfig("classification", "advanced", True, "morgan_r3", None, ("morgan_r3", "maccs", "descriptors", "compact"), (160, 220, None), (80, 120, 160, 220, 320, None)),
}
FORMAL_ABSOLUTE_TASKS = frozenset({
    "LOI",
    "PHRR",
    "THR",
    "UL94_V0",
    "Tg",
    "Char_yield",
    "TS_MPa",
    "FS_MPa",
})
DEFAULT_REGRESSION_MODELS = ("Ridge", "RF", "ExtraTrees", "GBDT", "XGB", "LGBM", "SVR", "SoftVote")
DEFAULT_CLASSIFICATION_MODELS = ("Logistic", "RF", "ExtraTrees", "GBDT", "XGB", "LGBM", "SVC", "SoftVote")


class NonEmptyColumnSelector(BaseEstimator, TransformerMixin):
    """Drop columns that are entirely missing in the fitted training fold."""

    def fit(self, X, y=None):
        frame = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.keep_mask_ = ~frame.isna().all(axis=0).to_numpy()
        if not self.keep_mask_.any():
            raise ValueError("All features are empty in this training fold.")
        return self

    def transform(self, X):
        frame = pd.DataFrame(X)
        if hasattr(self, "feature_names_in_") and set(self.feature_names_in_).issubset(frame.columns):
            frame = frame.loc[:, self.feature_names_in_]
        return frame.iloc[:, self.keep_mask_]

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_[self.keep_mask_]


class DuplicateColumnRemover(BaseEstimator, TransformerMixin):
    """Drop exactly duplicated numeric columns using training-fold data only.

    Duplicate aliases (for example ``Loading_total_FR wt%`` and
    ``Loading_total_FR``) and identical sparse fingerprint bits otherwise split
    importance and distort SelectKBest/SHAP rankings.  Fitting this transformer
    inside the pipeline avoids looking at the held-out fold.
    """

    def fit(self, X, y=None):
        array = np.asarray(X)
        if array.ndim != 2:
            raise ValueError(f"Expected 2D feature array, got shape={array.shape}")
        if array.shape[1] == 0:
            raise ValueError("No feature columns available for duplicate removal.")
        _, unique_indices = np.unique(array.T, axis=0, return_index=True)
        self.keep_indices_ = np.sort(unique_indices.astype(int))
        self.keep_mask_ = np.zeros(array.shape[1], dtype=bool)
        self.keep_mask_[self.keep_indices_] = True
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.keep_indices_]

    def get_support(self):
        return self.keep_mask_

    def get_feature_names_out(self, input_features=None):
        names = np.asarray(input_features if input_features is not None else [], dtype=object)
        return names[self.keep_mask_] if len(names) else names


class SafeSelectKBest(BaseEstimator, TransformerMixin):
    """Select at most K features after fold-specific variance filtering."""

    def __init__(self, score_func=f_regression, k: int | None = 100):
        self.score_func = score_func
        self.k = k

    def fit(self, X, y):
        n_features = int(np.asarray(X).shape[1])
        if self.k is None:
            k_use: int | str = "all"
        else:
            k_use = max(1, min(int(self.k), n_features))
        self.effective_k_ = n_features if k_use == "all" else int(k_use)
        self.selector_ = SelectKBest(score_func=self.score_func, k=k_use)
        self.selector_.fit(X, y)
        return self

    def transform(self, X):
        return self.selector_.transform(X)

    def get_support(self):
        return self.selector_.get_support()

    def get_feature_names_out(self, input_features=None):
        names = np.asarray(input_features if input_features is not None else [], dtype=object)
        return names[self.get_support()] if len(names) else names


@dataclass
class CandidateResult:
    view: str
    requested_k: int | None
    model_name: str
    inner_primary: float
    inner_secondary: float
    threshold: float | None = None
    oof_probability: np.ndarray | None = None
    error: str | None = None


@dataclass
class EvaluationOptions:
    outer_splits: int = 5
    inner_splits: int = 5
    random_state: int = 42
    split_strategy: str = "molecule"
    selection_scope: str = "fixed"
    screening_mode: str = "formulation"
    feature_scope: str = "all"
    use_bde: bool = False
    row_policy: str = "baseline_inclusive"
    conformal_alpha: float = 0.05
    include_tabpfn: bool = False
    model_names: tuple[str, ...] | None = None
    shap_stability: bool = False
    max_shap_samples: int = 80


@dataclass
class ScientificBundle:
    base: FeatureBundle
    df: pd.DataFrame
    unique_views: dict[str, pd.DataFrame]
    view_aliases: dict[str, str] = field(default_factory=dict)


def _view_signature(frame: pd.DataFrame) -> tuple[Any, ...]:
    # Column identity is enough to detect the current MACCS+descriptors duplicate
    # construction and avoids an expensive full-array comparison.
    return (frame.shape[1], tuple(map(str, frame.columns)))


ADVANCED_ONLY_DESCRIPTOR_KEYS = frozenset({
    "ExactMolWt", "MolMR", "LabuteASA", "NumValenceElectrons",
    "NumHeteroAtoms", "BertzCT", "BalabanJ", "AliphaticRings",
    "SaturatedRings", "AromaticCarbocycles", "AromaticHeterocycles",
    "SaturatedCarbocycles", "SaturatedHeterocycles", "Kappa1", "Kappa2",
    "Kappa3", "NHOHCount", "NOCount", "NumAmideBonds",
})


def _is_advanced_only_descriptor(column: Any) -> bool:
    name = str(column)
    return any(name == f"{prefix}_{key}" for prefix in ("main", "co", "curing") for key in ADVANCED_ONLY_DESCRIPTOR_KEYS)


def _is_v7_formula_feature(column: Any) -> bool:
    name = str(column)
    if name in {"Loading_x_Synergy", "P_loading_x_Synergy"}:
        return True
    for element in ("P", "N", "S", "B", "Si"):
        if name in {
            f"{element}_loading_real",
            f"{element}_loading_real_sq",
            f"{element}_loading_real_sqrt",
        }:
            return True
    for element in ("N", "Si", "B", "S"):
        if name in {
            f"P_{element}_loading_sum",
            f"P_{element}_loading_product",
            f"P_{element}_loading_ratio",
        }:
            return True
    return False


def prepare_scientific_bundle(input_path: str | Path) -> ScientificBundle:
    # Build the superset once, then remove task-inapplicable columns below.
    # This avoids rebuilding molecular fingerprints for every task while still
    # exactly reproducing each runner's descriptor/V7 switches.
    old_advanced = core.USE_ADVANCED_DESCRIPTORS
    old_v7 = core.USE_V7_FORMULA_FEATURES
    core.USE_ADVANCED_DESCRIPTORS = True
    core.USE_V7_FORMULA_FEATURES = True
    try:
        base = prepare_feature_bundle(input_path)
    finally:
        core.USE_ADVANCED_DESCRIPTORS = old_advanced
        core.USE_V7_FORMULA_FEATURES = old_v7
    enriched = standardize_dataset(base.df, base.colmap)
    unique: dict[str, pd.DataFrame] = {}
    aliases: dict[str, str] = {}
    signatures: dict[tuple[Any, ...], str] = {}
    for name, frame in base.views.items():
        signature = _view_signature(frame)
        if signature in signatures:
            aliases[name] = signatures[signature]
            continue
        signatures[signature] = name
        unique[name] = frame
    return ScientificBundle(base=base, df=enriched, unique_views=unique, view_aliases=aliases)


def _is_molecular_feature(column: Any) -> bool:
    """Return True for fingerprint/descriptor columns derived from molecular SMILES.

    Molecular blocks are identified by the explicit main/co/curing prefixes used
    by ``pipeline_core.build_feature_matrix``. Presence indicators are retained
    with the molecular-only scope so an all-zero block can be distinguished from
    an absent component.
    """
    name = str(column)
    lower = name.lower()
    if name in {"FR_present", "Main_FR_present", "Co_FR_present"}:
        return True
    return lower.startswith(("main_", "co_", "curing_"))


def _apply_feature_scope(matrix: pd.DataFrame, feature_scope: str) -> pd.DataFrame:
    """Restrict the feature matrix for information-source ablation.

    Scopes
    ------
    all
        Molecular structure + formulation/condition features.
    conditions_only
        Structured formulation, element, curing and test-condition features;
        all SMILES-derived fingerprints/descriptors and all BDE features removed.
    molecular_only
        Only SMILES-derived main/co/curing fingerprints/descriptors plus presence
        indicators. Loading, element contents, EP baselines and test conditions
        are removed.
    """
    scope = str(feature_scope).strip().lower()
    if scope == "all":
        return matrix
    molecular_columns = [c for c in matrix.columns if _is_molecular_feature(c)]
    if scope == "molecular_only":
        keep = molecular_columns
    elif scope == "conditions_only":
        keep = [
            c for c in matrix.columns
            if c not in molecular_columns and "bde" not in str(c).lower()
        ]
    else:
        raise ValueError(
            "feature_scope must be one of: all, conditions_only, molecular_only"
        )
    if not keep:
        raise ValueError(f"feature_scope={scope} produced an empty matrix")
    return matrix.loc[:, keep].copy()


def _filter_task_matrix(
    bundle: ScientificBundle,
    task: str,
    view: str,
    *,
    use_bde: bool,
    screening_mode: str,
    feature_scope: str = "all",
) -> pd.DataFrame:
    actual_view = bundle.view_aliases.get(view, view)
    if actual_view not in bundle.unique_views:
        raise KeyError(f"Unknown view={view}; available={sorted(bundle.unique_views)}")
    _old_bde_flag = core.USE_BDE_FEATURES
    core.USE_BDE_FEATURES = True
    try:
        matrix = core.filter_features_for_task(bundle.unique_views[actual_view], task)
    finally:
        core.USE_BDE_FEATURES = _old_bde_flag
    config = TASK_CONFIGS[task]
    if config.descriptor_mode == "basic":
        matrix = matrix.drop(
            columns=[c for c in matrix.columns if _is_advanced_only_descriptor(c)],
            errors="ignore",
        )
    if not config.use_v7_features:
        matrix = matrix.drop(
            columns=[c for c in matrix.columns if _is_v7_formula_feature(c)],
            errors="ignore",
        )
    if not use_bde:
        matrix = matrix.drop(columns=[c for c in matrix.columns if "bde" in str(c).lower()], errors="ignore")
    if screening_mode == "molecular":
        matrix = matrix.drop(columns=[c for c in matrix.columns if "ep_matrix" in str(c).lower()], errors="ignore")
    elif screening_mode != "formulation":
        raise ValueError("screening_mode must be 'formulation' or 'molecular'")
    matrix = _apply_feature_scope(matrix, feature_scope)
    return matrix.loc[:, ~matrix.columns.duplicated()].copy()


def _candidate_space(
    config: TaskConfig,
    bundle: ScientificBundle,
    scope: str,
) -> tuple[list[str], list[int | None]]:
    scope = scope.lower()
    if scope == "fixed":
        views = [config.current_view]
        ks = [config.current_k]
    elif scope == "curated":
        views = list(config.curated_views)
        ks = list(config.curated_k)
    elif scope == "full":
        views = list(bundle.unique_views)
        ks = list(config.full_k)
    else:
        raise ValueError("selection_scope must be fixed, curated or full")
    views = [bundle.view_aliases.get(v, v) for v in views]
    views = list(dict.fromkeys(v for v in views if v in bundle.unique_views))
    return views, ks


def _deduplicate_requested_k(ks: Iterable[int | None], n_raw_features: int) -> list[int | None]:
    seen: set[int | str] = set()
    result: list[int | None] = []
    for value in ks:
        effective: int | str = n_raw_features if value is None else min(int(value), n_raw_features)
        if effective in seen:
            continue
        seen.add(effective)
        result.append(value)
    return result


def _minimum_inner_available_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series | None,
    *,
    inner_splits: int,
    seed: int,
    classification: bool,
) -> int:
    """Estimate the minimum post-variance feature count using inner-training folds only."""
    counts: list[int] = []
    for train_idx, _ in _inner_splits(
        X_train, y_train, groups_train, n_splits=inner_splits,
        seed=seed, classification=classification,
    ):
        fold = X_train.iloc[train_idx]
        nonempty = fold.columns[~fold.isna().all(axis=0)]
        if len(nonempty) == 0:
            continue
        imputed = SimpleImputer(strategy="median").fit_transform(fold.loc[:, nonempty])
        deduplicated = DuplicateColumnRemover().fit_transform(imputed)
        variance = VarianceThreshold(threshold=1e-8)
        try:
            transformed = variance.fit_transform(deduplicated)
            counts.append(int(transformed.shape[1]))
        except ValueError:
            continue
    return min(counts) if counts else max(1, X_train.shape[1])


def _xgb_regressor(seed: int):
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.75, reg_alpha=0.3,
            reg_lambda=2.0, objective="reg:squarederror", random_state=seed,
            n_jobs=1,
        )
    except Exception:
        return GradientBoostingRegressor(random_state=seed)


def _lgbm_regressor(seed: int):
    try:
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=300, learning_rate=0.03, num_leaves=11,
            min_child_samples=12, subsample=0.8, colsample_bytree=0.75,
            reg_alpha=0.3, reg_lambda=2.0, random_state=seed, verbosity=-1,
            n_jobs=1,
        )
    except Exception:
        return GradientBoostingRegressor(random_state=seed)


def _xgb_classifier(seed: int):
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.75, reg_alpha=0.3,
            reg_lambda=2.0, eval_metric="logloss", random_state=seed,
            n_jobs=1,
        )
    except Exception:
        return GradientBoostingClassifier(random_state=seed)


def _lgbm_classifier(seed: int):
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=300, learning_rate=0.03, num_leaves=11,
            min_child_samples=12, subsample=0.8, colsample_bytree=0.75,
            reg_alpha=0.3, reg_lambda=2.0, random_state=seed, verbosity=-1,
            n_jobs=1,
        )
    except Exception:
        return GradientBoostingClassifier(random_state=seed)


def _regression_model(name: str, seed: int, include_tabpfn: bool = False):
    rf = RandomForestRegressor(n_estimators=350, min_samples_leaf=2, random_state=seed, n_jobs=1)
    et = ExtraTreesRegressor(n_estimators=450, min_samples_leaf=2, random_state=seed, n_jobs=1)
    xgb = _xgb_regressor(seed)
    lgbm = _lgbm_regressor(seed)
    models: dict[str, Any] = {
        "Ridge": Ridge(alpha=1.0),
        "RF": rf,
        "ExtraTrees": et,
        "GBDT": GradientBoostingRegressor(n_estimators=220, learning_rate=0.03, max_depth=2, subsample=0.85, random_state=seed),
        "XGB": xgb,
        "LGBM": lgbm,
        "SVR": SVR(C=5.0, epsilon=0.1, kernel="rbf"),
        "SoftVote": VotingRegressor([("xgb", xgb), ("lgbm", lgbm), ("rf", rf)]),
    }
    if include_tabpfn and name == "TabPFN":
        from common.tabpfn_utils import create_tabpfn_model
        return create_tabpfn_model("regression", random_state=seed, device="auto")
    if name not in models:
        raise KeyError(f"Unknown regression model: {name}")
    return models[name]


def _classification_model(name: str, seed: int, include_tabpfn: bool = False):
    rf = RandomForestClassifier(n_estimators=350, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=1)
    et = ExtraTreesClassifier(n_estimators=450, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=1)
    xgb = _xgb_classifier(seed)
    lgbm = _lgbm_classifier(seed)
    models: dict[str, Any] = {
        "Logistic": LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed),
        "RF": rf,
        "ExtraTrees": et,
        "GBDT": GradientBoostingClassifier(n_estimators=220, learning_rate=0.03, max_depth=2, subsample=0.85, random_state=seed),
        "XGB": xgb,
        "LGBM": lgbm,
        "SVC": SVC(C=5.0, kernel="rbf", probability=True, class_weight="balanced", random_state=seed),
        "SoftVote": VotingClassifier([("xgb", xgb), ("lgbm", lgbm), ("rf", rf)], voting="soft"),
    }
    if include_tabpfn and name == "TabPFN":
        from common.tabpfn_utils import create_tabpfn_model
        return create_tabpfn_model("classification", random_state=seed, device="auto")
    if name not in models:
        raise KeyError(f"Unknown classification model: {name}")
    return models[name]


def build_pipeline(
    *,
    task_type: TaskType,
    model_name: str,
    requested_k: int | None,
    seed: int,
    include_tabpfn: bool = False,
) -> Pipeline:
    score_func = f_regression if task_type == "regression" else f_classif
    model = (
        _regression_model(model_name, seed, include_tabpfn)
        if task_type == "regression"
        else _classification_model(model_name, seed, include_tabpfn)
    )
    scale = model_name in {"Ridge", "SVR", "Logistic", "SVC"}
    return Pipeline([
        ("nonempty", NonEmptyColumnSelector()),
        ("imputer", SimpleImputer(strategy="median")),
        ("deduplicate", DuplicateColumnRemover()),
        ("variance", VarianceThreshold(threshold=1e-8)),
        ("selector", SafeSelectKBest(score_func=score_func, k=requested_k)),
        ("scaler", StandardScaler() if scale else "passthrough"),
        ("model", model),
    ])


def regression_metrics(y_true, y_pred, prefix: str = "") -> dict[str, float]:
    return {
        f"{prefix}R2": float(r2_score(y_true, y_pred)),
        f"{prefix}RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        f"{prefix}MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def expected_calibration_error(y_true, probability, n_bins: int = 10) -> float:
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_arr)
    if total == 0:
        return float("nan")
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (p_arr >= left) & (p_arr < right if right < 1.0 else p_arr <= right)
        if not mask.any():
            continue
        accuracy = y_arr[mask].mean()
        confidence = p_arr[mask].mean()
        ece += mask.mean() * abs(accuracy - confidence)
    return float(ece)


def classification_metrics(y_true, y_pred, probability, prefix: str = "") -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    probability = np.asarray(probability, dtype=float)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    result = {
        f"{prefix}Accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        f"{prefix}Macro_F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        f"{prefix}Weighted_F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        f"{prefix}Brier": float(brier_score_loss(y_true, probability)),
        f"{prefix}ECE": expected_calibration_error(y_true, probability),
        f"{prefix}TN": int(tn), f"{prefix}FP": int(fp), f"{prefix}FN": int(fn), f"{prefix}TP": int(tp),
    }
    if len(np.unique(y_true)) == 2:
        result[f"{prefix}ROC_AUC"] = float(roc_auc_score(y_true, probability))
        result[f"{prefix}PR_AUC"] = float(average_precision_score(y_true, probability))
    else:
        result[f"{prefix}ROC_AUC"] = float("nan")
        result[f"{prefix}PR_AUC"] = float("nan")
    return result


def optimise_threshold(y_true, probability) -> tuple[float, float, float]:
    best = (0.5, -np.inf, -np.inf)
    for threshold in np.arange(0.20, 0.801, 0.01):
        pred = (np.asarray(probability) >= threshold).astype(int)
        macro = f1_score(y_true, pred, average="macro", zero_division=0)
        balanced = balanced_accuracy_score(y_true, pred)
        if (macro, balanced) > (best[1], best[2]):
            best = (float(threshold), float(macro), float(balanced))
    return best


def _inner_splits(X, y, groups, *, n_splits: int, seed: int, classification: bool):
    return [(fold.train_idx, fold.test_idx) for fold in iter_cv_folds(
        X, y, groups=groups, n_splits=n_splits, random_state=seed, classification=classification
    )]


def _probability_oof(pipeline: Pipeline, X, y, cv_splits) -> np.ndarray:
    try:
        return cross_val_predict(pipeline, X, y, cv=cv_splits, method="predict_proba", n_jobs=None)[:, 1]
    except Exception:
        scores = cross_val_predict(pipeline, X, y, cv=cv_splits, method="decision_function", n_jobs=None)
        return 1.0 / (1.0 + np.exp(-np.asarray(scores)))


def evaluate_candidate_inner(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series | None,
    task_type: TaskType,
    view: str,
    requested_k: int | None,
    model_name: str,
    inner_splits: int,
    seed: int,
    include_tabpfn: bool,
) -> CandidateResult:
    try:
        pipeline = build_pipeline(
            task_type=task_type, model_name=model_name, requested_k=requested_k,
            seed=seed, include_tabpfn=include_tabpfn,
        )
        cv_splits = _inner_splits(
            X_train, y_train, groups_train, n_splits=inner_splits,
            seed=seed, classification=task_type == "classification",
        )
        if task_type == "regression":
            pred = cross_val_predict(pipeline, X_train, y_train, cv=cv_splits, n_jobs=None)
            metrics = regression_metrics(y_train, pred, prefix="inner_")
            return CandidateResult(
                view, requested_k, model_name,
                inner_primary=metrics["inner_R2"],
                inner_secondary=-metrics["inner_RMSE"],
            )
        probability = _probability_oof(pipeline, X_train, y_train, cv_splits)
        threshold, macro, balanced = optimise_threshold(y_train, probability)
        return CandidateResult(
            view, requested_k, model_name,
            inner_primary=macro,
            inner_secondary=balanced,
            threshold=threshold,
            oof_probability=probability,
        )
    except Exception as exc:
        return CandidateResult(
            view, requested_k, model_name,
            inner_primary=-np.inf,
            inner_secondary=-np.inf,
            error=f"{type(exc).__name__}: {exc}",
        )


def _fit_platt_calibrator(y_true, probability):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(probability / (1.0 - probability)).reshape(-1, 1)
    calibrator = LogisticRegression(max_iter=2000)
    calibrator.fit(logits, np.asarray(y_true, dtype=int))
    return calibrator


def _apply_platt(calibrator, probability):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(probability / (1.0 - probability)).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


def _conformal_interval(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series | None,
    X_test: pd.DataFrame,
    *,
    alpha: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    folds = list(iter_cv_folds(
        X_train, y_train, groups=groups_train, n_splits=5,
        random_state=seed + 991, classification=False,
    ))
    calibration_fold = folds[0]
    proper_idx, calibration_idx = calibration_fold.train_idx, calibration_fold.test_idx
    conformal_model = clone(pipeline).fit(X_train.iloc[proper_idx], y_train.iloc[proper_idx])
    calibration_pred = conformal_model.predict(X_train.iloc[calibration_idx])
    residuals = np.abs(y_train.iloc[calibration_idx].to_numpy() - calibration_pred)
    n_cal = len(residuals)
    quantile_level = min(1.0, math.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal)
    try:
        q = float(np.quantile(residuals, quantile_level, method="higher"))
    except TypeError:  # numpy<1.22
        q = float(np.quantile(residuals, quantile_level, interpolation="higher"))
    point = np.asarray(conformal_model.predict(X_test), dtype=float)
    return point, point - q, point + q, q, n_cal


def _selected_feature_info(fitted: Pipeline) -> tuple[int, int]:
    variance = fitted.named_steps["variance"]
    selector = fitted.named_steps["selector"]
    n_after_variance = int(np.asarray(variance.get_support()).sum())
    effective_k = int(getattr(selector, "effective_k_", n_after_variance))
    return n_after_variance, effective_k


def _selected_feature_names(fitted: Pipeline) -> list[str]:
    names = fitted.named_steps["nonempty"].get_feature_names_out()
    names = np.asarray(names)[fitted.named_steps["deduplicate"].get_support()]
    names = names[fitted.named_steps["variance"].get_support()]
    names = names[fitted.named_steps["selector"].get_support()]
    return [str(value) for value in names]


def _feature_group(feature_name: str) -> str:
    """Map raw feature names to interpretable scientific groups."""
    name = str(feature_name).lower()
    normalized = (
        name.replace("_", " ")
        .replace("%", " ")
        .replace("/", " ")
    )

    # EP baseline must be checked before generic PHRR/loading matching.
    if "ep_matrix" in name:
        if any(token in name for token in [
            "loading_x_ep_matrix",
            "p_content_x_ep_matrix",
            "synergy_x_ep_matrix",
        ]):
            return "formulation_interaction"
        return "ep_baseline"

    if any(token in name for token in [
        "morgan",
        "maccs",
        "fingerprint",
        "fp_",
    ]):
        return "molecular_fingerprint"

    if any(token in name for token in [
        "molwt",
        "logp",
        "tpsa",
        "numh",
        "ring",
        "fractioncsp3",
        "descriptor",
    ]):
        return "molecular_descriptor"

    if any(token in normalized for token in [
        "loading",
        "fraction",
        "p loading",
    ]):
        return "formulation_loading"

    if (
        "content" in normalized
        or any(token in name for token in [
            "has_p",
            "has_n",
            "has_s",
            "has_b",
            "has_si",
        ])
        or any(token in normalized for token in [
            "n p ratio",
            "s p ratio",
            "b p ratio",
            "si p ratio",
        ])
    ):
        return "elemental_composition"

    if any(token in name for token in [
        "curing",
        "cure_temp",
        "preparation_method",
    ]):
        return "curing_and_preparation"

    if any(token in name for token in [
        "thickness",
        "cone_flux",
        "test_condition",
    ]):
        return "test_condition"

    if "bde" in name:
        return "bde"

    if (
        "interaction" in name
        or "__x__" in name
        or "*" in name
    ):
        return "interaction"

    return "other"
def _normalise_shap_array(values: Any, n_features: int) -> np.ndarray:
    """Convert SHAP outputs from different explainers to (n_samples, n_features)."""
    if isinstance(values, (list, tuple)):
        values = values[-1]
    if hasattr(values, "values"):
        values = values.values
    array = np.asarray(values)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim == 3:
        # Classification explainers commonly return (samples, features, classes).
        if array.shape[1] == n_features:
            array = array[..., -1]
        elif array.shape[2] == n_features:
            array = array[-1, ...]
    if array.ndim != 2:
        raise ValueError(f"Unsupported SHAP array shape: {array.shape}")
    if array.shape[1] != n_features and array.shape[0] == n_features:
        array = array.T
    if array.shape[1] != n_features:
        raise ValueError(f"SHAP feature mismatch: values={array.shape}, expected={n_features}")
    return np.asarray(array, dtype=float)


def _predict_output(model: Any, X: np.ndarray, task_type: TaskType) -> np.ndarray:
    if task_type == "classification":
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def _explain_single_model(
    model: Any,
    background: np.ndarray,
    sample: np.ndarray,
    names: list[str],
    task_type: TaskType,
) -> np.ndarray:
    import shap

    class_name = model.__class__.__name__.lower()
    tree_like = any(token in class_name for token in [
        "xgb", "lgbm", "randomforest", "extratrees", "gradientboosting", "decisiontree",
    ])
    linear_like = any(token in class_name for token in ["ridge", "linearregression", "logisticregression"])

    if tree_like:
        try:
            kwargs: dict[str, Any] = {"data": background, "feature_names": names}
            if task_type == "classification":
                kwargs["model_output"] = "probability"
            explainer = shap.TreeExplainer(model, **kwargs)
            return _normalise_shap_array(explainer(sample), len(names))
        except Exception:
            # Some model/version combinations cannot expose probabilities through
            # TreeExplainer. Fall back to a model-agnostic probability explainer.
            pass

    if linear_like:
        try:
            explainer = shap.LinearExplainer(model, background, feature_names=names)
            return _normalise_shap_array(explainer(sample), len(names))
        except Exception:
            pass

    predict_fn = lambda x: _predict_output(model, x, task_type)
    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.PermutationExplainer(predict_fn, masker, feature_names=names)
    explanation = explainer(
        sample,
        max_evals=max(2 * len(names) + 1, 3),
        silent=True,
    )
    return _normalise_shap_array(explanation, len(names))


def _explain_model_shap(
    model: Any,
    background: np.ndarray,
    sample: np.ndarray,
    names: list[str],
    task_type: TaskType,
) -> np.ndarray:
    """Explain a single estimator or a soft-voting ensemble on a common output scale."""
    if isinstance(model, (VotingRegressor, VotingClassifier)):
        estimators = list(model.estimators_)
        raw_weights = model.weights if model.weights is not None else [1.0] * len(estimators)
        weights = np.asarray(raw_weights, dtype=float)
        if len(weights) != len(estimators):
            weights = np.ones(len(estimators), dtype=float)
        component_values = [
            _explain_single_model(est, background, sample, names, task_type)
            for est in estimators
        ]
        stacked = np.stack(component_values, axis=0)
        return np.average(stacked, axis=0, weights=weights)
    return _explain_single_model(model, background, sample, names, task_type)


def _save_fold_shap(
    fitted: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    task_type: TaskType,
    output_path: Path,
    max_samples: int,
    seed: int,
    outer_fold: int,
    selected_view: str,
    selected_model: str,
    selected_requested_k: int | None,
    selected_effective_k: int,
) -> pd.DataFrame | None:
    try:
        rng = np.random.default_rng(seed)
        background = X_train.iloc[rng.choice(len(X_train), size=min(max_samples, len(X_train)), replace=False)]
        sample = X_test.iloc[rng.choice(len(X_test), size=min(max_samples, len(X_test)), replace=False)]

        def transform(frame: pd.DataFrame):
            value: Any = frame
            for step_name in ["nonempty", "imputer", "deduplicate", "variance", "selector", "scaler"]:
                step = fitted.named_steps[step_name]
                if step == "passthrough":
                    continue
                value = step.transform(value)
            return np.asarray(value)

        background_t = transform(background)
        sample_t = transform(sample)
        model = fitted.named_steps["model"]
        names = _selected_feature_names(fitted)
        array = _explain_model_shap(model, background_t, sample_t, names, task_type)

        result = pd.DataFrame({
            "feature": names,
            "feature_group": [_feature_group(name) for name in names],
            "mean_abs_shap": np.mean(np.abs(array), axis=0),
            "mean_signed_shap": np.mean(array, axis=0),
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        result["rank"] = np.arange(1, len(result) + 1)
        result["outer_fold"] = int(outer_fold)
        result["selected_view"] = selected_view
        result["selected_model"] = selected_model
        result["selected_requested_k"] = selected_requested_k
        result["selected_effective_k"] = int(selected_effective_k)
        result.to_csv(output_path, index=False, encoding="utf-8-sig")

        # V4 paper-figure update: preserve sample-level SHAP values so that a
        # genuine beeswarm and dependence plots can be generated later. Older
        # V4 outputs only stored fold-mean importance and therefore could not
        # reconstruct the distribution or feature-value direction.
        sample_rows: list[dict[str, Any]] = []
        sample_positions = list(sample.index)
        for sample_pos, original_index in enumerate(sample_positions):
            for feature_pos, feature_name in enumerate(names):
                raw_value = np.nan
                if feature_name in sample.columns:
                    raw_value = pd.to_numeric(
                        pd.Series([sample.iloc[sample_pos][feature_name]]), errors="coerce"
                    ).iloc[0]
                sample_rows.append({
                    "outer_fold": int(outer_fold),
                    "sample_index": original_index,
                    "feature": feature_name,
                    "feature_group": _feature_group(feature_name),
                    "feature_value_raw": raw_value,
                    "feature_value_transformed": float(sample_t[sample_pos, feature_pos]),
                    "shap_value": float(array[sample_pos, feature_pos]),
                    "selected_view": selected_view,
                    "selected_model": selected_model,
                    "selected_requested_k": selected_requested_k,
                    "selected_effective_k": int(selected_effective_k),
                })
        sample_path = output_path.with_name(
            output_path.stem.replace("_SHAP", "_SHAP_values_long") + ".csv.gz"
        )
        pd.DataFrame(sample_rows).to_csv(
            sample_path, index=False, encoding="utf-8-sig", compression="gzip"
        )
        return result
    except Exception as exc:
        warnings.warn(f"SHAP fold export skipped: {type(exc).__name__}: {exc}")
        return None


def _aggregate_shap_stability(shap_tables: list[pd.DataFrame], output_dir: Path, task: str) -> None:
    if not shap_tables:
        return

    long = pd.concat(shap_tables, ignore_index=True)
    fold_ids = sorted(pd.to_numeric(long["outer_fold"], errors="coerce").dropna().astype(int).unique())
    all_features = sorted(long["feature"].astype(str).unique())
    n_folds = len(fold_ids)
    missing_rank = len(all_features) + 1

    expanded_rows: list[dict[str, Any]] = []
    for fold_id in fold_ids:
        fold_table = long[long["outer_fold"] == fold_id].copy().set_index("feature")
        for feature in all_features:
            if feature in fold_table.index:
                row = fold_table.loc[feature]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                abs_value = float(row["mean_abs_shap"])
                signed_value = float(row["mean_signed_shap"])
                rank = int(row["rank"])
                group = str(row.get("feature_group", _feature_group(feature)))
                present = 1
            else:
                abs_value = 0.0
                signed_value = 0.0
                rank = missing_rank
                group = _feature_group(feature)
                present = 0
            expanded_rows.append({
                "outer_fold": fold_id,
                "feature": feature,
                "feature_group": group,
                "mean_abs_shap": abs_value,
                "mean_signed_shap": signed_value,
                "rank": rank,
                "selected": present,
                "top20": int(present == 1 and rank <= 20),
            })

    expanded = pd.DataFrame(expanded_rows)
    summary_rows: list[dict[str, Any]] = []
    for feature, sub in expanded.groupby("feature", sort=False):
        present = sub["selected"] == 1
        signed_present = sub.loc[present, "mean_signed_shap"]
        positive = int((signed_present > 0).sum())
        negative = int((signed_present < 0).sum())
        nonzero = positive + negative
        direction_consistency = max(positive, negative) / nonzero if nonzero else np.nan
        summary_rows.append({
            "feature": feature,
            "feature_group": sub["feature_group"].iloc[0],
            "mean_abs_shap_all_folds": float(sub["mean_abs_shap"].mean()),
            "mean_abs_shap_when_selected": float(sub.loc[present, "mean_abs_shap"].mean()) if present.any() else 0.0,
            "mean_signed_shap_all_folds": float(sub["mean_signed_shap"].mean()),
            "mean_signed_shap_when_selected": float(signed_present.mean()) if present.any() else 0.0,
            "selected_feature_frequency": float(sub["selected"].mean()),
            "top20_frequency": float(sub["top20"].mean()),
            "direction_consistency": float(direction_consistency) if pd.notna(direction_consistency) else np.nan,
            "positive_direction_fraction_when_selected": float(positive / nonzero) if nonzero else np.nan,
            "mean_rank_all_folds": float(sub["rank"].mean()),
            "folds_present": int(sub["selected"].sum()),
            "n_shap_folds": n_folds,
        })

    summary = pd.DataFrame(summary_rows).sort_values(
        ["top20_frequency", "selected_feature_frequency", "mean_abs_shap_all_folds"],
        ascending=[False, False, False],
    )
    summary.to_csv(output_dir / f"{task}_SHAP_stability_summary.csv", index=False, encoding="utf-8-sig")

    group_summary = summary.groupby("feature_group", as_index=False).agg(
        mean_abs_shap_all_folds_sum=("mean_abs_shap_all_folds", "sum"),
        mean_abs_shap_all_folds_mean=("mean_abs_shap_all_folds", "mean"),
        max_top20_frequency=("top20_frequency", "max"),
        mean_selected_feature_frequency=("selected_feature_frequency", "mean"),
        n_features=("feature", "nunique"),
    ).sort_values("mean_abs_shap_all_folds_sum", ascending=False)
    group_summary.to_csv(output_dir / f"{task}_SHAP_group_summary.csv", index=False, encoding="utf-8-sig")

    rank_pivot = expanded.pivot(index="feature", columns="outer_fold", values="rank")
    rank_pivot.corr(method="spearman").to_csv(
        output_dir / f"{task}_SHAP_rank_spearman.csv", encoding="utf-8-sig"
    )
    expanded.to_csv(output_dir / f"{task}_SHAP_all_folds_long.csv", index=False, encoding="utf-8-sig")



MATCHING_EP_BASELINE = {
    "LOI": "EP_matrix_LOI",
    "PHRR": "EP_matrix_PHRR",
    "THR": "EP_matrix_THR",
    "Tg": "EP_matrix_Tg",
    "Char_yield": "EP_matrix_CY",
    "TS_MPa": "EP_matrix_TS",
    "FS_MPa": "EP_matrix_FS",
}


def _is_neat_fr_derived_feature(column: Any) -> bool:
    """Return True for a feature that must be neutralized on a neat-EP row.

    Curing-agent-only and test-condition features are deliberately retained.
    FR-loading itself is also retained (and equals zero), because it is a real
    formulation coordinate rather than hypothetical FR identity information.
    """
    name = str(column)
    lower = name.lower()

    if name in {"FR_present", "Main_FR_present", "Co_FR_present"}:
        return True
    if lower.startswith(("main_", "co_")):
        return True

    # Interactions involving phosphorus/FR information must be removed even when
    # the other factor is a curing-agent feature.
    if lower.startswith("p_x_"):
        return True

    # Genuine curing-agent structure/composition is physically present in neat EP.
    if lower.startswith("curing_") or "curingagent_" in lower:
        return False

    # Loading_total_FR is a meaningful zero-valued formulation coordinate.
    if lower.startswith("loading_total_fr"):
        return False

    fr_tokens = (
        "fr_class",
        "preparation_method",
        "synergy",
        "main_fr_fraction",
        "co_fr_fraction",
        "p_content",
        "n_content",
        "s_content",
        "b_content",
        "si_content",
        "has_p",
        "has_n",
        "has_s",
        "has_b",
        "has_si",
        "has_al",
        "n/p",
        "s/p",
        "b/p",
        "si/p",
        "n_p_ratio",
        "s_p_ratio",
        "b_p_ratio",
        "si_p_ratio",
        "p_loading",
        "n_loading",
        "s_loading",
        "b_loading",
        "si_loading",
        "p_n_loading",
        "p_si_loading",
        "p_b_loading",
        "p_s_loading",
        "fr_main_bde",
        "bde_type",
    )
    return any(token in lower for token in fr_tokens)


def _apply_baseline_inclusive_mask(
    matrix: pd.DataFrame,
    df_task: pd.DataFrame,
    colmap: dict[str, str],
    task: str,
    row_policy: str,
) -> pd.DataFrame:
    """Apply the final Model-B-only neat-EP leakage guard.

    ``pipeline_core`` already zeros the main/co molecular blocks and several
    structured FR fields when loading is explicitly zero.  Later feature
    engineering can, however, recreate FR-derived quantities from the original
    metadata.  This task-level guard neutralizes those quantities without
    changing the frozen source CSV or the existing Model-A feature bundle.
    """
    policy = str(row_policy).strip().lower()
    if policy == "modified_only":
        return matrix.copy()
    if policy != "baseline_inclusive":
        raise ValueError("row_policy must be 'modified_only' or 'baseline_inclusive'")

    loading_col = colmap.get("Loading_total_FR wt%", "Loading_total_FR wt%")
    if loading_col not in df_task.columns:
        raise KeyError(f"Missing loading column for baseline-inclusive masking: {loading_col}")

    out = matrix.copy()
    loading = pd.to_numeric(df_task[loading_col], errors="coerce")
    neat = loading.eq(0.0).to_numpy()
    if not neat.any():
        return out

    zero_cols = [c for c in out.columns if _is_neat_fr_derived_feature(c)]
    if zero_cols:
        out.loc[neat, zero_cols] = 0.0

    # For a neat-EP row, the matching EP_matrix value equals the target by
    # definition.  Mask the baseline itself and every derived/interacting feature
    # that contains the same baseline token.  Modified formulations retain it.
    baseline = MATCHING_EP_BASELINE.get(task)
    if baseline:
        baseline_cols = [c for c in out.columns if baseline.lower() in str(c).lower()]
        if baseline_cols:
            out.loc[neat, baseline_cols] = np.nan

    return out


def _test_loading_masks(
    df_task: pd.DataFrame,
    test_idx,
    colmap: dict[str, str],
) -> tuple[np.ndarray, np.ndarray]:
    loading_col = colmap.get("Loading_total_FR wt%", "Loading_total_FR wt%")
    loading = pd.to_numeric(df_task.iloc[test_idx][loading_col], errors="coerce")
    return loading.eq(0.0).to_numpy(), loading.gt(0.0).to_numpy()


def evaluate_task_nested(
    bundle: ScientificBundle,
    task: str,
    output_dir: str | Path,
    options: EvaluationOptions,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if task not in TASK_CONFIGS:
        raise KeyError(f"Unsupported task={task}; available={sorted(TASK_CONFIGS)}")
    config = TASK_CONFIGS[task]
    output_dir = Path(output_dir)
    task_dir = output_dir / task / options.split_strategy / options.screening_mode
    if options.feature_scope != "all":
        task_dir = task_dir / options.feature_scope
    task_dir = task_dir / ("with_BDE" if options.use_bde else "without_BDE")
    task_dir.mkdir(parents=True, exist_ok=True)

    # A task directory encodes split/screening/scope/BDE but historically did not
    # encode row policy or selection scope. Refuse silent overwrites when a user
    # reuses the same --results root for scientifically different scenarios.
    existing_summary_path = task_dir / f"{task}_nested_summary.json"
    if existing_summary_path.exists():
        try:
            previous = json.loads(existing_summary_path.read_text(encoding="utf-8"))
        except Exception:
            previous = {}
        conflicts = []
        for key, current in {
            "row_policy": options.row_policy,
            "selection_scope": options.selection_scope,
            "split_strategy": options.split_strategy,
            "screening_mode": options.screening_mode,
            "feature_scope": options.feature_scope,
            "use_BDE": int(options.use_bde),
        }.items():
            if key in previous and str(previous.get(key)) != str(current):
                conflicts.append(f"{key}: existing={previous.get(key)!r}, requested={current!r}")
        if conflicts:
            raise RuntimeError(
                f"Refusing to overwrite incompatible existing results in {task_dir}. "
                + "; ".join(conflicts)
                + ". Use a different --results directory."
            )

    y = get_target(bundle.base, task)

    valid = core.build_task_valid_mask(
        bundle.df, bundle.base.colmap, task, target=y
    )

    if task in FORMAL_ABSOLUTE_TASKS:
        loading_column = bundle.base.colmap.get(
            "Loading_total_FR wt%",
            "Loading_total_FR wt%",
        )

        if loading_column not in bundle.df.columns:
            raise KeyError(
                f"Missing loading column required by formal row policy: "
                f"{loading_column}"
            )

        loading = pd.to_numeric(
            bundle.df[loading_column],
            errors="coerce",
        )

        row_policy = str(options.row_policy).strip().lower()
        if row_policy == "modified_only":
            valid = valid & loading.gt(0)
        elif row_policy == "baseline_inclusive":
            # Model B retains both confirmed neat EP (0) and modified (>0) rows.
            # Missing loading is not silently reclassified as a baseline row.
            valid = valid & loading.ge(0)
        else:
            raise ValueError(
                "row_policy must be 'modified_only' or 'baseline_inclusive'"
            )

    y_task = y.loc[valid].reset_index(drop=True)
    df_task = bundle.df.loc[valid].reset_index(drop=True)
    reference_col = bundle.base.colmap.get("Reference", "Reference")
    if options.split_strategy == "molecule":
        # Keep the strict workflow consistent with the main project rules.
        # Tg/TS/FS use MAIN+CO+CURING; other tasks use MAIN+CO.
        groups = get_groups(bundle.base, task).loc[valid].reset_index(drop=True)
    else:
        groups = build_group_labels(
            df_task, strategy=options.split_strategy, reference_col=reference_col
        )

    views, requested_ks = _candidate_space(config, bundle, options.selection_scope)
    matrices: dict[str, pd.DataFrame] = {}
    for view in views:
        matrix = _filter_task_matrix(
            bundle, task, view, use_bde=options.use_bde,
            screening_mode=options.screening_mode,
            feature_scope=options.feature_scope,
        ).loc[valid].reset_index(drop=True)
        matrix = _apply_baseline_inclusive_mask(
            matrix, df_task, bundle.base.colmap, task, options.row_policy
        )
        matrices[view] = matrix
    if not matrices:
        raise ValueError(f"No feature views available for task {task}")

    base_X = next(iter(matrices.values()))
    outer_folds = list(iter_cv_folds(
        base_X, y_task, groups=groups, n_splits=options.outer_splits,
        random_state=options.random_state, classification=config.task_type == "classification",
    ))

    if options.model_names:
        model_names = list(options.model_names)
    else:
        model_names = list(DEFAULT_REGRESSION_MODELS if config.task_type == "regression" else DEFAULT_CLASSIFICATION_MODELS)
    if options.include_tabpfn:
        model_names.append("TabPFN")

    fold_rows: list[dict[str, Any]] = []
    prediction_tables: list[pd.DataFrame] = []
    candidate_rows: list[dict[str, Any]] = []
    shap_tables: list[pd.DataFrame] = []

    for outer in outer_folds:
        train_idx, test_idx = outer.train_idx, outer.test_idx
        y_train, y_test = y_task.iloc[train_idx], y_task.iloc[test_idx]
        groups_train = groups.iloc[train_idx].reset_index(drop=True) if groups is not None else None
        if config.task_type == "classification" and (y_train.nunique() < 2 or y_test.nunique() < 2):
            warnings.warn(f"{task} outer fold {outer.fold_id} skipped: one class missing")
            continue

        candidates: list[CandidateResult] = []
        for view, X_all in matrices.items():
            X_train = X_all.iloc[train_idx].reset_index(drop=True)
            min_available = _minimum_inner_available_features(
                X_train, y_train.reset_index(drop=True), groups_train,
                inner_splits=options.inner_splits,
                seed=options.random_state + outer.fold_id,
                classification=config.task_type == "classification",
            )
            ks = _deduplicate_requested_k(requested_ks, min_available)
            for requested_k in ks:
                if requested_k is not None and requested_k > 500 and "TabPFN" in model_names:
                    tabpfn_allowed = False
                else:
                    tabpfn_allowed = True
                for model_name in model_names:
                    if model_name == "TabPFN" and not tabpfn_allowed:
                        continue
                    result = evaluate_candidate_inner(
                        X_train=X_train,
                        y_train=y_train.reset_index(drop=True),
                        groups_train=groups_train,
                        task_type=config.task_type,
                        view=view,
                        requested_k=requested_k,
                        model_name=model_name,
                        inner_splits=options.inner_splits,
                        seed=options.random_state + outer.fold_id,
                        include_tabpfn=options.include_tabpfn,
                    )
                    candidates.append(result)
                    candidate_rows.append({
                        "task": task, "outer_fold": outer.fold_id,
                        "view": view, "requested_k": requested_k,
                        "model": model_name,
                        "inner_primary": result.inner_primary,
                        "inner_secondary": result.inner_secondary,
                        "inner_threshold": result.threshold,
                        "error": result.error,
                    })

        valid_candidates = [c for c in candidates if np.isfinite(c.inner_primary)]
        if not valid_candidates:
            raise RuntimeError(f"No valid inner-CV candidate for {task}, outer fold {outer.fold_id}")
        selected = max(valid_candidates, key=lambda c: (c.inner_primary, c.inner_secondary))
        X_all = matrices[selected.view]
        X_train = X_all.iloc[train_idx].reset_index(drop=True)
        X_test = X_all.iloc[test_idx].reset_index(drop=True)
        fitted = build_pipeline(
            task_type=config.task_type,
            model_name=selected.model_name,
            requested_k=selected.requested_k,
            seed=options.random_state + outer.fold_id,
            include_tabpfn=options.include_tabpfn,
        ).fit(X_train, y_train.reset_index(drop=True))
        n_after_variance, effective_k = _selected_feature_info(fitted)

        base_row: dict[str, Any] = {
            "task": task,
            "outer_fold": outer.fold_id,
            "split_strategy": options.split_strategy,
            "screening_mode": options.screening_mode,
            "row_policy": options.row_policy,
            "use_BDE": int(options.use_bde),
            "descriptor_mode": config.descriptor_mode,
            "use_V7_formula_features": int(config.use_v7_features),
            "selected_view": selected.view,
            "selected_requested_k": selected.requested_k,
            "selected_effective_k": effective_k,
            "n_features_after_variance": n_after_variance,
            "selected_model": selected.model_name,
            "inner_primary": selected.inner_primary,
            "inner_secondary": selected.inner_secondary,
            "n_train": len(train_idx), "n_test": len(test_idx),
            "n_train_groups": int(groups.iloc[train_idx].nunique()) if groups is not None else np.nan,
            "n_test_groups": int(groups.iloc[test_idx].nunique()) if groups is not None else np.nan,
            "group_overlap": group_overlap(groups, train_idx, test_idx),
        }

        prediction_table = df_task.iloc[test_idx].copy()
        keep_meta = [c for c in [
            "Record_ID",
            bundle.base.colmap.get("FR_main"),
            bundle.base.colmap.get("FR_co"),
            bundle.base.colmap.get("SMILES_main"),
            bundle.base.colmap.get("SMILES_co"),
            reference_col,
            "Canonical_SMILES_main",
            "Murcko_scaffold_main",
        ] if c and c in prediction_table.columns]
        prediction_table = prediction_table.loc[:, list(dict.fromkeys(keep_meta))].reset_index(drop=True)
        prediction_table.insert(0, "task", task)
        prediction_table.insert(1, "outer_fold", outer.fold_id)
        prediction_table["y_true"] = y_test.to_numpy()
        loading_col_audit = bundle.base.colmap.get(
            "Loading_total_FR wt%", "Loading_total_FR wt%"
        )
        if loading_col_audit in df_task.columns:
            test_loading_audit = pd.to_numeric(
                df_task.iloc[test_idx][loading_col_audit], errors="coerce"
            ).reset_index(drop=True)
            prediction_table["Loading_total_FR_for_audit"] = test_loading_audit.to_numpy()
            prediction_table["Is_Neat_EP"] = test_loading_audit.eq(0.0).astype(int).to_numpy()
        else:
            prediction_table["Is_Neat_EP"] = 0
        # Persist protocol metadata directly in outer predictions so downstream
        # manuscript audits do not have to infer the scientific scenario from
        # directory names alone.
        prediction_table["split_strategy"] = options.split_strategy
        prediction_table["screening_mode"] = options.screening_mode
        prediction_table["row_policy"] = options.row_policy
        prediction_table["selection_scope"] = options.selection_scope
        prediction_table["use_BDE"] = int(options.use_bde)

        if config.task_type == "regression":
            y_pred = np.asarray(fitted.predict(X_test), dtype=float)
            base_row.update(regression_metrics(y_test, y_pred, prefix="outer_"))
            neat_test_mask = modified_test_mask = None
            if str(options.row_policy).lower() == "baseline_inclusive":
                neat_test_mask, modified_test_mask = _test_loading_masks(
                    df_task, test_idx, bundle.base.colmap
                )
                base_row["n_test_neat"] = int(neat_test_mask.sum())
                base_row["n_test_modified"] = int(modified_test_mask.sum())
                if int(modified_test_mask.sum()) >= 2:
                    base_row.update(
                        regression_metrics(
                            y_test.to_numpy()[modified_test_mask],
                            y_pred[modified_test_mask],
                            prefix="outer_modified_",
                        )
                    )
            conformal_pred, lower, upper, q, n_cal = _conformal_interval(
                fitted, X_train, y_train.reset_index(drop=True), groups_train, X_test,
                alpha=options.conformal_alpha, seed=options.random_state + outer.fold_id,
            )
            covered = (y_test.to_numpy() >= lower) & (y_test.to_numpy() <= upper)
            base_row.update({
                "conformal_alpha": options.conformal_alpha,
                "conformal_q": q,
                "conformal_n_calibration": n_cal,
                "outer_PICP": float(covered.mean()),
                "outer_MPIW": float(np.mean(upper - lower)),
            })
            if modified_test_mask is not None and int(modified_test_mask.sum()) >= 1:
                base_row["outer_modified_PICP"] = float(covered[modified_test_mask].mean())
                base_row["outer_modified_MPIW"] = float(
                    np.mean((upper - lower)[modified_test_mask])
                )
            prediction_table["y_pred"] = y_pred
            prediction_table["conformal_y_pred"] = conformal_pred
            prediction_table["PI_lower"] = lower
            prediction_table["PI_upper"] = upper
            prediction_table["PI_covered"] = covered.astype(int)
        else:
            raw_probability = np.asarray(fitted.predict_proba(X_test)[:, 1], dtype=float)
            # Fit probability calibration and final threshold only from outer-training OOF probabilities.
            if selected.oof_probability is None:
                inner_cv = _inner_splits(
                    X_train, y_train.reset_index(drop=True), groups_train,
                    n_splits=options.inner_splits, seed=options.random_state + outer.fold_id,
                    classification=True,
                )
                oof_probability = _probability_oof(
                    build_pipeline(
                        task_type="classification", model_name=selected.model_name,
                        requested_k=selected.requested_k,
                        seed=options.random_state + outer.fold_id,
                        include_tabpfn=options.include_tabpfn,
                    ),
                    X_train, y_train.reset_index(drop=True), inner_cv,
                )
            else:
                oof_probability = selected.oof_probability
            calibrator = _fit_platt_calibrator(y_train.reset_index(drop=True), oof_probability)
            calibrated_oof = _apply_platt(calibrator, oof_probability)
            final_threshold, _, _ = optimise_threshold(y_train.reset_index(drop=True), calibrated_oof)
            probability = _apply_platt(calibrator, raw_probability)
            y_pred = (probability >= final_threshold).astype(int)
            base_row["selected_threshold"] = final_threshold
            base_row.update(classification_metrics(y_test, y_pred, probability, prefix="outer_"))
            if str(options.row_policy).lower() == "baseline_inclusive":
                neat_test_mask, modified_test_mask = _test_loading_masks(
                    df_task, test_idx, bundle.base.colmap
                )
                base_row["n_test_neat"] = int(neat_test_mask.sum())
                base_row["n_test_modified"] = int(modified_test_mask.sum())
                if int(modified_test_mask.sum()) >= 2:
                    base_row.update(
                        classification_metrics(
                            y_test.to_numpy()[modified_test_mask],
                            y_pred[modified_test_mask],
                            probability[modified_test_mask],
                            prefix="outer_modified_",
                        )
                    )
            prediction_table["raw_V0_probability"] = raw_probability
            prediction_table["calibrated_V0_probability"] = probability
            prediction_table["threshold"] = final_threshold
            prediction_table["y_pred"] = y_pred

            fraction_pos, mean_pred = calibration_curve(y_test, probability, n_bins=8, strategy="quantile")
            pd.DataFrame({"mean_predicted_probability": mean_pred, "fraction_positive": fraction_pos}).to_csv(
                task_dir / f"{task}_outer_fold_{outer.fold_id}_calibration_curve.csv",
                index=False, encoding="utf-8-sig",
            )

        if options.shap_stability:
            shap_result = _save_fold_shap(
                fitted, X_train, X_test, task_type=config.task_type,
                output_path=task_dir / f"{task}_outer_fold_{outer.fold_id}_SHAP.csv",
                max_samples=options.max_shap_samples,
                seed=options.random_state + outer.fold_id,
                outer_fold=outer.fold_id,
                selected_view=selected.view,
                selected_model=selected.model_name,
                selected_requested_k=selected.requested_k,
                selected_effective_k=effective_k,
            )
            if shap_result is not None:
                shap_tables.append(shap_result)

        fold_rows.append(base_row)
        prediction_tables.append(prediction_table)
        # Use an absolute path and recreate the parent directory before saving.
        model_path = task_dir / f"{task}_outer_fold_{outer.fold_id}_model.joblib"

        if not model_path.is_absolute():
            project_root = Path(__file__).resolve().parents[1]
            model_path = project_root / model_path

        model_path = model_path.resolve()
        model_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            joblib.dump(fitted, model_path)
        except FileNotFoundError:
            # OneDrive or another process may temporarily disturb the directory.
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(fitted, model_path)

        print(f"[SAVE] model={model_path}")

    folds_df = pd.DataFrame(fold_rows)
    predictions_df = pd.concat(prediction_tables, ignore_index=True) if prediction_tables else pd.DataFrame()
    candidates_df = pd.DataFrame(candidate_rows)
    folds_df.to_csv(task_dir / f"{task}_outer_fold_metrics.csv", index=False, encoding="utf-8-sig")
    predictions_df.to_csv(task_dir / f"{task}_outer_predictions.csv", index=False, encoding="utf-8-sig")
    candidates_df.to_csv(task_dir / f"{task}_inner_candidate_results.csv", index=False, encoding="utf-8-sig")

    metric_columns = [c for c in folds_df.columns if c.startswith("outer_") and c != "outer_fold"]
    summary: dict[str, Any] = {
        "task": task,
        "task_type": config.task_type,
        "split_strategy": options.split_strategy,
        "screening_mode": options.screening_mode,
        "row_policy": options.row_policy,
        "feature_scope": options.feature_scope,
        "use_BDE": int(options.use_bde),
        "descriptor_mode": config.descriptor_mode,
        "use_V7_formula_features": int(config.use_v7_features),
        "configured_current_view": config.current_view,
        "configured_current_k": config.current_k,
        "outer_splits_requested": options.outer_splits,
        "outer_splits_completed": len(folds_df),
        "inner_splits": options.inner_splits,
        "selection_scope": options.selection_scope,
        "selection_rule": (
            (
                "feature view and K predefined before FINAL evaluation from development-stage sensitivity analyses; "
                "only the predictive model is selected using grouped inner CV within each outer-training set; "
                "UL94 threshold is estimated using outer-training data only; outer fold used once for evaluation"
            )
            if options.selection_scope == "fixed"
            else (
                "feature view, K and model selected using grouped inner CV within each outer-training set; "
                "UL94 threshold also selected using outer-training data only; outer fold used once for evaluation"
            )
        ),
        "formal_row_policy": (
            (
                "Loading_total_FR > 0 for absolute-property tasks; "
                "Neat EP retained for database tracing and Delta construction"
            )
            if task in FORMAL_ABSOLUTE_TASKS and options.row_policy == "modified_only"
            else (
                "Baseline-inclusive absolute-property evaluation: valid Loading_total_FR >= 0; "
                "neat-EP FR-derived features masked; task-matching EP_matrix feature masked "
                "on neat-EP rows; overall and modified-test-subset metrics reported"
            )
            if task in FORMAL_ABSOLUTE_TASKS and options.row_policy == "baseline_inclusive"
            else "task-specific valid target mask"
        ),
        "n_valid_formal_rows": int(valid.sum()),
        "n_neat_rows": int(
            pd.to_numeric(
                df_task[bundle.base.colmap.get("Loading_total_FR wt%", "Loading_total_FR wt%")],
                errors="coerce",
            ).eq(0.0).sum()
        ) if task in FORMAL_ABSOLUTE_TASKS else 0,
        "n_modified_rows": int(
            pd.to_numeric(
                df_task[bundle.base.colmap.get("Loading_total_FR wt%", "Loading_total_FR wt%")],
                errors="coerce",
            ).gt(0.0).sum()
        ) if task in FORMAL_ABSOLUTE_TASKS else int(valid.sum()),
        "neat_fr_feature_masking": bool(
            task in FORMAL_ABSOLUTE_TASKS and options.row_policy == "baseline_inclusive"
        ),
        "matching_baseline_leakage_guard": bool(
            task in MATCHING_EP_BASELINE and options.row_policy == "baseline_inclusive"
        ),
    }
    for column in metric_columns:
        values = pd.to_numeric(folds_df[column], errors="coerce")
        summary[f"{column}_mean"] = float(values.mean())
        summary[f"{column}_std"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0
    if not folds_df.empty:
        summary["selected_view_frequency"] = json.dumps(Counter(folds_df["selected_view"]).most_common(), ensure_ascii=False)
        summary["selected_model_frequency"] = json.dumps(Counter(folds_df["selected_model"]).most_common(), ensure_ascii=False)
        summary["selected_k_frequency"] = json.dumps(Counter(folds_df["selected_requested_k"].astype(str)).most_common(), ensure_ascii=False)
    pd.DataFrame([summary]).to_csv(task_dir / f"{task}_nested_summary.csv", index=False, encoding="utf-8-sig")
    (task_dir / f"{task}_nested_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if options.shap_stability:
        _aggregate_shap_stability(shap_tables, task_dir, task)

    # Refit a deployable model using the configuration most frequently selected
    # by inner CV across outer folds. Outer-test scores are not used to choose it.
    if not folds_df.empty:
        normalized_k = folds_df["selected_requested_k"].apply(
            lambda value: "ALL" if pd.isna(value) else int(value)
        )
        config_counts = Counter(
            zip(folds_df["selected_view"], normalized_k, folds_df["selected_model"])
        )
        final_view, final_k_token, final_model_name = config_counts.most_common(1)[0][0]
        final_k = None if final_k_token == "ALL" else int(final_k_token)
        final_X = matrices[str(final_view)]
        final_pipeline = build_pipeline(
            task_type=config.task_type, model_name=str(final_model_name),
            requested_k=None if final_k is None else int(final_k),
            seed=options.random_state, include_tabpfn=options.include_tabpfn,
        )
        model_bundle: dict[str, Any] = {
            "task": task,
            "task_type": config.task_type,
            "view": str(final_view),
            "requested_k": final_k,
            "model_name": str(final_model_name),
            "split_strategy": options.split_strategy,
            "screening_mode": options.screening_mode,
            "row_policy": options.row_policy,
            "feature_scope": options.feature_scope,
            "use_BDE": options.use_bde,
            "descriptor_mode": config.descriptor_mode,
            "use_V7_formula_features": config.use_v7_features,
        }
        if config.task_type == "regression":
            final_pipeline.fit(final_X, y_task)
            _, _, _, final_q, final_n_cal = _conformal_interval(
                final_pipeline, final_X, y_task, groups, final_X.iloc[:1],
                alpha=options.conformal_alpha, seed=options.random_state + 2026,
            )
            model_bundle.update({
                "pipeline": final_pipeline,
                "conformal_q": final_q,
                "conformal_alpha": options.conformal_alpha,
                "conformal_n_calibration": final_n_cal,
            })
        else:
            final_cv = _inner_splits(
                final_X, y_task, groups, n_splits=options.inner_splits,
                seed=options.random_state + 2026, classification=True,
            )
            final_oof = _probability_oof(final_pipeline, final_X, y_task, final_cv)
            final_calibrator = _fit_platt_calibrator(y_task, final_oof)
            final_calibrated_oof = _apply_platt(final_calibrator, final_oof)
            final_threshold, _, _ = optimise_threshold(y_task, final_calibrated_oof)
            final_pipeline.fit(final_X, y_task)
            model_bundle.update({
                "pipeline": final_pipeline,
                "probability_calibrator": final_calibrator,
                "threshold": final_threshold,
            })
        joblib.dump(model_bundle, task_dir / f"{task}_final_model_bundle.joblib")
        (task_dir / f"{task}_final_model_metadata.json").write_text(
            json.dumps({k: v for k, v in model_bundle.items() if k not in {"pipeline", "probability_calibrator"}}, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    return folds_df, predictions_df, summary


def run_nested_project(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    tasks: Iterable[str],
    split_strategies: Iterable[str],
    screening_modes: Iterable[str],
    options: EvaluationOptions,
    feature_scopes: Iterable[str] = ("all",),
    bde_modes: Iterable[bool] = (False,),
) -> pd.DataFrame:
    bundle = prepare_scientific_bundle(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save standardized, non-destructive data audit once per run.
    audit_dir = output_dir / "data_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    bundle.df.to_csv(audit_dir / "data_standardized_with_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"alias_view": alias, "canonical_view": canonical}
        for alias, canonical in bundle.view_aliases.items()
    ]).to_csv(audit_dir / "deduplicated_feature_view_aliases.csv", index=False, encoding="utf-8-sig")

    summaries: list[dict[str, Any]] = []
    for split_strategy in split_strategies:
        for screening_mode in screening_modes:
            for feature_scope in feature_scopes:
                for use_bde in bde_modes:
                    scenario_options = EvaluationOptions(**{
                        **options.__dict__,
                        "split_strategy": split_strategy,
                        "screening_mode": screening_mode,
                        "feature_scope": feature_scope,
                        "use_bde": use_bde,
                    })
                    for task in tasks:
                        print("=" * 88)
                        print(
                            f"[NESTED] task={task} split={split_strategy} mode={screening_mode} "
                            f"scope={feature_scope} row_policy={scenario_options.row_policy} "
                            f"BDE={'with' if use_bde else 'without'}"
                        )
                        _, _, summary = evaluate_task_nested(bundle, task, output_dir, scenario_options)
                        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "scientific_nested_summary_all.csv", index=False, encoding="utf-8-sig")
    (output_dir / "run_config.json").write_text(json.dumps({
        "input_path": str(input_path),
        "tasks": list(tasks),
        "split_strategies": list(split_strategies),
        "screening_modes": list(screening_modes),
        "feature_scopes": list(feature_scopes),
        "bde_modes": ["with_BDE" if value else "without_BDE" for value in bde_modes],
        "options": options.__dict__,
        "task_feature_settings": {
            name: {
                "descriptor_mode": config.descriptor_mode,
                "use_V7_formula_features": config.use_v7_features,
                "current_view": config.current_view,
                "current_k": config.current_k,
            }
            for name, config in TASK_CONFIGS.items()
        },
        "feature_view_aliases_removed": bundle.view_aliases,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_df
