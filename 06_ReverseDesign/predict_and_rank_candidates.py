# -*- coding: utf-8 -*-
"""Predict six DOPO/EP properties, score applicability domain, and Pareto-rank candidates.

This script uses the frozen deployable model bundles from the strict 5x5 nested
molecule-grouped validation. It never retrains or tunes models on candidates.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from candidate_library import read_csv_auto
from common.applicability_domain import build_training_reference, score_candidates
from common.pareto_ranking import Objective, rank_candidates
from common import pipeline_core as core
from common.final_result_paths import load_final_bundle
from common.scientific_evaluation import (
    _apply_baseline_inclusive_mask, _apply_feature_scope,
    _is_advanced_only_descriptor, _is_v7_formula_feature, TASK_CONFIGS,
)

TASKS = ["LOI", "PHRR", "THR", "UL94_V0", "Tg", "TS_MPa"]


def build_required_views(combined: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build only compact, descriptors and Morgan-r3 views needed by six models."""
    colmap = core.resolve_columns(combined)
    df = core.clean_dataframe(combined, colmap)
    if "P_content wt%" in colmap:
        df["P_loading"] = pd.to_numeric(df[colmap["P_content wt%"]], errors="coerce")
    else:
        df["P_loading"] = np.nan
    colmap["P_loading"] = "P_loading"
    if "UL94" in colmap:
        df["UL94_V0"] = df[colmap["UL94"]].map(
            lambda v: np.nan if pd.isna(v) or not str(v).strip() else int(str(v).strip().upper() == "V-0")
        )
        colmap["UL94_V0"] = "UL94_V0"

    old_advanced = core.USE_ADVANCED_DESCRIPTORS
    old_v7 = core.USE_V7_FORMULA_FEATURES
    old_bde = core.USE_BDE_FEATURES
    core.USE_ADVANCED_DESCRIPTORS = True
    core.USE_V7_FORMULA_FEATURES = True
    core.USE_BDE_FEATURES = True
    try:
        compact, _ = core.build_feature_matrix(df, colmap, fp_bits=128, use_p_interactions=False, fp_type="morgan", radius=2)
        descriptors, _ = core.build_feature_matrix(df, colmap, fp_bits=0, use_p_interactions=False, fp_type="descriptors", radius=0)
        morgan_r3, _ = core.build_feature_matrix(df, colmap, fp_bits=512, use_p_interactions=False, fp_type="morgan", radius=3)
    finally:
        core.USE_ADVANCED_DESCRIPTORS = old_advanced
        core.USE_V7_FORMULA_FEATURES = old_v7
        core.USE_BDE_FEATURES = old_bde
    return df, {"compact": compact, "descriptors": descriptors, "morgan_r3": morgan_r3}


def deployment_matrix(
    views: dict[str, pd.DataFrame],
    task: str,
    bundle: dict,
    df_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Build the frozen FINAL deployment matrix.

    The baseline-inclusive neat-EP masking is applied here as a safety guard.
    Candidate screening normally uses positive loading only, but this keeps the
    deployment path identical to the formal V5 protocol if a zero-loading row
    is ever supplied.
    """
    view = str(bundle["view"])
    if view not in views:
        raise KeyError(f"Required view {view} was not built; available={sorted(views)}")
    matrix = core.filter_features_for_task(views[view], task)
    if str(bundle.get("descriptor_mode", TASK_CONFIGS[task].descriptor_mode)) == "basic":
        matrix = matrix.drop(columns=[c for c in matrix if _is_advanced_only_descriptor(c)], errors="ignore")
    if not bool(bundle.get("use_V7_formula_features", TASK_CONFIGS[task].use_v7_features)):
        matrix = matrix.drop(columns=[c for c in matrix if _is_v7_formula_feature(c)], errors="ignore")
    if not bool(bundle.get("use_BDE", False)):
        matrix = matrix.drop(columns=[c for c in matrix if "bde" in str(c).lower()], errors="ignore")
    if str(bundle.get("screening_mode", "formulation")) == "molecular":
        matrix = matrix.drop(columns=[c for c in matrix if "ep_matrix" in str(c).lower()], errors="ignore")
    matrix = _apply_feature_scope(matrix, str(bundle.get("feature_scope", "all")))
    matrix = matrix.loc[:, ~matrix.columns.duplicated()].copy()
    colmap = core.resolve_columns(df_rows)
    matrix = _apply_baseline_inclusive_mask(
        matrix, df_rows, colmap, task, "baseline_inclusive"
    )
    return matrix
PREDICTION_COLUMNS = {
    "LOI": "LOI_pred", "PHRR": "PHRR_pred", "THR": "THR_pred",
    "UL94_V0": "V0_probability", "Tg": "Tg_pred", "TS_MPa": "TS_pred",
}
TARGET_COLUMNS = {
    "LOI": "LOI", "PHRR": "PHRR_kw_㎡", "THR": "THR_MJ_㎡",
    "Tg": "Tg_℃", "TS_MPa": "TS_MPa",
}


def locate_bundle(results_root: Path, task: str) -> Path:
    """Return the unique frozen FINAL bundle; never search/fallback by mtime."""
    bundle, path = load_final_bundle(task)
    return path



def repair_sklearn_compatibility(model) -> None:
    """Repair harmless private-attribute changes when loading older sklearn bundles."""
    try:
        steps = model.steps if hasattr(model, "steps") else []
    except Exception:
        steps = []
    for _, step in steps:
        if hasattr(step, "_fit_dtype") and not hasattr(step, "_fill_dtype"):
            step._fill_dtype = step._fit_dtype
        if hasattr(step, "steps"):
            repair_sklearn_compatibility(step)

def calibrated_probability(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    pipeline = bundle["pipeline"]
    if hasattr(pipeline, "predict_proba"):
        raw = np.asarray(pipeline.predict_proba(X))[:, 1]
    elif hasattr(pipeline, "decision_function"):
        decision = np.asarray(pipeline.decision_function(X), dtype=float)
        raw = 1.0 / (1.0 + np.exp(-decision))
    else:
        raw = np.asarray(pipeline.predict(X), dtype=float)
    calibrator = bundle.get("probability_calibrator")
    if calibrator is not None:
        raw = np.asarray(calibrator.predict_proba(raw.reshape(-1, 1)))[:, 1]
    return np.clip(raw, 0.0, 1.0)


def add_chemical_space(output: pd.DataFrame, training: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
    training_unique = (
        training[["FR_main", "SMILES_main"]]
        .dropna(subset=["SMILES_main"])
        .copy()
    )

    training_unique["_canonical"] = training_unique["SMILES_main"].map(
        lambda value: (
            Chem.MolToSmiles(Chem.MolFromSmiles(str(value)), canonical=True)
            if Chem.MolFromSmiles(str(value)) is not None
            else np.nan
        )
    )

    training_unique = (
        training_unique
        .dropna(subset=["_canonical"])
        .drop_duplicates("_canonical")
        .reset_index(drop=True)
    )
    candidate_unique = (
        output[["Candidate_ID", "Candidate_Name", "SMILES_main"]]
        .drop_duplicates("Candidate_ID")
        .copy()
    )

    candidate_unique["_canonical"] = candidate_unique["SMILES_main"].map(
        lambda value: (
            Chem.MolToSmiles(Chem.MolFromSmiles(str(value)), canonical=True)
            if Chem.MolFromSmiles(str(value)) is not None
            else np.nan
        )
    )

    candidate_unique = (
        candidate_unique
        .dropna(subset=["_canonical"])
        .drop_duplicates("_canonical")
    )
    records = []
    fps = []
    for status, frame, name_col in [
        ("Training", training_unique, "FR_main"), ("Candidate", candidate_unique, "Candidate_Name")
    ]:
        for _, row in frame.iterrows():
            mol = Chem.MolFromSmiles(str(row["SMILES_main"]))
            if mol is None:
                continue
            fps.append(np.asarray(generator.GetFingerprintAsNumPy(mol), dtype=float))
            records.append({
                "candidate_status": status,
                "Candidate_ID": row.get("Candidate_ID", ""),
                "space_name": row.get(name_col, ""),
                "space_smiles": Chem.MolToSmiles(mol, canonical=True),
            })
    if len(fps) < 3:
        return output, pd.DataFrame(records)
    coords = PCA(n_components=2, random_state=42).fit_transform(np.vstack(fps))
    space = pd.DataFrame(records)
    space["PC1"] = coords[:, 0]
    space["PC2"] = coords[:, 1]
    candidate_coords = space[space.candidate_status.eq("Candidate")][["Candidate_ID", "PC1", "PC2"]]
    return output.merge(candidate_coords, on="Candidate_ID", how="left"), space


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formulations", type=Path, default=ROOT / "results" / "06_ReverseDesign" / "FINAL_fixed" / "candidate_formulation_grid.csv")
    parser.add_argument("--training", type=Path, default=ROOT / "data" / "DOPO_EP_new_with_BDE.csv")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "06_ReverseDesign" / "FINAL_fixed")
    parser.add_argument("--reliable-threshold", type=float, default=0.70)
    parser.add_argument("--caution-threshold", type=float, default=0.50)
    args = parser.parse_args()

    formulations = read_csv_auto(args.formulations)
    training = read_csv_auto(args.training)
    if formulations.empty:
        raise ValueError("Candidate formulation grid is empty")

    # Append candidates to training before feature construction. This guarantees
    # identical one-hot columns and feature names to those used for model fitting.
    combined = pd.concat([training, formulations], ignore_index=True, sort=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared_rows, required_views = build_required_views(combined)
    candidate_positions = np.arange(len(training), len(training) + len(formulations))
    predicted = formulations.reset_index(drop=True).copy()
    model_manifest = []

    for task in TASKS:
        bundle, bundle_path = load_final_bundle(task)
        view = str(bundle["view"])
        X_all = deployment_matrix(required_views, task, bundle, prepared_rows)
        X_candidate = X_all.iloc[candidate_positions].copy()
        pipeline = bundle["pipeline"]
        repair_sklearn_compatibility(pipeline)
        pred_col = PREDICTION_COLUMNS[task]
        if task == "UL94_V0":
            probability = calibrated_probability(bundle, X_candidate)
            predicted[pred_col] = probability
            threshold = float(bundle.get("threshold", 0.5))
            predicted["UL94_V0_predicted_class"] = (probability >= threshold).astype(int)
            predicted["UL94_uncertainty"] = 4.0 * probability * (1.0 - probability)
        else:
            values = np.asarray(pipeline.predict(X_candidate), dtype=float)
            predicted[pred_col] = values
            q = float(bundle.get("conformal_q", np.nan))
            predicted[f"{task}_PI_lower"] = values - q if np.isfinite(q) else np.nan
            predicted[f"{task}_PI_upper"] = values + q if np.isfinite(q) else np.nan
            predicted[f"{task}_PI_width"] = 2.0 * q if np.isfinite(q) else np.nan
            target_col = TARGET_COLUMNS[task]
            target = pd.to_numeric(training[target_col], errors="coerce").dropna() if target_col in training else pd.Series(dtype=float)
            if not target.empty:
                low, high = target.quantile([0.01, 0.99])
                predicted[f"{task}_within_training_1_99pct"] = values >= low
                predicted[f"{task}_within_training_1_99pct"] &= values <= high
        predicted[f"{task}_model"] = bundle.get("model_name")
        predicted[f"{task}_view"] = bundle.get("view")
        predicted[f"{task}_K"] = "ALL" if bundle.get("requested_k") is None else bundle.get("requested_k")
        model_manifest.append({
            "task": task, "bundle": str(bundle_path), "model": bundle.get("model_name"),
            "view": bundle.get("view"), "K": bundle.get("requested_k"),
            "use_BDE": bundle.get("use_BDE"),
        })
        print(f"[OK] Predicted {task}: n={len(predicted)}, model={bundle.get('model_name')}, view={view}")

    # Applicability domain at unique-molecule level, then merge to all scenarios.
    unique_candidates = predicted[["Candidate_ID", "Candidate_Name", "SMILES_main"]].drop_duplicates("Candidate_ID")
    references = build_training_reference(training, smiles_col="SMILES_main", name_col="FR_main", radius=3, n_bits=2048)
    ad = score_candidates(
        unique_candidates, references,
        candidate_smiles_col="SMILES_main", candidate_name_col="Candidate_Name",
        radius=3, n_bits=2048,
        reliable_threshold=args.reliable_threshold, caution_threshold=args.caution_threshold,
        top_n=3,
    )
    ad_columns = [c for c in ad.columns if c not in {"Candidate_Name", "SMILES_main"}]
    predicted = predicted.merge(ad[ad_columns], on="Candidate_ID", how="left")
    predicted["max_similarity"] = pd.to_numeric(predicted["Max_Tanimoto_to_training"], errors="coerce")
    predicted["in_domain"] = predicted["Applicability_domain"].eq("In_domain")

    # A compact uncertainty summary for figures; individual widths remain available.
    uncertainty_parts = []
    for task in ["LOI", "PHRR", "THR", "Tg", "TS_MPa"]:
        c = f"{task}_PI_width"
        if c in predicted:
            values = pd.to_numeric(predicted[c], errors="coerce")
            scale = float(values.median()) if values.notna().any() and float(values.median()) > 0 else 1.0
            uncertainty_parts.append(values / scale)
    if "UL94_uncertainty" in predicted:
        uncertainty_parts.append(pd.to_numeric(predicted["UL94_uncertainty"], errors="coerce"))
    predicted["uncertainty"] = pd.concat(uncertainty_parts, axis=1).mean(axis=1) if uncertainty_parts else np.nan

    predicted, chemical_space = add_chemical_space(predicted, training)

    # ============================================================
    # Eligibility filtering before Pareto ranking
    # ============================================================

    # Regression predictions must remain inside the central
    # 1st–99th percentile range of the training targets.
    range_cols = [
        "LOI_within_training_1_99pct",
        "PHRR_within_training_1_99pct",
        "THR_within_training_1_99pct",
        "Tg_within_training_1_99pct",
        "TS_MPa_within_training_1_99pct",
    ]

    # These columns are required for formal candidate eligibility.
    required_eligibility_cols = [
        "smiles_valid",
        "contains_DOPO",
        "candidate_feasible",
        "in_domain",
        *range_cols,
    ]

    missing_eligibility_cols = [
        c for c in required_eligibility_cols
        if c not in predicted.columns
    ]

    if missing_eligibility_cols:
        raise RuntimeError(
            "Missing required candidate eligibility columns: "
            + ", ".join(missing_eligibility_cols)
        )

    # Normalize eligibility columns to strict Boolean values.
    for c in required_eligibility_cols:
        predicted[c] = predicted[c].fillna(False).astype(bool)

    # Separate the eligibility rules so the screening process
    # can be audited later.
    predicted["eligible_structure"] = (
        predicted["smiles_valid"]
        & predicted["contains_DOPO"]
        & predicted["candidate_feasible"]
    )

    predicted["eligible_ad"] = predicted["in_domain"]

    predicted["eligible_target_range"] = (
        predicted[range_cols].all(axis=1)
    )

    predicted["eligible_for_pareto"] = (
        predicted["eligible_structure"]
        & predicted["eligible_ad"]
        & predicted["eligible_target_range"]
    )

    # Keep all predictions for auditing, but only reliable
    # formulations are allowed to enter formal Pareto ranking.
    eligible = predicted.loc[
        predicted["eligible_for_pareto"]
    ].copy()

    if eligible.empty:
        raise RuntimeError(
            "No candidate formulation passed the formal eligibility rules. "
            "Check AD thresholds and training-range filters."
        )

    print("=" * 78)
    print("[CANDIDATE ELIGIBILITY]")
    print("=" * 78)
    print(f"[INFO] All formulations      : {len(predicted)}")
    print(f"[INFO] Eligible formulations : {len(eligible)}")
    print(
        f"[INFO] Eligible molecules    : "
        f"{eligible['Candidate_ID'].nunique()}"
    )


    # ============================================================
    # Multi-objective Pareto ranking
    # Only eligible formulations enter this stage.
    # ============================================================

    objectives = [
        Objective("LOI_pred", "max", 1.0),
        Objective("V0_probability", "max", 1.0),
        Objective("PHRR_pred", "min", 1.0),
        Objective("THR_pred", "min", 1.0),
        Objective("Tg_pred", "max", 0.5),
        Objective("TS_pred", "max", 0.5),
    ]

    ranked_all = rank_candidates(
        eligible,
        objectives,
        uncertainty_columns=[
            "LOI_PI_width",
            "PHRR_PI_width",
            "THR_PI_width",
            "Tg_PI_width",
            "TS_MPa_PI_width",
            "UL94_uncertainty",
        ],
        loading_column="Loading_total_FR wt%",
        loading_reference=10.0,
        uncertainty_penalty_weight=0.15,
        ad_penalty_weight=0.20,
        loading_penalty_weight=0.10,
    )

    ranked_all["pareto_flag"] = (
        ranked_all["Pareto_front"].astype(bool)
    )

    # Select the best eligible formulation for each molecule.
    best = (
        ranked_all
        .sort_values(
            ["Pareto_rank", "Final_ranking_score"],
            ascending=[True, False],
        )
        .drop_duplicates("Candidate_ID")
        .copy()
    )

    best = (
        best
        .sort_values(
            ["Pareto_rank", "Final_ranking_score"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )

    best["Candidate_rank"] = np.arange(
        1,
        len(best) + 1
    )

    # Formal Pareto-front candidate pool.
    pareto_pool = (
        best.loc[best["Pareto_rank"].eq(1)]
        .sort_values(
            "Final_ranking_score",
            ascending=False,
        )
        .reset_index(drop=True)
        .copy()
    )

    pareto_pool["Pareto_candidate_rank"] = np.arange(
        1,
        len(pareto_pool) + 1
    )
    
    # ============================================================
    # Save candidate screening outputs
    # ============================================================

    # 1. Complete prediction table:
    #    contains every formulation, including excluded ones.
    predicted.to_csv(
        args.output_dir / "candidate_predictions_all_formulations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 2. Formulations passing all formal eligibility rules.
    eligible.to_csv(
        args.output_dir / "candidate_eligible_formulations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 3. Eligible formulations after Pareto / score ranking.
    ranked_all.to_csv(
        args.output_dir / "candidate_eligible_ranked_formulations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 4. Best eligible formulation for each candidate molecule.
    best.to_csv(
        args.output_dir / "candidate_best_per_molecule.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Retain legacy file names for downstream paper scripts.
    best.to_csv(
        args.output_dir / "candidate_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    best.to_csv(
        args.output_dir / "ranked_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 5. Formal Pareto-front molecule pool.
    pareto_pool.to_csv(
        args.output_dir / "candidate_pareto_pool.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(model_manifest).to_csv(
        args.output_dir / "candidate_model_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    chemical_space.to_csv(
        args.output_dir / "candidate_chemical_space.csv",
        index=False,
        encoding="utf-8-sig",
    )
    # ============================================================
    # Screening funnel
    # Counts must be monotonically non-increasing.
    # ============================================================

    molecule_level = (
        predicted
        .sort_values("Formulation_ID")
        .drop_duplicates("Candidate_ID")
    )

    valid_ids = set(
        molecule_level.loc[
            molecule_level["smiles_valid"],
            "Candidate_ID",
        ]
    )

    dopo_ids = set(
        molecule_level.loc[
            molecule_level["smiles_valid"]
            & molecule_level["contains_DOPO"],
            "Candidate_ID",
        ]
    )

    feasible_ids = set(
        molecule_level.loc[
            molecule_level["smiles_valid"]
            & molecule_level["contains_DOPO"]
            & molecule_level["candidate_feasible"],
            "Candidate_ID",
        ]
    )

    in_domain_ids = set(
        predicted.loc[
            predicted["eligible_structure"]
            & predicted["eligible_ad"],
            "Candidate_ID",
        ]
    )

    eligible_ids = set(
        eligible["Candidate_ID"]
    )

    pareto_ids = set(
        ranked_all.loc[
            ranked_all["Pareto_front"],
            "Candidate_ID",
        ]
    )

    funnel = pd.DataFrame([
        {
            "stage": "Original candidate molecules",
            "count": int(
                molecule_level["Candidate_ID"].nunique()
            ),
        },
        {
            "stage": "Valid SMILES",
            "count": len(valid_ids),
        },
        {
            "stage": "Contains DOPO",
            "count": len(dopo_ids),
        },
        {
            "stage": "Candidate feasible for screening",
            "count": len(feasible_ids),
        },
        {
            "stage": "At least one in-domain formulation",
            "count": len(in_domain_ids),
        },
        {
            "stage": "At least one formally eligible formulation",
            "count": len(eligible_ids),
        },
        {
            "stage": "Pareto-front candidate molecules",
            "count": len(pareto_ids),
        },
    ])

    funnel.to_csv(
        args.output_dir / "candidate_screening_funnel.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.output_dir / "candidate_run_config.json").write_text(json.dumps({
        "tasks": TASKS, "training": str(args.training), "formulations": str(args.formulations),
        "reliable_threshold": args.reliable_threshold, "caution_threshold": args.caution_threshold,
        "models": model_manifest,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 78)
    print("[CANDIDATE SCREENING COMPLETE]")
    print("=" * 78)

    print(
        f"[DONE] All formulations: "
        f"{args.output_dir / 'candidate_predictions_all_formulations.csv'}"
    )

    print(
        f"[DONE] Eligible formulations: "
        f"{args.output_dir / 'candidate_eligible_formulations.csv'}"
    )

    print(
        f"[DONE] Best eligible formulation per molecule: "
        f"{args.output_dir / 'candidate_best_per_molecule.csv'}"
    )

    print(
        f"[DONE] Pareto candidate pool: "
        f"{args.output_dir / 'candidate_pareto_pool.csv'}"
    )

    print(
        f"[INFO] Eligible molecules: "
        f"{best['Candidate_ID'].nunique()}"
    )

    print(
        f"[INFO] Pareto-front molecules: "
        f"{pareto_pool['Candidate_ID'].nunique()}"
    )

if __name__ == "__main__":
    main()
