# -*- coding: utf-8 -*-
"""Create formulation/test-condition scenarios for eligible DOPO candidates.

The script does not invent arbitrary conditions. For each preparation method it
uses Q25/Q50/Q75 positive loadings and representative curing/test conditions
from the existing experimental database. Every chosen value is written to an
audit CSV.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from candidate_library import ROOT, PREPARATION_NUM, normalize_preparation, read_csv_auto


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace("%", "", regex=False), errors="coerce")


def mode_or_nan(series: pd.Series):
    value = series.dropna().astype(str).str.strip()
    value = value[value.ne("")]
    return value.mode().iloc[0] if not value.empty else np.nan


def median_or_nan(frame: pd.DataFrame, column: str):
    if column not in frame:
        return np.nan
    values = numeric(frame[column]).dropna()
    return float(values.median()) if not values.empty else np.nan


def atom_flags(smiles: object) -> dict[str, int]:
    mol = Chem.MolFromSmiles(str(smiles)) if pd.notna(smiles) and str(smiles).strip() else None
    present = {a.GetSymbol() for a in mol.GetAtoms()} if mol is not None else set()
    return {f"CuringAgent_Has_{e}": int(e in present) for e in ("N", "S", "P", "B", "F", "Cl")}


def ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if np.isfinite(numerator) and np.isfinite(denominator) and denominator > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, default=ROOT / "results" / "06_ReverseDesign" / "candidate_molecule_master.csv")
    parser.add_argument("--training", type=Path, default=ROOT / "data" / "DOPO_EP_new_with_BDE.csv")
    parser.add_argument("--max-loading", type=float, default=30.0)
    parser.add_argument("--loadings", default="", help="Optional fixed loading grid, e.g. 5,10,15")
    parser.add_argument("--cone-fluxes", default="35,50")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "06_ReverseDesign" / "candidate_formulation_grid.csv")
    args = parser.parse_args()

    master = read_csv_auto(args.master)
    training = read_csv_auto(args.training)
    eligible = master[master.get("Screening_Eligible", False).fillna(False).astype(bool)].copy()
    if eligible.empty:
        raise ValueError("No eligible candidates. Fill the manual candidate template and run candidate-build first.")

    if "Loading_total_FR wt%" not in training:
        raise KeyError("Training data lacks Loading_total_FR wt%")
    training = training.copy()
    training["_loading"] = numeric(training["Loading_total_FR wt%"])
    if "Preparation_Method" in training.columns:
        training["Preparation_Method"] = [
            normalize_preparation(value, "none")
            for value in training["Preparation_Method"]
        ]
    positive = training[training["_loading"] > 0].copy()
    cone_fluxes = [float(x) for x in args.cone_fluxes.split(",") if x.strip()]
    fixed_loadings = [float(x) for x in args.loadings.split(",") if x.strip()]

    rows: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    formulation_counter = 1
    for candidate in eligible.to_dict("records"):
        method = normalize_preparation(
            candidate.get("Preparation_Method", "DOPO-based (additive)"),
            str(candidate.get("Reactive_Group", "none")),
        )
        method_rows = positive[
            positive.get(
                "Preparation_Method",
                pd.Series(index=positive.index, dtype=str),
            ).astype(str).eq(method)
        ].copy()
        if method_rows.empty:
            method_rows = positive.copy()
            method_source = "global_positive_training_fallback"
        else:
            method_source = "same_preparation_method_training"

        if fixed_loadings:
            loadings = fixed_loadings
            loading_source = "user_fixed"
        else:
            q = method_rows["_loading"].dropna().quantile([0.25, 0.50, 0.75]).to_numpy(dtype=float)
            loadings = sorted({round(float(min(x, args.max_loading)), 2) for x in q if np.isfinite(x) and x > 0})
            loading_source = "Q25_Q50_Q75_training"
        if not loadings:
            raise ValueError(f"No positive loading scenarios available for {method}")

        curing = mode_or_nan(method_rows["Curing_Agent"]) if "Curing_Agent" in method_rows else np.nan
        curing_smiles = np.nan
        if "SMILES_Curing_Agent" in method_rows:
            pair = method_rows[["Curing_Agent", "SMILES_Curing_Agent"]].dropna()
            if pd.notna(curing) and not pair.empty:
                matched = pair[pair["Curing_Agent"].astype(str).eq(str(curing))]
                curing_smiles = mode_or_nan(matched["SMILES_Curing_Agent"]) if not matched.empty else mode_or_nan(pair["SMILES_Curing_Agent"])
        cure_temp = median_or_nan(method_rows, "Cure_Temp_Max")
        loi_thickness = median_or_nan(method_rows, "LOI_Thickness_mm")
        ul94_thickness = median_or_nan(method_rows, "UL94_Thickness_mm")
        cone_thickness = median_or_nan(method_rows, "Cone_Thickness_mm")

        baseline_cols = ["EP_matrix_LOI", "EP_matrix_PHRR", "EP_matrix_THR", "EP_matrix_Tg", "EP_matrix_CY", "EP_matrix_TS", "EP_matrix_FS"]
        baselines = {column: median_or_nan(method_rows, column) for column in baseline_cols if column in training}
        curing_flags = atom_flags(curing_smiles)

        for loading in loadings:
            for cone_flux in cone_fluxes:
                row = {column: np.nan for column in training.columns if not column.startswith("_")}
                row.update({
                    "Formulation_ID": f"FORM_{formulation_counter:06d}",
                    "Candidate_ID": candidate.get("Candidate_ID"),
                    "Candidate_Name": candidate.get("Candidate_Name"),
                    "Source_Type": candidate.get("Source_Type"),
                    "Design_Family": candidate.get("Design_Family"),
                    "Synthesis_Level": candidate.get("Synthesis_Level"),

                    "synthesis_feasible": candidate.get(
                        "synthesis_feasible",
                        False,
                    ),

                    "candidate_feasible": candidate.get(
                        "candidate_feasible",
                        candidate.get(
                            "synthesis_feasible",
                            False,
                        ),
                    ),

                    "smiles_valid": candidate.get(
                        "smiles_valid",
                        False,
                    ),

                    "contains_DOPO": candidate.get(
                        "contains_DOPO",
                        False,
                    ),

                    "is_unique": candidate.get(
                        "is_unique",
                        False,
                    ),
                    "candidate_status": "Candidate",
                    "FR_main": candidate.get("Candidate_Name"),
                    "FR_co": np.nan,
                    "Curing_Agent": curing,
                    "Preparation_Method": method,
                    "Preparation_Method_num": PREPARATION_NUM.get(method, candidate.get("Preparation_Method_num", 0)),
                    "Synergy_flag(single=0&synergy=1)": int(str(candidate.get("Design_Family", "P-only")) != "P-only"),
                    "Synergy_type": candidate.get("Design_Family", "P-only"),
                    "Main_FR_fraction": "100.00%",
                    "Co_FR_fraction": "0.00%",
                    "SMILES_main": candidate.get("Canonical_SMILES", candidate.get("SMILES_raw")),
                    "SMILES_co": np.nan,
                    "SMILES_Curing_Agent": curing_smiles,
                    "Loading_total_FR wt%": float(loading),
                    "Cure_Temp_Max": cure_temp,
                    "LOI_Thickness_mm": loi_thickness,
                    "UL94_Thickness_mm": ul94_thickness,
                    "Cone_Thickness_mm": cone_thickness,
                    "Cone_flux_kW_m2": float(cone_flux),
                    "Reference": "Virtual screening scenario",
                })
                row.update(baselines)
                row.update(curing_flags)

                # Molecular element percentages are converted to formulation-level
                # element contents: molecular wt% × total FR loading fraction.
                element_system = {}
                for element in ("P", "N", "S", "B", "Si"):
                    molecular_pct = pd.to_numeric(pd.Series([candidate.get(f"{element}_molecular_wt_pct")]), errors="coerce").iloc[0]
                    system_pct = float(molecular_pct * loading / 100.0) if np.isfinite(molecular_pct) else 0.0
                    row[f"{element}_content wt%"] = system_pct
                    row[f"Has_{element}"] = int(system_pct > 0)
                    element_system[element] = system_pct
                p = element_system["P"]
                for element in ("N", "S", "B", "Si"):
                    row[f"{element}/P ratio"] = ratio(element_system[element], p)
                rows.append(row)
                formulation_counter += 1

        audit.append({
            "Candidate_ID": candidate.get("Candidate_ID"), "Preparation_Method": method,
            "condition_source": method_source, "loading_source": loading_source,
            "loadings_wt_percent": ";".join(map(str, loadings)),
            "cone_fluxes_kW_m2": ";".join(map(str, cone_fluxes)),
            "Curing_Agent": curing, "SMILES_Curing_Agent": curing_smiles,
            "Cure_Temp_Max": cure_temp, "LOI_Thickness_mm": loi_thickness,
            "UL94_Thickness_mm": ul94_thickness, "Cone_Thickness_mm": cone_thickness,
        })

    grid = pd.DataFrame(rows)
    # Keep candidate identifiers first, then the original project schema.
    first = [c for c in [
        "Formulation_ID",
        "Candidate_ID",
        "Candidate_Name",
        "Source_Type",
        "Design_Family",
        "Synthesis_Level",
        "synthesis_feasible",
        "candidate_feasible",
        "smiles_valid",
        "contains_DOPO",
        "is_unique",
        "candidate_status"
    ] if c in grid]
    rest = [c for c in grid.columns if c not in first]
    grid = grid[first + rest]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(args.output, index=False, encoding="utf-8-sig")
    pd.DataFrame(audit).to_csv(args.output.with_name("candidate_formulation_condition_audit.csv"), index=False, encoding="utf-8-sig")
    print(f"[OK] Eligible molecules: {eligible['Candidate_ID'].nunique()}")
    print(f"[OK] Formulation scenarios: {len(grid)}")
    print(f"[OK] Saved: {args.output}")


if __name__ == "__main__":
    main()
