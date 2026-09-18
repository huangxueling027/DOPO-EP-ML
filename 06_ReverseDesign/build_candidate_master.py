# -*- coding: utf-8 -*-
"""Build the deduplicated DOPO molecule master from training and manual entries."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from candidate_library import ROOT, enrich_candidate_rows, read_csv_auto


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-seeds", type=Path, default=ROOT / "06_ReverseDesign" / "data" / "training_seed_molecules.csv")
    parser.add_argument("--manual", type=Path, default=ROOT / "06_ReverseDesign" / "data" / "manual_candidates_template.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "06_ReverseDesign" / "candidate_molecule_master.csv")
    args = parser.parse_args()

    if not args.training_seeds.exists():
        raise FileNotFoundError("Run `python -u run.py candidate-init` first.")
    seeds = read_csv_auto(args.training_seeds)
    training_canonical = seeds.get("Canonical_SMILES", pd.Series(dtype=str)).dropna().astype(str).tolist()

    manual = read_csv_auto(args.manual) if args.manual.exists() else pd.DataFrame()
    manual = manual.dropna(how="all").copy()
    if not manual.empty:
        manual = enrich_candidate_rows(manual, training_canonical)
        # Fill IDs only when omitted; never overwrite user IDs.
        missing_id = manual["Candidate_ID"].isna() | manual["Candidate_ID"].astype(str).str.strip().eq("")
        manual.loc[missing_id, "Candidate_ID"] = [f"CAND_{i:04d}" for i in range(1, int(missing_id.sum()) + 1)]
        manual["Source_Type"] = manual["Source_Type"].replace("", "designed").fillna("designed")
    else:
        manual = enrich_candidate_rows(pd.DataFrame(columns=["Candidate_ID", "Candidate_Name", "SMILES_raw"]), training_canonical)

    # Harmonise training seed columns with the validation flags used for manual rows.
    seed_master = seeds.copy()
    for column, value in {
        "smiles_valid": True, "contains_DOPO": True, "is_unique": True,
        "synthesis_feasible": True, "Training_Duplicate": True,
        "Screening_Eligible": False, "Exclusion_Reason": "training_reference",
    }.items():
        seed_master[column] = value
    if "Computed_DOPO_Count" not in seed_master:
        seed_master["Computed_DOPO_Count"] = seed_master.get("DOPO_Count", 1)
    if "Preparation_Method" not in seed_master:
        seed_master["Preparation_Method"] = seed_master.get("Preparation_Type", "")

    master = pd.concat([seed_master, manual], ignore_index=True, sort=False)
    master.insert(0, "Library_Row_ID", [f"LIB_{i:04d}" for i in range(1, len(master) + 1)])
    master["candidate_status"] = master["Source_Type"].astype(str).map(
        lambda x: "Training reference" if x == "training_seed" else "Candidate"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(args.output, index=False, encoding="utf-8-sig")
    exclusions = master[~master["Screening_Eligible"].fillna(False)].copy()
    exclusions.to_csv(args.output.with_name("candidate_exclusion_report.csv"), index=False, encoding="utf-8-sig")
    eligible = master[master["Screening_Eligible"].fillna(False)]

    print(f"[OK] Library rows: {len(master)}")
    print(f"[OK] Training references: {(master.candidate_status == 'Training reference').sum()}")
    print(f"[OK] New/literature candidate rows: {(master.candidate_status == 'Candidate').sum()}")
    print(f"[OK] Screening-eligible candidates: {len(eligible)}")
    print(f"[OK] Saved: {args.output}")
    if eligible.empty:
        print("[NEXT] Add candidate structures to 06_ReverseDesign/data/manual_candidates_template.csv")


if __name__ == "__main__":
    main()
