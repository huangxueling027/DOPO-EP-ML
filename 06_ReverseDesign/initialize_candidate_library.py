# -*- coding: utf-8 -*-
"""Initialize the DOPO candidate library from the existing experimental database.

Outputs
-------
training_seed_molecules.csv
    One row per unique main flame-retardant molecule already present in training.
design_seed_priority.csv
    Twenty chemically diverse/high-performing training molecules to use as design
    references. These are references, not new virtual-screening discoveries.
manual_candidates_template.csv
    Empty manual-entry table. Draw new candidates in ChemDraw and paste exported
    SMILES here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from candidate_library import (
    ROOT, canonicalize, dopo_count, element_mass_percent, empty_manual_template,
    infer_family, infer_reactive_groups, murcko, normalize_preparation, read_csv_auto,
)
from rdkit.Chem import Descriptors


def robust_score(series: pd.Series, direction: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index)
    lo, hi = valid.quantile([0.05, 0.95])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = valid.min(), valid.max()
    if hi <= lo:
        score = pd.Series(0.5, index=series.index)
    else:
        score = ((values.clip(lo, hi) - lo) / (hi - lo)).clip(0, 1)
    return 1 - score if direction == "min" else score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, default=ROOT / "data" / "DOPO_EP_new_with_BDE.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "06_ReverseDesign" / "data")
    parser.add_argument("--top-seeds", type=int, default=20)
    args = parser.parse_args()

    df = read_csv_auto(args.training)
    required = ["FR_main", "SMILES_main"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required training columns: {missing}")

    working = df.copy()
    working["Canonical_SMILES"] = working["SMILES_main"].map(lambda x: canonicalize(x)[0])
    working = working[working["Canonical_SMILES"].ne("")].copy()
    if "Loading_total_FR wt%" in working.columns:
        working["_loading"] = pd.to_numeric(working["Loading_total_FR wt%"], errors="coerce").fillna(0)
    else:
        working["_loading"] = 0.0
    positive = working[working["_loading"] > 0].copy()
    if positive.empty:
        positive = working.copy()

    identity_cols = [c for c in ["FR_main", "SMILES_main", "Canonical_SMILES", "Synergy_type", "Preparation_Method", "Reference"] if c in positive.columns]
    identity = positive.sort_values("_loading", ascending=False).drop_duplicates("Canonical_SMILES")[identity_cols].copy()
    identity = identity.reset_index(drop=True)
    identity.insert(0, "Candidate_ID", [f"TRAIN_{i:03d}" for i in range(1, len(identity) + 1)])
    identity = identity.rename(columns={"FR_main": "Candidate_Name", "SMILES_main": "SMILES_raw", "Reference": "Reference_ID"})
    identity["Source_Type"] = "training_seed"

    mols = [canonicalize(s)[1] for s in identity["Canonical_SMILES"]]
    identity["Design_Family"] = [infer_family(m) for m in mols]
    identity["DOPO_Count"] = [dopo_count(m) for m in mols]
    identity["Linker_Type"] = "review_manually"
    identity["Reactive_Group"] = [infer_reactive_groups(m) for m in mols]
    identity["Preparation_Type"] = [normalize_preparation(x, r) for x, r in zip(identity.get("Preparation_Method", ""), identity["Reactive_Group"])]
    identity["Synthesis_Level"] = "A"
    identity["Molecular_Weight"] = [float(Descriptors.MolWt(m)) if m is not None else np.nan for m in mols]
    identity["Murcko_Scaffold"] = [murcko(m) for m in mols]
    for element in ("P", "N", "S", "B", "Si"):
        identity[f"{element}_molecular_wt_pct"] = [element_mass_percent(m, element) for m in mols]
    identity["Training_Duplicate"] = True
    identity["Screening_Eligible"] = False
    identity["Notes"] = "Existing training molecule; use as design reference, not as a new discovery."

    # Aggregate measured properties at molecule level using medians from positive-loading rows.
    metric_map = {
        "LOI": ("LOI", "max"), "PHRR": ("PHRR_kw_㎡", "min"),
        "THR": ("THR_MJ_㎡", "min"), "Tg": ("Tg_℃", "max"),
        "TS": ("TS_MPa", "max"),
    }
    agg = positive.groupby("Canonical_SMILES", as_index=False).agg(
        n_formulations=("Canonical_SMILES", "size"),
        median_loading=("_loading", "median"),
        **{f"median_{name}": (column, "median") for name, (column, _) in metric_map.items() if column in positive.columns},
    )
    seeds = identity.merge(agg, on="Canonical_SMILES", how="left")
    score_cols = []
    for name, (_, direction) in metric_map.items():
        column = f"median_{name}"
        if column in seeds.columns:
            sc = f"score_{name}"
            seeds[sc] = robust_score(seeds[column], direction)
            score_cols.append(sc)
    seeds["available_objectives"] = seeds[score_cols].notna().sum(axis=1)
    seeds["design_priority_score"] = seeds[score_cols].mean(axis=1, skipna=True)
    # Avoid selecting only one family: first retain the best five per family, then global top.
    pool = seeds.sort_values("design_priority_score", ascending=False).groupby("Design_Family", group_keys=False).head(5)
    priority = pool.sort_values(["design_priority_score", "available_objectives"], ascending=[False, False]).head(args.top_seeds).copy()
    priority.insert(0, "Design_Seed_Rank", np.arange(1, len(priority) + 1))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    identity.to_csv(args.output_dir / "training_seed_molecules.csv", index=False, encoding="utf-8-sig")
    priority.to_csv(args.output_dir / "design_seed_priority.csv", index=False, encoding="utf-8-sig")
    template = args.output_dir / "manual_candidates_template.csv"
    if not template.exists():
        empty_manual_template().to_csv(template, index=False, encoding="utf-8-sig")

    print(f"[OK] Unique training main molecules: {len(identity)}")
    print(f"[OK] Design-priority seed molecules: {len(priority)}")
    print(f"[OK] Saved: {args.output_dir / 'training_seed_molecules.csv'}")
    print(f"[OK] Saved: {args.output_dir / 'design_seed_priority.csv'}")
    print(f"[NEXT] Fill new/literature structures in: {template}")


if __name__ == "__main__":
    main()
