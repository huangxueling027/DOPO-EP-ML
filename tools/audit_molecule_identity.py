# -*- coding: utf-8 -*-
"""Audit full-component molecular identities and alias/conflict groups."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import pipeline_core as core
from common.chem_standardization import standardize_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_data = ROOT / "data" / "DOPO_EP_new_with_BDE.csv"
    if not default_data.exists():
        default_data = ROOT / "data" / "DOPO_EP_new.csv"
    parser.add_argument("--input", type=Path, default=default_data)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "audit" / "molecule_identity")
    args = parser.parse_args()

    raw = core.read_csv_auto(str(args.input))
    colmap = core.resolve_columns(raw)
    clean = core.clean_dataframe(raw, colmap)
    audited = standardize_dataset(clean, colmap)

    raw_main_col = colmap.get("SMILES_main")
    raw_co_col = colmap.get("SMILES_co")
    raw_curing_col = colmap.get("SMILES_Curing_Agent")
    raw_pair = (
        "MAIN=" + (audited[raw_main_col].fillna("").astype(str).str.strip() if raw_main_col else "")
        + "|CO=" + (audited[raw_co_col].fillna("").astype(str).str.strip() if raw_co_col else "")
    )
    audited["Raw_main_co_pair"] = raw_pair
    audited["Canonical_main_co_curing"] = (
        audited["Canonical_main_co_pair"] + "|CURING=" + audited["Canonical_SMILES_curing"].fillna("")
    )

    # Multiple raw spellings mapped to one full-component canonical identity.
    alias = (
        audited.groupby("Canonical_main_co_pair", dropna=False)["Raw_main_co_pair"]
        .agg(n_raw_spellings="nunique", raw_spellings=lambda x: " || ".join(sorted(set(map(str, x)))))
        .reset_index()
    )
    alias = alias.loc[alias["n_raw_spellings"] > 1].sort_values("n_raw_spellings", ascending=False)

    # Name conflicts: one canonical structure represented by multiple material names.
    main_name_col = colmap.get("FR_main")
    if main_name_col:
        name_conflicts = (
            audited.groupby("Canonical_SMILES_main", dropna=False)[main_name_col]
            .agg(n_names="nunique", names=lambda x: " || ".join(sorted(set(str(v).strip() for v in x if pd.notna(v)))))
            .reset_index()
        )
        name_conflicts = name_conflicts.loc[name_conflicts["n_names"] > 1].sort_values("n_names", ascending=False)
    else:
        name_conflicts = pd.DataFrame()

    args.output.mkdir(parents=True, exist_ok=True)
    audited.to_csv(args.output / "molecule_identity_audit.csv", index=False, encoding="utf-8-sig")
    alias.to_csv(args.output / "canonical_alias_groups.csv", index=False, encoding="utf-8-sig")
    name_conflicts.to_csv(args.output / "canonical_name_conflicts.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame([{
        "n_rows": len(audited),
        "n_unique_raw_main_co": int(audited["Raw_main_co_pair"].nunique()),
        "n_unique_canonical_main_co": int(audited["Canonical_main_co_pair"].nunique()),
        "n_unique_canonical_main_co_curing": int(audited["Canonical_main_co_curing"].nunique()),
        "n_alias_groups": len(alias),
        "n_multicomponent_rows": int(audited["Multicomponent_flag"].sum()),
        "n_invalid_main_smiles": int((~audited["SMILES_main_valid"]).sum()),
    }])
    summary.to_csv(args.output / "molecule_identity_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(f"[DONE] Identity audit: {args.output}")


if __name__ == "__main__":
    main()
