# -*- coding: utf-8 -*-
"""
Build the final two-source candidate master:

1. Existing training-seed + designed candidate master
2. PubChem public candidates that passed:
   standardization -> deduplication -> applicability-domain filtering

Important:
For PubChem candidates, preparation method is a rule-based screening
scenario inferred from molecular functional groups. It is NOT treated
as experimentally verified synthesis/processing information.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from candidate_library import (
    ROOT,
    PREPARATION_NUM,
    infer_family,
    infer_reactive_groups,
    normalize_preparation,
    read_csv_auto,
)


# ============================================================
# Paths
# ============================================================

BASE_MASTER = (
    ROOT
    / "results"
    / "06_ReverseDesign"
    / "candidate_molecule_master.csv"
)

PUBLIC_POOL = (
    ROOT
    / "06_ReverseDesign"
    / "public_database"
    / "data"
    / "03_public_candidates"
    / "pubchem_public_candidates_for_prediction.csv"
)

OUT_DIR = (
    ROOT
    / "results"
    / "06_ReverseDesign"
    / "FINAL_fixed"
    / "combined"
)

OUT_MASTER = (
    OUT_DIR
    / "candidate_molecule_master_combined.csv"
)

OUT_AUDIT = (
    OUT_DIR
    / "combined_candidate_master_audit.csv"
)

OUT_SOURCE = (
    OUT_DIR
    / "combined_candidate_source_summary.csv"
)


# ============================================================
# Helpers
# ============================================================

def bool_series(frame, column, default=False):
    if column not in frame.columns:
        return pd.Series(
            default,
            index=frame.index,
            dtype=bool,
        )

    s = frame[column]

    if pd.api.types.is_bool_dtype(s):
        return s.fillna(default)

    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
        })
        .fillna(default)
        .astype(bool)
    )


def preparation_type_from_method(method):
    mapping = {
        "DOPO-based (additive)":
            "Additive",

        "DOPO-based (reactive)":
            "Reactive",

        "DOPO-based (Co-curing)":
            "Co-curing",

        "DOPO-based (Additive + Secondary Crosslinking)":
            "Additive + Secondary Crosslinking",
        "DOPO-based (additive+ Secondary Crosslinking)":
            "Additive + Secondary Crosslinking",
    }

    return mapping.get(
        str(method),
        "Additive",
    )


def first_available(row, columns, default=np.nan):
    for col in columns:
        if col in row.index:
            value = row[col]

            if pd.notna(value):
                text = str(value).strip()

                if text:
                    return value

    return default


# ============================================================
# Main
# ============================================================

def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not BASE_MASTER.exists():
        raise FileNotFoundError(
            f"Missing base candidate master:\n"
            f"{BASE_MASTER}"
        )

    if not PUBLIC_POOL.exists():
        raise FileNotFoundError(
            f"Missing PubChem prediction pool:\n"
            f"{PUBLIC_POOL}"
        )

    base = read_csv_auto(BASE_MASTER)
    public = read_csv_auto(PUBLIC_POOL)

    print("=" * 78)
    print("BUILD COMBINED CANDIDATE MASTER")
    print("=" * 78)

    print(
        "[LOAD] base master rows:",
        len(base),
    )

    print(
        "[LOAD] public prediction rows:",
        len(public),
    )

    # --------------------------------------------------------
    # Existing master: add unified downstream field
    # --------------------------------------------------------

    base = base.copy()

    base["candidate_feasible"] = (
        bool_series(
            base,
            "synthesis_feasible",
            default=False,
        )
    )

    # Existing screening logic remains unchanged.
    base["Screening_Eligible"] = (
        bool_series(
            base,
            "Screening_Eligible",
            default=False,
        )
    )

    # Explicit provenance
    base["Public_Record"] = False
    base["Public_Preparation_Known"] = False

    base["Preparation_Assumption"] = np.where(
        base["Source_Type"]
        .astype(str)
        .str.lower()
        .eq("designed"),
        "designed_library_rule",
        "training_record",
    )

    # --------------------------------------------------------
    # PubChem candidate enrichment
    # --------------------------------------------------------

    public = public.copy()

    required_public = [
        "Candidate_ID",
        "Candidate_Name",
        "Canonical_SMILES",
        "Applicability_domain",
        "Max_Tanimoto_to_training",
    ]

    missing = [
        c
        for c in required_public
        if c not in public.columns
    ]

    if missing:
        raise RuntimeError(
            "Public candidate pool is missing columns: "
            + ", ".join(missing)
        )

    # Formal safety gate.
    if not public[
        "Applicability_domain"
    ].eq("In_domain").all():
        raise RuntimeError(
            "Public prediction pool contains "
            "non-In_domain candidates."
        )

    # Model identity must remain unique.
    if (
        "Model_Identity_SMILES"
        in public.columns
        and public[
            "Model_Identity_SMILES"
        ].duplicated().any()
    ):
        raise RuntimeError(
            "Duplicate Model_Identity_SMILES "
            "found in public prediction pool."
        )

    mols = [
        Chem.MolFromSmiles(str(smi))
        if pd.notna(smi)
        else None
        for smi in public["Canonical_SMILES"]
    ]

    if any(mol is None for mol in mols):
        raise RuntimeError(
            "Invalid Canonical_SMILES exists "
            "in formal public prediction pool."
        )

    # --------------------------------------------------------
    # Family + functional-group inference
    # --------------------------------------------------------

    public["Design_Family"] = [
        infer_family(mol)
        for mol in mols
    ]

    public["Synergy_type"] = (
        public["Design_Family"]
    )

    public["Reactive_Group"] = [
        infer_reactive_groups(mol)
        for mol in mols
    ]

    # --------------------------------------------------------
    # Preparation method
    #
    # This is a screening scenario only.
    # It is NOT literature-verified processing information.
    # --------------------------------------------------------

    public["Preparation_Method"] = [
        normalize_preparation(
            "",
            reactive_group,
        )
        for reactive_group
        in public["Reactive_Group"]
    ]

    public["Preparation_Method_num"] = (
        public["Preparation_Method"]
        .map(PREPARATION_NUM)
    )

    public["Preparation_Type"] = (
        public["Preparation_Method"]
        .map(preparation_type_from_method)
    )

    public["Public_Preparation_Known"] = False

    public["Preparation_Assumption"] = (
        "rule_based_from_structure_for_screening_only"
    )

    # --------------------------------------------------------
    # Standard candidate-master fields
    # --------------------------------------------------------

    public["Source_Type"] = "pubchem"

    public["Public_Record"] = True

    public["Reference_ID"] = (
        "PubChem CID "
        + public["cid"]
        .astype("Int64")
        .astype(str)
        if "cid" in public.columns
        else public["Candidate_ID"]
    )

    public["SMILES_raw"] = np.where(
        public.get(
            "SMILES_raw",
            pd.Series(
                np.nan,
                index=public.index,
            )
        ).notna(),
        public.get(
            "SMILES_raw",
            public["Canonical_SMILES"],
        ),
        public["Canonical_SMILES"],
    )

    public["DOPO_Count"] = pd.to_numeric(
        public.get(
            "Computed_DOPO_Count",
            np.nan,
        ),
        errors="coerce",
    )

    public["Molecular_Weight"] = pd.to_numeric(
        public.get(
            "Computed_MW",
            public.get(
                "Molecular_Weight",
                np.nan,
            ),
        ),
        errors="coerce",
    )

    if "Murcko_Scaffold_SMILES" in public.columns:
        public["Murcko_Scaffold"] = (
            public[
                "Murcko_Scaffold_SMILES"
            ]
        )

    public["Synthesis_Level"] = (
        "PUBLIC_UNVERIFIED"
    )

    # Do NOT call PubChem records
    # experimentally synthesis-feasible.
    public["synthesis_feasible"] = False

    # But they passed the structural + AD gates
    # and can enter virtual screening.
    public["candidate_feasible"] = True

    public["Screening_Eligible"] = True

    public["Training_Duplicate"] = False

    public["smiles_valid"] = True
    public["contains_DOPO"] = True
    public["is_unique"] = True

    public["candidate_status"] = "Candidate"

    public["Exclusion_Reason"] = (
        "eligible_public_in_domain"
    )

    public["Notes"] = (
        "PubChem public-database candidate; "
        "standardized, deduplicated and inside "
        "the strict applicability domain. "
        "Preparation method is a rule-based "
        "screening scenario only; synthesis "
        "feasibility is not experimentally verified."
    )

    # --------------------------------------------------------
    # Element flags
    # --------------------------------------------------------

    for element in (
        "P",
        "N",
        "S",
        "B",
        "Si",
    ):
        col = (
            f"{element}_molecular_wt_pct"
        )

        values = pd.to_numeric(
            public.get(
                col,
                pd.Series(
                    0.0,
                    index=public.index,
                ),
            ),
            errors="coerce",
        ).fillna(0.0)

        public[f"Has_{element}"] = (
            values > 0
        ).astype(int)

    # --------------------------------------------------------
    # Library row IDs
    # --------------------------------------------------------

    public["Library_Row_ID"] = [
        f"PUBLIB_{i:04d}"
        for i in range(
            1,
            len(public) + 1,
        )
    ]

    # --------------------------------------------------------
    # No Candidate_ID collision
    # --------------------------------------------------------

    base_ids = set(
        base["Candidate_ID"]
        .astype(str)
    )

    public_ids = set(
        public["Candidate_ID"]
        .astype(str)
    )

    overlap_ids = (
        base_ids
        & public_ids
    )

    if overlap_ids:
        raise RuntimeError(
            "Candidate_ID collision between "
            "existing and public libraries: "
            + ", ".join(
                sorted(overlap_ids)[:10]
            )
        )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    combined = pd.concat(
        [
            base,
            public,
        ],
        ignore_index=True,
        sort=False,
    )

    # Safety
    if combined[
        "Candidate_ID"
    ].astype(str).duplicated().any():
        raise RuntimeError(
            "Combined master contains duplicate "
            "Candidate_ID values."
        )

    # --------------------------------------------------------
    # Audit
    # --------------------------------------------------------

    source_norm = (
        combined["Source_Type"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    combined[
        "_screening_bool"
    ] = bool_series(
        combined,
        "Screening_Eligible",
        default=False,
    )

    combined[
        "_candidate_feasible_bool"
    ] = bool_series(
        combined,
        "candidate_feasible",
        default=False,
    )

    source_rows = []

    for source in sorted(
        source_norm.unique()
    ):
        mask = source_norm.eq(source)

        source_rows.append({
            "Source_Type": source,
            "N_rows": int(mask.sum()),
            "N_screening_eligible": int(
                combined.loc[
                    mask,
                    "_screening_bool"
                ].sum()
            ),
            "N_candidate_feasible": int(
                combined.loc[
                    mask,
                    "_candidate_feasible_bool"
                ].sum()
            ),
        })

    source_summary = pd.DataFrame(
        source_rows
    )

    audit = pd.DataFrame([
        {
            "Metric":
                "base_master_rows",
            "Value":
                len(base),
        },
        {
            "Metric":
                "public_formal_rows",
            "Value":
                len(public),
        },
        {
            "Metric":
                "combined_rows",
            "Value":
                len(combined),
        },
        {
            "Metric":
                "combined_unique_candidate_ID",
            "Value":
                combined[
                    "Candidate_ID"
                ].nunique(),
        },
        {
            "Metric":
                "combined_screening_eligible",
            "Value":
                int(
                    combined[
                        "_screening_bool"
                    ].sum()
                ),
        },
        {
            "Metric":
                "public_in_domain",
            "Value":
                int(
                    (
                        source_norm.eq(
                            "pubchem"
                        )
                        & combined[
                            "Applicability_domain"
                        ].eq(
                            "In_domain"
                        )
                    ).sum()
                ),
        },
    ])

    # Drop temporary audit columns
    combined = combined.drop(
        columns=[
            "_screening_bool",
            "_candidate_feasible_bool",
        ],
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    combined.to_csv(
        OUT_MASTER,
        index=False,
        encoding="utf-8-sig",
    )

    audit.to_csv(
        OUT_AUDIT,
        index=False,
        encoding="utf-8-sig",
    )

    source_summary.to_csv(
        OUT_SOURCE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 78)
    print("COMBINED MASTER AUDIT")
    print("=" * 78)

    print(
        audit.to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print("SOURCE SUMMARY")
    print("=" * 78)

    print(
        source_summary.to_string(
            index=False
        )
    )

    print()
    print("[DONE]")
    print(
        "Combined master:",
        OUT_MASTER,
    )
    print(
        "Audit:",
        OUT_AUDIT,
    )
    print(
        "Source summary:",
        OUT_SOURCE,
    )


if __name__ == "__main__":
    main()
