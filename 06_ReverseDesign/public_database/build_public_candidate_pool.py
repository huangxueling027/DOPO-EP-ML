from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem


# ============================================================
# Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REVERSE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = REVERSE_DIR.parent

STANDARDIZED_PATH = (
    SCRIPT_DIR
    / "data"
    / "02_pubchem_standardized"
    / "pubchem_standardized_eligible.csv"
)

CANDIDATE_MASTER_PATH = (
    PROJECT_ROOT
    / "results"
    / "06_ReverseDesign"
    / "candidate_molecule_master.csv"
)

OUT_DIR = (
    SCRIPT_DIR
    / "data"
    / "03_public_candidates"
)

OUT_ALL = (
    OUT_DIR
    / "pubchem_public_candidates_all.csv"
)

OUT_UNIQUE = (
    OUT_DIR
    / "pubchem_public_candidates_unique.csv"
)

OUT_EXCLUDED = (
    OUT_DIR
    / "pubchem_exclusion_report.csv"
)

OUT_AUDIT = (
    OUT_DIR
    / "pubchem_public_pool_audit.csv"
)

OUT_OVERLAP = (
    OUT_DIR
    / "pubchem_overlap_with_existing.csv"
)


# ============================================================
# Helpers
# ============================================================

def canonicalize(smiles) -> str:
    if pd.isna(smiles):
        return ""

    text = str(smiles).strip()

    if not text:
        return ""

    try:
        mol = Chem.MolFromSmiles(text)
    except Exception:
        mol = None

    if mol is None:
        return ""

    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True,
    )
def model_identity_smiles(smiles) -> str:
    """
    Non-isomeric canonical identity used for ML candidate deduplication.

    The current molecular fingerprint/AD space does not distinguish some
    stereoisomers; therefore stereochemical variants should not occupy
    separate virtual-screening slots.
    """
    if pd.isna(smiles):
        return ""

    text = str(smiles).strip()

    if not text:
        return ""

    try:
        mol = Chem.MolFromSmiles(text)
    except Exception:
        mol = None

    if mol is None:
        return ""

    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=False,
    )

def first_existing_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col

    return None


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )


# ============================================================
# Main
# ============================================================

def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not STANDARDIZED_PATH.exists():
        raise FileNotFoundError(
            f"Missing standardized PubChem file:\n"
            f"{STANDARDIZED_PATH}"
        )

    if not CANDIDATE_MASTER_PATH.exists():
        raise FileNotFoundError(
            f"Missing candidate master:\n"
            f"{CANDIDATE_MASTER_PATH}"
        )

    pub = pd.read_csv(
        STANDARDIZED_PATH,
        encoding="utf-8-sig",
    )

    master = pd.read_csv(
        CANDIDATE_MASTER_PATH,
        encoding="utf-8-sig",
    )

    print("=" * 78)
    print("PUBLIC CANDIDATE POOL")
    print("=" * 78)

    print(
        f"[LOAD] PubChem standardized eligible: "
        f"{len(pub)}"
    )

    print(
        f"[LOAD] Existing candidate master: "
        f"{len(master)}"
    )

    # --------------------------------------------------------
    # PubChem canonical
    # --------------------------------------------------------

    if "Canonical_SMILES" not in pub.columns:
        raise KeyError(
            "PubChem standardized file lacks "
            "Canonical_SMILES."
        )

    pub["Canonical_SMILES"] = (
        pub["Canonical_SMILES"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    pub["Model_Identity_SMILES"] = (
        pub["Canonical_SMILES"]
        .apply(model_identity_smiles)
    )

    # --------------------------------------------------------
    # Existing master canonical structure
    # --------------------------------------------------------

    master_smiles_col = first_existing_column(
        master,
        [
            "Canonical_SMILES",
            "SMILES",
            "SMILES_main",
        ],
    )

    if master_smiles_col is None:
        raise KeyError(
            "Could not find a SMILES column in "
            "candidate_molecule_master.csv"
        )

    if master_smiles_col == "Canonical_SMILES":
        master["_canonical_compare"] = (
            master[master_smiles_col]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        master["_model_identity_compare"] = (
            master["_canonical_compare"]
            .apply(model_identity_smiles)
        )
    else:
        master["_canonical_compare"] = (
            master[master_smiles_col]
            .apply(canonicalize)
        )

    if "Source_Type" not in master.columns:
        raise KeyError(
            "candidate_molecule_master.csv lacks "
            "Source_Type."
        )

    source_norm = (
        master["Source_Type"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    training_mask = source_norm.eq(
        "training_seed"
    )

    designed_mask = source_norm.eq(
        "designed"
    )

    training = master[
        training_mask
    ].copy()

    designed = master[
        designed_mask
    ].copy()

    print(
        "[INFO] training_seed rows:",
        len(training),
    )

    print(
        "[INFO] designed rows:",
        len(designed),
    )

    # --------------------------------------------------------
    # Canonical lookup dictionaries
    # --------------------------------------------------------
    training_model_lookup = {}

    for _, row in training.iterrows():
        smi = row["_model_identity_compare"]

        if not smi:
            continue

        training_model_lookup.setdefault(
            smi,
            [],
        ).append(
            str(row.get("Candidate_ID", ""))
        )


    designed_model_lookup = {}

    for _, row in designed.iterrows():
        smi = row["_model_identity_compare"]

        if not smi:
            continue

        designed_model_lookup.setdefault(
            smi,
            [],
        ).append(
            str(row.get("Candidate_ID", ""))
        )

    training_lookup = {}

    for _, row in training.iterrows():
        smi = row["_canonical_compare"]

        if not smi:
            continue

        training_lookup.setdefault(
            smi,
            [],
        ).append(
            str(
                row.get(
                    "Candidate_ID",
                    "",
                )
            )
        )

    designed_lookup = {}

    for _, row in designed.iterrows():
        smi = row["_canonical_compare"]

        if not smi:
            continue

        designed_lookup.setdefault(
            smi,
            [],
        ).append(
            str(
                row.get(
                    "Candidate_ID",
                    "",
                )
            )
        )

    # --------------------------------------------------------
    # Match PubChem against existing chemical space
    # --------------------------------------------------------

    pub["Training_Duplicate"] = (
        pub["Canonical_SMILES"]
        .isin(training_lookup)
    )

    pub["Designed_Duplicate"] = (
        pub["Canonical_SMILES"]
        .isin(designed_lookup)
    )
    pub["Training_ModelIdentity_Duplicate"] = (
        pub["Model_Identity_SMILES"]
        .isin(training_model_lookup)
    )

    pub["Designed_ModelIdentity_Duplicate"] = (
        pub["Model_Identity_SMILES"]
        .isin(designed_model_lookup)
    )

    pub["PubChem_ModelIdentity_Duplicate"] = (
        pub["Model_Identity_SMILES"]
        .duplicated(keep="first")
    )
    
    pub["Matched_Training_Candidate_ID"] = (
        pub["Canonical_SMILES"]
        .map(
            lambda x:
                ";".join(
                    training_lookup.get(
                        x,
                        [],
                    )
                )
        )
    )

    pub["Matched_Designed_Candidate_ID"] = (
        pub["Canonical_SMILES"]
        .map(
            lambda x:
                ";".join(
                    designed_lookup.get(
                        x,
                        [],
                    )
                )
        )
    )

    pub["Existing_Structure_Duplicate"] = (
        pub["Training_Duplicate"]
        | pub["Designed_Duplicate"]
        | pub["Training_ModelIdentity_Duplicate"]
        | pub["Designed_ModelIdentity_Duplicate"]
        | pub["PubChem_ModelIdentity_Duplicate"]
    )

    # --------------------------------------------------------
    # Public-source status
    # --------------------------------------------------------

    pub["Source_Type"] = "pubchem"

    pub["Public_Record"] = True

    pub["Public_Source_Eligible"] = (
        ~pub["Existing_Structure_Duplicate"]
    )

    # Very important:
    # PubChem existence is NOT experimental synthesis proof.
    pub["Synthesis_Level"] = (
        "PUBLIC_UNVERIFIED"
    )

    pub["synthesis_feasible"] = False

    # Unified downstream screening gate.
    # Means "allowed to enter virtual screening",
    # not "experimentally proven synthesis-feasible".
    pub["candidate_feasible"] = (
        pub["Public_Source_Eligible"]
    )

    # --------------------------------------------------------
    # Exclusion reason
    # --------------------------------------------------------

    def public_exclusion_reason(row):
        reasons = []

        if bool(row["Training_Duplicate"]):
            reasons.append(
                "duplicate_training_structure"
            )

        if bool(row["Designed_Duplicate"]):
            reasons.append(
                "duplicate_designed_structure"
            )
        if bool(row["Training_ModelIdentity_Duplicate"]):
            reasons.append(
                "duplicate_training_model_identity"
            )

        if bool(row["Designed_ModelIdentity_Duplicate"]):
            reasons.append(
                "duplicate_designed_model_identity"
            )

        if bool(row["PubChem_ModelIdentity_Duplicate"]):
            reasons.append(
                "duplicate_pubchem_model_identity"
            )
        return ";".join(reasons)

    pub["Public_Exclusion_Reason"] = (
        pub.apply(
            public_exclusion_reason,
            axis=1,
        )
    )

    # --------------------------------------------------------
    # Save full annotated public pool
    # --------------------------------------------------------

    pub.to_csv(
        OUT_ALL,
        index=False,
        encoding="utf-8-sig",
    )

    # Only structurally new PubChem molecules
    unique = pub[
        pub["Public_Source_Eligible"]
    ].copy()

    unique.to_csv(
        OUT_UNIQUE,
        index=False,
        encoding="utf-8-sig",
    )

    excluded = pub[
        ~pub["Public_Source_Eligible"]
    ].copy()

    excluded.to_csv(
        OUT_EXCLUDED,
        index=False,
        encoding="utf-8-sig",
    )

    overlap = pub[
        pub["Existing_Structure_Duplicate"]
    ].copy()

    overlap.to_csv(
        OUT_OVERLAP,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Audit
    # --------------------------------------------------------

    n_raw = len(pub)

    n_training_dup = int(
        pub["Training_Duplicate"].sum()
    )

    n_designed_dup = int(
        pub["Designed_Duplicate"].sum()
    )

    n_both_dup = int(
        (
            pub["Training_Duplicate"]
            & pub["Designed_Duplicate"]
        ).sum()
    )

    n_existing_dup = int(
        pub[
            "Existing_Structure_Duplicate"
        ].sum()
    )

    n_unique = len(unique)

    audit = pd.DataFrame([
        {
            "Stage_Order": 1,
            "Stage":
                "standardization_eligible",
            "N": n_raw,
            "Removed_From_Previous": 0,
        },
        {
            "Stage_Order": 2,
            "Stage":
                "not_training_duplicate",
            "N": n_raw - n_training_dup,
            "Removed_From_Previous":
                n_training_dup,
        },
        {
            "Stage_Order": 3,
            "Stage":
                "not_existing_duplicate",
            "N": n_unique,
            "Removed_From_Previous":
                n_existing_dup,
        },
        {
            "Stage_Order": 4,
            "Stage":
                "public_unique_candidates",
            "N": n_unique,
            "Removed_From_Previous": 0,
        },
    ])

    audit.to_csv(
        OUT_AUDIT,
        index=False,
        encoding="utf-8-sig",
    )

    diagnostics = {
        "standardized_eligible":
            n_raw,

        "training_duplicates":
            n_training_dup,

        "designed_duplicates":
            n_designed_dup,

        "duplicate_of_both":
            n_both_dup,

        "existing_structure_duplicates":
            n_existing_dup,

        "public_unique_candidates":
            n_unique,
    }

    print()
    print("=" * 78)
    print("PUBLIC POOL AUDIT")
    print("=" * 78)

    print(
        audit.to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print("OVERLAP DIAGNOSTICS")
    print("=" * 78)

    for key, value in diagnostics.items():
        print(
            f"{key}: {value}"
        )

    print()
    print("[DONE]")
    print(
        "All annotated:",
        OUT_ALL,
    )
    print(
        "Unique public:",
        OUT_UNIQUE,
    )
    print(
        "Excluded:",
        OUT_EXCLUDED,
    )
    print(
        "Overlap:",
        OUT_OVERLAP,
    )
    print(
        "Audit:",
        OUT_AUDIT,
    )


if __name__ == "__main__":
    main()
