from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


# ============================================================
# Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REVERSE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = REVERSE_DIR.parent

if str(REVERSE_DIR) not in sys.path:
    sys.path.insert(0, str(REVERSE_DIR))

from candidate_library import DOPO_CORE_SMARTS


RAW_DIR = SCRIPT_DIR / "data" / "01_pubchem_raw"
OUT_DIR = SCRIPT_DIR / "data" / "02_pubchem_standardized"

Q1_PATH = RAW_DIR / "pubchem_Q1_substructure_raw.csv"
Q3_PATH = RAW_DIR / "pubchem_Q3_DOPO_H_raw.csv"

OUT_ALL = OUT_DIR / "pubchem_standardized_all.csv"
OUT_ELIGIBLE = OUT_DIR / "pubchem_standardized_eligible.csv"
OUT_AUDIT = OUT_DIR / "pubchem_standardization_audit.csv"


# ============================================================
# Chemistry helpers
# ============================================================

DOPO_QUERY = Chem.MolFromSmarts(DOPO_CORE_SMARTS)

if DOPO_QUERY is None:
    raise RuntimeError(
        f"Invalid DOPO SMARTS imported from candidate_library.py: "
        f"{DOPO_CORE_SMARTS}"
    )


ELEMENTS = {
    "P": 15,
    "N": 7,
    "S": 16,
    "B": 5,
    "Si": 14,
}


def safe_number(value, default=None):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def atom_element_mass_fraction(mol, atomic_num: int) -> float:
    if mol is None:
        return float("nan")

    total_mw = Descriptors.MolWt(mol)

    if total_mw <= 0:
        return float("nan")

    mass = 0.0

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == atomic_num:
            mass += atom.GetMass()

    return 100.0 * mass / total_mw


def process_row(row: pd.Series) -> dict:
    raw_smiles = str(row.get("SMILES", "") or "").strip()

    result = {
        "SMILES_raw": raw_smiles,
        "Canonical_SMILES": "",
        "Computed_InChIKey": "",
        "Computed_Formula": "",
        "Computed_MW": float("nan"),
        "Murcko_Scaffold_SMILES": "",
        "Computed_DOPO_Count": 0,
        "Computed_Formal_Charge": float("nan"),
        "Computed_Fragment_Count": float("nan"),
        "Computed_Isotopic_Atom_Count": float("nan"),
        "smiles_valid": False,
        "contains_DOPO": False,
        "single_component": False,
        "neutral": False,
        "non_isotopic": False,
    }

    if not raw_smiles:
        return result

    try:
        mol = Chem.MolFromSmiles(raw_smiles)
    except Exception:
        mol = None

    if mol is None:
        return result

    result["smiles_valid"] = True

    # Canonical structure
    canonical = Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True,
    )

    result["Canonical_SMILES"] = canonical

    try:
        result["Computed_InChIKey"] = Chem.MolToInchiKey(mol)
    except Exception:
        result["Computed_InChIKey"] = ""

    # Formula / MW
    try:
        result["Computed_Formula"] = rdMolDescriptors.CalcMolFormula(mol)
    except Exception:
        result["Computed_Formula"] = ""

    result["Computed_MW"] = float(Descriptors.MolWt(mol))

    # Murcko scaffold
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)

        if scaffold is not None and scaffold.GetNumAtoms() > 0:
            result["Murcko_Scaffold_SMILES"] = Chem.MolToSmiles(
                scaffold,
                canonical=True,
            )
    except Exception:
        pass

    # DOPO check: always re-check with project rule
    matches = mol.GetSubstructMatches(DOPO_QUERY)

    result["Computed_DOPO_Count"] = len(matches)
    result["contains_DOPO"] = len(matches) > 0

    # Component check
    fragment_count = len(Chem.GetMolFrags(mol))
    result["Computed_Fragment_Count"] = fragment_count

    raw_covalent = safe_number(
        row.get("Covalent_Unit_Count"),
        default=None,
    )

    if raw_covalent is None:
        result["single_component"] = fragment_count == 1
    else:
        result["single_component"] = (
            fragment_count == 1
            and int(raw_covalent) == 1
        )

    # Charge check
    formal_charge = sum(
        atom.GetFormalCharge()
        for atom in mol.GetAtoms()
    )

    result["Computed_Formal_Charge"] = formal_charge

    raw_charge = safe_number(
        row.get("Charge"),
        default=None,
    )

    if raw_charge is None:
        result["neutral"] = formal_charge == 0
    else:
        result["neutral"] = (
            formal_charge == 0
            and int(raw_charge) == 0
        )

    # Isotope check
    isotope_count = sum(
        1
        for atom in mol.GetAtoms()
        if atom.GetIsotope() != 0
    )

    result["Computed_Isotopic_Atom_Count"] = isotope_count

    raw_isotope = safe_number(
        row.get("Isotopic_Atom_Count"),
        default=None,
    )

    if raw_isotope is None:
        result["non_isotopic"] = isotope_count == 0
    else:
        result["non_isotopic"] = (
            isotope_count == 0
            and int(raw_isotope) == 0
        )

    # Element wt%
    for symbol, atomic_num in ELEMENTS.items():
        result[f"{symbol}_molecular_wt_pct"] = (
            atom_element_mass_fraction(
                mol,
                atomic_num,
            )
        )

    return result


# ============================================================
# Main
# ============================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    missing = [
        str(p)
        for p in [Q1_PATH, Q3_PATH]
        if not p.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing PubChem raw files:\n"
            + "\n".join(missing)
        )

    frames = []

    for query_id, path in [
        ("Q1", Q1_PATH),
        ("Q3", Q3_PATH),
    ]:
        df = pd.read_csv(
            path,
            encoding="utf-8-sig",
        )

        df["Query_ID"] = query_id
        df["Source_Type"] = "pubchem"
        df["Source_Database"] = "PubChem"
        df["Source_File"] = path.name

        frames.append(df)

        print(
            f"[LOAD] {query_id}: "
            f"{path.name}, rows={len(df)}"
        )

    raw = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    print(f"[INFO] Combined raw rows: {len(raw)}")

    processed = []

    for i, row in raw.iterrows():
        result = process_row(row)
        processed.append(result)

        if (i + 1) % 500 == 0:
            print(
                f"[INFO] Processed "
                f"{i + 1}/{len(raw)}"
            )

    chem = pd.DataFrame(processed)

    out = pd.concat(
        [
            raw.reset_index(drop=True),
            chem.reset_index(drop=True),
        ],
        axis=1,
    )

    # PubChem candidate ID
    out["Candidate_ID"] = (
        "PUBCHEM_"
        + out["cid"]
        .astype("Int64")
        .astype(str)
    )

    # Prefer PubChem name
    if "Name" in out.columns:
        out["Candidate_Name"] = out["Name"]
    else:
        out["Candidate_Name"] = out["Candidate_ID"]

    out["Synthesis_Level"] = "PUBLIC_UNVERIFIED"
    out["synthesis_feasible"] = False

    # --------------------------------------------------------
    # Canonical duplicate handling
    # --------------------------------------------------------

    valid_canonical = (
        out["smiles_valid"]
        & out["Canonical_SMILES"].astype(str).ne("")
    )

    out["PubChem_Canonical_Duplicate"] = False

    out.loc[
        valid_canonical,
        "PubChem_Canonical_Duplicate"
    ] = out.loc[
        valid_canonical,
        "Canonical_SMILES"
    ].duplicated(keep="first")

    out["is_unique_pubchem"] = (
        valid_canonical
        & ~out["PubChem_Canonical_Duplicate"]
    )

    # --------------------------------------------------------
    # Standardization eligibility
    # --------------------------------------------------------

    out["Standardization_Eligible"] = (
        out["smiles_valid"]
        & out["contains_DOPO"]
        & out["single_component"]
        & out["neutral"]
        & out["non_isotopic"]
        & out["is_unique_pubchem"]
    )

    # --------------------------------------------------------
    # Exclusion reason
    # --------------------------------------------------------

    def exclusion_reason(row):
        reasons = []

        if not row["smiles_valid"]:
            reasons.append("invalid_smiles")

        if row["smiles_valid"] and not row["contains_DOPO"]:
            reasons.append("project_DOPO_SMARTS_not_matched")

        if row["smiles_valid"] and not row["single_component"]:
            reasons.append("multi_component")

        if row["smiles_valid"] and not row["neutral"]:
            reasons.append("charged")

        if row["smiles_valid"] and not row["non_isotopic"]:
            reasons.append("isotopic")

        if (
            row["smiles_valid"]
            and row["PubChem_Canonical_Duplicate"]
        ):
            reasons.append("duplicate_canonical")

        return ";".join(reasons)

    out["Exclusion_Reason"] = out.apply(
        exclusion_reason,
        axis=1,
    )

    # --------------------------------------------------------
    # Audit funnel: cumulative, not independent counts
    # --------------------------------------------------------

    raw_mask = pd.Series(
        True,
        index=out.index,
    )

    valid_mask = (
        raw_mask
        & out["smiles_valid"]
    )

    dopo_mask = (
        valid_mask
        & out["contains_DOPO"]
    )

    single_mask = (
        dopo_mask
        & out["single_component"]
    )

    neutral_mask = (
        single_mask
        & out["neutral"]
    )

    isotope_mask = (
        neutral_mask
        & out["non_isotopic"]
    )

    unique_mask = (
        isotope_mask
        & out["is_unique_pubchem"]
    )

    stages = [
        ("raw_records", raw_mask),
        ("valid_smiles", valid_mask),
        ("project_DOPO_match", dopo_mask),
        ("single_component", single_mask),
        ("neutral", neutral_mask),
        ("non_isotopic", isotope_mask),
        ("unique_canonical", unique_mask),
        (
            "standardization_eligible",
            out["Standardization_Eligible"],
        ),
    ]

    audit_rows = []

    previous_n = len(out)

    for stage_order, (stage, mask) in enumerate(
        stages,
        start=1,
    ):
        n = int(mask.sum())

        audit_rows.append({
            "Stage_Order": stage_order,
            "Stage": stage,
            "N": n,
            "Removed_From_Previous": (
                0
                if stage_order == 1
                else previous_n - n
            ),
            "Fraction_of_Raw": (
                n / len(out)
                if len(out)
                else float("nan")
            ),
        })

        previous_n = n

    audit = pd.DataFrame(audit_rows)

    # Add useful independent diagnostics
    diagnostics = pd.DataFrame([
        {
            "Metric": "Q1_raw_rows",
            "Value": int(
                (out["Query_ID"] == "Q1").sum()
            ),
        },
        {
            "Metric": "Q3_raw_rows",
            "Value": int(
                (out["Query_ID"] == "Q3").sum()
            ),
        },
        {
            "Metric": "unique_CID",
            "Value": int(
                out["cid"].nunique()
            ),
        },
        {
            "Metric": "canonical_duplicates",
            "Value": int(
                out[
                    "PubChem_Canonical_Duplicate"
                ].sum()
            ),
        },
        {
            "Metric": "DOPO_nonmatches",
            "Value": int(
                (
                    out["smiles_valid"]
                    & ~out["contains_DOPO"]
                ).sum()
            ),
        },
        {
            "Metric": "multi_component_records",
            "Value": int(
                (
                    out["smiles_valid"]
                    & ~out["single_component"]
                ).sum()
            ),
        },
        {
            "Metric": "charged_records",
            "Value": int(
                (
                    out["smiles_valid"]
                    & ~out["neutral"]
                ).sum()
            ),
        },
        {
            "Metric": "isotopic_records",
            "Value": int(
                (
                    out["smiles_valid"]
                    & ~out["non_isotopic"]
                ).sum()
            ),
        },
    ])

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    out.to_csv(
        OUT_ALL,
        index=False,
        encoding="utf-8-sig",
    )

    eligible = out[
        out["Standardization_Eligible"]
    ].copy()

    eligible.to_csv(
        OUT_ELIGIBLE,
        index=False,
        encoding="utf-8-sig",
    )

    audit.to_csv(
        OUT_AUDIT,
        index=False,
        encoding="utf-8-sig",
    )

    diagnostics.to_csv(
        OUT_DIR / "pubchem_standardization_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 78)
    print("PUBCHEM STANDARDIZATION AUDIT")
    print("=" * 78)
    print(audit.to_string(index=False))

    print()
    print("=" * 78)
    print("DIAGNOSTICS")
    print("=" * 78)
    print(diagnostics.to_string(index=False))

    print()
    print("[DONE]")
    print(f"All rows:      {OUT_ALL}")
    print(f"Eligible rows: {OUT_ELIGIBLE}")
    print(f"Audit:         {OUT_AUDIT}")


if __name__ == "__main__":
    main()
