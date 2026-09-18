# -*- coding: utf-8 -*-
"""Shared utilities for the DOPO candidate molecule library.

The candidate library is intentionally separated into molecule identity,
formulation scenarios, and model predictions. One row in the molecule master
means one unique molecular structure; loading and test-condition scenarios are
created later in the formulation grid.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parents[1]

# The fused phosphaphenanthrene ring system. The P substituent is left open so
# P-H, P-C, P-N and related DOPO derivatives all match.
DOPO_CORE_SMARTS = "[P](=O)(*)1Oc2ccccc2-c2ccccc21"
DOPO_CORE = Chem.MolFromSmarts(DOPO_CORE_SMARTS)

MANUAL_COLUMNS = [
    "Candidate_ID", "Candidate_Name", "SMILES_raw", "Source_Type",
    "Reference_ID", "Design_Family", "DOPO_Count", "Linker_Type",
    "Reactive_Group", "Preparation_Type", "Synthesis_Level", "Notes",
]

ELEMENTS = ("P", "N", "S", "B", "Si")

PREPARATION_MAP = {
    "additive": "DOPO-based (additive)",
    "reactive": "DOPO-based (reactive)",
    "co-curing": "DOPO-based (Co-curing)",
    "co curing": "DOPO-based (Co-curing)",
    "cocuring": "DOPO-based (Co-curing)",
    "additive+secondary": "DOPO-based (Additive + Secondary Crosslinking)",
    "additive+secondary crosslinking": "DOPO-based (Additive + Secondary Crosslinking)",
    "additive + secondary crosslinking": "DOPO-based (Additive + Secondary Crosslinking)",
    "dopo-based (additive)": "DOPO-based (additive)",
    "dopo-based (reactive)": "DOPO-based (reactive)",
    "dopo-based (co-curing)": "DOPO-based (Co-curing)",
    "dopo-based (additive+ secondary crosslinking)": "DOPO-based (Additive + Secondary Crosslinking)",
    "dopo-based (additive + secondary crosslinking)": "DOPO-based (Additive + Secondary Crosslinking)",
}
PREPARATION_NUM = {
    "DOPO-based (additive)": 0,
    "DOPO-based (reactive)": 1,
    "DOPO-based (Co-curing)": 2,
    "DOPO-based (Additive + Secondary Crosslinking)": 3,
}



def read_csv_auto(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    last: Exception | None = None
    for encoding in ("utf-8-sig", "gb18030", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last = exc
    if last:
        raise last
    raise UnicodeError(f"Unable to read {path}")


def canonicalize(smiles: object) -> tuple[str, Chem.Mol | None]:
    text = "" if pd.isna(smiles) else str(smiles).strip()
    if not text:
        return "", None
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return "", None
    return Chem.MolToSmiles(mol, canonical=True), mol


def murcko(mol: Chem.Mol | None) -> str:
    if mol is None:
        return ""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, canonical=True) if scaffold.GetNumAtoms() else "ACYCLIC"
    except Exception:
        return ""


def dopo_count(mol: Chem.Mol | None) -> int:
    if mol is None or DOPO_CORE is None:
        return 0
    return len(mol.GetSubstructMatches(DOPO_CORE, uniquify=True))


def element_mass_percent(mol: Chem.Mol | None, symbol: str) -> float:
    if mol is None:
        return np.nan
    total = float(Descriptors.MolWt(mol))
    if total <= 0:
        return np.nan
    table = Chem.GetPeriodicTable()
    mass = sum(float(table.GetAtomicWeight(atom.GetAtomicNum())) for atom in mol.GetAtoms() if atom.GetSymbol() == symbol)
    return 100.0 * mass / total


def infer_family(mol: Chem.Mol | None) -> str:
    if mol is None:
        return "Unknown"
    present = {atom.GetSymbol() for atom in mol.GetAtoms()}
    extras = [x for x in ("N", "Si", "B", "S") if x in present]
    return "P-only" if not extras else "P-" + "-".join(extras)


def infer_reactive_groups(mol: Chem.Mol | None) -> str:
    if mol is None:
        return "unknown"
    patterns = {
        "NH2/NH": "[N;H1,H2;!$(N-[C,S,P]=[O,S,N])]",
        "OH": "[O;H1]",
        "epoxy": "[O;r3]1[C;r3][C;r3]1",
        "allyl/vinyl": "[C,c]=[C,c]",
        "carboxyl": "C(=O)[O;H1,-1]",
    }
    found = [name for name, smarts in patterns.items() if (q := Chem.MolFromSmarts(smarts)) is not None and mol.HasSubstructMatch(q)]
    return ";".join(found) if found else "none"


def normalize_preparation(value: object, reactive_group: str = "none") -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if text in PREPARATION_NUM:
        return text
    key = re.sub(r"\s+", " ", text.lower())
    if key in PREPARATION_MAP:
        return PREPARATION_MAP[key]
    # Conservative inference only when the user did not specify a method.
    if reactive_group not in {"", "none", "unknown"}:
        return "DOPO-based (reactive)"
    return "DOPO-based (additive)"


def synthesis_feasible(level: object) -> bool:
    text = "" if pd.isna(level) else str(level).strip().upper()
    return text in {"A", "B"}


def empty_manual_template() -> pd.DataFrame:
    return pd.DataFrame(columns=MANUAL_COLUMNS)


def enrich_candidate_rows(frame: pd.DataFrame, training_canonical: Iterable[str]) -> pd.DataFrame:
    """Validate and enrich manually entered or literature candidate rows."""
    out = frame.copy()
    for column in MANUAL_COLUMNS:
        if column not in out.columns:
            out[column] = ""
    out = out[MANUAL_COLUMNS + [c for c in out.columns if c not in MANUAL_COLUMNS]].copy()

    canonical_values: list[str] = []
    mols: list[Chem.Mol | None] = []
    for smiles in out["SMILES_raw"]:
        canonical, mol = canonicalize(smiles)
        canonical_values.append(canonical)
        mols.append(mol)
    out["Canonical_SMILES"] = canonical_values
    out["smiles_valid"] = [mol is not None for mol in mols]
    out["contains_DOPO"] = [dopo_count(mol) > 0 for mol in mols]
    out["Computed_DOPO_Count"] = [dopo_count(mol) for mol in mols]
    out["Molecular_Weight"] = [float(Descriptors.MolWt(mol)) if mol is not None else np.nan for mol in mols]
    out["Murcko_Scaffold"] = [murcko(mol) for mol in mols]
    for element in ELEMENTS:
        out[f"{element}_molecular_wt_pct"] = [element_mass_percent(mol, element) for mol in mols]
        out[f"Has_{element}"] = [int(mol is not None and any(a.GetSymbol() == element for a in mol.GetAtoms())) for mol in mols]

    inferred_family = [infer_family(mol) for mol in mols]
    out["Design_Family"] = out["Design_Family"].replace("", np.nan).fillna(pd.Series(inferred_family, index=out.index))
    inferred_reactive = [infer_reactive_groups(mol) for mol in mols]
    out["Reactive_Group"] = out["Reactive_Group"].replace("", np.nan).fillna(pd.Series(inferred_reactive, index=out.index))
    out["Preparation_Method"] = [normalize_preparation(v, r) for v, r in zip(out["Preparation_Type"], out["Reactive_Group"])]
    out["Preparation_Method_num"] = out["Preparation_Method"].map(PREPARATION_NUM)
    out["Synthesis_Level"] = out["Synthesis_Level"].replace("", "C").astype(str).str.upper()
    out["synthesis_feasible"] = out["Synthesis_Level"].map(synthesis_feasible)

    training_set = {str(x) for x in training_canonical if str(x)}
    out["Training_Duplicate"] = out["Canonical_SMILES"].isin(training_set)
    valid_canonical = out["Canonical_SMILES"].where(out["Canonical_SMILES"].ne(""), np.nan)
    out["is_unique"] = ~valid_canonical.duplicated(keep="first")
    out.loc[valid_canonical.isna(), "is_unique"] = False
    out["Screening_Eligible"] = (
        out["smiles_valid"]
        & out["contains_DOPO"]
        & out["is_unique"]
        & ~out["Training_Duplicate"]
        & out["synthesis_feasible"]
    )

    reasons = []
    for row in out.itertuples(index=False):
        r = []
        if not bool(row.smiles_valid): r.append("invalid_smiles")
        if bool(row.smiles_valid) and not bool(row.contains_DOPO): r.append("no_DOPO_core")
        if not bool(row.is_unique): r.append("duplicate_candidate")
        if bool(row.Training_Duplicate): r.append("training_duplicate")
        if not bool(row.synthesis_feasible): r.append("synthesis_level_C_or_missing")
        reasons.append(";".join(r) if r else "eligible")
    out["Exclusion_Reason"] = reasons
    return out
