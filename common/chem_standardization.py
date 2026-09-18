# -*- coding: utf-8 -*-
"""Chemical standardization and duplicate-audit utilities.

Identity handling is conservative and preserves every disconnected component.
This is essential for salts and multi-component flame retardants: counterions,
component multiplicity, and stoichiometry must not be discarded when forming
molecule groups or applicability-domain references.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

try:
    from rdkit.Chem.MolStandardize import rdMolStandardize
except Exception:  # pragma: no cover
    rdMolStandardize = None


@dataclass(frozen=True)
class StandardizedMolecule:
    original: str
    canonical_smiles: str
    structure_id: str
    inchikey: str
    murcko_scaffold: str
    is_valid: bool
    is_multicomponent: bool
    component_count: int
    standardization_note: str


def _standardize_fragment(fragment: Chem.Mol) -> Chem.Mol:
    mol = Chem.Mol(fragment)
    if rdMolStandardize is not None:
        try:
            mol = rdMolStandardize.Normalizer().normalize(mol)
            mol = rdMolStandardize.Reionizer().reionize(mol)
        except Exception:
            pass
    Chem.SanitizeMol(mol)
    return mol


def _full_component_canonical(mol: Chem.Mol) -> tuple[str, list[Chem.Mol]]:
    """Canonicalize all disconnected components and preserve multiplicity."""
    fragments = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True))
    if not fragments:
        fragments = [mol]
    standardized = [_standardize_fragment(fragment) for fragment in fragments]
    smiles = [Chem.MolToSmiles(item, canonical=True, isomericSmiles=True) for item in standardized]
    smiles.sort()
    return ".".join(smiles), standardized


def _full_scaffold(fragments: list[Chem.Mol]) -> str:
    scaffolds: list[str] = []
    for fragment in fragments:
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(fragment)
        if scaffold_mol.GetNumAtoms() > 0:
            scaffolds.append(Chem.MolToSmiles(scaffold_mol, canonical=True, isomericSmiles=True))
    scaffolds.sort()
    return ".".join(scaffolds)


def standardize_smiles(value: object) -> StandardizedMolecule:
    text = "" if pd.isna(value) else str(value).strip()
    if not text:
        return StandardizedMolecule("", "", "", "", "", False, False, 0, "empty")

    mol = Chem.MolFromSmiles(text)
    is_multicomponent = "." in text
    if mol is None:
        return StandardizedMolecule(text, "", "", "", "", False, is_multicomponent, 0, "invalid_smiles")

    try:
        canonical, fragments = _full_component_canonical(mol)
        full_mol = Chem.MolFromSmiles(canonical)
        inchikey = ""
        if full_mol is not None:
            try:
                inchikey = Chem.MolToInchiKey(full_mol)
            except Exception:
                inchikey = ""
        structure_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
        note = "full_component_canonicalized"
        if len(fragments) > 1:
            note += ";all_components_preserved"
        return StandardizedMolecule(
            text,
            canonical,
            structure_id,
            inchikey,
            _full_scaffold(fragments),
            True,
            len(fragments) > 1,
            len(fragments),
            note,
        )
    except Exception as exc:
        return StandardizedMolecule(
            text, "", "", "", "", False, is_multicomponent, 0,
            f"standardization_error:{type(exc).__name__}",
        )


def _series_or_blank(df: pd.DataFrame, column: str | None) -> pd.Series:
    if column and column in df.columns:
        return df[column]
    return pd.Series("", index=df.index, dtype=object)


def add_standardization_columns(df: pd.DataFrame, colmap: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    role_to_key = {
        "main": "SMILES_main",
        "co": "SMILES_co",
        "curing": "SMILES_Curing_Agent",
    }

    multicomponent_columns: list[str] = []
    for role, key in role_to_key.items():
        source_col = colmap.get(key)
        records = _series_or_blank(out, source_col).apply(standardize_smiles)
        out[f"Canonical_SMILES_{role}"] = records.apply(lambda x: x.canonical_smiles)
        out[f"Structure_ID_{role}"] = records.apply(lambda x: x.structure_id)
        out[f"InChIKey_{role}"] = records.apply(lambda x: x.inchikey)
        out[f"Murcko_scaffold_{role}"] = records.apply(lambda x: x.murcko_scaffold)
        out[f"SMILES_{role}_valid"] = records.apply(lambda x: x.is_valid)
        out[f"Multicomponent_{role}"] = records.apply(lambda x: x.is_multicomponent)
        out[f"Component_count_{role}"] = records.apply(lambda x: x.component_count)
        out[f"SMILES_{role}_standardization_note"] = records.apply(lambda x: x.standardization_note)
        multicomponent_columns.append(f"Multicomponent_{role}")

    out["SMILES_valid_flag"] = out["SMILES_main_valid"] & (
        _series_or_blank(out, colmap.get("SMILES_co")).astype(str).str.strip().eq("")
        | out["SMILES_co_valid"]
    )
    out["Multicomponent_flag"] = out[multicomponent_columns].any(axis=1)
    out["Canonical_main_co_pair"] = (
        "MAIN=" + out["Canonical_SMILES_main"].fillna("")
        + "|CO=" + out["Canonical_SMILES_co"].fillna("")
    )
    out["Canonical_main_co_curing"] = (
        out["Canonical_main_co_pair"]
        + "|CURING=" + out["Canonical_SMILES_curing"].fillna("")
    )
    return out


def _normalise_for_key(series: pd.Series, numeric: bool = False) -> pd.Series:
    if numeric:
        values = pd.to_numeric(series, errors="coerce")
        return values.round(6).astype("Float64").astype(str).replace("<NA>", "")
    return series.fillna("").astype(str).str.strip().str.lower()


def add_duplicate_audit_columns(df: pd.DataFrame, colmap: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    key_specs: list[tuple[str, bool]] = [
        ("Canonical_SMILES_main", False),
        ("Canonical_SMILES_co", False),
        ("Canonical_SMILES_curing", False),
    ]
    for std_key in [
        "Loading_total_FR wt%", "Main_FR_fraction", "Co_FR_fraction",
        "Cure_Temp_Max", "LOI_Thickness_mm", "UL94_Thickness_mm",
        "Cone_Thickness_mm", "Cone_flux_kW_m2",
    ]:
        actual = colmap.get(std_key)
        if actual and actual in out.columns:
            key_specs.append((actual, True))

    for std_key in ["Preparation_Method", "Reference"]:
        actual = colmap.get(std_key, std_key)
        if actual in out.columns:
            key_specs.append((actual, False))

    key_frame = pd.DataFrame(index=out.index)
    for column, numeric in key_specs:
        key_frame[column] = _normalise_for_key(out[column], numeric=numeric)

    key_text = key_frame.astype(str).agg("|".join, axis=1)
    out["Duplicate_group_id"] = key_text.apply(
        lambda text: hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    )
    group_size = out.groupby("Duplicate_group_id")["Duplicate_group_id"].transform("size")
    out["Duplicate_group_size"] = group_size.astype(int)
    out["Potential_parallel_or_duplicate"] = group_size.gt(1)
    return out


def standardize_dataset(df: pd.DataFrame, colmap: dict[str, str]) -> pd.DataFrame:
    return add_duplicate_audit_columns(add_standardization_columns(df, colmap), colmap)


def save_standardization_audit(df: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "data_standardized_with_audit.csv", index=False, encoding="utf-8-sig")
    df.loc[~df["SMILES_valid_flag"]].to_csv(
        output_dir / "invalid_smiles_rows.csv", index=False, encoding="utf-8-sig"
    )
    df.loc[df["Potential_parallel_or_duplicate"]].to_csv(
        output_dir / "potential_duplicate_or_parallel_rows.csv", index=False, encoding="utf-8-sig"
    )
    summary = pd.DataFrame([{
        "n_rows": len(df),
        "n_valid_main_smiles": int(df["SMILES_main_valid"].sum()),
        "n_invalid_main_smiles": int((~df["SMILES_main_valid"]).sum()),
        "n_unique_main_molecules": int(df.loc[df["SMILES_main_valid"], "Canonical_SMILES_main"].nunique()),
        "n_unique_main_scaffolds": int(df.loc[df["SMILES_main_valid"], "Murcko_scaffold_main"].replace("", np.nan).nunique()),
        "n_multicomponent_rows": int(df["Multicomponent_flag"].sum()),
        "n_potential_duplicate_rows": int(df["Potential_parallel_or_duplicate"].sum()),
        "n_potential_duplicate_groups": int(df.loc[df["Potential_parallel_or_duplicate"], "Duplicate_group_id"].nunique()),
    }])
    summary.to_csv(output_dir / "data_standardization_summary.csv", index=False, encoding="utf-8-sig")
