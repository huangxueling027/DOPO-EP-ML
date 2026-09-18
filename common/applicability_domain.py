# -*- coding: utf-8 -*-
"""Molecular similarity and applicability-domain utilities for virtual screening."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from common.chem_standardization import standardize_smiles


@dataclass
class MoleculeReference:
    name: str
    original_smiles: str
    canonical_smiles: str
    scaffold: str
    fingerprint: object


def _molecule_record(name: object, smiles: object, generator) -> MoleculeReference | None:
    text = "" if pd.isna(smiles) else str(smiles).strip()
    if not text:
        return None
    standardized = standardize_smiles(text)
    if not standardized.is_valid or not standardized.canonical_smiles:
        return None
    canonical = standardized.canonical_smiles
    mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        return None
    scaffold = standardized.murcko_scaffold
    fingerprint = generator.GetFingerprint(mol)
    return MoleculeReference(
        name="" if pd.isna(name) else str(name),
        original_smiles=text,
        canonical_smiles=canonical,
        scaffold=scaffold,
        fingerprint=fingerprint,
    )


def build_training_reference(
    training_df: pd.DataFrame,
    *,
    smiles_col: str,
    name_col: str | None = None,
    radius: int = 2,
    n_bits: int = 2048,
) -> list[MoleculeReference]:
    """Build a deduplicated training-molecule reference library."""
    if smiles_col not in training_df.columns:
        raise KeyError(f"Training SMILES column not found: {smiles_col}")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    names = training_df[name_col] if name_col and name_col in training_df.columns else pd.Series("", index=training_df.index)

    records: list[MoleculeReference] = []
    seen: set[str] = set()
    for name, smiles in zip(names, training_df[smiles_col]):
        record = _molecule_record(name, smiles, generator)
        if record is None or record.canonical_smiles in seen:
            continue
        records.append(record)
        seen.add(record.canonical_smiles)

    if not records:
        raise ValueError("No valid training molecule could be built from the supplied SMILES column.")
    return records


def score_candidates(
    candidates_df: pd.DataFrame,
    references: Iterable[MoleculeReference],
    *,
    candidate_smiles_col: str,
    candidate_name_col: str | None = None,
    radius: int = 2,
    n_bits: int = 2048,
    reliable_threshold: float = 0.70,
    caution_threshold: float = 0.50,
    top_n: int = 3,
) -> pd.DataFrame:
    """Append nearest-neighbour similarity, scaffold coverage and AD labels."""
    if candidate_smiles_col not in candidates_df.columns:
        raise KeyError(f"Candidate SMILES column not found: {candidate_smiles_col}")
    if not 0 <= caution_threshold <= reliable_threshold <= 1:
        raise ValueError("Thresholds must satisfy 0 <= caution <= reliable <= 1.")

    refs = list(references)
    ref_fingerprints = [item.fingerprint for item in refs]
    seen_scaffolds = {item.scaffold for item in refs if item.scaffold}
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    output_rows: list[dict[str, object]] = []
    names = (
        candidates_df[candidate_name_col]
        if candidate_name_col and candidate_name_col in candidates_df.columns
        else pd.Series("", index=candidates_df.index)
    )

    for position, (index, row) in enumerate(candidates_df.iterrows()):
        name = names.loc[index]
        smiles = row[candidate_smiles_col]
        record = _molecule_record(name, smiles, generator)
        base = row.to_dict()
        base["candidate_original_index"] = index

        if record is None:
            base.update({
                "Candidate_SMILES_valid": False,
                "Candidate_Canonical_SMILES": "",
                "Candidate_Murcko_scaffold": "",
                "Max_Tanimoto_to_training": np.nan,
                "Scaffold_seen_in_training": False,
                "Applicability_domain": "Invalid_SMILES",
                "AD_reason": "Candidate SMILES could not be parsed by RDKit.",
            })
            for rank in range(1, top_n + 1):
                base[f"Nearest_{rank}_name"] = ""
                base[f"Nearest_{rank}_SMILES"] = ""
                base[f"Nearest_{rank}_Tanimoto"] = np.nan
            output_rows.append(base)
            continue

        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(record.fingerprint, ref_fingerprints),
            dtype=float,
        )
        order = np.argsort(-similarities)
        max_similarity = float(similarities[order[0]])
        scaffold_seen = bool(record.scaffold and record.scaffold in seen_scaffolds)

        if max_similarity >= reliable_threshold:
            ad_label = "In_domain"
            reason = f"Maximum Tanimoto >= {reliable_threshold:.2f}."
        elif max_similarity >= caution_threshold:
            ad_label = "Caution"
            reason = f"Maximum Tanimoto is between {caution_threshold:.2f} and {reliable_threshold:.2f}."
        else:
            ad_label = "Extrapolation"
            reason = f"Maximum Tanimoto < {caution_threshold:.2f}."

        if not scaffold_seen and ad_label == "In_domain":
            ad_label = "Caution_new_scaffold"
            reason += " Murcko scaffold is absent from the training set."
        elif not scaffold_seen:
            reason += " Murcko scaffold is absent from the training set."

        base.update({
            "Candidate_SMILES_valid": True,
            "Candidate_Canonical_SMILES": record.canonical_smiles,
            "Candidate_Murcko_scaffold": record.scaffold,
            "Max_Tanimoto_to_training": max_similarity,
            "Scaffold_seen_in_training": scaffold_seen,
            "Applicability_domain": ad_label,
            "AD_reason": reason,
        })

        for rank in range(1, top_n + 1):
            if rank <= len(order):
                ref = refs[int(order[rank - 1])]
                base[f"Nearest_{rank}_name"] = ref.name
                base[f"Nearest_{rank}_SMILES"] = ref.canonical_smiles
                base[f"Nearest_{rank}_Tanimoto"] = float(similarities[order[rank - 1]])
            else:
                base[f"Nearest_{rank}_name"] = ""
                base[f"Nearest_{rank}_SMILES"] = ""
                base[f"Nearest_{rank}_Tanimoto"] = np.nan

        output_rows.append(base)

    return pd.DataFrame(output_rows)
