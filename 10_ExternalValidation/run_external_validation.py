# -*- coding: utf-8 -*-
"""
Independent external validation for corrected ML_DOPO_Project_V5

Design principles
-----------------
1. The frozen development database remains unchanged.
2. External samples are NEVER used for model/view/K/threshold selection.
3. Load only the frozen FINAL deployable bundle defined by the project manifest.
4. If a FINAL bundle is missing or mismatched, stop; never reconstruct, retune,
   or fall back to curated/V4/smoke results.
5. External metrics use positive-loading flame-retarded formulations only.
6. Neat EP rows are retained for feature/baseline tracing but are not counted
   as external "new-molecule" validation samples.
7. Cone PHRR/THR and MCC heat-release results are never mixed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import pipeline_core as core
from common.final_result_paths import load_final_bundle
from common import scientific_evaluation as se
from common.literature_feature_views import get_groups, get_target


DEFAULT_TRAIN = ROOT / "data" / "DOPO_EP_new_with_BDE.csv"
DEFAULT_EXTERNAL = (
    ROOT
    / "data"
    / "external_validation"
    / "DOPO_EP_External_Validation_Final_6_SMILES_Audited.xlsx"
)
DEFAULT_EXTERNAL_READY = (
    ROOT
    / "data"
    / "external_validation"
    / "External_Validation_V4_ready.csv"
)
LEGACY_EXTERNAL_READY = (
    ROOT
    / "results"
    / "10_ExternalValidation"
    / "External_Validation_V4_ready.csv"
)
DEFAULT_OUTPUT = ROOT / "results" / "10_ExternalValidation" / "FINAL_fixed"

DEFAULT_TASKS = [
    "LOI",
    "PHRR",
    "THR",
    "UL94_V0",
    "Tg",
    "TS_MPa",
]

CORE_EXTERNAL = {"MBFAP", "SPDO", "VH-DOPO"}

ELEMENTS = ["P", "N", "S", "B", "Si"]


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def read_csv_auto(path: Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError(f"Cannot read CSV: {path}")


def pick_column(df: pd.DataFrame, *names: str) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series(np.nan, index=df.index)


def numeric(series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def clean_text(series) -> pd.Series:
    return (
        pd.Series(series, index=series.index)
        .fillna("")
        .astype(str)
        .str.strip()
    )


def canonical_smiles(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True,
    )


def murcko(value) -> str:
    mol = Chem.MolFromSmiles(str(value)) if value else None
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return ""


def canonical_preparation(value: str) -> str:
    """Return the same canonical preparation labels used by the training data."""
    text = "" if value is None else str(value).strip()
    low = text.lower()

    if "secondary" in low:
        return "DOPO-based (Additive + Secondary Crosslinking)"
    if "co-cur" in low or "co curing" in low or "cocuring" in low:
        return "DOPO-based (Co-curing)"
    if "react" in low:
        return "DOPO-based (reactive)"
    return "DOPO-based (additive)"


def elemental_formulation_contents(
    smiles: str,
    total_loading_wt_pct: float,
) -> dict[str, float]:
    """
    Calculate formulation-level P/N/S/B/Si wt%:
        total FR wt% × elemental mass fraction in the FR molecule.
    """
    result = {e: 0.0 for e in ELEMENTS}

    if (
        smiles is None
        or pd.isna(total_loading_wt_pct)
        or float(total_loading_wt_pct) <= 0
    ):
        return result

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return result

    mw = Descriptors.MolWt(mol)
    if not np.isfinite(mw) or mw <= 0:
        return result

    pt = Chem.GetPeriodicTable()

    for element in ELEMENTS:
        element_mass = 0.0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == element:
                element_mass += float(pt.GetAtomicWeight(atom.GetAtomicNum()))

        molecular_fraction = element_mass / mw
        result[element] = (
            float(total_loading_wt_pct)
            * molecular_fraction
        )

    return result


# ---------------------------------------------------------------------
# Read external validation workbook
# ---------------------------------------------------------------------

def read_external_workbook(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"\nExternal workbook not found:\n{path}\n"
        )

    xl = pd.ExcelFile(path)

    preferred = [
        "最终外部验证表",
        "Sheet1",
    ]

    sheets = preferred + [
        x for x in xl.sheet_names if x not in preferred
    ]

    for sheet in sheets:
        if sheet not in xl.sheet_names:
            continue

        for header in [3, 0]:
            try:
                frame = pd.read_excel(
                    path,
                    sheet_name=sheet,
                    header=header,
                )
            except Exception:
                continue

            frame.columns = [
                str(c).strip() for c in frame.columns
            ]

            has_fr = "FR_main" in frame.columns
            has_smiles = any(
                c in frame.columns
                for c in ["SMILES_main", "Smile", "SMILES"]
            )
            has_target = any(
                c in frame.columns
                for c in [
                    "LOI",
                    "PHRR_kW_m2",
                    "THR_MJ_m2",
            "THR_MJ_㎡",
                    "UL94_rating",
                    "TS_MPa",
                ]
            )

            if has_fr and has_smiles and has_target:
                frame = frame.dropna(
                    how="all"
                ).reset_index(drop=True)

                print(
                    f"[DATA] external workbook sheet={sheet} "
                    f"header={header} rows={len(frame)}"
                )
                return frame

    raise RuntimeError(
        "\nThe supplied Excel does not contain the full external-validation "
        "table.\n"
        "Do NOT use the 9-column simplified workbook here.\n"
        "Use:\n"
        "DOPO_EP_External_Validation_Final_6_SMILES_Audited.xlsx\n"
    )


def _is_prepared_external_table(frame: pd.DataFrame, train: pd.DataFrame) -> bool:
    """Recognize the previously audited V4-schema external input table.

    The name ``V4_ready`` refers to column compatibility only. FINAL external
    validation still uses only the frozen FINAL V5 model bundles.
    """
    required = {"FR_main", "SMILES_main", "Loading_total_FR wt%"}
    if not required.issubset(frame.columns):
        return False

    target_cols = {
        "LOI", "UL94", "PHRR_kw_㎡", "PHRR_kW_m2",
        "THR_MJ_㎡", "THR_MJ_m2", "TS_MPa",
    }
    if not any(c in frame.columns for c in target_cols):
        return False

    overlap = len(set(frame.columns) & set(train.columns))
    minimum_overlap = max(20, int(0.50 * min(len(frame.columns), len(train.columns))))
    return overlap >= minimum_overlap


def _normalise_prepared_external_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Add stable IDs and neutral aliases without changing experimental values."""
    out = frame.copy().reset_index(drop=True)
    n = len(out)

    if "Record_ID" not in out.columns:
        out["Record_ID"] = [f"EXT_{i+1:03d}" for i in range(n)]
    else:
        rid = out["Record_ID"].fillna("").astype(str).str.strip()
        for i in out.index[rid.eq("")]:
            rid.loc[i] = f"EXT_{i+1:03d}"
        out["Record_ID"] = rid

    if "Sample_ID" not in out.columns:
        out["Sample_ID"] = [f"EXT_SAMPLE_{i+1:03d}" for i in range(n)]

    if "Standardized_total_FR_wt_pct" not in out.columns:
        out["Standardized_total_FR_wt_pct"] = pd.to_numeric(
            out.get("Loading_total_FR wt%", np.nan), errors="coerce"
        )

    alias_sources = {
        "PHRR_kW_m2": ["PHRR_kw_㎡", "PHRR_kw_m2"],
        "THR_MJ_m2": ["THR_MJ_㎡"],
        "UL94_rating": ["UL94"],
        "Tg_DMA_C": ["Tg_℃"],
        "TGA_char_yield_pct": ["Char_yield_％_700C"],
    }
    for target, sources in alias_sources.items():
        if target in out.columns and pd.Series(out[target]).notna().any():
            continue
        for source in sources:
            if source in out.columns:
                out[target] = out[source]
                break

    if "Tg_℃" in out.columns and "Tg_method" not in out.columns:
        tg = pd.to_numeric(out["Tg_℃"], errors="coerce")
        out["Tg_method"] = np.where(tg.notna(), "DMA", "")

    if "Reference" in out.columns and "Source_URL" not in out.columns:
        out["Source_URL"] = out["Reference"]

    if "Row_role" not in out.columns:
        loading = pd.to_numeric(out["Standardized_total_FR_wt_pct"], errors="coerce")
        out["Row_role"] = np.where(
            loading.fillna(0).eq(0), "neat_EP_baseline", "modified"
        )
    if "Validation_set" not in out.columns:
        out["Validation_set"] = "external_literature"

    return out


def _coerce_prepared_external_to_train_schema(
    train: pd.DataFrame,
    ready: pd.DataFrame,
) -> pd.DataFrame:
    """Copy a reviewed ready table into the exact frozen training schema."""
    meta = _normalise_prepared_external_metadata(ready)
    prepared = pd.DataFrame(np.nan, index=range(len(meta)), columns=train.columns)
    for col in prepared.columns:
        if col in meta.columns:
            prepared[col] = meta[col].values
    if "Record_ID" in prepared.columns:
        prepared["Record_ID"] = meta["Record_ID"].values
    return prepared


def resolve_external_source(requested: Path, train: pd.DataFrame):
    """Use audited Excel when available; otherwise reuse the reviewed ready CSV."""
    candidates = [Path(requested)]
    for candidate in [DEFAULT_EXTERNAL_READY, LEGACY_EXTERNAL_READY]:
        if candidate not in candidates:
            candidates.append(candidate)

    for path in candidates:
        if not path.exists():
            continue
        suffix = path.suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            frame = read_external_workbook(path)
            return frame, "workbook", path
        if suffix == ".csv":
            frame = read_csv_auto(path)
            if not _is_prepared_external_table(frame, train):
                raise RuntimeError(
                    "External CSV exists but is not a recognized audited ready table: "
                    f"{path}"
                )
            frame = _normalise_prepared_external_metadata(frame)
            print(f"[DATA] using audited prepared external CSV: {path} rows={len(frame)}")
            return frame, "prepared_csv", path

    checked = "\n  - ".join(str(x) for x in candidates)
    raise FileNotFoundError(
        "External validation input was not found. Checked:\n  - "
        + checked
        + "\nProvide the audited Excel workbook or External_Validation_V4_ready.csv."
    )


# ---------------------------------------------------------------------
# Structural identity / AD audit
# ---------------------------------------------------------------------

def audit_external_molecules(
    train: pd.DataFrame,
    external: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:

    train_colmap = core.resolve_columns(train)
    smi_col = train_colmap["SMILES_main"]
    name_col = train_colmap.get("FR_main", "FR_main")

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=3,
        fpSize=2048,
    )

    training = {}

    for _, row in train.iterrows():
        raw = row.get(smi_col)
        can = canonical_smiles(raw)
        if not can:
            continue

        if can not in training:
            mol = Chem.MolFromSmiles(can)
            training[can] = {
                "mol": mol,
                "fp": generator.GetFingerprint(mol),
                "scaffold": murcko(can),
                "names": set(),
            }

        if name_col in row.index:
            name = str(row.get(name_col, "")).strip()
            if name:
                training[can]["names"].add(name)

    ext_smiles = pick_column(
        external,
        "SMILES_main",
        "Smile",
        "SMILES",
    )

    unique_ext = pd.DataFrame({
        "FR_main": clean_text(external["FR_main"]),
        "SMILES_main": clean_text(ext_smiles),
    }).drop_duplicates("FR_main")

    rows = []

    for _, record in unique_ext.iterrows():
        fr = record["FR_main"]
        raw = record["SMILES_main"]

        mol = Chem.MolFromSmiles(raw)

        if mol is None:
            rows.append({
                "FR_main": fr,
                "SMILES_valid": "FAIL",
            })
            continue

        can = Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=True,
        )

        fp = generator.GetFingerprint(mol)
        scaffold = murcko(can)

        exact = can in training
        scaffold_seen = any(
            scaffold == meta["scaffold"]
            for meta in training.values()
        )

        similarities = []

        for train_can, meta in training.items():
            sim = DataStructs.TanimotoSimilarity(
                fp,
                meta["fp"],
            )
            similarities.append(
                (sim, train_can, meta)
            )

        similarities.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        max_sim, nearest_can, nearest_meta = similarities[0]

        if max_sim >= 0.70:
            ad = "in-domain"
        elif max_sim >= 0.50:
            ad = "caution"
        else:
            ad = "extrapolation"

        if exact:
            level = "Level 1"
        elif scaffold_seen:
            level = "Level 2"
        else:
            level = "Level 3"

        if exact:
            status = "Seen molecule - EXCLUDE as unseen-molecule validation"
        elif ad == "extrapolation":
            status = "Unseen molecule - OOD/extrapolation challenge"
        elif ad == "caution":
            status = "Unseen molecule - PASS with AD caution"
        else:
            status = "Unseen molecule - PASS"

        rows.append({
            "FR_main": fr,
            "SMILES_valid": "PASS",
            "SMILES_main": raw,
            "Canonical_SMILES_main": can,
            "Molecular_formula": Chem.rdMolDescriptors.CalcMolFormula(mol),
            "Exact_MW": Chem.rdMolDescriptors.CalcExactMolWt(mol),
            "Murcko_scaffold": scaffold,
            "V4_exact_canonical_match": "YES" if exact else "NO",
            "V4_scaffold_seen": "YES" if scaffold_seen else "NO",
            "Max_Tanimoto_to_V4": max_sim,
            "Nearest_V4_FR": " | ".join(
                sorted(nearest_meta["names"])
            ),
            "Nearest_V4_Canonical_SMILES": nearest_can,
            "AD_category": ad,
            "External_identity_level": level,
            "External_validation_status": status,
        })

    audit = pd.DataFrame(rows)

    audit.to_csv(
        output_dir / "external_molecule_identity_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"[AUDIT] V4 canonical molecules={len(training)}"
    )

    for _, r in audit.iterrows():
        if r.get("SMILES_valid") != "PASS":
            print(
                f"[AUDIT] {r['FR_main']}: INVALID SMILES"
            )
        else:
            print(
                f"[AUDIT] {r['FR_main']}: "
                f"exact={r['V4_exact_canonical_match']} "
                f"scaffold={r['V4_scaffold_seen']} "
                f"Smax={r['Max_Tanimoto_to_V4']:.3f} "
                f"AD={r['AD_category']} "
                f"{r['External_identity_level']}"
            )

    return audit


# ---------------------------------------------------------------------
# Convert external literature table to V4 raw schema
# ---------------------------------------------------------------------

def prepare_external_as_v4_rows(
    train: pd.DataFrame,
    external: pd.DataFrame,
    audit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    colmap = core.resolve_columns(train)
    n = len(external)

    prepared = pd.DataFrame(
        np.nan,
        index=np.arange(n),
        columns=train.columns,
    )

    def set_standard(std_name, values):
        column = colmap.get(std_name)

        if column is None and std_name in prepared.columns:
            column = std_name

        if column is not None and column in prepared.columns:
            prepared[column] = list(values)

    def set_first_existing(names, values):
        for name in names:
            if name in prepared.columns:
                prepared[name] = list(values)
                return name
        return None

    # --------------------------------------------------
    # External metadata
    # --------------------------------------------------

    fr = clean_text(external["FR_main"])

    smiles = clean_text(
        pick_column(
            external,
            "SMILES_main",
            "Smile",
            "SMILES",
        )
    )

    sample_id = clean_text(
        pick_column(
            external,
            "Sample_ID",
            "Sample",
        )
    )

    if sample_id.eq("").all():
        sample_id = pd.Series(
            [f"EXT_SAMPLE_{i+1:03d}" for i in range(n)]
        )

    record_id = clean_text(
        pick_column(
            external,
            "Record_ID",
        )
    )

    if record_id.eq("").all():
        record_id = pd.Series(
            [f"EXT_{i+1:03d}" for i in range(n)]
        )

    reported_loading = numeric(
        pick_column(
            external,
            "Reported_loading_wt_pct",
            "Reported_loading_wt%",
        )
    )

    standardized_loading = numeric(
        pick_column(
            external,
            "Standardized_total_FR_wt_pct",
            "Loading_total_FR wt%",
        )
    )

    ep_mass = numeric(
        pick_column(
            external,
            "EP_mass_or_fraction",
        )
    )
    fr_mass = numeric(
        pick_column(
            external,
            "FR_mass_or_fraction",
        )
    )
    curing_mass = numeric(
        pick_column(
            external,
            "Curing_mass_or_fraction",
        )
    )

    calculated_loading = (
        fr_mass
        / (ep_mass + fr_mass + curing_mass)
        * 100.0
    )

    standardized_loading = standardized_loading.where(
        standardized_loading.notna(),
        calculated_loading,
    )

    standardized_loading = standardized_loading.where(
        standardized_loading.notna(),
        reported_loading,
    )

    external = external.copy()
    external["Record_ID"] = record_id
    external["Sample_ID"] = sample_id
    external["SMILES_main"] = smiles
    external["Standardized_total_FR_wt_pct"] = standardized_loading

    # --------------------------------------------------
    # Identity fields
    # --------------------------------------------------

    set_standard("FR_main", fr)
    set_standard("SMILES_main", smiles)
    set_standard("FR_co", [""] * n)
    set_standard("SMILES_co", [""] * n)

    if "Record_ID" in prepared.columns:
        prepared["Record_ID"] = record_id

    # --------------------------------------------------
    # Loading and formulation
    # --------------------------------------------------

    set_standard(
        "Loading_total_FR wt%",
        standardized_loading,
    )

    prep_raw = clean_text(
        pick_column(
            external,
            "Preparation_Method",
        )
    )

    prep_canonical = prep_raw.apply(
        canonical_preparation
    )

    set_standard(
        "Preparation_Method",
        prep_canonical,
    )

    set_standard(
        "FR_class",
        prep_canonical,
    )

    curing = clean_text(
        pick_column(
            external,
            "Curing_Agent",
        )
    )
    set_standard("Curing_Agent", curing)

    synergy_raw = clean_text(
        pick_column(
            external,
            "Synergy_type",
        )
    )

    synergy_for_model = synergy_raw.copy()
    synergy_for_model.loc[
        standardized_loading.fillna(0).eq(0)
    ] = "None"

    set_standard(
        "Synergy_type",
        synergy_for_model,
    )

    synergy_flag = (
        standardized_loading.fillna(0).gt(0)
        & synergy_raw.str.contains(
            "N|S|B|Si",
            regex=True,
            case=False,
        )
    ).astype(int)

    set_standard(
        "Synergy_flag",
        synergy_flag,
    )

    set_standard(
        "Main_FR_fraction",
        standardized_loading.fillna(0).gt(0).astype(float),
    )
    set_standard(
        "Co_FR_fraction",
        [0.0] * n,
    )

    # --------------------------------------------------
    # Preparation method numeric code from V4 database
    # --------------------------------------------------

    pm_col = colmap.get("Preparation_Method")
    pm_num_col = colmap.get("Preparation_Method_num")

    if (
        pm_col in train.columns
        and pm_num_col in train.columns
    ):
        temp = pd.DataFrame({
            "method": train[pm_col]
            .fillna("")
            .astype(str)
            .apply(canonical_preparation),
            "num": pd.to_numeric(
                train[pm_num_col],
                errors="coerce",
            ),
        }).dropna()

        mapping = {}

        for key, sub in temp.groupby("method"):
            if len(sub):
                mapping[key] = float(
                    sub["num"].mode().iloc[0]
                )

        prep_num = prep_canonical.map(mapping)
        prepared[pm_num_col] = prep_num

    # --------------------------------------------------
    # Curing-agent SMILES and element flags
    # --------------------------------------------------

    curing_smiles_known = {
        "DDM": "Nc1ccc(Cc2ccc(N)cc2)cc1",
        "MPD": "Nc1cccc(N)c1",
    }

    curing_smiles = []

    train_curing_col = colmap.get("Curing_Agent")
    train_curing_smi_col = colmap.get(
        "SMILES_Curing_Agent"
    )

    for value in curing:
        smi = None

        if (
            train_curing_col in train.columns
            and train_curing_smi_col in train.columns
        ):
            mask = (
                train[train_curing_col]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.upper()
                .eq(str(value).upper())
            )

            candidates = (
                train.loc[
                    mask,
                    train_curing_smi_col,
                ]
                .dropna()
                .astype(str)
            )

            candidates = candidates[
                candidates.str.strip().ne("")
            ]

            if len(candidates):
                smi = candidates.mode().iloc[0]

        if not smi:
            smi = curing_smiles_known.get(
                str(value).upper(),
                "",
            )

        curing_smiles.append(smi)

    set_standard(
        "SMILES_Curing_Agent",
        curing_smiles,
    )

    curing_has_n = [
        1 if smi and Chem.MolFromSmiles(smi)
        and any(
            atom.GetSymbol() == "N"
            for atom in Chem.MolFromSmiles(smi).GetAtoms()
        )
        else 0
        for smi in curing_smiles
    ]

    set_standard(
        "CuringAgent_Has_N",
        curing_has_n,
    )

    for std in [
        "CuringAgent_Has_S",
        "CuringAgent_Has_P",
        "CuringAgent_Has_B",
        "CuringAgent_Has_F",
        "CuringAgent_Has_Cl",
    ]:
        set_standard(std, [0] * n)

    # --------------------------------------------------
    # Elemental formulation composition
    # --------------------------------------------------

    element_rows = []

    for smi, loading in zip(
        smiles,
        standardized_loading,
    ):
        element_rows.append(
            elemental_formulation_contents(
                smi,
                loading,
            )
        )

    for element in ELEMENTS:
        values = pd.Series([
            r[element] for r in element_rows
        ])

        if element == "P":
            set_standard(
                "P_content wt%",
                values,
            )
        else:
            set_standard(
                f"{element}_content wt%",
                values,
            )

        has_values = (
            values.fillna(0).gt(0)
        ).astype(int)

        set_standard(
            f"Has_{element}",
            has_values,
        )

    p_content = pd.Series([
        x["P"] for x in element_rows
    ])

    for element in ["N", "S", "B", "Si"]:
        e_content = pd.Series([
            x[element] for x in element_rows
        ])

        ratio = e_content / p_content.replace(
            0,
            np.nan,
        )

        set_standard(
            f"{element}/P ratio",
            ratio,
        )

    # --------------------------------------------------
    # Experimental/test-condition fields
    # --------------------------------------------------

    set_standard(
        "Cure_Temp_Max",
        numeric(
            pick_column(
                external,
                "Cure_Temp_Max_C",
                "Cure_Temp_Max",
            )
        ),
    )

    set_standard(
        "LOI_Thickness_mm",
        numeric(
            pick_column(
                external,
                "LOI_Thickness_mm",
            )
        ),
    )

    set_standard(
        "UL94_Thickness_mm",
        numeric(
            pick_column(
                external,
                "UL94_Thickness_mm",
            )
        ),
    )

    set_standard(
        "Cone_Thickness_mm",
        numeric(
            pick_column(
                external,
                "Cone_Thickness_mm",
            )
        ),
    )

    set_standard(
        "Cone_flux_kW_m2",
        numeric(
            pick_column(
                external,
                "Cone_flux_kW_m2",
            )
        ),
    )

    # --------------------------------------------------
    # Targets
    # --------------------------------------------------

    loi = numeric(
        pick_column(external, "LOI")
    )
    phrr = numeric(
        pick_column(
            external,
            "PHRR_kW_m2",
            "PHRR_kw_m2",
            "PHRR_kw_㎡",
        )
    )
    thr = numeric(
        pick_column(
            external,
            "THR_MJ_m2",
            "THR_MJ_㎡",
        )
    )
    ts = numeric(
        pick_column(
            external,
            "TS_MPa",
        )
    )
    fs = numeric(
        pick_column(
            external,
            "FS_MPa",
        )
    )

    ul94 = clean_text(
        pick_column(
            external,
            "UL94_rating",
            "UL94",
        )
    )

    char_temp = numeric(
        pick_column(
            external,
            "TGA_char_temp_C",
        )
    )
    char_raw = numeric(
        pick_column(
            external,
            "TGA_char_yield_pct",
        )
    )
    char700 = char_raw.where(
        char_temp.eq(700)
    )

    tg_raw = numeric(
        pick_column(
            external,
            "Tg_reported_C",
            "Tg_DMA_C",
        )
    )
    tg_method = clean_text(
        pick_column(
            external,
            "Tg_method",
        )
    )
    tg_dma = tg_raw.where(
        tg_method.str.contains(
            "DMA",
            case=False,
            regex=False,
        )
    )

    set_standard("LOI", loi)
    set_standard("PHRR", phrr)
    set_standard("THR", thr)
    set_standard("UL94", ul94)
    set_standard("Tg", tg_dma)
    set_standard("Char_yield", char700)
    set_standard("TS_MPa", ts)
    set_standard("FS_MPa", fs)

    # --------------------------------------------------
    # V0 numeric encoding, learned from existing V4 table
    # --------------------------------------------------

    ul94_num_col = colmap.get("UL94_num")
    ul94_col = colmap.get("UL94")

    if (
        ul94_num_col in train.columns
        and ul94_col in train.columns
    ):
        mapping = {}

        table = pd.DataFrame({
            "rating": train[ul94_col]
            .fillna("")
            .astype(str)
            .str.upper()
            .str.strip(),
            "num": pd.to_numeric(
                train[ul94_num_col],
                errors="coerce",
            ),
        }).dropna()

        for key, sub in table.groupby("rating"):
            if len(sub):
                mapping[key] = sub["num"].mode().iloc[0]

        prepared[ul94_num_col] = (
            ul94.str.upper().map(mapping)
        )

    # --------------------------------------------------
    # EP baseline fields
    # --------------------------------------------------

    set_standard(
        "EP_matrix_LOI",
        numeric(
            pick_column(
                external,
                "EP_matrix_LOI",
            )
        ),
    )

    set_standard(
        "EP_matrix_PHRR",
        numeric(
            pick_column(
                external,
                "EP_matrix_PHRR",
            )
        ),
    )

    set_standard(
        "EP_matrix_THR",
        numeric(
            pick_column(
                external,
                "EP_matrix_THR",
            )
        ),
    )

    set_standard(
        "EP_matrix_TS",
        numeric(
            pick_column(
                external,
                "EP_matrix_TS",
            )
        ),
    )

    set_standard(
        "EP_matrix_FS",
        numeric(
            pick_column(
                external,
                "EP_matrix_FS",
            )
        ),
    )

    ep_char = numeric(
        pick_column(
            external,
            "EP_matrix_TGA_char_pct",
        )
    ).where(char_temp.eq(700))

    set_standard(
        "EP_matrix_CY",
        ep_char,
    )

    # Delta fields are retained for traceability.
    set_standard(
        "Delta_LOI",
        loi - numeric(
            pick_column(
                external,
                "EP_matrix_LOI",
            )
        ),
    )

    set_standard(
        "Delta_PHRR",
        phrr - numeric(
            pick_column(
                external,
                "EP_matrix_PHRR",
            )
        ),
    )

    set_standard(
        "Delta_THR",
        thr - numeric(
            pick_column(
                external,
                "EP_matrix_THR",
            )
        ),
    )

    set_standard(
        "Delta_TS",
        ts - numeric(
            pick_column(
                external,
                "EP_matrix_TS",
            )
        ),
    )

    set_standard(
        "Delta_FS",
        fs - numeric(
            pick_column(
                external,
                "EP_matrix_FS",
            )
        ),
    )

    set_standard(
        "Reference",
        clean_text(
            pick_column(
                external,
                "Source_URL",
                "Paper_ID",
            )
        ),
    )

    # --------------------------------------------------
    # Optional raw EP category if present in V4
    # --------------------------------------------------

    ep_series = clean_text(
        pick_column(
            external,
            "EP_matrix",
        )
    )

    set_first_existing(
        [
            "EP_matrix",
            "EP_type",
            "Epoxy_matrix",
        ],
        ep_series,
    )

    # --------------------------------------------------
    # Attach audit fields to external metadata
    # --------------------------------------------------

    audit_small = audit.copy()

    external = external.merge(
        audit_small,
        on="FR_main",
        how="left",
        suffixes=("", "_audit"),
    )

    external["Standardized_total_FR_wt_pct"] = (
        standardized_loading
    )

    return prepared, external


# ---------------------------------------------------------------------
# Locate / reconstruct corrected V5 deployable model
# ---------------------------------------------------------------------

def find_existing_bundle(task: str):
    """Load the single frozen FINAL bundle from the manifest."""
    try:
        bundle, path = load_final_bundle(task)
    except FileNotFoundError:
        return None, None
    return path, bundle


def load_or_rebuild_bundle(
    task: str,
    train_bundle,
    output_dir: Path,
):
    """Load the frozen FINAL bundle only.

    The historical function name is retained for call compatibility, but no
    reconstruction or model selection is permitted during external validation.
    """
    path, model_bundle = find_existing_bundle(task)
    if model_bundle is None or path is None:
        raise FileNotFoundError(
            f"{task}: frozen FINAL bundle is missing. "
            "Generate/freeze the FINAL internal model before external validation; "
            "no curated/V4/rebuilt fallback is permitted."
        )
    print(f"[MODEL] {task}: using frozen FINAL bundle")
    print(f"[MODEL] {path}")
    return model_bundle, path


# ---------------------------------------------------------------------
# External target extraction
# ---------------------------------------------------------------------

def external_truth(
    external: pd.DataFrame,
    task: str,
) -> pd.Series:

    if task == "LOI":
        return numeric(
            pick_column(external, "LOI")
        )

    if task == "PHRR":
        value = numeric(
            pick_column(
                external,
                "PHRR_kW_m2",
                "PHRR_kw_m2",
            )
        )

        flux = numeric(
            pick_column(
                external,
                "Cone_flux_kW_m2",
            )
        )

        return value.where(
            flux.notna()
        )

    if task == "THR":
        value = numeric(
            pick_column(
                external,
                "THR_MJ_m2",
            "THR_MJ_㎡",
            )
        )

        flux = numeric(
            pick_column(
                external,
                "Cone_flux_kW_m2",
            )
        )

        return value.where(
            flux.notna()
        )

    if task == "UL94_V0":
        binary = numeric(
            pick_column(
                external,
                "UL94_V0_binary",
            )
        )

        if binary.notna().any():
            return binary

        rating = clean_text(
            pick_column(
                external,
                "UL94_rating",
                "UL94",
            )
        )

        return rating.apply(
            lambda x:
            np.nan if x == ""
            else (1 if x.upper() == "V-0" else 0)
        )

    if task == "Char_yield":
        value = numeric(
            pick_column(
                external,
                "TGA_char_yield_pct",
            )
        )

        temp = numeric(
            pick_column(
                external,
                "TGA_char_temp_C",
            )
        )

        return value.where(
            temp.eq(700)
        )

    if task == "Tg":
        value = numeric(
            pick_column(
                external,
                "Tg_reported_C",
                "Tg_DMA_C",
                "Tg_℃",
            )
        )

        method = clean_text(
            pick_column(
                external,
                "Tg_method",
            )
        )

        return value.where(
            method.str.contains(
                "DMA",
                case=False,
                regex=False,
            )
        )

    if task == "TS_MPa":
        return numeric(
            pick_column(
                external,
                "TS_MPa",
            )
        )

    if task == "FS_MPa":
        return numeric(
            pick_column(
                external,
                "FS_MPa",
            )
        )

    return pd.Series(
        np.nan,
        index=external.index,
    )


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def regression_external_metrics(
    y_true,
    y_pred,
):
    y_true = np.asarray(
        y_true,
        dtype=float,
    )
    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    n = len(y_true)

    result = {
        "n": n,
        "MAE": (
            float(mean_absolute_error(y_true, y_pred))
            if n else np.nan
        ),
        "RMSE": (
            float(
                math.sqrt(
                    mean_squared_error(
                        y_true,
                        y_pred,
                    )
                )
            )
            if n else np.nan
        ),
        "R2": np.nan,
        "Spearman": np.nan,
    }

    if n >= 3 and np.nanstd(y_true) > 0:
        result["R2"] = float(
            r2_score(
                y_true,
                y_pred,
            )
        )

    if n >= 3:
        yt_rank = pd.Series(y_true).rank()
        yp_rank = pd.Series(y_pred).rank()

        result["Spearman"] = float(
            yt_rank.corr(yp_rank)
        )

    return result


def classification_external_metrics(
    y_true,
    y_pred,
    probability,
):
    y_true = np.asarray(
        y_true,
        dtype=int,
    )
    y_pred = np.asarray(
        y_pred,
        dtype=int,
    )
    probability = np.asarray(
        probability,
        dtype=float,
    )

    result = {
        "n": len(y_true),
        "Accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "Balanced_Accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "Macro_F1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "ROC_AUC": np.nan,
    }

    if len(np.unique(y_true)) == 2:
        result["ROC_AUC"] = float(
            roc_auc_score(
                y_true,
                probability,
            )
        )

    return result


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

def make_task_figure(
    table: pd.DataFrame,
    task: str,
    output_dir: Path,
):
    usable = table[
        table["metric_eligible"] == 1
    ].copy()

    if usable.empty:
        return

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if task != "UL94_V0":
        fig, ax = plt.subplots(
            figsize=(5.5, 4.8)
        )

        ax.scatter(
            usable["experimental"],
            usable["predicted"],
            s=45,
        )

        low = min(
            usable["experimental"].min(),
            usable["predicted"].min(),
        )
        high = max(
            usable["experimental"].max(),
            usable["predicted"].max(),
        )

        pad = (
            (high - low) * 0.08
            if high > low else 1.0
        )

        ax.plot(
            [low - pad, high + pad],
            [low - pad, high + pad],
            "--",
            linewidth=1,
        )

        for _, r in usable.iterrows():
            ax.annotate(
                str(r["FR_main"]),
                (
                    r["experimental"],
                    r["predicted"],
                ),
                fontsize=6.5,
                xytext=(3, 3),
                textcoords="offset points",
            )

        ax.set_xlabel(
            f"Experimental {task}"
        )
        ax.set_ylabel(
            f"Predicted {task}"
        )
        ax.set_title(
            f"Independent external validation: {task}"
        )

        fig.tight_layout()

    else:
        fig, ax = plt.subplots(
            figsize=(max(7, 0.42 * len(usable)), 4.8)
        )

        x = np.arange(len(usable))

        ax.scatter(
            x,
            usable["predicted_probability"],
            s=45,
        )

        threshold = pd.to_numeric(
            usable["threshold"],
            errors="coerce",
        ).median()

        if pd.notna(threshold):
            ax.axhline(
                threshold,
                linestyle="--",
                linewidth=1,
                label=f"threshold={threshold:.3f}",
            )

        labels = [
            f"{a}\n{b}"
            for a, b in zip(
                usable["FR_main"],
                usable["Sample_ID"],
            )
        ]

        ax.set_xticks(
            x,
            labels,
            rotation=60,
            ha="right",
            fontsize=7,
        )

        ax.set_ylabel(
            "Predicted UL-94 V-0 probability"
        )
        ax.set_title(
            "Independent external validation: UL-94 V-0"
        )

        if pd.notna(threshold):
            ax.legend(
                frameon=False,
                fontsize=8,
            )

        fig.tight_layout()

    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(
            fig_dir
            / f"ExternalValidation_{task}.{suffix}",
            dpi=400 if suffix == "png" else None,
            bbox_inches="tight",
        )

    plt.close(fig)


def make_ad_figure(
    predictions: pd.DataFrame,
    output_dir: Path,
):
    data = predictions[
        (predictions["metric_eligible"] == 1)
        & predictions["relative_absolute_error_pct"].notna()
        & predictions["Max_Tanimoto_to_V4"].notna()
        & predictions["task"].ne("UL94_V0")
    ].copy()

    if data.empty:
        return

    fig, ax = plt.subplots(
        figsize=(6.0, 4.8)
    )

    ax.scatter(
        data["Max_Tanimoto_to_V4"],
        data["relative_absolute_error_pct"],
        s=40,
    )

    ax.axvline(
        0.50,
        linestyle="--",
        linewidth=1,
    )
    ax.axvline(
        0.70,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel(
        "Maximum Tanimoto similarity to V4 development set"
    )
    ax.set_ylabel(
        "Absolute relative error (%)"
    )
    ax.set_title(
        "External prediction error versus structural similarity"
    )

    fig.tight_layout()

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(
            fig_dir
            / f"ExternalValidation_AD_error_vs_similarity.{suffix}",
            dpi=400 if suffix == "png" else None,
            bbox_inches="tight",
        )

    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run(args):
    output_dir = Path(args.output)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = Path(args.train)
    external_path = Path(args.external)

    if not train_path.exists():
        raise FileNotFoundError(
            f"V4 training data not found: {train_path}"
        )

    train = read_csv_auto(
        train_path
    )

    external, external_source_kind, resolved_external_path = resolve_external_source(
        external_path, train
    )

    print("=" * 88)
    print("[EXTERNAL VALIDATION]")
    print("=" * 88)
    print(f"[DATA] frozen V5 training rows={len(train)}")
    print(f"[DATA] external literature rows={len(external)}")
    print(f"[DATA] external source kind={external_source_kind}")
    print(f"[DATA] external source={resolved_external_path}")

    # --------------------------------------------------
    # Structure audit
    # --------------------------------------------------

    audit = audit_external_molecules(
        train,
        external,
        output_dir,
    )

    if (
        audit["SMILES_valid"]
        .fillna("FAIL")
        .ne("PASS")
        .any()
    ):
        raise RuntimeError(
            "At least one external SMILES is invalid. "
            "Fix the structure before prediction."
        )

    if (
        audit["V4_exact_canonical_match"]
        .fillna("YES")
        .eq("YES")
        .any()
    ):
        print(
            "[WARN] At least one external molecule is already present "
            "in the V4 canonical development set."
        )

    # --------------------------------------------------
    # Prepare external rows in exact V4 raw schema
    # --------------------------------------------------

    if external_source_kind == "prepared_csv":
        external_v4 = _coerce_prepared_external_to_train_schema(train, external)
        external_meta = _normalise_prepared_external_metadata(external)
        external_meta = external_meta.merge(
            audit,
            on="FR_main",
            how="left",
            suffixes=("", "_audit"),
        )
        print(
            "[DATA] prepared external CSV already follows the project raw schema; "
            "reusing audited values without re-deriving formulation fields."
        )
    else:
        external_v4, external_meta = prepare_external_as_v4_rows(
            train,
            external,
            audit,
        )

    external_v4.to_csv(
        output_dir
        / "External_Validation_V4_ready.csv",
        index=False,
        encoding="utf-8-sig",
    )

    external_meta.to_csv(
        output_dir
        / "external_validation_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------
    # Build development-only feature bundle
    # --------------------------------------------------

    train_bundle = se.prepare_scientific_bundle(
        train_path
    )

    # --------------------------------------------------
    # Generate external features using same V4 feature generator
    # --------------------------------------------------

    merged = pd.concat(
        [
            train.reset_index(drop=True),
            external_v4.reset_index(drop=True),
        ],
        ignore_index=True,
        sort=False,
    )

    merged_path = (
        output_dir
        / "_merged_train_external_for_feature_generation.csv"
    )

    merged.to_csv(
        merged_path,
        index=False,
        encoding="utf-8-sig",
    )

    merged_bundle = se.prepare_scientific_bundle(
        merged_path
    )

    external_start = len(train)

    task_names = [
        x.strip()
        for x in args.tasks.split(",")
        if x.strip()
    ]

    all_predictions = []
    metric_rows = []
    model_rows = []

    for task in task_names:
        if task not in se.TASK_CONFIGS:
            print(
                f"[SKIP] unsupported task={task}"
            )
            continue

        y_true = external_truth(
            external_meta,
            task,
        )

        if y_true.notna().sum() == 0:
            print(
                f"[SKIP] {task}: no compatible external target values"
            )
            continue

        model_bundle, model_path = (
            load_or_rebuild_bundle(
                task,
                train_bundle,
                output_dir,
            )
        )

        if bool(
            model_bundle.get(
                "use_BDE",
                False,
            )
        ):
            raise RuntimeError(
                f"{task}: selected bundle uses BDE. "
                "External validation must use the formal without-BDE model."
            )

        view = str(
            model_bundle["view"]
        )

        matrix_all = se._filter_task_matrix(
            merged_bundle,
            task,
            view,
            use_bde=False,
            screening_mode=str(
                model_bundle.get(
                    "screening_mode",
                    "formulation",
                )
            ),
            feature_scope=str(
                model_bundle.get(
                    "feature_scope",
                    "all",
                )
            ),
        )
        matrix_all = se._apply_baseline_inclusive_mask(
            matrix_all,
            merged_bundle.df,
            merged_bundle.base.colmap,
            task,
            "baseline_inclusive",
        )

        X_ext = (
            matrix_all
            .iloc[external_start:]
            .reset_index(drop=True)
        )

        if len(X_ext) != len(external_meta):
            raise RuntimeError(
                f"{task}: external feature row mismatch "
                f"{len(X_ext)} != {len(external_meta)}"
            )

        pipe = model_bundle["pipeline"]

        pred_table = pd.DataFrame({
            "Record_ID": clean_text(
                pick_column(
                    external_meta,
                    "Record_ID",
                )
            ),
            "task": task,
            "FR_main": clean_text(
                external_meta["FR_main"]
            ),
            "Sample_ID": clean_text(
                pick_column(
                    external_meta,
                    "Sample_ID",
                )
            ),
            "Row_role": clean_text(
                pick_column(
                    external_meta,
                    "Row_role",
                )
            ),
            "Validation_set": clean_text(
                pick_column(
                    external_meta,
                    "Validation_set",
                )
            ),
            "Standardized_total_FR_wt_pct": numeric(
                external_meta[
                    "Standardized_total_FR_wt_pct"
                ]
            ),
            "experimental": y_true,
        })

        audit_cols = [
            "Canonical_SMILES_main",
            "Murcko_scaffold",
            "V4_exact_canonical_match",
            "V4_scaffold_seen",
            "Max_Tanimoto_to_V4",
            "Nearest_V4_FR",
            "AD_category",
            "External_identity_level",
            "External_validation_status",
        ]

        audit_per_row = (
            external_meta[["FR_main"]]
            .merge(
                audit[
                    ["FR_main"]
                    + [
                        c for c in audit_cols
                        if c in audit.columns
                    ]
                ],
                on="FR_main",
                how="left",
            )
        )

        for col in audit_cols:
            if col in audit_per_row.columns:
                pred_table[col] = (
                    audit_per_row[col].values
                )

        if model_bundle["task_type"] == "regression":
            prediction = np.asarray(
                pipe.predict(X_ext),
                dtype=float,
            )

            pred_table["predicted"] = prediction

            q = model_bundle.get(
                "conformal_q",
                np.nan,
            )

            if q is not None and np.isfinite(q):
                pred_table["PI_lower"] = prediction - float(q)
                pred_table["PI_upper"] = prediction + float(q)
                pred_table["PI_covered"] = (
                    (y_true >= prediction - float(q))
                    & (y_true <= prediction + float(q))
                ).astype("Int64")
            else:
                pred_table["PI_lower"] = np.nan
                pred_table["PI_upper"] = np.nan
                pred_table["PI_covered"] = np.nan

            pred_table["predicted_probability"] = np.nan
            pred_table["threshold"] = np.nan

        else:
            raw_prob = np.asarray(
                pipe.predict_proba(X_ext)[:, 1],
                dtype=float,
            )

            calibrator = model_bundle.get(
                "probability_calibrator"
            )

            if calibrator is not None:
                probability = se._apply_platt(
                    calibrator,
                    raw_prob,
                )
            else:
                probability = raw_prob
                warnings.warn(
                    "UL94 probability calibrator missing; "
                    "raw model probabilities are being used."
                )

            threshold = float(
                model_bundle.get(
                    "threshold",
                    0.5,
                )
            )

            prediction = (
                probability >= threshold
            ).astype(int)

            pred_table["predicted_probability"] = probability
            pred_table["raw_probability"] = raw_prob
            pred_table["threshold"] = threshold
            pred_table["predicted"] = prediction
            pred_table["PI_lower"] = np.nan
            pred_table["PI_upper"] = np.nan
            pred_table["PI_covered"] = np.nan

        # --------------------------------------------------
        # Formal metric eligibility:
        # positive-loading new FR formulations only
        # --------------------------------------------------

        loading = pred_table[
            "Standardized_total_FR_wt_pct"
        ]

        eligible = (
            loading.fillna(0).gt(0)
            & pred_table["experimental"].notna()
        )

        pred_table["metric_eligible"] = (
            eligible.astype(int)
        )

        pred_table["residual"] = (
            pred_table["predicted"]
            - pred_table["experimental"]
        )

        pred_table["absolute_error"] = (
            pred_table["residual"].abs()
        )

        denominator = (
            pred_table["experimental"]
            .abs()
            .replace(0, np.nan)
        )

        pred_table[
            "relative_absolute_error_pct"
        ] = (
            pred_table["absolute_error"]
            / denominator
            * 100.0
        )

        task_dir = (
            output_dir
            / "predictions_by_task"
        )
        task_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        pred_table.to_csv(
            task_dir
            / f"{task}_external_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )

        formal = pred_table[
            pred_table["metric_eligible"] == 1
        ].copy()

        if model_bundle["task_type"] == "regression":
            metrics = regression_external_metrics(
                formal["experimental"],
                formal["predicted"],
            )

            if "PI_covered" in formal.columns:
                coverage = pd.to_numeric(
                    formal["PI_covered"],
                    errors="coerce",
                )

                metrics["PICP"] = (
                    float(coverage.mean())
                    if coverage.notna().any()
                    else np.nan
                )

        else:
            metrics = classification_external_metrics(
                formal["experimental"],
                formal["predicted"],
                formal["predicted_probability"],
            )

        metrics.update({
            "task": task,
            "task_type": model_bundle["task_type"],
            "model_name": model_bundle.get("model_name"),
            "view": model_bundle.get("view"),
            "requested_k": model_bundle.get("requested_k"),
            "model_bundle_path": str(model_path),
        })

        metric_rows.append(metrics)

        model_rows.append({
            "task": task,
            "model_name": model_bundle.get("model_name"),
            "view": model_bundle.get("view"),
            "requested_k": model_bundle.get("requested_k"),
            "bundle_path": str(model_path),
            "bundle_rebuilt_for_external": int(
                bool(
                    model_bundle.get(
                        "external_validation_rebuilt",
                        False,
                    )
                )
            ),
        })

        all_predictions.append(
            pred_table
        )

        print(
            f"[PRED] {task}: "
            f"external rows={len(pred_table)}, "
            f"formal n={len(formal)}"
        )

        make_task_figure(
            pred_table,
            task,
            output_dir,
        )

    if not all_predictions:
        raise RuntimeError(
            "No external predictions were generated."
        )

    predictions_all = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    metrics_df = pd.DataFrame(
        metric_rows
    )

    models_df = pd.DataFrame(
        model_rows
    )

    predictions_all.to_csv(
        output_dir
        / "external_predictions_all_tasks.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics_df.to_csv(
        output_dir
        / "external_validation_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    models_df.to_csv(
        output_dir
        / "external_validation_model_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    make_ad_figure(
        predictions_all,
        output_dir,
    )

    run_info = {
        "V4_training_file": str(train_path),
        "V4_training_rows": len(train),
        "external_source": str(resolved_external_path),
        "external_source_kind": external_source_kind,
        "external_rows": len(external_meta),
        "tasks_requested": task_names,
        "formal_metric_policy": (
            "positive-loading external FR formulations only; "
            "Neat EP retained for tracing but excluded from external metrics"
        ),
        "model_policy": (
            "use frozen V5 FINAL final_model_bundle only; "
            "deployable model using V4 development data only"
        ),
        "BDE_policy": "without BDE only",
        "AD_thresholds": {
            "in_domain": "Smax >= 0.70",
            "caution": "0.50 <= Smax < 0.70",
            "extrapolation": "Smax < 0.50",
        },
    }

    (
        output_dir
        / "external_validation_run_manifest.json"
    ).write_text(
        json.dumps(
            run_info,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Temporary merged file is useful for reproducibility;
    # keep it, but clearly label that it is not a training database.
    print("=" * 88)
    print("[DONE] Independent external validation")
    print(f"[OUT] {output_dir}")
    print("=" * 88)
    print(metrics_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        default=str(DEFAULT_TRAIN),
    )

    parser.add_argument(
        "--external",
        default=str(DEFAULT_EXTERNAL),
        help=(
            "Audited external Excel workbook. If it is absent, the runner "
            "automatically falls back to data/external_validation/"
            "External_Validation_V4_ready.csv."
        ),
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
    )

    parser.add_argument(
        "--tasks",
        default=",".join(DEFAULT_TASKS),
    )

    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()