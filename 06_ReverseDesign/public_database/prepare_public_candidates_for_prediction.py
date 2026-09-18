# -*- coding: utf-8 -*-
"""Create the formal PubChem prediction pool from the public AD result.

Only public candidates labelled exactly ``In_domain`` are allowed to enter the
combined screening pool.  This step is intentionally separate from AD scoring
so the gate is explicit, auditable and reproducible.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REVERSE_DIR = SCRIPT_DIR.parent

DEFAULT_INPUT = (
    SCRIPT_DIR / "data" / "03_public_candidates" / "pubchem_public_AD.csv"
)
DEFAULT_OUTPUT = (
    SCRIPT_DIR / "data" / "03_public_candidates"
    / "pubchem_public_candidates_for_prediction.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--required-domain",
        default="In_domain",
        help="Only this exact AD label is retained. Default: In_domain.",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.input, encoding="utf-8-sig")
    required = {
        "Candidate_ID",
        "Candidate_Name",
        "Canonical_SMILES",
        "Applicability_domain",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Missing required columns in {args.input}: {missing}")

    selected = frame[
        frame["Applicability_domain"].astype(str).eq(args.required_domain)
    ].copy()

    if selected.empty:
        raise RuntimeError(
            f"No public candidates passed Applicability_domain == {args.required_domain!r}"
        )

    if selected["Candidate_ID"].astype(str).duplicated().any():
        dup = selected.loc[
            selected["Candidate_ID"].astype(str).duplicated(keep=False),
            "Candidate_ID",
        ].astype(str).tolist()
        raise RuntimeError(f"Duplicate Candidate_ID in public prediction pool: {dup[:10]}")

    if "Model_Identity_SMILES" in selected.columns:
        identity = selected["Model_Identity_SMILES"].fillna("").astype(str).str.strip()
        nonempty = identity.ne("")
        if identity[nonempty].duplicated().any():
            raise RuntimeError(
                "Duplicate Model_Identity_SMILES in public prediction pool."
            )

    # These fields are downstream screening flags.  They do NOT mean that
    # synthesis feasibility has been experimentally demonstrated.
    selected["Screening_Eligible"] = True
    selected["is_unique"] = True
    selected["candidate_status"] = "Public database candidate"

    # Keep the formal pool deterministic and consistent with the frozen V4 result:
    # highest structural similarity to the training set first.
    if "Max_Tanimoto_to_training" in selected.columns:
        selected = selected.sort_values(
            ["Max_Tanimoto_to_training", "Candidate_ID"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output, index=False, encoding="utf-8-sig")

    audit = pd.DataFrame([
        {"Metric": "public_AD_rows", "Value": len(frame)},
        {
            "Metric": f"required_domain_{args.required_domain}",
            "Value": len(selected),
        },
        {
            "Metric": "unique_candidate_ids",
            "Value": selected["Candidate_ID"].nunique(),
        },
    ])
    audit_path = args.output.with_name("pubchem_prediction_pool_audit.csv")
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")

    print("[OK] Public AD rows:", len(frame))
    print(f"[OK] {args.required_domain} candidates:", len(selected))
    print("[OK] Saved:", args.output)
    print("[OK] Audit:", audit_path)


if __name__ == "__main__":
    main()
