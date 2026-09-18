# -*- coding: utf-8 -*-
"""Add Tanimoto nearest neighbours, Murcko scaffold coverage and AD labels to candidates.

Example:
    python 06_ReverseDesign/candidate_applicability_domain.py \
        --candidates results/06_ReverseDesign/candidate_molecule_master.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.applicability_domain import build_training_reference, score_candidates
from common.pipeline_core import read_csv_auto


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_training = ROOT / "data" / "DOPO_EP_new_with_BDE.csv"
    if not default_training.exists():
        default_training = ROOT / "data" / "DOPO_EP_new.csv"
    parser.add_argument("--candidates", type=Path, default=ROOT / "results" / "06_ReverseDesign" / "candidate_molecule_master.csv")
    parser.add_argument("--training", type=Path, default=default_training)
    parser.add_argument("--candidate-smiles-col", default="Canonical_SMILES")
    parser.add_argument("--candidate-name-col", default="Candidate_Name")
    parser.add_argument("--training-smiles-col", default="SMILES_main")
    parser.add_argument("--training-name-col", default="FR_main")
    parser.add_argument("--reliable-threshold", type=float, default=0.70)
    parser.add_argument("--caution-threshold", type=float, default=0.50)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "06_ReverseDesign" / "candidate_applicability_domain.csv",
    )
    args = parser.parse_args()

    candidates = read_csv_auto(str(args.candidates))
    training = read_csv_auto(str(args.training))
    references = build_training_reference(
        training,
        smiles_col=args.training_smiles_col,
        name_col=args.training_name_col,
        radius=3,
        n_bits=2048,
    )
    result = score_candidates(
        candidates,
        references,
        candidate_smiles_col=args.candidate_smiles_col,
        candidate_name_col=args.candidate_name_col,
        reliable_threshold=args.reliable_threshold,
        caution_threshold=args.caution_threshold,
        top_n=args.top_n,
        radius=3,
        n_bits=2048,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))
    print(f"[DONE] Saved: {args.output}")


if __name__ == "__main__":
    main()
