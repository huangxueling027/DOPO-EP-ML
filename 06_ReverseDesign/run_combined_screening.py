# -*- coding: utf-8 -*-
"""Rebuild the complete frozen-raw PubChem + designed combined screening chain.

This runner does not download new PubChem data.  It starts from the frozen raw
PubChem query files already stored under public_database/data/01_pubchem_raw/.
It requires the canonical frozen deployment bundles from the strict 5x5
scientific validation before the two candidate-screen steps can run.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REVERSE_DIR = Path(__file__).resolve().parent
ROOT = REVERSE_DIR.parent
PUBLIC_DIR = REVERSE_DIR / "public_database"
RESULTS = ROOT / "results" / "06_ReverseDesign" / "FINAL_fixed"


def run(*parts: object) -> None:
    cmd = [sys.executable, "-u", *[str(p) for p in parts]]
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-loading", type=float, default=30.0)
    parser.add_argument("--reliable-threshold", type=float, default=0.70)
    parser.add_argument("--caution-threshold", type=float, default=0.50)
    parser.add_argument(
        "--skip-public-rebuild",
        action="store_true",
        help="Reuse the existing formal PubChem prediction pool.",
    )
    parser.add_argument(
        "--skip-screening",
        action="store_true",
        help="Rebuild through combined master only; do not run the six deployment models.",
    )
    args = parser.parse_args()

    public_data = PUBLIC_DIR / "data" / "03_public_candidates"
    unique_public = public_data / "pubchem_public_candidates_unique.csv"
    public_ad = public_data / "pubchem_public_AD.csv"
    public_prediction = public_data / "pubchem_public_candidates_for_prediction.csv"

    if not args.skip_public_rebuild:
        run(PUBLIC_DIR / "standardize_pubchem_candidates.py")
        run(PUBLIC_DIR / "build_public_candidate_pool.py")
        run(
            REVERSE_DIR / "candidate_applicability_domain.py",
            "--candidates", unique_public,
            "--candidate-smiles-col", "Canonical_SMILES",
            "--candidate-name-col", "Candidate_Name",
            "--reliable-threshold", args.reliable_threshold,
            "--caution-threshold", args.caution_threshold,
            "--output", public_ad,
        )
        run(
            PUBLIC_DIR / "prepare_public_candidates_for_prediction.py",
            "--input", public_ad,
            "--output", public_prediction,
        )

    if not public_prediction.exists():
        raise FileNotFoundError(
            f"Formal PubChem prediction pool not found: {public_prediction}"
        )

    run(REVERSE_DIR / "build_combined_candidate_master.py")

    if args.skip_screening:
        print("[DONE] Combined master rebuilt; screening skipped.")
        return

    master = RESULTS / "combined" / "candidate_molecule_master_combined.csv"
    for flux, folder in [(50, "combined_flux50"), (35, "combined_flux35")]:
        outdir = RESULTS / folder
        grid = outdir / "candidate_formulation_grid.csv"
        run(
            REVERSE_DIR / "build_formulation_grid.py",
            "--master", master,
            "--cone-fluxes", flux,
            "--max-loading", args.max_loading,
            "--output", grid,
        )
        run(
            REVERSE_DIR / "predict_and_rank_candidates.py",
            "--formulations", grid,
            "--output-dir", outdir,
            "--reliable-threshold", args.reliable_threshold,
            "--caution-threshold", args.caution_threshold,
        )

    run(
        REVERSE_DIR / "build_flux_stable_priority.py",
        "--flux50", RESULTS / "combined_flux50" / "ranked_candidates.csv",
        "--flux35", RESULTS / "combined_flux35" / "ranked_candidates.csv",
        "--output-dir", RESULTS / "final_priority_combined",
    )
    print("\n[DONE] Complete combined virtual-screening chain rebuilt.")


if __name__ == "__main__":
    main()
