# -*- coding: utf-8 -*-
"""Generate canonical SMILES/InChIKey/scaffold and duplicate-audit tables."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import pipeline_core as core
from common.chem_standardization import save_standardization_audit, standardize_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "DOPO_EP_new_with_BDE.csv")
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "scientific_validation" / "data_standardization")
    args = parser.parse_args()

    raw = core.read_csv_auto(str(args.input))
    colmap = core.resolve_columns(raw)
    cleaned = core.clean_dataframe(raw, colmap)
    standardized = standardize_dataset(cleaned, colmap)
    save_standardization_audit(standardized, args.results)
    print(f"[DONE] Standardization audit saved to: {args.results}")


if __name__ == "__main__":
    main()
