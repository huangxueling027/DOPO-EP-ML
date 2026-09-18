# -*- coding: utf-8 -*-
"""Audit Preparation_Method standardization without modifying frozen CSV files.

The final audited V5 release stores four canonical preparation-method labels.
This helper is intentionally read-only so running it cannot change data hashes.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    ROOT / "data" / "DOPO_EP_new.csv",
    ROOT / "data" / "DOPO_EP_new_with_BDE.csv",
]

EXPECTED = {
    "DOPO-based (additive)": 0,
    "DOPO-based (reactive)": 1,
    "DOPO-based (Co-curing)": 2,
    "DOPO-based (Additive + Secondary Crosslinking)": 3,
}


def read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeError(path)


def main() -> None:
    for path in FILES:
        if not path.exists():
            raise FileNotFoundError(path)
        df = read_csv(path)
        print("=" * 80)
        print(path.relative_to(ROOT))
        print(df["Preparation_Method"].value_counts(dropna=False))
        observed = set(df["Preparation_Method"].dropna().astype(str))
        if observed != set(EXPECTED):
            raise RuntimeError(
                f"Unexpected Preparation_Method labels: {sorted(observed)}"
            )
        if "Preparation_Method_num" not in df.columns:
            raise RuntimeError("Missing Preparation_Method_num")
        for label, code in EXPECTED.items():
            vals = pd.to_numeric(
                df.loc[df["Preparation_Method"].eq(label), "Preparation_Method_num"],
                errors="coerce",
            )
            if vals.isna().any() or not vals.eq(code).all():
                raise RuntimeError(f"Preparation code mismatch for {label}")
        print("[PASS] four canonical nominal categories; numeric code is metadata only")


if __name__ == "__main__":
    main()
