# -*- coding: utf-8 -*-
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

BASE_PATH = ROOT / "data" / "DOPO_EP_new.csv"
BDE_PATH = ROOT / "data" / "DOPO_EP_new_with_BDE.csv"


def read_csv(path: Path) -> pd.DataFrame:
    errors = []
    for encoding in ("utf-8-sig", "gb18030", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise UnicodeError(f"Unable to decode {path}\n" + "\n".join(errors))


def main() -> None:
    base = read_csv(BASE_PATH)
    bde = read_csv(BDE_PATH)

    if len(base) != len(bde):
        raise RuntimeError(
            f"Two data files have different row counts: {len(base)} vs {len(bde)}"
        )

    expected_ids = pd.Series(
        [f"DOPO_EP_{index:04d}" for index in range(1, len(base) + 1)],
        name="Record_ID",
    )

    # If both frozen files already contain the correct stable IDs, do not
    # rewrite them. This preserves their byte-level SHA256 hashes.
    if "Record_ID" in base.columns and "Record_ID" in bde.columns:
        base_ids = base["Record_ID"].reset_index(drop=True).astype(str)
        bde_ids = bde["Record_ID"].reset_index(drop=True).astype(str)
        if (
            not base["Record_ID"].isna().any()
            and not bde["Record_ID"].isna().any()
            and not base_ids.duplicated().any()
            and not bde_ids.duplicated().any()
            and base_ids.equals(expected_ids)
            and bde_ids.equals(expected_ids)
        ):
            print(f"[PASS] Record_ID already valid for {len(base)} rows; no files rewritten.")
            print(f"[PASS] First ID: {base_ids.iloc[0]}")
            print(f"[PASS] Last ID:  {base_ids.iloc[-1]}")
            return

    common_columns = [
        column
        for column in base.columns
        if column in bde.columns and column != "Record_ID"
    ]

    if not base[common_columns].equals(bde[common_columns]):
        raise RuntimeError(
            "The common fields of the two data files are not in exactly the same row order."
        )

    record_ids = expected_ids.tolist()

    # Make the operation idempotent: remove any existing ID column first,
    # then assign the same stable IDs to both frozen data files.
    base = base.drop(columns=["Record_ID"], errors="ignore")
    bde = bde.drop(columns=["Record_ID"], errors="ignore")

    base.insert(0, "Record_ID", record_ids)
    bde.insert(0, "Record_ID", record_ids)

    for name, frame in [
        ("DOPO_EP_new.csv", base),
        ("DOPO_EP_new_with_BDE.csv", bde),
    ]:
        if frame["Record_ID"].isna().any():
            raise RuntimeError(f"{name}: Record_ID contains missing values.")
        if frame["Record_ID"].duplicated().any():
            raise RuntimeError(f"{name}: duplicate Record_ID values detected.")

    base.to_csv(
        BASE_PATH,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )
    bde.to_csv(
        BDE_PATH,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )

    print(f"[PASS] Added Record_ID to {len(base)} rows.")
    print(f"[PASS] First ID: {base['Record_ID'].iloc[0]}")
    print(f"[PASS] Last ID:  {base['Record_ID'].iloc[-1]}")


if __name__ == "__main__":
    main()