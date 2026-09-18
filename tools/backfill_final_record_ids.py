# -*- coding: utf-8 -*-
"""One-time audit repair: backfill stable Record_ID into existing FINAL outer-prediction CSVs.

This script does NOT retrain models and does NOT alter predictions, folds, metrics, or model bundles.
It maps each existing outer-prediction row back to the frozen 599-row source table using
stable raw identity fields plus task target/loading. Existing files are backed up before writing.

Usage from the project root:
    python -u tools/backfill_final_record_ids.py
"""
from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_CANDIDATES = [
    ROOT / "data" / "DOPO_EP_new_with_BDE.csv",
    ROOT / "data" / "DOPO_EP_new.csv",
]
BACKUP_ROOT = ROOT / "results" / "audit" / "record_id_backfill_backup"

TARGET_ALIASES = {
    "LOI": ["LOI"],
    "PHRR": ["PHRR_kw_㎡", "PHRR_kW_m2"],
    "THR": ["THR_MJ_㎡", "THR_MJ_m2"],
    "UL94_V0": ["UL94_V0", "UL94_num", "UL94", "UL94_rating"],
    "Tg": ["Tg_℃", "Tg_DMA_C", "Tg"],
    "Char_yield": ["Char_yield_％_700C", "Char_yield_700C", "Char_yield"],
    "TS_MPa": ["TS_MPa"],
    "FS_MPa": ["FS_MPa"],
    "Delta_LOI": ["Delta_LOI"],
    "Delta_PHRR": ["Delta_PHRR"],
    "Delta_THR": ["Delta_THR"],
    "Delta_CY": ["Delta_CY"],
}

RAW_ID_FIELDS = ["FR_main", "FR_co", "SMILES_main", "SMILES_co", "Reference"]


def read_csv_flexible(path: Path) -> pd.DataFrame:
    last = None
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as exc:
            last = exc
    if last:
        raise last
    return pd.read_csv(path)


def norm_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def norm_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round(8)


def norm_target(series: pd.Series, task: str) -> pd.Series:
    """Normalize target values for stable identity matching.

    UL94_V0 is sometimes stored only as the original UL94 rating or UL94_num
    in the frozen 599-row CSV. Convert either representation to the binary
    V-0 target used in outer-prediction files. Other tasks remain numeric.
    """
    if task != "UL94_V0":
        return norm_num(series)

    # If already binary numeric (0/1), preserve it.
    num = pd.to_numeric(series, errors="coerce")
    non_missing = num.dropna()
    if len(non_missing) and set(non_missing.unique()).issubset({0, 1, 0.0, 1.0}):
        return num.round(8)

    # Project UL94_num convention: V-0=3, V-1=2, V-2=1, NR=0.
    if len(non_missing) and set(non_missing.unique()).issubset({0, 1, 2, 3, 0.0, 1.0, 2.0, 3.0}):
        out = pd.Series(np.nan, index=series.index, dtype=float)
        mask = num.notna()
        out.loc[mask] = (num.loc[mask] == 3).astype(float)
        return out

    # Original string ratings, e.g. V-0 / V-1 / V-2 / NR.
    txt = series.fillna("").astype(str).str.strip().str.upper().str.replace(" ", "", regex=False)
    out = pd.Series(np.nan, index=series.index, dtype=float)
    valid = txt.ne("")
    out.loc[valid] = 0.0
    out.loc[txt.isin({"V-0", "V0"})] = 1.0
    return out


def first_existing(frame: pd.DataFrame, names: list[str]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def infer_task(path: Path, frame: pd.DataFrame) -> str:
    if "task" in frame.columns and frame["task"].notna().any():
        return str(frame["task"].dropna().iloc[0])
    name = path.name
    if name.endswith("_outer_predictions.csv"):
        return name[: -len("_outer_predictions.csv")]
    raise RuntimeError(f"Cannot infer task for {path}")


def build_key_frame(source: pd.DataFrame, pred: pd.DataFrame, task: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    src = pd.DataFrame(index=source.index)
    out = pd.DataFrame(index=pred.index)

    for col in RAW_ID_FIELDS:
        if col not in source.columns or col not in pred.columns:
            raise RuntimeError(f"{task}: required identity field missing: {col}")
        src[col] = norm_text(source[col])
        out[col] = norm_text(pred[col])

    src_loading = first_existing(source, ["Loading_total_FR wt%", "Loading_total_FR_wt%"])
    pred_loading = first_existing(pred, ["Loading_total_FR_for_audit", "Loading_total_FR wt%", "Loading_total_FR_wt%"])
    if src_loading and pred_loading:
        src["__loading__"] = norm_num(source[src_loading])
        out["__loading__"] = norm_num(pred[pred_loading])

    src_target = first_existing(source, TARGET_ALIASES.get(task, [task]))
    pred_target = first_existing(pred, ["y_true", task])
    if src_target is None or pred_target is None:
        raise RuntimeError(
            f"{task}: cannot resolve target column; source candidates={TARGET_ALIASES.get(task, [task])}, "
            f"prediction target expected y_true"
        )
    src["__target__"] = norm_target(source[src_target], task)
    out["__target__"] = norm_target(pred[pred_target], task)
    return src, out


def tuple_key(row: np.ndarray) -> tuple:
    vals = []
    for value in row:
        if isinstance(value, float) and np.isnan(value):
            vals.append("<NA_NUM>")
        else:
            vals.append(value)
    return tuple(vals)


def sync_one(path: Path, source: pd.DataFrame) -> dict:
    pred = read_csv_flexible(path)
    task = infer_task(path, pred)

    if "Record_ID" in pred.columns:
        raw_ids = pred["Record_ID"]
        ids = raw_ids.fillna("").astype(str).str.strip()
        invalid = ids.eq("") | ids.str.lower().isin({"nan", "none", "<na>"})
        if (not invalid.any()) and ids.nunique(dropna=False) == len(pred):
            return {"file": str(path), "task": task, "rows": len(pred), "status": "already_ok", "ambiguous_groups": 0}

    src_key_df, pred_key_df = build_key_frame(source, pred, task)
    source_lookup: dict[tuple, list[int]] = defaultdict(list)
    for idx, row in zip(source.index, src_key_df.to_numpy(dtype=object)):
        source_lookup[tuple_key(row)].append(int(idx))

    pred_keys = [tuple_key(row) for row in pred_key_df.to_numpy(dtype=object)]
    occurrence: dict[tuple, int] = defaultdict(int)
    assigned: list[str] = []
    ambiguous_groups: set[tuple] = set()

    for row_no, key in enumerate(pred_keys):
        candidates = source_lookup.get(key, [])
        if not candidates:
            raise RuntimeError(
                f"{task}: no frozen-data match for prediction row {row_no} in {path}\n"
                f"Key={key}"
            )
        pos = occurrence[key]
        if pos >= len(candidates):
            raise RuntimeError(
                f"{task}: prediction has more occurrences than frozen data for key at row {row_no}; "
                f"pred_occurrence={pos + 1}, source_occurrences={len(candidates)}"
            )
        if len(candidates) > 1:
            ambiguous_groups.add(key)
        source_idx = candidates[pos]
        occurrence[key] += 1
        assigned.append(str(source.loc[source_idx, "Record_ID"]))

    if len(set(assigned)) != len(assigned):
        raise RuntimeError(f"{task}: Record_ID backfill would create duplicate IDs in {path}")

    # Verify every prediction occurrence consumed no more than the corresponding source occurrence count.
    for key, used in occurrence.items():
        if used > len(source_lookup[key]):
            raise RuntimeError(f"{task}: invalid occurrence mapping for key={key}")

    rel = path.relative_to(ROOT)
    backup = BACKUP_ROOT / rel
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        shutil.copy2(path, backup)

    if "Record_ID" in pred.columns:
        pred["Record_ID"] = assigned
    else:
        insert_at = 2 if "outer_fold" in pred.columns else 0
        pred.insert(insert_at, "Record_ID", assigned)
    pred.to_csv(path, index=False, encoding="utf-8-sig")

    return {
        "file": str(path),
        "task": task,
        "rows": len(pred),
        "status": "backfilled",
        "ambiguous_groups": len(ambiguous_groups),
    }


def main() -> None:
    data_path = next((p for p in DATA_CANDIDATES if p.exists()), None)
    if data_path is None:
        raise FileNotFoundError("Frozen source CSV not found under data/")
    source = read_csv_flexible(data_path)
    if "Record_ID" not in source.columns:
        raise RuntimeError(
            f"Frozen source table has no Record_ID: {data_path}. Run the approved data-release migration first."
        )
    if len(source) != 599:
        raise RuntimeError(f"Expected frozen dataset with 599 rows, found {len(source)}")
    if source["Record_ID"].isna().any() or source["Record_ID"].astype(str).duplicated().any():
        raise RuntimeError("Frozen Record_ID values are missing or duplicated; refusing to modify results.")

    roots = [
        ROOT / "results" / "scientific_validation" / "FINAL_core_fixed_baseline_inclusive_5x5",
        ROOT / "results" / "scientific_validation" / "FINAL_aux_fixed_baseline_inclusive_5x5",
        ROOT / "results" / "scientific_validation" / "FINAL_exploratory_fixed_baseline_inclusive_5x5",
        ROOT / "results" / "05_Shap" / "FINAL_core_fixed_baseline_inclusive_5x5",
        ROOT / "results" / "05_Shap" / "FINAL_aux_fixed_baseline_inclusive_5x5",
    ]
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(sorted(root.rglob("*_outer_predictions.csv")))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError("No FINAL outer-prediction CSVs found to synchronize.")

    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in files:
        result = sync_one(path, source)
        rows.append(result)
        print(
            f"[{result['status'].upper()}] {result['task']}: rows={result['rows']} "
            f"ambiguous_key_groups={result['ambiguous_groups']} -> {path.relative_to(ROOT)}"
        )

    report = pd.DataFrame(rows)
    report_path = ROOT / "results" / "audit" / "record_id_backfill_report.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False, encoding="utf-8-sig")
    print("\n[DONE] Stable Record_ID synchronization completed.")
    print(f"[BACKUP] {BACKUP_ROOT}")
    print(f"[REPORT] {report_path}")
    print("[NEXT] Run: python -u run.py paper-audit")


if __name__ == "__main__":
    main()
