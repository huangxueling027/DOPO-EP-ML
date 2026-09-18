# -*- coding: utf-8 -*-
"""One-time, auditable migration of the two frozen V5 CSVs to the corrected release.

This script is intentionally conservative. It will NOT touch the frozen data unless:
- both expected CSV files exist;
- both files contain exactly 599 data rows;
- their common columns (apart from Record_ID) are identical and in the same row order;
- Loading_total_FR wt% exists;
- PROJECT_RULES_LOCK.json and config/task_config.json are readable JSON.

When those checks pass, it:
1. backs up both CSVs plus lock/config under results/audit/;
2. converts the CSV bytes to UTF-8-SIG without changing cell text;
3. creates stable Record_ID values DOPO_EP_0001..DOPO_EP_0599;
4. synchronizes the frozen data-release identifier (this is not the FINAL modeling protocol id);
5. recalculates and freezes SHA256 hashes in PROJECT_RULES_LOCK.json;
6. performs a post-write self-check.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Frozen data-release identifier. Formal modeling protocol is configured separately.
TARGET_VERSION = "V5_CORRECTED_MODIFIED_FORMULATIONS_2026-09"
BASE_PATH = ROOT / "data" / "DOPO_EP_new.csv"
BDE_PATH = ROOT / "data" / "DOPO_EP_new_with_BDE.csv"
LOCK_PATH = ROOT / "PROJECT_RULES_LOCK.json"
CONFIG_PATH = ROOT / "config" / "task_config.json"
BACKUP_ROOT = ROOT / "results" / "audit"
ENCODINGS = ("utf-8-sig", "gb18030", "utf-8", "gbk")
EXPECTED_ROWS = 599


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv_cells(path: Path) -> tuple[list[str], list[list[str]], str]:
    raw = path.read_bytes()
    last_error: Exception | None = None
    for encoding in ENCODINGS:
        try:
            text = raw.decode(encoding)
            rows = list(csv.reader(io.StringIO(text, newline="")))
            if not rows:
                raise RuntimeError(f"Empty CSV: {path}")
            header = rows[0]
            data = rows[1:]
            if header and header[0].startswith("\ufeff"):
                header[0] = header[0].lstrip("\ufeff")
            return header, data, encoding
        except UnicodeDecodeError as exc:
            last_error = exc
    raise UnicodeError(f"Unable to decode {path}; last error: {last_error}")


def normalize_record_id(header: list[str], rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    if "Record_ID" in header:
        idx = header.index("Record_ID")
        header = [c for i, c in enumerate(header) if i != idx]
        rows = [[v for i, v in enumerate(row) if i != idx] for row in rows]
    new_header = ["Record_ID", *header]
    new_rows = [[f"DOPO_EP_{i:04d}", *row] for i, row in enumerate(rows, start=1)]
    return new_header, new_rows


def validate_rectangular(header: list[str], rows: list[list[str]], name: str) -> None:
    width = len(header)
    bad = [i for i, row in enumerate(rows, start=2) if len(row) != width]
    if bad:
        preview = ", ".join(map(str, bad[:10]))
        raise RuntimeError(f"{name}: non-rectangular CSV rows at lines {preview}")


def compare_common_fields(
    base_header: list[str], base_rows: list[list[str]],
    bde_header: list[str], bde_rows: list[list[str]],
) -> None:
    base_index = {c: i for i, c in enumerate(base_header) if c != "Record_ID"}
    bde_index = {c: i for i, c in enumerate(bde_header) if c != "Record_ID"}
    common = [c for c in base_header if c != "Record_ID" and c in bde_index]
    if not common:
        raise RuntimeError("The two CSVs have no common data columns.")
    for row_no, (a, b) in enumerate(zip(base_rows, bde_rows), start=2):
        for col in common:
            if a[base_index[col]] != b[bde_index[col]]:
                raise RuntimeError(
                    f"Shared-field mismatch before migration at CSV line {row_no}, column {col!r}. "
                    "The two frozen files are not the same row-aligned dataset; migration stopped."
                )


def choose_backup_dir() -> Path:
    base = BACKUP_ROOT / "v5_corrected_release_migration_backup"
    candidate = base
    n = 2
    while candidate.exists():
        candidate = Path(f"{base}_{n}")
        n += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def write_csv_utf8sig(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    for path in (BASE_PATH, BDE_PATH, LOCK_PATH, CONFIG_PATH):
        if not path.exists():
            raise FileNotFoundError(path)

    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    base_header, base_rows, base_encoding = read_csv_cells(BASE_PATH)
    bde_header, bde_rows, bde_encoding = read_csv_cells(BDE_PATH)
    validate_rectangular(base_header, base_rows, BASE_PATH.name)
    validate_rectangular(bde_header, bde_rows, BDE_PATH.name)

    if len(base_rows) != EXPECTED_ROWS or len(bde_rows) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_ROWS} rows in both files; got "
            f"{len(base_rows)} and {len(bde_rows)}. Migration stopped."
        )
    if "Loading_total_FR wt%" not in base_header or "Loading_total_FR wt%" not in bde_header:
        raise RuntimeError("Required column 'Loading_total_FR wt%' is missing. Migration stopped.")

    compare_common_fields(base_header, base_rows, bde_header, bde_rows)

    backup_dir = choose_backup_dir()
    for path in (BASE_PATH, BDE_PATH, LOCK_PATH, CONFIG_PATH):
        shutil.copy2(path, backup_dir / path.name)
    print(f"[BACKUP] {backup_dir.relative_to(ROOT)}")
    print(f"[INFO] before encodings: {BASE_PATH.name}={base_encoding}, {BDE_PATH.name}={bde_encoding}")
    print(f"[INFO] rows verified: {EXPECTED_ROWS}")
    print("[INFO] shared fields and row order verified before any write")

    base_header2, base_rows2 = normalize_record_id(base_header, base_rows)
    bde_header2, bde_rows2 = normalize_record_id(bde_header, bde_rows)
    write_csv_utf8sig(BASE_PATH, base_header2, base_rows2)
    write_csv_utf8sig(BDE_PATH, bde_header2, bde_rows2)

    config["project_version"] = TARGET_VERSION
    CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lock["version"] = TARGET_VERSION
    hashes = dict(lock.get("data_sha256", {}))
    hashes[BASE_PATH.name] = sha256_file(BASE_PATH)
    hashes[BDE_PATH.name] = sha256_file(BDE_PATH)
    lock["data_sha256"] = hashes
    LOCK_PATH.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Post-write checks.
    for path in (BASE_PATH, BDE_PATH):
        if path.read_bytes()[:3] != b"\xef\xbb\xbf":
            raise RuntimeError(f"Post-write UTF-8-SIG check failed: {path.name}")
    b_h, b_r, _ = read_csv_cells(BASE_PATH)
    d_h, d_r, _ = read_csv_cells(BDE_PATH)
    expected_ids = [f"DOPO_EP_{i:04d}" for i in range(1, EXPECTED_ROWS + 1)]
    if b_h[0] != "Record_ID" or d_h[0] != "Record_ID":
        raise RuntimeError("Post-write Record_ID header check failed.")
    if [r[0] for r in b_r] != expected_ids or [r[0] for r in d_r] != expected_ids:
        raise RuntimeError("Post-write Record_ID sequence check failed.")
    lock2 = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    config2 = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if lock2.get("version") != TARGET_VERSION or config2.get("project_version") != TARGET_VERSION:
        raise RuntimeError("Post-write version synchronization failed.")
    for path in (BASE_PATH, BDE_PATH):
        if lock2["data_sha256"].get(path.name) != sha256_file(path):
            raise RuntimeError(f"Post-write hash freeze failed: {path.name}")

    print(f"[PASS] release version synchronized: {TARGET_VERSION}")
    print(f"[PASS] {BASE_PATH.name} -> UTF-8-SIG + Record_ID DOPO_EP_0001..DOPO_EP_0599")
    print(f"[PASS] {BDE_PATH.name} -> UTF-8-SIG + Record_ID DOPO_EP_0001..DOPO_EP_0599")
    print("[PASS] PROJECT_RULES_LOCK.json SHA256 values re-frozen to the migrated bytes")
    print("[NEXT] python -u run.py validate")


if __name__ == "__main__":
    main()
