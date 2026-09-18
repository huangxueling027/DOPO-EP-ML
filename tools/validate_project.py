# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import ast
import json
import shutil
import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config" / "task_config.json").read_text(encoding="utf-8"))

def sha256_file(path: Path) -> str:
    """Calculate SHA256 of a file."""
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def validate_locked_data_hashes(project_root: Path) -> None:
    """
    Validate frozen data files against PROJECT_RULES_LOCK.json.
    Raise RuntimeError if any locked file is missing or modified.
    """
    lock_path = project_root / "PROJECT_RULES_LOCK.json"

    if not lock_path.exists():
        raise RuntimeError(
            f"[HASH FAIL] Missing lock file: {lock_path}"
        )

    with open(lock_path, "r", encoding="utf-8") as f:
        lock = json.load(f)

    expected_hashes = lock.get("data_sha256", {})

    if not expected_hashes:
        raise RuntimeError(
            "[HASH FAIL] PROJECT_RULES_LOCK.json does not contain data_sha256."
        )

    failures = []

    print("=" * 78)
    print("[DATA HASH VALIDATION]")
    print("=" * 78)

    for filename, expected in expected_hashes.items():

        path = project_root / "data" / filename

        if not path.exists():
            print(f"[HASH FAIL] {filename}: file missing")
            failures.append(filename)
            continue

        actual = sha256_file(path)

        if actual == expected:
            print(f"[HASH PASS] {filename}")
        else:
            print(f"[HASH FAIL] {filename}")
            print(f"  expected: {expected}")
            print(f"  actual:   {actual}")
            failures.append(filename)

    if failures:
        raise RuntimeError(
            "Frozen data hash validation failed for: "
            + ", ".join(failures)
        )

    print("[HASH PASS] All locked data files are unchanged.")
def read_csv_flexible(path: Path) -> pd.DataFrame:
    """Read project CSV files with the encodings used in this project."""
    errors: list[str] = []

    for encoding in ("utf-8-sig", "gb18030", "utf-8"):
        try:
            df = pd.read_csv(path, encoding=encoding)
            print(f"[INFO] File loaded with encoding: {encoding}")
            return df
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")

    raise UnicodeError(
        f"Unable to decode CSV: {path}\n" + "\n".join(errors)
    )


def validate_dataset_semantics(primary: pd.DataFrame, fallback: pd.DataFrame) -> list[str]:
    """Validate audited V5 data invariants beyond byte-level hashes."""
    errors: list[str] = []

    if len(primary) != 599 or len(fallback) != 599:
        errors.append(f"expected 599 rows in both frozen datasets; got {len(primary)} and {len(fallback)}")

    # The BDE-enriched primary file must contain every base-data column unchanged.
    missing_from_primary = [c for c in fallback.columns if c not in primary.columns]
    if missing_from_primary:
        errors.append("primary BDE file is missing fallback columns: " + ", ".join(missing_from_primary))
    elif "Record_ID" in primary.columns and "Record_ID" in fallback.columns:
        left = fallback.set_index("Record_ID")
        right = primary.set_index("Record_ID")
        if set(left.index) != set(right.index):
            errors.append("primary/fallback Record_ID sets differ")
        else:
            right = right.reindex(left.index)
            for col in left.columns:
                a = left[col]
                b = right[col]
                same = a.eq(b) | (a.isna() & b.isna())
                if not bool(same.all()):
                    errors.append(f"primary/fallback shared-column mismatch: {col}")
                    break

    if "Reference" not in primary.columns:
        errors.append("primary BDE file is missing Reference; reference-group validation would be invalid")
    elif primary["Reference"].isna().any() or primary["Reference"].astype(str).str.strip().eq("").any():
        errors.append("primary BDE file contains missing/blank Reference values")

    stray = [c for c in primary.columns if c in {"Delta_FS.1", "Delta_FS.2"}]
    stray += [c for c in fallback.columns if c in {"Delta_FS.1", "Delta_FS.2"}]
    if stray:
        errors.append("obsolete duplicate Delta_FS columns remain: " + ", ".join(sorted(set(stray))))

    expected_prep = {
        "DOPO-based (additive)": 0,
        "DOPO-based (reactive)": 1,
        "DOPO-based (Co-curing)": 2,
        "DOPO-based (Additive + Secondary Crosslinking)": 3,
    }
    for name, frame in (("primary", primary), ("fallback", fallback)):
        if "Preparation_Method" not in frame.columns or "Preparation_Method_num" not in frame.columns:
            errors.append(f"{name}: missing preparation-method columns")
            continue
        observed = set(frame["Preparation_Method"].dropna().astype(str))
        if observed != set(expected_prep):
            errors.append(f"{name}: preparation labels are not the four canonical categories: {sorted(observed)}")
        for label, code in expected_prep.items():
            vals = pd.to_numeric(frame.loc[frame["Preparation_Method"].eq(label), "Preparation_Method_num"], errors="coerce")
            if len(vals) and (vals.isna().any() or not vals.eq(code).all()):
                errors.append(f"{name}: Preparation_Method_num mismatch for {label}")

    delta_pairs = {
        "Delta_LOI": ("LOI", "EP_matrix_LOI"),
        "Delta_PHRR": ("PHRR_kw_㎡", "EP_matrix_PHRR"),
        "Delta_THR": ("THR_MJ_㎡", "EP_matrix_THR"),
        "Delta_Tg": ("Tg_℃", "EP_matrix_Tg"),
        "Delta_CY": ("Char_yield_％_700C", "EP_matrix_CY"),
        "Delta_TS": ("TS_MPa", "EP_matrix_TS"),
        "Delta_FS": ("FS_MPa", "EP_matrix_FS"),
    }
    for delta, (target, baseline) in delta_pairs.items():
        if not all(c in fallback.columns for c in (delta, target, baseline)):
            continue
        y = pd.to_numeric(fallback[target], errors="coerce")
        b = pd.to_numeric(fallback[baseline], errors="coerce")
        d = pd.to_numeric(fallback[delta], errors="coerce")
        expected = y - b
        comparable = y.notna() & b.notna()
        bad = comparable & (d.isna() | (d - expected).abs().gt(1e-6))
        if bad.any():
            errors.append(f"{delta}: {int(bad.sum())} rows violate Delta = modified - neat EP")
        extra = d.notna() & ~comparable
        if extra.any():
            errors.append(f"{delta}: {int(extra.sum())} values exist without both target and matched baseline")

    if not errors:
        print("[DATA SEMANTICS PASS] Record_ID/reference/preparation/Delta/base-vs-BDE invariants verified")
    return errors

def clean_python_cache() -> list[str]:
    """Remove Python-generated cache files without treating them as historical results."""
    removed: list[str] = []

    for cache_dir in sorted(ROOT.rglob("__pycache__"), reverse=True):
        if cache_dir.is_dir():
            for item in cache_dir.rglob("*"):
                if item.is_file():
                    removed.append(item.relative_to(ROOT).as_posix())
            shutil.rmtree(cache_dir, ignore_errors=True)

    for pyc in ROOT.rglob("*.pyc"):
        if pyc.is_file():
            removed.append(pyc.relative_to(ROOT).as_posix())
            pyc.unlink(missing_ok=True)

    return sorted(set(removed))


def check_python_syntax() -> list[str]:
    """Parse all Python files with ast; this checks syntax without creating .pyc files."""
    errors: list[str] = []
    for path in sorted(ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            source = path.read_text(encoding="utf-8-sig")
            ast.parse(source, filename=str(path))
        except UnicodeDecodeError:
            try:
                source = path.read_text(encoding="gb18030")
                ast.parse(source, filename=str(path))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"syntax/read error: {path.relative_to(ROOT).as_posix()}: {exc}")
        except SyntaxError as exc:
            errors.append(
                f"syntax error: {path.relative_to(ROOT).as_posix()}:{exc.lineno}: {exc.msg}"
            )
    return errors



def validate_final_protocol(config: dict, lock: dict) -> list[str]:
    """Validate the frozen FINAL modeling protocol without touching data hashes."""
    errors: list[str] = []
    formal = config.get("evaluation", {}).get("formal_protocol", {})
    expected = {
        "id": "FINAL_FIXED_BASELINE_INCLUSIVE_5X5",
        "selection_scope": "fixed",
        "row_policy": "baseline_inclusive",
        "split_strategy": "molecule",
        "screening_mode": "formulation",
        "feature_scope": "all",
        "bde_mode": "without",
        "outer_splits": 5,
        "inner_splits": 5,
    }
    for key, value in expected.items():
        if formal.get(key) != value:
            errors.append(
                f"FINAL protocol mismatch in task_config: {key}={formal.get(key)!r}, expected={value!r}"
            )

    expected_tasks = {
        "LOI": ("compact", 260),
        "PHRR": ("compact", 330),
        "THR": ("descriptors", 30),
        "UL94_V0": ("morgan_r3", "all"),
        "Tg": ("morgan_r3", 270),
        "TS_MPa": ("compact", 140),
        "Char_yield": ("full_interaction", 100),
        "FS_MPa": ("morgan_r3", 240),
    }
    tasks = config.get("tasks", {})
    for task, (view, kval) in expected_tasks.items():
        item = tasks.get(task, {})
        got_view = item.get("view")
        got_k = item.get("k")
        if str(got_view) != str(view):
            errors.append(f"FINAL task config mismatch: {task} view={got_view!r}, expected={view!r}")
        if kval == "all":
            if str(got_k).strip().lower() != "all":
                errors.append(f"FINAL task config mismatch: {task} k={got_k!r}, expected='all'")
        elif got_k != kval:
            errors.append(f"FINAL task config mismatch: {task} k={got_k!r}, expected={kval}")

    lock_protocol = lock.get("rules", {}).get("formal_protocol_id")
    if lock_protocol not in (None, "", expected["id"]):
        errors.append(
            f"PROJECT_RULES_LOCK formal_protocol_id={lock_protocol!r}, expected={expected['id']!r}"
        )
    if not errors:
        print("[FINAL PROTOCOL PASS] fixed + baseline_inclusive + molecule + formulation + without-BDE + 5x5")
        if not lock_protocol:
            print("[WARN] PROJECT_RULES_LOCK has no formal_protocol_id; run tools/sync_final_protocol_lock.py once.")
    return errors

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Check all Python files with AST parsing; no __pycache__ is created.",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Do not automatically remove existing __pycache__ and .pyc files.",
    )
    args = parser.parse_args()

    errors: list[str] = []
    # ============================================================
    # Frozen data SHA256 validation
    # ============================================================
    try:
        validate_locked_data_hashes(ROOT)
    except Exception as exc:
        errors.append(str(exc))

    # Configuration and frozen-data metadata must refer to the same release.
    lock_path = ROOT / "PROJECT_RULES_LOCK.json"
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        lock_version = str(lock.get("version", "")).strip()
        config_version = str(CONFIG.get("project_version", "")).strip()
        if lock_version != config_version:
            errors.append(
                f"project version mismatch: PROJECT_RULES_LOCK={lock_version!r}, "
                f"task_config={config_version!r}"
            )

        locked_names = set(lock.get("data_sha256", {}))
        configured_names = {
            Path(CONFIG["data"]["primary"]).name,
            Path(CONFIG["data"]["fallback"]).name,
        }
        if not configured_names.issubset(locked_names):
            errors.append(
                "configured primary/fallback data are not both protected by data_sha256"
            )
    except Exception as exc:
        errors.append(f"cannot validate project/config version consistency: {exc}")

    # FINAL modeling protocol is separate from the frozen data-release version.
    try:
        if lock_path.exists():
            lock_for_protocol = json.loads(lock_path.read_text(encoding="utf-8"))
            errors.extend(validate_final_protocol(CONFIG, lock_for_protocol))
    except Exception as exc:
        errors.append(f"cannot validate FINAL protocol metadata: {exc}")

    # Conversion backups should be archived outside the final data/ directory.
    data_backups = sorted((ROOT / "data").glob("*_before_encoding_conversion.csv"))
    if data_backups:
        errors.append(
            "ambiguous data backups remain in data/: "
            + ", ".join(p.name for p in data_backups)
            + ". Archive them outside the final project."
        )

    # The frozen project CSVs are required to retain UTF-8-SIG encoding.
    for rel in (CONFIG["data"]["primary"], CONFIG["data"]["fallback"]):
        path = ROOT / rel
        if path.exists():
            with path.open("rb") as fh:
                if fh.read(3) != b"\xef\xbb\xbf":
                    errors.append(f"locked CSV is not UTF-8-SIG: {rel}")

    # Stable row IDs are required for pooled outer-prediction integrity auditing.
    for rel in (CONFIG["data"]["primary"], CONFIG["data"]["fallback"]):
        path = ROOT / rel
        if not path.exists():
            continue
        try:
            frame = read_csv_flexible(path)
            if "Record_ID" not in frame.columns:
                errors.append(
                    f"missing Record_ID in {rel}; run tools/add_record_id.py only if the frozen lock/hash is intentionally updated"
                )
            else:
                ids = frame["Record_ID"]
                if ids.isna().any():
                    errors.append(f"Record_ID contains missing values: {rel}")
                if ids.astype(str).duplicated().any():
                    errors.append(f"Record_ID contains duplicates: {rel}")
                if len(frame) == 599:
                    expected = pd.Series([f"DOPO_EP_{i:04d}" for i in range(1, 600)])
                    if not ids.reset_index(drop=True).astype(str).equals(expected):
                        errors.append(f"Record_ID sequence is not DOPO_EP_0001..DOPO_EP_0599: {rel}")
        except Exception as exc:
            errors.append(f"cannot validate Record_ID in {rel}: {exc}")

    if not args.keep_cache:
        removed = clean_python_cache()
        if removed:
            print(f"[CLEAN] removed Python cache files: {len(removed)}")

    required = [
        "common/pipeline_core.py",
        "common/task_runner.py",
        "run_tasks.py",
        CONFIG["data"]["primary"],
    ]
    for rel in required:
        if not (ROOT / rel).exists():
            errors.append(f"missing: {rel}")

    # Results are allowed in the cleaned package. Warn only when active result
    # folders contain deterministic feature matrices that should be regenerated.
    active_result_roots = [
        ROOT / "results" / "main",
        ROOT / "results" / "bde_ablation",
        ROOT / "results" / "scientific_validation",
        ROOT / "results" / "05_Shap",
        ROOT / "results" / "06_ApplicabilityDomain",
    ]
    redundant_outputs: list[str] = []
    for result_root in active_result_roots:
        if result_root.exists():
            redundant_outputs.extend(
                path.relative_to(ROOT).as_posix()
                for path in result_root.rglob("feature_matrix_*.csv")
            )
    if redundant_outputs:
        print(f"[WARN] active results contain reproducible feature matrices: {len(redundant_outputs)}")

    data_path = ROOT / CONFIG["data"]["primary"]
    try:
        df = read_csv_flexible(data_path)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"cannot read primary data: {exc}")
        df = None

    # Cross-check the BDE-enriched primary dataset against the base fallback and
    # verify scientific data invariants that hashes alone cannot detect.
    if df is not None:
        try:
            fallback_df = read_csv_flexible(ROOT / CONFIG["data"]["fallback"])
            errors.extend(validate_dataset_semantics(df, fallback_df))
        except Exception as exc:
            errors.append(f"cannot validate dataset semantics: {exc}")

    loading_col = "Loading_total_FR wt%"
    if df is not None:
        if loading_col not in df.columns:
            errors.append(f"missing data column: {loading_col}")
        else:
            loading = pd.to_numeric(df[loading_col], errors="coerce")
            print(
                f"[DATA] rows={len(df)}, "
                f"explicit loading=0 rows={int(loading.eq(0).sum())}"
            )

    tasks = CONFIG["tasks"]
    for task in ("Tg", "TS_MPa", "FS_MPa"):
        if tasks[task]["group"] != "MAIN_CO_CURING":
            errors.append(f"wrong group rule for {task}")
    for task, cfg in tasks.items():
        if task not in {"Tg", "TS_MPa", "FS_MPa"} and cfg["group"] != "MAIN_CO":
            errors.append(f"wrong group rule for {task}")

    if args.compile:
        syntax_errors = check_python_syntax()
        errors.extend(syntax_errors)
        if not syntax_errors:
            print("[SYNTAX] all Python files parsed successfully")

    if errors:
        print("[FAIL]")
        for err in errors:
            print(" -", err)
        raise SystemExit(1)

    print("[PASS] Clean project structure and rules are valid.")


if __name__ == "__main__":
    main()
