# -*- coding: utf-8 -*-
"""Audit or convert the two locked project CSV files to UTF-8-SIG.

The default mode is CHECK ONLY and never writes data.  The corrected V5 release requires UTF-8-SIG for its two frozen project CSVs.

Use ``--execute`` only before a new data freeze if conversion is genuinely
needed. Backups are written outside data/ under results/audit/encoding_backups
to avoid leaving ambiguous duplicate datasets beside the frozen files.
After any real conversion, PROJECT_RULES_LOCK.json hashes must be deliberately
reviewed and re-frozen before model results are considered valid.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILES = [
    ROOT / "data" / "DOPO_EP_new.csv",
    ROOT / "data" / "DOPO_EP_new_with_BDE.csv",
]
ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")


def detect_and_read(path: Path) -> tuple[pd.DataFrame, str]:
    errors: list[str] = []
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc), enc
        except UnicodeDecodeError as exc:
            errors.append(f"{enc}: {exc}")
    raise UnicodeError(f"Unable to decode {path}\n" + "\n".join(errors))


def has_utf8_bom(path: Path) -> bool:
    with path.open("rb") as fh:
        return fh.read(3) == b"\xef\xbb\xbf"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Actually convert non-UTF-8-SIG files.")
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=ROOT / "results" / "audit" / "encoding_backups",
    )
    args = parser.parse_args()

    changed = 0
    for path in DEFAULT_FILES:
        print("=" * 80)
        print(f"[FILE] {path.relative_to(ROOT)}")
        if not path.exists():
            raise FileNotFoundError(path)

        df, detected = detect_and_read(path)
        bom = has_utf8_bom(path)
        print(f"[INFO] detected_read_encoding={detected}")
        print(f"[INFO] utf8_bom={bom}")
        print(f"[INFO] shape={df.shape}")

        if bom:
            print("[PASS] already UTF-8-SIG; no write needed.")
            continue

        if not args.execute:
            print("[CHECK] conversion would be required; no file was changed.")
            continue

        args.backup_dir.mkdir(parents=True, exist_ok=True)
        backup = args.backup_dir / path.name
        backup.write_bytes(path.read_bytes())
        df.to_csv(path, index=False, encoding="utf-8-sig")
        check = pd.read_csv(path, encoding="utf-8-sig")
        if check.shape != df.shape or not has_utf8_bom(path):
            raise RuntimeError(f"Conversion verification failed: {path}")
        changed += 1
        print(f"[BACKUP] {backup}")
        print("[SUCCESS] converted to UTF-8-SIG")

    print(f"\n[SUMMARY] changed_files={changed}")
    if changed:
        print(
            "[IMPORTANT] Frozen data bytes changed. Re-run validation, audit the data, "
            "and deliberately update PROJECT_RULES_LOCK.json before using model results."
        )


if __name__ == "__main__":
    main()
