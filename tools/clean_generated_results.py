# -*- coding: utf-8 -*-
"""Clean generated V4 result directories.

Dry-run by default.

The safe default scope is ``paper``: only regenerated paper-output directories
are targeted. Use ``--scope all --execute`` only when a complete reproducible
rerun from the frozen data and candidate source files is intentionally planned.
The script never touches data/, source code, configuration, or candidate source
tables under 06_ReverseDesign/.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

PAPER_DIRS = (
    "09_PaperFigures",
    "09_PaperTables",
    "09_MainTextTables",
    "09_PaperFigures_TEST",
    "09_PaperTables_TEST",
    "09_MainTextTables_TEST",
)


def targets_for_scope(scope: str) -> list[Path]:
    if not RESULTS.exists():
        return []
    if scope == "paper":
        return [RESULTS / name for name in PAPER_DIRS if (RESULTS / name).exists()]
    if scope == "all":
        return [p for p in RESULTS.iterdir() if p.name != "README.md"]
    raise ValueError(f"Unsupported scope={scope}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=["paper", "all"],
        default="paper",
        help="paper = only 09_* paper outputs (default); all = every generated result except results/README.md",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    targets = targets_for_scope(args.scope)
    action = "DELETE" if args.execute else "DRY-RUN"

    if args.scope == "all":
        print(
            "[WARN] scope=all removes all generated model/SHAP/AD/screening results. "
            "Use only when a full rerun is intended."
        )

    for path in sorted(targets):
        print(f"[{action}] {path.relative_to(ROOT)}")
        if args.execute:
            shutil.rmtree(path) if path.is_dir() else path.unlink()

    print(f"[SUMMARY] scope={args.scope} targets={len(targets)}")
    if not args.execute:
        print("[INFO] Nothing deleted. Add --execute to perform the cleanup.")


if __name__ == "__main__":
    main()
