# -*- coding: utf-8 -*-
"""Generate the audit, main figures, supplementary figures, and paper tables."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def call(script: str, *args: str) -> None:
    command = [sys.executable, "-u", str(HERE / script), *args]
    print("\n[RUN]", " ".join(command))
    subprocess.run(command, cwd=str(ROOT), check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", default="results")
    p.add_argument("--output-root", default="results/09_PaperFigures")
    p.add_argument("--tables-output", default="results/09_PaperTables")
    p.add_argument("--main-tables-output", default="results/09_MainTextTables")
    p.add_argument("--skip-audit", action="store_true")
    p.add_argument("--skip-main", action="store_true")
    p.add_argument("--skip-supplementary", action="store_true")
    p.add_argument("--skip-tables", action="store_true")
    args = p.parse_args()

    if not args.skip_audit:
        call("audit_paper_inputs.py", "--results-root", args.results_root, "--output", str(Path(args.output_root) / "audit"))
    if not args.skip_main:
        call("make_main_figures.py", "--results-root", args.results_root, "--output", str(Path(args.output_root) / "main"))
    if not args.skip_supplementary:
        call("make_supplementary_figures.py", "--results-root", args.results_root, "--output", str(Path(args.output_root) / "supplementary"))
    if not args.skip_tables:
        call("build_paper_tables.py", "--results-root", args.results_root, "--output", args.tables_output)
        call("build_main_text_tables.py", "--tables-root", args.tables_output, "--output", args.main_tables_output)
    print("\n[DONE] Paper output workflow completed. Check figure/table manifests before manuscript assembly.")


if __name__ == "__main__":
    main()
