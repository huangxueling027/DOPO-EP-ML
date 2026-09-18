# -*- coding: utf-8 -*-
"""Backward-compatible wrapper for all main and supplementary paper figures."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", default="results")
    p.add_argument("--output", default="results/09_PaperFigures")
    args = p.parse_args()
    for script, subfolder in [("make_main_figures.py", "main"), ("make_supplementary_figures.py", "supplementary")]:
        command = [sys.executable, "-u", str(HERE / script), "--results-root", args.results_root, "--output", str(Path(args.output) / subfolder)]
        print("[RUN]", " ".join(command))
        subprocess.run(command, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
