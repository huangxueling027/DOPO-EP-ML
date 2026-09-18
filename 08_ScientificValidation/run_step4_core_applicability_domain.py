#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-command runner for STEP 4 core applicability-domain validation."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "08_ScientificValidation" / "evaluate_applicability_domain.py"
    command = [
        sys.executable,
        "-u",
        str(script),
        "--results-root",
        str(project_root / "results" / "05_Shap" / "FINAL_core_fixed_baseline_inclusive_5x5"),
        "--output",
        str(project_root / "results" / "06_ApplicabilityDomain" / "FINAL_fixed_5x5"),
        "--tasks",
        "LOI,PHRR,THR,UL94_V0",
        "--radius",
        "3",
        "--nbits",
        "2048",
        "--reliable-threshold",
        "0.70",
        "--caution-threshold",
        "0.50",
        "--bootstrap-iterations",
        "1000",
        "--formats",
        "png,pdf",
        "--dpi",
        "600",
    ]
    print("=" * 96)
    print("[RUN] " + " ".join(command))
    print("=" * 96)
    return subprocess.call(command, cwd=project_root)


if __name__ == "__main__":
    raise SystemExit(main())
