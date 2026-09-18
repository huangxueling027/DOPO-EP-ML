# -*- coding: utf-8 -*-
"""Record Python and key package versions for reproducibility."""
from __future__ import annotations

import importlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = [
    "numpy", "pandas", "scipy", "sklearn", "xgboost", "lightgbm",
    "rdkit", "shap", "matplotlib", "joblib",
]


def version_of(name: str) -> str:
    try:
        module = importlib.import_module(name)
        return str(getattr(module, "__version__", "installed; version attribute unavailable"))
    except Exception as exc:
        return f"not available: {type(exc).__name__}: {exc}"


def main() -> None:
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {name: version_of(name) for name in PACKAGES},
    }
    outdir = ROOT / "results" / "environment"
    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / "environment_versions.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\n[SAVED] {output}")


if __name__ == "__main__":
    main()
