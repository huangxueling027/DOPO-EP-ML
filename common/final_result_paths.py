# -*- coding: utf-8 -*-
"""Single source of truth for the frozen V5 FINAL model/result paths.

This helper intentionally refuses to fall back to curated, modified-only,
smoke, V4, or mtime-selected result folders.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "config" / "final_model_manifest.json"

FINAL_TASK_CONFIGS = {
    "LOI": ("compact", 260, "core"),
    "PHRR": ("compact", 330, "core"),
    "THR": ("descriptors", 30, "core"),
    "UL94_V0": ("morgan_r3", None, "core"),
    "Tg": ("morgan_r3", 270, "auxiliary"),
    "TS_MPa": ("compact", 140, "auxiliary"),
    "Char_yield": ("full_interaction", 100, "exploratory"),
    "FS_MPa": ("morgan_r3", 240, "exploratory"),
}


def load_final_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Missing FINAL manifest: {MANIFEST_PATH}. "
            "Do not silently search old results."
        )
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8-sig"))
    expected = {
        "selection_scope": "fixed",
        "row_policy": "baseline_inclusive",
        "split_strategy": "molecule",
        "screening_mode": "formulation",
        "use_BDE": False,
    }
    bad = {k: (data.get(k), v) for k, v in expected.items() if data.get(k) != v}
    if bad:
        raise RuntimeError(f"FINAL manifest protocol mismatch: {bad}")
    return data


def formal_run_dir(task: str) -> Path:
    if task not in FINAL_TASK_CONFIGS:
        raise KeyError(f"Task is not in the frozen FINAL task set: {task}")
    _, _, tier = FINAL_TASK_CONFIGS[task]
    manifest = load_final_manifest()
    rel = manifest["formal_results"][tier]
    return ROOT / rel


def final_bundle_path(task: str) -> Path:
    run_dir = formal_run_dir(task)
    path = (
        run_dir
        / task
        / "molecule"
        / "formulation"
        / "without_BDE"
        / f"{task}_final_model_bundle.joblib"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Frozen FINAL bundle missing for {task}: {path}. "
            "Do not fall back to curated/V4/smoke results."
        )
    return path


def validate_final_bundle(task: str, bundle: dict[str, Any], path: Path | None = None) -> None:
    expected_view, expected_k, _ = FINAL_TASK_CONFIGS[task]
    problems: list[str] = []
    if str(bundle.get("task", task)) != task:
        problems.append(f"task={bundle.get('task')!r}")
    if str(bundle.get("view")) != expected_view:
        problems.append(f"view={bundle.get('view')!r}, expected={expected_view!r}")

    got_k = bundle.get("requested_k")
    if expected_k is None:
        if got_k is not None and str(got_k).strip().upper() != "ALL":
            problems.append(f"K={got_k!r}, expected=ALL")
    else:
        try:
            if int(float(got_k)) != int(expected_k):
                problems.append(f"K={got_k!r}, expected={expected_k}")
        except Exception:
            problems.append(f"K={got_k!r}, expected={expected_k}")

    checks = {
        "split_strategy": "molecule",
        "screening_mode": "formulation",
        "feature_scope": "all",
        "row_policy": "baseline_inclusive",
    }
    for key, value in checks.items():
        got = bundle.get(key, value if key == "feature_scope" else None)
        if got != value:
            problems.append(f"{key}={got!r}, expected={value!r}")

    if bool(bundle.get("use_BDE", False)):
        problems.append("use_BDE=True, expected without-BDE")

    # Older already-completed FINAL bundles may not store selection_scope.
    # If present, it must be fixed; if absent, exact task view/K above is authoritative.
    if bundle.get("selection_scope") not in (None, "", "fixed"):
        problems.append(
            f"selection_scope={bundle.get('selection_scope')!r}, expected='fixed' or legacy-missing"
        )

    if problems:
        where = f" ({path})" if path else ""
        raise RuntimeError(
            f"{task} bundle is not the frozen FINAL configuration{where}: "
            + "; ".join(problems)
        )


def load_final_bundle(task: str) -> tuple[dict[str, Any], Path]:
    path = final_bundle_path(task)
    bundle = joblib.load(path)
    validate_final_bundle(task, bundle, path)
    return bundle, path
