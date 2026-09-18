# -*- coding: utf-8 -*-
"""Synchronize FINAL modeling-protocol metadata into PROJECT_RULES_LOCK.json.

This tool does NOT modify any CSV, Record_ID, or data_sha256 value.  It only
updates descriptive protocol fields in the root lock file so the frozen data
release and the FINAL modeling protocol are documented consistently.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "PROJECT_RULES_LOCK.json"
CONFIG_PATH = ROOT / "config" / "task_config.json"
BACKUP_ROOT = ROOT / "results" / "audit" / "final_protocol_lock_backup"
PROTOCOL_ID = "FINAL_FIXED_BASELINE_INCLUSIVE_5X5"


def main() -> None:
    if not LOCK_PATH.exists():
        raise FileNotFoundError(LOCK_PATH)
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(CONFIG_PATH)

    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    formal = config.get("evaluation", {}).get("formal_protocol", {})
    if formal.get("id") != PROTOCOL_ID:
        raise RuntimeError(
            f"task_config formal protocol is {formal.get('id')!r}, expected {PROTOCOL_ID!r}."
        )

    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    candidate = BACKUP_ROOT / "PROJECT_RULES_LOCK.before_final_protocol.json"
    idx = 2
    while candidate.exists():
        candidate = BACKUP_ROOT / f"PROJECT_RULES_LOCK.before_final_protocol_{idx}.json"
        idx += 1
    shutil.copy2(LOCK_PATH, candidate)

    # Preserve the frozen data release identifier and hashes exactly.
    original_version = lock.get("version")
    original_hashes = dict(lock.get("data_sha256", {}))

    rules = dict(lock.get("rules", {}))
    rules.update({
        "formal_protocol_id": PROTOCOL_ID,
        "neat_ep_absolute_tasks": (
            "Primary formal absolute-property evaluation is baseline-inclusive. Valid neat EP rows are retained; "
            "on neat EP rows, FR-derived molecular/fingerprint/composition/synergy/interaction features are masked, "
            "and the task-matching EP_matrix feature is masked where present. Overall metrics are primary and "
            "modified-formulation test-subset metrics are sensitivity analysis."
        ),
        "delta_tasks": (
            "Neat EP is a baseline/reference source but is not a Delta target. Explicit Loading_total_FR=0 rows are "
            "excluded from Delta target evaluation; rows with unknown loading are retained only when the Delta target is valid."
        ),
        "strict_validation": (
            "FINAL evaluation uses 5x5 molecule-grouped nested CV with fold-internal preprocessing. Task-specific feature "
            "view/K are predefined from development-stage V4 sensitivity analyses; only the predictive model is selected "
            "by grouped inner CV within each outer-training fold."
        ),
        "formal_selection_scope": "fixed",
        "formal_row_policy": "baseline_inclusive",
        "formal_screening_mode": "formulation",
        "formal_split_strategy": "molecule",
        "formal_bde_mode": "without",
        "formal_outer_splits": 5,
        "formal_inner_splits": 5,
        "paper_output_policy": (
            "Main text uses FINAL fixed-view/K baseline-inclusive 5x5 molecule-grouped nested validation; overall metrics "
            "are primary, modified-formulation subset metrics are sensitivity analysis, and development repeated-group-holdout "
            "or curated/modified-only results must not silently replace FINAL results."
        ),
    })
    lock["rules"] = rules
    lock["results_status"] = (
        "FINAL fixed baseline-inclusive 5x5 core, auxiliary and exploratory evaluations completed. "
        "Downstream robustness, interpretation, screening and external-validation outputs must be synchronized to "
        "FINAL_FIXED_BASELINE_INCLUSIVE_5X5."
    )

    # Explicitly restore immutable release/hash fields.
    lock["version"] = original_version
    lock["data_sha256"] = original_hashes

    LOCK_PATH.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    check = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if check.get("version") != original_version:
        raise RuntimeError("Frozen data-release version changed unexpectedly.")
    if check.get("data_sha256") != original_hashes:
        raise RuntimeError("Frozen data SHA256 values changed unexpectedly.")
    if check.get("rules", {}).get("formal_protocol_id") != PROTOCOL_ID:
        raise RuntimeError("FINAL protocol metadata synchronization failed.")

    print(f"[BACKUP] {candidate.relative_to(ROOT)}")
    print(f"[PASS] PROJECT_RULES_LOCK protocol metadata -> {PROTOCOL_ID}")
    print("[PASS] data release version preserved:", original_version)
    print("[PASS] data_sha256 preserved unchanged")
    print("[NEXT] python -u run.py validate")


if __name__ == "__main__":
    main()
