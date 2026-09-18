# -*- coding: utf-8 -*-
"""Audit Model-B baseline-inclusive feature masking without changing source data."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.scientific_evaluation import (
    TASK_CONFIGS,
    MATCHING_EP_BASELINE,
    _apply_baseline_inclusive_mask,
    _candidate_space,
    _filter_task_matrix,
    _is_neat_fr_derived_feature,
    prepare_scientific_bundle,
)
from common.literature_feature_views import get_groups

DATA = ROOT / "data" / "DOPO_EP_new_with_BDE.csv"
CORE_TASKS = ("LOI", "PHRR", "THR", "UL94_V0")


def _numeric_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    a = left.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    b = right.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return bool(np.allclose(a, b, rtol=0.0, atol=0.0, equal_nan=True))


def main() -> None:
    bundle = prepare_scientific_bundle(DATA)
    loading_col = bundle.base.colmap.get("Loading_total_FR wt%", "Loading_total_FR wt%")
    loading = pd.to_numeric(bundle.df[loading_col], errors="coerce")
    neat = loading.eq(0.0).to_numpy()
    modified = loading.gt(0.0).to_numpy()

    print("=" * 88)
    print("[MODEL B | BASELINE-INCLUSIVE AUDIT]")
    print("=" * 88)
    print(f"[DATA] rows={len(bundle.df)} explicit_loading_0={int(neat.sum())} modified={int(modified.sum())}")

    failures: list[str] = []
    if len(bundle.df) != 599:
        failures.append(f"expected 599 rows, got {len(bundle.df)}")
    if int(neat.sum()) != 131:
        failures.append(f"expected 131 zero-loading rows, got {int(neat.sum())}")

    for task in CORE_TASKS:
        config = TASK_CONFIGS[task]
        views, _ = _candidate_space(config, bundle, "curated")
        for view in views:
            raw = _filter_task_matrix(
                bundle, task, view, use_bde=False,
                screening_mode="formulation", feature_scope="all",
            ).reset_index(drop=True)
            masked = _apply_baseline_inclusive_mask(
                raw, bundle.df.reset_index(drop=True), bundle.base.colmap,
                task, "baseline_inclusive",
            )

            fr_cols = [c for c in raw.columns if _is_neat_fr_derived_feature(c)]
            if fr_cols:
                vals = masked.loc[neat, fr_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
                nonzero = int(np.count_nonzero(np.abs(vals) > 1e-12))
                if nonzero:
                    failures.append(f"{task}/{view}: {nonzero} nonzero neat-EP FR-derived cells")

            baseline = MATCHING_EP_BASELINE.get(task)
            if baseline:
                baseline_cols = [c for c in masked.columns if baseline.lower() in str(c).lower()]
                if baseline_cols:
                    remaining = int(masked.loc[neat, baseline_cols].notna().sum().sum())
                    if remaining:
                        failures.append(f"{task}/{view}: {remaining} neat matching-baseline cells remain")

            # Model B must not alter any existing feature on modified formulations.
            common = list(raw.columns)
            if not _numeric_equal(raw.loc[modified, common], masked.loc[modified, common]):
                failures.append(f"{task}/{view}: modified rows changed by Model-B mask")

        groups = get_groups(bundle.base, task).reset_index(drop=True)
        neat_group_count = int(groups.loc[neat].nunique())
        print(f"[AUDIT] task={task} views={len(views)} neat_groups={neat_group_count} total_groups={groups.nunique()}")
        if neat_group_count <= 1:
            failures.append(f"{task}: neat rows collapsed to one molecule group")

    # Check whether physically identical neat-EP baselines are duplicated.  This
    # is an audit only: no row is automatically deleted.
    signature_keys = [
        "Reference", "Curing_Agent", "Cure_Temp_Max",
        "LOI_Thickness_mm", "UL94_Thickness_mm", "Cone_Thickness_mm",
        "Cone_flux_kW_m2", "LOI", "UL94", "PHRR", "THR", "Tg",
        "Char_yield", "TS_MPa", "FS_MPa",
    ]
    signature_cols = []
    for key in signature_keys:
        actual = bundle.base.colmap.get(key, key)
        if actual in bundle.df.columns and actual not in signature_cols:
            signature_cols.append(actual)
    neat_frame = bundle.df.loc[neat, signature_cols].copy()
    duplicate_signature_rows = int(neat_frame.duplicated(keep=False).sum())
    print(f"[AUDIT] duplicate_neat_baseline_signature_rows={duplicate_signature_rows}")
    if duplicate_signature_rows:
        print("[WARN] Duplicate physical baseline signatures detected; review before formal Model-B interpretation.")

    if failures:
        print("\n[FAIL]")
        for item in failures:
            print(" -", item)
        raise SystemExit(1)

    print("\n[PASS] Neat-EP FR-derived feature masking: OK")
    print("[PASS] Matching EP baseline leakage guard: OK")
    print("[PASS] Modified formulations unchanged by Model-B mask: OK")
    print("[PASS] Molecule groups remain based on original identities: OK")
    print("[PASS] Source CSV files were read only.")


if __name__ == "__main__":
    main()
