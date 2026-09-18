# -*- coding: utf-8 -*-
"""Build the final 35/50 kW/m² rank-stable combined candidate priority table."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_50 = ROOT / "results" / "06_ReverseDesign" / "FINAL_fixed" / "combined_flux50" / "ranked_candidates.csv"
DEFAULT_35 = ROOT / "results" / "06_ReverseDesign" / "FINAL_fixed" / "combined_flux35" / "ranked_candidates.csv"
DEFAULT_OUT = ROOT / "results" / "06_ReverseDesign" / "FINAL_fixed" / "final_priority_combined"

IDENTITY = [
    "Candidate_ID", "Candidate_Name", "Source_Type", "Design_Family", "Synthesis_Level"
]
METRICS = [
    "Candidate_rank", "Pareto_rank", "Final_ranking_score",
    "Loading_total_FR wt%", "LOI_pred", "PHRR_pred", "THR_pred",
    "V0_probability", "Tg_pred", "TS_pred", "max_similarity",
]


def _load(path: Path, suffix: str) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = set(IDENTITY + METRICS + ["Pareto_front"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"{path} is missing required columns: {missing}")

    if frame["Candidate_ID"].astype(str).duplicated().any():
        raise RuntimeError(f"Duplicate Candidate_ID in {path}")

    cols = IDENTITY + METRICS + ["Pareto_front"]
    out = frame[cols].copy()
    rename = {c: f"{c}_{suffix}" for c in METRICS + ["Pareto_front"]}
    return out.rename(columns=rename)


def _set_overlap(a: pd.DataFrame, b: pd.DataFrame, k: int) -> tuple[int, float, float]:
    sa = set(a.nsmallest(k, "Candidate_rank_50")["Candidate_ID"])
    sb = set(b.nsmallest(k, "Candidate_rank_35")["Candidate_ID"])
    inter = len(sa & sb)
    frac = inter / float(k) if k else np.nan
    union = len(sa | sb)
    jaccard = inter / float(union) if union else np.nan
    return inter, frac, jaccard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flux50", type=Path, default=DEFAULT_50)
    parser.add_argument("--flux35", type=Path, default=DEFAULT_35)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--final-top-n", type=int, default=20)
    args = parser.parse_args()

    a = _load(args.flux50, "50")
    b = _load(args.flux35, "35")

    # The identity fields must be the same for one candidate in both scenarios.
    merged = a.merge(
        b,
        on="Candidate_ID",
        how="inner",
        suffixes=("_identity50", "_identity35"),
        validate="one_to_one",
    )

    for col in IDENTITY[1:]:
        c50 = f"{col}_identity50"
        c35 = f"{col}_identity35"
        if c50 in merged and c35 in merged:
            mismatch = (
                merged[c50].fillna("").astype(str)
                != merged[c35].fillna("").astype(str)
            )
            if mismatch.any():
                bad = merged.loc[mismatch, "Candidate_ID"].astype(str).tolist()[:10]
                raise RuntimeError(f"{col} differs between 50/35 for: {bad}")
            merged[col] = merged[c50]
            merged = merged.drop(columns=[c50, c35])

    # Put identity fields first.
    ordered = ["Candidate_ID"] + IDENTITY[1:]
    other = [c for c in merged.columns if c not in ordered]
    merged = merged[ordered + other]

    merged["Rank_shift_35_minus_50"] = (
        pd.to_numeric(merged["Candidate_rank_35"], errors="coerce")
        - pd.to_numeric(merged["Candidate_rank_50"], errors="coerce")
    )
    merged["Abs_rank_shift"] = merged["Rank_shift_35_minus_50"].abs()
    merged["Mean_rank"] = merged[["Candidate_rank_50", "Candidate_rank_35"]].mean(axis=1)
    merged["Worst_rank"] = merged[["Candidate_rank_50", "Candidate_rank_35"]].max(axis=1)

    merged["Pareto_both"] = (
        merged["Pareto_front_50"].fillna(False).astype(bool)
        & merged["Pareto_front_35"].fillna(False).astype(bool)
    )
    merged["Top20_both"] = (
        (pd.to_numeric(merged["Candidate_rank_50"], errors="coerce") <= args.final_top_n)
        & (pd.to_numeric(merged["Candidate_rank_35"], errors="coerce") <= args.final_top_n)
    )
    merged["Final_priority"] = merged["Pareto_both"] & merged["Top20_both"]

    # Pareto_front booleans are redundant after the explicit Pareto_both flag
    # and were not part of the historical final stability table.
    merged = merged.drop(columns=["Pareto_front_50", "Pareto_front_35"])

    # Stable ranking statistics.
    r50 = pd.to_numeric(merged["Candidate_rank_50"], errors="coerce")
    r35 = pd.to_numeric(merged["Candidate_rank_35"], errors="coerce")
    rho, _ = spearmanr(r50, r35, nan_policy="omit")

    summary_rows: list[dict[str, float | str]] = []
    for k in (10, 20, 30, 50):
        # top-k overlap is computed from the common candidate population.
        s50 = set(merged.nsmallest(k, "Candidate_rank_50")["Candidate_ID"])
        s35 = set(merged.nsmallest(k, "Candidate_rank_35")["Candidate_ID"])
        inter = len(s50 & s35)
        union = len(s50 | s35)
        summary_rows.extend([
            {"Metric": f"Top{k}_overlap_n", "Value": inter},
            {"Metric": f"Top{k}_overlap_fraction", "Value": inter / float(k)},
            {"Metric": f"Top{k}_Jaccard", "Value": inter / float(union) if union else np.nan},
        ])

    summary_rows.extend([
        {"Metric": "n_flux50_eligible_molecules", "Value": len(a)},
        {"Metric": "n_flux35_eligible_molecules", "Value": len(b)},
        {"Metric": "n_common_eligible_molecules", "Value": len(merged)},
        {"Metric": "spearman_rank_correlation", "Value": float(rho)},
        {"Metric": "median_abs_rank_shift", "Value": float(merged["Abs_rank_shift"].median())},
        {"Metric": "mean_abs_rank_shift", "Value": float(merged["Abs_rank_shift"].mean())},
        {"Metric": "max_abs_rank_shift", "Value": float(merged["Abs_rank_shift"].max())},
        {"Metric": "pareto_flux50", "Value": int(a["Pareto_front_50"].fillna(False).astype(bool).sum())},
        {"Metric": "pareto_flux35", "Value": int(b["Pareto_front_35"].fillna(False).astype(bool).sum())},
        {"Metric": "pareto_intersection", "Value": int(merged["Pareto_both"].sum())},
    ])
    p50 = set(a.loc[a["Pareto_front_50"].fillna(False).astype(bool), "Candidate_ID"])
    p35 = set(b.loc[b["Pareto_front_35"].fillna(False).astype(bool), "Candidate_ID"])
    p_union = len(p50 | p35)
    summary_rows.append({
        "Metric": "pareto_jaccard",
        "Value": len(p50 & p35) / float(p_union) if p_union else np.nan,
    })

    final = merged[merged["Final_priority"]].copy()
    final["Mean_ranking_score"] = final[
        ["Final_ranking_score_50", "Final_ranking_score_35"]
    ].mean(axis=1)
    # Final priority is stability-first: minimize the worst 35/50 rank, then
    # use mean multi-objective score as the tie-breaker.
    final = final.sort_values(
        ["Worst_rank", "Mean_ranking_score", "Mean_rank", "Candidate_ID"],
        ascending=[True, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    final["Final_priority_rank"] = np.arange(1, len(final) + 1)

    summary_rows.append({
        "Metric": "final_priority_candidates",
        "Value": len(final),
    })

    source_order = list(dict.fromkeys(
        merged["Source_Type"].fillna("unknown").astype(str).tolist()
    ))
    source_rows = []
    for source in source_order:
        mask = merged["Source_Type"].fillna("unknown").astype(str).eq(source)
        source_rows.append({
            "Source_Type": source,
            "N_common_eligible": int(mask.sum()),
            "N_pareto_both": int((mask & merged["Pareto_both"]).sum()),
            "N_top20_both": int((mask & merged["Top20_both"]).sum()),
            "N_final_priority": int((mask & merged["Final_priority"]).sum()),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stability_out = merged.sort_values("Candidate_ID", kind="stable").reset_index(drop=True)
    stability_out.to_csv(
        args.output_dir / "flux35_50_rank_stability_combined.csv",
        index=False, encoding="utf-8-sig",
    )
    pd.DataFrame(summary_rows).to_csv(
        args.output_dir / "flux35_50_stability_summary_combined.csv",
        index=False, encoding="utf-8-sig",
    )
    final.to_csv(
        args.output_dir / "final_priority_candidates_combined.csv",
        index=False, encoding="utf-8-sig",
    )
    final.head(10).to_csv(
        args.output_dir / "final_top10_candidates_combined.csv",
        index=False, encoding="utf-8-sig",
    )
    pd.DataFrame(source_rows).to_csv(
        args.output_dir / "candidate_source_comparison.csv",
        index=False, encoding="utf-8-sig",
    )

    print("[OK] Common eligible candidates:", len(merged))
    print("[OK] Pareto in both:", int(merged["Pareto_both"].sum()))
    print("[OK] Top-20 in both:", int(merged["Top20_both"].sum()))
    print("[OK] Final priority candidates:", len(final))
    print("[OK] Spearman rank correlation:", f"{rho:.6f}")
    print("[DONE]", args.output_dir)


if __name__ == "__main__":
    main()
