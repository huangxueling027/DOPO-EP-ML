# -*- coding: utf-8 -*-
"""Rank predicted candidates by performance, uncertainty, applicability domain and loading.

Example objective syntax:
    --objectives "LOI_pred:max:1,V0_probability:max:1,PHRR_pred:min:1,THR_pred:min:1"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.pareto_ranking import Objective, rank_candidates
from common.pipeline_core import read_csv_auto


def parse_objectives(text: str) -> list[Objective]:
    objectives = []
    for item in text.split(","):
        parts = [part.strip() for part in item.split(":")]
        if len(parts) not in {2, 3}:
            raise ValueError(f"Invalid objective: {item}")
        objectives.append(Objective(parts[0], parts[1], float(parts[2]) if len(parts) == 3 else 1.0))
    return objectives


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--objectives",
        default="LOI_pred:max:1,V0_probability:max:1,PHRR_pred:min:1,THR_pred:min:1,Predicted_Tg:max:0.5,Predicted_TS:max:0.5,Predicted_FS:max:0.5",
    )
    parser.add_argument("--uncertainty-columns", default="LOI_PI_width,PHRR_PI_width,THR_PI_width,Tg_PI_width,TS_MPa_PI_width,UL94_uncertainty")
    parser.add_argument("--loading-column", default="Loading_total_FR wt%")
    parser.add_argument("--loading-reference", type=float, default=10.0)
    parser.add_argument("--uncertainty-penalty", type=float, default=0.15)
    parser.add_argument("--ad-penalty", type=float, default=0.20)
    parser.add_argument("--loading-penalty", type=float, default=0.10)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "06_ReverseDesign" / "candidate_ranked_pareto.csv")
    args = parser.parse_args()

    frame = read_csv_auto(str(args.input))
    result = rank_candidates(
        frame,
        parse_objectives(args.objectives),
        uncertainty_columns=[item.strip() for item in args.uncertainty_columns.split(",") if item.strip()],
        loading_column=args.loading_column,
        loading_reference=args.loading_reference,
        uncertainty_penalty_weight=args.uncertainty_penalty,
        ad_penalty_weight=args.ad_penalty,
        loading_penalty_weight=args.loading_penalty,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(result.head(30).to_string(index=False))
    print(f"[DONE] Saved: {args.output}")


if __name__ == "__main__":
    main()
