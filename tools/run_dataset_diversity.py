# -*- coding: utf-8 -*-
"""Generate molecule/scaffold diversity summaries without changing model training."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.dataset_diversity import analyse_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_data = ROOT / "data" / "DOPO_EP_new_with_BDE.csv"
    if not default_data.exists():
        default_data = ROOT / "data" / "DOPO_EP_new.csv"
    parser.add_argument("--input", type=Path, default=default_data)
    parser.add_argument(
        "--tasks",
        default="ALL,LOI,PHRR,THR,UL94_V0,Tg,Char_yield,TS_MPa,FS_MPa,Delta_LOI,Delta_PHRR,Delta_THR,Delta_CY",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "results" / "dataset_diversity",
    )
    args = parser.parse_args()
    tasks = [value.strip() for value in args.tasks.split(",") if value.strip()]
    _, summary, _ = analyse_dataset(args.input, args.results, tasks)
    print(summary.to_string(index=False))
    print(f"[DONE] Results saved to: {args.results}")


if __name__ == "__main__":
    main()
