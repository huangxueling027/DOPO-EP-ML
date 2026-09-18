# -*- coding: utf-8 -*-
"""Summarize the largest outer-test errors and their data-source patterns."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def parse_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "results" / "scientific_validation" / "FINAL_core_fixed_baseline_inclusive_5x5",
    )
    parser.add_argument("--tasks", default="LOI,PHRR,THR,UL94_V0")
    parser.add_argument("--split-strategy", default="molecule")
    parser.add_argument("--screening-mode", default="formulation")
    parser.add_argument("--bde-mode", choices=["without_BDE", "with_BDE"], default="without_BDE")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "scientific_validation" / "outer_error_analysis",
    )
    return parser.parse_args()


def locate_predictions(args: argparse.Namespace, task: str) -> Path:
    direct = (
        args.results_root
        / task
        / args.split_strategy
        / args.screening_mode
        / args.bde_mode
        / f"{task}_outer_predictions.csv"
    )
    if direct.exists():
        return direct
    matches = list(args.results_root.rglob(f"{task}_outer_predictions.csv"))
    matches = [path for path in matches if args.bde_mode in path.parts]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No outer predictions found for {task} under {args.results_root}")
    raise RuntimeError(f"Multiple prediction files found for {task}: {matches}")


def _group_summary(frame: pd.DataFrame, key: str, task_type: str) -> pd.DataFrame:
    if key not in frame.columns:
        return pd.DataFrame()
    if task_type == "regression":
        return (
            frame.groupby(key, dropna=False)
            .agg(
                n=("y_true", "size"),
                MAE=("abs_error", "mean"),
                median_abs_error=("abs_error", "median"),
                max_abs_error=("abs_error", "max"),
                mean_signed_error=("signed_error", "mean"),
            )
            .reset_index()
            .sort_values(["MAE", "n"], ascending=[False, False])
        )
    return (
        frame.groupby(key, dropna=False)
        .agg(
            n=("y_true", "size"),
            accuracy=("correct", "mean"),
            mean_probability=("probability", "mean"),
            mean_log_loss=("sample_log_loss", "mean"),
        )
        .reset_index()
        .sort_values(["accuracy", "n"], ascending=[True, False])
    )


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []

    for task in parse_csv(args.tasks):
        path = locate_predictions(args, task)
        frame = pd.read_csv(path, encoding="utf-8-sig")
        task_dir = args.output / task
        task_dir.mkdir(parents=True, exist_ok=True)
        task_type = "classification" if task == "UL94_V0" else "regression"

        if task_type == "regression":
            frame["signed_error"] = frame["y_pred"] - frame["y_true"]
            frame["abs_error"] = frame["signed_error"].abs()
            if {"PI_lower", "PI_upper"}.issubset(frame.columns):
                frame["PI_width"] = frame["PI_upper"] - frame["PI_lower"]
            top = frame.sort_values("abs_error", ascending=False).head(args.top_n)
        else:
            probability_col = next(
                (c for c in ["calibrated_V0_probability", "raw_V0_probability", "calibrated_probability", "probability", "y_probability", "y_prob"] if c in frame.columns),
                None,
            )
            prediction_col = next((c for c in ["y_pred", "predicted_class"] if c in frame.columns), None)
            if probability_col is None:
                raise KeyError(f"No probability column found in {path}; columns={list(frame.columns)}")
            if prediction_col is None:
                frame["y_pred"] = (frame[probability_col] >= 0.5).astype(int)
                prediction_col = "y_pred"
            p = np.clip(pd.to_numeric(frame[probability_col], errors="coerce"), 1e-8, 1 - 1e-8)
            y = pd.to_numeric(frame["y_true"], errors="coerce").astype(int)
            frame["probability"] = p
            frame["correct"] = (pd.to_numeric(frame[prediction_col], errors="coerce").astype(int) == y).astype(int)
            frame["sample_log_loss"] = -(y * np.log(p) + (1 - y) * np.log(1 - p))
            frame["classification_margin"] = np.abs(p - 0.5)
            top = frame.sort_values(["correct", "sample_log_loss"], ascending=[True, False]).head(args.top_n)

        frame.to_csv(task_dir / f"{task}_outer_predictions_with_errors.csv", index=False, encoding="utf-8-sig")
        top.to_csv(task_dir / f"{task}_top_{args.top_n}_error_cases.csv", index=False, encoding="utf-8-sig")
        for key, label in [
            ("FR_main", "molecule"),
            ("Reference", "reference"),
            ("Murcko_scaffold_main", "scaffold"),
            ("outer_fold", "outer_fold"),
        ]:
            summary = _group_summary(frame, key, task_type)
            if not summary.empty:
                summary.to_csv(task_dir / f"{task}_error_by_{label}.csv", index=False, encoding="utf-8-sig")

        manifest.append({
            "task": task,
            "task_type": task_type,
            "input_predictions": str(path),
            "n_rows": len(frame),
            "top_n": args.top_n,
            "output_dir": str(task_dir),
        })

    pd.DataFrame(manifest).to_csv(args.output / "outer_error_analysis_manifest.csv", index=False, encoding="utf-8-sig")
    (args.output / "run_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(pd.DataFrame(manifest).to_string(index=False))
    print(f"\n[DONE] {args.output}")


if __name__ == "__main__":
    main()
