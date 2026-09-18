# -*- coding: utf-8 -*-
"""Run development-stage task runners from the project root.

This file is intentionally lightweight. Formal manuscript metrics must come
from 08_ScientificValidation nested grouped CV, not from these development
repeated-holdout runs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common.task_runner import load_config, run_task

ROOT = Path(__file__).resolve().parent


def parse_tasks(text: str) -> list[str]:
    config = load_config()
    available = list(config.get("workflow_order", config["tasks"].keys()))
    if text.strip().upper() == "ALL":
        return available
    tasks = [item.strip() for item in text.split(",") if item.strip()]
    unknown = [task for task in tasks if task not in config["tasks"]]
    if unknown:
        raise KeyError(f"Unknown tasks: {unknown}; available={available}")
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["main", "bde_ablation"], default="main")
    parser.add_argument("--tasks", default="ALL")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    outputs: list[Path] = []
    for task in parse_tasks(args.tasks):
        if args.mode == "main":
            outputs.append(run_task(task, mode="main", bde="main", dry_run=args.dry_run))
        else:
            outputs.append(run_task(task, mode="bde_ablation", bde="without", dry_run=args.dry_run))
            outputs.append(run_task(task, mode="bde_ablation", bde="with", dry_run=args.dry_run))

    print("\n[DONE] Development task outputs")
    for path in outputs:
        print(" -", path.relative_to(ROOT) if path.is_absolute() and ROOT in path.parents else path)


if __name__ == "__main__":
    main()
