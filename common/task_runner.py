# -*- coding: utf-8 -*-
"""Single source of truth for task configuration and execution."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "task_config.json"
CORE = ROOT / "common" / "pipeline_core.py"


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def env_task_key(task: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in task).upper()


def input_path(config: dict[str, Any]) -> Path:
    primary = ROOT / config["data"]["primary"]
    fallback = ROOT / config["data"]["fallback"]
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Missing both project datasets: {primary} and {fallback}")


def result_dir(task: str, mode: str, use_bde: bool, config: dict[str, Any]) -> Path:
    category = config["tasks"][task]["category"]
    label = "with_BDE" if use_bde else "without_BDE"
    if mode == "main":
        return ROOT / "results" / "main" / category / task / label
    if mode == "bde_ablation":
        return ROOT / "results" / "bde_ablation" / task / label
    raise ValueError(f"Unsupported mode={mode}")


def build_env(task: str, *, mode: str = "main", bde: str = "main") -> tuple[dict[str, str], Path]:
    config = load_config()
    if task not in config["tasks"]:
        raise KeyError(f"Unknown task={task}; available={sorted(config['tasks'])}")
    task_cfg = config["tasks"][task]

    if bde == "main":
        use_bde = bool(task_cfg["main_bde"])
    elif bde == "with":
        use_bde = True
    elif bde == "without":
        use_bde = False
    else:
        raise ValueError("bde must be main, with or without")

    env = os.environ.copy()
    env["DOPO_TASKS"] = f"DELTA,{task}" if task.startswith("Delta_") else task
    env["DOPO_REPEAT_EVAL"] = "1"
    env["DOPO_REPEAT_SEEDS"] = ",".join(map(str, config["evaluation"]["seeds"]))
    env["DOPO_DESCRIPTOR_MODE"] = str(task_cfg["descriptor_mode"])
    env["DOPO_USE_V7_FEATURES"] = "1" if task_cfg["use_v7"] else "0"
    env["DOPO_USE_BDE_FEATURES"] = "1" if use_bde else "0"
    env["DOPO_INPUT_PATH"] = str(input_path(config))

    key = env_task_key(task)
    env[f"DOPO_{key}_VIEW"] = str(task_cfg["view"])
    env[f"DOPO_{key}_K"] = str(task_cfg["k"])

    # The legacy development pipeline uses dedicated UL-94 environment keys.
    # Keep them synchronized with task_config.json so run.py main / bde-ablation
    # remain functional for the classification task as well.
    if task_cfg["type"] == "classification" and task == "UL94_V0":
        env["DOPO_UL94_VIEWS"] = str(task_cfg["view"])
        k_value = task_cfg.get("k")
        env["DOPO_UL94_K_LIST"] = "all" if k_value in {None, "all", "ALL"} else str(k_value)
        if task_cfg.get("model"):
            env["DOPO_UL94_MODELS"] = str(task_cfg["model"])

    outdir = result_dir(task, mode, use_bde, config)
    env["DOPO_RESULTS"] = str(outdir)
    return env, outdir


def run_task(task: str, *, mode: str = "main", bde: str = "main", dry_run: bool = False) -> Path:
    env, outdir = build_env(task, mode=mode, bde=bde)
    print("=" * 88)
    print(f"[TASK] {task} | mode={mode} | BDE={'on' if env['DOPO_USE_BDE_FEATURES']=='1' else 'off'}")
    print(f"[DATA] {env['DOPO_INPUT_PATH']}")
    print(f"[VIEW/K] {env.get('DOPO_' + env_task_key(task) + '_VIEW')} / {env.get('DOPO_' + env_task_key(task) + '_K')}")
    print(f"[OUT] {outdir}")
    print("=" * 88)
    if not dry_run:
        outdir.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, str(CORE)], cwd=str(ROOT), env=env, check=True)
    return outdir
