# -*- coding: utf-8 -*-
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from common.task_runner import run_task

if __name__ == "__main__":
    run_task("Char_yield", mode="main", bde="main")
