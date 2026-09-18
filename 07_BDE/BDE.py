# -*- coding: utf-8 -*-
r"""
BDE.py

Final role / 最终定位：
    这是简化整理后的正式主脚本，用于 DOPO 衍生物 P-C / P-N 键 BDE(kJ/mol) 预测。

Patched note / 修改说明：
    本文件保留原 v10.3 文件名，但已按本轮运行结果完成 result-adjusted 修改：
    - 默认 best_selection 改为 strict_r2，使最终保存模型与 summary 第一名一致。
    - pc_pn 默认 K 搜索改为 1100/1200/1300，围绕已验证最优 k=1200 微调。
    - 低于 R²=0.80 时输出解释性提示，而不是暗示代码失败。

Why this version / 为什么用这一版：
    - 以 v10.2 为基础，因为 v10.2 已经按课题主线默认只训练 pc_pn，避免 all_types 拖慢运行。
    - 保留 v9 中对 R2 有帮助的扩展指纹块：Morgan whole/target、fragment、AtomPair、Torsion、Pattern。
    - 保留 v7 的 repeated-KFold / group-KFold / error diagnostics，用作小样本下更稳健的正式评估。
    - v10.4 根据本轮运行结果调整最终选择规则：默认 strict_r2，保证保存模型与 summary 第一名一致。
    - 默认 raw target，因为 v9/v10.2 结果显示 raw 优于 bond_residual。
    - 本轮 v10.3 运行结果：SoftVote_local/raw/k=1200 的 mean R2=0.7720，为严格 R2 最优；
      k=1000 的 MAE 仅低约 0.006，因此 v10.4 不再默认用 r2_mae 抢占最终模型。
    - 默认 K 搜索集中在 1100/1200/1300，围绕本轮结果中表现最优的 k=1200 继续微调。

Default training / 默认训练：
    tasks        = pc_pn only
    target_mode  = raw only
    split_mode   = repeated_kfold
    K list       = 1100,1200,1300 for pc_pn
    feature mode = extended

Default main-table update / 默认主表更新：
    普通运行 BDE.py 时，会在 BDE 模型训练/加载完成后自动同步：
        data/DOPO_EP_new.csv
        -> data/DOPO_EP_new_with_BDE.csv
    如只想训练 BDE 独立模型，不更新主表，请加：--no-update-main-bde

What to archive / 旧版本处理：
    - v4: quick baseline only, keep for historical comparison.
    - v6: best-score reference only, keep as tuning reference.
    - v7: formal evaluation reference, keep for method description.
    - v9: R2-BOOST search reference, keep for explaining how v10 was simplified.
    - v10.3-patched: result-adjusted thesis-use version.

Input module / 输入模块：
    A prediction input must contain at least:
        Smiles      : molecular SMILES string
        Bond_Type   : target bond type, P-C or P-N for the default pc_pn model

    Optional columns can improve prediction if available:
        Sample_ID, Linkage_Type, Target_Bond_Label_clean,
        Target_Bond_Subtype, Target_X_atom_class,
        Target_Bond_Count, Potential_PX_Bond_Count

Output module / 输出模块：
    The prediction output contains:
        BDE_pred_kJ_mol       : final selected prediction
        selected_model_task   : model used, usually pc_pn
        BDE_pred_std_kJ_mol   : final-ensemble seed standard deviation
        confidence_note       : reliability warning based on bond type/model/stability
        BDE_pred_pc_pn_kJ_mol : pc_pn prediction if available

Examples on Windows:
    set DOPO_BDE_INPUT=C:\path\to\DOPO_BDE.csv
    set DOPO_BDE_RESULTS=C:\path\to\results\07_BDE\BDE_pc_pn_final_v10_3
    python BDE.py

Create an input template:
    python BDE.py --make-template BDE_prediction_input_template.csv --skip-train

Predict one molecule after training/loading saved models:
    python BDE.py --skip-train --predict-smiles "O=P1(Oc2ccccc2-c2ccccc21)C..." --bond-type P-C

Predict a CSV table:
    python BDE.py --skip-train --predict-csv BDE_prediction_input_template.csv --smiles-col Smiles --bond-col Bond_Type

Useful switches / 常用开关：
    set DOPO_BDE_INPUT=C:\path\to\DOPO_BDE.csv
    set DOPO_BDE_RESULTS=C:\path\to\results\07_BDE\BDE_pc_pn_final_v10_3
    set DOPO_BDE_TASKS=pc_pn
    set DOPO_BDE_TARGET_MODES=raw
    set DOPO_BDE_K_LIST_PC_PN=1100,1200,1300
    set DOPO_BDE_FEATURE_MODE=extended
    set DOPO_BDE_USE_XGB=1
    set DOPO_BDE_N_REPEATS=5
    set DOPO_BDE_FINAL_ENSEMBLE=1
    set DOPO_BDE_ENABLE_SEED_DIAGNOSTICS=1

Optional compatibility / 可选兼容：
    If you really need P-O / P-S / P-H fallback prediction, run:
        set DOPO_BDE_TASKS=all_types,pc_pn
    But for the thesis main DOPO P-C/P-N workflow, keep the default pc_pn only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs, RDLogger
RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import Descriptors, Lipinski, MACCSkeys, rdMolDescriptors, rdPartialCharges

from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RepeatedKFold, KFold, GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import BayesianRidge, Ridge

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    XGBRegressor = None
    HAS_XGB = False

# -----------------------------------------------------------------------------
# Paths / configuration
# -----------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1] if len(Path(__file__).resolve().parents) > 1 else THIS_DIR

_DEFAULT_DATA_CANDIDATES = [
    # Simplified BDE project layout: keep the training CSV at 07_BDE/data/BDE.csv.
    THIS_DIR / "data" / "BDE.csv",
    THIS_DIR / "data" / "DOPO_BDE.csv",
    THIS_DIR / "DOPO_BDE.csv",
    THIS_DIR / "BDE.csv",
    # Backward-compatible paths for older folders.
    THIS_DIR / "data" / "raw" / "DOPO_BDE.csv",
    THIS_DIR.parent / "data" / "BDE.csv",
    THIS_DIR.parent / "data" / "DOPO_BDE.csv",
    THIS_DIR.parent / "data" / "raw" / "DOPO_BDE.csv",
    PROJECT_ROOT / "data" / "BDE.csv",
    PROJECT_ROOT / "data" / "DOPO_BDE.csv",
    PROJECT_ROOT / "data" / "raw" / "DOPO_BDE.csv",
    PROJECT_ROOT / "07_BDE" / "data" / "BDE.csv",
    PROJECT_ROOT / "07_BDE" / "data" / "DOPO_BDE.csv",
    PROJECT_ROOT / "07_BDE" / "data" / "raw" / "DOPO_BDE.csv",
    PROJECT_ROOT / "07_BDE_cleaned_final" / "data" / "raw" / "DOPO_BDE.csv",
]

def _first_existing(paths: Iterable[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    return list(paths)[0]


def _env_int_list(name: str, default: str) -> List[int]:
    raw = os.environ.get(name, default)
    out: List[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            raise ValueError(f"Environment variable {name} must be a comma-separated int list, got: {raw}")
    return out


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def log(msg: str, *, verbose_only: bool = False) -> None:
    if verbose_only and not VERBOSE:
        return
    print(msg, flush=True)


DATA_PATH = Path(os.environ.get("DOPO_BDE_INPUT", str(_first_existing(_DEFAULT_DATA_CANDIDATES))))
OUTDIR = Path(os.environ.get("DOPO_BDE_RESULTS", str(THIS_DIR / "results" / "BDE")))
OUTDIR.mkdir(parents=True, exist_ok=True)

REPEAT_SEEDS = _env_int_list("DOPO_BDE_SEEDS", "42,52,62,72,82")
TEST_SIZE = float(os.environ.get("DOPO_BDE_TEST_SIZE", "0.2"))

# Fingerprint/search settings.  Defaults are still practical, but stronger than the
# previous single-radius setup.  Use env vars to tune without editing code.
FP_BITS = int(os.environ.get("DOPO_BDE_FP_BITS", "2048"))
WHOLE_FP_RADII = _env_int_list("DOPO_BDE_WHOLE_FP_RADII", "2,3")
RADIUS = WHOLE_FP_RADII[0] if WHOLE_FP_RADII else 2  # kept for compatibility in output names
TARGET_FP_BITS = int(os.environ.get("DOPO_BDE_TARGET_FP_BITS", "512"))
TARGET_FP_RADII = _env_int_list("DOPO_BDE_TARGET_FP_RADII", "0,1,2,3")
FRAGMENT_BITS = int(os.environ.get("DOPO_BDE_FRAGMENT_BITS", "512"))
ATOMPAIR_BITS = int(os.environ.get("DOPO_BDE_ATOMPAIR_BITS", "512"))
TORSION_BITS = int(os.environ.get("DOPO_BDE_TORSION_BITS", "512"))
PATTERN_BITS = int(os.environ.get("DOPO_BDE_PATTERN_BITS", "1024"))
FEATURE_MODE = os.environ.get("DOPO_BDE_FEATURE_MODE", "extended").strip().lower()
USE_ATOMPAIR_FP = FEATURE_MODE != "compact" and _env_flag("DOPO_BDE_USE_ATOMPAIR_FP", "1")
USE_TORSION_FP = FEATURE_MODE != "compact" and _env_flag("DOPO_BDE_USE_TORSION_FP", "1")
USE_PATTERN_FP = FEATURE_MODE != "compact" and _env_flag("DOPO_BDE_USE_PATTERN_FP", "1")

K_LIST = _env_int_list("DOPO_BDE_K_LIST", "800,1200,1600,2000,2400")
SLOW_TUNING = _env_flag("DOPO_BDE_SLOW_TUNING", "0")
FAST_MODE = _env_flag("DOPO_BDE_FAST_MODE", "1")
USE_XGB = HAS_XGB and _env_flag("DOPO_BDE_USE_XGB", "1")
DROP_NON_DOPO = _env_flag("DOPO_BDE_DROP_NON_DOPO", "1")
VERBOSE = _env_flag("DOPO_BDE_VERBOSE", "0")
PRINT_EACH_MODEL = _env_flag("DOPO_BDE_PRINT_EACH_MODEL", "0")
SUMMARY_TOP_N = int(os.environ.get("DOPO_BDE_SUMMARY_TOP_N", "5"))
FAST_N_TREES = int(os.environ.get("DOPO_BDE_FAST_N_TREES", "260"))
SLOW_N_TREES = int(os.environ.get("DOPO_BDE_SLOW_N_TREES", "900"))
FINAL_MODEL_SEED = int(os.environ.get("DOPO_BDE_FINAL_SEED", "42"))

# v10.3-patched: final model selection. Default is strict_r2 so the saved model matches
# the top mean-R2 row in BDE_pc_pn_summary.csv. r2_mae is still available by env.
BEST_R2_TOL = float(os.environ.get("DOPO_BDE_BEST_R2_TOL", "0.005"))
BEST_SELECTION_MODE = os.environ.get("DOPO_BDE_BEST_SELECTION_MODE", "strict_r2").strip().lower()
FINAL_ENSEMBLE = _env_flag("DOPO_BDE_FINAL_ENSEMBLE", "1")
FINAL_ENSEMBLE_SEEDS = _env_int_list("DOPO_BDE_FINAL_ENSEMBLE_SEEDS", "42,52,62,72,82")

# V6: compare raw target fitting with a leakage-safe bond-mean residual target.
# bond_residual fits y - mean(Bond_Type) on the training split, then adds the
# corresponding training-split bond mean back at prediction time. This can improve
# stability for mixed P-C/P-N/P-O/P-S/P-H BDE ranges without using test labels as
# features.
TARGET_MODES = [m.strip().lower() for m in os.environ.get("DOPO_BDE_TARGET_MODES", "raw").split(",") if m.strip()]
TARGET_MODES = [m for m in TARGET_MODES if m in {"raw", "bond_residual"}] or ["raw"]
ENABLE_SEED_DIAGNOSTICS = _env_flag("DOPO_BDE_ENABLE_SEED_DIAGNOSTICS", "1")
R2_GOAL = float(os.environ.get("DOPO_BDE_TARGET_R2_GOAL", "0.8"))

# V8: keep the stricter V7 evaluation (repeated K-fold CV by default)
# while enabling the stronger V6 target engineering (raw + bond_residual) together.
SPLIT_MODE = os.environ.get("DOPO_BDE_SPLIT_MODE", "repeated_kfold").strip().lower()
N_SPLITS = int(os.environ.get("DOPO_BDE_N_SPLITS", "5"))
N_REPEATS = int(os.environ.get("DOPO_BDE_N_REPEATS", "5"))
CV_RANDOM_STATE = int(os.environ.get("DOPO_BDE_CV_RANDOM_STATE", "42"))
GROUP_COL = os.environ.get("DOPO_BDE_GROUP_COL", "Canonical_SMILES").strip()

# V8 diagnostics. These do not delete samples automatically; they only write
# candidate files for manual literature/data checking.
ENABLE_ERROR_DIAGNOSTICS = _env_flag("DOPO_BDE_ENABLE_ERROR_DIAGNOSTICS", "1")
OUTLIER_TOP_N = int(os.environ.get("DOPO_BDE_OUTLIER_TOP_N", "30"))
OUTLIER_ABS_THRESHOLD = float(os.environ.get("DOPO_BDE_OUTLIER_ABS_THRESHOLD", "50"))
OUTLIER_EFFECT_STEPS = _env_int_list("DOPO_BDE_OUTLIER_EFFECT_STEPS", "0,1,3,5,10,20")

# Optional target trimming, off by default because it changes the task.
ALLOW_Y_TRIM = _env_flag("DOPO_BDE_ALLOW_Y_TRIM", "0")
Y_MAX_IF_TRIM = float(os.environ.get("DOPO_BDE_Y_MAX_IF_TRIM", "500"))

# V9 R2-BOOST: a stronger search grid is used by default, but the official
# metric is still leakage-safe CV.  Reaching 0.90 is not forced by leakage.
# To run a faster smoke test, set DOPO_BDE_FAST_MODE=1 and DOPO_BDE_USE_XGB=0.
R2_BOOST_NOTE = (
    "v10.3-patched is the result-adjusted final BDE script: pc_pn + raw target by default, "
    "with v9 feature blocks (Pattern + AtomPair + Torsion) and v7 repeated-KFold diagnostics. "
    "It selects the strict highest mean-R2 model by default and is intended for thesis-use BDE feature generation, not for forcing R2 by leakage."
)

SUPPORTED_BOND_TYPES = ["P-C", "P-N", "P-O", "P-S", "P-H"]
TASKS_DEFAULT = [t.strip() for t in os.environ.get("DOPO_BDE_TASKS", "pc_pn").split(",") if t.strip()]


TASK_K_DEFAULTS = {
    # V4 keeps the search centered around the best V3 region while testing one higher K.
    "ALL_TYPES": "1200",
    # pc_pn benefited from larger K in V3; continue searching the high-K region.
    "PC_PN": "1100,1200,1300",
}


def get_k_list_for_task(task_name: str) -> List[int]:
    """Allow task-specific K search without editing the file.

    Priority:
      1) DOPO_BDE_K_LIST_<TASK>, e.g. DOPO_BDE_K_LIST_PC_PN
      2) DOPO_BDE_K_LIST, if the user explicitly set a global list
      3) built-in task-specific defaults from the latest optimization run
    """
    task_key = task_name.upper()
    env_key = f"DOPO_BDE_K_LIST_{task_key}"
    if env_key in os.environ:
        return _env_int_list(env_key, TASK_K_DEFAULTS.get(task_key, ",".join(str(k) for k in K_LIST)))
    if "DOPO_BDE_K_LIST" in os.environ:
        return K_LIST
    return _env_int_list(env_key, TASK_K_DEFAULTS.get(task_key, ",".join(str(k) for k in K_LIST)))


def task_k_lists(tasks: List[str]) -> Dict[str, List[int]]:
    """Report the actual K values used for each task."""
    out: Dict[str, List[int]] = {}
    for t in tasks:
        tt = "pc_pn" if t.lower() in {"pcpn", "pc_pn_classified"} else t.lower()
        out[tt] = get_k_list_for_task(tt)
    return out


CLASS_NUMERIC_COLS = ["Target_Bond_Count", "Potential_PX_Bond_Count"]
CLASS_CATEGORICAL_COLS = [
    "Linkage_Type",
    "Target_Bond_Label_clean",
    "Target_Bond_Subtype",
    "Target_X_atom_class",
]

# -----------------------------------------------------------------------------
# IO / cleaning
# -----------------------------------------------------------------------------
def read_csv_auto(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"]
    last_error = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"[INFO] File loaded: {path} | encoding={enc} | shape={df.shape}")
            return df
        except Exception as e:
            last_error = e
    searched = "\n".join(f"  - {p}" for p in _DEFAULT_DATA_CANDIDATES)
    raise RuntimeError(
        f"Cannot read {path}: {last_error}\n"
        f"Please check that DOPO_BDE.csv exists. The script searched these default locations:\n{searched}\n"
        f"You can also set DOPO_BDE_INPUT to the exact CSV path, for example:\n"
        f"set DOPO_BDE_INPUT=C:\\path\\to\\DOPO_BDE.csv"
    )


def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null", "na", "n/a"}:
        return ""
    return s


def normalize_bond_type(x) -> str:
    """Normalize P_C / DOPO_P-C / P–C / PH labels to P-C etc."""
    s = clean_text(x).upper()
    if not s:
        return "Unknown"
    s = s.replace("–", "-").replace("—", "-").replace("_", "-").replace(" ", "-")
    s = re.sub(r"-+", "-", s)
    if "P-C" in s or s == "PC":
        return "P-C"
    if "P-N" in s or s == "PN":
        return "P-N"
    if "P-O" in s or s == "PO":
        return "P-O"
    if "P-S" in s or s == "PS":
        return "P-S"
    if "P-H" in s or s == "PH":
        return "P-H"
    return s


def safe_mol(smiles: str):
    smiles = clean_text(smiles)
    if not smiles:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def canonical_smiles(smiles: str) -> str:
    mol = safe_mol(smiles)
    if mol is None:
        return ""
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return clean_text(smiles)


def resolve_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    cols = list(df.columns)
    smiles_candidates = ["Smiles", "SMILES", "SMILES_main", "smiles"]
    bond_candidates = [
        "Target_Bond_Label_clean", "Target_Bond_Label", "Bond_Type_norm", "Bond_Type", "BDE_Type", "Bond type", "bond_type",
    ]
    bde_candidates = ["BDE(kJ/mol)", "BDE(KJ/mol)", "BDE_kJ_mol", "BDE", "BDE(kcal/mol)", "BDE(Kcal/mol)"]

    def find(cands: List[str]) -> str:
        for c in cands:
            if c in cols:
                return c
        lower_map = {c.lower(): c for c in cols}
        for c in cands:
            if c.lower() in lower_map:
                return lower_map[c.lower()]
        raise KeyError(f"Cannot find any of {cands} in columns: {cols}")

    return find(smiles_candidates), find(bond_candidates), find(bde_candidates)


def bde_to_kj(series: pd.Series, col_name: str) -> pd.Series:
    y = pd.to_numeric(series, errors="coerce")
    name = str(col_name).lower()
    med = y.dropna().median()
    if "kcal" in name or ("kj" not in name and pd.notna(med) and med < 150):
        return y * 4.184
    return y


def ensure_classified_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    if "Bond_Type_norm" not in data.columns:
        data["Bond_Type_norm"] = data["Bond_Type"]
    if "Target_Bond_Label_clean" not in data.columns:
        if "Target_Bond_Label" in data.columns:
            data["Target_Bond_Label_clean"] = data["Target_Bond_Label"].apply(clean_text)
        else:
            data["Target_Bond_Label_clean"] = data["Bond_Type"].map({
                "P-C": "DOPO_P-C", "P-N": "DOPO_P-N", "P-O": "DOPO_P-O", "P-S": "DOPO_P-S", "P-H": "DOPO_P-H",
            }).fillna("")
    if "Linkage_Type" not in data.columns:
        data["Linkage_Type"] = data["Bond_Type"].map({
            "P-C": "DOPO-C", "P-N": "DOPO-N", "P-O": "DOPO-O", "P-S": "DOPO-S", "P-H": "DOPO-H",
        }).fillna("")
    if "Model_Use" not in data.columns:
        data["Model_Use"] = np.where(data["Bond_Type"].isin(["P-C", "P-N"]), "Use_pc_pn_model", "Use_all_types_model")
    for col in CLASS_NUMERIC_COLS:
        if col not in data.columns:
            data[col] = np.nan
    for col in CLASS_CATEGORICAL_COLS:
        if col not in data.columns:
            data[col] = ""
    return data


def load_clean_training_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = read_csv_auto(path)
    smi_col, bond_col, bde_col = resolve_columns(df)
    print(f"[INFO] Resolved columns: SMILES={smi_col}, Bond_Type={bond_col}, BDE={bde_col}")
    data = df.copy()
    data["Smiles"] = data[smi_col].apply(clean_text)
    data["Canonical_SMILES"] = data["Smiles"].apply(canonical_smiles)
    data["Bond_Type"] = data[bond_col].apply(normalize_bond_type)
    data = ensure_classified_columns(data)
    data["BDE_kJ_mol"] = bde_to_kj(data[bde_col], bde_col)
    data = data.dropna(subset=["BDE_kJ_mol"])
    data = data[data["Canonical_SMILES"].astype(str).str.len() > 0].copy()
    before = len(data)
    data = data.drop_duplicates(subset=["Canonical_SMILES", "Bond_Type", "BDE_kJ_mol"]).reset_index(drop=True)
    print(f"[INFO] valid={len(data)} | exact_duplicates_removed={before-len(data)}")
    print(f"[INFO] all Bond_Type distribution: {data['Bond_Type'].value_counts().to_dict()}")
    print(f"[INFO] BDE range: {data['BDE_kJ_mol'].min():.2f} - {data['BDE_kJ_mol'].max():.2f} kJ/mol")
    return data

# -----------------------------------------------------------------------------
# Task filtering
# -----------------------------------------------------------------------------
def task_filter(data: pd.DataFrame, task_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    task_name = task_name.lower().strip()
    if task_name == "all_types":
        use_mask = data["Bond_Type"].isin(SUPPORTED_BOND_TYPES).copy()
        # For all_types, do NOT remove P-O/P-S/P-H just because Model_Use says Exclude_from_pc_pn_model.
    elif task_name in {"pc_pn", "pcpn", "pc_pn_classified"}:
        task_name = "pc_pn"
        use_mask = data["Bond_Type"].isin(["P-C", "P-N"]).copy()
        if "Model_Use" in data.columns:
            model_use = data["Model_Use"].astype(str).str.lower()
            use_mask &= ~model_use.str.contains("exclude", na=False)
            if model_use.str.contains("use", na=False).any():
                use_mask &= model_use.str.contains("use", na=False)
    else:
        raise ValueError(f"Unknown task_name={task_name}; choose from all_types, pc_pn")

    if DROP_NON_DOPO and "Linkage_Type" in data.columns:
        linkage = data["Linkage_Type"].astype(str).str.lower()
        use_mask &= ~linkage.str.contains("non_dopo|non-dopo|non dopo", regex=True, na=False)
    if "Need_Manual_Check" in data.columns:
        manual = pd.to_numeric(data["Need_Manual_Check"], errors="coerce").fillna(0)
        use_mask &= manual.eq(0)

    model_data = data.loc[use_mask].reset_index(drop=True)
    if ALLOW_Y_TRIM:
        model_data = model_data[model_data["BDE_kJ_mol"] <= Y_MAX_IF_TRIM].reset_index(drop=True)
    excluded = data.loc[~use_mask].reset_index(drop=True)
    return model_data, excluded

# -----------------------------------------------------------------------------
# Feature engineering
# -----------------------------------------------------------------------------
def sanitize_feature_name(x) -> str:
    s = clean_text(x)
    if not s:
        return "Unknown"
    s = re.sub(r"[^0-9A-Za-z_\-]+", "_", s)
    s = s.replace("-", "_")
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "Unknown"


def morgan_fp(mol, n_bits: int = FP_BITS, radius: int = RADIUS) -> np.ndarray:
    arr = np.zeros((n_bits,), dtype=np.int8)
    if mol is None:
        return arr
    try:
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        DataStructs.ConvertToNumpyArray(fp, arr)
    except Exception:
        pass
    return arr


def maccs_fp(mol) -> np.ndarray:
    arr = np.zeros((167,), dtype=np.int8)
    if mol is None:
        return arr
    try:
        fp = MACCSkeys.GenMACCSKeys(mol)
        DataStructs.ConvertToNumpyArray(fp, arr)
    except Exception:
        pass
    return arr


def _bitvect_to_array(fp, n_bits: int) -> np.ndarray:
    arr = np.zeros((n_bits,), dtype=np.int8)
    try:
        DataStructs.ConvertToNumpyArray(fp, arr)
    except Exception:
        pass
    return arr


def morgan_multi_fp(mol, radii: Optional[List[int]] = None, n_bits: int = FP_BITS) -> Dict[str, np.ndarray]:
    radii = WHOLE_FP_RADII if radii is None else radii
    out: Dict[str, np.ndarray] = {}
    for radius in radii:
        arr = np.zeros((n_bits,), dtype=np.int8)
        if mol is not None:
            try:
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                arr = _bitvect_to_array(fp, n_bits)
            except Exception:
                pass
        out[f"morgan_r{radius}"] = arr
    return out


def atom_pair_fp(mol, n_bits: int = ATOMPAIR_BITS) -> np.ndarray:
    arr = np.zeros((n_bits,), dtype=np.int8)
    if mol is None or n_bits <= 0:
        return arr
    try:
        fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
        arr = _bitvect_to_array(fp, n_bits)
    except Exception:
        pass
    return arr


def torsion_fp(mol, n_bits: int = TORSION_BITS) -> np.ndarray:
    arr = np.zeros((n_bits,), dtype=np.int8)
    if mol is None or n_bits <= 0:
        return arr
    try:
        fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=n_bits)
        arr = _bitvect_to_array(fp, n_bits)
    except Exception:
        pass
    return arr


def pattern_fp(mol, n_bits: int = PATTERN_BITS) -> np.ndarray:
    arr = np.zeros((n_bits,), dtype=np.int8)
    if mol is None or n_bits <= 0:
        return arr
    try:
        fp = Chem.PatternFingerprint(mol, fpSize=n_bits)
        arr = _bitvect_to_array(fp, n_bits)
    except Exception:
        pass
    return arr


def calc_desc(mol) -> Dict[str, float]:
    keys = [
        "MolWt", "TPSA", "LogP", "HBD", "HBA", "RotBonds", "RingCount", "AromaticRings",
        "HeavyAtomCount", "FractionCSP3", "NumHeteroatoms", "NumValenceElectrons", "BertzCT",
        "BalabanJ", "MolMR", "ExactMolWt", "MaxPartialCharge", "MinPartialCharge",
        "MaxAbsPartialCharge", "MinAbsPartialCharge", "FpDensityMorgan1", "FpDensityMorgan2",
        "FpDensityMorgan3", "BCUT2D_MWHI", "BCUT2D_MWLOW", "BCUT2D_CHGHI", "BCUT2D_CHGLO",
        "BCUT2D_LOGPHI", "BCUT2D_LOGPLOW", "Kappa1", "Kappa2", "Kappa3", "HallKierAlpha",
        "NumAliphaticRings", "NumSaturatedRings", "NumAromaticHeterocycles", "NumAromaticCarbocycles",
    ]
    if mol is None:
        return {k: 0.0 for k in keys}

    def safe(fn, default=0.0):
        try:
            v = fn(mol)
            if v is None or not np.isfinite(float(v)):
                return default
            return float(v)
        except Exception:
            return default

    return {
        "MolWt": safe(Descriptors.MolWt),
        "TPSA": safe(Descriptors.TPSA),
        "LogP": safe(Descriptors.MolLogP),
        "HBD": safe(Lipinski.NumHDonors),
        "HBA": safe(Lipinski.NumHAcceptors),
        "RotBonds": safe(Lipinski.NumRotatableBonds),
        "RingCount": safe(Lipinski.RingCount),
        "AromaticRings": safe(Lipinski.NumAromaticRings),
        "HeavyAtomCount": safe(Descriptors.HeavyAtomCount),
        "FractionCSP3": safe(Descriptors.FractionCSP3),
        "NumHeteroatoms": safe(Descriptors.NumHeteroatoms),
        "NumValenceElectrons": safe(Descriptors.NumValenceElectrons),
        "BertzCT": safe(Descriptors.BertzCT),
        "BalabanJ": safe(Descriptors.BalabanJ),
        "MolMR": safe(Descriptors.MolMR),
        "ExactMolWt": safe(Descriptors.ExactMolWt),
        "MaxPartialCharge": safe(Descriptors.MaxPartialCharge),
        "MinPartialCharge": safe(Descriptors.MinPartialCharge),
        "MaxAbsPartialCharge": safe(Descriptors.MaxAbsPartialCharge),
        "MinAbsPartialCharge": safe(Descriptors.MinAbsPartialCharge),
        "FpDensityMorgan1": safe(Descriptors.FpDensityMorgan1),
        "FpDensityMorgan2": safe(Descriptors.FpDensityMorgan2),
        "FpDensityMorgan3": safe(Descriptors.FpDensityMorgan3),
        "BCUT2D_MWHI": safe(Descriptors.BCUT2D_MWHI),
        "BCUT2D_MWLOW": safe(Descriptors.BCUT2D_MWLOW),
        "BCUT2D_CHGHI": safe(Descriptors.BCUT2D_CHGHI),
        "BCUT2D_CHGLO": safe(Descriptors.BCUT2D_CHGLO),
        "BCUT2D_LOGPHI": safe(Descriptors.BCUT2D_LOGPHI),
        "BCUT2D_LOGPLOW": safe(Descriptors.BCUT2D_LOGPLOW),
        "Kappa1": safe(Descriptors.Kappa1),
        "Kappa2": safe(Descriptors.Kappa2),
        "Kappa3": safe(Descriptors.Kappa3),
        "HallKierAlpha": safe(Descriptors.HallKierAlpha),
        "NumAliphaticRings": safe(Descriptors.NumAliphaticRings),
        "NumSaturatedRings": safe(Descriptors.NumSaturatedRings),
        "NumAromaticHeterocycles": safe(Descriptors.NumAromaticHeterocycles),
        "NumAromaticCarbocycles": safe(Descriptors.NumAromaticCarbocycles),
    }


def smiles_text_features(smiles: str) -> Dict[str, float]:
    s = clean_text(smiles)
    return {
        "smi_len": float(len(s)),
        "smi_count_P": float(s.count("P")),
        "smi_count_N": float(s.count("N")),
        "smi_count_O": float(s.count("O")),
        "smi_count_S": float(s.count("S")),
        "smi_count_F": float(s.count("F")),
        "smi_count_Cl": float(s.count("Cl") + s.count("cl")),
        "smi_count_Br": float(s.count("Br") + s.count("br")),
        "smi_count_ring_digits": float(sum(ch.isdigit() for ch in s)),
        "smi_count_aromatic_c": float(s.count("c")),
        "smi_count_branch": float(s.count("(") + s.count(")")),
        "smi_count_double": float(s.count("=")),
        "smi_count_hash": float(s.count("#")),
        "smi_count_brackets": float(s.count("[") + s.count("]")),
        "smi_count_dots": float(s.count(".")),
        "smi_ratio_P": float(s.count("P")) / max(len(s), 1),
        "smi_ratio_N": float(s.count("N")) / max(len(s), 1),
        "smi_ratio_O": float(s.count("O")) / max(len(s), 1),
        "smi_has_aromatic": float(1 if any(ch in s for ch in ["c", "n", "o", "s"]) else 0),
        "smi_has_hetero": float(1 if any(c in s for c in "PNOFS") else 0),
    }


def _target_symbol_for_bond_type(bond_type: str) -> Optional[str]:
    return {"P-C": "C", "P-N": "N", "P-O": "O", "P-S": "S", "P-H": "H"}.get(normalize_bond_type(bond_type))


def target_bond_pairs(mol, bond_type: str) -> List[Tuple[int, Optional[int], Optional[int]]]:
    """Return target P-X pairs as (P_idx, X_idx, bond_idx). X_idx/bond_idx may be None for implicit P-H."""
    target_symbol = _target_symbol_for_bond_type(bond_type)
    pairs: List[Tuple[int, Optional[int], Optional[int]]] = []
    if mol is None or target_symbol is None:
        return pairs

    if target_symbol == "H":
        # Explicit P-H, if present.
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom(); a2 = bond.GetEndAtom()
            if a1.GetSymbol() == "P" and a2.GetSymbol() == "H":
                pairs.append((a1.GetIdx(), a2.GetIdx(), bond.GetIdx()))
            elif a2.GetSymbol() == "P" and a1.GetSymbol() == "H":
                pairs.append((a2.GetIdx(), a1.GetIdx(), bond.GetIdx()))
        # Most SMILES store P-H as implicit H on P; use P atom as target center.
        if not pairs:
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == "P" and atom.GetTotalNumHs() > 0:
                    pairs.append((atom.GetIdx(), None, None))
        return pairs

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom(); a2 = bond.GetEndAtom()
        if a1.GetSymbol() == "P" and a2.GetSymbol() == target_symbol:
            pairs.append((a1.GetIdx(), a2.GetIdx(), bond.GetIdx()))
        elif a2.GetSymbol() == "P" and a1.GetSymbol() == target_symbol:
            pairs.append((a2.GetIdx(), a1.GetIdx(), bond.GetIdx()))
    return pairs


def _get_gasteiger_mol(mol):
    if mol is None:
        return None
    try:
        m = Chem.Mol(mol)
        rdPartialCharges.ComputeGasteigerCharges(m)
        return m
    except Exception:
        return mol


def _safe_charge(atom) -> float:
    try:
        v = float(atom.GetProp("_GasteigerCharge"))
        return v if np.isfinite(v) else 0.0
    except Exception:
        return 0.0


def target_local_features(mol, bond_type: str) -> Dict[str, float]:
    keys = [
        "target_pair_count", "target_bond_order_mean", "target_bond_in_ring_mean", "target_bond_conj_mean",
        "target_P_charge_mean", "target_X_charge_mean", "target_PX_charge_diff_mean",
        "target_P_degree_mean", "target_X_degree_mean", "target_P_valence_mean", "target_X_valence_mean",
        "target_X_aromatic_mean", "target_X_ring_mean", "target_X_sp2_mean", "target_X_sp3_mean",
        "target_P_neighbor_C_mean", "target_P_neighbor_N_mean", "target_P_neighbor_O_mean", "target_P_neighbor_S_mean",
        "target_X_neighbor_C_mean", "target_X_neighbor_N_mean", "target_X_neighbor_O_mean", "target_X_neighbor_S_mean",
        "target_X_neighbor_hetero_mean", "target_P_doubleO_mean", "target_P_implicit_H_mean",
        "mol_P_count", "mol_PC_bond_count", "mol_PN_bond_count", "mol_PO_bond_count", "mol_PS_bond_count", "mol_PH_implicit_count", "mol_P_doubleO_count",
    ]
    out = {k: 0.0 for k in keys}
    if mol is None:
        return out

    pairs = target_bond_pairs(mol, bond_type)
    out["target_pair_count"] = float(len(pairs))

    p_atoms = [a for a in mol.GetAtoms() if a.GetSymbol() == "P"]
    out["mol_P_count"] = float(len(p_atoms))
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom(); a2 = bond.GetEndAtom()
        if "P" not in {a1.GetSymbol(), a2.GetSymbol()}:
            continue
        other = a2 if a1.GetSymbol() == "P" else a1
        osym = other.GetSymbol()
        if osym == "C": out["mol_PC_bond_count"] += 1.0
        elif osym == "N": out["mol_PN_bond_count"] += 1.0
        elif osym == "O":
            out["mol_PO_bond_count"] += 1.0
            if str(bond.GetBondType()).upper() == "DOUBLE": out["mol_P_doubleO_count"] += 1.0
        elif osym == "S": out["mol_PS_bond_count"] += 1.0
        elif osym == "H": out["mol_PH_implicit_count"] += 1.0
    out["mol_PH_implicit_count"] += float(sum(a.GetTotalNumHs() for a in p_atoms))

    if not pairs:
        return out

    cmol = _get_gasteiger_mol(mol)
    rows = []
    for p_idx, x_idx, bond_idx in pairs:
        p_atom = mol.GetAtomWithIdx(p_idx)
        x_atom = mol.GetAtomWithIdx(x_idx) if x_idx is not None else None
        p_charge = _safe_charge(cmol.GetAtomWithIdx(p_idx)) if cmol is not None else 0.0
        x_charge = _safe_charge(cmol.GetAtomWithIdx(x_idx)) if (cmol is not None and x_idx is not None) else 0.0
        bond = mol.GetBondWithIdx(bond_idx) if bond_idx is not None else None
        p_neighbors = [n for n in p_atom.GetNeighbors()]
        p_symbols = [n.GetSymbol() for n in p_neighbors]
        if x_atom is not None:
            x_neighbors = [n for n in x_atom.GetNeighbors() if n.GetIdx() != p_idx]
            x_symbols = [n.GetSymbol() for n in x_neighbors]
            hyb = str(x_atom.GetHybridization()).upper()
            x_atomic_num = x_atom.GetAtomicNum()
            x_degree = x_atom.GetDegree()
            x_valence = x_atom.GetTotalValence()
            x_arom = float(x_atom.GetIsAromatic())
            x_ring = float(x_atom.IsInRing())
        else:
            x_symbols = []
            hyb = "S"
            x_atomic_num = 1
            x_degree = 1
            x_valence = 1
            x_arom = 0.0
            x_ring = 0.0

        p_double_o = 0
        for b in p_atom.GetBonds():
            other = b.GetOtherAtom(p_atom)
            if other.GetSymbol() == "O" and str(b.GetBondType()).upper() == "DOUBLE":
                p_double_o += 1

        rows.append({
            "target_bond_order": float(bond.GetBondTypeAsDouble()) if bond is not None else 1.0,
            "target_bond_in_ring": float(bond.IsInRing()) if bond is not None else 0.0,
            "target_bond_conj": float(bond.GetIsConjugated()) if bond is not None else 0.0,
            "target_P_charge": p_charge,
            "target_X_charge": x_charge,
            "target_PX_charge_diff": p_charge - x_charge,
            "target_P_degree": float(p_atom.GetDegree()),
            "target_X_degree": float(x_degree),
            "target_P_valence": float(p_atom.GetTotalValence()),
            "target_X_valence": float(x_valence),
            "target_X_aromatic": x_arom,
            "target_X_ring": x_ring,
            "target_X_sp2": float("SP2" in hyb),
            "target_X_sp3": float("SP3" in hyb),
            "target_P_neighbor_C": float(p_symbols.count("C")),
            "target_P_neighbor_N": float(p_symbols.count("N")),
            "target_P_neighbor_O": float(p_symbols.count("O")),
            "target_P_neighbor_S": float(p_symbols.count("S")),
            "target_X_neighbor_C": float(x_symbols.count("C")),
            "target_X_neighbor_N": float(x_symbols.count("N")),
            "target_X_neighbor_O": float(x_symbols.count("O")),
            "target_X_neighbor_S": float(x_symbols.count("S")),
            "target_X_neighbor_hetero": float(sum(1 for s in x_symbols if s not in {"C", "H"})),
            "target_P_doubleO": float(p_double_o),
            "target_P_implicit_H": float(p_atom.GetTotalNumHs()),
        })
    tmp = pd.DataFrame(rows)
    for col in tmp.columns:
        out[f"{col}_mean"] = float(tmp[col].mean())
    return out


def target_centered_fingerprint(mol, bond_type: str, n_bits: int = TARGET_FP_BITS) -> np.ndarray:
    """Morgan fingerprints centered on target P/X atoms for configurable radii."""
    out_by_radius = []
    pairs = target_bond_pairs(mol, bond_type)
    for radius in TARGET_FP_RADII:
        arr = np.zeros((n_bits,), dtype=np.int8)
        if mol is not None and pairs:
            for p_idx, x_idx, _ in pairs:
                from_atoms = [p_idx] if x_idx is None else [p_idx, x_idx]
                try:
                    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, fromAtoms=from_atoms)
                    tmp = _bitvect_to_array(fp, n_bits)
                    arr = np.maximum(arr, tmp)
                except Exception:
                    pass
        out_by_radius.append(arr)
    if not out_by_radius:
        return np.zeros((0,), dtype=np.int8)
    return np.concatenate(out_by_radius)


def target_fragment_hash(mol, bond_type: str, n_bits: int = FRAGMENT_BITS) -> np.ndarray:
    """Hashed canonical local fragments within radius 1-4 around target P/X atoms."""
    arr = np.zeros((n_bits,), dtype=np.int8)
    pairs = target_bond_pairs(mol, bond_type)
    if mol is None or not pairs:
        return arr
    try:
        dist = Chem.GetDistanceMatrix(mol)
    except Exception:
        return arr
    for p_idx, x_idx, _ in pairs:
        centers = [p_idx] if x_idx is None else [p_idx, x_idx]
        for rad in [1, 2, 3, 4]:
            atom_ids = [i for i in range(mol.GetNumAtoms()) if any(dist[c, i] <= rad for c in centers)]
            try:
                smi = Chem.MolFragmentToSmiles(mol, atomsToUse=atom_ids, canonical=True)
                # Python hash is salted across sessions. Use a deterministic simple hash.
                h = 0
                for ch in f"{rad}|{smi}":
                    h = (h * 131 + ord(ch)) % n_bits
                arr[h] = 1
            except Exception:
                pass
    return arr


def add_classified_features(feat: Dict[str, float], row: pd.Series, bond_type: str) -> None:
    for col in CLASS_NUMERIC_COLS:
        val = row.get(col, np.nan)
        feat[f"class_{col}"] = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
    for col in CLASS_CATEGORICAL_COLS:
        val = clean_text(row.get(col, ""))
        if val:
            feat[f"class_{col}_{sanitize_feature_name(val)}"] = 1.0
    fallback = {
        "P-C": ("DOPO_C", "DOPO_P_C"),
        "P-N": ("DOPO_N", "DOPO_P_N"),
        "P-O": ("DOPO_O", "DOPO_P_O"),
        "P-S": ("DOPO_S", "DOPO_P_S"),
        "P-H": ("DOPO_H", "DOPO_P_H"),
    }
    if bond_type in fallback:
        link, label = fallback[bond_type]
        feat[f"class_Linkage_fallback_{link}"] = 1.0
        feat[f"class_Target_Bond_Label_fallback_{label}"] = 1.0


def build_prediction_frame(smiles: str, bond_type: str) -> pd.DataFrame:
    btype = normalize_bond_type(bond_type)
    label = {"P-C": "DOPO_P-C", "P-N": "DOPO_P-N", "P-O": "DOPO_P-O", "P-S": "DOPO_P-S", "P-H": "DOPO_P-H"}.get(btype, "")
    linkage = {"P-C": "DOPO-C", "P-N": "DOPO-N", "P-O": "DOPO-O", "P-S": "DOPO-S", "P-H": "DOPO-H"}.get(btype, "")
    return pd.DataFrame([{
        "Smiles": clean_text(smiles),
        "Bond_Type": btype,
        "Canonical_SMILES": canonical_smiles(smiles),
        "Linkage_Type": linkage,
        "Target_Bond_Label_clean": label,
        "Target_Bond_Subtype": label,
        "Target_X_atom_class": _target_symbol_for_bond_type(btype) or "",
        "Target_Bond_Count": np.nan,
        "Potential_PX_Bond_Count": np.nan,
    }])


def build_features(df: pd.DataFrame, smiles_col: str = "Smiles", bond_col: str = "Bond_Type") -> pd.DataFrame:
    """Build leakage-safe features for BDE prediction.

    Feature blocks:
      - RDKit descriptors + SMILES text features
      - whole-molecule Morgan fingerprints at configurable radii
      - MACCS keys
      - target-bond-centered Morgan fingerprints
      - local target-fragment hash
      - optional AtomPair / TopologicalTorsion / Pattern fingerprints
    """
    desc_rows: List[Dict[str, float]] = []
    whole_fps_by_radius: Dict[str, List[np.ndarray]] = {f"morgan_r{r}": [] for r in WHOLE_FP_RADII}
    maccs_fps: List[np.ndarray] = []
    target_fps: List[np.ndarray] = []
    frag_fps: List[np.ndarray] = []
    atompair_fps: List[np.ndarray] = []
    torsion_fps: List[np.ndarray] = []
    pattern_fps: List[np.ndarray] = []
    cache = {}

    for _, row in df.iterrows():
        smiles = clean_text(row.get(smiles_col, ""))
        bond_type = normalize_bond_type(row.get(bond_col, "Unknown"))
        class_key = tuple(clean_text(row.get(c, "")) for c in CLASS_CATEGORICAL_COLS)
        num_key = tuple(clean_text(row.get(c, "")) for c in CLASS_NUMERIC_COLS)
        cache_key = (smiles, bond_type, class_key, num_key, FEATURE_MODE, tuple(WHOLE_FP_RADII), tuple(TARGET_FP_RADII))
        if cache_key in cache:
            cached = cache[cache_key]
            desc_rows.append(dict(cached["feat"]))
            for key in whole_fps_by_radius:
                whole_fps_by_radius[key].append(cached[key].copy())
            maccs_fps.append(cached["maccs"].copy())
            target_fps.append(cached["target"].copy())
            frag_fps.append(cached["fragment"].copy())
            if USE_ATOMPAIR_FP:
                atompair_fps.append(cached["atompair"].copy())
            if USE_TORSION_FP:
                torsion_fps.append(cached["torsion"].copy())
            if USE_PATTERN_FP:
                pattern_fps.append(cached["pattern"].copy())
            continue

        mol = safe_mol(smiles)
        feat: Dict[str, float] = {}
        feat.update(calc_desc(mol))
        feat.update(smiles_text_features(smiles))
        feat.update(target_local_features(mol, bond_type))
        add_classified_features(feat, row, bond_type)
        for bt in SUPPORTED_BOND_TYPES:
            feat[f"Bond_is_{bt.replace('-', '')}"] = float(bond_type == bt)
        feat["Bond_is_Unknown"] = float(bond_type == "Unknown")

        multi = morgan_multi_fp(mol)
        maccs = maccs_fp(mol)
        tfp = target_centered_fingerprint(mol, bond_type)
        ffp = target_fragment_hash(mol, bond_type)
        apfp = atom_pair_fp(mol) if USE_ATOMPAIR_FP else np.zeros((0,), dtype=np.int8)
        torfp = torsion_fp(mol) if USE_TORSION_FP else np.zeros((0,), dtype=np.int8)
        patfp = pattern_fp(mol) if USE_PATTERN_FP else np.zeros((0,), dtype=np.int8)

        cached = {"feat": dict(feat), "maccs": maccs.copy(), "target": tfp.copy(), "fragment": ffp.copy(),
                  "atompair": apfp.copy(), "torsion": torfp.copy(), "pattern": patfp.copy()}
        for key, arr in multi.items():
            cached[key] = arr.copy()
        cache[cache_key] = cached

        desc_rows.append(feat)
        for key in whole_fps_by_radius:
            whole_fps_by_radius[key].append(multi.get(key, np.zeros((FP_BITS,), dtype=np.int8)))
        maccs_fps.append(maccs)
        target_fps.append(tfp)
        frag_fps.append(ffp)
        if USE_ATOMPAIR_FP:
            atompair_fps.append(apfp)
        if USE_TORSION_FP:
            torsion_fps.append(torfp)
        if USE_PATTERN_FP:
            pattern_fps.append(patfp)

    blocks = [pd.DataFrame(desc_rows, index=df.index)]
    for key, arrs in whole_fps_by_radius.items():
        blocks.append(pd.DataFrame(np.asarray(arrs), index=df.index, columns=[f"{key}_{i}" for i in range(FP_BITS)]))
    blocks.append(pd.DataFrame(np.asarray(maccs_fps), index=df.index, columns=[f"maccs_{i}" for i in range(167)]))
    blocks.append(pd.DataFrame(np.asarray(target_fps), index=df.index, columns=[f"target_morgan_{i}" for i in range(TARGET_FP_BITS * len(TARGET_FP_RADII))]))
    blocks.append(pd.DataFrame(np.asarray(frag_fps), index=df.index, columns=[f"target_fragment_{i}" for i in range(FRAGMENT_BITS)]))
    if USE_ATOMPAIR_FP:
        blocks.append(pd.DataFrame(np.asarray(atompair_fps), index=df.index, columns=[f"atompair_{i}" for i in range(ATOMPAIR_BITS)]))
    if USE_TORSION_FP:
        blocks.append(pd.DataFrame(np.asarray(torsion_fps), index=df.index, columns=[f"torsion_{i}" for i in range(TORSION_BITS)]))
    if USE_PATTERN_FP:
        blocks.append(pd.DataFrame(np.asarray(pattern_fps), index=df.index, columns=[f"pattern_{i}" for i in range(PATTERN_BITS)]))

    X = pd.concat(blocks, axis=1)
    return X.replace([np.inf, -np.inf], np.nan)

# -----------------------------------------------------------------------------
# Models / preprocessing
# -----------------------------------------------------------------------------
def model_pool(seed: int, task_name: str = "pc_pn") -> Dict[str, object]:
    """Fast project-focused model pool.

    Kept based on v9 results:
      - SoftVote_local was the best pc_pn model in v9.
      - ExtraTrees_sqrt_local is retained as a strong, stable comparison.
      - RF_sqrt_local is retained as a baseline.

    Removed from default search for speed:
      - all standalone XGB variants, extra RF/ET variants, GBDT-only search,
        Ridge/KNN/SVR/KernelRidge/HistGB/BayesianRidge.

    XGB handling:
      - DOPO_BDE_USE_XGB=1 includes XGB only inside SoftVote_local by default.
      - Standalone XGB models are still excluded unless DOPO_BDE_RUN_STANDALONE_XGB=1.
      - This keeps most of the v9 gain while avoiding the full v9 runtime.
    """
    n_tree = SLOW_N_TREES if SLOW_TUNING else FAST_N_TREES
    models: Dict[str, object] = {
        "RF_sqrt_local": RandomForestRegressor(
            n_estimators=n_tree,
            max_features="sqrt",
            min_samples_leaf=1,
            random_state=seed,
            n_jobs=-1,
        ),
        "ExtraTrees_sqrt_local": ExtraTreesRegressor(
            n_estimators=n_tree,
            max_features="sqrt",
            min_samples_leaf=1,
            random_state=seed,
            n_jobs=-1,
        ),
    }

    vote_estimators = [
        ("rf", RandomForestRegressor(
            n_estimators=max(70, n_tree // 2),
            max_features="sqrt",
            min_samples_leaf=1,
            random_state=seed,
            n_jobs=-1,
        )),
        ("et", ExtraTreesRegressor(
            n_estimators=max(70, n_tree // 2),
            max_features="sqrt",
            min_samples_leaf=1,
            random_state=seed,
            n_jobs=-1,
        )),
        ("gb", GradientBoostingRegressor(
            n_estimators=80,
            learning_rate=0.045,
            max_depth=3,
            subsample=0.85,
            random_state=seed,
        )),
    ]

    if USE_XGB:
        vote_estimators.append(("xgb", XGBRegressor(
            n_estimators=200,
            learning_rate=0.035,
            max_depth=3,
            subsample=0.90,
            colsample_bytree=0.78,
            reg_lambda=3.0,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )))

    models["SoftVote_local"] = VotingRegressor(vote_estimators, n_jobs=None)

    # v10.3-patched default: keep only models that mattered in v9/v10 to reduce runtime.
    # Set DOPO_BDE_SLIM_MODEL_POOL=0 to recover the wider v10 pool.
    if _env_flag("DOPO_BDE_SLIM_MODEL_POOL", "1"):
        keep = {"RF_sqrt_local", "ExtraTrees_sqrt_local", "SoftVote_local"}
        if _env_flag("DOPO_BDE_RUN_STANDALONE_XGB", "0"):
            keep.update({"XGB_local", "XGB_shallow_local"})
        models = {k: v for k, v in models.items() if k in keep}
    return models

def fit_preprocessor(X_train: pd.DataFrame, y_train: pd.Series, k_best: int) -> dict:
    non_all_nan_cols = list(X_train.columns[~X_train.isna().all()])
    imputer = SimpleImputer(strategy="median")
    Xtr = imputer.fit_transform(X_train[non_all_nan_cols])

    vt = VarianceThreshold(threshold=1e-8)
    Xtr = vt.fit_transform(Xtr)
    vt_cols = list(np.array(non_all_nan_cols)[vt.get_support()])

    selector = None
    final_cols = vt_cols
    if k_best and k_best > 0 and Xtr.shape[1] > k_best:
        selector = SelectKBest(f_regression, k=min(k_best, Xtr.shape[1]))
        Xtr = selector.fit_transform(Xtr, y_train)
        final_cols = list(np.array(vt_cols)[selector.get_support()])

    return {
        "non_all_nan_cols": non_all_nan_cols,
        "imputer": imputer,
        "variance_threshold": vt,
        "selector": selector,
        "vt_cols": vt_cols,
        "final_cols": final_cols,
        "k_best": k_best,
    }


def transform_with_preprocessor(X: pd.DataFrame, pre: dict) -> np.ndarray:
    X_use = X.copy()
    for c in pre["non_all_nan_cols"]:
        if c not in X_use.columns:
            X_use[c] = np.nan
    X_use = X_use[pre["non_all_nan_cols"]]
    arr = pre["imputer"].transform(X_use)
    arr = pre["variance_threshold"].transform(arr)
    if pre.get("selector") is not None:
        arr = pre["selector"].transform(arr)
    return arr


def evaluate(y_true, y_pred) -> Dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def _make_stratify(y: pd.Series, bond: pd.Series) -> Optional[pd.Series]:
    """Use stratification only when every stratum has at least 2 samples."""
    # For all_types, P-S/P-H may have one sample, so normal bond stratification is impossible.
    vc = bond.value_counts()
    if len(vc) > 1 and int(vc.min()) >= 2:
        return bond.reset_index(drop=True)
    return None


def _bond_mean_table(y_values: pd.Series, bond_values: pd.Series) -> Tuple[Dict[str, float], float]:
    """Compute training-split bond means for residual target mode."""
    tmp = pd.DataFrame({"y": pd.Series(y_values).astype(float).reset_index(drop=True),
                        "bond": pd.Series(bond_values).apply(normalize_bond_type).reset_index(drop=True)})
    global_mean = float(tmp["y"].mean())
    means = tmp.groupby("bond")["y"].mean().to_dict()
    return {str(k): float(v) for k, v in means.items()}, global_mean


def _bond_mean_vector(bond_values: pd.Series, means: Dict[str, float], global_mean: float) -> np.ndarray:
    bonds = pd.Series(bond_values).apply(normalize_bond_type)
    return bonds.map(lambda b: means.get(str(b), global_mean)).astype(float).to_numpy()


def _infer_bond_from_feature_frame(X: pd.DataFrame) -> pd.Series:
    """Infer Bond_Type from one-hot bond features in X for prediction-time residual correction."""
    out = pd.Series(["Unknown"] * len(X), index=X.index, dtype="object")
    mapping = {
        "Bond_is_PC": "P-C", "Bond_is_PN": "P-N", "Bond_is_PO": "P-O",
        "Bond_is_PS": "P-S", "Bond_is_PH": "P-H",
    }
    for col, bond in mapping.items():
        if col in X.columns:
            mask = pd.to_numeric(X[col], errors="coerce").fillna(0).astype(float) > 0.5
            out.loc[mask] = bond
    return out


def _fit_predict_target_mode(model, X_train_sel: np.ndarray, y_train: pd.Series, X_test_sel: np.ndarray,
                             bond_train: pd.Series, bond_test: pd.Series, target_mode: str):
    """Fit one estimator in raw or bond-residual target mode and return predictions plus target metadata."""
    target_mode = target_mode.lower().strip()
    if target_mode == "bond_residual":
        means, global_mean = _bond_mean_table(y_train, bond_train)
        train_base = _bond_mean_vector(bond_train, means, global_mean)
        test_base = _bond_mean_vector(bond_test, means, global_mean)
        fitted = clone(model).fit(X_train_sel, y_train.to_numpy(dtype=float) - train_base)
        pred = fitted.predict(X_test_sel) + test_base
        return fitted, pred, means, global_mean
    fitted = clone(model).fit(X_train_sel, y_train)
    pred = fitted.predict(X_test_sel)
    return fitted, pred, {}, float(pd.Series(y_train).astype(float).mean())


def _iter_splits(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame, bond: pd.Series, task_name: str):
    """Yield train/test indices for the selected evaluation protocol."""
    mode = SPLIT_MODE
    n = len(y)
    if mode in {"repeated_kfold", "rkf", "repeated_cv"}:
        n_splits = max(2, min(N_SPLITS, n))
        rkf = RepeatedKFold(n_splits=n_splits, n_repeats=N_REPEATS, random_state=CV_RANDOM_STATE)
        for split_id, (tr, te) in enumerate(rkf.split(X, y), start=1):
            repeat = (split_id - 1) // n_splits + 1
            fold = (split_id - 1) % n_splits + 1
            yield tr, te, {"split": "repeated_kfold", "split_id": split_id, "repeat": repeat, "fold": fold, "seed": split_id}
        return

    if mode in {"kfold", "cv"}:
        n_splits = max(2, min(N_SPLITS, n))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=CV_RANDOM_STATE)
        for split_id, (tr, te) in enumerate(kf.split(X, y), start=1):
            yield tr, te, {"split": "kfold", "split_id": split_id, "repeat": 1, "fold": split_id, "seed": split_id}
        return

    if mode in {"group_kfold", "group"}:
        if GROUP_COL in meta.columns:
            groups = meta[GROUP_COL].astype(str).fillna("Unknown")
        elif "Canonical_SMILES" in meta.columns:
            groups = meta["Canonical_SMILES"].astype(str).fillna("Unknown")
        else:
            groups = pd.Series(range(n), index=meta.index).astype(str)
        n_groups = int(groups.nunique())
        n_splits = max(2, min(N_SPLITS, n_groups))
        gkf = GroupKFold(n_splits=n_splits)
        for split_id, (tr, te) in enumerate(gkf.split(X, y, groups=groups), start=1):
            yield tr, te, {"split": f"group_kfold:{GROUP_COL}", "split_id": split_id, "repeat": 1, "fold": split_id, "seed": split_id}
        return

    # V4/V6-compatible repeated random holdout.
    stratify = _make_stratify(y, bond)
    for split_id, seed in enumerate(REPEAT_SEEDS, start=1):
        split_kwargs = dict(test_size=TEST_SIZE, random_state=seed)
        if stratify is not None:
            split_kwargs["stratify"] = stratify
        indices = np.arange(n)
        tr, te = train_test_split(indices, **split_kwargs)
        yield np.asarray(tr), np.asarray(te), {
            "split": "random_holdout" if stratify is None else "stratified_random_holdout",
            "split_id": split_id, "repeat": 1, "fold": split_id, "seed": seed,
        }


def train_eval_random_splits(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame, bond: pd.Series, task_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate candidate models using the selected split protocol.

    V8 default: RepeatedKFold, which is more stable for this small BDE dataset
    than a few random holdout splits. The function name is kept for backward
    compatibility with older versions of the script.
    """
    all_results = []
    all_preds = []
    k_values = get_k_list_for_task(task_name)

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    meta = meta.reset_index(drop=True)
    bond = bond.reset_index(drop=True)

    for train_idx, test_idx, split_info in _iter_splits(X, y, meta, bond, task_name):
        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)
        meta_test = meta.iloc[test_idx].reset_index(drop=True)
        bond_train = bond.iloc[train_idx].reset_index(drop=True)
        bond_test = bond.iloc[test_idx].reset_index(drop=True)

        for k_best in k_values:
            pre = fit_preprocessor(X_train, y_train, k_best)
            X_train_sel = transform_with_preprocessor(X_train, pre)
            X_test_sel = transform_with_preprocessor(X_test, pre)
            split_best = None
            log(
                f"[RUN] {task_name} split={split_info['split']} repeat={split_info['repeat']} "
                f"fold={split_info['fold']} k={k_best} train={len(y_train)} test={len(y_test)} "
                f"features={X_train_sel.shape[1]}",
                verbose_only=not VERBOSE,
            )

            for name, model in model_pool(int(split_info.get("seed", 42)), task_name).items():
                for target_mode in TARGET_MODES:
                    try:
                        fitted, pred, means, global_mean = _fit_predict_target_mode(
                            model, X_train_sel, y_train, X_test_sel, bond_train, bond_test, target_mode
                        )
                        test_m = evaluate(y_test, pred)
                        row = {
                            "task": task_name,
                            "seed": int(split_info.get("seed", 0)),
                            "split_id": int(split_info.get("split_id", 0)),
                            "repeat": int(split_info.get("repeat", 1)),
                            "fold": int(split_info.get("fold", 1)),
                            "k_best": k_best,
                            "model": name,
                            "target_mode": target_mode,
                            "model_variant": f"{name}__{target_mode}",
                            "test_R2": test_m["R2"],
                            "test_RMSE": test_m["RMSE"],
                            "test_MAE": test_m["MAE"],
                            "n_samples": len(y),
                            "test_size": TEST_SIZE if str(split_info.get("split", "")).endswith("holdout") else np.nan,
                            "selected_features": X_train_sel.shape[1],
                            "split": split_info.get("split", SPLIT_MODE),
                        }
                        all_results.append(row)
                        if split_best is None or row["test_R2"] > split_best["test_R2"]:
                            split_best = row

                        pred_df = meta_test.copy()
                        pred_df["task"] = task_name
                        pred_df["seed"] = row["seed"]
                        pred_df["split_id"] = row["split_id"]
                        pred_df["repeat"] = row["repeat"]
                        pred_df["fold"] = row["fold"]
                        pred_df["split"] = row["split"]
                        pred_df["k_best"] = k_best
                        pred_df["model"] = name
                        pred_df["target_mode"] = target_mode
                        pred_df["model_variant"] = row["model_variant"]
                        pred_df["BDE_true_kJ_mol"] = y_test.reset_index(drop=True)
                        pred_df["BDE_pred_kJ_mol"] = pred
                        pred_df["abs_error_kJ_mol"] = np.abs(pred_df["BDE_true_kJ_mol"] - pred_df["BDE_pred_kJ_mol"])
                        all_preds.append(pred_df)
                        if PRINT_EACH_MODEL:
                            log(f"    {name}/{target_mode}: R2={test_m['R2']:.4f}, RMSE={test_m['RMSE']:.2f}, MAE={test_m['MAE']:.2f}")
                    except Exception as e:
                        log(f"[WARN] task={task_name} model={name} target_mode={target_mode} failed at split={split_info.get('split_id')} k={k_best}: {e}")

            if split_best is not None and not PRINT_EACH_MODEL:
                log(
                    f"    best={split_best['model']}/{split_best.get('target_mode','raw')} "
                    f"R2={split_best['test_R2']:.4f} RMSE={split_best['test_RMSE']:.2f} MAE={split_best['test_MAE']:.2f}",
                    verbose_only=not VERBOSE,
                )

    return pd.DataFrame(all_results), pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()

def summary_from_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    group_cols = ["task", "model", "target_mode", "model_variant", "k_best"]
    for c in group_cols:
        if c not in results.columns:
            results[c] = "raw" if c == "target_mode" else results.get("model", "model")
    return results.groupby(group_cols, as_index=False).agg(
        test_R2_mean=("test_R2", "mean"),
        test_R2_std=("test_R2", "std"),
        test_R2_max=("test_R2", "max"),
        test_R2_min=("test_R2", "min"),
        test_RMSE_mean=("test_RMSE", "mean"),
        test_MAE_mean=("test_MAE", "mean"),
        n_runs=("split_id", "count"),
        n_samples=("n_samples", "first"),
        selected_features=("selected_features", "first"),
    ).sort_values(["task", "test_R2_mean", "test_RMSE_mean"], ascending=[True, False, True])



def save_and_print_target_mode_breakdown(summary: pd.DataFrame, task_name: str, outdir: Path, top_n: int = SUMMARY_TOP_N) -> None:
    """Save and print per-target-mode summaries so raw/bond_residual are easy to compare."""
    sub_all = summary[summary["task"] == task_name].copy()
    if sub_all.empty or "target_mode" not in sub_all.columns:
        return

    best_rows = []
    print(f"\n[TARGET MODE CHECK] {task_name}")
    for target_mode in sorted(sub_all["target_mode"].dropna().astype(str).unique()):
        sub = sub_all[sub_all["target_mode"].astype(str) == target_mode].copy()
        if sub.empty:
            continue
        sub.to_csv(outdir / f"BDE_{task_name}_summary_{target_mode}.csv", index=False, encoding="utf-8-sig")
        best = sub.sort_values(["test_R2_mean", "test_RMSE_mean"], ascending=[False, True]).iloc[0]
        best_rows.append(best.to_dict())
        print(
            f"  - {target_mode}: model={best['model']}, k={int(best['k_best'])}, "
            f"mean_R2={float(best['test_R2_mean']):.4f}, std_R2={float(best['test_R2_std']):.4f}, "
            f"MAE={float(best['test_MAE_mean']):.2f}"
        )
    if best_rows:
        pd.DataFrame(best_rows).to_csv(outdir / f"BDE_{task_name}_target_mode_best_rows.csv", index=False, encoding="utf-8-sig")


def select_best_summary_row(summary: pd.DataFrame, task_name: str) -> pd.Series:
    """Select the practical best model for final prediction.

    v10.3-patched default: select the highest mean R2 so the final saved model is
    consistent with the first row of the summary table.

    Optional: set DOPO_BDE_BEST_SELECTION_MODE=r2_mae to use the older practical
    rule that chooses lower MAE among near-best R2 models.
    """
    sub = summary[summary["task"] == task_name].copy()
    if sub.empty:
        raise ValueError(f"No summary rows found for task={task_name}")
    if BEST_SELECTION_MODE in {"strict_r2", "r2_only"}:
        return sub.sort_values(["test_R2_mean", "test_RMSE_mean"], ascending=[False, True]).iloc[0]

    best_r2 = float(sub["test_R2_mean"].max())
    near = sub[sub["test_R2_mean"] >= best_r2 - BEST_R2_TOL].copy()
    near = near.sort_values(
        ["test_MAE_mean", "test_RMSE_mean", "test_R2_std", "test_R2_mean"],
        ascending=[True, True, True, False],
    )
    return near.iloc[0]


def best_selection_note(best: pd.Series, summary: pd.DataFrame, task_name: str) -> str:
    sub = summary[summary["task"] == task_name].copy()
    strict = sub.sort_values(["test_R2_mean", "test_RMSE_mean"], ascending=[False, True]).iloc[0]
    if BEST_SELECTION_MODE in {"strict_r2", "r2_only"}:
        return "strict_r2: selected by highest mean R2."
    if str(best["model"]) == str(strict["model"]) and str(best.get("target_mode", "raw")) == str(strict.get("target_mode", "raw")) and int(best["k_best"]) == int(strict["k_best"]):
        return f"r2_mae: selected by highest mean R2; no lower-MAE model within R2 tolerance {BEST_R2_TOL}."
    return (
        f"r2_mae: selected lower-MAE model within R2 tolerance {BEST_R2_TOL}. "
        f"Strict-R2 best was {strict['model']}/{strict.get('target_mode', 'raw')} k={int(strict['k_best'])} "
        f"R2={float(strict['test_R2_mean']):.4f}, MAE={float(strict['test_MAE_mean']):.2f}."
    )


def _target_baseline_from_X(X: pd.DataFrame, bond_means: Dict[str, float], global_mean: float) -> np.ndarray:
    inferred = _infer_bond_from_feature_frame(X)
    return _bond_mean_vector(inferred, bond_means, global_mean)


def train_final_model(X: pd.DataFrame, y: pd.Series, model_name: str, k_best: int, task_name: str,
                      bond: Optional[pd.Series] = None, target_mode: str = "raw",
                      seed: int = FINAL_MODEL_SEED) -> dict:
    pre = fit_preprocessor(X, y, k_best)
    X_arr = transform_with_preprocessor(X, pre)

    target_mode = (target_mode or "raw").lower().strip()
    if bond is None:
        bond = _infer_bond_from_feature_frame(X)
    bond_means, global_mean = _bond_mean_table(y, bond)
    train_base = _bond_mean_vector(bond, bond_means, global_mean) if target_mode == "bond_residual" else np.zeros(len(y), dtype=float)
    y_fit = y.to_numpy(dtype=float) - train_base if target_mode == "bond_residual" else y

    model_seeds = FINAL_ENSEMBLE_SEEDS if FINAL_ENSEMBLE else [seed]
    fitted_models = []
    for model_seed in model_seeds:
        fitted_models.append(clone(model_pool(model_seed, task_name)[model_name]).fit(X_arr, y_fit))

    return {
        "preprocessor": pre,
        "model": fitted_models[0],                 # backward-compatible single model
        "models": fitted_models,                   # final-seed ensemble
        "model_name": model_name,
        "target_mode": target_mode,
        "bond_means": bond_means,
        "global_mean": global_mean,
        "task": task_name,
        "k_best": k_best,
        "unit": "kJ/mol",
        "feature_columns": list(X.columns),
        "fp_bits": FP_BITS,
        "whole_fp_radii": WHOLE_FP_RADII,
        "radius": RADIUS,
        "target_fp_bits": TARGET_FP_BITS,
        "target_fp_radii": TARGET_FP_RADII,
        "fragment_bits": FRAGMENT_BITS,
        "atompair_bits": ATOMPAIR_BITS if USE_ATOMPAIR_FP else 0,
        "torsion_bits": TORSION_BITS if USE_TORSION_FP else 0,
        "pattern_bits": PATTERN_BITS if USE_PATTERN_FP else 0,
        "feature_mode": FEATURE_MODE,
        "target_modes": TARGET_MODES,
        "enable_seed_diagnostics": ENABLE_SEED_DIAGNOSTICS,
        "target_R2_goal": R2_GOAL,
        "supported_bond_types": SUPPORTED_BOND_TYPES,
        "final_ensemble": bool(FINAL_ENSEMBLE),
        "final_ensemble_seeds": model_seeds,
        "best_selection_mode": BEST_SELECTION_MODE,
        "best_r2_tol": BEST_R2_TOL,
    }


def predict_with_pack(model_pack: dict, X_new: pd.DataFrame) -> np.ndarray:
    mean, _ = predict_with_pack_detail(model_pack, X_new)
    return mean


def predict_with_pack_detail(model_pack: dict, X_new: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Return mean and std prediction for a saved pack.

    The std reflects final-ensemble seed variance, not experimental uncertainty.
    For target_mode=bond_residual, bond means learned from the training data are
    added back after predicting residual BDE.
    """
    X_arr = transform_with_preprocessor(X_new, model_pack["preprocessor"])
    models = model_pack.get("models")
    if models:
        preds = np.vstack([m.predict(X_arr) for m in models])
    else:
        preds = np.vstack([model_pack["model"].predict(X_arr)])

    if model_pack.get("target_mode", "raw") == "bond_residual":
        base = _target_baseline_from_X(
            X_new,
            model_pack.get("bond_means", {}),
            float(model_pack.get("global_mean", 0.0)),
        )
        preds = preds + base.reshape(1, -1)
    return preds.mean(axis=0), preds.std(axis=0)


# -----------------------------------------------------------------------------
# Training all tasks
# -----------------------------------------------------------------------------
def train_one_task(data: pd.DataFrame, task_name: str) -> Tuple[Optional[dict], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_data, excluded = task_filter(data, task_name)
    task_dir = OUTDIR / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    model_data.to_csv(task_dir / f"DOPO_BDE_{task_name}_model_rows.csv", index=False, encoding="utf-8-sig")
    excluded.to_csv(task_dir / f"DOPO_BDE_{task_name}_excluded_rows.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print(f"[TASK] {task_name}")
    print("=" * 78)
    print(f"[INFO] task={task_name} n={len(model_data)} distribution={model_data['Bond_Type'].value_counts().to_dict() if len(model_data) else {}}")
    if len(model_data) < 20:
        print(f"[WARN] task={task_name} skipped because usable rows < 20")
        return None, pd.DataFrame(), pd.DataFrame(), model_data

    print(f"[INFO] Building features for {task_name} ...")
    X = build_features(model_data, "Smiles", "Bond_Type")
    y = model_data["BDE_kJ_mol"].astype(float).reset_index(drop=True)
    bond = model_data["Bond_Type"].reset_index(drop=True)
    print(f"[INFO] task={task_name} X shape={X.shape}")

    meta_cols = [
        "ID", "Smiles", "Canonical_SMILES", "Bond_Type", "Linkage_Type", "Target_Bond_Label_clean",
        "Target_Bond_Subtype", "Target_X_atom_class", "Target_Bond_Count", "Potential_PX_Bond_Count", "Model_Use",
    ]
    meta_cols = [c for c in meta_cols if c in model_data.columns]
    meta = model_data[meta_cols].copy().reset_index(drop=True)

    results, preds = train_eval_random_splits(X, y, meta, bond, task_name)
    results.to_csv(task_dir / f"BDE_{task_name}_all_results.csv", index=False, encoding="utf-8-sig")
    preds.to_csv(task_dir / f"BDE_{task_name}_all_predictions.csv", index=False, encoding="utf-8-sig")
    summary = summary_from_results(results)
    summary.to_csv(task_dir / f"BDE_{task_name}_summary.csv", index=False, encoding="utf-8-sig")
    save_and_print_target_mode_breakdown(summary, task_name, task_dir)

    if summary.empty:
        print(f"[WARN] task={task_name} no successful model")
        return None, results, preds, model_data

    # V4 best selection: R2 first, then MAE when R2 is essentially tied.
    best = select_best_summary_row(summary, task_name)
    best_model = str(best["model"])
    best_target_mode = str(best.get("target_mode", "raw"))
    best_k = int(best["k_best"])
    selection_note = best_selection_note(best, summary, task_name)
    print(f"\n[TASK SUMMARY] {task_name} top {SUMMARY_TOP_N}")
    print(summary[summary["task"] == task_name].head(SUMMARY_TOP_N).to_string(index=False))
    print(
        f"\n[BEST {task_name}] model={best_model}, target_mode={best_target_mode}, k={best_k}, "
        f"mean_R2={float(best['test_R2_mean']):.4f}, std_R2={float(best['test_R2_std']):.4f}, "
        f"MAE={float(best['test_MAE_mean']):.2f}, max_seed_R2={float(best['test_R2_max']):.4f}"
    )
    print(f"[BEST RULE] {selection_note}")

    pack = train_final_model(X, y, best_model, best_k, task_name=task_name, bond=bond, target_mode=best_target_mode, seed=FINAL_MODEL_SEED)
    pack["best_cv_metrics"] = {k: (float(v) if isinstance(v, (int, float, np.number)) else v) for k, v in best.to_dict().items()}
    pack["best_selection_note"] = selection_note
    pack_path = task_dir / f"BDE_best_model_{task_name}.joblib"
    joblib.dump(pack, pack_path)
    # Also save a flat compatibility copy in OUTDIR.
    joblib.dump(pack, OUTDIR / f"BDE_best_model_{task_name}.joblib")
    print(f"[INFO] Saved final model: {pack_path}")

    if not preds.empty:
        for group_col in ["Bond_Type", "Target_Bond_Subtype", "Target_X_atom_class"]:
            if group_col in preds.columns:
                err = preds.groupby(["task", "model", "k_best", group_col], as_index=False).agg(
                    n=("abs_error_kJ_mol", "count"),
                    MAE=("abs_error_kJ_mol", "mean"),
                    MedAE=("abs_error_kJ_mol", "median"),
                ).sort_values(["task", "model", "k_best", "MAE"], ascending=[True, True, True, False])
                err.to_csv(task_dir / f"BDE_{task_name}_error_by_{group_col}.csv", index=False, encoding="utf-8-sig")

    return pack, results, preds, model_data



def _sample_key_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in ["ID", "Canonical_SMILES", "Smiles", "Bond_Type", "Target_Bond_Label_clean", "Target_Bond_Subtype", "Target_X_atom_class"]:
        if c in df.columns:
            cols.append(c)
    if "ID" in cols:
        # ID plus chemistry labels is usually enough and avoids grouping unrelated duplicate IDs across tasks.
        return [c for c in ["ID", "Canonical_SMILES", "Bond_Type", "Target_Bond_Label_clean", "Target_Bond_Subtype", "Target_X_atom_class"] if c in df.columns]
    return cols or ["Bond_Type"]


def _pooled_metrics(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty or df["BDE_true_kJ_mol"].nunique() < 2:
        return {"pooled_R2": np.nan, "pooled_RMSE": np.nan, "pooled_MAE": np.nan, "n_pred_rows": int(len(df))}
    y_true = pd.to_numeric(df["BDE_true_kJ_mol"], errors="coerce")
    y_pred = pd.to_numeric(df["BDE_pred_kJ_mol"], errors="coerce")
    mask = y_true.notna() & y_pred.notna()
    if mask.sum() < 2 or y_true[mask].nunique() < 2:
        return {"pooled_R2": np.nan, "pooled_RMSE": np.nan, "pooled_MAE": np.nan, "n_pred_rows": int(mask.sum())}
    return {
        "pooled_R2": float(r2_score(y_true[mask], y_pred[mask])),
        "pooled_RMSE": float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))),
        "pooled_MAE": float(mean_absolute_error(y_true[mask], y_pred[mask])),
        "n_pred_rows": int(mask.sum()),
    }


def generate_error_diagnostics(preds: pd.DataFrame, combined_summary: pd.DataFrame, outdir: Path) -> None:
    """Write V7 diagnostic files for manual data checking.

    These diagnostics never remove samples from training. They only identify
    high-error rows that should be checked against the original literature.
    """
    if preds.empty or combined_summary.empty:
        return

    diag_dir = outdir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    selected_rows = []
    selected_pred_blocks = []
    outlier_blocks = []
    manual_blocks = []
    effect_rows = []
    error_group_rows = []

    for task in sorted(combined_summary["task"].dropna().unique()):
        try:
            best = select_best_summary_row(combined_summary, str(task))
        except Exception:
            continue
        model = str(best["model"])
        target_mode = str(best.get("target_mode", "raw"))
        k_best = int(best["k_best"])
        selected_rows.append(best.to_dict())

        sub = preds[
            (preds["task"].astype(str) == str(task)) &
            (preds["model"].astype(str) == model) &
            (preds["target_mode"].astype(str) == target_mode) &
            (pd.to_numeric(preds["k_best"], errors="coerce") == k_best)
        ].copy()
        if sub.empty:
            continue
        sub["selected_model"] = model
        sub["selected_target_mode"] = target_mode
        sub["selected_k_best"] = k_best
        selected_pred_blocks.append(sub)

        key_cols = _sample_key_columns(sub)
        agg_dict = {
            "appeared_in_test_n": ("abs_error_kJ_mol", "count"),
            "BDE_true_mean_kJ_mol": ("BDE_true_kJ_mol", "mean"),
            "BDE_pred_mean_kJ_mol": ("BDE_pred_kJ_mol", "mean"),
            "abs_error_mean_kJ_mol": ("abs_error_kJ_mol", "mean"),
            "abs_error_max_kJ_mol": ("abs_error_kJ_mol", "max"),
            "abs_error_median_kJ_mol": ("abs_error_kJ_mol", "median"),
        }
        outliers = sub.groupby(key_cols, as_index=False).agg(**agg_dict)
        outliers.insert(0, "task", task)
        outliers["selected_model"] = model
        outliers["selected_target_mode"] = target_mode
        outliers["selected_k_best"] = k_best
        outliers = outliers.sort_values("abs_error_mean_kJ_mol", ascending=False).reset_index(drop=True)
        outliers["error_rank_in_task"] = np.arange(1, len(outliers) + 1)
        outlier_blocks.append(outliers.head(OUTLIER_TOP_N))

        manual = outliers[(outliers["abs_error_mean_kJ_mol"] >= OUTLIER_ABS_THRESHOLD) | (outliers["error_rank_in_task"] <= min(OUTLIER_TOP_N, 20))].copy()
        manual["manual_check_reason"] = np.where(
            manual["abs_error_mean_kJ_mol"] >= OUTLIER_ABS_THRESHOLD,
            f"mean_abs_error >= {OUTLIER_ABS_THRESHOLD} kJ/mol",
            f"top-{min(OUTLIER_TOP_N, 20)} error sample in task",
        )
        manual_blocks.append(manual)

        # Error by available chemistry labels for selected config.
        for group_col in ["Bond_Type", "Target_Bond_Label_clean", "Target_Bond_Subtype", "Target_X_atom_class"]:
            if group_col in sub.columns:
                g = sub.groupby(group_col, as_index=False).agg(
                    n_pred_rows=("abs_error_kJ_mol", "count"),
                    MAE=("abs_error_kJ_mol", "mean"),
                    MedAE=("abs_error_kJ_mol", "median"),
                    MaxAE=("abs_error_kJ_mol", "max"),
                )
                g.insert(0, "task", task)
                g.insert(1, "group_col", group_col)
                g["selected_model"] = model
                g["selected_target_mode"] = target_mode
                g["selected_k_best"] = k_best
                error_group_rows.append(g)

        # Diagnostic only: pooled metric after removing top-N unique high-error samples.
        base_metrics = _pooled_metrics(sub)
        top_keys = outliers[key_cols].copy()
        for n_remove in OUTLIER_EFFECT_STEPS:
            n_remove = int(n_remove)
            if n_remove <= 0:
                filtered = sub.copy()
            else:
                remove_keys = set(map(tuple, top_keys.head(n_remove).astype(str).to_numpy()))
                row_keys = list(map(tuple, sub[key_cols].astype(str).to_numpy()))
                keep_mask = [rk not in remove_keys for rk in row_keys]
                filtered = sub.loc[keep_mask].copy()
            m = _pooled_metrics(filtered)
            m.update({
                "task": task,
                "selected_model": model,
                "selected_target_mode": target_mode,
                "selected_k_best": k_best,
                "removed_top_unique_error_samples": n_remove,
                "diagnostic_only": True,
                "note": "Do not report this as final performance unless removed samples are independently confirmed as data/label errors.",
            })
            effect_rows.append(m)

    if selected_rows:
        pd.DataFrame(selected_rows).to_csv(diag_dir / "BDE_selected_configs_for_diagnostics.csv", index=False, encoding="utf-8-sig")
    if selected_pred_blocks:
        pd.concat(selected_pred_blocks, ignore_index=True).to_csv(diag_dir / "BDE_selected_config_all_predictions.csv", index=False, encoding="utf-8-sig")
    if outlier_blocks:
        pd.concat(outlier_blocks, ignore_index=True).to_csv(diag_dir / "BDE_top_outliers_by_config.csv", index=False, encoding="utf-8-sig")
    if manual_blocks:
        pd.concat(manual_blocks, ignore_index=True).to_csv(diag_dir / "BDE_manual_check_candidates.csv", index=False, encoding="utf-8-sig")
    if error_group_rows:
        pd.concat(error_group_rows, ignore_index=True).to_csv(diag_dir / "BDE_error_by_group_selected_config.csv", index=False, encoding="utf-8-sig")
    if effect_rows:
        pd.DataFrame(effect_rows).to_csv(diag_dir / "BDE_outlier_removal_effect_diagnostic.csv", index=False, encoding="utf-8-sig")

    print("\n[ERROR DIAGNOSTICS] files saved in diagnostics/:")
    print("  - BDE_selected_configs_for_diagnostics.csv")
    print("  - BDE_selected_config_all_predictions.csv")
    print("  - BDE_top_outliers_by_config.csv")
    print("  - BDE_manual_check_candidates.csv")
    print("  - BDE_error_by_group_selected_config.csv")
    print("  - BDE_outlier_removal_effect_diagnostic.csv")

def train_all_tasks(tasks: List[str]) -> Dict[str, dict]:
    data = load_clean_training_data(DATA_PATH)
    conflict = (
        data.groupby(["Canonical_SMILES", "Bond_Type"])["BDE_kJ_mol"]
        .agg(["count", "nunique", "min", "max", "std"])
        .reset_index()
    )
    conflict = conflict[conflict["nunique"] > 1].copy()
    conflict.to_csv(OUTDIR / "BDE_conflict_same_structure_bond.csv", index=False, encoding="utf-8-sig")
    print(f"[CHECK] same Canonical_SMILES + Bond_Type with multiple BDE: {len(conflict)} groups")

    packs: Dict[str, dict] = {}
    all_results = []
    all_preds = []
    task_stats = {}
    for task in tasks:
        task = "pc_pn" if task.lower() in {"pcpn", "pc_pn_classified"} else task.lower()
        pack, results, preds, model_data = train_one_task(data, task)
        if pack is not None:
            packs[task] = pack
        if not results.empty:
            all_results.append(results)
        if not preds.empty:
            all_preds.append(preds)
        task_stats[task] = {
            "n_samples": int(len(model_data)),
            "distribution": model_data["Bond_Type"].value_counts().to_dict() if len(model_data) else {},
            "bde_min": float(model_data["BDE_kJ_mol"].min()) if len(model_data) else None,
            "bde_max": float(model_data["BDE_kJ_mol"].max()) if len(model_data) else None,
        }

    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_results.to_csv(OUTDIR / "BDE_all_tasks_all_results.csv", index=False, encoding="utf-8-sig")
        combined_summary = summary_from_results(combined_results)
        combined_summary.to_csv(OUTDIR / "BDE_all_tasks_summary.csv", index=False, encoding="utf-8-sig")
        if "target_mode" in combined_summary.columns:
            combined_summary.to_csv(OUTDIR / "BDE_all_tasks_summary_by_target_mode.csv", index=False, encoding="utf-8-sig")
        print("\n" + "=" * 78)
        print("[COMBINED SUMMARY] Top rows per task")
        print("=" * 78)
        for task in tasks:
            tt = "pc_pn" if task.lower() in {"pcpn", "pc_pn_classified"} else task.lower()
            sub = combined_summary[combined_summary["task"] == tt].head(SUMMARY_TOP_N)
            if not sub.empty:
                print(f"\n[{tt}]")
                print(sub.to_string(index=False))
                best_mean = float(sub.iloc[0]["test_R2_mean"])
                if best_mean >= R2_GOAL:
                    print(f"[TARGET CHECK] {tt}: best mean R2={best_mean:.4f} >= target {R2_GOAL:.2f}.")
                else:
                    print(f"[TARGET CHECK] {tt}: best mean R2={best_mean:.4f} < optional target {R2_GOAL:.2f}; current pc_pn data did not reach the optional target under leakage-safe repeated-KFold. This is not a code failure; report the value honestly or improve/verify data.")

                if "target_mode" in combined_summary.columns:
                    print(f"[TARGET MODE BEST] {tt}:")
                    mode_best_rows = []
                    for target_mode in sorted(combined_summary.loc[combined_summary["task"] == tt, "target_mode"].dropna().astype(str).unique()):
                        mode_sub = combined_summary[(combined_summary["task"] == tt) & (combined_summary["target_mode"].astype(str) == target_mode)].copy()
                        if mode_sub.empty:
                            continue
                        mode_best = mode_sub.sort_values(["test_R2_mean", "test_RMSE_mean"], ascending=[False, True]).iloc[0]
                        mode_best_rows.append(mode_best.to_dict())
                        print(
                            f"  - {target_mode}: model={mode_best['model']}, k={int(mode_best['k_best'])}, "
                            f"mean_R2={float(mode_best['test_R2_mean']):.4f}, std_R2={float(mode_best['test_R2_std']):.4f}, "
                            f"MAE={float(mode_best['test_MAE_mean']):.2f}"
                        )
                    if mode_best_rows:
                        pd.DataFrame(mode_best_rows).to_csv(OUTDIR / f"BDE_{tt}_target_mode_best_rows.csv", index=False, encoding="utf-8-sig")

        if ENABLE_SEED_DIAGNOSTICS:
            seed_diag = (
                combined_results.sort_values(["task", "seed", "test_R2"], ascending=[True, True, False])
                .groupby(["task", "seed"], as_index=False)
                .head(1)
                .sort_values(["task", "seed"])
            )
            seed_diag.to_csv(OUTDIR / "BDE_seed_best_diagnostics.csv", index=False, encoding="utf-8-sig")
            seed_stats = seed_diag.groupby("task", as_index=False).agg(
                seed_best_R2_mean=("test_R2", "mean"),
                seed_best_R2_std=("test_R2", "std"),
                seed_best_R2_min=("test_R2", "min"),
                seed_best_R2_max=("test_R2", "max"),
                seed_best_MAE_mean=("test_MAE", "mean"),
            )
            seed_stats.to_csv(OUTDIR / "BDE_seed_best_diagnostics_summary.csv", index=False, encoding="utf-8-sig")
            print("\n[SEED DIAGNOSTICS] best row per seed saved:")
            print("  - BDE_seed_best_diagnostics.csv")
            print("  - BDE_seed_best_diagnostics_summary.csv")

    combined_preds = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    if all_preds:
        combined_preds.to_csv(OUTDIR / "BDE_all_tasks_all_predictions.csv", index=False, encoding="utf-8-sig")
        if ENABLE_ERROR_DIAGNOSTICS and all_results:
            try:
                generate_error_diagnostics(combined_preds, summary_from_results(pd.concat(all_results, ignore_index=True)), OUTDIR)
            except Exception as e:
                print(f"[WARN] error diagnostics failed: {e}")

    run_summary = {
        "data_path": str(DATA_PATH),
        "outdir": str(OUTDIR),
        "tasks": tasks,
        "task_stats": task_stats,
        "seeds": REPEAT_SEEDS,
        "test_size": TEST_SIZE,
        "split_mode": SPLIT_MODE,
        "n_splits": N_SPLITS,
        "n_repeats": N_REPEATS,
        "cv_random_state": CV_RANDOM_STATE,
        "group_col": GROUP_COL,
        "global_k_list": K_LIST,
        "task_k_lists": task_k_lists(tasks),
        "final_model_seed": FINAL_MODEL_SEED,
        "best_selection_mode": BEST_SELECTION_MODE,
        "best_r2_tol": BEST_R2_TOL,
        "final_ensemble": FINAL_ENSEMBLE,
        "final_ensemble_seeds": FINAL_ENSEMBLE_SEEDS,
        "fp_bits": FP_BITS,
        "whole_fp_radii": WHOLE_FP_RADII,
        "radius": RADIUS,
        "target_fp_bits": TARGET_FP_BITS,
        "target_fp_radii": TARGET_FP_RADII,
        "fragment_bits": FRAGMENT_BITS,
        "atompair_bits": ATOMPAIR_BITS if USE_ATOMPAIR_FP else 0,
        "torsion_bits": TORSION_BITS if USE_TORSION_FP else 0,
        "pattern_bits": PATTERN_BITS if USE_PATTERN_FP else 0,
        "feature_mode": FEATURE_MODE,
        "target_modes": TARGET_MODES,
        "enable_seed_diagnostics": ENABLE_SEED_DIAGNOSTICS,
        "target_R2_goal": R2_GOAL,
        "enable_error_diagnostics": ENABLE_ERROR_DIAGNOSTICS,
        "outlier_top_n": OUTLIER_TOP_N,
        "outlier_abs_threshold": OUTLIER_ABS_THRESHOLD,
        "outlier_effect_steps": OUTLIER_EFFECT_STEPS,
        "use_xgb": USE_XGB,
        "fast_mode": FAST_MODE,
        "slow_tuning": SLOW_TUNING,
        "allow_y_trim": ALLOW_Y_TRIM,
        "y_max_if_trim": Y_MAX_IF_TRIM,
        "note": "V10.3 final pc_pn version: default pc_pn + raw target + focused K search + v9 feature blocks + v7 repeated-KFold diagnostics. Use v9 only as a tuning/reference script; use this file for thesis-use BDE feature generation.",
    }
    with open(OUTDIR / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)
    return packs

# -----------------------------------------------------------------------------
# Input / output module
# -----------------------------------------------------------------------------
PREDICTION_REQUIRED_COLUMNS = ["Smiles", "Bond_Type"]
PREDICTION_OPTIONAL_COLUMNS = [
    "Sample_ID", "Linkage_Type", "Target_Bond_Label_clean", "Target_Bond_Subtype",
    "Target_X_atom_class", "Target_Bond_Count", "Potential_PX_Bond_Count", "Notes",
]
PREDICTION_OUTPUT_COLUMNS = [
    "Smiles", "Canonical_SMILES", "Bond_Type", "selected_model_task",
    "BDE_pred_kJ_mol", "BDE_pred_std_kJ_mol", "prediction_agreement_note", "confidence_note",
    "BDE_pred_pc_pn_kJ_mol", "BDE_pred_pc_pn_std_kJ_mol",
    "BDE_pred_all_types_kJ_mol", "BDE_pred_all_types_std_kJ_mol",
    "pcpn_alltypes_abs_diff_kJ_mol",
]


def make_prediction_template(out_path: Path) -> Path:
    """Create a CSV template for unseen-SMILES BDE prediction."""
    template = pd.DataFrame([
        {
            "Sample_ID": "example_1",
            "Smiles": "O=P1(Oc2ccccc2-c2ccccc21)C...",
            "Bond_Type": "P-C",
            "Linkage_Type": "DOPO-C",
            "Target_Bond_Label_clean": "DOPO_P-C",
            "Target_Bond_Subtype": "",
            "Target_X_atom_class": "C",
            "Target_Bond_Count": "",
            "Potential_PX_Bond_Count": "",
            "Notes": "Replace this row with your real molecule. Bond_Type must be P-C/P-N/P-O/P-S/P-H.",
        },
        {
            "Sample_ID": "example_2",
            "Smiles": "",
            "Bond_Type": "P-N",
            "Linkage_Type": "DOPO-N",
            "Target_Bond_Label_clean": "DOPO_P-N",
            "Target_Bond_Subtype": "",
            "Target_X_atom_class": "N",
            "Target_Bond_Count": "",
            "Potential_PX_Bond_Count": "",
            "Notes": "For prediction, at least Smiles and Bond_Type are required.",
        },
    ])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved prediction input template: {out_path}")
    return out_path


def autodetect_column(df: pd.DataFrame, candidates: List[str], role: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    raise KeyError(f"Cannot find {role} column. Candidates={candidates}; available={list(df.columns)}")


def standardize_prediction_input(
    df: pd.DataFrame,
    smiles_col: Optional[str] = None,
    bond_col: Optional[str] = None,
) -> pd.DataFrame:
    """Convert a user prediction table to the internal input schema."""
    smiles_col = smiles_col or autodetect_column(df, ["Smiles", "SMILES", "SMILES_main", "smiles"], "SMILES")
    bond_col = bond_col or autodetect_column(df, ["Bond_Type", "BDE_Type", "Target_Bond_Label_clean", "Target_Bond_Label", "Bond type", "bond_type"], "bond type")

    out = pd.DataFrame(index=df.index)
    out["Smiles"] = df[smiles_col].apply(clean_text)
    out["Bond_Type"] = df[bond_col].apply(normalize_bond_type)
    out["Canonical_SMILES"] = out["Smiles"].apply(canonical_smiles)

    for col in PREDICTION_OPTIONAL_COLUMNS:
        if col in df.columns:
            out[col] = df[col]

    out = ensure_classified_columns(out)

    invalid_smiles = out["Canonical_SMILES"].astype(str).str.len().eq(0)
    invalid_bond = ~out["Bond_Type"].isin(SUPPORTED_BOND_TYPES)
    out["input_valid"] = ~(invalid_smiles | invalid_bond)
    out["input_warning"] = ""
    out.loc[invalid_smiles, "input_warning"] += "Invalid or empty SMILES. "
    out.loc[invalid_bond, "input_warning"] += "Unsupported Bond_Type; use P-C/P-N/P-O/P-S/P-H. "
    return out


def write_io_guide(outdir: Path = OUTDIR) -> Path:
    guide = """BDE prediction input/output guide
=================================

Purpose:
  Train on DOPO_BDE.csv and predict BDE(kJ/mol) for unseen DOPO derivative SMILES.
  Default thesis-use mode is DOPO P-C/P-N only: pc_pn.

Required prediction input columns:
  Smiles      Molecular SMILES string.
  Bond_Type   Target bond type: P-C or P-N for the default pc_pn model.

Recommended optional columns:
  Sample_ID, Linkage_Type, Target_Bond_Label_clean, Target_Bond_Subtype,
  Target_X_atom_class, Target_Bond_Count, Potential_PX_Bond_Count, Notes.

Model selection rule:
  P-C/P-N        -> pc_pn model first.
  P-O/P-S/P-H    -> only available if you explicitly train all_types with DOPO_BDE_TASKS=all_types,pc_pn; otherwise no suitable model is returned.

Main output columns:
  BDE_pred_kJ_mol             Final selected BDE prediction.
  selected_model_task         Model used for the final prediction.
  confidence_note             Reliability warning.
  BDE_pred_pc_pn_kJ_mol       Prediction from pc_pn if available.
  BDE_pred_all_types_kJ_mol   Prediction from all_types if explicitly trained.

Typical commands:
  python BDE.py
  python BDE.py --make-template BDE_prediction_input_template.csv --skip-train
  python BDE.py --skip-train --predict-smiles "SMILES_HERE" --bond-type P-C
  python BDE.py --skip-train --predict-csv input.csv --smiles-col Smiles --bond-col Bond_Type
"""
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "BDE_prediction_io_guide.txt"
    path.write_text(guide, encoding="utf-8")
    return path

# -----------------------------------------------------------------------------
# Prediction from SMILES / CSV
# -----------------------------------------------------------------------------
def load_saved_packs(outdir: Path = OUTDIR) -> Dict[str, dict]:
    packs = {}
    for task in ["pc_pn", "all_types"]:
        candidates = [outdir / f"BDE_best_model_{task}.joblib", outdir / task / f"BDE_best_model_{task}.joblib"]
        for p in candidates:
            if p.exists():
                packs[task] = joblib.load(p)
                break
    return packs


def choose_model_for_bond(bond_type: str, packs: Dict[str, dict]) -> Optional[str]:
    """Choose the model according to the clarified BDE prediction role.

    P-C / P-N are the main DOPO targets, so pc_pn is preferred.
    P-O / P-S / P-H are only covered by all_types; using pc_pn for them would
    be conceptually wrong, so no fallback to pc_pn is used for those bond types.
    """
    b = normalize_bond_type(bond_type)
    if b in {"P-C", "P-N"}:
        for t in ["pc_pn", "all_types"]:
            if t in packs:
                return t
    if b in {"P-O", "P-S", "P-H"}:
        if "all_types" in packs:
            return "all_types"
        return None
    if "all_types" in packs:
        return "all_types"
    if "pc_pn" in packs:
        return "pc_pn"
    return None


def prediction_agreement_note(diff: Optional[float]) -> str:
    if diff is None or not np.isfinite(diff):
        return "Only one suitable model prediction is available."
    if diff <= 10:
        return "High agreement between pc_pn and all_types predictions."
    if diff <= 25:
        return "Moderate agreement between pc_pn and all_types predictions."
    return "Low agreement between pc_pn and all_types predictions; manually inspect the molecule/target bond."


def confidence_note_for_prediction(bond_type: str, selected_task: Optional[str], pred_std: Optional[float] = None, diff: Optional[float] = None) -> str:
    b = normalize_bond_type(bond_type)
    notes: List[str] = []
    if selected_task is None:
        return "No suitable model was available for this bond type."
    if b in {"P-C", "P-N"} and selected_task == "pc_pn":
        notes.append("Main application domain: DOPO P-C/P-N prediction; use as trend-level BDE feature.")
    elif b in {"P-C", "P-N"} and selected_task == "all_types":
        notes.append("Fallback all_types model used because pc_pn was unavailable.")
    elif b in {"P-O", "P-S", "P-H"}:
        notes.append("Low-confidence extrapolation: few training samples for P-O/P-S/P-H; use only as rough reference.")
    else:
        notes.append("Unknown bond type; prediction may be unreliable.")

    if pred_std is not None and np.isfinite(pred_std):
        if pred_std > 20:
            notes.append("High final-ensemble variance; prediction is less stable.")
        elif pred_std > 10:
            notes.append("Moderate final-ensemble variance.")
    if diff is not None and np.isfinite(diff) and b in {"P-C", "P-N"}:
        if diff > 25:
            notes.append("pc_pn and all_types disagree strongly.")
        elif diff > 10:
            notes.append("pc_pn and all_types show moderate disagreement.")
    return " ".join(notes)


def predict_single_smiles(smiles: str, bond_type: str, packs: Dict[str, dict], return_all: bool = True) -> Dict[str, object]:
    pred_frame = build_prediction_frame(smiles, bond_type)
    X_new = build_features(pred_frame, "Smiles", "Bond_Type")
    out: Dict[str, object] = {
        "Smiles": clean_text(smiles),
        "Canonical_SMILES": canonical_smiles(smiles),
        "Bond_Type": normalize_bond_type(bond_type),
    }
    all_model_preds: Dict[str, object] = {}
    all_model_stds: Dict[str, object] = {}
    all_model_meta: Dict[str, Dict[str, object]] = {}
    for task, pack in packs.items():
        try:
            mean_pred, std_pred = predict_with_pack_detail(pack, X_new)
            all_model_preds[task] = float(mean_pred[0])
            all_model_stds[task] = float(std_pred[0])
            all_model_meta[task] = {"model_name": pack.get("model_name", ""), "target_mode": pack.get("target_mode", "raw"), "k_best": pack.get("k_best", "")}
        except Exception as e:
            all_model_preds[task] = f"ERROR: {e}"
            all_model_stds[task] = np.nan
            all_model_meta[task] = {"model_name": "ERROR", "target_mode": "ERROR", "k_best": ""}
    chosen = choose_model_for_bond(bond_type, packs)
    selected_pred = all_model_preds.get(chosen, np.nan) if chosen else np.nan
    selected_std = all_model_stds.get(chosen, np.nan) if chosen else np.nan

    pc_pred = all_model_preds.get("pc_pn", np.nan)
    all_pred = all_model_preds.get("all_types", np.nan)
    diff = None
    if isinstance(pc_pred, (int, float)) and isinstance(all_pred, (int, float)) and np.isfinite(pc_pred) and np.isfinite(all_pred):
        diff = abs(float(pc_pred) - float(all_pred))

    out["selected_model_task"] = chosen
    out["selected_model_name"] = all_model_meta.get(chosen, {}).get("model_name", "") if chosen else ""
    out["selected_target_mode"] = all_model_meta.get(chosen, {}).get("target_mode", "") if chosen else ""
    out["selected_k_best"] = all_model_meta.get(chosen, {}).get("k_best", "") if chosen else ""
    out["BDE_pred_kJ_mol"] = selected_pred
    out["BDE_pred_std_kJ_mol"] = selected_std
    out["pcpn_alltypes_abs_diff_kJ_mol"] = diff if diff is not None else np.nan
    out["prediction_agreement_note"] = prediction_agreement_note(diff)
    out["confidence_note"] = confidence_note_for_prediction(bond_type, chosen, pred_std=selected_std, diff=diff)
    if return_all:
        for task, val in all_model_preds.items():
            out[f"BDE_pred_{task}_kJ_mol"] = val
            out[f"BDE_pred_{task}_std_kJ_mol"] = all_model_stds.get(task, np.nan)
            out[f"BDE_model_{task}"] = all_model_meta.get(task, {}).get("model_name", "")
            out[f"BDE_target_mode_{task}"] = all_model_meta.get(task, {}).get("target_mode", "")
            out[f"BDE_k_best_{task}"] = all_model_meta.get(task, {}).get("k_best", "")
    return out


def predict_csv_table(input_csv: Path, smiles_col: Optional[str], bond_col: Optional[str], packs: Dict[str, dict], out_csv: Optional[Path] = None) -> Path:
    df_raw = read_csv_auto(input_csv)
    df_in = standardize_prediction_input(df_raw, smiles_col=smiles_col, bond_col=bond_col)

    preds = []
    for _, row in df_in.iterrows():
        if not bool(row.get("input_valid", False)):
            base = {
                "Smiles": row.get("Smiles", ""),
                "Canonical_SMILES": row.get("Canonical_SMILES", ""),
                "Bond_Type": row.get("Bond_Type", "Unknown"),
                "selected_model_task": None,
                "BDE_pred_kJ_mol": np.nan,
                "confidence_note": row.get("input_warning", "Invalid input."),
            }
            for task in packs:
                base[f"BDE_pred_{task}_kJ_mol"] = np.nan
            preds.append(base)
        else:
            preds.append(predict_single_smiles(row["Smiles"], row["Bond_Type"], packs, return_all=True))

    pred_df = pd.DataFrame(preds)
    out_df = pd.concat([df_raw.reset_index(drop=True), pred_df.add_prefix("pred_")], axis=1)
    if out_csv is None:
        out_csv = input_csv.with_name(input_csv.stem + "_with_BDE_predictions.csv")
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved BDE prediction CSV: {out_csv}")
    return out_csv


# -----------------------------------------------------------------------------
# Main table synchronization / 主数据库同步
# -----------------------------------------------------------------------------
def _default_main_base_csv() -> Path:
    """Default authoritative main table: data/DOPO_EP_new.csv."""
    candidates = [
        PROJECT_ROOT / "data" / "DOPO_EP_new.csv",
        THIS_DIR.parent / "data" / "DOPO_EP_new.csv",
        THIS_DIR / "data" / "DOPO_EP_new.csv",
    ]
    return _first_existing(candidates)


def _default_main_bde_csv() -> Path:
    """Default enriched main table: data/DOPO_EP_new_with_BDE.csv."""
    candidates = [
        PROJECT_ROOT / "data" / "DOPO_EP_new_with_BDE.csv",
        THIS_DIR.parent / "data" / "DOPO_EP_new_with_BDE.csv",
        THIS_DIR / "data" / "DOPO_EP_new_with_BDE.csv",
    ]
    return _first_existing(candidates)


def _is_missing_cell(x) -> bool:
    s = clean_text(x)
    return s == "" or s.lower() in {"none", "nan", "null", "na", "n/a"}


def _to_float_or_nan(x) -> float:
    s = clean_text(x)
    if _is_missing_cell(s):
        return np.nan
    try:
        return float(str(s).replace("%", ""))
    except Exception:
        return np.nan


def _format_number_cell(x, ndigits: int = 6) -> str:
    try:
        v = float(x)
        if not np.isfinite(v):
            return ""
        return (f"{v:.{ndigits}f}").rstrip("0").rstrip(".")
    except Exception:
        return ""


def update_main_table_with_bde(
    base_csv: Path,
    current_bde_csv: Optional[Path],
    out_csv: Path,
    packs: Optional[Dict[str, dict]] = None,
) -> Path:
    """Synchronize DOPO_EP_new_with_BDE.csv with DOPO_EP_new.csv.

    Purpose / 用途：
      1) Treat DOPO_EP_new.csv as the authoritative source for all original
         experimental columns.
      2) Preserve BDE enrichment columns already present in DOPO_EP_new_with_BDE.csv.
      3) If trained/saved BDE model packs are available, refresh BDE predictions
         from SMILES_main + BDE_Type.
      4) Recalculate FR_main_BDE_final_kJ_mol, FR_main_BDE_available,
         FR_main_BDE_source and FR_main_BDE_low_confidence.

    This function is intentionally conservative: it does not modify the BDE
    training set 07_BDE/data/BDE.csv.
    """
    base_csv = Path(base_csv)
    out_csv = Path(out_csv)
    current_bde_csv = Path(current_bde_csv) if current_bde_csv else None

    base_df = read_csv_auto(base_csv)
    if current_bde_csv and current_bde_csv.exists():
        bde_df = read_csv_auto(current_bde_csv)
        if len(base_df) != len(bde_df):
            raise ValueError(
                f"Row count mismatch: base={len(base_df)} rows, current_bde={len(bde_df)} rows. "
                "Please align rows before syncing."
            )
        extra_cols = [c for c in bde_df.columns if c not in base_df.columns]
        out_df = pd.concat([base_df.reset_index(drop=True), bde_df[extra_cols].reset_index(drop=True)], axis=1)
    else:
        out_df = base_df.copy()
        extra_cols = []

    # Optional prediction refresh using BDE model packs.
    if packs and "SMILES_main" in out_df.columns and "BDE_Type" in out_df.columns:
        pred_rows: List[Dict[str, object]] = []
        for _, row in out_df.iterrows():
            smiles = clean_text(row.get("SMILES_main", ""))
            bond_type = normalize_bond_type(row.get("BDE_Type", "Unknown"))
            if not smiles or bond_type not in SUPPORTED_BOND_TYPES:
                pred_rows.append({})
                continue
            try:
                pred_rows.append(predict_single_smiles(smiles, bond_type, packs, return_all=True))
            except Exception as e:
                pred_rows.append({"confidence_note": f"Prediction failed: {e}"})

        pred_df = pd.DataFrame(pred_rows)
        mapping = {
            "Smiles": "pred_Smiles",
            "Canonical_SMILES": "pred_Canonical_SMILES",
            "Bond_Type": "pred_Bond_Type",
            "selected_model_task": "pred_selected_model_task",
            "selected_model_name": "pred_selected_model_name",
            "selected_target_mode": "pred_selected_target_mode",
            "selected_k_best": "pred_selected_k_best",
            "BDE_pred_kJ_mol": "pred_BDE_pred_kJ_mol",
            "BDE_pred_std_kJ_mol": "pred_BDE_pred_std_kJ_mol",
            "confidence_note": "pred_confidence_note",
            "prediction_agreement_note": "pred_prediction_agreement_note",
            "pcpn_alltypes_abs_diff_kJ_mol": "pred_pcpn_alltypes_abs_diff_kJ_mol",
            "BDE_pred_pc_pn_kJ_mol": "pred_BDE_pred_pc_pn_kJ_mol",
            "BDE_pred_pc_pn_std_kJ_mol": "pred_BDE_pred_pc_pn_std_kJ_mol",
            "BDE_model_pc_pn": "pred_BDE_model_pc_pn",
            "BDE_target_mode_pc_pn": "pred_BDE_target_mode_pc_pn",
            "BDE_k_best_pc_pn": "pred_BDE_k_best_pc_pn",
            "BDE_pred_all_types_kJ_mol": "pred_BDE_pred_all_types_kJ_mol",
            "BDE_pred_all_types_std_kJ_mol": "pred_BDE_pred_all_types_std_kJ_mol",
            "BDE_model_all_types": "pred_BDE_model_all_types",
            "BDE_target_mode_all_types": "pred_BDE_target_mode_all_types",
            "BDE_k_best_all_types": "pred_BDE_k_best_all_types",
        }
        for src_col, dst_col in mapping.items():
            if src_col in pred_df.columns:
                out_df[dst_col] = pred_df[src_col]

        # FR_main_BDE_* aliases used by the main LOI/PHRR/THR/UL94 pipeline.
        alias_mapping = {
            "pred_BDE_pred_kJ_mol": "FR_main_BDE_pred_kJ_mol",
            "pred_BDE_pred_std_kJ_mol": "FR_main_BDE_pred_std_kJ_mol",
            "pred_selected_model_task": "FR_main_BDE_model_task",
            "pred_selected_model_name": "FR_main_BDE_model_name",
            "pred_selected_target_mode": "FR_main_BDE_target_mode",
            "pred_selected_k_best": "FR_main_BDE_k_best",
            "pred_confidence_note": "FR_main_BDE_confidence_note",
            "pred_prediction_agreement_note": "FR_main_BDE_prediction_agreement_note",
            "pred_pcpn_alltypes_abs_diff_kJ_mol": "FR_main_BDE_pcpn_alltypes_diff_kJ_mol",
            "pred_BDE_pred_pc_pn_kJ_mol": "FR_main_BDE_pc_pn_pred_kJ_mol",
            "pred_BDE_pred_pc_pn_std_kJ_mol": "FR_main_BDE_pc_pn_pred_std_kJ_mol",
            "pred_BDE_pred_all_types_kJ_mol": "FR_main_BDE_all_types_pred_kJ_mol",
            "pred_BDE_pred_all_types_std_kJ_mol": "FR_main_BDE_all_types_pred_std_kJ_mol",
        }
        for src_col, dst_col in alias_mapping.items():
            if src_col in out_df.columns:
                out_df[dst_col] = out_df[src_col]

    # Recalculate final BDE columns. Measured/original value has priority over prediction.
    measured = pd.to_numeric(out_df.get("FR_main_BDE_KJ/mol", pd.Series([np.nan] * len(out_df))), errors="coerce")
    pred = pd.to_numeric(out_df.get("FR_main_BDE_pred_kJ_mol", pd.Series([np.nan] * len(out_df))), errors="coerce")
    if pred.isna().all() and "pred_BDE_pred_kJ_mol" in out_df.columns:
        pred = pd.to_numeric(out_df["pred_BDE_pred_kJ_mol"], errors="coerce")

    final = measured.where(measured.notna(), pred)
    out_df["FR_main_BDE_final_kJ_mol"] = final
    out_df["FR_main_BDE_available"] = final.notna().astype(int)

    source = np.where(measured.notna(), "measured_or_original",
                      np.where(pred.notna(), "predicted_by_BDE.py", "missing"))
    out_df["FR_main_BDE_source"] = source

    note_col = "FR_main_BDE_confidence_note" if "FR_main_BDE_confidence_note" in out_df.columns else "pred_confidence_note"
    if note_col in out_df.columns:
        notes = out_df[note_col].astype(str).str.lower()
        low_conf = (
            final.isna()
            | notes.str.contains("low-confidence|unknown|no suitable|failed|high final-ensemble", regex=True, na=False)
        )
    else:
        low_conf = final.isna()
    out_df["FR_main_BDE_low_confidence"] = low_conf.astype(int)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] Main table synchronized with BDE features: {out_csv}")
    print(f"[INFO] base_columns={len(base_df.columns)}, preserved_extra_columns={len(extra_cols)}, rows={len(out_df)}")
    print(f"[INFO] FR_main_BDE_available={int(out_df['FR_main_BDE_available'].sum())}/{len(out_df)}")
    return out_csv


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final thesis-use DOPO P-C/P-N BDE model training and prediction.")
    parser.add_argument("--tasks", default=",".join(TASKS_DEFAULT), help="Comma-separated tasks. Default pc_pn only; all_types is kept only for compatibility.")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and load existing joblib models from OUTDIR.")
    parser.add_argument("--predict-smiles", default=None, help="Single SMILES string for BDE prediction.")
    parser.add_argument("--bond-type", default="P-C", help="Target bond type for --predict-smiles, e.g. P-C, P-N, P-O, P-S, P-H.")
    parser.add_argument("--predict-csv", default=None, help="CSV file containing SMILES and bond-type columns for batch prediction.")
    parser.add_argument("--smiles-col", default=None, help="SMILES column name for --predict-csv. If omitted, the script auto-detects it.")
    parser.add_argument("--bond-col", default=None, help="Bond type column name for --predict-csv. If omitted, the script auto-detects it.")
    parser.add_argument("--out-csv", default=None, help="Output CSV path for --predict-csv.")
    parser.add_argument("--make-template", default=None, help="Create a prediction input template CSV at this path.")
    parser.add_argument("--update-main-bde", action="store_true",
                        help="Force update data/DOPO_EP_new_with_BDE.csv from data/DOPO_EP_new.csv and refresh BDE feature columns. Kept for compatibility; auto-update is ON by default.")
    parser.add_argument("--no-update-main-bde", action="store_true",
                        help="Disable automatic update of data/DOPO_EP_new_with_BDE.csv after BDE training/loading.")
    parser.add_argument("--main-base-csv", default=None,
                        help="Authoritative main table CSV. Default: data/DOPO_EP_new.csv")
    parser.add_argument("--main-bde-csv", default=None,
                        help="Current enriched BDE table to preserve extra BDE columns. Default: data/DOPO_EP_new_with_BDE.csv")
    parser.add_argument("--main-out-csv", default=None,
                        help="Output path for updated main BDE table. Default: same as --main-bde-csv")
    parser.add_argument("--verbose", action="store_true", help="Print each model result instead of compact logs.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    global VERBOSE, PRINT_EACH_MODEL
    if args.verbose:
        VERBOSE = True
        PRINT_EACH_MODEL = True
    tasks = [t.strip().lower() for t in args.tasks.split(",") if t.strip()]
    tasks = ["pc_pn" if t in {"pcpn", "pc_pn_classified"} else t for t in tasks]

    print("=" * 78)
    print("[BDE | final pc_pn model]")
    print("=" * 78)
    print(f"[INFO] DATA_PATH={DATA_PATH}")
    print(f"[INFO] OUTDIR={OUTDIR}")
    print(f"[INFO] tasks={tasks}")
    print(f"[INFO] split_mode={SPLIT_MODE}, n_splits={N_SPLITS}, n_repeats={N_REPEATS}, cv_random_state={CV_RANDOM_STATE}")
    print(f"[INFO] seeds={REPEAT_SEEDS}, test_size={TEST_SIZE}, global_K_LIST={K_LIST}, XGB={USE_XGB}, fast_mode={FAST_MODE}, slow_tuning={SLOW_TUNING}")
    print(f"[INFO] task_K_LISTS={task_k_lists(tasks)}, final_model_seed={FINAL_MODEL_SEED}")
    print(f"[INFO] best_selection={BEST_SELECTION_MODE}, r2_tol={BEST_R2_TOL}, final_ensemble={FINAL_ENSEMBLE}, ensemble_seeds={FINAL_ENSEMBLE_SEEDS if FINAL_ENSEMBLE else [FINAL_MODEL_SEED]}")
    print(f"[INFO] target_modes={TARGET_MODES}, seed_diagnostics={ENABLE_SEED_DIAGNOSTICS}, error_diagnostics={ENABLE_ERROR_DIAGNOSTICS}, target_R2_goal={R2_GOAL}")
    print(f"[INFO] feature_mode={FEATURE_MODE}, whole_radii={WHOLE_FP_RADII}, target_radii={TARGET_FP_RADII}, atomPair={USE_ATOMPAIR_FP}, torsion={USE_TORSION_FP}, pattern={USE_PATTERN_FP}")
    print(f"[INFO] R2_BOOST_NOTE={R2_BOOST_NOTE}")

    guide_path = write_io_guide(OUTDIR)
    print(f"[INFO] IO guide saved: {guide_path}")

    if args.make_template:
        make_prediction_template(Path(args.make_template))

    needs_prediction_models = bool(args.predict_smiles or args.predict_csv)
    if args.skip_train:
        packs = load_saved_packs(OUTDIR)
        if needs_prediction_models and not packs:
            raise RuntimeError(f"--skip-train was used, but no saved models found in {OUTDIR}")
    else:
        packs = train_all_tasks(tasks)

    if args.predict_smiles:
        pred = predict_single_smiles(args.predict_smiles, args.bond_type, packs, return_all=True)
        print("\n[PREDICT SMILES RESULT]")
        print(json.dumps(pred, ensure_ascii=False, indent=2))
        pd.DataFrame([pred]).to_csv(OUTDIR / "BDE_single_smiles_prediction.csv", index=False, encoding="utf-8-sig")
        print(f"[INFO] Saved: {OUTDIR / 'BDE_single_smiles_prediction.csv'}")

    if args.predict_csv:
        out_csv = Path(args.out_csv) if args.out_csv else None
        predict_csv_table(Path(args.predict_csv), args.smiles_col, args.bond_col, packs, out_csv=out_csv)

    # Auto-update main DOPO+EP table by default.
    # 普通运行 BDE.py 时，训练/加载 BDE 模型后自动刷新 data/DOPO_EP_new_with_BDE.csv。
    # Use --no-update-main-bde if you only want to train the standalone BDE model.
    auto_update_main_bde = (not args.no_update_main_bde) or args.update_main_bde
    if auto_update_main_bde:
        base_csv = Path(args.main_base_csv) if args.main_base_csv else _default_main_base_csv()
        current_bde_csv = Path(args.main_bde_csv) if args.main_bde_csv else _default_main_bde_csv()
        out_main_csv = Path(args.main_out_csv) if args.main_out_csv else current_bde_csv
        print("\n" + "=" * 78)
        print("[AUTO UPDATE] Synchronizing data/DOPO_EP_new_with_BDE.csv")
        print("=" * 78)
        print(f"[INFO] main_base_csv={base_csv}")
        print(f"[INFO] main_bde_csv={current_bde_csv}")
        print(f"[INFO] main_out_csv={out_main_csv}")
        update_main_table_with_bde(base_csv, current_bde_csv, out_main_csv, packs=packs)

    print("\n" + "=" * 78)
    print("[SUCCESS] Workflow completed")
    print("=" * 78)
    print(f"[INFO] Output directory: {OUTDIR}")
    print("[INFO] Key files:")
    print("  - BDE_all_tasks_summary.csv")
    print("  - BDE_all_tasks_all_results.csv")
    print("  - pc_pn/BDE_pc_pn_summary.csv")
    print("  - pc_pn/BDE_pc_pn_summary_raw.csv")
    print("  - pc_pn/BDE_best_model_pc_pn.joblib")
    print("  - BDE_best_model_pc_pn.joblib")
    print("  - diagnostics/BDE_manual_check_candidates.csv")
    print("  - diagnostics/BDE_top_outliers_by_config.csv")
    print("  - BDE_single_smiles_prediction.csv if --predict-smiles was used")


if __name__ == "__main__":
    main()
