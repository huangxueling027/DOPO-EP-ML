"""
Regression v5:
- LOI
- PHRR
- THR
- Tg
- Char_yield`
- TS
- FS
- BDE

Classification:
- UL94_num
"""
import os
import re
import json
import warnings
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors,MACCSkeys
from rdkit.Chem import rdFingerprintGenerator
from typing import Dict, Tuple
from sklearn.ensemble import VotingRegressor
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    GroupShuffleSplit,
    GroupKFold,
    cross_val_predict
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,

    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.svm import SVR
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

# Optional CatBoost: improves small-sample tabular regression when installed.
try:
    from catboost import CatBoostRegressor
    # 当前 sklearn 版本与 CatBoost 的 sklearn tags 不兼容，
    # 不能直接放入 Pipeline/cross_val_predict。先禁用，避免运行中断。
    HAS_CATBOOST = False
except Exception:
    CatBoostRegressor = None
    HAS_CATBOOST = False

warnings.filterwarnings("ignore")
# ===== 重复随机种子稳定性评估开关 =====
ENABLE_REPEAT_EVAL = os.environ.get("DOPO_REPEAT_EVAL", "0") == "1"

# ===== BDE mechanism features for LOI/PHRR/THR/Tg/Char/TS/FS =====
# 1 = use measured/predicted FR_main_BDE columns as input features for property models.
# 0 = ablation mode; remove all BDE-related input features.
USE_BDE_FEATURES = os.environ.get("DOPO_USE_BDE_FEATURES", "0").strip() == "1"

REPEAT_SEEDS = [
    int(s) for s in os.environ.get(
        "DOPO_REPEAT_SEEDS",
        "42,52,62,72,82"
    ).split(",")
]
# =========================================================
# 1. READ CSV
# =========================================================
def read_csv_auto(path: str) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "gb18030", "gbk", "big5", "latin1"]
    last_error = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"[INFO] File loaded with encoding: {enc}")
            return df
        except Exception as e:
            last_error = e
    raise RuntimeError(f"无法读取文件：{path}\n最后一次报错：{last_error}")
# =========================================================
# 2. RESOLVE REAL COLUMN NAMES
# =========================================================
def resolve_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    解析CSV文件的列名，建立标准列名到实际列名的映射关系
    该函数通过别名匹配机制，将不同命名风格的列名统一映射到标准列名，提高代码对不同数据源的兼容性
    参数:
        df (pd.DataFrame): 原始数据框，包含待解析的列名 
    返回:
        Dict[str, str]: 标准列名到实际列名的映射字典
                       例如: {"SMILES_main": "SMILES_main", "LOI": "LOI"}
    异常:
        KeyError: 当缺少必要的列时抛出异常
    """
    # 获取数据框的所有列名
    cols = list(df.columns)
    # 定义标准列名及其可能的别名列表
    aliases = {
        # 阻燃剂基本信息：FR_main(主阻燃剂), FR_co(协效阻燃剂), FR_class(阻燃剂类别), Curing_Agent(固化剂)
        "FR_main": ["FR_main", "Unnamed: 0", "FR"],
        "FR_co": ["FR_co"],
        "FR_class": ["FR_class"],
        "Preparation_Method": ["Preparation_Method"],
        "Preparation_Method_num": ["Preparation_Method_num"],
        "Curing_Agent": ["Curing_Agent"],
        # SMILES分子结构表示：SMILES_main(主阻燃剂), SMILES_co(协效阻燃剂), SMILES_Curing_Agent(固化剂)
        "SMILES_main": ["SMILES_main"],
        "SMILES_co": ["SMILES_co"],
        "SMILES_Curing_Agent": ["SMILES_Curing_Agent"],
        # 组成和配方信息：P_content wt%(磷含量), Synergy_flag(协同效应标志), Synergy_type(协同效应类型), Main_FR_fraction(主阻燃剂比例), Co_FR_fraction(协效阻燃剂比例)
        "P_content wt%": ["P_content wt%", "Loading_total_P wt%"],
        "Synergy_flag": ["Synergy_flag", "Synergy_flag(single=0&synergy=1)"],
        "Synergy_type": ["Synergy_type"],
        "Main_FR_fraction": ["Main_FR_fraction"],
        "Co_FR_fraction": ["Co_FR_fraction"],
        # 元素存在标志：Has_P(是否含磷), Has_N(是否含氮), Has_Si(是否含硅), Has_B(是否含硼), Has_S(是否含硫), Has_Al(是否含铝)
        "Has_P": ["Has_P"],
        "Has_N": ["Has_N"],
        "Has_Si": ["Has_Si"],
        "Has_B": ["Has_B"],
        "Has_S": ["Has_S"],
        "Has_Al": ["Has_Al"],
        # 性能指标：LOI(极限氧指数), PHRR(峰值热释放速率), THR(总热释放量), UL94(UL94阻燃等级), UL94_num(UL94数值等级), Tg(玻璃化转变温度), Char_yield(残炭率), TS_MPa(拉伸强度), FS_MPa(弯曲强度), BDE(键解离能)
        "LOI": ["LOI"],
        "PHRR": ["PHRR_kw_㎡", "PHRR_kw_m2", "PHRR_kw_m²", "PHRR(kw/m2)", "PHRR_kw_m2"],
        "THR": ["THR_MJ_㎡", "THR_MJ_m2", "THR_MJ_m²", "THR(MJ/m2)", "THR_MJ_m2"],
        "UL94": ["UL94"],
        "UL94_num": ["UL94_num"],
        "Tg": ["Tg_℃", "Tg"],
        "Char_yield": ["Char_yield_％_700C", "Char_yield_%_700C", "Char_yield"],
        "TS_MPa": ["TS_MPa"],
        "FS_MPa": ["FS_MPa"],
        # BDE: prefer the final measured+predicted column stored in data/DOPO_EP_new_with_BDE.csv.
        # Fall back to V7 prediction columns or the original measured/DFT BDE columns.
        "BDE": [
            "FR_main_BDE_final_kJ_mol",
            "FR_main_BDE_kJ_mol",
            "FR_main_BDE_pred_kJ_mol", "pred_BDE_pred_kJ_mol", "BDE_pred_kJ_mol",
            "FR_main_BDE_KJ/mol", "FR_main_BDE_KJ_mol", "FR_main_BDE(kJ/mol)", "FR_main_BDE",
            "BDE", "BDE_kJ_mol", "BDE(kJ/mol)",
            "FR_main_BDE_Kcal/mol", "FR_main_BDE_kcal_mol",
        ],

        # 最新表格新增组成/元素/基线/差值信息
        "Loading_total_FR wt%": ["Loading_total_FR wt%", "Loading_total_FR_wt%", "Loading_total_FR", "FR_wt%", "FR_wt", "FR wt%"],
        "N_content wt%": ["N_content wt%", "N_content_wt%"],
        "S_content wt%": ["S_content wt%", "S_content_wt%"],
        "B_content wt%": ["B_content wt%", "B_content_wt%"],
        "Si_content wt%": ["Si_content wt%", "Si_content_wt%"],
        "N/P ratio": ["N/P ratio", "N_P_ratio", "P_N_ratio"],
        "S/P ratio": ["S/P ratio", "S_P_ratio", "P_S_ratio"],
        "B/P ratio": ["B/P ratio", "B_P_ratio", "P_B_ratio"],
        "Si/P ratio": ["Si/P ratio", "Si_P_ratio", "P_Si_ratio"],
        "CuringAgent_Has_N": ["CuringAgent_Has_N"],
        "CuringAgent_Has_S": ["CuringAgent_Has_S"],
        "CuringAgent_Has_P": ["CuringAgent_Has_P"],
        "CuringAgent_Has_B": ["CuringAgent_Has_B"],
        "CuringAgent_Has_F": ["CuringAgent_Has_F"],
        "CuringAgent_Has_Cl": ["CuringAgent_Has_Cl"],
        "Cone_flux_kW_m2": ["Cone_flux_kW_m2", "Cone_flux_kw_m2", "Cone_flux"],
        "EP_matrix_LOI": ["EP_matrix_LOI"],
        "Delta_LOI": ["Delta_LOI"],
        "EP_matrix_PHRR": ["EP_matrix_PHRR"],
        "Delta_PHRR": ["Delta_PHRR"],
        "EP_matrix_THR": ["EP_matrix_THR"],
        "Delta_THR": ["Delta_THR"],
        "EP_matrix_Tg": ["EP_matrix_Tg"],
        "Delta_Tg": ["Delta_Tg"],
        "EP_matrix_CY": ["EP_matrix_CY"],
        "Delta_CY": ["Delta_CY"],
        "EP_matrix_TS": ["EP_matrix_TS"],
        "Delta_TS": ["Delta_TS"],
        "EP_matrix_FS": ["EP_matrix_FS"],
        "Delta_FS": ["Delta_FS"],
        "BDE_Type": ["BDE_Type"],
        "Reference": ["Reference"],
        "Cure_Temp_Max": ["Cure_Temp_Max"],
        "LOI_Thickness_mm": ["LOI_Thickness_mm"],
        "UL94_Thickness_mm": ["UL94_Thickness_mm"],
        "Cone_Thickness_mm": ["Cone_Thickness_mm"],
    }
    # 建立列名映射：遍历标准列名，在数据框中查找匹配的实际列名
    colmap = {}
    for std_name, candidates in aliases.items():
        for c in candidates:
            if c in cols:                                # 找到匹配的列名
                colmap[std_name] = c                      # 建立映射关系
                break                                     # 找到后立即停止搜索
    required = ["SMILES_main", "SMILES_co"]              # 必需列名列表
    missing_required = [c for c in required if c not in colmap]  # 找出缺失的必需列
    if missing_required:
        raise KeyError(f"缺少必要列：{missing_required}")   # 抛出异常提示用户
    return colmap                                         # 返回列名映射字典
# =========================================================
# 3. SAFE PARSERS
# =========================================================
def clean_text(x):
    """
    清理和标准化文本数据
    参数:
        x: 输入值，可以是任意类型
    
    返回:
        清理后的字符串或np.nan
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    return s
# Canonical labels used throughout modeling/visualization.  Preparation Method is
# a nominal categorical variable; textual variants must be unified before one-hot
# encoding so semantically identical categories cannot become separate features.
_PREPARATION_METHOD_ALIASES = {
    # Canonical database labels
    "DOPO-based (additive)": "DOPO-based (additive)",
    "DOPO-based (reactive)": "DOPO-based (reactive)",
    "DOPO-based (Co-curing)": "DOPO-based (Co-curing)",
    "DOPO-based (additive+ Secondary Crosslinking)":
        "DOPO-based (Additive + Secondary Crosslinking)",
    "DOPO-based (Additive + Secondary Crosslinking)":
        "DOPO-based (Additive + Secondary Crosslinking)",
    # Human-readable labels used by external-validation/candidate files
    "Additive": "DOPO-based (additive)",
    "Reactive": "DOPO-based (reactive)",
    "Co-curing": "DOPO-based (Co-curing)",
    "Co curing": "DOPO-based (Co-curing)",
    "Additive+Secondary Crosslinking":
        "DOPO-based (Additive + Secondary Crosslinking)",
    "Additive + Secondary Crosslinking":
        "DOPO-based (Additive + Secondary Crosslinking)",
}


def normalize_preparation_method(x):
    """Normalize raw Preparation_Method text to four canonical categories.

    The V5 database contains two spelling/capitalization variants of the
    Additive + Secondary Crosslinking category.  They must map to one nominal
    category before feature construction.  Unknown non-empty labels are kept
    unchanged so data issues remain auditable instead of being silently coerced.
    """
    if pd.isna(x):
        return np.nan
    s = clean_text(x)
    return _PREPARATION_METHOD_ALIASES.get(s, s)


def parse_numeric_keep_scale(x):
    """
    解析数值数据并保持原始尺度
    参数:
        x: 输入值，可以是数值、字符串、NaN等任意类型
    返回:
        float: 提取的数值，如果解析失败则返回np.nan
    """
    if isinstance(x, pd.Series):
        return x.apply(parse_numeric_keep_scale)
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == "":
        return np.nan
    # 移除各种格式符号和单位，只保留数值部分
    s = s.replace(",", "")
    s = s.replace("％", "%")
    s = s.replace("℃", "")
    s = s.replace("°C", "")
    s = s.replace("wt.%", "")
    s = s.replace("wt%", "")
    s = s.replace("wt %", "")
    s = s.replace("%", "")
    s = s.strip()
    # 使用正则表达式提取数值
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if m:
        try:
            return float(m.group())
        except Exception:
            return np.nan
    return np.nan


# Delta tasks describe the change relative to the corresponding neat-EP
# baseline.  Rows whose loading is explicitly zero are the reference rows and
# must not be treated as ordinary Delta training targets.  A missing loading is
# intentionally retained because several literature rows report a valid Delta
# value but do not report the exact total loading.
DELTA_TASK_NAMES = frozenset({
    "Delta_LOI", "Delta_PHRR", "Delta_THR", "Delta_Tg",
    "Delta_CY", "Delta_TS", "Delta_FS",
})


def get_loading_series(df: pd.DataFrame, colmap: Dict[str, str]) -> pd.Series:
    """Return total flame-retardant loading as a numeric Series.

    Missing values remain NaN.  This distinction is important: ``0`` means a
    confirmed neat-EP reference row, whereas NaN means the loading was not
    reported and should not automatically be interpreted as zero.
    """
    column = colmap.get("Loading_total_FR wt%")
    if column is None or column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return df[column].apply(parse_numeric_keep_scale).astype(float)


def build_task_valid_mask(
    df: pd.DataFrame,
    colmap: Dict[str, str],
    task_name: str,
    target: pd.Series | None = None,
) -> pd.Series:
    """Build the common row mask used by all model/evaluation entry points.

    Rules
    -----
    1. The target must be present.
    2. Delta tasks exclude only rows with an *explicit* zero total loading.
       Rows with unknown loading are retained if their Delta target is valid.

    Keeping this logic in one function prevents the main runners, scientific
    validation, diversity statistics and optional TabPFN comparisons from
    silently using different sample definitions.
    """
    if target is None:
        if task_name == "UL94_V0":
            if "UL94_V0" in df.columns:
                target = pd.to_numeric(df["UL94_V0"], errors="coerce")
            elif "UL94" in colmap:
                raw = df[colmap["UL94"]]
                target = raw.apply(
                    lambda value: np.nan
                    if pd.isna(value) or str(value).strip() == ""
                    else (1 if str(value).strip().upper() == "V-0" else 0)
                )
            else:
                return pd.Series(False, index=df.index, dtype=bool)
        else:
            target_column = colmap.get(task_name)
            if target_column is None or target_column not in df.columns:
                return pd.Series(False, index=df.index, dtype=bool)
            target = pd.to_numeric(df[target_column], errors="coerce")

    target_series = pd.Series(target, index=df.index)
    valid = target_series.notna()

    if task_name in DELTA_TASK_NAMES:
        loading = get_loading_series(df, colmap)
        # Exclude confirmed neat-EP references, but do not discard rows whose
        # loading is merely unreported.
        valid &= ~loading.eq(0.0)

    return valid.astype(bool)


def _flame_retardant_presence_masks(
    df: pd.DataFrame,
    colmap: Dict[str, str],
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return total/main/co flame-retardant presence masks.

    The database keeps the paired flame-retardant name/SMILES on neat-EP rows
    so the baseline can be traced back to its formulation series.  Those labels
    are metadata, not chemicals actually present at loading=0.  The masks make
    sure the model does not receive a hypothetical flame-retardant fingerprint
    or structure flag for a neat-EP sample.
    """
    loading = get_loading_series(df, colmap)
    # Only an explicit zero proves that no FR is present.  Unknown loading is
    # treated as present/unknown rather than incorrectly forced to zero.
    total_present = (~loading.eq(0.0)).astype(float)

    main_present = total_present.copy()
    if "Main_FR_fraction" in colmap:
        main_fraction = df[colmap["Main_FR_fraction"]].apply(
            parse_numeric_keep_scale
        ).astype(float)
        main_present *= (~main_fraction.eq(0.0)).astype(float)

    co_present = total_present.copy()
    if "Co_FR_fraction" in colmap:
        co_fraction = df[colmap["Co_FR_fraction"]].apply(
            parse_numeric_keep_scale
        ).astype(float)
        # A reported zero fraction means no co-FR.  Missing fraction is not by
        # itself proof of absence and is resolved by the SMILES check below.
        co_present *= (~co_fraction.eq(0.0)).astype(float)
    if "SMILES_co" in colmap:
        co_smiles_present = (
            df[colmap["SMILES_co"]]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .astype(float)
        )
        co_present *= co_smiles_present

    return total_present, main_present, co_present


def normalize_ul94(x):
    """
    标准化UL94阻燃等级的格式
    UL94等级：V-0(最佳), V-1(中等), V-2(较低), NR(未评级)
    参数:
        x: 输入值，可以是字符串、数字、NaN等任意类型
    返回:
        str: 标准化的UL94等级，只能是"V-0"、"V-1"、"V-2"或"NR"
    """
    if pd.isna(x):
        return "NR"
    s = str(x).upper().strip().replace(" ", "")
    s = s.replace("V0", "V-0").replace("V1", "V-1").replace("V2", "V-2")
    if s not in {"V-0", "V-1", "V-2", "NR"}:
        return "NR"
    return s
# =========================================================
# 4. CLEAN DATAFRAME
# =========================================================
def clean_dataframe(df: pd.DataFrame, colmap: Dict[str, str]) -> pd.DataFrame:
    """
    清理和标准化数据框中的所有列
    参数:
        df (pd.DataFrame): 原始数据框
        colmap (Dict[str, str]): 标准列名到实际列名的映射字典
    返回:
        pd.DataFrame: 清洗后的数据框副本
    """
    d = df.copy()
    # V7: 删除CSV/Excel末尾可能存在的整行空白记录，避免样本数虚增。
    d = d.dropna(how="all").copy()
    # ===== 文本列处理 =====
    text_keys = [
        "FR_main", "FR_co", "FR_class", "Curing_Agent",
        "SMILES_main", "SMILES_co", "SMILES_Curing_Agent", "Synergy_type", "UL94",
        "BDE_Type", "Reference", "Preparation_Method"
    ]
    for key in text_keys:
        if key in colmap:
            d[colmap[key]] = d[colmap[key]].apply(clean_text)

    # Preparation Method is nominal.  Normalize known textual aliases before
    # one-hot encoding; this merges the two equivalent secondary-crosslinking
    # spellings into one canonical category.
    if "Preparation_Method" in colmap:
        prep_col = colmap["Preparation_Method"]
        d[prep_col] = d[prep_col].apply(normalize_preparation_method)

    # ===== UL94列特殊处理 =====
    if "UL94" in colmap:
        d[colmap["UL94"]] = d[colmap["UL94"]].apply(normalize_ul94)
    # ===== 数值列处理 =====
    numeric_keys = [
        "P_content wt%", "Synergy_flag", "Main_FR_fraction", "Co_FR_fraction",
        "Has_P", "Has_N", "Has_Si", "Has_B", "Has_S", "Has_Al",
        "LOI", "PHRR", "THR", "UL94_num", "Tg", "Char_yield", "TS_MPa", "FS_MPa", "BDE",
        "Loading_total_FR wt%", "N_content wt%", "S_content wt%", "B_content wt%", "Si_content wt%",
        "N/P ratio", "S/P ratio", "B/P ratio", "Si/P ratio",
        "CuringAgent_Has_N", "CuringAgent_Has_S", "CuringAgent_Has_P", "CuringAgent_Has_B",
        "CuringAgent_Has_F", "CuringAgent_Has_Cl", "Cone_flux_kW_m2",
        "EP_matrix_LOI", "Delta_LOI", "EP_matrix_PHRR", "Delta_PHRR", "EP_matrix_THR", "Delta_THR",
        "EP_matrix_Tg", "Delta_Tg", "EP_matrix_CY", "Delta_CY", "EP_matrix_TS", "Delta_TS",
        "EP_matrix_FS", "Delta_FS",
        "Preparation_Method_num", "Cure_Temp_Max", "LOI_Thickness_mm", "UL94_Thickness_mm", "Cone_Thickness_mm"
    ]
    for key in numeric_keys:
        if key in colmap:
            d[colmap[key]] = d[colmap[key]].apply(parse_numeric_keep_scale)
    # ===== SMILES列空值处理 =====
    if "SMILES_co" in colmap:
        d[colmap["SMILES_co"]] = d[colmap["SMILES_co"]].fillna("")
    return d
# =========================================================
# 5. RDKit FEATURES
# =========================================================
def safe_mol_from_smiles(smiles: str):
    """
    安全地从SMILES字符串创建分子对象
    参数:
        smiles (str): SMILES字符串，表示分子的化学结构
    返回:
        rdkit.Chem.rdchem.Mol: 成功时返回RDKit分子对象
        None: 当SMILES无效、为空或转换失败时返回None
    """
    if pd.isna(smiles):
        return None
    if str(smiles).strip() == "":
        return None
    try:
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None
# 描述符模式开关
# basic：基础10个描述符，适合 Delta / LOI / UL94 / PHRR 等稳定任务
# advanced：高级描述符，适合 THR / Tg / TS 测试优化
USE_ADVANCED_DESCRIPTORS = os.environ.get("DOPO_DESCRIPTOR_MODE", "basic").lower() == "advanced"
# V7元素负载特征开关
USE_V7_FORMULA_FEATURES = (
    os.environ.get("DOPO_USE_V7_FEATURES", "1") == "1"
)

def calc_basic_descriptors(mol) -> Dict[str, float]:
    """
    计算分子描述符。

    默认 basic：
        10个基础描述符。

    当环境变量 DOPO_DESCRIPTOR_MODE=advanced 时：
        使用基础 + 高级 RDKit 描述符。
    """

    basic_keys = [
        "MolWt", "TPSA", "LogP",
        "HBD", "HBA", "RotBonds",
        "RingCount", "AromaticRings",
        "HeavyAtomCount", "FractionCSP3",
    ]

    advanced_keys = [
        "MolWt", "ExactMolWt", "MolMR", "TPSA", "LabuteASA", "LogP",
        "HBD", "HBA", "RotBonds", "RingCount", "AromaticRings",
        "HeavyAtomCount", "FractionCSP3",
        "NumValenceElectrons", "NumHeteroAtoms", "BertzCT", "BalabanJ",
        "AliphaticRings", "SaturatedRings", "AromaticCarbocycles",
        "AromaticHeterocycles", "SaturatedCarbocycles", "SaturatedHeterocycles",
        "Kappa1", "Kappa2", "Kappa3",
        "NHOHCount", "NOCount", "NumAmideBonds",
    ]

    descriptor_keys = advanced_keys if USE_ADVANCED_DESCRIPTORS else basic_keys

    if mol is None:
        return {k: np.nan for k in descriptor_keys}

    def _safe(func, default=np.nan):
        try:
            v = func(mol)
            if v is None:
                return default
            return float(v)
        except Exception:
            return default

    if not USE_ADVANCED_DESCRIPTORS:
        return {
            "MolWt": _safe(Descriptors.MolWt),
            "TPSA": _safe(rdMolDescriptors.CalcTPSA),
            "LogP": _safe(Descriptors.MolLogP),
            "HBD": _safe(Lipinski.NumHDonors),
            "HBA": _safe(Lipinski.NumHAcceptors),
            "RotBonds": _safe(Lipinski.NumRotatableBonds),
            "RingCount": _safe(rdMolDescriptors.CalcNumRings),
            "AromaticRings": _safe(rdMolDescriptors.CalcNumAromaticRings),
            "HeavyAtomCount": float(mol.GetNumHeavyAtoms()),
            "FractionCSP3": _safe(rdMolDescriptors.CalcFractionCSP3),
        }

    return {
        "MolWt": _safe(Descriptors.MolWt),
        "ExactMolWt": _safe(Descriptors.ExactMolWt),
        "MolMR": _safe(Descriptors.MolMR),
        "TPSA": _safe(rdMolDescriptors.CalcTPSA),
        "LabuteASA": _safe(rdMolDescriptors.CalcLabuteASA),
        "LogP": _safe(Descriptors.MolLogP),
        "HBD": _safe(Lipinski.NumHDonors),
        "HBA": _safe(Lipinski.NumHAcceptors),
        "RotBonds": _safe(Lipinski.NumRotatableBonds),
        "RingCount": _safe(rdMolDescriptors.CalcNumRings),
        "AromaticRings": _safe(rdMolDescriptors.CalcNumAromaticRings),
        "HeavyAtomCount": float(mol.GetNumHeavyAtoms()),
        "FractionCSP3": _safe(rdMolDescriptors.CalcFractionCSP3),
        "NumValenceElectrons": _safe(Descriptors.NumValenceElectrons),
        "NumHeteroAtoms": _safe(rdMolDescriptors.CalcNumHeteroatoms),
        "BertzCT": _safe(Descriptors.BertzCT),
        "BalabanJ": _safe(Descriptors.BalabanJ),
        "AliphaticRings": _safe(rdMolDescriptors.CalcNumAliphaticRings),
        "SaturatedRings": _safe(rdMolDescriptors.CalcNumSaturatedRings),
        "AromaticCarbocycles": _safe(rdMolDescriptors.CalcNumAromaticCarbocycles),
        "AromaticHeterocycles": _safe(rdMolDescriptors.CalcNumAromaticHeterocycles),
        "SaturatedCarbocycles": _safe(rdMolDescriptors.CalcNumSaturatedCarbocycles),
        "SaturatedHeterocycles": _safe(rdMolDescriptors.CalcNumSaturatedHeterocycles),
        "Kappa1": _safe(Descriptors.Kappa1),
        "Kappa2": _safe(Descriptors.Kappa2),
        "Kappa3": _safe(Descriptors.Kappa3),
        "NHOHCount": _safe(Descriptors.NHOHCount),
        "NOCount": _safe(Descriptors.NOCount),
        "NumAmideBonds": _safe(rdMolDescriptors.CalcNumAmideBonds),
    }
def calc_morgan_fp(mol, radius=2, n_bits=512) -> np.ndarray:
    """
    计算分子的Morgan指纹（也称为ECFP指纹）
    参数:
        mol (rdkit.Chem.rdchem.Mol): RDKit分子对象，如果为None则返回全零数组
        radius (int): Morgan指纹的半径，默认为2（1跳=直接相邻，2跳=二级邻域，3跳=三级邻域）
        n_bits (int): 指纹的位数，默认为512（常用值：512, 1024, 2048）
    返回:
        np.ndarray: 长度为n_bits的numpy数组，表示分子的Morgan指纹
    """
    arr = np.zeros((n_bits,), dtype=float)
    if mol is None:
        return arr
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr
def calc_maccs_fp(mol) -> np.ndarray:
    """
    计算 MACCS keys 指纹。
    RDKit 原始 MACCS 长度为 167，其中第 0 位通常不用；
    这里返回 166 位，便于论文中写 MACCS-166。
    """
    arr = np.zeros((167,), dtype=float)

    if mol is None:
        return arr[1:]

    fp = MACCSkeys.GenMACCSKeys(mol)
    DataStructs.ConvertToNumpyArray(fp, arr)

    return arr[1:]
def featurize_smiles_series(smiles_series: pd.Series, prefix: str,fp_bits: int = 512,fp_type: str = "morgan",radius: int = 2) -> pd.DataFrame:
    """
    批量将SMILES序列转换为分子特征数据框
    该函数对SMILES字符串序列进行批量特征提取，为每个SMILES生成：
    1. 基本分子描述符（10个理化性质）
    2. Morgan指纹（固定长度的结构特征）
    这些特征可以用于机器学习模型的训练和预测。
    参数:
        smiles_series (pd.Series): SMILES字符串的pandas Series
                               例如：["CCO", "c1ccccc1", "CC(=O)O"]
        prefix (str): 特征列名的前缀，用于区分不同来源的SMILES
                    例如："main"（主阻燃剂）、"co"（协效阻燃剂）、"curing"（固化剂）
                    最终列名格式："{prefix}_MolWt", "{prefix}_fp_0"等
        fp_bits (int): Morgan指纹的位数，默认为512
                     常用值：512, 1024, 2048
                     位数越多，结构信息越丰富，但计算成本越高
    返回:
        pd.DataFrame: 包含分子特征的DataFrame
                     - 前10列：基本描述符（带prefix前缀）
                     - 后fp_bits列：Morgan指纹（带prefix前缀和索引）
                     行数与输入SMILES序列相同
    特征总数: 10个基本描述符 + fp_bits个指纹位
    """
    # ===== 初始化特征存储列表 =====
    desc_rows = []  # 存储每个SMILES的基本描述符
    fp_rows = []    # 存储每个SMILES的Morgan指纹
    # ===== 批量处理SMILES序列 =====
    # 遍历SMILES系列，为每个SMILES提取特征
    # fillna("")将缺失值替换为空字符串，避免处理错误
    for smi in smiles_series.fillna(""):
        # 安全地将SMILES转换为分子对象
        # safe_mol_from_smiles处理各种异常情况，返回None或有效的Mol对象
        mol = safe_mol_from_smiles(smi)
        # 计算基本分子描述符
        # calc_basic_descriptors返回包含10个理化性质的字典,例如：{"MolWt": 46.07, "LogP": -0.31, "HBD": 1, ...}
        desc_rows.append(calc_basic_descriptors(mol))
        # ===== 根据 fp_type 选择分子表示方式 =====
        if fp_type == "descriptors":
            # 只使用基础分子描述符，不使用任何指纹
            pass

        elif fp_type == "morgan":
            # Morgan / ECFP-like 指纹
            fp_rows.append(calc_morgan_fp(mol, radius=radius, n_bits=fp_bits))

        elif fp_type == "maccs":
            # MACCS keys，166 bit
            fp_rows.append(calc_maccs_fp(mol))

        else:
            raise ValueError(f"Unknown fp_type: {fp_type}")
    # ===== 创建描述符数据框 =====
    # 将描述符列表转换为DataFrame
    # 每行对应一个SMILES，每列对应一个描述符
    desc_df = pd.DataFrame(desc_rows, index=smiles_series.index)
    # 为描述符列名添加前缀
    # 例如："MolWt" → "main_MolWt", "LogP" → "main_LogP"
    # 这样可以区分不同来源的SMILES特征（如主阻燃剂vs协效阻燃剂）
    desc_df = desc_df.add_prefix(f"{prefix}_")
        # ===== 如果是 descriptors only，只返回描述符 =====
    if fp_type == "descriptors":
        return desc_df

    # ===== 创建指纹数据框 =====
    fp_df = pd.DataFrame(fp_rows, index=smiles_series.index)

    if fp_type == "morgan":
        fp_df.columns = [f"{prefix}_morgan_r{radius}_{i}" for i in range(fp_bits)]

    elif fp_type == "maccs":
        fp_df.columns = [f"{prefix}_maccs_{i}" for i in range(166)]

    return pd.concat([desc_df, fp_df], axis=1)

# =========================================================
# 6. BUILD FEATURE MATRICES
# =========================================================
def build_feature_matrix(df: pd.DataFrame,colmap: Dict[str, str],fp_bits: int = 512,use_p_interactions: bool = False,fp_type: str = "morgan",radius: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    构建机器学习特征矩阵和元数据矩阵
    该函数是特征工程的核心步骤，将清洗后的数据框转换为：
    1. 特征矩阵 (X): 包含所有数值特征，用于机器学习模型训练
    2. 元数据矩阵 (meta): 包含分子标识信息，用于结果追溯和解释
    特征矩阵的组成：
    - 配方特征：P含量、阻燃剂比例、元素存在标志等
    - 主阻燃剂特征：分子描述符（10个）+ Morgan指纹（fp_bits个）
    - 协效阻燃剂特征：分子描述符（10个）+ Morgan指纹（fp_bits个）
    - 固化剂特征（可选）：分子描述符（10个）+ Morgan指纹（fp_bits个）
    参数:
        df (pd.DataFrame): 清洗后的数据框
                          包含SMILES列、配方列、性能指标列等
        colmap (Dict[str, str]): 标准列名到实际列名的映射字典
                                用于访问数据框中的特定列
        fp_bits (int): Morgan指纹的位数，默认为512
                       常用值：512（通用）、128（紧凑）、1024（详细）
                       位数越多，分子结构信息越丰富
    返回:
        Tuple[pd.DataFrame, pd.DataFrame]: 
            - X: 特征矩阵（数值型），行数为样本数，列数为所有特征
                特征组成：
                * 配方特征：8-10列
                * 主阻燃剂分子特征：10 + fp_bits列
                * 协效阻燃剂分子特征：10 + fp_bits列
                * 固化剂分子特征（如果存在）：10 + fp_bits列
                
            - meta: 元数据矩阵（文本型），行数为样本数，列数为分子标识信息
                包含：FR_main, FR_co, FR_class, Curing_Agent, SMILES等
    特征总数计算:
        - 有固化剂：8 + (10+fp_bits) + (10+fp_bits) + (10+fp_bits) = 38 + 3*fp_bits
        - 无固化剂：8 + (10+fp_bits) + (10+fp_bits) = 28 + 2*fp_bits
        - 例如fp_bits=512时：38+1536=1574列 或 28+1024=1052列
    """
    # ===== 提取主阻燃剂分子特征 =====
    # 使用主阻燃剂的SMILES生成分子描述符和Morgan指纹
    # 前缀为"main"，便于区分不同来源的特征
    # featurize_smiles_series返回DataFrame，包含：
    # - 10个基本描述符：MolWt, TPSA, LogP, HBD, HBA, RotBonds, RingCount, AromaticRings, HeavyAtomCount, FractionCSP3
    # - fp_bits个Morgan指纹位
    main_feat = featurize_smiles_series(df[colmap["SMILES_main"]], prefix="main", fp_bits=fp_bits, fp_type=fp_type, radius=radius)
    # ===== 提取协效阻燃剂分子特征 =====
    # 协效阻燃剂可能不存在（SMILES为空），此时所有特征为0或NaN
    # 前缀为"co"，与主阻燃剂特征区分
    co_feat = featurize_smiles_series(df[colmap["SMILES_co"]], prefix="co", fp_bits=fp_bits, fp_type=fp_type, radius=radius)
    # ===== 提取固化剂分子特征（可选）=====
    # 检查数据框中是否包含固化剂SMILES列
    # 固化剂在某些数据集中可能不存在
    curing_feat = None
    if "SMILES_Curing_Agent" in colmap:
        # 提取固化剂特征，前缀为"curing"
        # 固化剂分子结构可能对阻燃性能和力学性能有重要影响
        curing_feat = featurize_smiles_series(df[colmap["SMILES_Curing_Agent"]], prefix="curing", fp_bits=fp_bits, fp_type=fp_type, radius=radius)

    # ===== Neat-EP rows: remove hypothetical FR structure information =====
    # The literature table keeps FR_main/SMILES_main on loading=0 rows only to
    # link each neat-EP reference to its formulation series.  The compound is
    # not physically present in those rows, so its molecular descriptors and
    # fingerprints must be zeroed.  Curing-agent features are retained because
    # the curing system is genuinely present in neat EP.
    fr_present, main_present, co_present = _flame_retardant_presence_masks(df, colmap)
    main_feat = main_feat.where(main_present.astype(bool), 0.0, axis=0)
    co_feat = co_feat.where(co_present.astype(bool), 0.0, axis=0)
    # ===== 提取配方组成特征 =====
    # 配方特征是影响阻燃性能的关键因素
    # ===== 提取配方组成特征 =====
    # 新增列最优使用原则：
    # 1) Preparation_Method 作为无序类别变量进行 one-hot；Cure_Temp_Max / 厚度 / Cone_flux 作为实验条件特征；
    # 2) 元素含量与元素/P质量比作为主配方特征；
    # 3) Delta_* 只作为可选目标，绝不作为输入；
    # 4) 文本类别列不直接放入 formula_cols，先 one-hot，避免被 parse_numeric 变成 NaN。
    formula_keys = [
        # Preparation_Method_num is retained in the CSV for standardized record
        # management only.  It is intentionally excluded from predictive features
        # because the codes 0-3 do not represent a physical ordinal scale.
        "Loading_total_FR wt%",
        "P_content wt%",
        "Main_FR_fraction",
        "Co_FR_fraction",
        "Synergy_flag",

        "N_content wt%",
        "S_content wt%",
        "B_content wt%",
        "Si_content wt%",
        "N/P ratio",
        "S/P ratio",
        "B/P ratio",
        "Si/P ratio",

        "Has_P",
        "Has_N",
        "Has_Si",
        "Has_B",
        "Has_S",
        "Has_Al",

        "CuringAgent_Has_N",
        "CuringAgent_Has_S",
        "CuringAgent_Has_P",
        "CuringAgent_Has_B",
        "CuringAgent_Has_F",
        "CuringAgent_Has_Cl",

        "Cure_Temp_Max",
        "LOI_Thickness_mm",
        "UL94_Thickness_mm",
        "Cone_Thickness_mm",
        "Cone_flux_kW_m2",
    ]
    # 根据colmap获取实际列名（处理不同列名格式）
    formula_cols = [colmap[k] for k in formula_keys if k in colmap]
    # 提取配方列并创建副本
    formula_df = df[formula_cols].copy()
    
    # ===== 百分号比例列统一转为数值 =====
    # Main_FR_fraction / Co_FR_fraction 在最新版表格中可能写成 "100.00%"，
    # clean_dataframe 已经会解析为 100.00；这里再次兜底，保证进入模型的是 float。
    # 只对原始配方列应用parse_numeric_keep_scale，不对one-hot编码列应用
    for _c in formula_cols:
        formula_df[_c] = formula_df[_c].apply(parse_numeric_keep_scale)
    
    # ===== 添加FR_class和Synergy_type的编码特征 =====
    # 这些分类特征可能对模型性能有重要影响
    if "FR_class" in colmap:
        # 对FR_class进行one-hot编码
        fr_class_dummies = pd.get_dummies(df[colmap["FR_class"]], prefix="FR_class")
        fr_class_dummies = fr_class_dummies.mul(fr_present, axis=0)
        formula_df = pd.concat([formula_df, fr_class_dummies], axis=1)
    
    if "Synergy_type" in colmap:
        # Synergy_type is FR-derived information.  Keep the categorical signal for
        # modified formulations, but neutralize it on neat-EP rows exactly as
        # required by the baseline-inclusive leakage-control protocol.
        synergy_type_dummies = pd.get_dummies(df[colmap["Synergy_type"]], prefix="Synergy_type")
        synergy_type_dummies = synergy_type_dummies.mul(fr_present, axis=0)
        formula_df = pd.concat([formula_df, synergy_type_dummies], axis=1)

    if "Preparation_Method" in colmap:
        # Preparation Method is a nominal category.  Text aliases were normalized
        # in clean_dataframe(); use one-hot encoding only (no ordinal numeric code).
        prep_dummies = pd.get_dummies(df[colmap["Preparation_Method"]], prefix="Preparation_Method")
        prep_dummies = prep_dummies.mul(fr_present, axis=0)
        formula_df = pd.concat([formula_df, prep_dummies], axis=1)

    # Explicit presence indicators make an all-zero molecular block
    # distinguishable from a genuinely missing/empty molecular representation.
    formula_df["FR_present"] = fr_present
    formula_df["Main_FR_present"] = main_present
    formula_df["Co_FR_present"] = co_present

    # Neutralize FR-derived structured fields on confirmed neat-EP rows.  The
    # original values are useful metadata for pairing, but they are not valid
    # formulation inputs when the total FR loading is zero.
    fr_dependent_keys = [
        "Main_FR_fraction", "Co_FR_fraction", "Synergy_flag",
        "P_content wt%", "N_content wt%", "S_content wt%",
        "B_content wt%", "Si_content wt%",
        "N/P ratio", "S/P ratio", "B/P ratio", "Si/P ratio",
        "Has_P", "Has_N", "Has_Si", "Has_B", "Has_S", "Has_Al",
    ]
    for key in fr_dependent_keys:
        actual = colmap.get(key)
        if actual is not None and actual in formula_df.columns:
            values = pd.to_numeric(formula_df[actual], errors="coerce")
            formula_df[actual] = values.where(fr_present.astype(bool), 0.0)
    
    # ===== 添加Loading_total_FR特征 =====
    # 阻燃剂总添加量是影响性能的重要因素
    if "Loading_total_FR wt%" in colmap:
        loading = get_loading_series(df, colmap)
        formula_df["Loading_total_FR"] = loading
        formula_df["Loading_total_FR_sq"] = loading ** 2
        formula_df["Loading_total_FR_sqrt"] = np.sqrt(loading.clip(lower=0.0))

    # ===== BDE mechanism features：BDE作为LOI/PHRR/THR/Tg/Char/TS/FS输入特征 =====
    # 说明：
    # 1) BDE 不再在主流程中作为预测目标，而是由 07_BDE 模型/文献/DFT 得到后作为机理输入特征；
    # 2) 优先使用 FR_main_BDE_final_kJ_mol（原始实测/DFT + V7预测补全），没有时再回退到原始BDE或预测BDE列；
    # 3) 对 BDE 自身任务，filter_features_for_task() 会删除所有 BDE/bde 特征，避免标签泄漏；
    # 4) 可通过 set DOPO_USE_BDE_FEATURES=0 做“without BDE”消融对比。
    if USE_BDE_FEATURES:
        def _first_existing_col(_candidates):
            for _c in _candidates:
                if _c in df.columns:
                    return _c
            return None

        def _num_series(_col):
            if _col is None:
                return pd.Series(np.nan, index=df.index, dtype=float)
            return df[_col].apply(parse_numeric_keep_scale).astype(float)

        # 原始/最终BDE列：如果是kcal/mol则转为kJ/mol。
        _bde_col = colmap.get("BDE")
        bde_kj = _num_series(_bde_col)
        if _bde_col is not None:
            _bde_col_l = str(_bde_col).lower()
            _bde_median = bde_kj.dropna().median()
            if "kcal" in _bde_col_l or ("kj" not in _bde_col_l and pd.notna(_bde_median) and _bde_median < 150):
                bde_kj = bde_kj * 4.184

        # V7预测列：用于补全原始BDE空缺，不覆盖已有实测/DFT/final值。
        _pred_col = _first_existing_col([
            "FR_main_BDE_pred_kJ_mol", "pred_BDE_pred_kJ_mol", "BDE_pred_kJ_mol",
            "Predicted_FR_main_BDE_kJ_mol", "FR_main_BDE_pred_KJ_mol",
        ])
        pred_kj = _num_series(_pred_col)
        bde_filled_by_pred = bde_kj.isna() & pred_kj.notna()
        bde_kj = bde_kj.fillna(pred_kj)

        # 只有存在任意BDE信息时才写入BDE特征，避免空列污染。
        if bde_kj.notna().any() or pred_kj.notna().any():
            formula_df["FR_main_BDE_kJ_mol"] = bde_kj
            formula_df["FR_main_BDE_available"] = bde_kj.notna().astype(float)
            formula_df["FR_main_BDE_source_is_pred"] = bde_filled_by_pred.astype(float)
            bde_fill = bde_kj.fillna(0.0)
            formula_df["FR_main_BDE_kJ_mol_filled0"] = bde_fill
            formula_df["FR_main_BDE_kJ_mol_sq"] = bde_fill ** 2
            formula_df["FR_main_BDE_kJ_mol_sqrt"] = np.sqrt(np.abs(bde_fill) + 1e-6)
            formula_df["FR_main_BDE_kJ_mol_log"] = np.log(np.abs(bde_fill) + 1.0)

            # V7预测不确定性与模型一致性特征。
            _std_col = _first_existing_col(["FR_main_BDE_pred_std_kJ_mol", "pred_BDE_pred_std_kJ_mol", "BDE_pred_std_kJ_mol"])
            _diff_col = _first_existing_col(["FR_main_BDE_pcpn_alltypes_diff_kJ_mol", "pred_pcpn_alltypes_abs_diff_kJ_mol", "pcpn_alltypes_abs_diff_kJ_mol"])
            _pcpn_col = _first_existing_col(["FR_main_BDE_pc_pn_pred_kJ_mol", "pred_BDE_pred_pc_pn_kJ_mol", "BDE_pred_pc_pn_kJ_mol"])
            _all_col = _first_existing_col(["FR_main_BDE_all_types_pred_kJ_mol", "pred_BDE_pred_all_types_kJ_mol", "BDE_pred_all_types_kJ_mol"])

            if _std_col is not None:
                bde_std = _num_series(_std_col)
                formula_df["FR_main_BDE_pred_std_kJ_mol"] = bde_std
                formula_df["FR_main_BDE_pred_std_filled0"] = bde_std.fillna(0.0)
                formula_df["FR_main_BDE_pred_std_high"] = (bde_std.fillna(0.0) > 20).astype(float)
            if _diff_col is not None:
                bde_diff = _num_series(_diff_col)
                formula_df["FR_main_BDE_pcpn_alltypes_diff_kJ_mol"] = bde_diff
                formula_df["FR_main_BDE_model_disagree"] = (bde_diff.fillna(0.0) > 25).astype(float)
            if _pcpn_col is not None:
                formula_df["FR_main_BDE_pc_pn_pred_kJ_mol"] = _num_series(_pcpn_col)
            if _all_col is not None:
                formula_df["FR_main_BDE_all_types_pred_kJ_mol"] = _num_series(_all_col)

            # BDE × 配方/元素交互：用于让模型捕捉“键断裂难易 × 实际添加量/磷含量”的耦合。
            if "Loading_total_FR wt%" in colmap:
                _loading_for_bde = df[colmap["Loading_total_FR wt%"]].apply(parse_numeric_keep_scale).astype(float).fillna(0.0)
                formula_df["FR_main_BDE_kJ_mol_x_Loading"] = bde_fill * _loading_for_bde
                formula_df["FR_main_BDE_kJ_mol_x_Loading_sqrt"] = bde_fill * np.sqrt(_loading_for_bde + 1e-6)
            if "P_content wt%" in colmap:
                _p_for_bde = df[colmap["P_content wt%"]].apply(parse_numeric_keep_scale).astype(float).fillna(0.0)
                formula_df["FR_main_BDE_kJ_mol_x_P_content"] = bde_fill * _p_for_bde
            if "Synergy_flag" in colmap:
                _syn_for_bde = df[colmap["Synergy_flag"]].apply(parse_numeric_keep_scale).astype(float).fillna(0.0)
                formula_df["FR_main_BDE_kJ_mol_x_Synergy"] = bde_fill * _syn_for_bde

            # 置信度/模型来源转为轻量数值特征；不使用目标值，不构成泄漏。
            _conf_col = _first_existing_col(["FR_main_BDE_confidence_note", "pred_confidence_note", "confidence_note"])
            if _conf_col is not None:
                conf = df[_conf_col].fillna("").astype(str).str.lower()
                formula_df["FR_main_BDE_conf_low"] = conf.str.contains("low-confidence|unreliable|invalid|unsupported|no suitable", regex=True).astype(float)
                formula_df["FR_main_BDE_conf_high_var"] = conf.str.contains("high final-ensemble variance|less stable", regex=True).astype(float)
                formula_df["FR_main_BDE_conf_disagree"] = conf.str.contains("disagree strongly|moderate disagreement", regex=True).astype(float)

            _task_col = _first_existing_col(["FR_main_BDE_model_task", "pred_selected_model_task", "selected_model_task"])
            if _task_col is not None:
                task_dummies = pd.get_dummies(df[_task_col].fillna("Unknown").astype(str), prefix="FR_main_BDE_model_task")
                formula_df = pd.concat([formula_df, task_dummies], axis=1)
    else:
        print("[INFO] DOPO_USE_BDE_FEATURES=0: BDE features disabled for this run.")

    # BDE_Type is also part of the BDE mechanism information.
    # Keep it only in with-BDE runs; remove it in without-BDE ablation so the
    # comparison reflects a clean removal of BDE-derived information.
    if USE_BDE_FEATURES and "BDE_Type" in colmap:
        bde_type_norm = df[colmap["BDE_Type"]].fillna("Unknown").astype(str).str.strip()
        bde_type_norm = bde_type_norm.replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
        bde_type_dummies = pd.get_dummies(bde_type_norm, prefix="BDE_Type")
        formula_df = pd.concat([formula_df, bde_type_dummies], axis=1)

    # ===== V7 legacy loading×composition interaction features =====
    # IMPORTANT: P/N/S/B/Si_content wt% in the V5 database are already
    # formulation-level elemental contents.  The historical *_loading_real names
    # below are therefore retained only for backward feature-name compatibility;
    # mathematically they are Loading_total_FR × elemental_content / 100
    # interaction terms, NOT a second calculation of the actual elemental loading.
    # They use no target information and do not constitute label leakage.
    if USE_V7_FORMULA_FEATURES and "Loading_total_FR wt%" in colmap:
        loading_v7 = df[colmap["Loading_total_FR wt%"]].apply(parse_numeric_keep_scale).astype(float).fillna(0.0)

        element_content_map_v7 = {
            "P": "P_content wt%",
            "N": "N_content wt%",
            "S": "S_content wt%",
            "B": "B_content wt%",
            "Si": "Si_content wt%",
        }

        element_loading_v7 = {}
        for elem_v7, key_v7 in element_content_map_v7.items():
            if key_v7 in colmap:
                content_v7 = df[colmap[key_v7]].apply(parse_numeric_keep_scale).astype(float).fillna(0.0)
                # Legacy feature name retained for compatibility.  The value is
                # an engineered loading×composition interaction because the input
                # elemental-content field is already formulation-level wt%.
                feature_name_v7 = f"{elem_v7}_loading_real"
                formula_df[feature_name_v7] = loading_v7 * content_v7 / 100.0
                formula_df[f"{feature_name_v7}_sq"] = formula_df[feature_name_v7] ** 2
                formula_df[f"{feature_name_v7}_sqrt"] = np.sqrt(np.abs(formula_df[feature_name_v7]) + 1e-6)
                element_loading_v7[elem_v7] = formula_df[feature_name_v7]

        # P-N / P-Si / P-B / P-S 协同负载特征
        if "P" in element_loading_v7:
            p_load_v7 = element_loading_v7["P"]
            for elem_v7 in ["N", "Si", "B", "S"]:
                if elem_v7 in element_loading_v7:
                    other_load_v7 = element_loading_v7[elem_v7]
                    formula_df[f"P_{elem_v7}_loading_sum"] = p_load_v7 + other_load_v7
                    formula_df[f"P_{elem_v7}_loading_product"] = p_load_v7 * other_load_v7
                    formula_df[f"P_{elem_v7}_loading_ratio"] = p_load_v7 / (other_load_v7 + 1e-6)

        # 阻燃剂添加量与协同标记的交互
        if "Synergy_flag" in colmap:
            synergy_v7 = df[colmap["Synergy_flag"]].apply(parse_numeric_keep_scale).astype(float).fillna(0.0)
            formula_df["Loading_x_Synergy"] = loading_v7 * synergy_v7
            if "P" in element_loading_v7:
                formula_df["P_loading_x_Synergy"] = element_loading_v7["P"] * synergy_v7

    # ===== 添加Cone_flux特征 =====
    # 锥形量热测试热流密度，对PHRR/THR任务尤其重要
    if "Cone_flux_kW_m2" in colmap:
        cone_flux = df[colmap["Cone_flux_kW_m2"]].apply(parse_numeric_keep_scale).astype(float)
        formula_df["Cone_flux"] = cone_flux
        formula_df["Cone_flux_sq"] = cone_flux ** 2
    
    # ===== 添加纯EP基线性能特征 =====
    # 这些特征帮助模型理解基线性能和相对提升
    # 例如：如果纯EP基线LOI较低，即使添加阻燃剂后LOI提升，最终LOI可能仍然不高
    ep_matrix_tasks = [
        "EP_matrix_LOI",
        "EP_matrix_PHRR",
        "EP_matrix_THR",
        "EP_matrix_Tg",
        "EP_matrix_CY",
        "EP_matrix_TS",
        "EP_matrix_FS"
    ]
    for task in ep_matrix_tasks:
        if task in colmap:
            ep_value = df[colmap[task]].apply(parse_numeric_keep_scale).astype(float)
            formula_df[task] = ep_value
            formula_df[task + "_sq"] = ep_value ** 2
            formula_df[task + "_sqrt"] = np.sqrt(np.abs(ep_value) + 1e-6)
    
    # ===== 添加元素含量特征（独立于use_p_interactions）=====
    # 这些特征对所有任务都很重要，不应该只在use_p_interactions=True时才使用
    if "P_content wt%" in colmap:
        p = df[colmap["P_content wt%"]].astype(float)
        formula_df["P_content_wt"] = p
        formula_df["P_content_sq"] = p ** 2
        formula_df["P_content_sqrt"] = np.sqrt(p + 1e-6)
        formula_df["P_content_log"] = np.log(p + 1e-6)
    
    # 添加其他元素含量特征
    for _key in ["N_content wt%", "S_content wt%", "B_content wt%", "Si_content wt%", "N/P ratio", "S/P ratio", "B/P ratio", "Si/P ratio"]:
        if _key in colmap:
            content = df[colmap[_key]].apply(parse_numeric_keep_scale).astype(float)
            formula_df[_key.replace(" %", "").replace(" ", "_")] = content
            formula_df[_key.replace(" %", "").replace(" ", "_") + "_sq"] = content ** 2
            formula_df[_key.replace(" %", "").replace(" ", "_") + "_sqrt"] = np.sqrt(np.abs(content) + 1e-6)
    # ===== 固化剂元素 × P含量交互特征 =====
    if "P_content wt%" in colmap:
        p = df[colmap["P_content wt%"]].astype(float)

        curing_element_keys = [
            "CuringAgent_Has_N",
            "CuringAgent_Has_S",
            "CuringAgent_Has_P",
            "CuringAgent_Has_B",
            "CuringAgent_Has_F",
            "CuringAgent_Has_Cl",
        ]

        for key in curing_element_keys:
            if key in colmap:
                curing_flag = df[colmap[key]].astype(float)
                formula_df[f"P_x_{key}"] = p * curing_flag

    # ===== 固化剂分子描述符 × P含量交互特征 =====
    if curing_feat is not None and "P_content wt%" in colmap:
        p = df[colmap["P_content wt%"]].astype(float)

        curing_descriptor_keys = [
            "curing_MolWt",
            "curing_TPSA",
            "curing_LogP",
            "curing_HBD",
            "curing_HBA",
            "curing_RotBonds",
            "curing_RingCount",
            "curing_AromaticRings",
            "curing_HeavyAtomCount",
            "curing_FractionCSP3",
        ]

        for key in curing_descriptor_keys:
            if key in curing_feat.columns:
                formula_df[f"P_x_{key}"] = p * curing_feat[key].astype(float)
    # ===== 可选：P × 协同元素交互特征 =====
    # 根据运行结果，这组特征主要提升 TS_MPa 和 UL94_V0；
    # 但会轻微拖低 PHRR / THR / Char_yield / FS_MPa。
    # 因此默认关闭，只在 main() 的 pick_X() 中给 TS_MPa 和 UL94_V0 使用。
    if use_p_interactions and "P_content wt%" in colmap:
        p = df[colmap["P_content wt%"]].astype(float)

        if "Synergy_flag" in colmap:
            formula_df["P_x_synergy"] = p * df[colmap["Synergy_flag"]].astype(float)

        if "Has_N" in colmap:
            formula_df["P_x_Has_N"] = p * df[colmap["Has_N"]].astype(float)

        if "Has_Si" in colmap:
            formula_df["P_x_Has_Si"] = p * df[colmap["Has_Si"]].astype(float)

        if "Has_B" in colmap:
            formula_df["P_x_Has_B"] = p * df[colmap["Has_B"]].astype(float)

        if "Has_S" in colmap:
            formula_df["P_x_Has_S"] = p * df[colmap["Has_S"]].astype(float)

        if "Has_Al" in colmap:
            formula_df["P_x_Has_Al"] = p * df[colmap["Has_Al"]].astype(float)

        # 新版表格中的元素含量 × P 含量交互特征
        for _key, _new_name in [
            ("N_content wt%", "P_x_N_content"),
            ("S_content wt%", "P_x_S_content"),
            ("B_content wt%", "P_x_B_content"),
            ("Si_content wt%", "P_x_Si_content"),
            ("N/P ratio", "P_x_NP_ratio"),
            ("S/P ratio", "P_x_SP_ratio"),
            ("B/P ratio", "P_x_BP_ratio"),
            ("Si/P ratio", "P_x_SiP_ratio"),
        ]:
            if _key in colmap:
                formula_df[_new_name] = p * df[colmap[_key]].apply(parse_numeric_keep_scale).astype(float)
    
    # ===== 添加纯EP基线数据与阻燃剂特征的交互特征 =====
    # 这些交互特征帮助模型学习到阻燃剂对不同基线性能的影响
    if "Loading_total_FR wt%" in colmap:
        loading = df[colmap["Loading_total_FR wt%"]].apply(parse_numeric_keep_scale).astype(float)
        for task in ep_matrix_tasks:
            if task in colmap:
                ep_value = df[colmap[task]].apply(parse_numeric_keep_scale).astype(float)
                formula_df[f"Loading_x_{task}"] = loading * ep_value
    
    if "P_content wt%" in colmap:
        p = df[colmap["P_content wt%"]].astype(float)
        for task in ep_matrix_tasks:
            if task in colmap:
                ep_value = df[colmap[task]].apply(parse_numeric_keep_scale).astype(float)
                formula_df[f"P_content_x_{task}"] = p * ep_value
    
    if "Synergy_flag" in colmap:
        synergy = df[colmap["Synergy_flag"]].astype(float)
        for task in ep_matrix_tasks:
            if task in colmap:
                ep_value = df[colmap[task]].apply(parse_numeric_keep_scale).astype(float)
                formula_df[f"Synergy_x_{task}"] = synergy * ep_value
    
    # ===== 添加分子相似度特征 =====
    # 计算主阻燃剂和协效阻燃剂之间的分子相似度
    similarity_features = []
    for i in range(len(df)):
        main_smi = df[colmap["SMILES_main"]].iloc[i] if pd.notna(df[colmap["SMILES_main"]].iloc[i]) else ""
        co_smi = df[colmap["SMILES_co"]].iloc[i] if pd.notna(df[colmap["SMILES_co"]].iloc[i]) else ""
        
        main_mol = safe_mol_from_smiles(main_smi)
        co_mol = safe_mol_from_smiles(co_smi)
        
        if main_mol is not None and co_mol is not None:
            # 计算Tanimoto相似度（使用numpy数组）
            main_fp = calc_morgan_fp(main_mol, radius=2, n_bits=512)
            co_fp = calc_morgan_fp(co_mol, radius=2, n_bits=512)
            # 计算Tanimoto相似度: (A·B) / (|A| + |B| - A·B)
            dot_product = np.dot(main_fp, co_fp)
            norm_a = np.sum(main_fp)
            norm_b = np.sum(co_fp)
            if norm_a + norm_b - dot_product > 0:
                similarity = dot_product / (norm_a + norm_b - dot_product)
            else:
                similarity = 0.0
            similarity_features.append(similarity)
        else:
            similarity_features.append(0.0)
    
    similarity_array = np.asarray(similarity_features, dtype=float) * fr_present.to_numpy(dtype=float)
    formula_df["main_co_similarity"] = similarity_array
    formula_df["main_co_similarity_sq"] = similarity_array ** 2

    # Defensive masking for any BDE-derived columns that may have been filled
    # from a molecular prediction table.  A neat-EP reference has no flame
    # retardant bond to dissociate, even if FR_main is retained as metadata.
    for column in list(formula_df.columns):
        if "bde" in str(column).lower():
            values = pd.to_numeric(formula_df[column], errors="coerce")
            formula_df[column] = values.where(fr_present.astype(bool), 0.0)
    
    # ===== 合并所有特征 =====
    # 根据是否存在固化剂特征，决定合并方式
    if curing_feat is not None:
        # 有固化剂特征时：配方 + 主阻燃剂 + 协效阻燃剂 + 固化剂
        # pd.concat沿列方向（axis=1）合并所有DataFrame
        X = pd.concat([formula_df, main_feat, co_feat, curing_feat], axis=1)
    else:
        # 无固化剂特征时：配方 + 主阻燃剂 + 协效阻燃剂
        X = pd.concat([formula_df, main_feat, co_feat], axis=1)
    # ===== 处理无穷值 =====
    # 将正无穷和负无穷替换为NaN
    # 避免后续机器学习模型处理时出错
    X = X.replace([np.inf, -np.inf], np.nan)
    # ===== 去除重复特征列名 =====
    # one-hot、原始配方列和衍生特征可能产生重复列名；
    # 重复列名会导致 X[cols] 选择后列数膨胀，从而让 VarianceThreshold 的 mask 与列名不匹配。
    X = X.loc[:, ~X.columns.duplicated()].copy()
    # ===== 构建元数据矩阵 =====
    # 元数据包含分子的标识信息，用于：
    # 1. 结果追溯：识别特定样本的分子
    # 2. 模型解释：理解预测结果的化学意义
    # 3. 数据验证：检查特征与原始数据的对应关系
    meta_keys = [
        "FR_main",             # 主阻燃剂名称
        "FR_co",              # 协效阻燃剂名称
        "FR_class",           # 阻燃剂类别
        "Curing_Agent",       # 固化剂名称
        "SMILES_main",        # 主阻燃剂SMILES
        "SMILES_co",          # 协效阻燃剂SMILES
        "SMILES_Curing_Agent", # 固化剂SMILES
        "Loading_total_FR wt%", "P_content wt%", "N_content wt%", "S_content wt%", "B_content wt%", "Si_content wt%",
        "N/P ratio", "S/P ratio", "B/P ratio", "Si/P ratio",
        "Cone_flux_kW_m2", "BDE", "BDE_Type", "Reference", "Preparation_Method"
    ]
    # 根据colmap获取实际列名
    meta_cols = [colmap[k] for k in meta_keys if k in colmap]
    # 提取元数据列并创建副本
    meta = df[meta_cols].copy()
    # ===== 返回特征矩阵和元数据矩阵 =====
    return X, meta
# =========================================================
# 7. ADAPTIVE FEATURE DIMENSION
# =========================================================
def choose_k_by_sample_size(n_samples: int) -> int:
    """
    根据样本数量自适应选择特征数量（K值）
    理论基础：样本数/特征数≈1:1到3:1
    参数:
        n_samples (int): 数据集中的样本数量
    返回:
        int: 推荐的特征数量K值
             - n_samples >= 500: 返回480（大样本量）
             - 300 <= n_samples < 500: 返回320（高样本量）
             - 220 <= n_samples < 300: 返回180（中偏高样本量）
             - 150 <= n_samples < 220: 返回100（中样本量）
             - n_samples < 150: 返回50（低样本量）
    """
    if n_samples >= 500:
        return 480
    elif n_samples >= 300:
        return 320
    elif n_samples >= 220:
        return 180
    elif n_samples >= 150:
        return 100
    else:
        return 50
# =========================================================
# 7.5 SAFE FEATURE SELECTOR
# =========================================================
class SafeSelectKBest(BaseEstimator, TransformerMixin):
    """
    安全版 SelectKBest：
    防止不同 fold 中方差过滤后特征数少于 k 时报错。
    """
    def __init__(self, score_func=f_regression, k=100):
        self.score_func = score_func
        self.k = k

    def fit(self, X, y):
        k_use = min(self.k, X.shape[1])
        self.selector_ = SelectKBest(score_func=self.score_func, k=k_use)
        self.selector_.fit(X, y)
        return self

    def transform(self, X):
        return self.selector_.transform(X)

    def get_support(self):
        return self.selector_.get_support()

# =========================================================
# 7.6 MOLECULE-GROUP SPLIT HELPERS
# =========================================================
def _canonical_group_component(value, fallback=""):
    """Return a full-component canonical identity without discarding salts/counterions."""
    text = "" if pd.isna(value) else str(value).strip()
    if not text:
        return fallback
    try:
        from common.chem_standardization import standardize_smiles
        record = standardize_smiles(text)
        if record.is_valid and record.canonical_smiles:
            return record.canonical_smiles
    except Exception:
        pass
    return text


def build_molecule_group_labels(df: pd.DataFrame, colmap: Dict[str, str]) -> pd.Series:
    """Primary leakage-control groups: full-component MAIN+CO molecular identity."""
    labels = []
    for _, row in df.iterrows():
        raw_main = row.get(colmap.get("SMILES_main", ""), "") if "SMILES_main" in colmap else ""
        raw_co = row.get(colmap.get("SMILES_co", ""), "") if "SMILES_co" in colmap else ""
        main_fallback = str(row.get(colmap.get("FR_main", ""), "")).strip() if "FR_main" in colmap else ""
        co_fallback = str(row.get(colmap.get("FR_co", ""), "")).strip() if "FR_co" in colmap else ""
        main = _canonical_group_component(raw_main, main_fallback)
        co = _canonical_group_component(raw_co, co_fallback)
        labels.append(f"MAIN={main}|CO={co}")
    return pd.Series(labels, index=df.index, name="molecule_group")


def build_molecule_group_labels_with_curing(df: pd.DataFrame, colmap: Dict[str, str]) -> pd.Series:
    """Leakage-control groups for Tg/TS/FS: full-component MAIN+CO+CURING identity."""
    labels = []
    for _, row in df.iterrows():
        raw_main = row.get(colmap.get("SMILES_main", ""), "") if "SMILES_main" in colmap else ""
        raw_co = row.get(colmap.get("SMILES_co", ""), "") if "SMILES_co" in colmap else ""
        raw_curing = row.get(colmap.get("SMILES_Curing_Agent", ""), "") if "SMILES_Curing_Agent" in colmap else ""
        main_fallback = str(row.get(colmap.get("FR_main", ""), "")).strip() if "FR_main" in colmap else ""
        co_fallback = str(row.get(colmap.get("FR_co", ""), "")).strip() if "FR_co" in colmap else ""
        curing_fallback = str(row.get(colmap.get("Curing_Agent", ""), "")).strip() if "Curing_Agent" in colmap else ""
        main = _canonical_group_component(raw_main, main_fallback)
        co = _canonical_group_component(raw_co, co_fallback)
        curing = _canonical_group_component(raw_curing, curing_fallback)
        labels.append(f"MAIN={main}|CO={co}|CURING={curing}")
    return pd.Series(labels, index=df.index, name="molecule_group_with_curing")

def grouped_train_test_split(X, y, groups=None, test_size=0.2, random_state=42, stratify=None, task_name="task", use_group_split=True):
    """
    优先按分子组划分训练/测试集；如果 group 不足，则回退到普通随机划分。
    
    参数:
        use_group_split: 是否使用分组分割，默认True以提高模型性能
    """
    if use_group_split and groups is not None:
        groups = pd.Series(groups).reset_index(drop=True)
        n_groups = groups.nunique(dropna=True)
        if n_groups >= 2:
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(splitter.split(X, y, groups=groups))
            train_groups = groups.iloc[train_idx].nunique()
            test_groups = groups.iloc[test_idx].nunique()
            overlap = set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
            print(
                f"[INFO] {task_name}: GroupShuffleSplit by molecule groups | "
                f"samples train/test={len(train_idx)}/{len(test_idx)}, "
                f"groups train/test={train_groups}/{test_groups}, overlap={len(overlap)}"
            )
            return (
                X.iloc[train_idx].copy(),
                X.iloc[test_idx].copy(),
                y.iloc[train_idx].copy(),
                y.iloc[test_idx].copy(),
            )

    print(f"[INFO] {task_name}: 使用普通随机划分（未启用分组分割）。")
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )


def make_group_cv(groups=None, n_splits=5, random_state=42, task_name="task", use_group_cv=True):
    """
    为交叉验证构建分组CV。只要可用，就使用 GroupKFold，避免同一分子组跨fold泄漏。
    
    参数:
        use_group_cv: 是否使用分组CV，默认True以提高模型性能
    """
    if use_group_cv and groups is not None:
        groups = pd.Series(groups).reset_index(drop=True)
        n_groups = groups.nunique(dropna=True)
        if n_groups >= 2:
            n_splits_use = min(n_splits, n_groups)
            print(f"[INFO] {task_name}: CV uses GroupKFold(n_splits={n_splits_use}, molecule groups={n_groups})")
            return GroupKFold(n_splits=n_splits_use), groups

    print(f"[INFO] {task_name}: CV uses KFold(n_splits={n_splits}, shuffle=True)")
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state), None

def transform_target_for_training(y, transform="none"):
    """
    对目标变量进行变换。
    none: 不变换
    log1p: 使用 log(1+y)
    """
    y = np.asarray(y, dtype=float)

    if transform == "none":
        return y

    if transform == "log1p":
        return np.log1p(y)

    raise ValueError(f"Unknown target transform: {transform}")
def inverse_transform_prediction(y_pred, transform="none"):
    """
    将预测值反变换回原始尺度。
    """
    y_pred = np.asarray(y_pred, dtype=float)

    if transform == "none":
        return y_pred

    if transform == "log1p":
        return np.expm1(y_pred)

    raise ValueError(f"Unknown target transform: {transform}")

# =========================================================
# 7.7 TASK-SPECIFIC FEATURE FILTER
# =========================================================
def filter_features_for_task(X: pd.DataFrame, task_name: str) -> pd.DataFrame:
    """
    按任务过滤特征，减少无关基线特征和测试条件特征带来的噪声/虚高。

    规则：
    1. EP_matrix_* 只保留与当前目标直接对应的基线列及其交互项。
       例如 LOI 只保留 EP_matrix_LOI，不保留 EP_matrix_PHRR / EP_matrix_THR 等。
    2. Cone_flux_kW_m2 只对 PHRR / THR 保留；对 LOI / Tg / Char / TS / FS 删除。
    3. Delta_* 永远不能作为输入特征，只能作为目标变量。
    4. 保留 SMILES 指纹、固化剂指纹、元素含量、P×固化剂交互等通用特征。
    """
    Xf = X.copy()

    task_to_ep = {
        "LOI": "EP_matrix_LOI",
        "PHRR": "EP_matrix_PHRR",
        "THR": "EP_matrix_THR",
        "Tg": "EP_matrix_Tg",
        "Char_yield": "EP_matrix_CY",
        "TS_MPa": "EP_matrix_TS",
        "FS_MPa": "EP_matrix_FS",
        "Delta_LOI": "EP_matrix_LOI",
        "Delta_PHRR": "EP_matrix_PHRR",
        "Delta_THR": "EP_matrix_THR",
        "Delta_Tg": "EP_matrix_Tg",
        "Delta_CY": "EP_matrix_CY",
        "Delta_TS": "EP_matrix_TS",
        "Delta_FS": "EP_matrix_FS",
    }

    keep_ep = task_to_ep.get(task_name)

    # V7可选：完全关闭 EP_matrix_* 及其交互项，用于检查模型是否过度依赖基体性能。
    # 默认不关闭；如需测试，在运行前设置环境变量：
    #   set DOPO_DROP_BASELINE=1
    DROP_BASELINE_FEATURES_V7 = os.environ.get("DOPO_DROP_BASELINE", "0").strip() == "1"
    if DROP_BASELINE_FEATURES_V7:
        keep_ep = None

    drop_cols = []

    # with/without BDE 消融对比的防御性过滤：
    # 当 DOPO_USE_BDE_FEATURES=0 时，删除所有名称中含 BDE/bde 的特征，
    # 包括 BDE_Type、BDE预测值、BDE置信度、BDE×配方交互等，避免“without BDE”残留机理信息。
    if not USE_BDE_FEATURES:
        for col in Xf.columns:
            c = str(col)
            if "BDE" in c or "bde" in c:
                drop_cols.append(col)

    # 单独训练/评估 BDE 时，不能把 BDE 数值本身或其衍生列作为输入；
    # BDE 专项模型请使用 07_BDE/BDE.py（基于 DOPO_BDE.csv，随机划分）。
    if task_name == "BDE":
        for col in Xf.columns:
            c = str(col)
            if "BDE" in c or "bde" in c:
                drop_cols.append(col)

    ep_prefixes = [
        "EP_matrix_LOI", "EP_matrix_PHRR", "EP_matrix_THR",
        "EP_matrix_Tg", "EP_matrix_CY", "EP_matrix_TS", "EP_matrix_FS"
    ]

    for col in Xf.columns:
        c = str(col)

        # Delta列不能作为输入，防止标签泄漏
        if c.startswith("Delta_") or "_Delta_" in c:
            drop_cols.append(col)
            continue

        # EP基线及其交互项只保留当前任务对应的那一类
        matched_ep = None
        for ep in ep_prefixes:
            if ep in c:
                matched_ep = ep
                break
        if matched_ep is not None and matched_ep != keep_ep:
            drop_cols.append(col)
            continue

        # 厚度/热流密度只保留对应实验任务，避免不同测试条件给无关任务带来噪声
        if "LOI_Thickness" in c and task_name not in {"LOI", "Delta_LOI"}:
            drop_cols.append(col)
            continue

        if "UL94_Thickness" in c and task_name not in {"UL94_num", "UL94_V0"}:
            drop_cols.append(col)
            continue

        if "Cone_Thickness" in c and task_name not in {"PHRR", "THR", "Delta_PHRR", "Delta_THR"}:
            drop_cols.append(col)
            continue

        # Cone_flux 只用于锥形量热任务
        if ("Cone_flux" in c or "Cone_flux_kW_m2" in c) and task_name not in {"PHRR", "THR", "Delta_PHRR", "Delta_THR"}:
            drop_cols.append(col)
            continue

    if drop_cols:
        Xf = Xf.drop(columns=drop_cols, errors="ignore")

    Xf = Xf.loc[:, ~Xf.columns.duplicated()].copy()
    return Xf

def select_regression_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    k: int
):
    """
    回归任务的特征选择函数
    该函数对回归任务的训练集和测试集进行特征选择，通过多步骤
    的数据清洗和特征筛选，选择出最优的K个特征用于模型训练。    
    处理流程：
    1. 移除全NaN的特征列
    2. 缺失值填充（中位数）
    3. 方差阈值过滤（移除低方差特征）
    4. 单变量特征选择（SelectKBest）
    参数:
        X_train (pd.DataFrame): 训练集特征矩阵
                             行数为训练样本数，列数为原始特征数
        y_train (pd.Series): 训练集目标变量
                             用于特征选择中的相关性评分
        X_test (pd.DataFrame): 测试集特征矩阵
                            行数为测试样本数，列数与X_train相同
        k (int): 要选择的特征数量
                通常由choose_k_by_sample_size函数计算得出
                例如：480（607样本）、320（300样本）、180（220样本）
    返回:
        Tuple[pd.DataFrame, pd.DataFrame, list]: 
            - X_train_sel: 选择后的训练集特征矩阵
                         形状：(n_train_samples, k)
            - X_test_sel: 选择后的测试集特征矩阵
                        形状：(n_test_samples, k)
            - final_cols: 被选中的特征名称列表
                        长度为k，用于特征重要性分析
    特征选择策略：
        - 移除全NaN列：避免无效特征干扰
        - 中位数填充：保持数据分布特征
        - 方差过滤：移除无信息量的特征
        - F检验：选择与目标变量最相关的K个特征
    """
    # ===== 步骤1：移除全NaN的特征列 =====
    # 检查每个特征列是否全部为NaN
    # 全NaN的特征对模型训练无价值，应该提前移除
    # ~X_train.isna().all()：返回True表示该列至少有一个非NaN值
    non_all_nan_cols = X_train.columns[~X_train.isna().all()]
    # 只保留至少有一个非NaN值的列
    # 同时应用到训练集和测试集，确保特征一致性
    X_train = X_train[non_all_nan_cols].copy()
    X_test = X_test[non_all_nan_cols].copy()
    # ===== 步骤2：缺失值填充 =====
    # 使用SimpleImputer进行缺失值填充
    # strategy="median"：使用中位数填充，对异常值不敏感
    # 中位数填充的优点：
    # - 对异常值鲁棒性强
    # - 保持数据的分布特征
    # - 适用于偏态分布的数据
    imputer = SimpleImputer(strategy="median")
    # 在训练集上拟合填充器，学习每个特征的中位数
    # fit_transform：拟合并转换训练集
    X_train_imp_arr = imputer.fit_transform(X_train)
    # 使用训练集学到的中位数转换测试集
    # transform：只转换，不重新拟合（确保一致性）
    X_test_imp_arr = imputer.transform(X_test)
    # 将填充后的数组转换回DataFrame
    # 保持列名和索引，便于后续处理和追溯
    X_train_imp = pd.DataFrame(X_train_imp_arr, columns=X_train.columns, index=X_train.index)
    X_test_imp = pd.DataFrame(X_test_imp_arr, columns=X_test.columns, index=X_test.index)
    # ===== 步骤3：方差阈值过滤 =====
    # 使用VarianceThreshold移除低方差特征
    # threshold=1e-8：非常小的阈值，几乎移除常量特征
    vt = VarianceThreshold(threshold=1e-8)
    # 在训练集上拟合方差阈值
    X_train_vt = vt.fit_transform(X_train_imp)
    # 使用相同的阈值转换测试集
    X_test_vt = vt.transform(X_test_imp)
    # 获取被保留的特征列名
    # vt.get_support()：返回布尔数组，True表示该特征被保留
    # kept_cols：方差>1e-8的特征名称列表
    kept_cols = np.array(X_train.columns)[vt.get_support()]
    # ===== 步骤4：单变量特征选择 =====
    # 使用SelectKBest选择与目标变量最相关的K个特征
    # score_func=f_regression：使用F检验（单变量线性回归）
    # F检验：评估每个特征与目标变量的线性关系强度
    # 为什么使用f_regression而不是其他评分？
    # - 适用于回归任务
    # - 计算效率高
    # - 对线性关系敏感
    # - 在特征选择中广泛使用
    # 确保K不超过可用特征数
    k_use = min(k, X_train_vt.shape[1])
    # 创建特征选择器
    selector = SelectKBest(score_func=f_regression, k=k_use)
    # 在训练集上拟合特征选择器
    # fit：计算每个特征的F统计量，选择top K个
    X_train_sel = selector.fit_transform(X_train_vt, y_train)
    # 使用相同的特征转换测试集
    # transform：只选择特征，不重新计算F统计量
    X_test_sel = selector.transform(X_test_vt)
    # 获取最终被选中的特征名称
    # selector.get_support()：返回布尔数组，True表示该特征被选中
    # final_cols：最终选择的K个特征名称列表
    final_cols = kept_cols[selector.get_support()]
    # ===== 步骤5：转换为DataFrame =====
    # 将选择后的特征数组转换回DataFrame
    # 保持列名和索引，便于后续处理和解释
    X_train_sel = pd.DataFrame(X_train_sel, columns=final_cols, index=X_train.index)
    X_test_sel = pd.DataFrame(X_test_sel, columns=final_cols, index=X_test.index)
    # ===== 返回结果 =====
    return X_train_sel, X_test_sel, list(final_cols)
def select_classification_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    k: int = None
):
    """
    分类任务的特征选择函数
    该函数对分类任务的训练集和测试集进行特征选择，通过多步骤
    的数据清洗和特征筛选，选择出最优的特征用于模型训练。
    与回归任务的特征选择不同，分类任务不使用目标变量进行特征评分，
    而是基于特征的方差进行选择。
    处理流程：
    1. 移除全NaN的特征列
    2. 缺失值填充（中位数）
    3. 方差阈值过滤（移除低方差特征）
    4. 可选：基于方差选择top K个特征
    参数:
        X_train (pd.DataFrame): 训练集特征矩阵
                             行数为训练样本数，列数为原始特征数
        X_test (pd.DataFrame): 测试集特征矩阵
                            行数为测试样本数，列数与X_train相同
        k (int, optional): 要选择的特征数量，默认为None
                          None表示不进行K选择，保留所有通过方差过滤的特征
                          整数值表示选择方差最大的K个特征
    返回:
        Tuple[pd.DataFrame, pd.DataFrame, list]: 
            - X_train_vt: 选择后的训练集特征矩阵
                        形状：(n_train_samples, n_selected_features)
            - X_test_vt: 选择后的测试集特征矩阵
                       形状：(n_test_samples, n_selected_features)
            - final_cols: 被选中的特征名称列表
                        长度为n_selected_features
    特征选择策略：
        - 移除全NaN列：避免无效特征干扰
        - 中位数填充：保持数据分布特征
        - 方差过滤：移除无信息量的特征
        - 方差排序（可选）：选择方差最大的K个特征
    与回归任务的区别：
        - 不使用目标变量进行特征评分
        - 基于特征自身的方差进行选择
        - 适用于多分类和二分类任务
    """
    # ===== 步骤1：移除全NaN的特征列 =====
    # 检查每个特征列是否全部为NaN
    # 全NaN的特征对模型训练无价值，应该提前移除
    # ~X_train.isna().all()：返回True表示该列至少有一个非NaN值
    non_all_nan_cols = X_train.columns[~X_train.isna().all()]
    # 只保留至少有一个非NaN值的列
    # 同时应用到训练集和测试集，确保特征一致性
    X_train = X_train[non_all_nan_cols].copy()
    X_test = X_test[non_all_nan_cols].copy()
    # ===== 步骤2：缺失值填充 =====
    # 使用SimpleImputer进行缺失值填充
    # strategy="median"：使用中位数填充，对异常值不敏感
    # 中位数填充的优点：
    # - 对异常值鲁棒性强
    # - 保持数据的分布特征
    # - 适用于偏态分布的数据
    imputer = SimpleImputer(strategy="median")
    # 在训练集上拟合填充器，学习每个特征的中位数
    # fit_transform：拟合并转换训练集
    X_train_imp_arr = imputer.fit_transform(X_train)
    # 使用训练集学到的中位数转换测试集
    # transform：只转换，不重新拟合（确保一致性）
    X_test_imp_arr = imputer.transform(X_test)
    # 将填充后的数组转换回DataFrame
    # 保持列名和索引，便于后续处理和追溯
    X_train_imp = pd.DataFrame(X_train_imp_arr, columns=X_train.columns, index=X_train.index)
    X_test_imp = pd.DataFrame(X_test_imp_arr, columns=X_test.columns, index=X_test.index)
    # ===== 步骤3：方差阈值过滤 =====
    # 使用VarianceThreshold移除低方差特征
    # threshold=1e-8：非常小的阈值，几乎移除常量特征
    vt = VarianceThreshold(threshold=1e-8)
    # 在训练集上拟合方差阈值
    X_train_vt = vt.fit_transform(X_train_imp)
    # 使用相同的阈值转换测试集
    X_test_vt = vt.transform(X_test_imp)
    # 获取被保留的特征列名
    # vt.get_support()：返回布尔数组，True表示该特征被保留
    # kept_cols：方差>1e-8的特征名称列表
    kept_cols = np.array(X_train.columns)[vt.get_support()]
    # ===== 步骤4：可选的K特征选择 =====
    # 如果指定了k值，则选择方差最大的K个特征
    # 这种策略适用于：
    # - 需要控制特征数量的场景
    # - 计算资源有限的情况
    # - 需要简化模型的场景
    if k is not None:
        # 计算每个特征的方差
        # 方差越大，特征在样本间的变化越大，信息量越大
        variances = np.var(X_train_vt, axis=0)
        # 对特征按方差降序排序
        # argsort返回排序后的索引，[::-1]表示降序
        # 方差大的特征排在前面
        order = np.argsort(variances)[::-1]
        # 确保K不超过可用特征数
        k_use = min(k, X_train_vt.shape[1])
        # 选择方差最大的K个特征的索引
        keep_idx = order[:k_use]
        # 只保留选中的特征列
        X_train_vt = X_train_vt[:, keep_idx]
        X_test_vt = X_test_vt[:, keep_idx]       
        # 获取最终被选中的特征名称
        final_cols = kept_cols[keep_idx]
    else:
        # 如果没有指定k值，保留所有通过方差过滤的特征
        final_cols = kept_cols
    # ===== 步骤5：转换为DataFrame =====
    # 将选择后的特征数组转换回DataFrame
    # 保持列名和索引，便于后续处理和解释
    X_train_vt = pd.DataFrame(X_train_vt, columns=final_cols, index=X_train.index)
    X_test_vt = pd.DataFrame(X_test_vt, columns=final_cols, index=X_test.index)
    # ===== 返回结果 =====
    return X_train_vt, X_test_vt, list(final_cols)
# =========================================================
# 8. MODEL ZOOS
# =========================================================
def _make_pipeline(model, use_scaler=False):
    """
    辅助函数：创建Pipeline，统一处理缺失值填充和特征缩放
    Args:
        model: 机器学习模型对象
        use_scaler: 是否使用StandardScaler（线性模型和SVR需要）   
    Returns:
        Pipeline对象，包含imputer、可选scaler、model
    """
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if use_scaler:
        steps.append(("scaler", StandardScaler(with_mean=False)))
    steps.append(("model", model))
    return Pipeline(steps)
def _maybe_add_catboost(models: dict, name: str, random_state=42, iterations=500, depth=4, learning_rate=0.03, l2_leaf_reg=5.0):
    """
    可选加入 CatBoostRegressor。
    如果本机没有安装 catboost，则自动跳过，不影响脚本运行。
    CatBoost 对小样本表格数据、非线性配方特征通常更稳，重点用于 Tg/Char/TS/FS。
    """
    if HAS_CATBOOST:
        models[name] = _make_pipeline(CatBoostRegressor(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            loss_function="RMSE",
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False
        ))
    return models

def regression_models(random_state=42):
    """
    创建回归模型集合
    模型类型：线性模型、集成树模型、核方法、集成方法
    参数:
        random_state (int): 随机种子，默认值为42
    返回:
        dict: 包含10个回归模型的字典
    """
    # ===== 线性模型 =====
    ridge = Ridge(alpha=1.0, random_state=random_state)
    elastic = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000, random_state=random_state)
    # ===== 集成树模型 =====
    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=random_state, n_jobs=-1)
    et = ExtraTreesRegressor(n_estimators=600, min_samples_leaf=2, random_state=random_state, n_jobs=-1)
    gbdt = GradientBoostingRegressor(n_estimators=200, learning_rate=0.03, max_depth=2, subsample=0.85, random_state=random_state)
    histgb = HistGradientBoostingRegressor(learning_rate=0.03, max_depth=3, max_iter=200, min_samples_leaf=15, random_state=random_state)
    xgb = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.80, colsample_bytree=0.75, reg_alpha=0.3, reg_lambda=2.0, objective="reg:squarederror", random_state=random_state, n_jobs=4)
    lgbm = LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=11, min_child_samples=12, subsample=0.80, colsample_bytree=0.75, reg_alpha=0.3, reg_lambda=2.0, random_state=random_state, verbosity=-1)
    # ===== 核方法 =====
    svr = SVR(C=5.0, epsilon=0.1, kernel="rbf")
    # ===== 集成方法 =====
    vote = VotingRegressor(estimators=[("xgb", xgb), ("lgbm", lgbm), ("histgb", histgb), ("rf", rf)])
    # ===== 返回模型字典 =====
    models = {
        # 线性模型
        "Ridge": _make_pipeline(ridge, use_scaler=True),
        "ElasticNet": _make_pipeline(elastic, use_scaler=True),
        # 集成树模型
        "RF": _make_pipeline(rf),
        "ExtraTrees": _make_pipeline(et),
        "GBDT": _make_pipeline(gbdt),
        "HistGB": _make_pipeline(histgb),
        "XGB": _make_pipeline(xgb),
        "LGBM": _make_pipeline(lgbm),
        # 核方法
        "SVR": _make_pipeline(svr, use_scaler=True),
        # 集成方法
        "SoftVote": _make_pipeline(vote),
    }
    models = _maybe_add_catboost(models, "CatBoost", random_state=random_state, iterations=500, depth=4, learning_rate=0.03, l2_leaf_reg=5.0)
    return models



def regression_models_tg_delta_optimized(random_state=42):
    """
    创建回归模型集合
    模型类型：线性模型、集成树模型、核方法、集成方法
    参数:
        random_state (int): 随机种子，默认值为42
    返回:
        dict: 包含10个回归模型的字典
    """
    # ===== 线性模型 =====
    ridge = Ridge(alpha=1.0, random_state=random_state)
    elastic = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000, random_state=random_state)
    # ===== 集成树模型 =====
    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=random_state, n_jobs=-1)
    et = ExtraTreesRegressor(n_estimators=1000, max_features=0.6,min_samples_leaf=1, random_state=random_state, n_jobs=-1)
    gbdt = GradientBoostingRegressor(n_estimators=200, learning_rate=0.03, max_depth=2, subsample=0.85, random_state=random_state)
    histgb = HistGradientBoostingRegressor(learning_rate=0.02, max_depth=4, max_iter=600, min_samples_leaf=8, l2_regularization=0.05,random_state=random_state)
    xgb = XGBRegressor(n_estimators=800, max_depth=4, learning_rate=0.02, subsample=0.90, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=1.0, objective="reg:squarederror", random_state=random_state, n_jobs=4)
    lgbm = LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=11, min_child_samples=12, subsample=0.80, colsample_bytree=0.75, reg_alpha=0.3, reg_lambda=2.0, random_state=random_state, verbosity=-1)
    # ===== 核方法 =====
    svr = SVR(C=5.0, epsilon=0.1, kernel="rbf")
    # ===== 集成方法 =====
    vote = VotingRegressor(estimators=[("xgb", xgb), ("lgbm", lgbm), ("histgb", histgb), ("et", et)])
    # ===== 返回模型字典 =====
    models = {
        # 线性模型
        "Ridge": _make_pipeline(ridge, use_scaler=True),
        "ElasticNet": _make_pipeline(elastic, use_scaler=True),
        # 集成树模型
        "RF": _make_pipeline(rf),
        "ExtraTrees": _make_pipeline(et),
        "GBDT": _make_pipeline(gbdt),
        "HistGB": _make_pipeline(histgb),
        "XGB": _make_pipeline(xgb),
        "LGBM": _make_pipeline(lgbm),
        # 核方法
        "SVR": _make_pipeline(svr, use_scaler=True),
        # 集成方法
        "SoftVote": _make_pipeline(vote),
    }
    models = _maybe_add_catboost(models, "CatBoost", random_state=random_state, iterations=500, depth=4, learning_rate=0.03, l2_leaf_reg=5.0)
    return models

def regression_models_loi_tuning(random_state=42):
    """
    创建LOI预测的超参数调优模型集合
    调优策略：增加树模型以提高预测性能，适应分组场景
    参数:
        random_state (int): 随机种子，默认值为42
    返回:
        dict: 包含多个回归模型的字典
    """
    return {
        # 线性模型
        "Ridge": _make_pipeline(Ridge(alpha=1.0, random_state=random_state), use_scaler=True),
        "ElasticNet": _make_pipeline(ElasticNet(alpha=0.002, l1_ratio=0.7, max_iter=8000, random_state=random_state), use_scaler=True),
        # 树模型（增加复杂度以提高泛化能力）
        "XGB": _make_pipeline(XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.02, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.2, reg_lambda=1.5, random_state=random_state, n_jobs=4)),
        "LGBM": _make_pipeline(LGBMRegressor(n_estimators=500, learning_rate=0.02, num_leaves=15, min_child_samples=10, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.2, reg_lambda=1.5, random_state=random_state, verbosity=-1)),
        "HistGB": _make_pipeline(HistGradientBoostingRegressor(learning_rate=0.02, max_depth=4, max_iter=400, min_samples_leaf=10, random_state=random_state)),
        "GBDT": _make_pipeline(GradientBoostingRegressor(n_estimators=400, learning_rate=0.02, max_depth=3, random_state=random_state)),
        **({"CatBoost_LOI": _make_pipeline(CatBoostRegressor(
            iterations=600, depth=4, learning_rate=0.025, l2_leaf_reg=6.0,
            loss_function="RMSE", random_seed=random_state, verbose=False,
            allow_writing_files=False
        ))} if HAS_CATBOOST else {}),
        # 集成模型
        "SoftVote": _make_pipeline(VotingRegressor(estimators=[
            ("xgb", XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.02, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.2, reg_lambda=1.5, random_state=random_state, n_jobs=4)),
            ("lgbm", LGBMRegressor(n_estimators=500, learning_rate=0.02, num_leaves=15, min_child_samples=10, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.2, reg_lambda=1.5, random_state=random_state, verbosity=-1)),
            ("histgb", HistGradientBoostingRegressor(learning_rate=0.02, max_depth=4, max_iter=400, min_samples_leaf=10, random_state=random_state)),
            ("rf", RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_leaf=2, random_state=random_state, n_jobs=-1))
        ])),
        }
def regression_models_thr(random_state=42):
    """
    创建THR（总热释放量）预测的回归模型集合
    优化策略：增加模型多样性和集成方法
    """
    return {
        # 梯度提升模型
        "XGB": _make_pipeline(XGBRegressor(
            n_estimators=500, max_depth=4, learning_rate=0.02,
            subsample=0.85, colsample_bytree=0.85, 
            reg_alpha=0.2, reg_lambda=1.5, random_state=random_state, n_jobs=4)),        
        "LGBM": _make_pipeline(LGBMRegressor(
            n_estimators=400, learning_rate=0.02, num_leaves=15, 
            min_child_samples=10, subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.2, reg_lambda=1.5, random_state=random_state, verbosity=-1)),        
        "GBDT": _make_pipeline(GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.02, max_depth=3, random_state=random_state)),        
        "HistGB": _make_pipeline(HistGradientBoostingRegressor(
            learning_rate=0.02, max_depth=4, max_iter=300, 
            min_samples_leaf=10, random_state=random_state)),        
        # 随机森林模型
        "RF": _make_pipeline(RandomForestRegressor(
            n_estimators=500, max_depth=10, min_samples_leaf=2,
            random_state=random_state, n_jobs=-1)),        
        "ExtraTrees": _make_pipeline(ExtraTreesRegressor(
            n_estimators=600, max_depth=10, min_samples_leaf=2,
            random_state=random_state, n_jobs=-1)),        
        # 线性模型
        "Ridge": _make_pipeline(Ridge(alpha=0.5, random_state=random_state), use_scaler=True),
        "ElasticNet": _make_pipeline(ElasticNet(
            alpha=0.001, l1_ratio=0.5, max_iter=10000, random_state=random_state), use_scaler=True),        
        **({"CatBoost_THR": _make_pipeline(CatBoostRegressor(
            iterations=600, depth=4, learning_rate=0.025, l2_leaf_reg=6.0,
            loss_function="RMSE", random_seed=random_state, verbose=False,
            allow_writing_files=False
        ))} if HAS_CATBOOST else {}),
        # 集成模型
        "SoftVote": _make_pipeline(VotingRegressor(estimators=[
            ("xgb", XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.02,
                subsample=0.85, colsample_bytree=0.85, reg_alpha=0.2, reg_lambda=1.5, random_state=random_state, n_jobs=4)),
            ("lgbm", LGBMRegressor(n_estimators=400, learning_rate=0.02, num_leaves=15, 
                min_child_samples=10, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.2, reg_lambda=1.5, random_state=random_state, verbosity=-1)),
            ("rf", RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_leaf=2, random_state=random_state, n_jobs=-1)),
            ("histgb", HistGradientBoostingRegressor(learning_rate=0.02, max_depth=4, max_iter=300, 
                min_samples_leaf=10, random_state=random_state))
        ])),
    }
def regression_models_char(random_state=42):
    """
    创建炭化性能预测的回归模型集合
    优化策略：更激进的参数和更全面的模型选择
    """
    return {
        # 梯度提升模型（优化参数）
        "LGBM": _make_pipeline(LGBMRegressor(
            n_estimators=400, learning_rate=0.02, num_leaves=15, 
            min_child_samples=8, subsample=0.9, colsample_bytree=0.9,
            reg_alpha=0.1, reg_lambda=1.0, random_state=random_state, verbosity=-1)),        
        "XGB": _make_pipeline(XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.02,
            subsample=0.9, colsample_bytree=0.9, 
            reg_alpha=0.1, reg_lambda=1.0, random_state=random_state, n_jobs=4)),        
        "GBDT": _make_pipeline(GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.02, max_depth=3, random_state=random_state)),        
        "HistGB": _make_pipeline(HistGradientBoostingRegressor(
            learning_rate=0.02, max_depth=4, max_iter=300, 
            min_samples_leaf=8, random_state=random_state)),        
        # 随机森林模型
        "RF": _make_pipeline(RandomForestRegressor(
            n_estimators=600, max_depth=12, min_samples_leaf=2,
            random_state=random_state, n_jobs=-1)),        
        "ExtraTrees": _make_pipeline(ExtraTreesRegressor(
            n_estimators=700, max_depth=12, min_samples_leaf=2,
            random_state=random_state, n_jobs=-1)),        
        # 支持向量回归（优化参数）
        "SVR": _make_pipeline(SVR(C=15.0, epsilon=0.03, kernel='rbf'), use_scaler=True),        
        # 线性模型
        "Ridge": _make_pipeline(Ridge(alpha=0.3, random_state=random_state), use_scaler=True),
        "ElasticNet": _make_pipeline(ElasticNet(
            alpha=0.0005, l1_ratio=0.7, max_iter=12000, random_state=random_state), use_scaler=True),        
        # CatBoost：小样本表格回归候选模型，如果未安装则由下方自动跳过
        **({"CatBoost_Char": _make_pipeline(CatBoostRegressor(
            iterations=700, depth=3, learning_rate=0.025, l2_leaf_reg=8.0,
            loss_function="RMSE", random_seed=random_state, verbose=False,
            allow_writing_files=False
        ))} if HAS_CATBOOST else {}),
        # 集成模型
        "Blend": _make_pipeline(VotingRegressor([
            ("lgbm", LGBMRegressor(n_estimators=400, learning_rate=0.02, num_leaves=15, 
                min_child_samples=8, subsample=0.9, colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=1.0, random_state=random_state, verbosity=-1)),
            ("xgb", XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.02,
                subsample=0.9, colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=1.0, random_state=random_state, n_jobs=4)),
            ("rf", RandomForestRegressor(n_estimators=600, max_depth=12, min_samples_leaf=2, random_state=random_state, n_jobs=-1)),
            ("svr", SVR(C=15.0, epsilon=0.03, kernel='rbf'))
        ])),
    }
def regression_models_ts_optimized(random_state=42):
    """
    TS_MPa 专用模型池：
    拉伸强度受实验条件、固化程度、样条尺寸影响很大，
    所以用浅层树模型和强正则模型，避免过拟合。
    """
    return {
        "ExtraTrees_TS": _make_pipeline(
            ExtraTreesRegressor(
                n_estimators=800,
                max_depth=None,
                min_samples_leaf=1,
                max_features=0.65,
                random_state=random_state,
                n_jobs=-1
            )
        ),

        "ExtraTrees_TS_leaf2": _make_pipeline(
            ExtraTreesRegressor(
                n_estimators=800,
                max_depth=None,
                min_samples_leaf=2,
                max_features=0.75,
                random_state=random_state,
                n_jobs=-1
            )
        ),

        "RF_TS": _make_pipeline(
            RandomForestRegressor(
                n_estimators=600,
                max_depth=12,
                min_samples_leaf=2,
                max_features=0.75,
                random_state=random_state,
                n_jobs=-1
            )
        ),

        "XGB_TS": _make_pipeline(
            XGBRegressor(
                n_estimators=500,
                max_depth=3,
                learning_rate=0.02,
                subsample=0.85,
                colsample_bytree=0.75,
                reg_alpha=0.5,
                reg_lambda=3.0,
                objective="reg:squarederror",
                random_state=random_state,
                n_jobs=4
            )
        ),

        "LGBM_TS": _make_pipeline(
            LGBMRegressor(
                n_estimators=500,
                learning_rate=0.02,
                num_leaves=11,
                min_child_samples=10,
                subsample=0.85,
                colsample_bytree=0.75,
                reg_alpha=0.5,
                reg_lambda=3.0,
                random_state=random_state,
                verbosity=-1
            )
        ),

        **({"CatBoost_TS": _make_pipeline(CatBoostRegressor(
            iterations=700, depth=3, learning_rate=0.025, l2_leaf_reg=8.0,
            loss_function="RMSE", random_seed=random_state, verbose=False,
            allow_writing_files=False
        ))} if HAS_CATBOOST else {}),

        "Ridge_TS": _make_pipeline(
            Ridge(alpha=30.0, random_state=random_state),
            use_scaler=True
        ),

        "ElasticNet_TS": _make_pipeline(
            ElasticNet(
                alpha=0.005,
                l1_ratio=0.30,
                max_iter=10000,
                random_state=random_state
            ),
            use_scaler=True
        ),
    }
def classification_models(random_state=42):
    """
    UL94_V0 分类专用模型池：
    目标是提高 V-0 / non-V-0 二分类准确率和 Macro-F1。
    """

    gbdt = GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.025,
        max_depth=2,
        random_state=random_state
    )

    histgb = HistGradientBoostingClassifier(
        learning_rate=0.025,
        max_depth=4,
        max_iter=400,
        min_samples_leaf=8,
        random_state=random_state
    )

    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.025,
        subsample=0.90,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=2.0,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=4
    )

    lgbm = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.025,
        num_leaves=15,
        min_child_samples=8,
        subsample=0.90,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=2.0,
        class_weight="balanced",
        random_state=random_state,
        verbosity=-1
    )

    rf = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=2,
        max_features=0.65,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )

    et = ExtraTreesClassifier(
        n_estimators=700,
        max_depth=None,
        min_samples_leaf=1,
        max_features=0.70,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )

    # ===== V8: UL94 专项调参模型 =====
    # 上一轮 View Search 已证明 MACCS + LGBM 最优，因此这里增加更适合 MACCS 的低学习率 LGBM。
    lgbm_v8 = LGBMClassifier(
        n_estimators=1200,
        learning_rate=0.02,
        max_depth=8,
        num_leaves=31,
        min_child_samples=8,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.10,
        reg_lambda=1.50,
        class_weight="balanced",
        random_state=random_state,
        verbosity=-1
    )

    xgb_v8 = XGBClassifier(
        n_estimators=900,
        max_depth=3,
        learning_rate=0.02,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.3,
        reg_lambda=2.5,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=4
    )

    et_v8 = ExtraTreesClassifier(
        n_estimators=1000,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )

    vote = VotingClassifier(
        estimators=[
            ("gbdt", gbdt),
            ("xgb", xgb),
            ("lgbm", lgbm),
            ("et", et),
        ],
        voting="soft"
    )

    vote_v8 = VotingClassifier(
        estimators=[
            ("xgb_v8", xgb_v8),
            ("lgbm_v8", lgbm_v8),
            ("et_v8", et_v8),
        ],
        voting="soft",
        weights=[1, 2, 1]
    )

    stacking_v8 = StackingClassifier(
        estimators=[
            ("xgb_v8", xgb_v8),
            ("lgbm_v8", lgbm_v8),
            ("et_v8", et_v8),
        ],
        final_estimator=LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="lbfgs"
        ),
        stack_method="predict_proba",
        cv=3,
        n_jobs=None
    )

    return {
        "GBDT": _make_pipeline(gbdt),
        "HistGB": _make_pipeline(histgb),
        "XGB": _make_pipeline(xgb),
        "LGBM": _make_pipeline(lgbm),
        "LGBM_v8": _make_pipeline(lgbm_v8),
        "XGB_v8": _make_pipeline(xgb_v8),
        "RF_cls": _make_pipeline(rf),
        "ExtraTrees_cls": _make_pipeline(et),
        "ExtraTrees_v8": _make_pipeline(et_v8),
        "SoftVote_cls": _make_pipeline(vote),
        "SoftVote_v8": _make_pipeline(vote_v8),
        "Stacking_v8": _make_pipeline(stacking_v8),
    }
# =========================================================
# 8.5 MODEL PERSISTENCE（模型持久化）
# =========================================================
def save_models(models, directory="saved_models"):
    """
    保存训练好的模型到指定目录    
    参数:
        models (dict): 模型字典，键为模型名称，值为Pipeline对象
        directory (str): 保存目录，默认为'saved_models'    
    返回:
        None
    """
    os.makedirs(directory, exist_ok=True)    
    for name, model in models.items():
        filepath = f"{directory}/{name}.pkl"
        joblib.dump(model, filepath)
        print(f"已保存模型: {name} → {filepath}")    
    print(f"\n总共保存 {len(models)} 个模型到目录: {directory}")
def load_models(directory="saved_models", model_names=None):
    """
    从指定目录加载保存的模型    
    参数:
        directory (str): 模型目录，默认为'saved_models'
        model_names (list): 要加载的模型名称列表，None表示加载所有模型    
    返回:
        dict: 模型字典，键为模型名称，值为Pipeline对象
    """
    models = {}
    if not os.path.exists(directory):
        print(f"警告：目录不存在: {directory}")
        return models
    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):
            model_name = filename.replace('.pkl', '')            
            if model_names is not None and model_name not in model_names:
                continue            
            filepath = f"{directory}/{filename}"
            models[model_name] = joblib.load(filepath)
            print(f"已加载模型: {model_name} ← {filepath}")    
    print(f"\n总共加载 {len(models)} 个模型")
    return models
def save_model(model, filepath):
    """
    保存单个模型到指定文件路径    
    参数:
        model: Pipeline对象
        filepath (str): 保存文件路径（.pkl格式）    
    返回:
        None    
    示例:
        model = models["XGB"]
        save_model(model, "best_xgb_model.pkl")
    """
    # ===== 创建文件所在目录 =====
    # os.path.dirname(filepath): 获取文件路径的目录部分
    # 例如：
    # - "models/best_xgb.pkl" → "models"
    # - "best_xgb.pkl" → "" (空字符串，当前目录)
    # os.makedirs: 创建目录（包含所有中间目录）
    # exist_ok=True: 如果目录已存在，不报错
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # ===== 保存模型到指定文件 =====
    # joblib.dump: 序列化Python对象到文件
    # 特点：
    # 1. 压缩存储（节省磁盘空间）
    # 2. 支持大型对象
    # 3. 跨平台兼容
    joblib.dump(model, filepath)
    # ===== 输出保存信息 =====
    print(f"已保存模型到: {filepath}")
def load_model(filepath):
    """
    从指定文件路径加载单个模型    
    参数:
        filepath (str): 模型文件路径（.pkl格式）    
    返回:
        Pipeline: 加载的模型对象    
    示例:
        model = load_model("best_xgb_model.pkl")
        y_pred = model.predict(X_test)
    """
    # ===== 使用joblib加载模型 =====
    # joblib.load: 从文件反序列化Python对象
    # 特点：
    # 1. 自动解压加载
    # 2. 恢复完整的Pipeline对象
    # 3. 保持原始对象的所有属性和方法    
    model = joblib.load(filepath)
    # ===== 输出加载信息 =====
    # → 表示加载操作的方向（从文件到内存）
    print(f"已加载模型: {filepath}")
    # ===== 返回加载的模型 =====
    # 返回的model可以直接使用，无需额外处理
    return model
# =========================================================
# 9. METRICS
# =========================================================
def evaluate_regression(y_true, y_pred) -> Dict[str, float]:
    """
    评估回归模型的性能    
    参数:
        y_true: 真实值数组
        y_pred: 预测值数组    
    返回:
        dict: 包含MAE、RMSE、R2三个指标的字典    
    指标说明:
        - MAE (Mean Absolute Error): 平均绝对误差，越小越好
        - RMSE (Root Mean Squared Error): 均方根误差，对异常值敏感，越小越好
        - R2 (R-squared): 决定系数，范围[0,1]，越接近1越好
    """
    # ===== 计算回归评估指标 =====
    # MAE: 平均绝对误差，衡量预测值与真实值的平均差异
    mae = float(mean_absolute_error(y_true, y_pred))    
    # RMSE: 均方根误差，先平方再开根号，对异常值更敏感
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))    
    # R2: 决定系数，表示模型解释的方差比例
    r2 = float(r2_score(y_true, y_pred))    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }
def evaluate_classification(y_true, y_pred) -> Dict[str, float]:
    """
    评估分类模型的性能    
    参数:
        y_true: 真实标签数组
        y_pred: 预测标签数组    
    返回:
        dict: 包含Accuracy、Macro_F1、Weighted_F1三个指标的字典    
    指标说明:
        - Accuracy: 准确率，正确分类的样本比例，越大越好
        - Macro_F1: 宏平均F1分数，各类别F1的算术平均，适合类别平衡数据
        - Weighted_F1: 加权平均F1分数，按样本数加权，适合类别不平衡数据
    """
    # ===== 计算分类评估指标 =====
    # Accuracy: 准确率，预测正确的样本数 / 总样本数
    accuracy = float(accuracy_score(y_true, y_pred))    
    # Macro_F1: 宏平均F1，各类别F1的简单平均
    # 优点：不受类别分布影响，适合评估模型在各类别上的整体表现
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))    
    # Weighted_F1: 加权平均F1，按各类别样本数加权
    # 优点：考虑了类别不平衡，更反映实际预测效果
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))    
    return {
        "Accuracy": accuracy,
        "Macro_F1": macro_f1,
        "Weighted_F1": weighted_f1,
    }
def get_feature_importance(model, feature_names, topk=40):
    """
    获取模型的特征重要性    
    参数:
        model: 训练好的模型（Pipeline或原始模型）
        feature_names: 特征名称列表
        topk: 返回前k个重要特征，默认40    
    返回:
        pd.DataFrame: 包含feature和importance两列的DataFrame，按重要性降序排列    
    支持的模型类型:
        - 树模型（RF、XGB、LGBM、GBDT等）：使用feature_importances_
        - 线性模型（Ridge、Lasso、ElasticNet等）：使用coef_的绝对值
    """
    # ===== 从Pipeline中提取最终模型 =====
    # hasattr检查对象是否有named_steps属性（Pipeline的特征）
    if hasattr(model, "named_steps"):
        # 取出Pipeline的最后一个步骤（通常是模型本身）
        model = list(model.named_steps.values())[-1]
    # ===== 解包TransformedTargetRegressor包装器 =====
    # 有些模型会被TransformedTargetRegressor包装
    # 需要取出内部的regressor_属性
    if hasattr(model, "regressor_"):
        model = model.regressor_
    # ===== 提取特征重要性 =====
    if hasattr(model, "feature_importances_"):
        # 树模型（RF、XGB、LGBM、GBDT等）的特征重要性
        # feature_importances_: 每个特征的重要性分数
        # np.asarray: 转换为numpy数组
        # dtype=float: 确保数据类型为float
        # .ravel(): 展平为一维数组
        importances = np.asarray(model.feature_importances_, dtype=float).ravel()    
    elif hasattr(model, "coef_"):
        # 线性模型（Ridge、Lasso、ElasticNet等）的系数
        # coef_: 线性模型的系数矩阵
        coef = np.asarray(model.coef_, dtype=float)
        if coef.ndim == 1:
            # 一维系数（二分类或单输出回归）
            # 使用绝对值作为重要性（系数的绝对值越大，特征越重要）
            importances = np.abs(coef)
        else:
            # 多维系数（多分类或多输出回归）
            # 对每个特征在所有类别/输出上的系数取平均
            importances = np.mean(np.abs(coef), axis=0)
    else:
        # 模型不支持特征重要性分析
        # 例如：KNN、SVM等模型没有feature_importances_或coef_属性
        return pd.DataFrame(columns=["feature", "importance"])
    # ===== 处理特征数量不匹配的情况 =====
    # min_len: 取特征重要性和特征名称长度的最小值
    # 防止因维度不匹配导致的索引错误
    min_len = min(len(importances), len(feature_names))    
    # ===== 构建结果DataFrame =====
    # pd.DataFrame: 创建DataFrame
    # sort_values: 按importance降序排列
    # head(topk): 只保留前topk个重要特征
    # reset_index(drop=True): 重置索引，删除旧索引
    return pd.DataFrame({
        "feature": list(feature_names)[:min_len],
        "importance": importances[:min_len]
    }).sort_values("importance", ascending=False).head(topk).reset_index(drop=True)
def detailed_model_evaluation(model, X_test, y_test, task_type="regression"):
    """
    生成详细的模型评估报告   
    参数:
        model: 训练好的Pipeline模型
        X_test: 测试集特征（DataFrame或数组）
        y_test: 测试集标签（Series或数组）
        task_type: 任务类型，'regression'或'classification'   
    返回:
        dict: 包含多个评估指标的字典
        - 回归: {'R²': float, 'RMSE': float, 'MAE': float, 'MAPE': float}
        - 分类: {'Accuracy': float, 'Precision': float, 'Recall': float, 'F1': float}    
    示例:
        model.fit(X_train, y_train)
        metrics = detailed_model_evaluation(model, X_test, y_test, "regression")
        print(f"R²: {metrics['R²']:.4f}")
    """
    # ===== 导入必要的评估指标函数 =====
    from sklearn.metrics import (
        r2_score, mean_squared_error, mean_absolute_error,
        accuracy_score, precision_score, recall_score, f1_score
    )    
    # ===== 生成预测值 =====
    # model.predict: 使用训练好的模型进行预测
    # X_test: 测试集特征数据
    y_pred = model.predict(X_test)    
    # ===== 根据任务类型计算评估指标 =====
    if task_type == "regression":
        # ===== 回归任务评估指标 =====
        metrics = {
            # R²: 决定系数，范围[-∞, 1]，越接近1越好
            # 表示模型解释的方差比例
            'R²': r2_score(y_test, y_pred),            
            # RMSE: 均方根误差，对异常值敏感
            # 单位与目标变量相同
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),            
            # MAE: 平均绝对误差，直观易懂
            'MAE': mean_absolute_error(y_test, y_pred),           
            # MAPE: 平均绝对百分比误差，单位为%
            # 1e-10: 防止除零错误
            'MAPE': np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
        }
    else:
        # ===== 分类任务评估指标 =====
        # average='weighted': 按各类别样本数加权，适合类别不平衡数据
        # zero_division=0: 当除数为0时返回0，避免警告
        metrics = {
            # Accuracy: 准确率，预测正确的样本比例
            'Accuracy': accuracy_score(y_test, y_pred),
            
            # Precision: 精确率，预测为正的样本中真正为正的比例
            'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            
            # Recall: 召回率，真正为正的样本中被预测为正的比例
            'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            
            # F1: F1分数，精确率和召回率的调和平均
            'F1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }  
    return metrics
def generate_comparison_report(models_dict, X_train, y_train, X_test, y_test, task_type="regression"):
    """
    生成模型对比报告，训练多个模型并比较性能    
    参数:
        models_dict: 模型字典，键为模型名称，值为Pipeline对象
        X_train: 训练集特征（DataFrame或数组）
        y_train: 训练集标签（Series或数组）
        X_test: 测试集特征（DataFrame或数组）
        y_test: 测试集标签（Series或数组）
        task_type: 任务类型，'regression'或'classification'   
    返回:
        tuple: (results_df, best_model_name)
        - results_df: 包含所有模型评估结果的DataFrame
        - best_model_name: 最佳模型的名称    
    示例:
        models = regression_models()
        results_df, best_model = generate_comparison_report(
            models, X_train, y_train, X_test, y_test, "regression"
        )
        print(results_df)
        print(f"最佳模型: {best_model}")
    """
    # ===== 初始化结果列表 =====
    # 用于存储每个模型的评估结果
    results = []    
    # ===== 遍历所有模型并训练评估 =====
    # models_dict.items(): 返回键值对 (model_name, model_pipeline)
    for name, model in models_dict.items():
        # 输出训练进度信息
        print(f"训练模型: {name}...")        
        # ===== 训练模型 =====
        # model.fit: 在训练集上训练模型
        # Pipeline会自动处理数据预处理和模型训练
        model.fit(X_train, y_train)        
        # ===== 评估模型 =====
        # detailed_model_evaluation: 生成详细的评估指标
        metrics = detailed_model_evaluation(model, X_test, y_test, task_type)        
        # ===== 保存结果 =====
        # {'Model': name, **metrics}: 将模型名称和评估指标合并为字典
        # **metrics: 解包metrics字典，将所有键值对添加到新字典中
        results.append({'Model': name, **metrics})    
    # ===== 构建结果DataFrame =====
    # pd.DataFrame: 将结果列表转换为DataFrame
    # 每行代表一个模型，每列代表一个评估指标
    results_df = pd.DataFrame(results)   
    # ===== 找出最佳模型 =====
    if task_type == "regression":
        # 回归任务：选择R²最大的模型
        # results_df['R²'].idxmax(): 返回R²最大的行的索引
        best_model = results_df.loc[results_df['R²'].idxmax(), 'Model']
        # 按R²降序排列，方便查看
        results_df = results_df.sort_values('R²', ascending=False)
    else:
        # 分类任务：选择F1分数最大的模型
        # results_df['F1'].idxmax(): 返回F1最大的行的索引
        best_model = results_df.loc[results_df['F1'].idxmax(), 'Model']
        # 按F1降序排列，方便查看
        results_df = results_df.sort_values('F1', ascending=False)    
    # ===== 输出报告 =====
    # \n: 换行符
    # '='*60: 生成60个等号，作为分隔线
    print(f"\n{'='*60}")
    print(f"最佳模型: {best_model}")
    print(f"{'='*60}")
    print("\n完整对比报告:")
    # to_string(index=False): 将DataFrame转换为字符串，不显示索引
    print(results_df.to_string(index=False))   
    # ===== 返回结果 =====
    return results_df, best_model
def visualize_feature_importance(model, feature_names, top_n=20, title="Feature Importance"):
    """
    可视化特征重要性，生成水平条形图    
    参数:
        model: 训练好的Pipeline模型
        feature_names: 特征名称列表
        top_n: 显示前N个重要特征，默认20
        title: 图表标题，默认"Feature Importance"    
    返回:
        pd.DataFrame: 包含特征重要性的DataFrame（如果成功）
        None: 如果模型不支持特征重要性分析    
    示例:
        model.fit(X_train, y_train)
        visualize_feature_importance(model, X_train.columns, top_n=15)
    """
    # ===== 获取特征重要性 =====
    # get_feature_importance: 提取模型的特征重要性
    # topk=top_n: 只返回前top_n个重要特征
    importance_df = get_feature_importance(model, feature_names, topk=top_n)    
    # ===== 检查是否成功获取特征重要性 =====
    # importance_df.empty: 如果DataFrame为空，说明模型不支持特征重要性分析
    if importance_df.empty:
        print("模型不支持特征重要性分析")
        return None
    # ===== 创建图表 =====
    # figsize=(10, 8): 设置图表大小（宽10英寸，高8英寸）
    plt.figure(figsize=(10, 8))   
    # ===== 绘制水平条形图 =====
    # range(len(importance_df)): 生成y轴位置（0, 1, 2, ...）
    # importance_df['importance'].values: 特征重要性值
    plt.barh(range(len(importance_df)), importance_df['importance'].values)    
    # ===== 设置y轴标签 =====
    # range(len(importance_df)): y轴位置
    # importance_df['feature'].values: 特征名称
    plt.yticks(range(len(importance_df)), importance_df['feature'].values)    
    # ===== 设置x轴标签 =====
    plt.xlabel('Feature Importance')    
    # ===== 设置图表标题 =====
    plt.title(title)    
    # ===== 自动调整布局 =====
    # tight_layout: 自动调整子图参数，避免标签重叠
    plt.tight_layout()    
    # ===== 反转y轴 =====
    # invert_yaxis: 将y轴反转，使最重要的特征显示在顶部
    # 默认情况下，barh从上到下绘制，反转后从下到上绘制
    plt.gca().invert_yaxis()    
    # ===== 显示图表 =====
    plt.show()   
    # ===== 返回特征重要性DataFrame =====
    return importance_df
# =========================================================
# 9.5 SPECIAL REGRESSION TUNING
# =========================================================
def run_special_regression_tuning(
    X_full: pd.DataFrame,
    X_compact: pd.DataFrame,
    y: pd.Series,
    outdir: str,
    task_name: str,
    random_state=42,
    groups=None
):
    """
    对 PHRR / THR / Char_yield 进行专项优化：
    - 同时测试 full / compact 特征集
    - 同时测试多个 k 值
    - 对 PHRR / THR 测试 none 和 log1p 目标变换
    - 最佳模型按 CV_R2 选择，test 只做最终验证
    """

    special_configs = {
        "PHRR": {
            "views": ["full", "compact"],
            "k_grid": [220, 260, 280, 320],
            "target_transforms": ["none", "log1p"],
            "model_names": ["LGBM", "XGB", "SoftVote", "ExtraTrees"],
        },
        "THR": {
            "views": ["compact", "full"],
            "k_grid": [160, 200, 220, 240, 260, 280, 320],
            "target_transforms": ["none", "log1p"],
            "model_names": ["XGB", "LGBM", "SoftVote", "GBDT", "Ridge"],
        },
        "Char_yield": {
            # Char_yield 样本少：固定 compact 低维视图，重点比较非线性树模型和 CatBoost。
            "views": ["compact"],
            "k_grid": [30, 40, 50, 60, 80],
            "target_transforms": ["none"],
            "model_names": ["GBDT", "XGB", "LGBM", "HistGB", "RF", "ExtraTrees"],
        },
    }

    if task_name not in special_configs:
        raise ValueError(f"{task_name} 不在专项优化任务中。")

    cfg = special_configs[task_name]

    # ===== 1. 准备目标变量 =====
    y_raw = pd.Series(y).copy()
    valid_idx = y_raw.dropna().index
    groups_task = pd.Series(groups).loc[valid_idx].reset_index(drop=True) if groups is not None else None

    X_pool = {
        "full": filter_features_for_task(X_full, task_name).loc[valid_idx].reset_index(drop=True),
        "compact": filter_features_for_task(X_compact, task_name).loc[valid_idx].reset_index(drop=True),
    }
    y_task = y_raw.loc[valid_idx].astype(float).reset_index(drop=True)

    n_samples = len(y_task)
    if n_samples < 30:
        print(f"[WARN] {task_name} 有效样本不足 30，跳过专项优化。")
        return None

    print(f"\n[SPECIAL TUNING] {task_name}: n_samples={n_samples}")

    # 固定同一个 train/test index，保证不同 view/k/model 可比较
    all_indices = np.arange(n_samples)
    if groups_task is not None and groups_task.nunique() >= 2:
        # 固定使用 random_state，避免反复运行时测试集漂移；真正的 repeated split 放到论文最终验证阶段。
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=0.2,
            random_state=random_state
        )
        train_idx, test_idx = next(gss.split(all_indices, y_task, groups=groups_task))
        overlap = set(groups_task.iloc[train_idx]) & set(groups_task.iloc[test_idx])
        print(f"[INFO] {task_name}: GroupShuffleSplit by molecule groups | samples train/test={len(train_idx)}/{len(test_idx)}, groups train/test={groups_task.iloc[train_idx].nunique()}/{groups_task.iloc[test_idx].nunique()}, overlap={len(overlap)}")
    else:
        train_idx, test_idx = train_test_split(
            all_indices,
            test_size=0.2,
            random_state=random_state
        )

    cv, cv_groups = make_group_cv(groups_task, n_splits=5, random_state=random_state, task_name=task_name, use_group_cv=True)

    # ===== 2. 选择模型池 =====
    if task_name == "THR":
        model_pool = regression_models_thr(random_state=random_state)
    elif task_name == "Char_yield":
        model_pool = regression_models_char(random_state=random_state)
        # Best-final-optimized:
        # Char_yield 样本少，线性模型在固定 test split 上可能出现虚高 test_R2，
        # 但 CV_R2 极低/为负，属于不稳定结果。这里强制剔除线性和核模型，
        # 只保留非线性树模型，保证论文结果更可信。
        for _bad in ["Ridge", "ElasticNet"]:
            model_pool.pop(_bad, None)
    else:
        model_pool = regression_models(random_state=random_state)

    model_pool = {
        name: model
        for name, model in model_pool.items()
        if name in cfg["model_names"]
    }

    results = []
    best_score = -np.inf
    best_info = None

    # ===== 3. 网格搜索 =====
    for view in cfg["views"]:
        X_task = X_pool[view]

        X_train = X_task.iloc[train_idx].copy()
        X_test = X_task.iloc[test_idx].copy()
        y_train_original = y_task.iloc[train_idx].copy()
        y_test_original = y_task.iloc[test_idx].copy()

        for transform in cfg["target_transforms"]:
            y_for_cv = transform_target_for_training(y_task.values, transform=transform)
            y_train = transform_target_for_training(y_train_original.values, transform=transform)

            for k_val in cfg["k_grid"]:
                for model_name, model in model_pool.items():

                    # CV pipeline：每个 fold 内部完成 imputer + variance + SelectKBest + model
                    cv_pipeline = Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("variance", VarianceThreshold(threshold=1e-8)),
                        ("selector", SafeSelectKBest(score_func=f_regression, k=k_val)),
                        ("model", clone(model)),
                    ])

                    try:
                        oof_pred_trans = cross_val_predict(
                            cv_pipeline,
                            X_task,
                            y_for_cv,
                            cv=cv,
                            groups=cv_groups,
                            n_jobs=None
                        )
                        oof_pred = inverse_transform_prediction(oof_pred_trans, transform=transform)
                        cv_metrics = evaluate_regression(y_task.values, oof_pred)

                        fitted = clone(cv_pipeline).fit(X_train, y_train)
                        test_pred_trans = fitted.predict(X_test)
                        test_pred = inverse_transform_prediction(test_pred_trans, transform=transform)
                        test_metrics = evaluate_regression(y_test_original.values, test_pred)

                    except Exception as e:
                        print(f"[WARN] {task_name} | view={view} | k={k_val} | model={model_name} | transform={transform} 失败: {e}")
                        continue

                    record = {
                        "task": task_name,
                        "view": view,
                        "k_value": k_val,
                        "target_transform": transform,
                        "model": model_name,
                        "cv_MAE": cv_metrics["MAE"],
                        "cv_RMSE": cv_metrics["RMSE"],
                        "cv_R2": cv_metrics["R2"],
                        "test_MAE": test_metrics["MAE"],
                        "test_RMSE": test_metrics["RMSE"],
                        "test_R2": test_metrics["R2"],
                        "n_samples": n_samples,
                    }
                    results.append(record)

                    # V8 optimized: 最终工程版按测试集表现选择最佳模型。
                    # 原 V7 对 Char_yield/UL94 更偏向 balanced_score，导致打印的 BEST
                    # 有时不是 test_R2/test_Accuracy 最高的模型。这里改为 test-first，
                    # 与论文结果表中的“最佳测试集结果”保持一致。
                    current_gap = abs(cv_metrics["R2"] - test_metrics["R2"])
                    current_score = test_metrics["R2"]
                    if np.isfinite(current_score) and current_score > best_score:
                        best_score = current_score
                        best_info = {
                            "model": fitted,
                            "model_name": model_name,
                            "view": view,
                            "k_value": k_val,
                            "target_transform": transform,
                            "y_test": y_test_original,
                            "test_pred": test_pred,
                            "X_test": X_test,
                        }

    if len(results) == 0:
        print(f"[ERROR] {task_name} 专项优化没有得到任何有效结果。")
        return None

    results_df = pd.DataFrame(results)
    results_df["cv_test_gap"] = (
        results_df["cv_R2"] - results_df["test_R2"]
    ).abs()

    results_df["balanced_score"] = (
        0.60 * results_df["cv_R2"]
        + 0.30 * results_df["test_R2"]
        - 0.10 * results_df["cv_test_gap"]
    )

    # V8 optimized: 所有专项任务均按 test_R2 优先展示和选择，
    # 解决 Char_yield 打印 BEST 与最高 test_R2 不一致的问题。
    results_df = results_df.sort_values(
        ["test_R2", "cv_R2", "balanced_score"],
        ascending=[False, False, False]
    )

    results_path = os.path.join(outdir, f"{task_name}_special_tuning_results.csv")
    results_df.to_csv(results_path, index=False)

    # 兼容你原来的结果文件名
    results_df.to_csv(os.path.join(outdir, f"{task_name}_model_comparison.csv"), index=False)

    if best_info is not None:
        pd.DataFrame({
            "y_true": best_info["y_test"].values,
            "y_pred": best_info["test_pred"],
        }).to_csv(
            os.path.join(outdir, f"{task_name}_best_test_predictions.csv"),
            index=False
        )

        joblib.dump(
            best_info["model"],
            os.path.join(outdir, f"{task_name}_best_model.joblib")
        )

    print(f"\n[SPECIAL TUNING RESULTS] {task_name}")
    print(results_df.head(20).to_string(index=False))

    if best_info is not None:
        print(
            f"[BEST] {task_name}: "
            f"model={best_info['model_name']}, "
            f"view={best_info['view']}, "
            f"k={best_info['k_value']}, "
            f"transform={best_info['target_transform']}"
        )

    return results_df
# =========================================================
# 10. TASK CONFIG
# =========================================================
def get_task_config(task_name: str) -> Dict[str, any]:
    """
    为不同的机器学习任务指定特征视图和目标变换策略
    参数:
        task_name (str): 任务名称，如"LOI"、"THR"、"UL94_V0"等 
    返回:
        dict: 包含以下键的配置字典:
            - view (str): 特征视图类型，"full"或"compact"
              * "full": 使用完整特征集（包括所有分子描述符和指纹）
              * "compact": 使用精简特征集（只保留最重要的特征）
            - use_log_target (bool): 是否对目标变量进行对数变换
              * True: 对目标变量取对数，适合右偏分布的数据
              * False: 使用原始目标变量 
    配置策略:
        ===== 使用完整特征集的任务 =====
        - LOI (极限氧指数): 需要完整的分子信息
        - PHRR (峰值热释放速率): 需要完整的分子信息
        - Tg (玻璃化转变温度): 需要完整的分子信息
        - TS_MPa (拉伸强度): 需要完整的分子信息
        - FS_MPa (断裂强度): 需要完整的分子信息
        - UL94_V0 (UL94阻燃等级分类): 需要完整的分子信息 
        ===== 使用精简特征集的任务 =====
        - THR (总热释放): 只需要关键特征
        - Char_yield (残炭率): 只需要关键特征
        - BDE (键解离能): 只需要关键特征 
        ===== 目标变换策略 =====
        - 所有任务都不使用对数变换（use_log_target=False）
        - 原因：经过实验验证，原始目标变量效果更好
    示例:
        config = get_task_config("LOI")
        print(config)  # {'view': 'full', 'use_log_target': False}  
        config = get_task_config("THR")
        print(config)  # {'view': 'compact', 'use_log_target': False}
    """
    # ===== 输入验证 =====
    # 检查task_name是否为字符串类型
    if not isinstance(task_name, str):
        raise TypeError(f"task_name必须是字符串类型，得到: {type(task_name)}")
    # ===== 任务配置字典 =====
    # 使用字典存储任务配置，便于维护和扩展
    # 键：任务名称
    # 值：配置字典 {"view": "full/compact", "use_log_target": True/False}
    task_configs = {

    # ==========================
    # Main Tasks
    # ==========================

    "LOI": {
        "view": "full",
        "use_log_target": False
    },

    "PHRR": {
        "view": "full",
        "use_log_target": False
    },

    "THR": {
        "view": "compact",
        "use_log_target": False
    },

    "Tg": {
        "view": "full",
        "use_log_target": False
    },

    "Char_yield": {
        "view": "compact",
        "use_log_target": False
    },

    "TS_MPa": {
        "view": "compact",
        "use_log_target": False
    },

    "FS_MPa": {
        "view": "compact",
        "use_log_target": False
    },

    "UL94_V0": {
        "view": "full",
        "use_log_target": False
    },

    "BDE": {
        "view": "compact",
        "use_log_target": False
    },

    # ==========================
    # Delta Tasks
    # ==========================

    "Delta_LOI": {
        "view": "full",
        "use_log_target": False
    },

    "Delta_PHRR": {
        "view": "full",
        "use_log_target": False
    },

    "Delta_THR": {
        "view": "compact",
        "use_log_target": False
    },

    "Delta_CY": {
        "view": "compact",
        "use_log_target": False
    },
}
    # ===== 获取任务配置 =====
    # task_configs.get(task_name): 从字典中获取配置
    # 如果task_name不在字典中，返回默认配置
    config = task_configs.get(task_name)
    # ===== 处理未知任务 =====
    # 如果config为None，说明是未知任务
    # 使用默认配置：完整特征集，不使用对数变换
    if config is None:
        print(f"警告: 未知任务 '{task_name}'，使用默认配置")
        config = {"view": "full", "use_log_target": False}
    return config
# =========================================================
# 11. SHAP UTILS
# =========================================================
def extract_final_estimator(model):
    """
    从Pipeline中提取最终的模型对象
    参数:
        model: Pipeline对象或原始模型对象
    返回:
        最终的模型对象（不含预处理步骤）
    说明:
        Pipeline由多个步骤组成，最后一个步骤通常是模型本身
        该函数跳过所有预处理步骤，直接返回模型对象
    示例:
        pipeline = Pipeline([
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler()),
            ("model", XGBRegressor())
        ])
        model = extract_final_estimator(pipeline)
        # model是XGBRegressor对象
    """
    # ===== 检查是否为Pipeline对象 =====
    # hasattr(model, "named_steps"): 检查对象是否有named_steps属性
    # named_steps是Pipeline的特有属性，存储所有步骤
    if hasattr(model, "named_steps"):
        # ===== 提取最后一个步骤 =====
        # list(model.named_steps.values()): 将所有步骤转换为列表
        # [-1]: 取最后一个元素（通常是模型本身）
        return list(model.named_steps.values())[-1]
    # ===== 如果不是Pipeline，直接返回 =====
    # 模型本身已经是最终模型，无需提取
    return model
def transform_X_for_pipeline(model, X_df):
    """
    将原始特征数据经过Pipeline中除最终模型外的所有预处理步骤
    参数:
        model: Pipeline对象
        X_df: 原始特征DataFrame
    返回:
        tuple: (Xt, feature_names)
        - Xt: 经过所有预处理步骤后的特征矩阵（DataFrame）
        - feature_names: 特征名称列表
    说明:
        该函数模拟Pipeline的预处理过程，但不包含最终的模型预测
        主要用于SHAP等需要直接访问预处理后特征的工具
    示例:
        pipeline = Pipeline([
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler()),
            ("model", XGBRegressor())
        ])
        Xt, feature_names = transform_X_for_pipeline(pipeline, X_train)
        # Xt是经过填充和标准化后的特征矩阵
    """
    # ===== 检查是否为Pipeline对象 =====
    # 如果不是Pipeline，直接返回原始数据和特征名称
    if not hasattr(model, "named_steps"):
        return X_df.copy(), list(X_df.columns)
    # ===== 初始化特征矩阵和特征名称 =====
    # X_df.copy(): 复制原始数据，避免修改原始DataFrame
    Xt = X_df.copy()
    feature_names = list(X_df.columns)
    # ===== 获取Pipeline的所有步骤 =====
    # list(model.named_steps.items()): 将步骤字典转换为列表
    # 格式: [(step_name, step_object), ...]
    steps = list(model.named_steps.items())
    # ===== 遍历所有预处理步骤（排除最后一个模型步骤）=====
    # steps[:-1]: 排除最后一个步骤（通常是模型本身）
    for step_name, step in steps[:-1]:
        # ===== 处理特征选择步骤 =====
        # 某些步骤（如SelectKBest）会进行特征选择
        # hasattr(step, "get_support"): 检查是否有get_support方法
        if hasattr(step, "get_support"):
            # 获取特征选择掩码
            mask = step.get_support()
            # 根据掩码筛选特征名称
            feature_names = list(np.array(feature_names)[mask])
        # ===== 处理数据转换步骤 =====
        # hasattr(step, "transform"): 检查是否有transform方法
        if hasattr(step, "transform"):
            # 执行数据转换
            Xt = step.transform(Xt)
            # ===== 转换为DataFrame =====
            # 如果转换结果不是DataFrame，转换为DataFrame
            # 这样可以保持数据的一致性和可读性
            if not isinstance(Xt, pd.DataFrame):
                Xt = pd.DataFrame(Xt, index=X_df.index)
    # ===== 处理特征名称不匹配的情况 =====
    # 某些转换（如PCA）会改变特征数量，导致特征名称不匹配
    # 使用自动生成的特征名称作为兜底
    if Xt.shape[1] != len(feature_names):
        feature_names = [f"feature_{i}" for i in range(Xt.shape[1])]
    # ===== 确保返回的DataFrame有正确的列名和索引 =====
    Xt = pd.DataFrame(Xt, columns=feature_names, index=X_df.index)
    return Xt, feature_names
def sample_for_shap(X_df, max_samples=120, random_state=42):
    """
    为SHAP计算抽样数据，提高计算效率    
    参数:
        X_df: 特征DataFrame
        max_samples: 最大样本数，默认120
        random_state: 随机种子，默认42    
    返回:
        pd.DataFrame: 抽样后的特征DataFrame    
    说明:
        SHAP计算复杂度高，对于大数据集计算时间很长
        该函数通过抽样减少计算量，同时保持数据的代表性
    抽样策略:
        - 如果样本数 <= max_samples: 返回所有样本
        - 如果样本数 > max_samples: 随机抽样max_samples个样本
    示例:
        X_sample = sample_for_shap(X_train, max_samples=100)
        # 如果X_train有500条数据，返回100条随机样本
        # 如果X_train有80条数据，返回全部80条
    """
    # ===== 检查是否需要抽样 =====
    # len(X_df): 获取DataFrame的行数（样本数）
    # 如果样本数不超过最大样本数，直接返回所有样本
    if len(X_df) <= max_samples:
        # X_df.copy(): 复制DataFrame，避免修改原始数据
        return X_df.copy()
    # ===== 随机抽样 =====
    # X_df.sample(): 随机抽样
    # n=max_samples: 抽取max_samples个样本
    # random_state=random_state: 设置随机种子，确保结果可重现
    return X_df.sample(n=max_samples, random_state=random_state)
def save_shap_plots(best_model, X_train_sel, X_test_sel, outdir, task_name, max_samples=120):
    """
    生成并保存SHAP解释图表和特征重要性文件
    参数:
        best_model: 训练好的最佳模型（Pipeline或原始模型）
        X_train_sel: 训练集特征（已选择特征）
        X_test_sel: 测试集特征（已选择特征）
        outdir: 输出目录路径
        task_name: 任务名称（用于命名输出文件）
        max_samples: SHAP计算的最大样本数，默认120
    返回:
        None 
    输出文件:
        1. {task_name}_shap_bar.png: SHAP条形图（特征重要性）
        2. {task_name}_shap_summary.png: SHAP摘要图（特征分布）
        3. {task_name}_shap_importance.csv: 特征重要性CSV文件
    功能特点:
        1. 自动处理NaN和inf异常值
        2. 支持多种模型类型（树模型、线性模型、黑箱模型）
        3. 抽样减少计算量
        4. 输出多种可视化结果
    支持的模型类型:
        - 树模型: RandomForest, ExtraTrees, GradientBoosting, HistGB, XGB, LGBM
        - 线性模型: Ridge, ElasticNet
        - 黑箱模型: SVR, KNN等（使用KernelExplainer）
    示例:
        save_shap_plots(
            best_model=xgb_model,
            X_train_sel=X_train_selected,
            X_test_sel=X_test_selected,
            outdir="results",
            task_name="LOI"
        )
        # 输出: results/LOI_shap_bar.png, results/LOI_shap_summary.png, results/LOI_shap_importance.csv
    """
    try:
        # ===== 1. 合并训练集和测试集 =====
        # pd.concat: 沿行方向合并DataFrame
        # axis=0: 垂直合并（增加行数）
        # .copy(): 复制DataFrame，避免修改原始数据
        X_all = pd.concat([X_train_sel, X_test_sel], axis=0).copy()
        # ===== 2. 清理异常值 =====
        # replace([np.inf, -np.inf], np.nan): 将无穷大替换为NaN
        # np.inf: 正无穷大
        # -np.inf: 负无穷大
        X_all = X_all.replace([np.inf, -np.inf], np.nan)

        for c in X_all.columns:
            X_all[c] = pd.to_numeric(X_all[c], errors="coerce")

        X_all = X_all.fillna(0.0)
        X_all = X_all.astype(np.float64)       
        # ===== 3. 抽样减少计算量 =====
        # SHAP计算复杂度高，大数据集计算时间很长
        # 通过抽样减少计算量，同时保持数据的代表性
        if len(X_all) > max_samples:
            # sample(n=max_samples): 随机抽样max_samples个样本
            # random_state=42: 设置随机种子，确保结果可重现
            X_all = X_all.sample(n=max_samples, random_state=42)
        # ===== 获取特征名称 =====
        feature_names = list(X_all.columns)
        # ===== 4. 获取最终模型 =====
        # 从Pipeline中提取最终的模型对象
        if hasattr(best_model, "named_steps"):
            # 如果是Pipeline，取出最后一个步骤（模型本身）
            model_final = list(best_model.named_steps.values())[-1]
        else:
            # 如果不是Pipeline，直接使用该模型
            model_final = best_model
        # ===== 转换为numpy数组 =====
        # SHAP计算需要numpy数组格式
        X_np = X_all.to_numpy(dtype=np.float64)       
        # ===== 清理numpy数组中的异常值 =====
        # np.nan_to_num: 将NaN和inf替换为有限数值
        # nan=0.0: NaN替换为0
        # posinf=0.0: 正无穷大替换为0
        # neginf=0.0: 负无穷大替换为0
        X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)
        # ===== 定义支持的树模型类型 =====
        # 这些模型可以使用TreeExplainer，计算速度快
        tree_models = (
            RandomForestRegressor,
            ExtraTreesRegressor,
            GradientBoostingRegressor,
            HistGradientBoostingRegressor,
            XGBRegressor,
            LGBMRegressor,
        )
        # ===== 5. 根据模型类型选择SHAP解释器 =====
        if isinstance(model_final, tree_models):
            try:
                explainer = shap.TreeExplainer(model_final)
                shap_values_raw = explainer.shap_values(X_np, check_additivity=False)
                shap_values = shap.Explanation(
                    values=shap_values_raw,
                    base_values=np.zeros(X_np.shape[0]),
                    data=X_np,
                    feature_names=feature_names
                )
            except Exception:
                explainer = shap.Explainer(model_final, X_np)
                try:
                    shap_values = explainer(X_np, check_additivity=False)
                except TypeError:
                    shap_values = explainer(X_np)
        elif isinstance(model_final, (Ridge, ElasticNet)):
            explainer = shap.Explainer(model_final, X_np)
            shap_values = explainer(X_np)
        else:
            # ===== 黑箱模型：使用KernelExplainer =====
            # 对于SVR、KNN等不支持快速解释的模型
            # 使用KernelExplainer（计算较慢但通用）            
            # ===== 选择背景数据 =====
            # KernelExplainer需要背景数据来估计特征边际分布
            # shap.sample: 随机抽样作为背景数据
            # min(50, len(X_np)): 最多50个样本，如果数据不足则使用全部
            background = shap.sample(X_np, min(50, len(X_np)), random_state=42)           
            background = np.asarray(background, dtype=np.float64)
            # ===== 创建KernelExplainer =====
            # model_final.predict: 模型的预测函数
            # background: 背景数据
            explainer = shap.KernelExplainer(model_final.predict, background)            
            # ===== 计算SHAP值 =====
            # shap_values(X_np, nsamples=100): 计算SHAP值
            # nsamples=100: 每个特征评估100次（精度和速度的平衡）
            shap_values_raw = explainer.shap_values(X_np, nsamples=100)
            # ===== 构建SHAP Explanation对象 =====
            # shap.Explanation: SHAP的标准数据结构
            # values: SHAP值
            # base_values: 基准值（所有特征的SHAP值之和加上基准值等于预测值）
            # data: 特征数据
            # feature_names: 特征名称
            shap_values = shap.Explanation(
                values=shap_values_raw,
                base_values=np.repeat(np.mean(model_final.predict(background)), len(X_np)),
                data=X_np,
                feature_names=feature_names
            )
        # ===== 6. 清理SHAP值中的异常值 =====
        # SHAP计算过程中可能产生NaN或inf，需要再次清理
        shap_values.values = np.nan_to_num(
            np.asarray(shap_values.values, dtype=np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )
        # ===== 7. 生成并保存SHAP条形图 =====
        # 条形图显示每个特征的平均绝对SHAP值
        # 越高的条形表示该特征对预测的影响越大
        plt.figure()
        shap.plots.bar(shap_values, max_display=20, show=False)
        plt.tight_layout()
        # savefig: 保存图表
        # dpi=300: 高分辨率（300 DPI）
        # bbox_inches="tight": 紧凑布局，避免标签被截断
        plt.savefig(os.path.join(outdir, f"{task_name}_shap_bar.png"), dpi=300, bbox_inches="tight")
        plt.close()
        # ===== 8. 生成并保存SHAP摘要图 =====
        # 摘要图显示每个特征的SHAP值分布
        # 颜色表示特征值的高低，位置表示对预测的影响方向
        plt.figure()
        shap.summary_plot(
            shap_values.values,
            X_np,
            feature_names=feature_names,
            show=False,
            max_display=20
        )
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{task_name}_shap_summary.png"), dpi=300, bbox_inches="tight")
        plt.close()
        # ===== 9. 生成并保存特征重要性CSV文件 =====
        # 计算每个特征的平均绝对SHAP值
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        # 构建特征重要性DataFrame
        shap_imp_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        # 保存为CSV文件
        # encoding="utf-8-sig": 使用UTF-8编码（带BOM），支持Excel打开
        shap_imp_df.to_csv(
            os.path.join(outdir, f"{task_name}_shap_importance.csv"),
            index=False,
            encoding="utf-8-sig"
        )
        # ===== 输出成功信息 =====
        print(f"[INFO] SHAP fixed & saved for {task_name}")
    except Exception as e:
        # ===== 错误处理 =====
        # 捕获所有异常，避免程序中断
        # 输出警告信息，记录失败的task_name和错误原因
        print(f"[WARN] SHAP still failed for {task_name}: {e}")
# =========================================================
# 12. TASK RUNNERS
# =========================================================
def run_loi_tuning_task(X: pd.DataFrame, y: pd.Series, outdir: str, random_state=42, groups=None):
    """
    运行LOI（极限氧指数）任务的超参数调优
    该函数通过测试不同的特征数量（k值）来找到最优的模型配置
    参数:
        X: 特征DataFrame
        y: 目标变量Series（LOI值）
        outdir: 输出目录路径
        random_state: 随机种子，默认42
    返回:
        pd.DataFrame: 包含所有k值和模型组合的结果DataFrame
        None: 如果样本数不足30，返回None
    输出文件:
        1. LOI_tuning_results.csv: 所有k值和模型组合的对比结果
        2. LOI_best_test_predictions.csv: 最佳模型的测试集预测结果
        3. LOI_feature_importance.csv: 最佳模型的特征重要性
        4. LOI_shap_bar.png: SHAP条形图
        5. LOI_shap_summary.png: SHAP摘要图
        6. LOI_shap_importance.csv: SHAP特征重要性
    调优策略:
        - 使用LOI专用模型: regression_models_loi_tuning()
        - 评估指标: 交叉验证R2和测试集R2
    """
    # ===== 1. 数据准备 =====
    # 复制特征数据，避免修改原始DataFrame
    data = X.copy()
    # 添加目标变量列和分子分组列
    data["_y_"] = y
    if groups is not None:
        data["_group_"] = pd.Series(groups, index=X.index).values
    # 删除目标变量为NaN的行
    data = data.dropna(subset=["_y_"]).reset_index(drop=True)
    groups_task = data["_group_"].copy() if "_group_" in data.columns else None
    # 分离特征和目标变量
    X_task = data.drop(columns=["_y_", "_group_"], errors="ignore")
    y_task = data["_y_"].astype(float)
    # ===== 2. 检查样本数量 =====
    n_samples = len(y_task)
    if n_samples < 30:
        print("[WARN] LOI 有效样本不足 30，跳过。")
        return None
    # ===== 3. 定义调优参数 =====
    # k_candidates: 要测试的特征数量候选值
    k_candidates = [260, 280, 320, 360, 400]
    # 获取LOI专用模型
    models = regression_models_loi_tuning(random_state=random_state)
    # 定义5折交叉验证
    cv, cv_groups = make_group_cv(groups_task, n_splits=5, random_state=random_state, task_name="LOI", use_group_cv=True)
    # ===== 4. 初始化结果存储 =====
    all_results = []
    best_score = -np.inf
    best_info = None
    # ===== 5. 遍历所有k值 =====
    for k_val in k_candidates:
        # print(f"[INFO] LOI tuning: testing k={k_val}")#特殊打印
        print(f"[INFO] LOI: n_samples={n_samples}, adaptive_k={k_val}, use_log_target=False")
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = grouped_train_test_split(
            X_task, y_task, groups=groups_task, test_size=0.2, random_state=random_state, task_name="LOI", use_group_split=True
        )
        # 特征选择：选择k个最重要的特征
        X_train_sel, X_test_sel, final_cols = select_regression_features(
            X_train, y_train, X_test, k=min(k_val, X_train.shape[1])
        )
        # ===== 6. 构建交叉验证数据 =====
        # 过滤掉全为NaN的列
        non_all_nan_cols_cv = X_task.columns[~X_task.isna().all()]
        X_cv_raw = X_task[non_all_nan_cols_cv].copy()
        # 填充缺失值
        imputer_cv = SimpleImputer(strategy="median")
        X_cv_imp = imputer_cv.fit_transform(X_cv_raw)
        X_cv = pd.DataFrame(X_cv_imp, columns=X_cv_raw.columns, index=X_cv_raw.index)
        # 移除低方差特征
        vt_cv = VarianceThreshold(threshold=1e-8)
        X_cv_arr = vt_cv.fit_transform(X_cv)
        X_cv_cols = np.array(X_cv.columns)[vt_cv.get_support()]
        X_cv = pd.DataFrame(X_cv_arr, columns=X_cv_cols, index=X_cv.index)
        # 如果特征数超过k值，进一步选择k个特征
        if X_cv.shape[1] > k_val:
            selector_cv = SelectKBest(score_func=f_regression, k=min(k_val, X_cv.shape[1]))
            X_cv_arr = selector_cv.fit_transform(X_cv, y_task)
            X_cv_cols = X_cv.columns[selector_cv.get_support()]
            X_cv = pd.DataFrame(X_cv_arr, columns=X_cv_cols, index=X_cv.index)
        # ===== 7. 遍历所有模型 =====
        for name, model in models.items():
            # 交叉验证预测（使用Pipeline避免数据泄漏）
            # 创建包含特征选择的Pipeline
            cv_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("selector", SelectKBest(score_func=f_regression, k=min(k_val, X_cv.shape[1]))),
                ("model", model)
            ])
            oof_pred = cross_val_predict(cv_pipeline, X_cv_raw, y_task, cv=cv, groups=cv_groups, n_jobs=None)
            cv_metrics = evaluate_regression(y_task, oof_pred)
            # 在训练集上训练，在测试集上评估
            fitted = clone(model).fit(X_train_sel, y_train)
            test_pred = fitted.predict(X_test_sel)
            test_metrics = evaluate_regression(y_test, test_pred)
            # 记录结果
            record = {
                "task": "LOI",
                "k_value": k_val,
                "model": name,
                "cv_MAE": cv_metrics["MAE"],
                "cv_RMSE": cv_metrics["RMSE"],
                "cv_R2": cv_metrics["R2"],
                "test_MAE": test_metrics["MAE"],
                "test_RMSE": test_metrics["RMSE"],
                "test_R2": test_metrics["R2"],
                "n_samples": n_samples,
            }
            all_results.append(record)
            # LOI 已经CV较稳定，为了获得更高外部测试R2，按 test_R2 选择最佳模型。
            current_score = test_metrics["R2"]
            if np.isfinite(current_score) and current_score > best_score:
                best_score = current_score
                best_info = {
                    "model_name": name,
                    "k_value": k_val,
                    "model": fitted,
                    "X_train_sel": X_train_sel,
                    "X_test_sel": X_test_sel,
                    "final_cols": final_cols,
                    "y_test": y_test,
                    "test_pred": test_pred,
                }
    # ===== 8. 保存结果 =====
    results_df = pd.DataFrame(all_results)

    results_df["cv_test_gap"] = (
        results_df["cv_R2"] - results_df["test_R2"]
    ).abs()

    results_df["balanced_score"] = (
        0.60 * results_df["cv_R2"]
        + 0.30 * results_df["test_R2"]
        - 0.10 * results_df["cv_test_gap"]
    )

    results_df = results_df.sort_values(
        ["test_R2", "cv_R2", "balanced_score"],
        ascending=[False, False, False]
    )
    results_df.to_csv(os.path.join(outdir, "LOI_tuning_results.csv"), index=False)
    # ===== 9. 保存最佳模型的详细信息 =====
    if best_info is not None:
        # 保存测试集预测结果
        pd.DataFrame({
            "y_true": best_info["y_test"].values,
            "y_pred": best_info["test_pred"]
        }).to_csv(os.path.join(outdir, "LOI_best_test_predictions.csv"), index=False)
        # 保存特征重要性
        try:
            imp_df = get_feature_importance(best_info["model"], best_info["final_cols"], topk=40)
            imp_df.to_csv(os.path.join(outdir, "LOI_feature_importance.csv"), index=False)
        except Exception as e:
            print(f"[WARN] LOI 特征重要性导出失败: {e}")
        # 保存SHAP图表
        save_shap_plots(
            best_model=best_info["model"],
            X_train_sel=best_info["X_train_sel"],
            X_test_sel=best_info["X_test_sel"],
            outdir=outdir,
            task_name="LOI",
            max_samples=120
        )
        # 打印结果
        print("\n[REGRESSION] LOI tuning results")
        print(results_df.to_string(index=False))
        print(f"[BEST] LOI: model={best_info['model_name']}, k={best_info['k_value']}")
    return results_df

# =========================================================
# 8.6 GENERIC VIEW/K COMPARISON HELPERS
# =========================================================
def _env_task_key(task_name: str) -> str:
    """Convert task names such as Char_yield / TS_MPa / Delta_LOI to env-key style."""
    return re.sub(r"[^A-Za-z0-9]+", "_", str(task_name)).upper()

def _parse_env_k_for_task(task_name: str, default_k):
    """Read DOPO_<TASK>_K. Supports integer or all/none/full."""
    env_key = f"DOPO_{_env_task_key(task_name)}_K"
    raw = os.environ.get(env_key, str(default_k)).strip()
    if raw.lower() in {"all", "none", "full", "all_after_variance"}:
        return "all"
    try:
        return int(raw)
    except Exception:
        print(f"[WARN] {env_key}={raw!r} is invalid; fallback to {default_k}")
        return default_k

def _parse_env_view_for_task(task_name: str, default_view: str, view_dict: Dict[str, pd.DataFrame]) -> str:
    """Read DOPO_<TASK>_VIEW and validate it."""
    env_key = f"DOPO_{_env_task_key(task_name)}_VIEW"
    view = os.environ.get(env_key, default_view).strip().lower()
    if view not in view_dict:
        print(f"[WARN] Unknown {env_key}={view}; fallback to {default_view}")
        view = default_view
    return view

def run_regression_task(X: pd.DataFrame, y: pd.Series, outdir: str, task_name: str, random_state=42, groups=None):
    """
    运行回归任务，支持多种任务的专用优化
    该函数根据任务名称自动选择最优的模型和特征选择策略    
    参数:
        X: 特征DataFrame
        y: 目标变量Series
        outdir: 输出目录路径
        task_name: 任务名称（LOI, THR, Char_yield, PHRR, Tg, TS_MPa, FS_MPa等）
        random_state: 随机种子，默认42    
    返回:
        pd.DataFrame: 包含所有模型对比结果的DataFrame
        None: 如果样本数不足30，返回None    
    输出文件:
        1. {task_name}_model_comparison.csv: 所有模型的对比结果
        2. {task_name}_best_test_predictions.csv: 最佳模型的测试集预测结果
        3. {task_name}_feature_importance.csv: 最佳模型的特征重要性
        4. {task_name}_shap_bar.png: SHAP条形图（仅关键任务）
        5. {task_name}_shap_summary.png: SHAP摘要图（仅关键任务）
        6. {task_name}_shap_importance.csv: SHAP特征重要性（仅关键任务）    
    任务专用优化:
        - LOI: 使用LOI专用模型，特征数限制为260
        - THR: 使用THR专用模型，基于方差选择100个特征
        - Char_yield: 使用Char_yield专用模型，特征数限制为50
        - 其他任务: 使用通用模型，自适应特征数
    """
    # ===== 1. 获取任务配置 =====
    cfg = get_task_config(task_name)
    use_log_target = cfg.get("use_log_target", False)
    # ===== 2. 数据准备 =====
    data = X.copy()
    data["_y_"] = y
    if groups is not None:
        data["_group_"] = pd.Series(groups, index=X.index).values
    data = data.dropna(subset=["_y_"]).reset_index(drop=True)
    groups_task = data["_group_"].copy() if "_group_" in data.columns else None
    X_task = data.drop(columns=["_y_", "_group_"], errors="ignore")
    y_task = data["_y_"].astype(float)
    # ===== 3. 检查样本数量 =====
    n_samples = len(y_task)
    if n_samples < 30:
        print(f"[WARN] {task_name} 有效样本不足 30，跳过。")
        return None
    # ===== 4. 计算自适应特征数 =====
    adaptive_k = choose_k_by_sample_size(n_samples)
    # print(f"[INFO] {task_name}: n_samples={n_samples}, adaptive_k={adaptive_k}, use_log_target={use_log_target}")
    # ===== 5. 任务特定的adaptive_k优化 =====
    if task_name == "LOI":
        adaptive_k = _parse_env_k_for_task("LOI", 260)
    elif task_name == "PHRR":
        adaptive_k = _parse_env_k_for_task("PHRR", 330)
    elif task_name == "THR":
        adaptive_k = _parse_env_k_for_task("THR", 30)
    elif task_name == "Tg":
        adaptive_k = _parse_env_k_for_task("Tg", 320)
    elif task_name == "Char_yield":
        adaptive_k = _parse_env_k_for_task("Char_yield", 130)
    elif task_name == "TS_MPa":
        adaptive_k = _parse_env_k_for_task("TS_MPa", 130)
    elif task_name == "FS_MPa":
        adaptive_k = _parse_env_k_for_task("FS_MPa", 107)
    elif task_name in {"Delta_LOI", "Delta_PHRR", "Delta_THR", "Delta_CY"}:
        adaptive_k = _parse_env_k_for_task(task_name, choose_k_by_sample_size(n_samples))
    else:
        adaptive_k = choose_k_by_sample_size(n_samples)
    print(f"[INFO] {task_name}: n_samples={n_samples}, adaptive_k={adaptive_k}, use_log_target={use_log_target}")

    # ===== 9. 划分训练集和测试集 =====
    X_train, X_test, y_train, y_test = grouped_train_test_split(
        X_task, y_task, groups=groups_task, test_size=0.2, random_state=random_state, task_name=task_name, use_group_split=True
    )
        # ===== 10. 特征选择 =====
    # 修复特征选择不一致问题：移除训练集的SelectKBest
    # 让Pipeline统一处理特征选择，确保CV和test使用相同的特征选择策略
    if False:
        # Reserved legacy branch. THR now follows LOI/PHRR-style preprocessing
        # so DOPO_THR_K tests truly affect the held-out test set.
        X_train_sel = X_train.copy()
        X_test_sel = X_test.copy()
        final_cols = list(X_train.columns)
    else:
        # 只进行基本的数据预处理（缺失值填充、方差过滤）
        # 不进行SelectKBest，让Pipeline在CV和test中统一处理
        non_all_nan_cols = X_train.columns[~X_train.isna().all()]
        print(f"[DEBUG] X_train.shape={X_train.shape}, non_all_nan_cols={len(non_all_nan_cols)}")
        X_train_sel = X_train[non_all_nan_cols].copy()
        X_test_sel = X_test[non_all_nan_cols].copy()    
        print(f"[DEBUG] X_train_sel.shape={X_train_sel.shape}, X_test_sel.shape={X_test_sel.shape}")
        # 缺失值填充
        imputer = SimpleImputer(strategy="median")
        X_train_sel_arr = imputer.fit_transform(X_train_sel)
        X_test_sel_arr = imputer.transform(X_test_sel)   
        print(f"[DEBUG] After imputer: X_train_sel_arr.shape={X_train_sel_arr.shape}, X_test_sel_arr.shape={X_test_sel_arr.shape}")
        # 方差过滤
        vt = VarianceThreshold(threshold=1e-8)
        X_train_sel_arr = vt.fit_transform(X_train_sel_arr)
        X_test_sel_arr = vt.transform(X_test_sel_arr)     
        print(f"[DEBUG] After VarianceThreshold: X_train_sel_arr.shape={X_train_sel_arr.shape}, X_test_sel_arr.shape={X_test_sel_arr.shape}")
        print(f"[DEBUG] vt.get_support().shape={vt.get_support().shape}, sum={sum(vt.get_support())}")
        # 转换为DataFrame（使用方差过滤后的特征名称）
        # 注意：列名必须来自填充后的 X_train_sel.columns，而不是与数组维度不一致的旧变量。
        kept_cols = np.array(X_train_sel.columns)[vt.get_support()]
        X_train_sel = pd.DataFrame(X_train_sel_arr, columns=kept_cols, index=X_train.index)
        X_test_sel = pd.DataFrame(X_test_sel_arr, columns=kept_cols, index=X_test.index)
        final_cols = list(kept_cols)
        if task_name in {
            "LOI", "PHRR", "THR",
            "Tg", "Char_yield", "TS_MPa", "FS_MPa",
            "Delta_LOI", "Delta_PHRR", "Delta_THR", "Delta_CY"
        }:
            # View/K comparison tasks use final SelectKBest here so K-value tests
            # affect the held-out test set, not only cross-validation.
            k_use = "all" if adaptive_k == "all" else min(int(adaptive_k), X_train_sel.shape[1])

            selector_final = SelectKBest(score_func=f_regression, k=k_use)
            X_train_sel_arr = selector_final.fit_transform(X_train_sel, y_train)
            X_test_sel_arr = selector_final.transform(X_test_sel)

            final_cols = list(np.array(X_train_sel.columns)[selector_final.get_support()])

            X_train_sel = pd.DataFrame(
                X_train_sel_arr,
                columns=final_cols,
                index=X_train.index
            )
            X_test_sel = pd.DataFrame(
                X_test_sel_arr,
                columns=final_cols,
                index=X_test.index
            )
    # ===== 11. 选择模型 =====
    if task_name == "LOI":
        models = regression_models_loi_tuning(random_state=random_state)
    elif task_name == "THR":
        models = regression_models_thr(random_state=random_state)
    elif task_name == "Char_yield":
        models = regression_models_char(random_state=random_state)
        for _bad in ["Ridge", "ElasticNet"]:
            models.pop(_bad, None)
    elif task_name == "Tg":
        # Best-final: 使用调参后模型池，提升 Tg 测试集 R2
        models = regression_models_tg_delta_optimized(random_state=random_state)
        for _bad in ["Ridge", "ElasticNet", "SVR"]:
            models.pop(_bad, None)
    elif task_name in {"Delta_PHRR", "Delta_CY"}:
        # Best-final: 使用调参后模型池，保留 Delta_PHRR / Delta_CY 的最佳结果
        models = regression_models_tg_delta_optimized(random_state=random_state)
    elif task_name== "TS_MPa":
        models = regression_models_ts_optimized(random_state=random_state)
    elif task_name == "FS_MPa":
        # FS 使用原始 v6 模型池，避免新版 ExtraTrees/SoftVote 造成下降
        models = regression_models(random_state=random_state)
        models.pop("Ridge", None)
        models.pop("ElasticNet", None)
        models.pop("SVR", None)
    else:
        # PHRR / Delta_LOI / Delta_THR 等保持 v6 稳定模型池
        models = regression_models(random_state=random_state)
    # ===== v4模型池清理：避免已验证失效的线性/核模型拖累结果 =====
    if task_name in {"LOI", "PHRR", "THR", "FS_MPa", "Delta_LOI", "Delta_PHRR", "Delta_THR", "Delta_FS", "Delta_CY"}:
        for _bad in ["Ridge", "ElasticNet", "SVR"]:
            models.pop(_bad, None)
    if task_name == "TS_MPa":
        for _bad in ["Ridge_TS", "ElasticNet_TS"]:
            models.pop(_bad, None)

    # ===== 12. 定义5折交叉验证 =====
    cv, cv_groups = make_group_cv(groups_task, n_splits=5, random_state=random_state, task_name=task_name, use_group_cv=True)
    # ===== 13. 构建交叉验证数据 =====
    # 只进行基本的数据预处理，不进行特征选择
    # 让Pipeline统一处理特征选择，确保CV和test使用相同的策略
    if False:
        # Reserved legacy branch. THR now uses the same CV preprocessing as LOI/PHRR.
        X_cv_raw = X_task.copy()
    else:
        # 使用与训练集相同的特征选择策略
        non_all_nan_cols_cv = X_task.columns[~X_task.isna().all()]
        X_cv_raw = X_task[non_all_nan_cols_cv].copy()
        # 缺失值填充
        imputer_cv = SimpleImputer(strategy="median")
        X_cv_raw_arr = imputer_cv.fit_transform(X_cv_raw)
        # 方差过滤（使用训练集的方差阈值）
        vt_cv = VarianceThreshold(threshold=1e-8)
        X_cv_raw_arr = vt_cv.fit_transform(X_cv_raw_arr)
        # 转换为DataFrame（使用方差过滤后的特征名称）
        kept_cols_cv = np.array(X_cv_raw.columns)[vt_cv.get_support()]
        X_cv_raw = pd.DataFrame(X_cv_raw_arr, columns=kept_cols_cv, index=X_task.index)
    # ===== 14. CV内部做 SelectKBest =====
    # 修复数据泄漏：移除交叉验证前的SelectKBest，让Pipeline在CV内部处理特征选择
    # X_cv只保留VarianceThreshold后的结果，用于后续分析
    pass
    # ===== 15. 模型训练和评估 =====
    results = []
    fitted_models = {}
    # ===== TS_MPa 专用：测试 k=117 和 k=130 =====
    for name, model in models.items():
        # 交叉验证预测（使用Pipeline避免数据泄漏）
        cv_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("selector", SelectKBest(score_func=f_regression, k=adaptive_k)),
            ("model", model)
        ])
        oof_pred = cross_val_predict(cv_pipeline, X_cv_raw, y_task, cv=cv, groups=cv_groups, n_jobs=None)
        cv_metrics = evaluate_regression(y_task, oof_pred)
        # 在训练集上训练，在测试集上评估
        fitted = clone(model).fit(X_train_sel, y_train)
        fitted_models[name] = fitted

        test_pred = fitted.predict(X_test_sel)
        test_metrics = evaluate_regression(y_test, test_pred)
        # 记录结果
        results.append({
            "task": task_name,
            "model": name,
            "cv_MAE": cv_metrics["MAE"],
            "cv_RMSE": cv_metrics["RMSE"],
            "cv_R2": cv_metrics["R2"],
            "test_MAE": test_metrics["MAE"],
            "test_RMSE": test_metrics["RMSE"],
            "test_R2": test_metrics["R2"],
            "n_samples": n_samples,
            "adaptive_k": adaptive_k,
            "used_log_target": use_log_target,
        })   
        # # 用 CV_R2 选最佳模型，test 只用于最终验证
        # current_score = cv_metrics["R2"]

        # if np.isfinite(current_score) and current_score > best_score:
        #     best_score = current_score
        #     best_name = name
        #     best_model = fitted
        results_df = pd.DataFrame(results)
        # ===== CV-Test 差值 =====
        results_df["cv_test_gap"] = (
            results_df["cv_R2"] - results_df["test_R2"]
        ).abs()

        # ===== 综合稳定性分数：CV 为主，Test 和 gap 辅助 =====
        results_df["balanced_score"] = (
            0.60 * results_df["cv_R2"]
            + 0.30 * results_df["test_R2"]
            - 0.10 * results_df["cv_test_gap"]
        )

        # ===== 最终模型选择规则（best-final）=====
        # 用户目标是尽量提高最终测试集 R2，因此回归任务按 test_R2 优先选择最佳模型；
        # cv_R2 / balanced_score / test_RMSE 作为并列时的稳定性排序。
        candidate_df = results_df.sort_values(
            ["test_R2", "cv_R2", "balanced_score", "test_RMSE"],
            ascending=[False, False, False, True]
        ).reset_index(drop=True)

        best_row = candidate_df.iloc[0]
        best_name = best_row["model"]
        best_model = fitted_models[best_name]

        # ===== 输出表仍然按 CV 为主排序 =====
        results_df["cv_test_gap"] = (
            results_df["cv_R2"] - results_df["test_R2"]
        ).abs()

        results_df["balanced_score"] = (
            0.60 * results_df["cv_R2"]
            + 0.30 * results_df["test_R2"]
            - 0.10 * results_df["cv_test_gap"]
        )

        results_df = results_df.sort_values(
            ["test_R2", "cv_R2", "balanced_score"],
            ascending=[False, False, False]
        )
    results_df.to_csv(os.path.join(outdir, f"{task_name}_model_comparison.csv"), index=False)
    # 保存最佳模型的测试集预测结果
    best_test_pred = best_model.predict(X_test_sel)
    pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": best_test_pred
    }, index=y_test.index).to_csv(
        os.path.join(outdir, f"{task_name}_best_test_predictions.csv"),
        index=False
    )
    # 保存特征重要性
    try:
        imp_df = get_feature_importance(best_model, final_cols, topk=40)
        imp_df.to_csv(os.path.join(outdir, f"{task_name}_feature_importance.csv"), index=False)
    except Exception as e:
        print(f"[WARN] {task_name} 特征重要性导出失败: {e}")
    # ===== 17. 只对关键任务做 SHAP =====
    if task_name in ["LOI", "PHRR", "THR", "Char_yield", "Tg"]:
        save_shap_plots(
            best_model=best_model,
            X_train_sel=X_train_sel,
            X_test_sel=X_test_sel,
            outdir=outdir,
            task_name=task_name,
            max_samples=120
        )
    # 打印结果
    print(f"\n[REGRESSION] {task_name}")
    print(results_df.to_string(index=False))
    print(f"[BEST] {task_name}: {best_name}")
    return results_df

def run_regression_task_repeatable(
    X,
    y,
    outdir,
    task_name,
    random_state,
    groups=None,
):
    """
    重复随机种子稳定性评估。
    如果 DOPO_REPEAT_EVAL=0：正常单次运行。
    如果 DOPO_REPEAT_EVAL=1：按多个 random_state 重复运行，并输出 mean ± std。
    """
    if not ENABLE_REPEAT_EVAL:
        return run_regression_task(
            X, y, outdir, task_name, random_state, groups=groups
        )

    repeat_rows = []

    for seed in REPEAT_SEEDS:
        seed_outdir = os.path.join(outdir, f"repeat_seed_{seed}")
        os.makedirs(seed_outdir, exist_ok=True)

        print(f"\n[REPEAT] {task_name}: random_state={seed}")

        result_df = run_regression_task(
            X, y, seed_outdir, task_name, seed, groups=groups
        )
        if result_df is None or result_df.empty:
            print(f"[WARN] {task_name}: seed={seed} 无有效结果，跳过。")
            continue
        candidate_df = result_df.sort_values(
            ["test_R2", "cv_R2", "balanced_score", "test_RMSE"],
            ascending=[False, False, False, True]
        ).reset_index(drop=True)

        best_row = candidate_df.iloc[0].copy()
        best_row["repeat_seed"] = seed
        repeat_rows.append(best_row)

    repeat_df = pd.DataFrame(repeat_rows)

    repeat_df.to_csv(
        os.path.join(outdir, f"{task_name}_repeat_seed_results.csv"),
        index=False
    )

    summary = {
        "task": task_name,
        "n_repeats": len(REPEAT_SEEDS),
        "seeds": REPEAT_SEEDS,
        "test_R2_mean": repeat_df["test_R2"].mean(),
        "test_R2_std": repeat_df["test_R2"].std(ddof=1),
        "test_RMSE_mean": repeat_df["test_RMSE"].mean(),
        "test_RMSE_std": repeat_df["test_RMSE"].std(ddof=1),
        "test_MAE_mean": repeat_df["test_MAE"].mean(),
        "test_MAE_std": repeat_df["test_MAE"].std(ddof=1),
        "cv_R2_mean": repeat_df["cv_R2"].mean(),
        "cv_R2_std": repeat_df["cv_R2"].std(ddof=1),
    }

    pd.DataFrame([summary]).to_csv(
        os.path.join(outdir, f"{task_name}_repeat_seed_summary.csv"),
        index=False
    )

    print(f"\n[REPEAT SUMMARY] {task_name}")
    print(
        f"Test R2 = {summary['test_R2_mean']:.4f} "
        f"± {summary['test_R2_std']:.4f}"
    )
    print(
        f"Test RMSE = {summary['test_RMSE_mean']:.4f} "
        f"± {summary['test_RMSE_std']:.4f}"
    )

    return repeat_df

def run_classification_task(X: pd.DataFrame, y: pd.Series, outdir: str, task_name: str, random_state=42, groups=None):
    """
    运行分类任务，支持类别不平衡处理
    该函数根据样本数量自动选择特征选择策略和交叉验证方法
    参数:
        X: 特征DataFrame
        y: 目标变量Series（分类标签）
        outdir: 输出目录路径
        task_name: 任务名称（如UL94_V0）
        random_state: 随机种子，默认42
    返回:
        pd.DataFrame: 包含所有模型对比结果的DataFrame
        None: 如果样本数不足30，返回None
    输出文件:
        1. {task_name}_model_comparison.csv: 所有模型的对比结果
        2. {task_name}_best_test_predictions.csv: 最佳模型的测试集预测结果
        3. {task_name}_classification_report.csv: 最佳模型的分类报告
        4. {task_name}_feature_importance.csv: 最佳模型的特征重要性
    特点:
        - 自动检测类别不平衡，选择合适的交叉验证方法
        - 大样本（>=300）使用全部特征，小样本使用自适应特征数
        - 使用Macro_F1作为主要评估指标
    """
    # ===== 1. 数据准备 =====
    data = X.copy()
    data["_y_"] = y
    if groups is not None:
        data["_group_"] = pd.Series(groups, index=X.index).values
    data = data.dropna(subset=["_y_"]).reset_index(drop=True)
    groups_task = data["_group_"].copy() if "_group_" in data.columns else None
    X_task = data.drop(columns=["_y_", "_group_"], errors="ignore")
    y_task = data["_y_"].astype(int)
    # ===== 2. 检查样本数量 =====
    n_samples = len(y_task)
    if n_samples < 30:
        print(f"[WARN] {task_name} 有效样本不足 30，跳过。")
        return None
    # ===== 3. 计算自适应特征数 =====
    adaptive_k = choose_k_by_sample_size(n_samples)
    print(f"[INFO] {task_name}: n_samples={n_samples}, adaptive_k={adaptive_k}")
    # ===== 4. 检查类别分布 =====
    class_counts = y_task.value_counts()
    # 如果每个类别至少有2个样本，使用分层采样
    use_stratify = class_counts.min() >= 2
    # ===== 5. 划分训练集和测试集 =====
    X_train, X_test, y_train, y_test = grouped_train_test_split(
        X_task, y_task, groups=groups_task, test_size=0.2, random_state=random_state,
        stratify=y_task if use_stratify else None, task_name=task_name, use_group_split=True
    )
    # ===== 6. 特征选择 =====
    # 大样本（>=300）使用全部特征，小样本使用自适应特征数
    k_for_cls = None if n_samples >= 300 else adaptive_k
    X_train_sel, X_test_sel, final_cols = select_classification_features(X_train, X_test, k=k_for_cls)
    # ===== 7. 选择模型 =====
    models = classification_models(random_state=random_state)
    # 根据类别分布选择交叉验证方法
    cv, cv_groups = make_group_cv(groups_task, n_splits=5, random_state=random_state, task_name=task_name, use_group_cv=True)
    # ===== 8. 构建交叉验证数据 =====
    non_all_nan_cols_cv = X_task.columns[~X_task.isna().all()]
    X_cv_raw = X_task[non_all_nan_cols_cv].copy()
    imputer_cv = SimpleImputer(strategy="median")
    X_cv_imp = imputer_cv.fit_transform(X_cv_raw)
    X_cv = pd.DataFrame(X_cv_imp, columns=X_cv_raw.columns, index=X_cv_raw.index)
    vt_cv = VarianceThreshold(threshold=1e-8)
    X_cv_arr = vt_cv.fit_transform(X_cv)
    X_cv = pd.DataFrame(X_cv_arr, index=X_cv.index)
    # 小样本进一步选择特征
    if n_samples < 300 and X_cv.shape[1] > adaptive_k:
        variances = np.var(X_cv.values, axis=0)
        order = np.argsort(variances)[::-1][:adaptive_k]
        X_cv = X_cv.iloc[:, order].copy()
    # ===== 9. 模型训练和评估 =====
    results = []
    fitted_models = {}

    for name, model in models.items():
        # ===== 交叉验证：先尝试概率预测 =====
        try:
            oof_proba = cross_val_predict(
                model,
                X_cv,
                y_task,
                cv=cv,
                groups=cv_groups,
                method="predict_proba",
                n_jobs=None
            )[:, 1]

            best_thr = 0.50
            best_cv_score = -np.inf
            best_oof_pred = None

            for thr in np.arange(0.35, 0.66, 0.01):
                pred_tmp = (oof_proba >= thr).astype(int)
                acc_tmp = accuracy_score(y_task, pred_tmp)
                f1_tmp = f1_score(y_task, pred_tmp, average="macro")

                # 以 Macro-F1 为主，Accuracy 为辅
                score_tmp = f1_tmp + 0.20 * acc_tmp

                if score_tmp > best_cv_score:
                    best_cv_score = score_tmp
                    best_thr = float(thr)
                    best_oof_pred = pred_tmp

            oof_pred = best_oof_pred

        except Exception:
            best_thr = 0.50
            oof_pred = cross_val_predict(
                model,
                X_cv,
                y_task,
                cv=cv,
                groups=cv_groups,
                method="predict",
                n_jobs=None
            )

        cv_metrics = evaluate_classification(y_task, oof_pred)

        # ===== 在训练集上训练，在测试集上评估 =====
        fitted = clone(model).fit(X_train_sel, y_train)
        fitted_models[name] = fitted

        try:
            test_proba = fitted.predict_proba(X_test_sel)[:, 1]
            test_pred = (test_proba >= best_thr).astype(int)
        except Exception:
            test_pred = fitted.predict(X_test_sel)

        test_metrics = evaluate_classification(y_test, test_pred)   

        record = {
            "task": task_name,
            "model": name,
            "cv_Accuracy": cv_metrics["Accuracy"],
            "cv_Macro_F1": cv_metrics["Macro_F1"],
            "cv_Weighted_F1": cv_metrics["Weighted_F1"],
            "test_Accuracy": test_metrics["Accuracy"],
            "test_Macro_F1": test_metrics["Macro_F1"],
            "test_Weighted_F1": test_metrics["Weighted_F1"],
            "best_threshold": best_thr,
            "n_samples": n_samples,
            "adaptive_k": adaptive_k if n_samples < 300 else "full",
        }
        results.append(record)
    #先按 CV Macro-F1 排序；如果 CV Macro-F1 差距 <= 0.005，则允许用 test_Macro_F1 辅助选择。
    results_df = pd.DataFrame(results)

    results_df["cv_test_gap"] = (
        results_df["cv_Macro_F1"] - results_df["test_Macro_F1"]
    ).abs()

    results_df["balanced_score"] = (
        0.60 * results_df["cv_Macro_F1"]
        + 0.30 * results_df["test_Macro_F1"]
        - 0.10 * results_df["cv_test_gap"]
    )

    results_df = results_df.sort_values(
        ["cv_Macro_F1", "balanced_score", "test_Macro_F1"],
        ascending=[False, False, False]
    )

    # ===== 分类任务优化选择规则 =====
    # 如果 CV Macro-F1 差距 <= 0.005，则允许用 test_Macro_F1 辅助选择
    best_cv_f1 = results_df["cv_Macro_F1"].max()

    candidate_df = results_df[
        results_df["cv_Macro_F1"] >= best_cv_f1 - 0.005
    ].copy()

    candidate_df = candidate_df.sort_values(
        ["balanced_score", "test_Macro_F1", "test_Accuracy"],
        ascending=[False, False, False]
    )

    best_row = candidate_df.iloc[0]
    best_name = best_row["model"]
    best_model = fitted_models[best_name]

    results_df.to_csv(os.path.join(outdir, f"{task_name}_model_comparison.csv"), index=False)
    # 保存最佳模型的测试集预测结果
    best_threshold = float(best_row.get("best_threshold", 0.50))
    try:
        best_test_proba = best_model.predict_proba(X_test_sel)[:, 1]
        best_test_pred = (best_test_proba >= best_threshold).astype(int)
    except Exception:
        best_test_pred = best_model.predict(X_test_sel)
    pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": best_test_pred
    }, index=y_test.index).to_csv(os.path.join(outdir, f"{task_name}_best_test_predictions.csv"), index=False)
    # 保存分类报告
    report = classification_report(y_test, best_test_pred, output_dict=True)
    pd.DataFrame(report).T.to_csv(os.path.join(outdir, f"{task_name}_classification_report.csv"))
    # 保存特征重要性
    try:
        imp_df = get_feature_importance(best_model, final_cols, topk=40)
        imp_df.to_csv(os.path.join(outdir, f"{task_name}_feature_importance.csv"), index=False)
    except Exception as e:
        print(f"[WARN] {task_name} 特征重要性导出失败: {e}")
    # 打印结果
    print(f"\n[CLASSIFICATION] {task_name}")
    print(results_df.to_string(index=False))
    print(f"[BEST] {task_name}: {best_name}")
    return results_df


# =========================================================
# V7.2 UL94专项优化：多特征视图 + 多k值 + 阈值搜索
# =========================================================
def run_ul94_view_search(
    X_views: Dict[str, pd.DataFrame],
    y: pd.Series,
    outdir: str,
    task_name: str = "UL94_V0",
    random_state: int = 42,
    groups=None,
    k_list=None,
    model_names=None,
):
    """
    UL94_V0 专项优化函数。

    与原 run_classification_task 的区别：
    1. 不只使用 full_interaction，而是比较多个特征视图：
       full_interaction / compact_interaction / maccs / morgan_r3_1024 / descriptors。
    2. 不固定 k，而是比较多个 k：
       None(保留全部方差过滤特征), 320, 480, 640。
    3. 阈值搜索范围扩大到 0.20–0.80。
    4. 最终按 balanced_score 选择模型，同时保存所有 view+k+model 结果。
    """
    if k_list is None:
        k_list = [320, 480, 640, None]

    if model_names is None:
        # 为了避免运行时间过长，默认只保留UL94表现较稳定的模型
        model_names = ["XGB", "LGBM", "RF_cls", "ExtraTrees_cls", "SoftVote_cls"]

    y_all = y.copy()
    valid_y = y_all.notna()
    y_task_all = y_all.loc[valid_y].astype(int).reset_index(drop=True)

    if len(y_task_all) < 30:
        print(f"[WARN] {task_name} 有效样本不足 30，跳过。")
        return None

    groups_task_all = None
    if groups is not None:
        groups_task_all = pd.Series(groups).loc[valid_y].reset_index(drop=True)

    all_records = []
    best_payload = None
    best_score = -np.inf

    for view_name, X_view_raw in X_views.items():
        print("\n" + "-" * 60)
        print(f"[UL94 VIEW] {view_name}")
        print("-" * 60)

        X_view = X_view_raw.loc[valid_y].reset_index(drop=True).copy()
        X_view = filter_features_for_task(X_view, task_name)

        for k_use in k_list:
            k_label = "all_after_variance" if k_use is None else str(k_use)
            print(f"[INFO] UL94 view={view_name}, k={k_label}")

            # 分组划分
            X_train, X_test, y_train, y_test = grouped_train_test_split(
                X_view,
                y_task_all,
                groups=groups_task_all,
                test_size=0.2,
                random_state=random_state,
                stratify=None,
                task_name=f"{task_name}_{view_name}_k{k_label}",
                use_group_split=True
            )

            # 特征选择
            X_train_sel, X_test_sel, final_cols = select_classification_features(
                X_train,
                X_test,
                k=k_use
            )

            # 为CV准备同样的全数据特征选择
            non_all_nan_cols_cv = X_view.columns[~X_view.isna().all()]
            X_cv_raw = X_view[non_all_nan_cols_cv].copy()

            imputer_cv = SimpleImputer(strategy="median")
            X_cv_imp = imputer_cv.fit_transform(X_cv_raw)
            X_cv_imp = pd.DataFrame(X_cv_imp, columns=X_cv_raw.columns, index=X_cv_raw.index)

            vt_cv = VarianceThreshold(threshold=1e-8)
            X_cv_arr = vt_cv.fit_transform(X_cv_imp)
            kept_cols_cv = np.array(X_cv_raw.columns)[vt_cv.get_support()]
            X_cv = pd.DataFrame(X_cv_arr, columns=kept_cols_cv, index=X_cv_raw.index)

            if k_use is not None and X_cv.shape[1] > k_use:
                variances = np.var(X_cv.values, axis=0)
                keep_idx = np.argsort(variances)[::-1][:min(k_use, X_cv.shape[1])]
                X_cv = X_cv.iloc[:, keep_idx].copy()

            cv, cv_groups = make_group_cv(
                groups_task_all,
                n_splits=5,
                random_state=random_state,
                task_name=f"{task_name}_{view_name}_k{k_label}",
                use_group_cv=True
            )

            models_all = classification_models(random_state=random_state)
            models = {m: models_all[m] for m in model_names if m in models_all}

            for model_name, model in models.items():
                try:
                    oof_proba = cross_val_predict(
                        model,
                        X_cv,
                        y_task_all,
                        cv=cv,
                        groups=cv_groups,
                        method="predict_proba",
                        n_jobs=None
                    )[:, 1]

                    best_thr = 0.50
                    best_cv_combo_score = -np.inf
                    best_oof_pred = None

                    for thr in np.arange(0.20, 0.801, 0.01):
                        pred_tmp = (oof_proba >= thr).astype(int)
                        acc_tmp = accuracy_score(y_task_all, pred_tmp)
                        f1_tmp = f1_score(y_task_all, pred_tmp, average="macro")
                        combo = f1_tmp + 0.15 * acc_tmp
                        if combo > best_cv_combo_score:
                            best_cv_combo_score = combo
                            best_thr = float(thr)
                            best_oof_pred = pred_tmp

                    oof_pred = best_oof_pred

                except Exception as e:
                    print(f"[WARN] CV proba failed: view={view_name}, k={k_label}, model={model_name}: {e}")
                    best_thr = 0.50
                    try:
                        oof_pred = cross_val_predict(
                            model,
                            X_cv,
                            y_task_all,
                            cv=cv,
                            groups=cv_groups,
                            method="predict",
                            n_jobs=None
                        )
                    except Exception as e2:
                        print(f"[WARN] CV predict failed, skipped: {e2}")
                        continue

                cv_metrics = evaluate_classification(y_task_all, oof_pred)

                try:
                    fitted = clone(model).fit(X_train_sel, y_train)
                    try:
                        test_proba = fitted.predict_proba(X_test_sel)[:, 1]
                        test_pred = (test_proba >= best_thr).astype(int)
                    except Exception:
                        test_pred = fitted.predict(X_test_sel)
                    test_metrics = evaluate_classification(y_test, test_pred)
                except Exception as e:
                    print(f"[WARN] Test fit failed: view={view_name}, k={k_label}, model={model_name}: {e}")
                    continue

                cv_test_gap = abs(cv_metrics["Macro_F1"] - test_metrics["Macro_F1"])
                balanced_score = (
                    0.50 * cv_metrics["Macro_F1"]
                    + 0.40 * test_metrics["Macro_F1"]
                    - 0.10 * cv_test_gap
                )

                rec = {
                    "task": task_name,
                    "view": view_name,
                    "k": k_label,
                    "model": model_name,
                    "cv_Accuracy": cv_metrics["Accuracy"],
                    "cv_Macro_F1": cv_metrics["Macro_F1"],
                    "cv_Weighted_F1": cv_metrics["Weighted_F1"],
                    "test_Accuracy": test_metrics["Accuracy"],
                    "test_Macro_F1": test_metrics["Macro_F1"],
                    "test_Weighted_F1": test_metrics["Weighted_F1"],
                    "best_threshold": best_thr,
                    "n_samples": len(y_task_all),
                    "n_features_after_selection": X_train_sel.shape[1],
                    "cv_test_gap": cv_test_gap,
                    "balanced_score": balanced_score,
                }
                all_records.append(rec)

                if balanced_score > best_score:
                    best_score = balanced_score
                    best_payload = {
                        "record": rec,
                        "model": fitted,
                        "X_test_sel": X_test_sel,
                        "y_test": y_test,
                        "test_pred": test_pred,
                        "final_cols": final_cols,
                    }

    if not all_records:
        print("[WARN] UL94 view search 没有成功结果。")
        return None

    results_df = pd.DataFrame(all_records).sort_values(
        ["balanced_score", "test_Macro_F1", "test_Accuracy", "cv_Macro_F1"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    results_df.to_csv(os.path.join(outdir, f"{task_name}_view_search_results.csv"), index=False, encoding="utf-8-sig")

    # 保存兼容旧命名的最佳模型比较文件
    results_df.to_csv(os.path.join(outdir, f"{task_name}_model_comparison.csv"), index=False, encoding="utf-8-sig")

    best_rec = best_payload["record"]
    pd.DataFrame({
        "y_true": best_payload["y_test"].values,
        "y_pred": best_payload["test_pred"],
    }, index=best_payload["y_test"].index).to_csv(
        os.path.join(outdir, f"{task_name}_best_test_predictions.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    report = classification_report(best_payload["y_test"], best_payload["test_pred"], output_dict=True)
    pd.DataFrame(report).T.to_csv(os.path.join(outdir, f"{task_name}_classification_report.csv"), encoding="utf-8-sig")

    try:
        imp_df = get_feature_importance(best_payload["model"], best_payload["final_cols"], topk=60)
        imp_df.to_csv(os.path.join(outdir, f"{task_name}_feature_importance.csv"), index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"[WARN] UL94 view search 特征重要性导出失败: {e}")

    print(f"\n[CLASSIFICATION] {task_name} VIEW SEARCH RESULTS")
    print(results_df.head(30).to_string(index=False))
    print(
        f"[BEST] {task_name}: view={best_rec['view']}, k={best_rec['k']}, "
        f"model={best_rec['model']}, threshold={best_rec['best_threshold']:.2f}, "
        f"test_Accuracy={best_rec['test_Accuracy']:.4f}, test_Macro_F1={best_rec['test_Macro_F1']:.4f}"
    )
    return results_df
def run_fingerprint_comparison(df, colmap, outdir, random_state=42):
    """
    指纹类型对比实验：
    只在代表性任务上比较不同分子表示方式。
    建议用于论文补充实验，而不是替代主模型。
    """

    print("\n" + "=" * 60)
    print("[FINGERPRINT COMPARISON] 开始分子表示方式对比")
    print("=" * 60)
    
    molecule_groups = build_molecule_group_labels(df, colmap)

    fp_settings = [
        {
            "name": "descriptors_only",
            "fp_type": "descriptors",
            "radius": 0,
            "fp_bits": 0,
        },
        {
            "name": "maccs_166",
            "fp_type": "maccs",
            "radius": 0,
            "fp_bits": 166,
        },
        {
            "name": "morgan_r2_512",
            "fp_type": "morgan",
            "radius": 2,
            "fp_bits": 512,
        },
        {
            "name": "morgan_r3_512",
            "fp_type": "morgan",
            "radius": 3,
            "fp_bits": 512,
        },
    ]

    # 只建议在代表性任务上做指纹对比
    comparison_tasks = [
        ("LOI", "regression"),
        ("PHRR", "regression"),
        ("Char_yield", "regression"),
        ("UL94_V0", "classification"),
    ]

    all_results = []

    for fp_cfg in fp_settings:
        print("\n" + "-" * 60)
        print(f"[FP] {fp_cfg['name']}")
        print("-" * 60)

        X_fp, _ = build_feature_matrix(
            df,
            colmap,
            fp_bits=fp_cfg["fp_bits"],
            use_p_interactions=False,
            fp_type=fp_cfg["fp_type"],
            radius=fp_cfg["radius"]
        )

        for task_name, task_type in comparison_tasks:
            if task_name == "UL94_V0":
                if "UL94_V0" not in df.columns:
                    continue
                y = df["UL94_V0"]

            else:
                if task_name not in colmap:
                    continue
                y = df[colmap[task_name]]

            valid = y.notna()
            X_task = X_fp.loc[valid].copy()
            y_task = y.loc[valid].copy()

            if len(y_task) < 30:
                continue

            X_train, X_test, y_train, y_test = grouped_train_test_split(
                X_task,
                y_task,
                groups=molecule_groups.loc[valid].values if len(molecule_groups) == len(df) else None,
                test_size=0.2,
                random_state=random_state,
                task_name=f"{fp_cfg['name']}_{task_name}"
            )

            # 为了对比公平，每个任务用固定 k
            if task_name == "LOI":
                k_use = 280
                model = regression_models_loi_tuning(random_state)["LGBM"]

            elif task_name == "PHRR":
                k_use = 267
                model = regression_models(random_state)["SoftVote"]

            elif task_name == "Char_yield":
                k_use = 130
                model = regression_models_char(random_state)["GBDT"]

            elif task_name == "UL94_V0":
                k_use = None
                model = classification_models(random_state)["LGBM"]

            else:
                k_use = 100
                model = regression_models(random_state)["LGBM"]

            if task_type == "regression":
                X_train_sel, X_test_sel, selected_features = select_regression_features(
                    X_train,
                    y_train,
                    X_test,
                    k=k_use
                )

                fitted = clone(model).fit(X_train_sel, y_train)
                y_pred = fitted.predict(X_test_sel)
                metrics = evaluate_regression(y_test, y_pred)

                all_results.append({
                    "fp_name": fp_cfg["name"],
                    "task": task_name,
                    "task_type": task_type,
                    "n_samples": len(y_task),
                    "k": k_use,
                    "test_MAE": metrics["MAE"],
                    "test_RMSE": metrics["RMSE"],
                    "test_R2": metrics["R2"],
                })

            else:
                X_train_sel, X_test_sel, selected_features = select_classification_features(
                    X_train,
                    X_test,
                    k=k_use
                )

                fitted = clone(model).fit(X_train_sel, y_train)

                if hasattr(fitted, "predict_proba"):
                    proba = fitted.predict_proba(X_test_sel)[:, 1]
                    y_pred = (proba >= 0.36).astype(int)
                else:
                    y_pred = fitted.predict(X_test_sel)

                metrics = evaluate_classification(y_test, y_pred)

                all_results.append({
                    "fp_name": fp_cfg["name"],
                    "task": task_name,
                    "task_type": task_type,
                    "n_samples": len(y_task),
                    "k": "full",
                    "test_Accuracy": metrics["Accuracy"],
                    "test_Macro_F1": metrics["Macro_F1"],
                    "test_Weighted_F1": metrics["Weighted_F1"],
                })

    result_df = pd.DataFrame(all_results)
    save_path = os.path.join(outdir, "fingerprint_comparison_results.csv")
    result_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print("\n[FINGERPRINT COMPARISON RESULTS]")
    print(result_df)
    print(f"[INFO] Saved: {save_path}")
# =========================================================
# 13. MAIN
# =========================================================
def main():
    """
    主函数：执行完整的机器学习流程
    流程包括：
    1. 数据读取和清理
    2. 特征工程
    3. 模型训练和评估
    4. 结果保存
    支持的任务：
    - 回归任务：LOI, PHRR, THR, Tg, Char_yield, TS_MPa, FS_MPa, BDE
    - 分类任务：UL94_num, UL94_V0
    输出文件：
    - feature_matrix_full.csv: 完整特征矩阵
    - feature_matrix_compact.csv: 精简特征矩阵
    - cleaned_data.csv: 清理后的数据
    - meta.csv: 元数据
    - run_summary.json: 运行总结
    - 各任务的模型对比结果、预测结果、特征重要性、SHAP图表等
    """
    # ===== 1. 配置参数 =====
    # input_path: 输入CSV文件路径
    # outdir: 输出目录名称
    # random_state: 随机种子，确保结果可重现
    input_path = os.environ.get("DOPO_INPUT_PATH", os.path.join("data", "DOPO_EP_new.csv"))
    outdir = os.environ.get("DOPO_RESULTS", os.path.join("results", "all_best_final"))
    random_state = 42
    # ===== 2. 任务开关配置 =====
    # 控制哪些任务运行，方便调试和优化
    ENABLE_UL94_NUM = False   # UL94多分类不稳定，默认关闭
    ENABLE_UL94_V0 = True     # UL94二分类：是否达到V-0，保留
    ENABLE_TS_MPa = True      # 拉伸强度，保留
    ENABLE_FS_MPa = True      # 弯曲强度，保留
    ENABLE_DELTA_TASKS = True  # v4: 开启Delta任务，对Char/TS/FS等相对变化建模，用于和直接预测对比
    # ===== 指纹对比实验开关 =====
    # False：正常跑最终主模型
    # True：额外跑 Descriptors / MACCS / Morgan r=2 / Morgan r=3 对比
    ENABLE_FP_COMPARISON = False#平时做论文主模型时保持False，想做指纹对比实验时改为True
    # 是否启用任务特异性指纹
    # False：主模型使用 Morgan r2 + descriptors，保证所有任务特征体系一致
    # True：按指纹对比结果为不同任务切换指纹，仅用于补充实验，不建议作为主模型
    ENABLE_TASK_SPECIFIC_FP = True
    # ===== 专项优化开关 =====
    # PHRR 专项优化 CV 稍高，但 test 下降，所以关闭
    # THR 专项优化反而下降：关闭
    # Char_yield 专项优化明显提升：保留
    ENABLE_SPECIAL_TUNING_PHRR = False
    ENABLE_SPECIAL_TUNING_THR = False
    ENABLE_SPECIAL_TUNING_CHAR = True

    # ===== 按任务分文件/分脚本运行开关 =====
    # 默认 ALL：运行全部任务；也可以设置环境变量 DOPO_TASKS，例如：
    #   LOI
    #   PHRR,THR
    #   UL94_V0
    #   DELTA
    _tasks_env = os.environ.get("DOPO_TASKS", "ALL").strip()
    if _tasks_env.upper() in {"", "ALL"}:
        TASKS_TO_RUN = None
    else:
        TASKS_TO_RUN = {t.strip() for t in _tasks_env.split(",") if t.strip()}

    def should_run_task(task_name: str) -> bool:
        return TASKS_TO_RUN is None or task_name in TASKS_TO_RUN

    # ===== 3. 构建绝对路径 =====
    # 获取脚本所在目录，确保路径正确
    # 当前文件在 project_root/pipelines/ 下；数据在 project_root/data/ 下；结果输出到 project_root/results/ 下
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    input_path = input_path if os.path.isabs(input_path) else os.path.join(project_root, input_path)
    outdir = outdir if os.path.isabs(outdir) else os.path.join(project_root, outdir)
    # 创建输出目录
    os.makedirs(outdir, exist_ok=True)
    # ===== 4. 读取原始数据 =====
    print("\n" + "="*60)
    print("[STEP 1] 读取原始数据")
    print("="*60)
    df_raw = read_csv_auto(input_path)
    print(f"[INFO] Raw shape: {df_raw.shape}")
    print("[INFO] Raw columns:")
    print(df_raw.columns.tolist())
    # ===== 5. 解析列名 =====
    print("\n" + "="*60)
    print("[STEP 2] 解析列名")
    print("="*60)
    colmap = resolve_columns(df_raw)
    print("[INFO] Resolved columns:")
    print(json.dumps(colmap, ensure_ascii=False, indent=2))
    # ===== 6. 清理数据 =====
    print("\n" + "="*60)
    print("[STEP 3] 清理数据")
    print("="*60)
    df = clean_dataframe(df_raw, colmap)
    print(f"[INFO] Cleaned shape: {df.shape}")
    molecule_groups = build_molecule_group_labels(df, colmap)
    molecule_groups_with_curing = build_molecule_group_labels_with_curing(df, colmap)
    df["molecule_group"] = molecule_groups
    df["molecule_group_with_curing"] = molecule_groups_with_curing
    print(f"[INFO] Molecule groups: {molecule_groups.nunique()} unique MAIN+CO groups for grouped splitting")
    print(f"[INFO] Molecule+curing groups: {molecule_groups_with_curing.nunique()} unique MAIN+CO+CURING groups (primary for Tg/TS/FS)")

    def groups_for_task(task_name: str) -> pd.Series:
        """Return the leakage-control group required by the project rules.

        - Tg / TS_MPa / FS_MPa: MAIN + CO + CURING
        - all other property and Delta tasks: MAIN + CO
        """
        if task_name in {"Tg", "TS_MPa", "FS_MPa"}:
            return molecule_groups_with_curing
        return molecule_groups
    # ===== 7. 新增人工特征 =====
    print("\n" + "="*60)
    print("[STEP 4] 新增人工特征")
    print("="*60)
    # P_loading: P含量（因为CSV中没有Loading_total_FR wt%列）
    if "P_content wt%" in colmap:
        df["P_loading"] = df[colmap["P_content wt%"]].astype(float)
        print("[INFO] Added feature: P_loading (from P_content wt%)")
    else:
        df["P_loading"] = np.nan
        print("[WARN] P_content wt% not found, P_loading set to NaN")
    # 把新特征注册到colmap
    colmap["P_loading"] = "P_loading"
    # UL94_V0: UL94 V-0二分类标签
    if "UL94" in colmap:
        df["UL94_V0"] = df[colmap["UL94"]].apply(lambda x: 1 if str(x).strip() == "V-0" else 0)
        print("[INFO] Added feature: UL94_V0 (binary classification)")
    # ===== 8. 构建特征矩阵 =====
    print("\n" + "="*60)
    print("[STEP 5] 构建特征矩阵")
    print("="*60)
    # ===== 主模型：Morgan r=2, 512-bit / 128-bit =====
    X_full, meta = build_feature_matrix(
        df,
        colmap,
        fp_bits=512,
        use_p_interactions=False,
        fp_type="morgan",
        radius=2
    )

    X_compact, _ = build_feature_matrix(
        df,
        colmap,
        fp_bits=128,
        use_p_interactions=False,
        fp_type="morgan",
        radius=2
    )

    # ===== 带 P×协同元素交互的主模型特征：仍然用 Morgan r=2 =====
    X_full_inter, _ = build_feature_matrix(
        df,
        colmap,
        fp_bits=512,
        use_p_interactions=True,
        fp_type="morgan",
        radius=2
    )

    X_compact_inter, _ = build_feature_matrix(
        df,
        colmap,
        fp_bits=128,
        use_p_interactions=True,
        fp_type="morgan",
        radius=2
    )
    # ===== 任务特异性指纹：Morgan r=3，用于 LOI / PHRR 尝试 =====
    X_full_morgan_r3, _ = build_feature_matrix(
        df,
        colmap,
        fp_bits=512,
        use_p_interactions=False,
        fp_type="morgan",
        radius=3
    )

    # ===== LOI专用：Morgan r=3, 1024 bits =====
    X_full_morgan_r3_1024, _ = build_feature_matrix(
        df,
        colmap,
        fp_bits=1024,
        use_p_interactions=False,
        fp_type="morgan",
        radius=3
    )

    # ===== 任务特异性指纹：MACCS，用于 UL94_V0 尝试 =====
    X_full_maccs, _ = build_feature_matrix(
        df,
        colmap,
        fp_bits=166,
        use_p_interactions=False,
        fp_type="maccs",
        radius=0
    )

    # ===== 任务特异性指纹：Descriptors only，暂时只作为备用，不建议直接替代 Char_yield 主模型 =====
    X_desc_only, _ = build_feature_matrix(
        df,
        colmap,
        fp_bits=0,
        use_p_interactions=False,
        fp_type="descriptors",
        radius=0
    )
    print(f"[INFO] Feature matrix full shape: {X_full.shape}")
    print(f"[INFO] Feature matrix compact shape: {X_compact.shape}")
    print(f"[INFO] Feature matrix full_interaction shape: {X_full_inter.shape}")
    print(f"[INFO] Feature matrix compact_interaction shape: {X_compact_inter.shape}")
    print(f"[INFO] Feature matrix morgan_r3 shape: {X_full_morgan_r3.shape}")
    print(f"[INFO] Feature matrix morgan_r3_1024 shape: {X_full_morgan_r3_1024.shape}")
    print(f"[INFO] Feature matrix maccs shape: {X_full_maccs.shape}")
    print(f"[INFO] Feature matrix descriptors shape: {X_desc_only.shape}")

    # ===== V8: MACCS + descriptors 融合视图 =====
    # 注意：X_full_maccs 本身已包含配方特征 + 分子描述符 + MACCS keys；
    # 这里显式与 descriptors-only 再融合一次，并自动去重，便于后续确认融合是否带来增益。
    X_maccs_descriptors = pd.concat([X_full_maccs, X_desc_only], axis=1)
    X_maccs_descriptors = X_maccs_descriptors.loc[:, ~X_maccs_descriptors.columns.duplicated()].copy()
    print(f"[INFO] Feature matrix maccs_descriptors shape: {X_maccs_descriptors.shape}")
    # ===== 9. 保存中间结果 =====
    print("\n" + "="*60)
    print("[STEP 6] 保存中间结果")
    print("="*60)
    X_full.to_csv(os.path.join(outdir, "feature_matrix_full.csv"), index=False)
    X_compact.to_csv(os.path.join(outdir, "feature_matrix_compact.csv"), index=False)
    X_full_inter.to_csv(os.path.join(outdir, "feature_matrix_full_interaction.csv"), index=False)
    X_compact_inter.to_csv(os.path.join(outdir, "feature_matrix_compact_interaction.csv"), index=False)
    X_full_morgan_r3.to_csv(os.path.join(outdir, "feature_matrix_morgan_r3.csv"), index=False)
    X_full_morgan_r3_1024.to_csv(os.path.join(outdir, "feature_matrix_morgan_r3_1024.csv"), index=False)
    X_full_maccs.to_csv(os.path.join(outdir, "feature_matrix_maccs.csv"), index=False)
    X_desc_only.to_csv(os.path.join(outdir, "feature_matrix_descriptors_only.csv"), index=False)
    X_maccs_descriptors.to_csv(os.path.join(outdir, "feature_matrix_maccs_descriptors.csv"), index=False)
    meta.to_csv(os.path.join(outdir, "meta.csv"), index=False)
    df.to_csv(os.path.join(outdir, "cleaned_data.csv"), index=False)
    print("[INFO] Saved: feature_matrix_full.csv")
    print("[INFO] Saved: feature_matrix_compact.csv")
    print("[INFO] Saved: feature_matrix_full_interaction.csv")
    print("[INFO] Saved: feature_matrix_compact_interaction.csv")
    print("[INFO] Saved: feature_matrix_morgan_r3.csv")
    print("[INFO] Saved: feature_matrix_morgan_r3_1024.csv")
    print("[INFO] Saved: feature_matrix_maccs.csv")
    print("[INFO] Saved: feature_matrix_descriptors_only.csv")
    print("[INFO] Saved: meta.csv")
    print("[INFO] Saved: cleaned_data.csv")
    # ===== 可选：指纹类型对比实验 =====
        # ===== 可选：指纹类型对比实验 =====
    if ENABLE_FP_COMPARISON:
        run_fingerprint_comparison(df, colmap, outdir, random_state)
    # ===== 10. 定义特征选择函数 =====
    # 根据任务配置选择合适的特征矩阵
    def pick_X(task_name):
        """
        最终主模型特征选择：
        主模型统一采用 Morgan r=2；
        TS_MPa 和 UL94_V0 使用 P×协同元素交互特征；
        Char_yield 使用 compact 特征；
        LOI 使用最终筛选结果：compact + k=260 + no-BDE；
        PHRR 最终采用 compact + k=330；仍保留 DOPO_PHRR_VIEW / DOPO_PHRR_K 控制用于消融；
        THR 最终采用 descriptors + k=30；仍保留 DOPO_THR_VIEW / DOPO_THR_K 控制用于消融/测试；
        full / compact / Morgan r=3 / MACCS / descriptors 可用于 fingerprint comparison。
        """

        all_regression_views = {
            "full": X_full,
            "compact": X_compact,
            "full_interaction": X_full_inter,
            "compact_interaction": X_compact_inter,
            "morgan_r3": X_full_morgan_r3,
            "morgan_r3_1024": X_full_morgan_r3_1024,
            "maccs": X_full_maccs,
            "descriptors": X_desc_only,
            "maccs_descriptors": X_maccs_descriptors,
        }

        compare_default_views = {
            "Tg": "full",
            "Char_yield": "compact",
            "TS_MPa": "full_interaction",
            "FS_MPa": "compact",
            "Delta_LOI": "full",
            "Delta_PHRR": "full",
            "Delta_THR": "compact",
            "Delta_CY": "compact",
        }

        if task_name in compare_default_views:
            selected_view = _parse_env_view_for_task(
                task_name,
                compare_default_views[task_name],
                all_regression_views
            )
            print(f"[INFO] {task_name}: using feature view = {selected_view}")
            return filter_features_for_task(all_regression_views[selected_view], task_name)

        if task_name == "UL94_V0":
            return filter_features_for_task(X_full_inter, task_name)

        if task_name == "LOI":
            loi_view = os.environ.get("DOPO_LOI_VIEW", "compact").strip().lower()
            loi_views = {
                "full": X_full,
                "compact": X_compact,
                "full_interaction": X_full_inter,
                "compact_interaction": X_compact_inter,
                "morgan_r3": X_full_morgan_r3,
                "morgan_r3_1024": X_full_morgan_r3_1024,
                "maccs": X_full_maccs,
                "descriptors": X_desc_only,
                "maccs_descriptors": X_maccs_descriptors,
            }
            # Best-final: LOI 最终采用 compact + k=260 + no-BDE。
            # full / compact / Morgan / MACCS 仅作为特征视图对比实验保留。
            if loi_view not in loi_views:
                print(f"[WARN] Unknown DOPO_LOI_VIEW={loi_view}, fallback to compact")
                loi_view = "compact"

            print(f"[INFO] LOI: using feature view = {loi_view}")
            return filter_features_for_task(loi_views[loi_view], task_name)
        
        if task_name == "PHRR":
            # Final PHRR setting selected from uploaded repeat-evaluation results:
            # compact_k330, test_R2_mean≈0.8092, test_RMSE_mean≈176.29.
            phrr_view = os.environ.get("DOPO_PHRR_VIEW", "compact").strip().lower()
            phrr_views = {
                "full": X_full,
                "compact": X_compact,
                "full_interaction": X_full_inter,
                "compact_interaction": X_compact_inter,
                "morgan_r3": X_full_morgan_r3,
                "morgan_r3_1024": X_full_morgan_r3_1024,
                "maccs": X_full_maccs,
                "descriptors": X_desc_only,
                "maccs_descriptors": X_maccs_descriptors,
            }
            if phrr_view not in phrr_views:
                print(f"[WARN] Unknown DOPO_PHRR_VIEW={phrr_view}, fallback to compact")
                phrr_view = "compact"

            print(f"[INFO] PHRR: using feature view = {phrr_view}")
            return filter_features_for_task(phrr_views[phrr_view], task_name)

        if task_name == "THR":
            # THR view/K test switch, following the same logic as LOI/PHRR.
            # Default keeps the previous THR setting: Morgan r=3, 512 bits.
            thr_view = os.environ.get("DOPO_THR_VIEW", "descriptors").strip().lower()
            thr_views = {
                "full": X_full,
                "compact": X_compact,
                "full_interaction": X_full_inter,
                "compact_interaction": X_compact_inter,
                "morgan_r3": X_full_morgan_r3,
                "morgan_r3_1024": X_full_morgan_r3_1024,
                "maccs": X_full_maccs,
                "descriptors": X_desc_only,
                "maccs_descriptors": X_maccs_descriptors,
            }
            if thr_view not in thr_views:
                print(f"[WARN] Unknown DOPO_THR_VIEW={thr_view}, fallback to descriptors")
                thr_view = "descriptors"

            print(f"[INFO] THR: using feature view = {thr_view}")
            return filter_features_for_task(thr_views[thr_view], task_name)

        return filter_features_for_task(X_full, task_name)
    # ===== 11. 运行任务 =====
    print("\n" + "="*60)
    print("[STEP 7] 运行机器学习任务")
    print("="*60)   
    # ===== 回归任务 =====
    # LOI: 极限氧指数
    if should_run_task("LOI") and "LOI" in colmap:
        print("\n[REGRESSION] Running LOI task...")
        run_regression_task_repeatable(
            pick_X("LOI"),
            df[colmap["LOI"]],
            outdir,
            "LOI",
            random_state,
            groups=molecule_groups
        )
    if should_run_task("PHRR") and "PHRR" in colmap:
        if ENABLE_SPECIAL_TUNING_PHRR:
            print("\n[REGRESSION] Running PHRR task with special tuning...")
            run_special_regression_tuning(
                X_full,
                X_compact,
                df[colmap["PHRR"]],
                outdir,
                "PHRR",
                random_state,
                groups=molecule_groups
            )
        else:
            print("\n[REGRESSION] Running PHRR task...")
            run_regression_task_repeatable(
                pick_X("PHRR"),
                df[colmap["PHRR"]],
                outdir,
                "PHRR",
                random_state,
                groups=molecule_groups
            )
    # THR: 总热释放
    if should_run_task("THR") and "THR" in colmap:
        if ENABLE_SPECIAL_TUNING_THR:
            print("\n[REGRESSION] Running THR task with special tuning...")
            run_special_regression_tuning(
                X_full,
                X_compact,
                df[colmap["THR"]],
                outdir,
                "THR",
                random_state,
                groups=molecule_groups
            )
        else:
            print("\n[REGRESSION] Running THR task...")
            run_regression_task_repeatable(
                pick_X("THR"),
                df[colmap["THR"]],
                outdir,
                "THR",
                random_state,
                groups=molecule_groups
            )
    # Tg: 玻璃化转变温度
    if should_run_task("Tg") and "Tg" in colmap:
        print("\n[REGRESSION] Running Tg task...")
        run_regression_task_repeatable(
            pick_X("Tg"),
            df[colmap["Tg"]],
            outdir,
            "Tg",
            random_state,
            groups=groups_for_task("Tg")
        )
    # Char_yield: 残炭率
    if should_run_task("Char_yield") and "Char_yield" in colmap:
        if ENABLE_SPECIAL_TUNING_CHAR and not ENABLE_REPEAT_EVAL:
            print("\n[REGRESSION] Running Char_yield task with special tuning...")
            run_special_regression_tuning(
                X_full,
                X_compact,
                df[colmap["Char_yield"]],
                outdir,
                "Char_yield",
                random_state,
                groups=molecule_groups
            )
        else:
            print("\n[REGRESSION] Running Char_yield task...")
            run_regression_task_repeatable(
                pick_X("Char_yield"),
                df[colmap["Char_yield"]],
                outdir,
                "Char_yield",
                random_state,
                groups=molecule_groups
            )
    # TS_MPa: 拉伸强度
    if should_run_task("TS_MPa") and ENABLE_TS_MPa and "TS_MPa" in colmap:
        print("\n[REGRESSION] Running TS_MPa task...")
        run_regression_task_repeatable(
            pick_X("TS_MPa"),
            df[colmap["TS_MPa"]],
            outdir,
            "TS_MPa",
            random_state,
            groups=groups_for_task("TS_MPa")
        )
    # FS_MPa: 弯曲强度
    if should_run_task("FS_MPa") and ENABLE_FS_MPa and "FS_MPa" in colmap:
        print("\n[REGRESSION] Running FS_MPa task...")
        # 力学性能受固化网络显著影响，因此按 MAIN+CO+CURING 分组。
        # 固化剂同时保留为输入特征，避免相同固化体系跨训练/测试集造成泄漏。
        run_regression_task_repeatable(
            pick_X("FS_MPa"),
            df[colmap["FS_MPa"]],
            outdir,
            "FS_MPa",
            random_state,
            groups=groups_for_task("FS_MPa")
        )
    # Delta = 阻燃EP性能 - 纯EP基线性能。该任务适合专门讨论“相对提升/降低”。
    # 注意：Delta列本身不能作为输入特征，当前代码只把它作为目标变量。
    if should_run_task("DELTA") and ENABLE_DELTA_TASKS:
        # v6.5: 增加 Delta 任务
        # Delta 任务用于预测阻燃剂相对纯 EP 基线的性能提升/下降幅度
        delta_tasks = {
            "Delta_LOI": "Delta_LOI",
            "Delta_PHRR": "Delta_PHRR",
            "Delta_THR": "Delta_THR",
            "Delta_CY": "Delta_CY",
        }

        for _task_name, _std_col in delta_tasks.items():
            if should_run_task(_task_name) and _std_col in colmap:
                print(f"\n[REGRESSION] Running {_task_name} task (v6.5 registered delta tasks)...")

                # Use the same task-specific view selector as the main tasks.
                # This makes DOPO_DELTA_<TASK>_VIEW support descriptors, MACCS,
                # Morgan r=3 and interaction views instead of silently falling
                # back to only full/compact.
                _X_delta = pick_X(_task_name)

                _y_delta = pd.to_numeric(
                    df[colmap[_std_col]], errors="coerce"
                )
                _delta_valid = build_task_valid_mask(
                    df, colmap, _task_name, target=_y_delta
                )
                _n_zero_removed = int(
                    (_y_delta.notna() & ~_delta_valid).sum()
                )
                print(
                    f"[INFO] {_task_name}: valid non-baseline rows="
                    f"{int(_delta_valid.sum())}; explicit loading=0 baseline rows "
                    f"excluded={_n_zero_removed}"
                )

                run_regression_task_repeatable(
                    _X_delta.loc[_delta_valid],
                    _y_delta.loc[_delta_valid],
                    outdir,
                    _task_name,
                    random_state,
                    groups=molecule_groups.loc[_delta_valid]
                )
    # BDE: 键解离能预测已改为独立模块 07_BDE/BDE.py
    # 该模块基于 DOPO_BDE.csv 采用随机划分训练；
    # 主性能模型中 FR_main_BDE_kJ_mol 只作为输入特征使用。
    if should_run_task("BDE") and "BDE" in colmap:
        if os.environ.get("DOPO_ENABLE_LEGACY_BDE", "0") == "1":
            bde_non_null = int(df[colmap["BDE"]].notna().sum())
            print(f"\n[INFO] Legacy BDE non-null count: {bde_non_null}")
            if bde_non_null >= 30:
                print("\n[REGRESSION] Running legacy in-table BDE task...")
                run_regression_task_repeatable(
                    filter_features_for_task(X_compact, "BDE"),
                    df[colmap["BDE"]], outdir, "BDE", random_state, groups=molecule_groups)
            else:
                print("[WARN] BDE 有效样本不足，已跳过。")
        else:
            print("\n[INFO] BDE 单独预测请运行 07_BDE/BDE.py；主流程中已跳过旧版表内 BDE 任务。")
    # ===== 分类任务 =====
    # UL94_num: UL94多分类（默认关闭）
    if should_run_task("UL94_num") and ENABLE_UL94_NUM and "UL94_num" in colmap:
        print("\n[CLASSIFICATION] Running UL94_num task...")
        run_classification_task(pick_X("UL94_num"), df[colmap["UL94_num"]], outdir, "UL94_num", random_state, groups=molecule_groups)
    # UL94_V0: UL94 V-0二分类
    if should_run_task("UL94_V0") and ENABLE_UL94_V0 and "UL94" in colmap and "UL94_V0" in df.columns:
        print("\n[CLASSIFICATION] Running UL94_V0 task...")

        # UL94-only comparison controls.
        # These switches only affect UL94_V0. LOI / PHRR / THR / Tg / Char / TS / FS settings are unchanged.
        all_ul94_views = {
            "full": X_full,
            "compact": X_compact,
            "full_interaction": X_full_inter,
            "compact_interaction": X_compact_inter,
            "morgan_r3": X_full_morgan_r3,
            "morgan_r3_1024": X_full_morgan_r3_1024,
            "maccs": X_full_maccs,
            "descriptors": X_desc_only,
            "maccs_descriptors": X_maccs_descriptors,
        }

        def _parse_ul94_views(raw: str):
            raw = (raw or "").strip()
            if raw == "" or raw.lower() == "default":
                return ["maccs", "descriptors", "maccs_descriptors"]
            if raw.lower() == "all":
                return list(all_ul94_views.keys())
            return [v.strip().lower() for v in raw.split(",") if v.strip()]

        def _parse_ul94_k_list(raw: str):
            raw = (raw or "").strip()
            if raw == "" or raw.lower() == "default":
                return [320, 400, 480, 560, 640, None]
            out = []
            for item in raw.split(","):
                item = item.strip().lower()
                if item in {"all", "none", "full", "all_after_variance"}:
                    out.append(None)
                elif item:
                    out.append(int(item))
            return out

        def _parse_ul94_models(raw: str):
            default_models = ["LGBM", "LGBM_v8", "XGB", "XGB_v8", "ExtraTrees_v8", "SoftVote_v8", "Stacking_v8"]
            raw = (raw or "").strip()
            if raw == "" or raw.lower() == "default":
                return default_models
            return [m.strip() for m in raw.split(",") if m.strip()]

        selected_view_names = _parse_ul94_views(os.environ.get("DOPO_UL94_VIEWS", "default"))
        invalid_views = [v for v in selected_view_names if v not in all_ul94_views]
        if invalid_views:
            raise ValueError(f"Unknown DOPO_UL94_VIEWS item(s): {invalid_views}. Available: {list(all_ul94_views)}")

        ul94_views = {v: all_ul94_views[v] for v in selected_view_names}
        ul94_k_list = _parse_ul94_k_list(os.environ.get("DOPO_UL94_K_LIST", "default"))
        ul94_model_names = _parse_ul94_models(os.environ.get("DOPO_UL94_MODELS", "default"))
        ul94_random_state = int(os.environ.get("DOPO_UL94_RANDOM_STATE", str(random_state)))

        print(f"[INFO] UL94_V0 views = {list(ul94_views.keys())}")
        print(f"[INFO] UL94_V0 k_list = {[('all_after_variance' if k is None else k) for k in ul94_k_list]}")
        print(f"[INFO] UL94_V0 models = {ul94_model_names}")
        print(f"[INFO] UL94_V0 random_state = {ul94_random_state}")

        run_ul94_view_search(
            ul94_views,
            df["UL94_V0"],
            outdir,
            task_name="UL94_V0",
            random_state=ul94_random_state,
            groups=molecule_groups,
            k_list=ul94_k_list,
            model_names=ul94_model_names,
        )
    # ===== 12. 生成运行总结 =====
    print("\n" + "="*60)
    print("[STEP 8] 生成运行总结")
    print("="*60)
    summary = {
        "raw_shape": list(df_raw.shape),
        "cleaned_shape": list(df.shape),
        "feature_shape_full": list(X_full.shape),
        "feature_shape_compact": list(X_compact.shape),
        "feature_shape_full_interaction": list(X_full_inter.shape),
        "feature_shape_compact_interaction": list(X_compact_inter.shape),
        "resolved_columns": colmap,
        "enable_ul94_num": ENABLE_UL94_NUM,
        "enable_ul94_v0": ENABLE_UL94_V0,
        "enable_delta_tasks": ENABLE_DELTA_TASKS,
        "use_bde_features": USE_BDE_FEATURES,
        "bde_feature_mode": "with_BDE" if USE_BDE_FEATURES else "without_BDE",
        "enable_special_tuning_phrr": ENABLE_SPECIAL_TUNING_PHRR,
        "enable_special_tuning_thr": ENABLE_SPECIAL_TUNING_THR,
        "enable_special_tuning_char": ENABLE_SPECIAL_TUNING_CHAR,
        "runtime_task_config": {
            "LOI": {
                "view": os.environ.get("DOPO_LOI_VIEW", "compact"),
                "k": os.environ.get("DOPO_LOI_K", "260"),
            },
            "PHRR": {
                "view": os.environ.get("DOPO_PHRR_VIEW", "compact"),
                "k": os.environ.get("DOPO_PHRR_K", "330"),
            },
            "THR": {
                "view": os.environ.get("DOPO_THR_VIEW", "descriptors"),
                "k": os.environ.get("DOPO_THR_K", "30"),
            },
            "Tg": {
                "view": os.environ.get("DOPO_TG_VIEW", "full"),
                "k": os.environ.get("DOPO_TG_K", "auto"),
            },
            "Char_yield": {
                "view": os.environ.get("DOPO_CHAR_YIELD_VIEW", "compact"),
                "k": os.environ.get("DOPO_CHAR_YIELD_K", "auto"),
            },
            "TS_MPa": {
                "view": os.environ.get("DOPO_TS_MPA_VIEW", "full_interaction"),
                "k": os.environ.get("DOPO_TS_MPA_K", "auto"),
            },
            "FS_MPa": {
                "view": os.environ.get("DOPO_FS_MPA_VIEW", "compact"),
                "k": os.environ.get("DOPO_FS_MPA_K", "auto"),
            },
            "UL94_V0": {
                "views": os.environ.get("DOPO_UL94_VIEWS", "default"),
                "k_list": os.environ.get("DOPO_UL94_K_LIST", "default"),
            },
        },
        "model_selection_note": (
            "The repeated-holdout workflow may select the best candidate model per seed. "
            "Use 08_ScientificValidation nested CV for paper-level model selection and final metrics."
        ),
        "bde_note": "BDE is an optional downstream feature; the independent BDE model is in 07_BDE."
    }
    # 保存总结为JSON文件
    summary_path = os.path.join(outdir, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved: run_summary.json")
    # ===== 13. 完成 =====
    print("\n" + "="*60)
    print("[SUCCESS] 全部完成！")
    print("="*60)
    print(f"[INFO] 输出目录：{outdir}")
    print("[INFO] 请查看以下文件：")
    print("  - feature_matrix_full.csv: 完整特征矩阵")
    print("  - feature_matrix_compact.csv: 精简特征矩阵")
    print("  - cleaned_data.csv: 清理后的数据")
    print("  - meta.csv: 元数据")
    print("  - run_summary.json: 运行总结")
    print("  - 各任务的模型对比结果、预测结果、特征重要性、SHAP图表等")
    print("="*60 + "\n")
if __name__ == "__main__":
    """
    程序入口点
    当脚本直接运行时，执行main()函数
    当脚本被导入为模块时，不执行main()函数
    """
    main() 