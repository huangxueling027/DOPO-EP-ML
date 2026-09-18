# -*- coding: utf-8 -*-
"""Shared utilities for reproducible manuscript figures and supplementary tables.

The helpers support corrected V5 result-directory names with limited legacy compatibility. Every
selected input file is recorded in a manifest so that a figure can be traced
back to its source CSV.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]

# Global manuscript palette: restored to the V4 publication style.
# Figure 5 intentionally uses a separate local palette from the supplied reference board.
PALETTE = {
    # Updated global manuscript palette based on the user-supplied pastel board.
    # Layouts remain unchanged; only the overall colour language is harmonised.
    "red_dark": "#8A352E",
    "red": "#BA3E45",
    "coral": "#D69D98",
    "pink": "#F5B99E",
    "rose": "#FEE8DD",
    "blue_dark": "#3A4B6E",
    "blue": "#4E6691",
    "sky": "#9FBBD5",
    "teal": "#4C9AC9",
    "cyan": "#C8E0EF",
    "blue_light": "#E7F2F6",
    "purple_dark": "#4E6691",
    "purple": "#9FBBD5",
    "lavender": "#E9D8D2",
    "green_dark": "#8E8B52",
    "green": "#C9C780",
    "green_light": "#F5F5DC",
    "orange_dark": "#D05928",
    "orange": "#E2745E",
    "peach": "#FFE0D1",
    "gold": "#D05928",
    "gold_light": "#FFE0D1",
    "slate": "#A7AFB8",
    "gray": "#A7AFB8",
    "gray_light": "#E9D8D2",
    "dark": "#3A4B6E",
    "black": "#000000",
    "light_gray": "#F5F5DC",
}

# Backward-compatible semantic aliases used throughout the plotting scripts.
PALETTE["yellow"] = PALETTE["gold"]

TASK_COLORS = {
    "LOI": PALETTE["blue"],
    "PHRR": PALETTE["red"],
    "THR": PALETTE["orange"],
    "UL94": PALETTE["blue_dark"],
    "UL94_V0": PALETTE["blue_dark"],
    "Tg": PALETTE["purple"],
    "Char_yield": PALETTE["gold"],
    "TS_MPa": PALETTE["green"],
    "FS_MPa": PALETTE["slate"],
    "Delta_LOI": PALETTE["sky"],
    "Delta_PHRR": PALETTE["coral"],
    "Delta_THR": PALETTE["peach"],
    "Delta_CY": PALETTE["gold"],
}

CATEGORY_COLORS = [
    PALETTE["blue"], PALETTE["red"], PALETTE["green"],
    PALETTE["purple"], PALETTE["orange"], PALETTE["gold"],
    PALETTE["teal"], PALETTE["slate"], PALETTE["coral"],
]
SERIES_COLORS = CATEGORY_COLORS

SPLIT_COLORS = {
    "molecule": PALETTE["blue"],
    "scaffold": PALETTE["purple"],
    "reference": PALETTE["orange"],
}
SCENARIO_COLORS = {
    "Conditions only": PALETTE["gray"],
    "Molecular only": PALETTE["purple"],
    "Combined": PALETTE["blue"],
    "Combined + BDE": PALETTE["red"],
}
ZONE_COLORS = {
    "In_domain": PALETTE["green"],
    "Caution": PALETTE["gold"],
    "Extrapolation": PALETTE["red"],
}

RED_CMAP = LinearSegmentedColormap.from_list(
    "paper_red", [PALETTE["rose"], PALETTE["pink"], PALETTE["red"], PALETTE["red_dark"]]
)
BLUE_CMAP = LinearSegmentedColormap.from_list(
    "paper_blue", [PALETTE["blue_light"], PALETTE["sky"], PALETTE["blue"], PALETTE["blue_dark"]]
)
PURPLE_CMAP = LinearSegmentedColormap.from_list(
    "paper_purple", ["#F1EDF7", PALETTE["lavender"], PALETTE["purple"], PALETTE["purple_dark"]]
)
GREEN_CMAP = LinearSegmentedColormap.from_list(
    "paper_green", ["#EDF6F1", PALETTE["green_light"], PALETTE["green"], PALETTE["green_dark"]]
)
ORANGE_CMAP = LinearSegmentedColormap.from_list(
    "paper_orange", ["#FFF4EC", PALETTE["peach"], PALETTE["orange"], PALETTE["orange_dark"]]
)
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "paper_diverging",
    [PALETTE["blue_dark"], PALETTE["blue"], PALETTE["blue_light"], "#FFFFFF", PALETTE["rose"], PALETTE["red"], PALETTE["red_dark"]],
)

_CURRENT_FIGURE_TIER = "supplementary"

TASK_LABELS = {
    "LOI": "LOI",
    "PHRR": "PHRR",
    "THR": "THR",
    "UL94": "UL-94",
    "UL94_V0": "UL-94",
    "Tg": "$T_g$",
    "Char_yield": "Char yield",
    "TS_MPa": "TS",
    "FS_MPa": "FS",
    "Delta_LOI": "$\\Delta$LOI",
    "Delta_PHRR": "$\\Delta$PHRR",
    "Delta_THR": "$\\Delta$THR",
    "Delta_CY": "$\\Delta$Char",
}

TARGET_COLUMNS = {
    "LOI": "LOI",
    "PHRR": "PHRR_kw_㎡",
    "THR": "THR_MJ_㎡",
    "UL94_V0": "UL94",
    "Tg": "Tg_℃",
    "Char_yield": "Char_yield_％_700C",
    "TS_MPa": "TS_MPa",
    "FS_MPa": "FS_MPa",
    "Delta_LOI": "Delta_LOI",
    "Delta_PHRR": "Delta_PHRR",
    "Delta_THR": "Delta_THR",
    "Delta_CY": "Delta_CY",
}

TARGET_AXIS_LABELS = {
    "LOI": "LOI (%)",
    "PHRR": "PHRR (kW m$^{-2}$)",
    "THR": "THR (MJ m$^{-2}$)",
    "Tg": "$T_g$ (°C)",
    "Char_yield": "Char yield at 700 °C (%)",
    "TS_MPa": "Tensile strength (MPa)",
    "FS_MPa": "Flexural strength (MPa)",
    "Delta_LOI": "$\\Delta$LOI (%)",
    "Delta_PHRR": "$\\Delta$PHRR (kW m$^{-2}$)",
    "Delta_THR": "$\\Delta$THR (MJ m$^{-2}$)",
    "Delta_CY": "$\\Delta$Char yield (%)",
}

CORE_REGRESSION_TASKS = ["LOI", "PHRR", "THR", "Tg", "TS_MPa"]
ALL_REGRESSION_TASKS = [
    "LOI", "PHRR", "THR", "Tg", "Char_yield", "TS_MPa", "FS_MPa",
    "Delta_LOI", "Delta_PHRR", "Delta_THR", "Delta_CY",
]


def apply_paper_style(tier: str = "supplementary") -> None:
    """Apply consistent journal typography and geometry.

    Main-text figures use slightly larger type, stronger axes and more generous
    spacing. Supplementary figures remain compact but retain the same visual
    language.
    """
    global _CURRENT_FIGURE_TIER
    _CURRENT_FIGURE_TIER = tier
    main = tier == "main"
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans", "Microsoft YaHei", "SimHei"],
        "font.size": 9.6 if main else 8.5,
        "font.weight": "medium",
        "axes.labelsize": 9.9 if main else 8.7,
        "axes.labelweight": "medium",
        "axes.titlesize": 10.5 if main else 9.3,
        "axes.titleweight": "semibold",
        "axes.titlepad": 7 if main else 5,
        "xtick.labelsize": 8.4 if main else 7.5,
        "ytick.labelsize": 8.4 if main else 7.5,
        "legend.fontsize": 7.8 if main else 6.9,
        "legend.frameon": False,
        "legend.handlelength": 1.7,
        "legend.handletextpad": 0.45,
        "legend.columnspacing": 1.0,
        "axes.linewidth": 0.9 if main else 0.75,
        "axes.edgecolor": PALETTE["dark"],
        "axes.labelcolor": PALETTE["black"],
        "axes.titlecolor": PALETTE["black"],
        "xtick.color": PALETTE["black"],
        "ytick.color": PALETTE["black"],
        "text.color": PALETTE["black"],
        "text.antialiased": True,
        "xtick.major.width": 0.8 if main else 0.65,
        "ytick.major.width": 0.8 if main else 0.65,
        "xtick.major.size": 3.5 if main else 3.0,
        "ytick.major.size": 3.5 if main else 3.0,
        "lines.linewidth": 1.5 if main else 1.2,
        "lines.markersize": 4.5 if main else 3.8,
        "patch.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "path",
        "savefig.transparent": False,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "grid.color": PALETTE["light_gray"],
        "grid.linewidth": 0.65 if main else 0.55,
        "grid.alpha": 0.85,
    })


def apply_main_style() -> None:
    apply_paper_style("main")


def apply_supplementary_style() -> None:
    apply_paper_style("supplementary")


def read_csv_auto(path: Path, **kwargs) -> pd.DataFrame:
    last: Exception | None = None
    for encoding in ("utf-8-sig", "gb18030", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last = exc
    if last:
        raise last
    raise UnicodeError(f"Unable to read {path}")


def _strengthen_export_text(fig: plt.Figure) -> None:
    """Improve readability after a wide figure is scaled to manuscript width.

    Several manuscript figures are drawn at 12--16 inches for layout reasons,
    but are inserted into Word/WPS at roughly journal-column width.  Very small
    6--7 pt annotations therefore become visually weak even in a 900-dpi PNG.
    This export-only pass gently enlarges small text and uses a medium weight
    without changing the underlying data, colours, or panel geometry.
    """
    width_in = float(fig.get_size_inches()[0])
    if _CURRENT_FIGURE_TIER == "main":
        scale = min(max(width_in / 11.0, 1.0), 1.20)
        min_size = 8.0
    else:
        scale = min(max(width_in / 11.5, 1.0), 1.12)
        min_size = 7.2

    for txt in fig.findobj(mpl.text.Text):
        content = txt.get_text()
        if not isinstance(content, str) or not content.strip():
            continue
        size = float(txt.get_fontsize())
        # Increase only the sizes that tend to become weak after document scaling.
        if size <= 11.0:
            txt.set_fontsize(max(min_size, size * scale))
        # Keep intentionally bold/semibold labels unchanged; strengthen normal text.
        weight = txt.get_fontweight()
        if weight in (None, "normal", 400):
            txt.set_fontweight("medium")


def save_figure(fig: plt.Figure, output_base: Path, *, dpi: int = 900, apply_layout: bool = True) -> list[Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    _strengthen_export_text(fig)
    # Harmonise legends and remove accidental box styling before export.
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.set_frame_on(False)
            for txt in legend.get_texts():
                txt.set_color(PALETTE["black"])
    paths = [
        output_base.with_suffix(".pdf"),
        output_base.with_suffix(".png"),
        output_base.with_suffix(".svg"),
    ]
    pad = 0.075 if _CURRENT_FIGURE_TIER == "main" else 0.06
    # Apply a final layout pass before export so labels, titles, legends, and
    # panel annotations do not overlap with plot content or with neighbouring panels.
    if apply_layout:
        try:
            rect = (0.0, 0.0, 1.0, 0.965) if getattr(fig, "_suptitle", None) is not None else None
            fig.tight_layout(pad=1.15 if _CURRENT_FIGURE_TIER == "main" else 0.95, rect=rect)
        except Exception:
            pass
    fig.savefig(paths[0], bbox_inches="tight", pad_inches=pad)
    fig.savefig(paths[1], dpi=max(int(dpi), 900), bbox_inches="tight", pad_inches=pad)
    fig.savefig(paths[2], bbox_inches="tight", pad_inches=pad)
    plt.close(fig)
    return paths


def clean_axes(ax: plt.Axes, *, grid: str | None = None, boxed: bool = False) -> None:
    width = 0.9 if _CURRENT_FIGURE_TIER == "main" else 0.75
    if boxed:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(width)
            spine.set_color(PALETTE["dark"])
    else:
        ax.spines[["top", "right"]].set_visible(False)
        for side in ["left", "bottom"]:
            ax.spines[side].set_linewidth(width)
            ax.spines[side].set_color(PALETTE["dark"])
    ax.tick_params(direction="out", width=width, color=PALETTE["dark"])
    if grid:
        ax.grid(axis=grid, color=PALETTE["light_gray"], linewidth=0.65 if _CURRENT_FIGURE_TIER == "main" else 0.55, alpha=0.85)
        ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str) -> None:
    size = 11.3 if _CURRENT_FIGURE_TIER == "main" else 9.6
    ax.text(-0.11, 1.035, label, transform=ax.transAxes, fontsize=size,
            fontweight="bold", va="bottom", ha="left", color=PALETTE["black"], clip_on=False)


def task_color(task: str, default: str | None = None) -> str:
    return TASK_COLORS.get(task, default or PALETTE["blue"])


def categorical_color(index: int) -> str:
    return CATEGORY_COLORS[index % len(CATEGORY_COLORS)]


def metric_box(ax: plt.Axes, text: str, *, x: float = 0.04, y: float = 0.96, fontsize: float | None = None) -> None:
    ax.text(
        x, y, text, transform=ax.transAxes, va="top", ha="left",
        fontsize=fontsize or (7.7 if _CURRENT_FIGURE_TIER == "main" else 6.8),
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": PALETTE["gray_light"], "linewidth": 0.55, "alpha": 0.92},
    )


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def task_label(task: str) -> str:
    return TASK_LABELS.get(task, task)


def target_axis_label(task: str) -> str:
    return TARGET_AXIS_LABELS.get(task, task_label(task))


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if len(y) < 2:
        return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "n": int(len(y))}
    return {
        "R2": float(r2_score(y, p)),
        "RMSE": float(mean_squared_error(y, p) ** 0.5),
        "MAE": float(mean_absolute_error(y, p)),
        "n": int(len(y)),
    }


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int], probability: Sequence[float] | None = None) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    out = {
        "Accuracy": float(accuracy_score(y, p)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y, p)),
        "Macro_F1": float(f1_score(y, p, average="macro", zero_division=0)),
        "n": int(len(y)),
    }
    if probability is not None:
        prob = np.asarray(probability, dtype=float)
        mask = np.isfinite(prob)
        if mask.sum() and len(np.unique(y[mask])) == 2:
            out["ROC_AUC"] = float(roc_auc_score(y[mask], prob[mask]))
    return out


def normalise_ul94(series: pd.Series) -> pd.Series:
    """Convert textual or numeric UL-94 labels to binary V-0 / non-V-0."""
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")
        # Historical coding: V-0=3, V-1=2, V-2=1, NR=0; binary exports use 0/1.
        return pd.Series(np.where(values.isna(), np.nan, np.where(values.isin([1]), 1, np.where(values >= 3, 1, 0))), index=series.index)
    text = series.astype(str).str.strip().str.upper().str.replace("–", "-", regex=False)
    return pd.Series(np.where(series.isna(), np.nan, text.str.contains(r"^V-?0$|V0", regex=True).astype(int)), index=series.index)


def canonical_smiles(smiles: object) -> str | None:
    if pd.isna(smiles) or not str(smiles).strip():
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None


def murcko_scaffold(smiles: object) -> str | None:
    canonical = canonical_smiles(smiles)
    if not canonical:
        return None
    mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, canonical=True) if scaffold.GetNumAtoms() else "ACYCLIC"
    except Exception:
        return None



def unique_main_molecule_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per canonical main-FR molecule with non-baseline labels.

    The database contains matched neat-EP rows that retain the associated
    flame-retardant identity but carry ``Synergy_type=Neat EP`` and loading 0.
    Selecting the first row per SMILES would therefore mislabel chemical-space
    points.  This helper canonicalises SMILES and chooses labels preferentially
    from positive-loading formulation rows.
    """
    required = [c for c in ["FR_main", "SMILES_main", "Synergy_type", "Preparation_Method", "Loading_total_FR wt%"] if c in df.columns]
    if "SMILES_main" not in required:
        raise KeyError("SMILES_main is required")
    work = df[required].copy()
    work["Canonical_SMILES_main"] = work["SMILES_main"].map(canonical_smiles)
    work = work.dropna(subset=["Canonical_SMILES_main"])
    if "Loading_total_FR wt%" in work:
        work["__loading"] = numeric(work["Loading_total_FR wt%"])
    else:
        work["__loading"] = np.nan

    def choose(group: pd.DataFrame, column: str, excluded: set[str] | None = None) -> object:
        if column not in group:
            return np.nan
        preferred = group.loc[group["__loading"].fillna(0) > 0, column].dropna().astype(str).str.strip()
        values = preferred if not preferred.empty else group[column].dropna().astype(str).str.strip()
        if excluded:
            filtered = values[~values.str.lower().isin({x.lower() for x in excluded})]
            if not filtered.empty:
                values = filtered
        values = values[~values.str.lower().isin({"", "nan", "none", "missing"})]
        return values.value_counts().index[0] if not values.empty else np.nan

    rows = []
    for canonical, group in work.groupby("Canonical_SMILES_main", sort=False):
        rows.append({
            "FR_main": choose(group, "FR_main"),
            "SMILES_main": canonical,
            "Canonical_SMILES_main": canonical,
            "Synergy_type": choose(group, "Synergy_type", {"Neat EP"}),
            "Preparation_Method": choose(group, "Preparation_Method"),
            "max_loading_wt_percent": float(group["__loading"].max()) if group["__loading"].notna().any() else np.nan,
            "record_count": int(len(group)),
        })
    return pd.DataFrame(rows)

def morgan_matrix(smiles_values: Iterable[object], *, radius: int = 2, nbits: int = 1024) -> tuple[np.ndarray, list[bool]]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    rows: list[np.ndarray] = []
    valid: list[bool] = []
    for value in smiles_values:
        mol = Chem.MolFromSmiles(str(value)) if pd.notna(value) else None
        if mol is None:
            valid.append(False)
            continue
        arr = np.zeros(nbits, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), arr)
        rows.append(arr)
        valid.append(True)
    return np.asarray(rows), valid


@dataclass
class LocatedFile:
    path: Path
    score: float
    reason: str


class ResultIndex:
    """Fast path resolver for projects that contain several V4/V5 result versions."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.csv_files = sorted(set(self.root.rglob("*.csv")) | set(self.root.rglob("*.csv.gz"))) if self.root.exists() else []
        self.json_files = sorted(self.root.rglob("*.json")) if self.root.exists() else []
        self.selected: list[dict[str, object]] = []

    @staticmethod
    def _score(path: Path, prefer: Sequence[str], avoid: Sequence[str]) -> float:
        text = path.as_posix().lower()
        score = 0.0
        for i, token in enumerate(prefer):
            if token.lower() in text:
                score += 10.0 - min(i, 8) * 0.3
        for token in avoid:
            if token.lower() in text:
                score -= 20.0
        if "final" in text:
            score += 3.0
        if "5x5" in text:
            score += 2.0
        if "fixed_baseline_inclusive" in text:
            score += 25.0
        elif "fixed" in text:
            score += 4.0
        if "baseline_inclusive" in text:
            score += 4.0
        if "curated" in text or "modified_only" in text:
            score -= 25.0
        if "smoke" in text or "2x2" in text:
            score -= 20.0
        try:
            score += path.stat().st_mtime / 1e12
        except OSError:
            pass
        return score

    def locate(
        self,
        filename: str,
        *,
        prefer: Sequence[str] = (),
        avoid: Sequence[str] = (),
        required_path_tokens: Sequence[str] = (),
        optional: bool = False,
        label: str | None = None,
    ) -> Path | None:
        candidates = [p for p in self.csv_files if p.name == filename]
        if required_path_tokens:
            candidates = [p for p in candidates if all(t.lower() in p.as_posix().lower() for t in required_path_tokens)]
        if not candidates:
            if optional:
                return None
            raise FileNotFoundError(f"Cannot find {filename} under {self.root}")
        ranked = sorted(
            (LocatedFile(p, self._score(p, prefer, avoid), "token score") for p in candidates),
            key=lambda item: (item.score, item.path.as_posix()), reverse=True,
        )
        chosen = ranked[0]
        self.selected.append({
            "item": label or filename,
            "selected_path": str(chosen.path),
            "candidate_count": len(candidates),
            "score": chosen.score,
            "prefer_tokens": ";".join(prefer),
            "avoid_tokens": ";".join(avoid),
        })
        return chosen.path

    def locate_any(
        self,
        filenames: Sequence[str],
        *,
        prefer: Sequence[str] = (),
        avoid: Sequence[str] = (),
        optional: bool = False,
        label: str | None = None,
    ) -> Path | None:
        candidates: list[Path] = []
        for name in filenames:
            candidates.extend(p for p in self.csv_files if p.name == name)
        if not candidates:
            if optional:
                return None
            raise FileNotFoundError(f"Cannot find any of {filenames} under {self.root}")
        ranked = sorted(
            (LocatedFile(p, self._score(p, prefer, avoid), "token score") for p in candidates),
            key=lambda item: (item.score, item.path.as_posix()), reverse=True,
        )
        chosen = ranked[0]
        self.selected.append({
            "item": label or "|".join(filenames),
            "selected_path": str(chosen.path),
            "candidate_count": len(candidates),
            "score": chosen.score,
            "prefer_tokens": ";".join(prefer),
            "avoid_tokens": ";".join(avoid),
        })
        return chosen.path

    def all_named(self, filename: str) -> list[Path]:
        return [p for p in self.csv_files if p.name == filename]

    def save_manifest(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.selected).to_csv(path, index=False, encoding="utf-8-sig")


def task_preferences(task: str, *, use_bde: bool | None = False, purpose: str = "main") -> tuple[list[str], list[str]]:
    prefer = [task.lower(), "molecule", "formulation", "fixed", "baseline_inclusive", "final", "5x5"]
    avoid = [
        "smoke", "2x2", "old", "archive", "pre_audit",
        "curated", "modified_only",
        "screening_mode_ablation", "molecular_only", "conditions_only",
    ]
    if purpose == "shap":
        prefer += ["05_shap", "final_core_fixed_baseline_inclusive_5x5", "final_aux_fixed_baseline_inclusive_5x5"]
    elif purpose == "split":
        prefer += ["scientific_validation", "final_grouping_sensitivity_5x5"]
        avoid += ["05_shap"]
    elif task in {"LOI", "PHRR", "THR", "UL94_V0"}:
        prefer += ["scientific_validation", "final_core_fixed_baseline_inclusive_5x5"]
        avoid += ["05_shap", "06_applicabilitydomain"]
    elif task in {"Tg", "TS_MPa"}:
        prefer += ["scientific_validation", "final_aux_fixed_baseline_inclusive_5x5"]
        avoid += ["05_shap", "06_applicabilitydomain"]
    elif task in {"Char_yield", "FS_MPa"}:
        prefer += ["scientific_validation", "final_exploratory_fixed_baseline_inclusive_5x5"]
        avoid += ["05_shap"]
    elif task.startswith("Delta_"):
        prefer += ["scientific_validation", "final_delta_fixed_5x5"]
        avoid += ["05_shap"]
    if use_bde is True:
        prefer += ["with_bde", "final_bde_paired_ablation_5x5"]
        avoid += ["without_bde"]
    elif use_bde is False:
        prefer += ["without_bde"]
        avoid += ["with_bde"]
    return prefer, avoid


def _canonical_task_path(index: ResultIndex, task: str, filename: str, *, use_bde: bool | None, purpose: str) -> Path | None:
    """Return the manuscript-canonical FINAL path when it exists.

    This prevents copied nested outputs under SHAP/AD folders from outranking
    the authoritative scientific-validation files merely because they are newer.
    """
    if purpose == "shap":
        group = "FINAL_core_fixed_baseline_inclusive_5x5" if task in {"LOI", "PHRR", "THR", "UL94_V0"} else "FINAL_aux_fixed_baseline_inclusive_5x5"
        path = index.root / "05_Shap" / group / task / "molecule" / "formulation" / "without_BDE" / filename
        return path if path.exists() else None

    if use_bde is True:
        path = index.root / "scientific_validation" / "FINAL_BDE_paired_ablation_5x5" / task / "molecule" / "formulation" / "with_BDE" / filename
        return path if path.exists() else None

    if task in {"LOI", "PHRR", "THR", "UL94_V0"}:
        group = "FINAL_core_fixed_baseline_inclusive_5x5"
    elif task in {"Tg", "TS_MPa"}:
        group = "FINAL_aux_fixed_baseline_inclusive_5x5"
    elif task in {"Char_yield", "FS_MPa"}:
        group = "FINAL_exploratory_fixed_baseline_inclusive_5x5"
    elif task.startswith("Delta_"):
        group = "FINAL_Delta_fixed_5x5"
    else:
        return None
    path = index.root / "scientific_validation" / group / task / "molecule" / "formulation" / "without_BDE" / filename
    return path if path.exists() else None


def locate_task_file(index: ResultIndex, task: str, kind: str, *, use_bde: bool | None = False, optional: bool = False, purpose: str = "main") -> Path | None:
    filename = f"{task}_{kind}.csv"
    canonical = _canonical_task_path(index, task, filename, use_bde=use_bde, purpose=purpose)
    if canonical is not None:
        index.selected.append({
            "item": f"{task}:{kind}:{use_bde}",
            "selected_path": str(canonical),
            "candidate_count": 1,
            "score": 1000.0,
            "prefer_tokens": "canonical FINAL path",
            "avoid_tokens": "no fallback used",
        })
        return canonical
    prefer, avoid = task_preferences(task, use_bde=use_bde, purpose=purpose)
    return index.locate(filename, prefer=prefer, avoid=avoid, optional=optional, label=f"{task}:{kind}:{use_bde}")


def probability_column(frame: pd.DataFrame) -> str | None:
    for col in [
        "calibrated_V0_probability", "raw_V0_probability", "calibrated_probability",
        "probability", "y_probability", "y_prob",
    ]:
        if col in frame.columns:
            return col
    return None


def model_caption(metrics: pd.DataFrame) -> str:
    parts: list[str] = []
    for col, prefix in [
        ("selected_model", "Model"),
        ("selected_view", "View"),
        ("selected_requested_k", "K"),
    ]:
        if col in metrics.columns and metrics[col].notna().any():
            value = metrics[col].astype(str).value_counts().index[0]
            parts.append(f"{prefix}: {value}")
    return " | ".join(parts)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def add_no_data(ax: plt.Axes, message: str) -> None:
    ax.axis("off")
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center",
            color=PALETTE["gray"], fontsize=9, wrap=True)


def jitter(n: int, width: float = 0.07, seed: int = 42) -> np.ndarray:
    if n <= 1:
        return np.zeros(n)
    rng = np.random.default_rng(seed)
    return rng.normal(0, width, n)


def robust_lowess_like(x: np.ndarray, y: np.ndarray, bins: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Dependency-free binned-median trend used instead of forced linear fits."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 4 or np.unique(x).size < 3:
        return np.asarray([]), np.asarray([])
    edges = np.unique(np.quantile(x, np.linspace(0, 1, min(bins, len(x)) + 1)))
    if len(edges) < 3:
        return np.asarray([]), np.asarray([])
    centers: list[float] = []
    medians: list[float] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (x >= lo) & (x <= hi if hi == edges[-1] else x < hi)
        if sel.sum() >= 2:
            centers.append(float(np.median(x[sel])))
            medians.append(float(np.median(y[sel])))
    return np.asarray(centers), np.asarray(medians)


apply_supplementary_style()
