# -*- coding: utf-8 -*-
"""Generate all Python-based main-text figures required by the drawing checklist.

Figures are generated from frozen outer-test predictions and analysis CSVs.
Missing upstream results are skipped and reported in a manifest instead of being
silently replaced by invented values.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from paper_utils import (
    ALL_REGRESSION_TASKS,
    CORE_REGRESSION_TASKS,
    BLUE_CMAP,
    DIVERGING_CMAP,
    GREEN_CMAP,
    ORANGE_CMAP,
    PURPLE_CMAP,
    PALETTE,
    SCENARIO_COLORS,
    SPLIT_COLORS,
    ZONE_COLORS,
    ROOT,
    SERIES_COLORS,
    TARGET_COLUMNS,
    ResultIndex,
    add_no_data,
    apply_main_style,
    canonical_smiles,
    categorical_color,
    classification_metrics,
    clean_axes,
    jitter,
    locate_task_file,
    metric_box,
    model_caption,
    morgan_matrix,
    murcko_scaffold,
    normalise_ul94,
    numeric,
    panel_label,
    probability_column,
    read_csv_auto,
    regression_metrics,
    robust_lowess_like,
    safe_name,
    save_figure,
    target_axis_label,
    task_color,
    task_label,
    unique_main_molecule_table,
    write_json,
)

PREPARATION_DISPLAY_MAP = {
    "dopo-based (additive)": "Additive",
    "dopo-based (reactive)": "Reactive",
    "dopo-based (co-curing)": "Co-curing",
    "dopo-based (additive+ secondary crosslinking)":
        "Additive + secondary crosslinking",
    "dopo-based (additive + secondary crosslinking)":
        "Additive + secondary crosslinking",
}
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "data" / "DOPO_EP_new_with_BDE.csv")
    p.add_argument("--results-root", type=Path, default=ROOT / "results")
    p.add_argument("--output", type=Path, default=ROOT / "results" / "09_PaperFigures" / "main")
    p.add_argument("--dpi", type=int, default=900)
    return p.parse_args()


def _task_scaffold_counts(df: pd.DataFrame) -> pd.DataFrame:
    if "SMILES_main" not in df.columns:
        raise KeyError("SMILES_main is required for scaffold counts")
    work = df.copy()
    work["__scaffold"] = work["SMILES_main"].map(murcko_scaffold)
    rows = []
    for task in ["LOI", "PHRR", "THR", "UL94_V0", "Tg", "Char_yield", "TS_MPa", "FS_MPa"]:
        column = TARGET_COLUMNS[task]
        if column not in work.columns:
            continue
        if task == "UL94_V0":
            valid = work[column].notna()
        else:
            valid = numeric(work[column]).notna()
        rows.append({
            "task": task,
            "n_samples": int(valid.sum()),
            "n_unique_main_molecules": int(work.loc[valid, "SMILES_main"].dropna().nunique()),
            "n_unique_main_scaffolds": int(work.loc[valid, "__scaffold"].dropna().nunique()),
        })
    return pd.DataFrame(rows)


def fig2_dataset_composition(df: pd.DataFrame, output: Path) -> None:
    """Create V5 Figure 2 using the established V4 publication layout."""
    counts = _task_scaffold_counts(df)
    counts.to_csv(output / "Fig2a_task_sample_scaffold_counts.csv", index=False, encoding="utf-8-sig")

    # V4 layout retained deliberately:
    #   (a) task coverage | (b) five target distributions
    #   (c) PCA chemical space | (d) preparation method + synergy type
    fig = plt.figure(figsize=(16.0, 9.2))
    outer = fig.add_gridspec(
        2, 2,
        width_ratios=[1.0, 1.85],
        height_ratios=[1.0, 1.0],
        wspace=0.26,
        hspace=0.34,
    )

    # (a) Samples and scaffolds
    ax = fig.add_subplot(outer[0, 0])
    order = [
        t for t in ["LOI", "PHRR", "THR", "UL94_V0", "Tg", "TS_MPa", "Char_yield", "FS_MPa"]
        if t in counts.task.values
    ]
    tab = counts.set_index("task").loc[order]
    y = np.arange(len(tab))
    h = 0.34
    ax.barh(y + h / 2, tab["n_samples"], height=h, color=PALETTE["blue"], label="Valid records")
    ax.barh(y - h / 2, tab["n_unique_main_scaffolds"], height=h, color=PALETTE["gold"], label="Unique scaffolds")
    ax.set_yticks(y, [task_label(t) for t in order])
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.legend(frameon=False, ncol=1, loc="lower right", fontsize=7)
    clean_axes(ax, grid="x")
    xmax = max(tab["n_samples"]) if len(tab) else 1
    for i, (n, s) in enumerate(zip(tab["n_samples"], tab["n_unique_main_scaffolds"])):
        ax.text(float(n) + xmax * 0.012, i + h / 2, str(int(n)), va="center", fontsize=7)
        ax.text(float(s) + xmax * 0.012, i - h / 2, str(int(s)), va="center", fontsize=7)
    panel_label(ax, "(a)")

    # (b) Five target distributions grouped into one panel
    tasks = ["LOI", "PHRR", "THR", "Tg", "TS_MPa"]
    sub_b = outer[0, 1].subgridspec(1, 5, wspace=0.40)
    rows = []
    b_axes = []
    for i, task in enumerate(tasks):
        axb = fig.add_subplot(sub_b[0, i])
        b_axes.append(axb)
        col = TARGET_COLUMNS[task]
        values = numeric(df[col]).dropna() if col in df.columns else pd.Series(dtype=float)
        if values.empty:
            add_no_data(axb, f"Missing {col}")
            continue
        axb.hist(values, bins="auto", color=task_color(task), edgecolor="white", linewidth=0.55, alpha=0.92)
        median = float(values.median())
        axb.axvline(median, color=PALETTE["dark"], linestyle="--", linewidth=1)
        axb.set_title(task_label(task), fontsize=8.2, pad=3)
        axb.set_xlabel(target_axis_label(task), fontsize=7.2)
        if i == 0:
            axb.set_ylabel("Frequency")
        axb.text(
            0.04, 0.96,
            f"n={len(values)}\nMedian={median:.2f}",
            transform=axb.transAxes,
            va="top",
            fontsize=6.5,
        )
        clean_axes(axb)
        rows.append({"task": task, "n": len(values), "median": median, "min": values.min(), "max": values.max()})
    pd.DataFrame(rows).to_csv(output / "Fig2b_target_distribution_summary.csv", index=False, encoding="utf-8-sig")
    if b_axes:
        panel_label(b_axes[0], "(b)")

    # (c) Unique-main-molecule PCA chemical space
    axc = fig.add_subplot(outer[1, 0])
    moltab = unique_main_molecule_table(df)
    matrix, valid = morgan_matrix(moltab["SMILES_main"], radius=2, nbits=1024)
    moltab = moltab.loc[np.asarray(valid)].reset_index(drop=True)
    if len(moltab) >= 3:
        coords = PCA(n_components=2, random_state=42).fit_transform(matrix)
        moltab["PC1"], moltab["PC2"] = coords[:, 0], coords[:, 1]
        categories = moltab.get("Synergy_type", pd.Series("Unknown", index=moltab.index)).fillna("Unknown").astype(str)
        top = categories.value_counts().head(7).index
        categories = categories.where(categories.isin(top), "Other")
        for i, category in enumerate(categories.value_counts().index):
            sel = categories.eq(category)
            axc.scatter(
                moltab.loc[sel, "PC1"], moltab.loc[sel, "PC2"],
                s=28, color=categorical_color(i), alpha=0.78,
                edgecolor="white", linewidth=0.35, label=category,
            )
        axc.set_xlabel("PC1 (Morgan fingerprint)")
        axc.set_ylabel("PC2 (Morgan fingerprint)")
        axc.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=6.4)
        clean_axes(axc)
        moltab.to_csv(output / "Fig2c_unique_main_PCA_coordinates.csv", index=False, encoding="utf-8-sig")
    else:
        add_no_data(axc, "Insufficient valid molecules for PCA")
    panel_label(axc, "(c)")

    # (d) Preparation method and synergy type grouped into one panel.
    # Keep the corrected V5 display mapping while restoring the V4 geometry.
    sub_d = outer[1, 1].subgridspec(1, 2, wspace=0.42)
    d_axes = []
    for k, (column, color, label) in enumerate([
        ("Preparation_Method", PALETTE["green"], "Preparation method"),
        ("Synergy_type", PALETTE["purple"], "Synergy type"),
    ]):
        axd = fig.add_subplot(sub_d[0, k])
        d_axes.append(axd)
        if column not in df.columns:
            add_no_data(axd, f"Missing {column}")
            continue
        values = df[column].fillna("Missing").astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
        if column == "Preparation_Method":
            normalized = values.str.lower()
            values = normalized.map(PREPARATION_DISPLAY_MAP).fillna(values)
        counts_s = values.value_counts()
        if column == "Synergy_type" and len(counts_s) > 9:
            kept = counts_s.head(8).copy()
            kept.loc["Other"] = counts_s.iloc[8:].sum()
            counts_s = kept
        counts_s = counts_s.sort_values()
        labels = [x.replace("DOPO-based (", "").replace(")", "") for x in counts_s.index]
        axd.barh(np.arange(len(counts_s)), counts_s.values, color=color)
        axd.set_yticks(np.arange(len(counts_s)), labels)
        axd.set_xlabel("Number of records")
        axd.set_title(label, fontsize=8.5)
        clean_axes(axd, grid="x")
        xmax_d = max(counts_s.values) if len(counts_s) else 1
        for j, v in enumerate(counts_s.values):
            axd.text(v + xmax_d * 0.015, j, str(int(v)), va="center", fontsize=7)
    if d_axes:
        panel_label(d_axes[0], "(d)")

    fig.subplots_adjust(left=0.07, right=0.975, top=0.97, bottom=0.08)
    # Keep the current V5 filename so downstream paper scripts do not break.
    save_figure(fig, output / "Fig2_dataset_composition_overview")


def _load_regression_bundle(index: ResultIndex, task: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_path = locate_task_file(index, task, "outer_predictions", use_bde=False)
    met_path = locate_task_file(index, task, "outer_fold_metrics", use_bde=False)
    return read_csv_auto(pred_path), read_csv_auto(met_path)


def fig3_outer_predictions(index: ResultIndex, output: Path) -> None:
    tasks = ["LOI", "PHRR", "THR", "Tg", "TS_MPa"]
    fig, axes = plt.subplots(1, 5, figsize=(16.2, 3.55))
    summary_rows = []
    audit_rows = []
    for i, (ax, task) in enumerate(zip(axes, tasks)):
        try:
            pred, folds = _load_regression_bundle(index, task)
        except Exception as exc:
            add_no_data(ax, str(exc))
            continue
        y = numeric(pred["y_true"])
        p = numeric(pred["y_pred"])
        valid = y.notna() & p.notna()
        y, p = y[valid], p[valid]
        metrics = regression_metrics(y, p)
        lo, hi = float(min(y.min(), p.min())), float(max(y.max(), p.max()))
        pad = (hi - lo) * 0.04 if hi > lo else 1.0
        ax.scatter(y, p, s=20, color=task_color(task), alpha=0.72, edgecolor="white", linewidth=0.30)
        ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], linestyle="--", color=PALETTE["slate"], linewidth=1.15)
        ax.set_xlim(lo-pad, hi+pad)
        ax.set_ylim(lo-pad, hi+pad)
        ax.set_xlabel("Experimental")
        if i == 0:
            ax.set_ylabel("Outer-test prediction")
        selected_models = (
            folds["selected_model"]
            .dropna()
            .astype(str)
            .unique()
            if "selected_model" in folds.columns
            else []
        )

        if len(selected_models) == 1:
            caption = selected_models[0]
        elif len(selected_models) > 1:
            caption = "Models varied across outer folds"
        else:
            caption = "Nested grouped CV"
        ax.set_title(f"{task_label(task)}", fontsize=8.8)
        metric_box(ax, f"$R^2$={metrics['R2']:.3f}\nRMSE={metrics['RMSE']:.2f}\nMAE={metrics['MAE']:.2f}")
        ax.set_aspect("equal", adjustable="box")
        clean_axes(ax)
        panel_label(ax, f"({chr(97+i)})")
        summary_rows.append({"task": task, **metrics})
        # Each row should occur once in the pooled outer-test predictions.
        id_col = next(
            (
                c
                for c in [
                    "Record_ID",
                    "original_index",
                    "row_index",
                    "sample_index",
                    "ID",
                ]
                if c in pred.columns
            ),
            None,
        )
        duplicate_count = int(pred[id_col].duplicated().sum()) if id_col else np.nan
        audit_rows.append({"task": task, "n_prediction_rows": len(pred), "identifier_column": id_col, "duplicate_identifier_rows": duplicate_count,
                           "outer_folds": int(pred["outer_fold"].nunique()) if "outer_fold" in pred.columns else np.nan})
    fig.tight_layout(pad=1.15, w_pad=1.0, h_pad=1.0)
    save_figure(fig, output / "Fig3_core_regression_outer_predictions")
    pd.DataFrame(summary_rows).to_csv(output / "Fig3_metrics_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audit_rows).to_csv(output / "Fig3_outer_prediction_uniqueness_audit.csv", index=False, encoding="utf-8-sig")


def _horizontal_fold_plot(ax: plt.Axes, rows: pd.DataFrame, order: list[str], *, xlabel: str, grey_tasks: set[str] | None = None) -> None:
    grey_tasks = grey_tasks or set()
    for i, task in enumerate(order):
        vals = numeric(rows.loc[rows.task.eq(task), "score"]).dropna().to_numpy()
        if len(vals) == 0:
            continue
        color = PALETTE["gray"] if task in grey_tasks else task_color(task)
        ax.scatter(vals, np.full(len(vals), i) + jitter(len(vals), 0.045, 42+i), s=26,
                   color=color, alpha=0.7, edgecolor="white", linewidth=0.3)
        mean = float(np.mean(vals)); std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        ax.errorbar(mean, i, xerr=std, marker="D", color=color, markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.0, capsize=3, linewidth=1.25, markersize=4.5)
    ax.axvline(0, linestyle="--", color=PALETTE["dark"], linewidth=0.8)
    ax.set_yticks(range(len(order)), [task_label(t) for t in order])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    clean_axes(ax, grid="x")


def fig4_stability_and_split(index: ResultIndex, output: Path) -> None:
    """Plot FINAL V5 stability/grouping data using the established V4 layout."""
    regression_rows = []
    ul94_rows = []
    for task in ["LOI", "PHRR", "THR", "Tg", "TS_MPa", "Char_yield", "FS_MPa"]:
        path = locate_task_file(index, task, "outer_fold_metrics", use_bde=False, optional=True)
        if path is None:
            continue
        frame = read_csv_auto(path)
        if "outer_R2" not in frame.columns:
            continue
        for _, row in frame.iterrows():
            regression_rows.append({"task": task, "outer_fold": row.get("outer_fold"), "score": row["outer_R2"], "source": str(path)})

    path = locate_task_file(index, "UL94_V0", "outer_fold_metrics", use_bde=False, optional=True)
    if path:
        frame = read_csv_auto(path)
        if "outer_Macro_F1" in frame.columns:
            for _, row in frame.iterrows():
                ul94_rows.append({"task": "UL94_V0", "outer_fold": row.get("outer_fold"), "score": row["outer_Macro_F1"], "source": str(path)})

    reg = pd.DataFrame(regression_rows)
    cls = pd.DataFrame(ul94_rows)
    reg.to_csv(output / "Fig4a_regression_outer_fold_scores.csv", index=False, encoding="utf-8-sig")
    cls.to_csv(output / "Fig4b_UL94_outer_fold_scores.csv", index=False, encoding="utf-8-sig")

    # Corrected V5 source rule: only use the frozen FINAL grouping-sensitivity run.
    split_records = []
    for task in ["LOI", "PHRR", "THR", "UL94_V0"]:
        filename = f"{task}_outer_fold_metrics.csv"
        for p in index.all_named(filename):
            text_path = p.as_posix().lower()
            if not any(token in text_path for token in ["final_grouping_sensitivity_5x5", "three_split_validation"]):
                continue
            split = next((s for s in ["molecule", "scaffold", "reference"] if f"/{s}/" in text_path or f"\\{s}\\" in text_path), None)
            if split is None or "smoke" in text_path:
                continue
            frame = read_csv_auto(p)
            metric = "outer_Macro_F1" if task == "UL94_V0" else "outer_R2"
            if metric not in frame.columns:
                continue
            for _, row in frame.iterrows():
                split_records.append({"task": task, "split": split, "outer_fold": row.get("outer_fold"), "score": row[metric], "source": str(p)})

    split_df = pd.DataFrame(split_records).drop_duplicates(["task", "split", "outer_fold"], keep="last") if split_records else pd.DataFrame()
    split_df.to_csv(output / "Fig4_split_sensitivity_outer_folds.csv", index=False, encoding="utf-8-sig")

    # V4 geometry: wide regression panels on the left, narrow UL-94 panels on the right.
    fig, axes = plt.subplots(
        2, 2,
        figsize=(12.4, 8.4),
        gridspec_kw={"width_ratios": [3.2, 1.2], "height_ratios": [1.0, 0.95]},
    )

    order = [
        t for t in ["LOI", "PHRR", "THR", "Tg", "TS_MPa", "Char_yield", "FS_MPa"]
        if not reg.empty and t in reg.task.unique()
    ]
    if order:
        _horizontal_fold_plot(axes[0, 0], reg, order, xlabel="Outer-fold $R^2$", grey_tasks={"Char_yield", "FS_MPa"})
    else:
        add_no_data(axes[0, 0], "No regression fold metrics")
    axes[0, 0].set_title("Outer-fold model stability", fontsize=9)
    panel_label(axes[0, 0], "(a)")

    if not cls.empty:
        _horizontal_fold_plot(axes[0, 1], cls, ["UL94_V0"], xlabel="Outer-fold Macro-F1")
        axes[0, 1].set_xlim(max(0, float(cls.score.min()) - 0.1), min(1.0, float(cls.score.max()) + 0.1))
    else:
        add_no_data(axes[0, 1], "No UL-94 fold metrics")
    axes[0, 1].set_title("UL-94 stability", fontsize=9)
    panel_label(axes[0, 1], "(b)")

    for ax, tasks, xlabel, title in [
        (axes[1, 0], ["LOI", "PHRR", "THR"], "$R^2$", "Grouping-strategy sensitivity"),
        (axes[1, 1], ["UL94_V0"], "Macro-F1", "UL-94 grouping sensitivity"),
    ]:
        if split_df.empty:
            add_no_data(ax, "FINAL grouping-sensitivity results are missing")
            ax.set_title(title, fontsize=9)
            continue
        positions = np.arange(len(tasks))
        offsets = {"molecule": -0.22, "scaffold": 0.0, "reference": 0.22}
        for j, split_name in enumerate(["molecule", "scaffold", "reference"]):
            for i, task in enumerate(tasks):
                vals = numeric(split_df.loc[(split_df.task == task) & (split_df.split == split_name), "score"]).dropna().to_numpy()
                if len(vals) == 0:
                    continue
                y0 = positions[i] + offsets[split_name]
                ax.scatter(
                    vals,
                    np.full(len(vals), y0) + jitter(len(vals), 0.025, 100 + i + j),
                    s=22,
                    color=SPLIT_COLORS[split_name],
                    alpha=0.60,
                )
                ax.errorbar(
                    np.mean(vals), y0,
                    xerr=np.std(vals, ddof=1) if len(vals) > 1 else 0,
                    marker="o",
                    color=SPLIT_COLORS[split_name],
                    capsize=3,
                    linewidth=1.2,
                    label=split_name if i == 0 else None,
                )
        ax.axvline(0, linestyle="--", color=PALETTE["dark"], linewidth=0.8)
        ax.set_yticks(positions, [task_label(t) for t in tasks])
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=9)
        clean_axes(ax, grid="x")
        ax.legend(frameon=False, loc="best", fontsize=7)

    panel_label(axes[1, 0], "(c)")
    panel_label(axes[1, 1], "(d)")
    fig.tight_layout(pad=1.2, w_pad=1.15, h_pad=1.35)
    # Keep the current V5 filename while restoring the V4 visual form.
    save_figure(fig, output / "Fig4_stability_and_grouping_sensitivity")


def fig5_ul94(index: ResultIndex, output: Path) -> None:
    """UL-94 diagnostics in V4 layout with the user-specified Figure-5 palette."""
    pred_path = locate_task_file(index, "UL94_V0", "outer_predictions", use_bde=False)
    frame = read_csv_auto(pred_path)

    y = numeric(frame["y_true"])
    pred = numeric(frame["y_pred"])
    prob_col = probability_column(frame)
    if prob_col is None:
        raise KeyError(f"No probability column in {pred_path}")
    prob = numeric(frame[prob_col]).clip(0, 1)

    valid = y.notna() & pred.notna() & prob.notna()
    y = y[valid].astype(int).to_numpy()
    pred = pred[valid].astype(int).to_numpy()
    prob = prob[valid].to_numpy()

    metrics = classification_metrics(y, pred, prob)
    fpr, tpr, _ = roc_curve(y, prob)
    precision, recall, _ = precision_recall_curve(y, prob)
    frac_pos, mean_pred = calibration_curve(
        y, prob,
        n_bins=min(10, max(4, len(y) // 20)),
        strategy="quantile",
    )
    cm = confusion_matrix(y, pred, labels=[0, 1])

    # Figure 5 only: palette taken from the supplied reference image.
    fig5_blue_dark = "#3A4B6E"
    fig5_blue = "#4C9AC9"
    fig5_blue_mid = "#9FBBD5"
    fig5_blue_pale = "#C8E0EF"
    fig5_blue_light = "#E7F2F6"
    fig5_red = "#BA3E45"
    fig5_olive = "#C9C780"
    fig5_gray = "#A7AFB8"
    fig5_dark = "#2F3A4A"

    cm_cmap = LinearSegmentedColormap.from_list(
        "ul94_reference_blue",
        [fig5_blue_light, fig5_blue_pale, fig5_blue_mid, fig5_blue, "#4E6691", fig5_blue_dark],
    )

    # Restore the V4 2 x 2 geometry and information density.
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.8))

    # (a) Confusion matrix
    ax = axes[0, 0]
    im = ax.imshow(cm, cmap=cm_cmap, vmin=0, vmax=float(cm.max()) if cm.size else 1)
    text_threshold = 0.55 * float(cm.max()) if cm.size else 0
    for (i, j), v in np.ndenumerate(cm):
        ax.text(
            j, i, str(v),
            ha="center", va="center",
            fontsize=12,
            color="white" if v >= text_threshold else fig5_dark,
            fontweight="medium",
        )
    ax.set_xticks([0, 1], ["Non-V-0", "V-0"])
    ax.set_yticks([0, 1], ["Non-V-0", "V-0"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    cbar = fig.colorbar(im, ax=ax, fraction=0.050, pad=0.045)
    cbar.set_label("Number of samples\n(light → dark = larger)", fontsize=7.5)
    vmax_cm = int(cm.max()) if cm.size else 1
    cbar_ticks = np.unique(np.linspace(0, vmax_cm, min(5, vmax_cm + 1)).round().astype(int))
    cbar.set_ticks(cbar_ticks)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_linewidth(0.6)
    panel_label(ax, "(a)")

    # (b) ROC curve
    ax = axes[0, 1]
    ax.plot(fpr, tpr, color=fig5_blue, linewidth=2, label=f"AUC={auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], "--", color=fig5_gray, linewidth=1)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.legend(frameon=False)
    clean_axes(ax)
    panel_label(ax, "(b)")

    # (c) Precision-recall curve
    ax = axes[1, 0]
    ap = average_precision_score(y, prob)
    baseline = float(np.mean(y))
    ax.plot(recall, precision, color=fig5_red, linewidth=2, label=f"AP={ap:.3f}")
    ax.axhline(baseline, linestyle="--", color=fig5_gray, linewidth=1, label=f"Baseline={baseline:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(frameon=False)
    clean_axes(ax)
    panel_label(ax, "(c)")

    # (d) Calibration curve
    ax = axes[1, 1]
    ax.plot(
        mean_pred, frac_pos,
        marker="o",
        color=fig5_olive,
        linewidth=2,
        markerfacecolor="white",
        markeredgecolor=fig5_olive,
        markeredgewidth=1.0,
    )
    ax.plot([0, 1], [0, 1], "--", color=fig5_gray, linewidth=1)
    ax.set_xlabel("Mean predicted V-0 probability")
    ax.set_ylabel("Observed V-0 fraction")
    clean_axes(ax)
    panel_label(ax, "(d)")

    fig.suptitle(
        (
            "UL-94 classification | "
            f"Accuracy={metrics['Accuracy']:.3f} | "
            f"Balanced accuracy={metrics['Balanced_Accuracy']:.3f} | "
            f"Macro-F1={metrics['Macro_F1']:.3f} | "
            f"ROC-AUC={metrics.get('ROC_AUC', np.nan):.3f}"
        ),
        y=1.01,
        fontsize=10.2,
        fontweight="semibold",
    )

    fig.tight_layout(pad=1.25, w_pad=1.0, h_pad=1.0)
    save_figure(fig, output / "Fig5_UL94_classification_diagnostics")

    # Preserve the V5 audit-friendly metric export.
    metric_row = dict(metrics)
    metric_row.update({
        "evaluation_scope": "pooled outer-of-fold predictions",
        "note": "Table 3 reports five-outer-fold mean ± SD; pooled metrics can differ slightly.",
        "average_precision": ap,
        "positive_class_baseline": baseline,
    })
    pd.DataFrame([metric_row]).to_csv(output / "Fig5_UL94_metrics.csv", index=False, encoding="utf-8-sig")


def fig6_information_source(index: ResultIndex, output: Path) -> None:
    """Plot the frozen FINAL information-source ablation.

    The FINAL ablation contains three controlled sources: all formulation
    information (Combined), conditions-only, and molecular-only. BDE is kept
    separate in Fig. 7 as a paired mechanistic-descriptor ablation.
    """
    base = index.root / "scientific_validation" / "FINAL_information_source_ablation_5x5"
    records = []
    scope_labels = {
        "all": "Combined",
        "conditions_only": "Conditions only",
        "molecular_only": "Molecular only",
    }
    for scope, label in scope_labels.items():
        for task in ["LOI", "PHRR", "THR", "UL94_V0"]:
            if scope == "all":
                path = base / task / "molecule" / "formulation" / "without_BDE" / f"{task}_outer_fold_metrics.csv"
            else:
                path = base / task / "molecule" / "formulation" / scope / "without_BDE" / f"{task}_outer_fold_metrics.csv"
            if not path.exists():
                continue
            frame = read_csv_auto(path)
            metric = "outer_Macro_F1" if task == "UL94_V0" else "outer_R2"
            if metric not in frame.columns:
                continue
            for _, row in frame.iterrows():
                records.append({
                    "task": task,
                    "scenario": scope,
                    "scenario_label": label,
                    "outer_fold": row.get("outer_fold"),
                    "metric": row.get(metric),
                    "source": str(path),
                })
    data = pd.DataFrame(records)
    if data.empty:
        raise FileNotFoundError(f"FINAL information-source ablation results are missing under {base}")
    required = {(task, scope) for task in ["LOI", "PHRR", "THR", "UL94_V0"] for scope in scope_labels}
    observed = set(zip(data["task"].astype(str), data["scenario"].astype(str)))
    missing = sorted(required - observed)
    if missing:
        raise FileNotFoundError(f"FINAL information-source ablation is incomplete; missing task/scope pairs: {missing}")
    data.to_csv(output / "Fig6_information_source_ablation_data.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [3, 1.2]})
    scenarios = list(scope_labels.values())
    for ax, tasks, ylabel in [(axes[0], ["LOI", "PHRR", "THR"], "Outer-fold $R^2$"), (axes[1], ["UL94_V0"], "Outer-fold Macro-F1")]:
        x = np.arange(len(tasks)); offsets = np.linspace(-0.22, 0.22, len(scenarios))
        for j, scenario in enumerate(scenarios):
            for i, task in enumerate(tasks):
                vals = numeric(data.loc[(data.task.astype(str)==task) & (data.scenario_label.astype(str)==scenario), "metric"]).dropna().to_numpy()
                if len(vals)==0:
                    continue
                color = SCENARIO_COLORS.get(scenario, categorical_color(j))
                ax.scatter(np.full(len(vals), x[i]+offsets[j])+jitter(len(vals),0.018,300+i+j), vals,
                           color=color, s=22, alpha=0.55)
                ax.errorbar(x[i]+offsets[j], np.mean(vals), yerr=np.std(vals,ddof=1) if len(vals)>1 else 0,
                            marker=["o","s","^"][j%3], color=color, capsize=3, linewidth=1, label=scenario if i==0 else None)
        ax.set_xticks(x, [task_label(t) for t in tasks]); ax.set_ylabel(ylabel); ax.axhline(0, ls="--", lw=0.8, color=PALETTE["dark"])
        clean_axes(ax, grid="y")
    panel_label(axes[0], "(a)"); panel_label(axes[1], "(b)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94), pad=1.15, w_pad=1.1)
    save_figure(fig, output / "Fig6_information_source_ablation")


def _aggregate_bde_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    if not {"BDE_true_kJ_mol", "BDE_pred_kJ_mol"}.issubset(frame.columns):
        raise KeyError("BDE prediction columns are missing")
    key_cols = [c for c in ["ID", "Canonical_SMILES", "Smiles", "Bond_Type"] if c in frame.columns]
    if not key_cols:
        frame = frame.copy(); frame["__row"] = np.arange(len(frame)); key_cols = ["__row"]
    agg = frame.groupby(key_cols, dropna=False, as_index=False).agg(
        BDE_true_kJ_mol=("BDE_true_kJ_mol", "mean"),
        BDE_pred_kJ_mol=("BDE_pred_kJ_mol", "mean"),
        BDE_pred_std_kJ_mol=("BDE_pred_kJ_mol", "std"),
        n_test_appearances=("BDE_pred_kJ_mol", "size"),
    )
    agg["abs_error_kJ_mol"] = (agg["BDE_true_kJ_mol"]-agg["BDE_pred_kJ_mol"]).abs()
    return agg


def fig7_bde(index: ResultIndex, output: Path) -> None:
    bde_path = index.locate("BDE_selected_config_all_predictions.csv", prefer=["07_bde", "diagnostics"], optional=True)
    bde = _aggregate_bde_predictions(read_csv_auto(bde_path)) if bde_path else pd.DataFrame()
    paired_records = []
    paired_root = index.root / "scientific_validation" / "FINAL_BDE_paired_ablation_5x5"
    for task in ["LOI", "PHRR", "THR", "Tg", "TS_MPa", "UL94_V0"]:
        p0 = paired_root / task / "molecule" / "formulation" / "without_BDE" / f"{task}_outer_fold_metrics.csv"
        p1 = paired_root / task / "molecule" / "formulation" / "with_BDE" / f"{task}_outer_fold_metrics.csv"
        if not p0.exists() or not p1.exists():
            continue
        a, b = read_csv_auto(p0), read_csv_auto(p1)
        metric = "outer_Macro_F1" if task=="UL94_V0" else "outer_R2"
        if metric not in a or metric not in b: continue
        merged = a[["outer_fold",metric]].merge(b[["outer_fold",metric]],on="outer_fold",suffixes=("_without","_with"))
        for _, r in merged.iterrows():
            paired_records.append({"task":task,"outer_fold":r.outer_fold,"without_BDE":r[f"{metric}_without"],"with_BDE":r[f"{metric}_with"],
                                   "delta":r[f"{metric}_with"]-r[f"{metric}_without"],"metric":metric})
    paired = pd.DataFrame(paired_records)
    bde.to_csv(output / "Fig7_BDE_aggregated_predictions.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(output / "Fig7_BDE_paired_ablation.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.3))
    ax = axes[0,0]
    if bde.empty:
        add_no_data(ax, "Run 07_BDE/run_BDE.py to generate BDE predictions")
    else:
        y=numeric(bde.BDE_true_kJ_mol); p=numeric(bde.BDE_pred_kJ_mol); m=regression_metrics(y,p)
        lo,hi=min(y.min(),p.min()),max(y.max(),p.max())
        ax.scatter(y,p,color=PALETTE["blue"],s=25,alpha=0.65,edgecolor="white",linewidth=0.3)
        ax.plot([lo,hi],[lo,hi],"--",color=PALETTE["dark"],lw=1)
        ax.set_xlabel("Experimental BDE (kJ mol$^{-1}$)"); ax.set_ylabel("Predicted BDE (kJ mol$^{-1}$)")
        metric_box(ax, f"$R^2$={m['R2']:.3f}\nRMSE={m['RMSE']:.2f}\nMAE={m['MAE']:.2f}")
        clean_axes(ax)
    panel_label(ax,"(a)")

    ax=axes[0,1]
    if bde.empty or "Bond_Type" not in bde.columns:
        add_no_data(ax,"Bond_Type not available")
    else:
        types=[x for x in ["P-C","P-N"] if x in bde.Bond_Type.astype(str).unique()]
        data=[numeric(bde.loc[bde.Bond_Type.astype(str)==x,"abs_error_kJ_mol"]).dropna() for x in types]
        bp=ax.boxplot(data,tick_labels=[f"{x}\n(n={len(v)})" for x,v in zip(types,data)],patch_artist=True,showfliers=True)
        for patch,color in zip(bp['boxes'],[PALETTE['blue'],PALETTE['red']]): patch.set_facecolor(color); patch.set_alpha(0.82)
        ax.set_ylabel("Absolute error (kJ mol$^{-1}$)"); clean_axes(ax,grid="y")
    panel_label(ax,"(b)")

    ax=axes[1,0]
    if paired.empty:
        add_no_data(ax,"Run matched with/without-BDE 5×5 validation")
    else:
        tasks=list(dict.fromkeys(paired.task))
        for i,task in enumerate(tasks):
            sub=paired[paired.task==task]
            for _,r in sub.iterrows():
                ax.plot([i-0.16,i+0.16],[r.without_BDE,r.with_BDE],color=PALETTE["gray"],alpha=.55,lw=.9)
                ax.scatter(i-0.16,r.without_BDE,s=24,color=PALETTE["blue"],edgecolor="white",linewidth=.3,zorder=2)
                ax.scatter(i+0.16,r.with_BDE,s=24,color=PALETTE["red"],edgecolor="white",linewidth=.3,zorder=2)
            ax.scatter(i-0.16,sub.without_BDE.mean(),marker="D",s=46,color=PALETTE["blue_dark"],zorder=3)
            ax.scatter(i+0.16,sub.with_BDE.mean(),marker="D",s=46,color=PALETTE["red_dark"],zorder=3)
        ax.set_xticks(range(len(tasks)),[task_label(t) for t in tasks],rotation=25,ha="right")
        ax.set_ylabel("Outer-fold $R^2$ / Macro-F1"); ax.axhline(0,ls="--",lw=.8,color=PALETTE['dark']); clean_axes(ax,grid="y")
        ax.text(.02,.02,"Blue: without BDE   Red: with BDE",transform=ax.transAxes,fontsize=7)
    panel_label(ax,"(c)")

    ax=axes[1,1]
    if paired.empty:
        add_no_data(ax,"No paired ablation data")
    else:
        delta=paired.groupby("task",as_index=False).agg(mean_delta=("delta","mean"),std_delta=("delta","std"),n=("delta","size"))
        colors=[PALETTE['green'] if x>=0 else PALETTE['red'] for x in delta.mean_delta]
        ax.barh(np.arange(len(delta)),delta.mean_delta,xerr=delta.std_delta,color=colors,capsize=3)
        ax.set_yticks(np.arange(len(delta)),[task_label(t) for t in delta.task]); ax.invert_yaxis(); ax.axvline(0,ls="--",lw=.8,color=PALETTE['dark'])
        ax.set_xlabel("Mean $\\Delta R^2$ / $\\Delta$Macro-F1"); clean_axes(ax,grid="x")
        delta.to_csv(output / "Fig7d_BDE_mean_delta.csv",index=False,encoding="utf-8-sig")
    panel_label(ax,"(d)")
    fig.tight_layout(pad=1.15, w_pad=1.0, h_pad=1.0)
    save_figure(fig,output/"Fig7_BDE_model_and_ablation")


def _shap_summary_path(index: ResultIndex, task: str) -> Path | None:
    return index.locate(f"{task}_SHAP_stability_summary.csv", prefer=["05_shap",task,"5x5"], avoid=["smoke","2x2"], optional=True)


def _shap_long_paths(index: ResultIndex, task: str) -> list[Path]:
    return [p for p in index.csv_files if p.name.startswith(f"{task}_outer_fold_") and p.name.endswith("_SHAP_values_long.csv.gz") and "smoke" not in p.as_posix().lower()]


def _beeswarm(ax: plt.Axes, long: pd.DataFrame, top_features: list[str]) -> None:
    # Custom lightweight beeswarm: all sample SHAP points, coloured by feature value.
    for i, feature in enumerate(top_features[::-1]):
        sub=long[long.feature.astype(str)==feature].copy()
        shap=numeric(sub.shap_value); raw=numeric(sub.get("feature_value_raw",sub.get("feature_value_transformed")))
        valid=shap.notna(); shap=shap[valid].to_numpy(); raw=raw[valid].to_numpy()
        if len(shap)==0: continue
        if np.isfinite(raw).sum()>1 and np.nanmax(raw)>np.nanmin(raw):
            c=(raw-np.nanmin(raw))/(np.nanmax(raw)-np.nanmin(raw))
        else: c=np.full(len(raw),.5)
        ax.scatter(shap,np.full(len(shap),i)+jitter(len(shap),.10,600+i),c=c,cmap=DIVERGING_CMAP,vmin=0,vmax=1,s=12,alpha=.65,edgecolor="none")
    ax.axvline(0,ls="--",lw=.8,color=PALETTE['gray'])
    ax.set_yticks(range(len(top_features)),[f.replace("Morgan_","Morgan bit ") for f in top_features[::-1]])
    ax.set_xlabel("SHAP value"); clean_axes(ax,grid="x")


def fig8_shap(index: ResultIndex, output: Path) -> None:
    """Generate Fig. 8: cross-fold SHAP structure–formulation–property analysis.

    Panel (c) reports the within-task share of aggregated mean absolute SHAP
    values.  It is intentionally labelled as a SHAP attribution *share*, not a
    physical/causal contribution, because absolute SHAP scales differ among
    targets and SHAP does not establish causality.
    """
    tasks = ["LOI", "PHRR", "UL94_V0"]
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.0))
    axes_flat = list(axes.ravel())

    # (a-b) Cross-fold-stable sample-level SHAP distributions.
    for k, task in enumerate(tasks[:2]):
        ax = axes_flat[k]
        summary_path = _shap_summary_path(index, task)
        long_paths = _shap_long_paths(index, task)
        if long_paths:
            long = pd.concat([read_csv_auto(p) for p in long_paths], ignore_index=True)
            if summary_path:
                s = read_csv_auto(summary_path).sort_values(
                    "mean_abs_shap_all_folds", ascending=False
                )
                top = s.head(15).feature.astype(str).tolist()
            else:
                top = (
                    long.assign(a=numeric(long.shap_value).abs())
                    .groupby("feature").a.mean()
                    .sort_values(ascending=False)
                    .head(15).index.astype(str).tolist()
                )
            _beeswarm(ax, long, top)
        elif summary_path:
            s = (
                read_csv_auto(summary_path)
                .sort_values("mean_abs_shap_all_folds", ascending=False)
                .head(15)
                .sort_values("mean_abs_shap_all_folds")
            )
            ax.barh(
                np.arange(len(s)),
                s.mean_abs_shap_all_folds,
                color=task_color(task),
            )
            ax.set_yticks(np.arange(len(s)), s.feature.astype(str))
            ax.set_xlabel("Mean |SHAP| across folds")
            clean_axes(ax, grid="x")
            ax.text(
                .02, .02,
                "Aggregate fallback; rerun SHAP to export sample-level beeswarm data",
                transform=ax.transAxes,
                fontsize=6.3,
            )
        else:
            add_no_data(ax, f"Missing {task} SHAP results")

        ax.set_title(task_label(task))
        ax.tick_params(axis="y", labelsize=7.0)
        panel_label(ax, f"({chr(97 + k)})")

    # (c) Within-task share of aggregated mean absolute SHAP by feature group.
    # This is a model-attribution summary, not a causal/physical contribution.
    ax = axes_flat[2]
    group_rows = []
    for task in ["LOI", "PHRR", "THR", "UL94_V0"]:
        p = index.locate(
            f"{task}_SHAP_group_summary.csv",
            prefer=["05_shap", task, "5x5"],
            avoid=["smoke"],
            optional=True,
        )
        if not p:
            continue
        g = read_csv_auto(p).head(8)
        for _, r in g.iterrows():
            group_rows.append(
                {
                    "task": task,
                    "feature_group": r.feature_group,
                    "mean_abs_shap_sum": r.mean_abs_shap_all_folds_sum,
                }
            )

    groups = pd.DataFrame(group_rows)
    # Keep the feature-group legend local to panel (c), as in the earlier layout.

    if groups.empty:
        add_no_data(ax, "Missing SHAP group summaries")
    else:
        pivot = (
            groups.pivot_table(
                index="feature_group",
                columns="task",
                values="mean_abs_shap_sum",
                aggfunc="sum",
            )
            .fillna(0)
        )
        pivot = pivot.loc[
            pivot.sum(axis=1).sort_values(ascending=False).head(8).index
        ]

        # Absolute SHAP magnitudes are task-scale dependent.  Convert them to
        # within-task percentages before comparing feature-group composition.
        totals = pivot.sum(axis=0).replace(0, np.nan)
        pivot_pct = pivot.divide(totals, axis=1).fillna(0) * 100.0
        task_order = [
            t for t in ["LOI", "PHRR", "THR", "UL94_V0"]
            if t in pivot_pct.columns
        ]
        pivot_pct = pivot_pct[task_order]

        group_colors = {
            "molecular_descriptor": "#D65F5F",
            "ep_baseline": "#D4A94D",
            "molecular_fingerprint": "#67A88A",
            "formulation_loading": "#4C78A8",
            "elemental_composition": "#8E77B6",
            "formulation_interaction": "#B279A2",
            "curing_and_preparation": "#6FA3B8",
            "test_condition": "#94A3B8",
            "bde": "#59A14F",
            "interaction": "#B279A2",
            "other": "#E79A63",
        }
        group_display = {
            "molecular_descriptor": "molecular descriptor",
            "ep_baseline": "EP baseline",
            "molecular_fingerprint": "molecular fingerprint",
            "formulation_loading": "formulation/loading",
            "elemental_composition": "elemental composition",
            "formulation_interaction": "formulation interaction",
            "curing_and_preparation": "curing/preparation",
            "test_condition": "test condition",
            "bde": "BDE",
            "interaction": "interaction",
            "other": "other",
        }

        bottom = np.zeros(len(pivot_pct.columns))
        x = np.arange(len(pivot_pct.columns))
        for i, (group, row) in enumerate(pivot_pct.iterrows()):
            bars = ax.bar(
                x,
                row.values,
                bottom=bottom,
                label=group_display.get(str(group), str(group).replace("_", " ")),
                color=group_colors.get(str(group), categorical_color(i)),
                edgecolor="white",
                linewidth=0.45,
                width=0.78,
            )
            bottom += row.values

        ax.set_xticks(x, [task_label(t) for t in pivot_pct.columns])
        ax.set_ylabel("Aggregated mean absolute SHAP share (%)")
        ax.set_ylim(0, 100)
        ax.tick_params(axis="x", labelsize=8.0)
        clean_axes(ax, grid="y")
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=6.4,
            ncol=1,
            borderaxespad=0.0,
            handlelength=1.2,
            handletextpad=0.45,
            labelspacing=0.30,
        )

        out = groups.copy()
        out["aggregated_mean_abs_shap_share_percent"] = (
            out.groupby("task")["mean_abs_shap_sum"]
            .transform(lambda x: x / x.sum() * 100 if x.sum() else 0)
        )
        # Backward-compatible alias for any downstream audit script that still
        # expects the previous column name.
        out["contribution_percent_within_task"] = (
            out["aggregated_mean_abs_shap_share_percent"]
        )
        out.to_csv(
            output / "Fig8c_SHAP_feature_group_contributions.csv",
            index=False,
            encoding="utf-8-sig",
        )
    panel_label(ax, "(c)")

    # (d) Representative dependence plot for a stable continuous LOI feature.
    ax = axes_flat[3]
    task = "LOI"
    long_paths = _shap_long_paths(index, task)
    summary_path = _shap_summary_path(index, task)
    if long_paths and summary_path:
        long = pd.concat([read_csv_auto(p) for p in long_paths], ignore_index=True)
        s = read_csv_auto(summary_path)
        candidates = s[
            ~s.feature.astype(str).str.contains("Morgan|MACCS|bit", case=False, regex=True)
        ].sort_values("mean_abs_shap_all_folds", ascending=False)
        feature = next(
            (
                f
                for f in candidates.feature.astype(str)
                if f in set(long.feature.astype(str))
                and numeric(
                    long.loc[
                        long.feature.astype(str) == f,
                        "feature_value_raw",
                    ]
                ).nunique() > 2
            ),
            None,
        )
        if feature:
            sub = long[long.feature.astype(str) == feature]
            x = numeric(sub.feature_value_raw).to_numpy()
            y = numeric(sub.shap_value).to_numpy()
            mask = np.isfinite(x) & np.isfinite(y)
            ax.scatter(
                x[mask],
                y[mask],
                color=PALETTE["purple"],
                alpha=.52,
                s=18,
                edgecolor="white",
                linewidth=.2,
            )
            tx, ty = robust_lowess_like(x[mask], y[mask])
            if len(tx):
                ax.plot(tx, ty, color=PALETTE["red_dark"], lw=2)
            ax.axhline(0, ls="--", lw=.8, color=PALETTE["gray"])
            ax.set_xlabel(feature)
            ax.set_ylabel("SHAP value")
            clean_axes(ax, grid="y")
        else:
            add_no_data(ax, "No stable continuous raw feature found")
    else:
        # Existing results may contain only fold-aggregated SHAP values.  Use
        # UL-94 stable-feature importance as an explicit fallback rather than
        # fabricating sample-level dependence values.
        fallback = _shap_summary_path(index, "UL94_V0")
        if fallback:
            fs = (
                read_csv_auto(fallback)
                .sort_values("mean_abs_shap_all_folds", ascending=False)
                .head(15)
                .sort_values("mean_abs_shap_all_folds")
            )
            ax.barh(
                np.arange(len(fs)),
                fs.mean_abs_shap_all_folds,
                color=PALETTE["purple"],
            )
            ax.set_yticks(np.arange(len(fs)), fs.feature.astype(str))
            ax.set_xlabel("Mean |SHAP| across folds")
            ax.set_title("UL-94 stable features")
            ax.tick_params(axis="y", labelsize=7.0)
            clean_axes(ax, grid="x")
            ax.text(
                .02, .02,
                "Aggregate fallback; rerun SHAP for dependence plots",
                transform=ax.transAxes,
                fontsize=6.3,
            )
        else:
            add_no_data(ax, "Rerun SHAP with updated exporter for dependence data")
    panel_label(ax, "(d)")

    fig.tight_layout(
        pad=1.15,
        w_pad=1.45,
        h_pad=1.25,
    )
    save_figure(fig, output / "Fig8_SHAP_structure_property_relationships")

def _locate_ad(index: ResultIndex, task: str, suffix: str) -> Path | None:
    return index.locate(f"{task}_{suffix}.csv",prefer=["06_applicabilitydomain","applicability","FINAL_fixed_5x5",task],avoid=["smoke"],optional=True)


def fig9_applicability_domain(index: ResultIndex, output: Path) -> None:
    task="LOI"
    pred_path=_locate_ad(index,task,"outer_predictions_with_AD")
    zone_path=_locate_ad(index,task,"AD_zone_metrics")
    scaffold_path=_locate_ad(index,task,"AD_scaffold_metrics")
    sens_path=_locate_ad(index,task,"AD_threshold_sensitivity")
    fig,axes=plt.subplots(2,2,figsize=(10.8,8.3))

    # (a) Similarity versus absolute error, with the frozen 0.50/0.70 zones.
    if pred_path:
        pred=read_csv_auto(pred_path)
        if "absolute_error" not in pred.columns and {"y_true","y_pred"}.issubset(pred.columns):
            pred["absolute_error"]=(numeric(pred.y_true)-numeric(pred.y_pred)).abs()
        ax=axes[0,0]
        x=numeric(pred.get("AD_similarity",pd.Series(dtype=float))).to_numpy()
        y=numeric(pred.get("absolute_error",pd.Series(dtype=float))).to_numpy()
        mask=np.isfinite(x)&np.isfinite(y)
        zones=pred.get("AD_zone",pd.Series("Unknown",index=pred.index)).astype(str)
        for i,z in enumerate(["In_domain","Caution","Extrapolation","Invalid_or_missing"]):
            sel=mask & zones.eq(z).to_numpy()
            if sel.sum():
                ax.scatter(x[sel],y[sel],s=18,alpha=.55,color=ZONE_COLORS.get(z, PALETTE["gray"]),label=z.replace("_"," "))
        tx,ty=robust_lowess_like(x[mask],y[mask])
        if len(tx): ax.plot(tx,ty,color=PALETTE['red_dark'],lw=2,label="Binned median")
        ax.axvline(.5,ls="--",lw=.8,color=PALETTE['gray'])
        ax.axvline(.7,ls="--",lw=.8,color=PALETTE['gray'])
        ax.set_xlabel("Maximum Tanimoto similarity")
        ax.set_ylabel("Absolute error")
        ax.legend(frameon=False,fontsize=7)
        clean_axes(ax)
        pred.to_csv(output/f"Fig9_{task}_AD_predictions_used.csv",index=False,encoding="utf-8-sig")
    else:
        add_no_data(axes[0,0],"Run applicability-domain analysis")
    panel_label(axes[0,0],"(a)")

    # (b) Performance in the three applicability zones. V4 uses the generic
    # 'group' column rather than 'AD_zone' in its summary CSV.
    ax=axes[0,1]
    if zone_path:
        z=read_csv_auto(zone_path)
        zone_col=next((c for c in ["AD_zone","group","zone"] if c in z.columns),None)
        metric=next((c for c in ["MAE","Macro_F1","Accuracy"] if c in z.columns and numeric(z[c]).notna().any()),None)
        if zone_col and metric:
            order=[x for x in ["In_domain","Caution","Extrapolation","Invalid_or_missing"] if x in z[zone_col].astype(str).unique()]
            vals=[]; lows=[]; highs=[]; ns=[]
            for zone in order:
                sub=z[z[zone_col].astype(str)==zone]
                value=float(numeric(sub[metric]).mean())
                vals.append(value)
                low_col=f"{metric}_bootstrap_CI_low"; high_col=f"{metric}_bootstrap_CI_high"
                low=float(numeric(sub[low_col]).mean()) if low_col in sub and numeric(sub[low_col]).notna().any() else value
                high=float(numeric(sub[high_col]).mean()) if high_col in sub and numeric(sub[high_col]).notna().any() else value
                lows.append(max(0,value-low)); highs.append(max(0,high-value))
                ns.append(int(numeric(sub.get("n",pd.Series([0]))).sum()))
            yerr=np.vstack([lows,highs]) if any(v>0 for v in lows+highs) else None
            ax.bar(np.arange(len(order)),vals,yerr=yerr,color=[ZONE_COLORS.get(z, PALETTE["gray"]) for z in order],capsize=3)
            ax.set_xticks(np.arange(len(order)),[f"{x.replace('_',' ')}\n(n={n})" for x,n in zip(order,ns)],rotation=15,ha="right")
            ax.set_ylabel(metric.replace("_"," "))
            clean_axes(ax,grid="y")
        else:
            add_no_data(ax,"No supported AD-zone metric column")
    else:
        add_no_data(ax,"Missing AD zone metrics")
    panel_label(ax,"(b)")

    # (c) Seen versus novel Murcko scaffold performance. V4 again stores the
    # label in 'group'.
    ax=axes[1,0]
    if scaffold_path:
        sc=read_csv_auto(scaffold_path)
        group_col=next((c for c in ["scaffold_zone","scaffold_status","Scaffold_status","group"] if c in sc.columns),None)
        metric=next((c for c in ["MAE","Macro_F1","Accuracy"] if c in sc.columns and numeric(sc[c]).notna().any()),None)
        if group_col and metric:
            order=[x for x in ["Seen_scaffold","New_scaffold"] if x in sc[group_col].astype(str).unique()]
            if not order: order=sc[group_col].dropna().astype(str).drop_duplicates().tolist()
            vals=[]; lows=[]; highs=[]; ns=[]
            for group in order:
                sub=sc[sc[group_col].astype(str)==group]
                value=float(numeric(sub[metric]).mean()); vals.append(value)
                low_col=f"{metric}_bootstrap_CI_low"; high_col=f"{metric}_bootstrap_CI_high"
                low=float(numeric(sub[low_col]).mean()) if low_col in sub and numeric(sub[low_col]).notna().any() else value
                high=float(numeric(sub[high_col]).mean()) if high_col in sub and numeric(sub[high_col]).notna().any() else value
                lows.append(max(0,value-low)); highs.append(max(0,high-value)); ns.append(int(numeric(sub.get("n",pd.Series([0]))).sum()))
            yerr=np.vstack([lows,highs]) if any(v>0 for v in lows+highs) else None
            ax.bar(np.arange(len(order)),vals,yerr=yerr,color=[PALETTE['blue'],PALETTE['orange']][:len(order)],capsize=3)
            ax.set_xticks(np.arange(len(order)),[f"{x.replace('_',' ')}\n(n={n})" for x,n in zip(order,ns)])
            ax.set_ylabel(metric); clean_axes(ax,grid="y")
        else:
            add_no_data(ax,f"Inspect {scaffold_path.name}; columns={list(sc.columns)[:6]}")
    elif pred_path:
        pred=read_csv_auto(pred_path)
        group_col=next((c for c in ["scaffold_novelty","scaffold_status"] if c in pred.columns),None)
        if group_col and "absolute_error" in pred:
            g=pred.groupby(group_col).absolute_error.mean()
            ax.bar(g.index.astype(str),g.values,color=[PALETTE['blue'],PALETTE['orange']][:len(g)])
            ax.set_ylabel("MAE"); clean_axes(ax,grid="y")
        else:
            add_no_data(ax,"Seen/new scaffold field missing")
    else:
        add_no_data(ax,"Missing scaffold AD results")
    panel_label(ax,"(c)")

    # (d) Threshold sensitivity at the prespecified caution threshold closest
    # to 0.50. Each line is an AD zone; this avoids connecting the three zone
    # rows as if they were repeated measurements of one series.
    ax=axes[1,1]
    if sens_path:
        ss=read_csv_auto(sens_path)
        xcol=next((c for c in ["reliable_threshold","threshold"] if c in ss.columns),None)
        group_col=next((c for c in ["AD_zone_temp","group","AD_zone","zone"] if c in ss.columns),None)
        metric=next((c for c in ["MAE","Macro_F1","Accuracy"] if c in ss.columns and numeric(ss[c]).notna().any()),None)
        if xcol and group_col and metric:
            if "caution_threshold" in ss.columns and numeric(ss.caution_threshold).notna().any():
                available=np.sort(numeric(ss.caution_threshold).dropna().unique())
                chosen=float(available[np.argmin(np.abs(available-.50))])
                plot=ss[np.isclose(numeric(ss.caution_threshold),chosen)].copy()
            else:
                chosen=np.nan; plot=ss.copy()
            for i,zone in enumerate([z for z in ["In_domain","Caution","Extrapolation"] if z in plot[group_col].astype(str).unique()]):
                sub=plot[plot[group_col].astype(str)==zone].sort_values(xcol)
                ax.plot(numeric(sub[xcol]),numeric(sub[metric]),marker="o",color=ZONE_COLORS.get(zone, categorical_color(i)),label=zone.replace("_"," "),markerfacecolor="white",markeredgewidth=.9)
            ax.set_xlabel("Reliable-domain threshold")
            ax.set_ylabel(metric)
            title="Threshold sensitivity" if not np.isfinite(chosen) else f"Caution threshold fixed at {chosen:.2f}"
            ax.set_title(title,fontsize=9)
            ax.legend(frameon=False,fontsize=7)
            clean_axes(ax,grid="y")
        else:
            add_no_data(ax,"Threshold sensitivity columns not recognised")
    else:
        add_no_data(ax,"Missing threshold-sensitivity results")
    panel_label(ax,"(d)")
    fig.suptitle("Applicability domain and error analysis (LOI)",y=1.01,fontsize=11)
    fig.tight_layout(pad=1.15, w_pad=1.0, h_pad=1.0)
    save_figure(fig,output/"Fig9_applicability_domain_and_error_analysis")

def fig10_virtual_screening(index: ResultIndex, output: Path) -> None:
    """Combined designed + PubChem virtual-screening summary.

    The manuscript-facing Fig. 10 always prefers the audited combined 50 kW/m²
    screening outputs.  Historical designed-only folders are retained only as
    a backward-compatible fallback.
    """
    candidate_path = index.locate(
        "candidate_best_per_molecule.csv",
        prefer=["06_reversedesign", "combined_flux50"],
        avoid=["final_flux50", "sensitivity_flux35", "smoke", "2x2"],
        optional=True,
        label="Fig10 combined eligible candidates",
    )
    if not candidate_path:
        candidate_path = index.locate_any(
            ["candidate_predictions.csv", "ranked_candidates.csv", "pareto_candidates.csv", "candidate_ranking.csv"],
            prefer=["06_reversedesign", "candidate", "pareto"],
            avoid=["sensitivity_flux35", "smoke", "2x2"],
            optional=True,
            label="Fig10 legacy candidate fallback",
        )
    if not candidate_path:
        raise FileNotFoundError("Virtual-screening predictions are not yet available; Fig. 10 is intentionally skipped")

    df = read_csv_auto(candidate_path)

    all_path = index.locate(
        "candidate_predictions_all_formulations.csv",
        prefer=["06_reversedesign", "combined_flux50"],
        avoid=["final_flux50", "sensitivity_flux35", "smoke", "2x2"],
        optional=True,
        label="Fig10 combined all formulations",
    )
    all_df = read_csv_auto(all_path) if all_path else df

    final_path = index.locate(
        "final_priority_candidates_combined.csv",
        prefer=["06_reversedesign", "final_priority_combined"],
        avoid=["final_priority/", "smoke", "2x2"],
        optional=True,
        label="Fig10 combined final priority",
    )
    final = read_csv_auto(final_path) if final_path else pd.DataFrame()

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))

    # (a) Chemical-space panel with separate Training / Designed / PubChem classes.
    space_path = index.locate(
        "candidate_chemical_space.csv",
        prefer=["06_reversedesign", "combined_flux50"],
        avoid=["final_flux50", "sensitivity_flux35"],
        optional=True,
        label="Fig10 combined chemical space",
    )
    space = read_csv_auto(space_path) if space_path else df.copy()
    xcol = next((c for c in ["PC1", "UMAP1", "PCA1"] if c in space.columns), None)
    ycol = next((c for c in ["PC2", "UMAP2", "PCA2"] if c in space.columns), None)
    if xcol and ycol:
        source_map = (
            all_df[["Candidate_ID", "Source_Type"]]
            .dropna(subset=["Candidate_ID"])
            .drop_duplicates("Candidate_ID")
            if {"Candidate_ID", "Source_Type"}.issubset(all_df.columns)
            else pd.DataFrame(columns=["Candidate_ID", "Source_Type"])
        )
        if "Candidate_ID" in space.columns and not source_map.empty:
            space = space.merge(source_map, on="Candidate_ID", how="left", suffixes=("", "_source"))
        status = space.get("candidate_status", pd.Series("Candidate", index=space.index)).fillna("Candidate").astype(str)
        source = space.get("Source_Type", pd.Series("", index=space.index)).fillna("").astype(str).str.lower()
        display = pd.Series("Candidate", index=space.index, dtype=object)
        display.loc[status.str.lower().eq("training")] = "Training"
        display.loc[~status.str.lower().eq("training") & source.eq("designed")] = "Designed"
        display.loc[~status.str.lower().eq("training") & source.eq("pubchem")] = "PubChem"
        colors = {"Training": PALETTE["gray"], "Designed": PALETTE["purple"], "PubChem": PALETTE["green"]}
        for label in [x for x in ["Training", "Designed", "PubChem", "Candidate"] if x in display.unique()]:
            mask = display.eq(label)
            axes[0].scatter(
                numeric(space.loc[mask, xcol]), numeric(space.loc[mask, ycol]),
                s=23, color=colors.get(label, PALETTE["blue"]), alpha=.62,
                label=label, edgecolor="white", linewidth=.25,
            )
        axes[0].set_xlabel(xcol)
        axes[0].set_ylabel(ycol)
        axes[0].legend(frameon=False, loc="best")
        clean_axes(axes[0])
        space.assign(Display_group=display).to_csv(output / "Fig10_chemical_space_data_used.csv", index=False, encoding="utf-8-sig")
    else:
        add_no_data(axes[0], "Run combined candidate chemical-space mapping")

    # (b) Predicted LOI-PHRR plane, separated by candidate source.
    x = next((c for c in ["LOI_pred", "pred_LOI", "LOI"] if c in df.columns), None)
    y = next((c for c in ["PHRR_pred", "pred_PHRR", "PHRR"] if c in df.columns), None)
    if x and y:
        source = df.get("Source_Type", pd.Series("candidate", index=df.index)).fillna("candidate").astype(str).str.lower()
        source_colors = {"designed": PALETTE["purple"], "pubchem": PALETTE["green"], "candidate": PALETTE["pink"]}
        for label in source.unique():
            mask = source.eq(label)
            axes[1].scatter(
                numeric(df.loc[mask, x]), numeric(df.loc[mask, y]), s=24,
                color=source_colors.get(label, PALETTE["pink"]), alpha=.62,
                label=label.capitalize(), edgecolor="white", linewidth=.25,
            )
        if not final.empty and "Candidate_ID" in final.columns and "Candidate_ID" in df.columns:
            top_ids = set(final.head(20)["Candidate_ID"].astype(str))
            top_mask = df["Candidate_ID"].astype(str).isin(top_ids)
            axes[1].scatter(
                numeric(df.loc[top_mask, x]), numeric(df.loc[top_mask, y]),
                s=58, facecolors="none", edgecolors=PALETTE["black"], linewidths=.8,
                label="Final priority",
            )
        axes[1].set_xlabel("Predicted LOI")
        axes[1].set_ylabel("Predicted PHRR (kW m$^{-2}$)")
        axes[1].legend(frameon=False, loc="best")
        clean_axes(axes[1])
    else:
        add_no_data(axes[1], "Predicted LOI/PHRR columns missing")

    # (c) Standardised heatmap of final combined priority candidates.
    if not final.empty:
        preferred = [
            "LOI_pred_50", "V0_probability_50", "PHRR_pred_50",
            "THR_pred_50", "Tg_pred_50", "TS_pred_50",
        ]
        metric_cols = [c for c in preferred if c in final.columns]
        top = final.head(min(15, len(final))).copy()
        labels = top.get("Candidate_Name", top.get("Candidate_ID", pd.Series(index=top.index, dtype=object))).astype(str)
    else:
        metric_cols = [c for c in ["LOI_pred", "V0_probability", "PHRR_pred", "THR_pred", "Tg_pred", "TS_pred"] if c in df.columns]
        top = df.head(min(15, len(df))).copy()
        labels = top.get("Candidate_Name", top.get("Candidate_ID", pd.Series(index=top.index, dtype=object))).astype(str)

    if metric_cols:
        arr = top[metric_cols].apply(pd.to_numeric, errors="coerce")
        arr = (arr - arr.mean()) / arr.std(ddof=0).replace(0, np.nan)
        im = axes[2].imshow(arr.fillna(0).to_numpy(), aspect="auto", cmap=DIVERGING_CMAP)
        xlabels = [c.replace("_pred_50", "").replace("_50", "").replace("V0_probability", "V-0 prob.") for c in metric_cols]
        axes[2].set_xticks(range(len(metric_cols)), xlabels, rotation=45, ha="right")
        axes[2].set_yticks(range(len(top)), [s[:20] for s in labels])
        plt.colorbar(im, ax=axes[2], fraction=.046, label="Standardised value")
        if not final.empty:
            top.to_csv(output / "Fig10_final_priority_data_used.csv", index=False, encoding="utf-8-sig")
    else:
        add_no_data(axes[2], "Candidate multi-objective metrics missing")

    for i, ax in enumerate(axes):
        panel_label(ax, f"({chr(97+i)})")
    fig.tight_layout(pad=1.15, w_pad=1.0, h_pad=1.0)
    save_figure(fig, output / "Fig10_virtual_screening")
    df.to_csv(output / "Fig10_candidate_data_used.csv", index=False, encoding="utf-8-sig")


def _draw_molecule_matplotlib(ax: plt.Axes, smiles: str) -> bool:
    """Draw a 2D chemical structure without rdMolDraw2D/Chem.Draw.

    This intentionally relies only on RDKit core molecule parsing plus the
    rdDepictor coordinate generator, then renders atoms/bonds with Matplotlib.
    It is therefore compatible with Windows environments where rdMolDraw2D.dll
    cannot be loaded but the core RDKit chemistry modules work normally.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return False

    # Generate 2D coordinates when possible. If the RDKit coordinate generator
    # is unavailable, retain a deterministic circular graph layout as a last
    # fallback so figure generation never depends on rdMolDraw2D.
    coords = None
    try:
        from rdkit.Chem import rdDepictor
        rdDepictor.Compute2DCoords(mol)
        conf = mol.GetConformer()
        coords = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                           for i in range(mol.GetNumAtoms())], dtype=float)
    except Exception:
        n_atoms = max(1, mol.GetNumAtoms())
        theta = np.linspace(0, 2 * np.pi, n_atoms, endpoint=False)
        coords = np.column_stack([np.cos(theta), np.sin(theta)])

    if coords.size == 0:
        return False

    # Centre and scale while retaining molecular aspect ratio.
    coords = coords - np.nanmean(coords, axis=0, keepdims=True)
    span = np.nanmax(coords, axis=0) - np.nanmin(coords, axis=0)
    scale = float(max(span.max(), 1e-8))
    coords = coords / scale

    def _offset_segment(p1: np.ndarray, p2: np.ndarray, offset: float) -> tuple[np.ndarray, np.ndarray]:
        vec = p2 - p1
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-12:
            return p1, p2
        normal = np.array([-vec[1], vec[0]]) / norm
        return p1 + offset * normal, p2 + offset * normal

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx(); j = bond.GetEndAtomIdx()
        p1, p2 = coords[i], coords[j]
        btype = bond.GetBondType()
        if btype == Chem.BondType.DOUBLE:
            for off in (-0.010, 0.010):
                q1, q2 = _offset_segment(p1, p2, off)
                ax.plot([q1[0], q2[0]], [q1[1], q2[1]], color="black", lw=0.80, solid_capstyle="round")
        elif btype == Chem.BondType.TRIPLE:
            for off in (-0.014, 0.0, 0.014):
                q1, q2 = _offset_segment(p1, p2, off)
                ax.plot([q1[0], q2[0]], [q1[1], q2[1]], color="black", lw=0.72, solid_capstyle="round")
        elif btype == Chem.BondType.AROMATIC:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="black", lw=0.78, solid_capstyle="round")
        else:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="black", lw=0.88, solid_capstyle="round")

    # Carbon atoms remain implicit; heteroatoms are labelled in conventional
    # element notation. Charges are appended when present.
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol == "C":
            continue
        idx = atom.GetIdx(); x, y = coords[idx]
        charge = atom.GetFormalCharge()
        charge_txt = ""
        if charge > 0:
            charge_txt = "+" if charge == 1 else f"{charge}+"
        elif charge < 0:
            charge_txt = "−" if charge == -1 else f"{abs(charge)}−"
        label = f"{symbol}{charge_txt}"
        ax.text(x, y, label, ha="center", va="center", fontsize=6.1,
                bbox=dict(boxstyle="round,pad=0.08", facecolor="white", edgecolor="none", alpha=0.95),
                zorder=5)

    xmin, ymin = np.nanmin(coords, axis=0); xmax, ymax = np.nanmax(coords, axis=0)
    padx = max(0.06, (xmax - xmin) * 0.08)
    pady = max(0.06, (ymax - ymin) * 0.08)
    ax.set_xlim(xmin - padx, xmax + padx)
    ax.set_ylim(ymin - pady, ymax + pady)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    return True


def fig11_priority_candidate_structures(index: ResultIndex, output: Path) -> None:
    """Draw structures of the final Top-10 cross-flux priority candidates."""
    import textwrap

    final_path = index.locate(
        "final_priority_candidates_combined.csv",
        prefer=["06_reversedesign", "final_priority_combined"],
        avoid=["final_priority/", "smoke", "2x2"],
        optional=True,
        label="Fig11 combined final priority",
    )
    if not final_path:
        raise FileNotFoundError("final_priority_candidates_combined.csv is required for Fig11")

    final = read_csv_auto(final_path)
    rank_col = next((c for c in ["Final_priority_rank", "Overall_rank", "Candidate_rank", "Rank"] if c in final.columns), None)
    if rank_col:
        final = final.sort_values(rank_col, kind="stable")
    top = final.head(min(10, len(final))).copy()

    # The compact cross-flux ranking table intentionally does not contain the
    # full SMILES. Recover them by Candidate_ID from the audited 50 kW/m2
    # candidate table, which uses exactly the same candidate identities.
    smiles_col = next((c for c in ["Candidate_Canonical_SMILES", "SMILES_main", "pred_Canonical_SMILES"] if c in top.columns), None)
    structure_source = final_path
    if smiles_col is None and "Candidate_ID" in top.columns:
        source_path = index.locate(
            "candidate_best_per_molecule.csv",
            prefer=["06_reversedesign", "combined_flux50"],
            avoid=["final_flux50", "sensitivity_flux35", "smoke", "2x2"],
            optional=True,
            label="Fig11 candidate structure lookup",
        )
        if source_path:
            source = read_csv_auto(source_path)
            source_smiles = next((c for c in ["Candidate_Canonical_SMILES", "SMILES_main", "pred_Canonical_SMILES"] if c in source.columns), None)
            if source_smiles and "Candidate_ID" in source.columns:
                lookup = source[["Candidate_ID", source_smiles]].dropna(subset=["Candidate_ID"]).drop_duplicates("Candidate_ID")
                top = top.merge(lookup, on="Candidate_ID", how="left")
                smiles_col = source_smiles
                structure_source = source_path
    if smiles_col is None:
        raise KeyError("Could not recover candidate SMILES for Fig11")

    n = len(top)
    ncols = 5 if n > 5 else max(1, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.55, nrows * 3.85))
    axes = np.atleast_1d(axes).ravel()
    rows = []

    for ax in axes:
        ax.axis("off")

    for ax, (_, row) in zip(axes, top.iterrows()):
        raw_smiles = row.get(smiles_col, "")
        canonical = canonical_smiles(raw_smiles) if pd.notna(raw_smiles) else None
        ok = _draw_molecule_matplotlib(ax, canonical or raw_smiles) if (canonical or raw_smiles) else False
        if not ok:
            add_no_data(ax, "Structure unavailable")

        rank_val = row.get(rank_col, len(rows) + 1) if rank_col else len(rows) + 1
        try:
            rank_txt = str(int(float(rank_val)))
        except Exception:
            rank_txt = str(rank_val)
        name = str(row.get("Candidate_Name", row.get("Candidate_ID", f"Candidate {rank_txt}")))
        family = str(row.get("Design_Family", "")).strip()
        title = f"#{rank_txt} " + textwrap.fill(name, width=18)
        if family and family.lower() not in {"nan", "none"}:
            title += f"\n{family}"
        ax.set_title(title, fontsize=7.0, pad=8.0)
        rows.append({
            "rank": rank_txt,
            "candidate_id": row.get("Candidate_ID", ""),
            "candidate_name": name,
            "design_family": family,
            "smiles_used": canonical or raw_smiles,
            "ranking_source_file": str(final_path),
            "structure_source_file": str(structure_source),
            "drawing_backend": "Matplotlib + RDKit core (no rdMolDraw2D)",
        })

    for ax in axes[len(top):]:
        ax.axis("off")

    fig.tight_layout(pad=1.1, w_pad=0.9, h_pad=1.45)
    save_figure(fig, output / "Fig11_top10_priority_candidate_structures")
    pd.DataFrame(rows).to_csv(output / "Fig11_top10_priority_candidate_structures.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    apply_main_style()
    args=parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams["savefig.dpi"] = args.dpi
    plt.rcParams["figure.dpi"] = min(max(220, args.dpi // 2), 450)
    df=read_csv_auto(args.data)
    index=ResultIndex(args.results_root)
    manifest=[]
    jobs=[
        ("Fig2",fig2_dataset_composition,(df,args.output)),
        ("Fig3",fig3_outer_predictions,(index,args.output)),
        ("Fig4",fig4_stability_and_split,(index,args.output)),
        ("Fig5",fig5_ul94,(index,args.output)),
        ("Fig6",fig6_information_source,(index,args.output)),
        ("Fig7",fig7_bde,(index,args.output)),
        ("Fig8",fig8_shap,(index,args.output)),
        ("Fig9",fig9_applicability_domain,(index,args.output)),
        ("Fig10",fig10_virtual_screening,(index,args.output)),
        ("Fig11",fig11_priority_candidate_structures,(index,args.output)),
    ]
    for name,func,func_args in jobs:
        try:
            func(*func_args); manifest.append({"item":name,"status":"generated","error":""}); print(f"[OK] {name}")
        except Exception as exc:
            manifest.append({"item":name,"status":"skipped","error":f"{type(exc).__name__}: {exc}"}); print(f"[SKIP] {name}: {exc}")
    pd.DataFrame(manifest).to_csv(args.output/"main_figure_manifest.csv",index=False,encoding="utf-8-sig")
    index.save_manifest(args.output/"selected_input_files.csv")
    write_json(args.output/"run_config.json",vars(args))
    print(f"\n[DONE] Main figures: {args.output}")


if __name__=="__main__":
    main()
