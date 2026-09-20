# -*- coding: utf-8 -*-
"""Generate a compact supplementary-figure set from frozen V5 results.

This compact mode keeps the analyses needed to support the manuscript but
reduces redundancy in the Supplementary Information. S1 remains a manual
literature-screening flowchart. The code-generated SI figures are streamlined
to S2-S11 so that repository outputs match the frozen Supplementary Information exactly.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.warning")
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve, roc_curve

from paper_utils import (
    ALL_REGRESSION_TASKS,
    BLUE_CMAP,
    DIVERGING_CMAP,
    GREEN_CMAP,
    ORANGE_CMAP,
    PURPLE_CMAP,
    PALETTE,
    SPLIT_COLORS,
    ROOT,
    SERIES_COLORS,
    TARGET_COLUMNS,
    ResultIndex,
    add_no_data,
    apply_supplementary_style,
    categorical_color,
    clean_axes,
    jitter,
    locate_task_file,
    metric_box,
    morgan_matrix,
    murcko_scaffold,
    normalise_ul94,
    numeric,
    panel_label,
    probability_column,
    read_csv_auto,
    regression_metrics,
    robust_lowess_like,
    save_figure,
    target_axis_label,
    task_color,
    task_label,
    unique_main_molecule_table,
    write_json,
)
from make_main_figures import (
    _aggregate_bde_predictions,
    _beeswarm,
    _locate_ad,
    _shap_long_paths,
    _shap_summary_path,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "data" / "DOPO_EP_new_with_BDE.csv")
    p.add_argument("--bde-data", type=Path, default=ROOT / "07_BDE" / "data" / "BDE.csv")
    p.add_argument("--results-root", type=Path, default=ROOT / "results")
    p.add_argument("--output", type=Path, default=ROOT / "results" / "09_PaperFigures" / "supplementary")
    p.add_argument("--dpi", type=int, default=900)
    return p.parse_args()


def fig_s2_missingness(df: pd.DataFrame, out: Path) -> None:
    fields = [
        ("LOI", "LOI"), ("LOI thickness", "LOI_Thickness_mm"), ("UL-94", "UL94"),
        ("UL-94 thickness", "UL94_Thickness_mm"), ("PHRR", "PHRR_kw_㎡"),
        ("THR", "THR_MJ_㎡"), ("Cone flux", "Cone_flux_kW_m2"),
        ("Cone thickness", "Cone_Thickness_mm"), ("Tg", "Tg_℃"),
        ("Char yield", "Char_yield_％_700C"), ("TS", "TS_MPa"), ("FS", "FS_MPa"),
        ("Cure temperature", "Cure_Temp_Max"), ("EP baseline LOI", "EP_matrix_LOI"),
        ("EP baseline PHRR", "EP_matrix_PHRR"), ("EP baseline THR", "EP_matrix_THR"),
        ("EP baseline Tg", "EP_matrix_Tg"), ("EP baseline char", "EP_matrix_CY"),
        ("EP baseline TS", "EP_matrix_TS"), ("EP baseline FS", "EP_matrix_FS"),
    ]
    rows = [{"field": label, "column": col, "missing_rate_percent": float(df[col].isna().mean()*100)}
            for label, col in fields if col in df.columns]
    tab = pd.DataFrame(rows).sort_values("missing_rate_percent")
    fig, ax = plt.subplots(figsize=(8.0, max(5.0, 0.30*len(tab)+1.5)))
    ax.barh(np.arange(len(tab)), tab.missing_rate_percent, color=PALETTE["orange"], alpha=0.88)
    ax.set_yticks(np.arange(len(tab)), tab.field)
    ax.set_xlabel("Missing rate (%)"); ax.set_xlim(0, 105); clean_axes(ax, grid="x")
    for i, v in enumerate(tab.missing_rate_percent): ax.text(v+1, i, f"{v:.1f}%", va="center", fontsize=7)
    save_figure(fig, out/"FigS2_key_field_missingness")
    tab.to_csv(out/"FigS2_missingness_data.csv", index=False, encoding="utf-8-sig")


def fig_s3_target_distributions(df: pd.DataFrame, out: Path) -> None:
    tasks = ["LOI","PHRR","THR","Tg","Char_yield","TS_MPa","FS_MPa","UL94_V0"]
    fig, axes = plt.subplots(2,4,figsize=(14.5,7.0))
    rows=[]
    for i,(ax,task) in enumerate(zip(axes.ravel(),tasks)):
        col=TARGET_COLUMNS[task]
        if col not in df: add_no_data(ax,f"Missing {col}"); continue
        if task=="UL94_V0":
            labels=df[col].fillna("Missing").astype(str).value_counts()
            ax.bar(np.arange(len(labels)),labels.values,color=task_color(task),alpha=.9)
            ax.set_xticks(np.arange(len(labels)),labels.index,rotation=35,ha="right")
            ax.set_ylabel("Count")
            for j,v in enumerate(labels.values): ax.text(j,v+max(labels.values)*.02,str(int(v)),ha="center",fontsize=7)
            rows.append({"task":task,"n":int(df[col].notna().sum()),"median":np.nan,"min":np.nan,"max":np.nan})
        else:
            v=numeric(df[col]).dropna()
            ax.hist(v,bins="auto",color=task_color(task),edgecolor="white",linewidth=.5,alpha=.9)
            ax.axvline(v.median(),ls="--",lw=1,color=PALETTE['dark'])
            ax.text(.04,.96,f"n={len(v)}\nMedian={v.median():.2f}\nRange={v.min():.2f}–{v.max():.2f}",transform=ax.transAxes,va="top",fontsize=7)
            ax.set_xlabel(target_axis_label(task)); ax.set_ylabel("Frequency")
            rows.append({"task":task,"n":len(v),"median":v.median(),"min":v.min(),"max":v.max()})
        clean_axes(ax); panel_label(ax,f"({chr(97+i)})")
    fig.tight_layout(pad=.9,w_pad=.75,h_pad=.75); save_figure(fig,out/"FigS3_all_absolute_target_distributions")
    pd.DataFrame(rows).to_csv(out/"FigS3_distribution_summary.csv",index=False,encoding="utf-8-sig")


def fig_s4_delta(df: pd.DataFrame, out: Path) -> None:
    tasks=["Delta_LOI","Delta_PHRR","Delta_THR","Delta_CY"]
    fig,axes=plt.subplots(2,2,figsize=(10.5,7.4)); rows=[]
    for i,(ax,task) in enumerate(zip(axes.ravel(),tasks)):
        col=TARGET_COLUMNS[task]
        if col not in df: add_no_data(ax,f"Missing {col}");continue
        v=numeric(df[col]).dropna(); ax.hist(v,bins="auto",color=task_color(task),edgecolor="white",linewidth=.5,alpha=.9)
        ax.axvline(0,ls="--",lw=1,color=PALETTE['dark'])
        pos=int((v>0).sum()); neg=int((v<0).sum()); zero=int((v==0).sum())
        ax.text(.04,.96,f"n={len(v)}\nPositive={pos}\nNegative={neg}\nZero={zero}",transform=ax.transAxes,va="top",fontsize=8)
        ax.set_xlabel(target_axis_label(task)); ax.set_ylabel("Frequency");clean_axes(ax);panel_label(ax,f"({chr(97+i)})")
        baseline=next((c for c in ["EP_matrix_LOI","EP_matrix_PHRR","EP_matrix_THR","EP_matrix_CY"] if c.endswith(task.replace("Delta_","")) and c in df),None)
        rows.append({"task":task,"n":len(v),"positive":pos,"negative":neg,"zero":zero,"baseline_matched_n":int(df.loc[v.index,baseline].notna().sum()) if baseline else np.nan})
    fig.tight_layout(pad=.9,w_pad=.75,h_pad=.75);save_figure(fig,out/"FigS4_delta_distributions_and_baseline_matching")
    pd.DataFrame(rows).to_csv(out/"FigS4_delta_summary.csv",index=False,encoding="utf-8-sig")


def _unique_molecules(df: pd.DataFrame) -> tuple[pd.DataFrame,np.ndarray]:
    tab=unique_main_molecule_table(df)
    matrix,valid=morgan_matrix(tab.SMILES_main,radius=2,nbits=1024)
    return tab.loc[np.asarray(valid)].reset_index(drop=True),matrix


def fig_s5_chemical_space(df: pd.DataFrame, out: Path) -> None:
    tab,matrix=_unique_molecules(df)
    if len(tab)<3: raise ValueError("Too few valid unique molecules")
    pca=PCA(n_components=min(20,matrix.shape[0]-1,matrix.shape[1]),random_state=42).fit_transform(matrix)
    coords=pca[:,:2]
    try:
        import umap  # type: ignore
        um=umap.UMAP(n_components=2,random_state=42,n_neighbors=min(15,max(3,len(tab)//8)),min_dist=.15).fit_transform(pca)
        method="UMAP"
    except Exception:
        um=coords.copy(); method="PCA fallback (install umap-learn for UMAP)"
    tab["PCA1"],tab["PCA2"],tab["UMAP1"],tab["UMAP2"]=coords[:,0],coords[:,1],um[:,0],um[:,1]
    fig,axes=plt.subplots(2,2,figsize=(11.5,9.2))
    axes[0,0].scatter(tab.PCA1,tab.PCA2,s=25,color=PALETTE['blue'],alpha=.7);axes[0,0].set_xlabel("PC1");axes[0,0].set_ylabel("PC2");axes[0,0].set_title("Unique main flame retardants")
    axes[0,1].scatter(tab.UMAP1,tab.UMAP2,s=25,color=PALETTE['purple'],alpha=.7);axes[0,1].set_xlabel("UMAP1");axes[0,1].set_ylabel("UMAP2");axes[0,1].set_title(method)
    prep_display = {
        "DOPO-based (additive)": "Additive",
        "DOPO-based (reactive)": "Reactive",
        "DOPO-based (Co-curing)": "Co-curing",
        "DOPO-based (additive+ Secondary Crosslinking)": "Additive + secondary crosslinking",
        "DOPO-based (Additive + Secondary Crosslinking)": "Additive + secondary crosslinking",
    }
    for ax,x,y,column,title in [(axes[1,0],"UMAP1","UMAP2","Synergy_type","Synergy type"),(axes[1,1],"UMAP1","UMAP2","Preparation_Method","Preparation method")]:
        cats=tab.get(column,pd.Series("Unknown",index=tab.index)).fillna("Unknown").astype(str)
        if column == "Preparation_Method":
            cats = cats.map(prep_display).fillna(cats)
        top=cats.value_counts().head(7).index;cats=cats.where(cats.isin(top),"Other")
        for i,c in enumerate(cats.value_counts().index):
            s=cats.eq(c);ax.scatter(tab.loc[s,x],tab.loc[s,y],s=25,color=categorical_color(i),alpha=.72,label=c)
        ax.set_xlabel(x);ax.set_ylabel(y);ax.set_title(title);ax.legend(frameon=False,bbox_to_anchor=(0.5,-0.18),loc="upper center",fontsize=6.3,ncol=2)
    for i,ax in enumerate(axes.ravel()):clean_axes(ax);panel_label(ax,f"({chr(97+i)})")
    fig.tight_layout(pad=.9,w_pad=.75,h_pad=.75);save_figure(fig,out/"FigS5_complete_chemical_space")
    tab.to_csv(out/"FigS5_chemical_space_coordinates.csv",index=False,encoding="utf-8-sig")


def fig_s6_scaffolds(df: pd.DataFrame, out: Path) -> None:
    tab=df[["SMILES_main"]].dropna().copy();tab["scaffold"]=tab.SMILES_main.map(murcko_scaffold)
    counts=tab.scaffold.dropna().value_counts();top=counts.head(20).sort_values();cum=counts.cumsum()/counts.sum()*100
    fig,axes=plt.subplots(1,2,figsize=(12.5,5.4))
    axes[0].barh(np.arange(len(top)),top.values,color=PALETTE['blue']);axes[0].set_yticks(np.arange(len(top)),[f"Scaffold {counts.index.get_loc(x)+1}" for x in top.index]);axes[0].set_xlabel("Experimental records")
    axes[1].plot(np.arange(1,len(cum)+1),cum.values,color=PALETTE['red'],lw=2);axes[1].axhline(80,ls="--",color=PALETTE['gray']);axes[1].set_xlabel("Scaffolds ranked by frequency");axes[1].set_ylabel("Cumulative record coverage (%)")
    for i,ax in enumerate(axes):clean_axes(ax,grid="x" if i==0 else "y");panel_label(ax,f"({chr(97+i)})")
    fig.tight_layout(pad=.9,w_pad=.75,h_pad=.75);save_figure(fig,out/"FigS6_scaffold_frequency_and_coverage")
    pd.DataFrame({"scaffold_rank":np.arange(1,len(counts)+1),"scaffold_smiles":counts.index,"record_count":counts.values,"cumulative_coverage_percent":cum.values}).to_csv(out/"FigS6_scaffold_frequency_table.csv",index=False,encoding="utf-8-sig")


def _seed_from_path(path: Path) -> int | None:
    match=re.search(r"(?:repeat_)?seed_(\d+)",path.as_posix(),re.I)
    return int(match.group(1)) if match else None


def _development_candidates(index: ResultIndex, task: str) -> pd.DataFrame:
    """Collect five-seed development model comparisons from results/main.

    The development workflow froze one feature view and K value per task.  The
    figure therefore compares candidate models across the five grouped-holdout
    seeds and annotates the frozen view/K instead of pretending that the strict
    nested inner-candidate table is a development view search.
    """
    files=[p for p in index.csv_files
           if p.name==f"{task}_model_comparison.csv"
           and "/results/main/" in ("/"+p.as_posix().lower())
           and "without_bde" in p.as_posix().lower()]
    blocks=[]
    for path in files:
        try:
            frame=read_csv_auto(path)
        except Exception:
            continue
        frame["seed"]=_seed_from_path(path)
        frame["source_file"]=str(path)
        blocks.append(frame)
    return pd.concat(blocks,ignore_index=True,sort=False) if blocks else pd.DataFrame()


def fig_s4_frozen_configs(index: ResultIndex, out: Path) -> None:
    """Final S4: frozen task configurations and model-selection frequencies.

    This figure intentionally reads only the five FINAL outer-fold metric files.
    It does not fall back to historical development-stage view/K scans, because
    the frozen Supplementary Information caption refers specifically to model
    selection frequencies across the FINAL outer folds.
    """
    tasks = ["LOI", "PHRR", "THR", "UL94_V0", "Tg", "TS_MPa"]
    config = json.loads((ROOT / "config" / "task_config.json").read_text(encoding="utf-8"))["tasks"]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2))
    rows = []

    for ax, task in zip(axes.ravel(), tasks):
        metrics_path = locate_task_file(index, task, "outer_fold_metrics", use_bde=False, optional=True)
        if not metrics_path:
            add_no_data(ax, f"Missing FINAL fold metrics for {task}")
            continue

        folds = read_csv_auto(metrics_path)
        selected_col = next((c for c in ["selected_model", "model_name", "model"] if c in folds.columns), None)
        if not selected_col:
            add_no_data(ax, "Selected-model column not recognised")
            continue

        view = str(config.get(task, {}).get("view", ""))
        kval = str(config.get(task, {}).get("k", "ALL"))
        if "selected_view" in folds and folds["selected_view"].notna().any():
            view = str(folds["selected_view"].astype(str).mode().iloc[0])
        if "selected_requested_k" in folds and folds["selected_requested_k"].notna().any():
            kval_raw = folds["selected_requested_k"].mode().iloc[0]
            kval = "ALL" if str(kval_raw).lower() in {"nan", "none", "all"} else str(kval_raw).replace(".0", "")
        elif str(kval).lower() == "all":
            kval = "ALL"

        counts = folds[selected_col].astype(str).value_counts()
        y = np.arange(len(counts))
        ax.barh(y, counts.to_numpy(), color=task_color(task), alpha=.80)
        ax.set_yticks(y, counts.index.astype(str))
        ax.invert_yaxis()
        ax.set_xlim(0, 5.25)
        ax.set_xticks(range(0, 6))
        ax.set_xlabel("FINAL outer folds selected (n / 5)")
        ax.set_title(f"{task_label(task)}  |  view={view}, K={kval}", fontsize=9)
        clean_axes(ax, grid="x")

        for rank, (model, count) in enumerate(counts.items(), start=1):
            rows.append({
                "task": task,
                "rank": rank,
                "selected_model": model,
                "outer_fold_count": int(count),
                "n_outer_folds": int(len(folds)),
                "frozen_view": view,
                "frozen_K": kval,
                "source_file": str(metrics_path),
            })

    fig.suptitle("Frozen task-specific configurations and model-selection frequencies across the FINAL outer folds", fontsize=11, y=1.01)
    fig.tight_layout(pad=.9, w_pad=.75, h_pad=.75)
    save_figure(fig, out / "FigS4_frozen_task_specific_configurations_and_model_selection_frequencies")
    pd.DataFrame(rows).to_csv(out / "FigS4_model_selection_frequencies.csv", index=False, encoding="utf-8-sig")


def _all_fold_rows(index: ResultIndex) -> pd.DataFrame:
    rows=[]
    for task in ALL_REGRESSION_TASKS+["UL94_V0"]:
        p=locate_task_file(index,task,"outer_fold_metrics",use_bde=False,optional=True)
        if not p:continue
        f=read_csv_auto(p);metric="outer_Macro_F1" if task=="UL94_V0" else "outer_R2"
        if metric not in f:continue
        for _,r in f.iterrows():rows.append({"task":task,"outer_fold":r.get("outer_fold"),"score":r[metric],"metric":metric,"source":str(p)})
    return pd.DataFrame(rows)


def _fold_panel(ax:plt.Axes,data:pd.DataFrame,order:list[str],ylabel:str)->None:
    for i,t in enumerate(order):
        vals=numeric(data.loc[data.task==t,"score"]).dropna().to_numpy();
        if not len(vals):continue
        ax.scatter(np.full(len(vals),i)+jitter(len(vals),.05,800+i),vals,color=task_color(t),s=25,alpha=.65)
        ax.errorbar(i,np.mean(vals),yerr=np.std(vals,ddof=1) if len(vals)>1 else 0,marker="D",color=task_color(t),markerfacecolor="white",markeredgewidth=.9,capsize=3)
    ax.axhline(0,ls="--",lw=.8,color=PALETTE['dark']);ax.set_xticks(range(len(order)),[task_label(t) for t in order],rotation=30,ha="right");ax.set_ylabel(ylabel);clean_axes(ax,grid="y")


def fig_s8_all_stability(index: ResultIndex,out:Path)->None:
    data=_all_fold_rows(index);data.to_csv(out/"FigS8_all_outer_fold_scores.csv",index=False,encoding="utf-8-sig")
    fig,axes=plt.subplots(1,3,figsize=(15.2,4.4),gridspec_kw={"width_ratios":[2.7,1.8,1.0]})
    absolute=[t for t in ["LOI","PHRR","THR","Tg","TS_MPa","FS_MPa","Char_yield"] if not data.empty and t in data.task.unique()]
    delta=[t for t in ["Delta_LOI","Delta_PHRR","Delta_THR","Delta_CY"] if not data.empty and t in data.task.unique()]
    _fold_panel(axes[0],data,absolute,"Outer-fold $R^2$") if absolute else add_no_data(axes[0],"Absolute-task results missing")
    _fold_panel(axes[1],data,delta,"Outer-fold $R^2$") if delta else add_no_data(axes[1],"Delta-task results missing")
    _fold_panel(axes[2],data,["UL94_V0"],"Outer-fold Macro-F1") if not data.empty and "UL94_V0" in data.task.unique() else add_no_data(axes[2],"UL-94 results missing")
    for i,ax in enumerate(axes):panel_label(ax,f"({chr(97+i)})")
    fig.tight_layout(pad=.9,w_pad=.75,h_pad=.75);save_figure(fig,out/"FigS8_all_task_outer_fold_stability")


def fig_s9_split_folds(index:ResultIndex,out:Path)->None:
    tasks=["LOI","PHRR","THR","UL94_V0"];records=[]
    for task in tasks:
        for p in index.all_named(f"{task}_outer_fold_metrics.csv"):
            text=p.as_posix().lower()
            if not any(token in text for token in ["final_grouping_sensitivity_5x5", "three_split_validation"]):
                continue
            split=next((s for s in ["molecule","scaffold","reference"] if f"/{s}/" in text or f"\\{s}\\" in text),None)
            if not split or "smoke" in text:continue
            f=read_csv_auto(p);metric="outer_Macro_F1" if task=="UL94_V0" else "outer_R2"
            if metric not in f:continue
            for _,r in f.iterrows():records.append({"task":task,"split":split,"outer_fold":r.get("outer_fold"),"score":r[metric]})
    data=pd.DataFrame(records).drop_duplicates(["task","split","outer_fold"],keep="last") if records else pd.DataFrame()
    fig,axes=plt.subplots(1,4,figsize=(14.5,3.8))
    for ax,task in zip(axes,tasks):
        sub=data[data.task==task] if not data.empty else pd.DataFrame()
        if sub.empty:add_no_data(ax,"Missing split results");continue
        for i,split in enumerate(["molecule","scaffold","reference"]):
            vals=numeric(sub.loc[sub.split==split,"score"]).dropna().to_numpy();ax.scatter(np.full(len(vals),i)+jitter(len(vals),.05,900+i),vals,color=SPLIT_COLORS[split],s=25)
            if len(vals):ax.errorbar(i,np.mean(vals),yerr=np.std(vals,ddof=1) if len(vals)>1 else 0,marker="D",color=SPLIT_COLORS[split],markerfacecolor="white",markeredgewidth=.9,capsize=3)
        ax.axhline(0,ls="--",lw=.8,color=PALETTE['gray']);ax.set_xticks(range(3),["Molecule","Scaffold","Reference"],rotation=30,ha="right");ax.set_ylabel("Macro-F1" if task=="UL94_V0" else "$R^2$");ax.set_title(task_label(task));clean_axes(ax,grid="y")
    fig.tight_layout(pad=.9,w_pad=.75,h_pad=.75);save_figure(fig,out/"FigS9_three_split_outer_fold_results");data.to_csv(out/"FigS9_split_fold_data.csv",index=False,encoding="utf-8-sig")

def fig_s5_null_tests(index:ResultIndex,out:Path)->None:
    # Prefer the manuscript-canonical FINAL diagnostics directory.  This keeps
    # S5 tied to the current project rather than to copied/stale outputs from an
    # older project version.  Fall back to indexed discovery only when needed.
    final_dir = index.root / "scientific_validation" / "FINAL_null_tests"
    perm0 = final_dir / "y_scrambling_all_results.csv"
    summ0 = final_dir / "null_test_summary.csv"
    perm = perm0 if perm0.exists() else index.locate("y_scrambling_all_results.csv",prefer=["FINAL_null_tests","null_tests"],optional=True)
    summ = summ0 if summ0.exists() else index.locate("null_test_summary.csv",prefer=["FINAL_null_tests","null_tests"],optional=True)
    if not perm or not summ:
        raise FileNotFoundError("Missing results/scientific_validation/FINAL_null_tests outputs; run: D:/Anaconda/python.exe -u run.py null-tests")
    p=read_csv_auto(perm);s=read_csv_auto(summ);tasks=[t for t in ["LOI","PHRR","THR","UL94_V0"] if t in p.task.unique()]
    fig,axes=plt.subplots(1,len(tasks),figsize=(3.6*len(tasks),3.6),squeeze=False)
    for ax,task in zip(axes.ravel(),tasks):
        sub=p[p.task==task].copy();row=s[s.task==task].iloc[0];metric=str(row.primary_metric)
        # Match the inferential unit used for the empirical p value: one null
        # value per complete permutation replicate, averaged across outer folds.
        if "permutation" in sub.columns:
            sub[metric]=numeric(sub[metric])
            vals=sub.groupby("permutation",sort=True)[metric].mean().dropna()
        else:
            vals=numeric(sub[metric]).dropna()
        ax.hist(vals,bins="auto",color=PALETTE['gray_light'],edgecolor=PALETTE['slate'],linewidth=.5,alpha=.95);ax.axvline(row.observed_mean,color=task_color(task),lw=2,label="Observed");ax.axvline(row.null_95_percentile,color=PALETTE['red'],ls="--",label="Null 95th");ax.axvline(row.dummy_mean,color=PALETTE['gold'],ls=":",label="Dummy")
        ax.set_xlabel(metric.replace("_","-"));ax.set_ylabel("Count");ax.set_title(f"{task_label(task)}\np={row.empirical_p_value:.3g}");clean_axes(ax);ax.legend(frameon=False,fontsize=6.5)
    fig.tight_layout(pad=.9,w_pad=.75,h_pad=.75);save_figure(fig,out/"FigS5_dummy_and_y_scrambling")


def fig_s6_learning_curves(index:ResultIndex,out:Path)->None:
    # Use the FINAL learning-curve summary from the current project whenever it
    # exists; do not silently use a similarly named file from an older run.
    final_path = index.root / "scientific_validation" / "FINAL_learning_curves" / "learning_curve_summary.csv"
    path = final_path if final_path.exists() else index.locate("learning_curve_summary.csv",prefer=["FINAL_learning_curves","learning_curves"],optional=True)
    if not path:
        raise FileNotFoundError("Missing results/scientific_validation/FINAL_learning_curves/learning_curve_summary.csv; run: D:/Anaconda/python.exe -u run.py learning-curves")
    d=read_csv_auto(path);fig,axes=plt.subplots(1,3,figsize=(13.5,3.8))
    groups=[(["LOI","PHRR","THR"],"R2_mean","R2_std","Core regression"),(["Tg","TS_MPa"],"R2_mean","R2_std","Auxiliary regression"),(["UL94_V0"],"Macro_F1_mean","Macro_F1_std","UL-94")]
    for ax,(tasks,mean,std,title) in zip(axes,groups):
        for i,t in enumerate(tasks):
            s=d[d.task==t].sort_values("train_fraction");
            if s.empty or mean not in s:continue
            ax.errorbar(s.train_fraction*100,s[mean],yerr=s[std] if std in s else None,marker="o",color=task_color(t),markerfacecolor="white",markeredgewidth=.9,capsize=3,label=task_label(t))
        ax.set_xlabel("Training molecule groups (%)");ax.set_ylabel("Macro-F1" if tasks==["UL94_V0"] else "$R^2$");ax.set_title(title);ax.legend(frameon=False);clean_axes(ax,grid="y")
    fig.tight_layout(pad=.9,w_pad=.75,h_pad=.75);save_figure(fig,out/"FigS6_learning_curves")


def fig_s12_bde_dataset(bde_path:Path,out:Path)->None:
    df=read_csv_auto(bde_path);bond=next((c for c in ["Bond_Type","bond_type"] if c in df),None);target=next((c for c in ["BDE_kJ_mol","BDE (kJ/mol)","BDE_kJ/mol","BDE"] if c in df),None)
    if target is None:
        target=next((c for c in df.columns if "BDE" in c.upper() and "pred" not in c.lower()),None)
    if not bond or not target:raise KeyError(f"BDE columns not recognised: {list(df.columns)}")
    fig,axes=plt.subplots(1,3,figsize=(12.5,3.8));counts=df[bond].astype(str).value_counts();axes[0].bar(counts.index,counts.values,color=[categorical_color(i) for i in range(len(counts))]);axes[0].tick_params(axis='x',rotation=30);axes[0].set_ylabel("Count")
    v=numeric(df[target]).dropna();axes[1].hist(v,bins="auto",color=PALETTE['purple'],alpha=.88);axes[1].set_xlabel("BDE (kJ mol$^{-1}$)");axes[1].set_ylabel("Frequency")
    types=[x for x in ["P-C","P-N"] if x in counts.index];data=[numeric(df.loc[df[bond].astype(str)==x,target]).dropna() for x in types];bp=axes[2].boxplot(data,tick_labels=[f"{x}\n(n={len(a)})" for x,a in zip(types,data)],patch_artist=True);[patch.set_facecolor(c) for patch,c in zip(bp['boxes'],[PALETTE['blue'],PALETTE['red']])];axes[2].set_ylabel("BDE (kJ mol$^{-1}$)")
    for i,ax in enumerate(axes):clean_axes(ax);panel_label(ax,f"({chr(97+i)})")
    fig.tight_layout(pad=.9,w_pad=.75,h_pad=.75);save_figure(fig,out/"FigS12_BDE_dataset_composition")


def fig_s13_bde_diagnostics(index:ResultIndex,out:Path)->None:
    p=index.locate("BDE_selected_config_all_predictions.csv",prefer=["07_bde","diagnostics"],optional=True)
    if not p:raise FileNotFoundError("Run BDE model diagnostics")
    raw=read_csv_auto(p);d=_aggregate_bde_predictions(raw);fig,axes=plt.subplots(2,3,figsize=(14.0,8.0))
    y=numeric(d.BDE_true_kJ_mol);pred=numeric(d.BDE_pred_kJ_mol);res=y-pred;m=regression_metrics(y,pred);lo=min(y.min(),pred.min());hi=max(y.max(),pred.max())
    axes[0,0].scatter(y,pred,s=20,color=PALETTE['blue'],alpha=.6);axes[0,0].plot([lo,hi],[lo,hi],"--",color=PALETTE['dark']);axes[0,0].set_xlabel("True");axes[0,0].set_ylabel("Predicted");metric_box(axes[0,0],f"$R^2$={m['R2']:.3f}")
    axes[0,1].scatter(pred,res,s=20,color=PALETTE['coral'],alpha=.6);axes[0,1].axhline(0,ls="--",color=PALETTE['dark']);axes[0,1].set_xlabel("Predicted");axes[0,1].set_ylabel("Residual")
    axes[0,2].hist(d.abs_error_kJ_mol,bins="auto",color=PALETTE['gold']);axes[0,2].set_xlabel("Absolute error");axes[0,2].set_ylabel("Frequency")
    if "Bond_Type" in d:
        types=[x for x in ["P-C","P-N"] if x in d.Bond_Type.astype(str).unique()];data=[d.loc[d.Bond_Type.astype(str)==x,"abs_error_kJ_mol"] for x in types];axes[1,0].boxplot(data,tick_labels=[f"{x}\n(n={len(a)})" for x,a in zip(types,data)]);axes[1,0].set_ylabel("Absolute error")
    else:add_no_data(axes[1,0],"Bond type unavailable")
    top=d.sort_values("abs_error_kJ_mol",ascending=False).head(8);axes[1,1].barh(np.arange(len(top)),top.abs_error_kJ_mol[::-1],color=PALETTE['orange']);axes[1,1].set_yticks(np.arange(len(top)),[f"Sample {i}" for i in range(len(top),0,-1)]);axes[1,1].set_xlabel("Absolute error")
    cv=index.locate_any(["BDE_seed_best_diagnostics.csv","BDE_pc_pn_all_results.csv","BDE_pc_pn_summary.csv"],prefer=["07_bde"],optional=True)
    if cv:
        c=read_csv_auto(cv);col=next((x for x in ["test_R2","R2","test_R2_mean"] if x in c),None)
        if col:axes[1,2].hist(numeric(c[col]).dropna(),bins="auto",color=PALETTE['purple']);axes[1,2].set_xlabel("Cross-validation $R^2$");axes[1,2].set_ylabel("Count")
        else:add_no_data(axes[1,2],"R² column unavailable")
    else:add_no_data(axes[1,2],"CV diagnostic file missing")
    for i,ax in enumerate(axes.ravel()):clean_axes(ax);panel_label(ax,f"({chr(97+i)})")
    fig.tight_layout(pad=.9,w_pad=.75,h_pad=.75);save_figure(fig,out/"FigS13_BDE_complete_diagnostics");top.to_csv(out/"FigS13_top8_BDE_errors.csv",index=False,encoding="utf-8-sig")


def fig_s8_shap(index:ResultIndex,out:Path)->None:
    tasks=["LOI","PHRR","THR","UL94_V0"]
    has_sample_level=all(_shap_long_paths(index,t) for t in tasks)
    fig,axes=plt.subplots(2,2,figsize=(13.0,10.0))
    for ax,task in zip(axes.ravel(),tasks):
        long_paths=_shap_long_paths(index,task); s_path=_shap_summary_path(index,task)
        if long_paths:
            long=pd.concat([read_csv_auto(p) for p in long_paths],ignore_index=True)
            summary=read_csv_auto(s_path).sort_values("mean_abs_shap_all_folds",ascending=False) if s_path else None
            top=summary.head(15).feature.astype(str).tolist() if summary is not None else long.groupby("feature").shap_value.apply(lambda x:numeric(x).abs().mean()).sort_values(ascending=False).head(15).index.tolist()
            _beeswarm(ax,long,top)
        elif s_path:
            summary=read_csv_auto(s_path).sort_values("mean_abs_shap_all_folds",ascending=False).head(15).sort_values("mean_abs_shap_all_folds")
            ax.barh(np.arange(len(summary)),summary.mean_abs_shap_all_folds,color=task_color(task))
            ax.set_yticks(np.arange(len(summary)),summary.feature.astype(str)); ax.set_xlabel("Mean |SHAP| across folds")
            ax.text(.02,.02,"Fold-aggregated importance; not a beeswarm",transform=ax.transAxes,fontsize=7)
        else:
            add_no_data(ax,"Missing SHAP output")
        ax.set_title(task_label(task)); clean_axes(ax,grid="x")
    fig.tight_layout(pad=.9,w_pad=.75,h_pad=.75)
    name="FigS8_core_task_SHAP_beeswarms" if has_sample_level else "FigS8_core_task_SHAP_stable_feature_importance"
    save_figure(fig,out/name)

def fig_s15_shap_stability(index:ResultIndex,out:Path)->None:
    tasks=["LOI","PHRR","THR","UL94_V0","Tg","TS_MPa"];fig,axes=plt.subplots(2,3,figsize=(13.0,8.5))
    rows=[]
    for ax,task in zip(axes.ravel(),tasks):
        p=index.locate(f"{task}_SHAP_rank_spearman.csv",prefer=["05_shap",task],avoid=["smoke"],optional=True)
        if not p:add_no_data(ax,"Missing rank correlation");continue
        c=read_csv_auto(p,index_col=0);arr=c.apply(pd.to_numeric,errors="coerce").to_numpy();im=ax.imshow(arr,vmin=0,vmax=1,cmap=PURPLE_CMAP);ax.set_xticks(range(len(c.columns)),c.columns);ax.set_yticks(range(len(c.index)),c.index);ax.set_title(task_label(task)); clean_axes(ax,boxed=True)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if np.isfinite(arr[i,j]):ax.text(j,i,f"{arr[i,j]:.2f}",ha="center",va="center",fontsize=6,color="white" if abs(arr[i,j])>.55 else PALETTE["black"])
        upper=arr[np.triu_indices_from(arr,k=1)];rows.append({"task":task,"mean_pairwise_spearman":np.nanmean(upper),"min":np.nanmin(upper),"max":np.nanmax(upper)})
    if rows:
        fig.subplots_adjust(right=.92,wspace=.34,hspace=.34)
        cax=fig.add_axes([.94,.18,.014,.64])
        fig.colorbar(im,cax=cax,label="Spearman correlation")
    save_figure(fig,out/"FigS15_SHAP_cross_fold_rank_stability");pd.DataFrame(rows).to_csv(out/"FigS15_SHAP_stability_summary.csv",index=False,encoding="utf-8-sig")


def fig_s16_dependence(index:ResultIndex,out:Path)->None:
    tasks=["LOI","PHRR","THR","UL94_V0"]
    available=[t for t in tasks if _shap_long_paths(index,t) and _shap_summary_path(index,t)]
    if not available:
        raise FileNotFoundError("Sample-level SHAP values are absent. Rerun: python -u run.py shap --tasks LOI,PHRR,THR,UL94_V0 --results results/05_Shap/FINAL_core_fixed_baseline_inclusive_5x5")
    fig,axes=plt.subplots(2,4,figsize=(15.5,7.2)); manifest=[]
    for col,task in enumerate(tasks):
        paths=_shap_long_paths(index,task); s_path=_shap_summary_path(index,task)
        if not paths or not s_path:
            add_no_data(axes[0,col],"Sample-level SHAP missing"); add_no_data(axes[1,col],"Rerun SHAP exporter"); continue
        long=pd.concat([read_csv_auto(p) for p in paths],ignore_index=True); summary=read_csv_auto(s_path)
        candidates=summary[~summary.feature.astype(str).str.contains("Morgan|MACCS|bit",case=False,regex=True)].sort_values("mean_abs_shap_all_folds",ascending=False)
        chosen=[]
        for feature in candidates.feature.astype(str):
            sub=long[long.feature.astype(str)==feature]
            if "feature_value_raw" in sub and numeric(sub.feature_value_raw).nunique()>1: chosen.append(feature)
            if len(chosen)==2: break
        for row,feature in enumerate(chosen):
            ax=axes[row,col]; sub=long[long.feature.astype(str)==feature]
            x=numeric(sub.feature_value_raw).to_numpy(); y=numeric(sub.shap_value).to_numpy(); mask=np.isfinite(x)&np.isfinite(y)
            if np.unique(x[mask]).size<=3:
                cats=sorted(np.unique(x[mask])); data=[y[mask & (x==c)] for c in cats]
                ax.boxplot(data,tick_labels=[str(c) for c in cats]); ax.set_xlabel(feature)
            else:
                ax.scatter(x[mask],y[mask],s=16,color=task_color(task),alpha=.48,edgecolor="white",linewidth=.18)
                tx,ty=robust_lowess_like(x[mask],y[mask])
                if len(tx): ax.plot(tx,ty,color=PALETTE['red_dark'],lw=1.8)
                ax.set_xlabel(feature)
            ax.axhline(0,ls="--",lw=.8,color=PALETTE['gray']); ax.set_ylabel("SHAP value"); clean_axes(ax,grid="y")
            manifest.append({"task":task,"feature":feature,"panel_row":row+1})
        for row in range(len(chosen),2): add_no_data(axes[row,col],"No second stable continuous feature")
        axes[0,col].set_title(task_label(task))
    fig.tight_layout(pad=.9,w_pad=.75,h_pad=.75); save_figure(fig,out/"FigS16_key_continuous_SHAP_dependence")
    pd.DataFrame(manifest).to_csv(out/"FigS16_selected_features.csv",index=False,encoding="utf-8-sig")

def _parse_morgan_feature(name:str)->tuple[str,int,int]|None:
    m=re.search(r"(?P<prefix>main|co|curing).*morgan_r(?P<radius>\d+)_(?P<bit>\d+)",name,re.I)
    if not m:return None
    return m.group("prefix").lower(),int(m.group("radius")),int(m.group("bit"))


def fig_s17_morgan_mapping(df:pd.DataFrame,index:ResultIndex,out:Path)->None:
    # RDKit's 2D drawing extension (rdMolDraw2D) can fail to load on some
    # Windows/Anaconda installations even when Chem and Morgan fingerprints
    # work normally. Keep S17 mapping available and make PNG highlights
    # optional instead of aborting the complete supplementary-figure workflow.
    try:
        from rdkit.Chem import Draw
        draw_available = True
        draw_error = ""
    except Exception as exc:
        Draw = None
        draw_available = False
        draw_error = f"{type(exc).__name__}: {exc}"
        print(f"[WARN] RDKit Draw unavailable; S17 will export mapping CSV without PNG highlights: {draw_error}")

    records=[];image_dir=out/"FigS17_RDKit_highlights"
    if draw_available:
        image_dir.mkdir(parents=True,exist_ok=True)
    for task in ["LOI","PHRR","THR","UL94_V0"]:
        p=_shap_summary_path(index,task)
        if not p:continue
        s=read_csv_auto(p).sort_values("mean_abs_shap_all_folds",ascending=False)
        features=[f for f in s.feature.astype(str) if _parse_morgan_feature(f)][:3]
        for feature in features:
            parsed=_parse_morgan_feature(feature)
            if not parsed:continue
            prefix,radius,bit=parsed;smiles_col={"main":"SMILES_main","co":"SMILES_co","curing":"SMILES_Curing_Agent"}[prefix];name_col={"main":"FR_main","co":"FR_co","curing":"Curing_Agent"}[prefix]
            if smiles_col not in df:continue
            nbits=1024 if bit>=512 else 512;found=False
            for _,row in df[[c for c in [name_col,smiles_col] if c in df]].dropna(subset=[smiles_col]).drop_duplicates(smiles_col).iterrows():
                mol=Chem.MolFromSmiles(str(row[smiles_col]));
                if mol is None:continue
                bit_info={};AllChem.GetMorganFingerprintAsBitVect(mol,radius,nBits=nbits,bitInfo=bit_info)
                if bit not in bit_info:continue
                atom_idx,rad=bit_info[bit][0];env=Chem.FindAtomEnvironmentOfRadiusN(mol,rad,atom_idx);atoms={atom_idx}
                for bond_idx in env:
                    b=mol.GetBondWithIdx(bond_idx);atoms.update([b.GetBeginAtomIdx(),b.GetEndAtomIdx()])
                path = ""
                if draw_available:
                    image=Draw.MolToImage(mol,size=(700,450),highlightAtoms=sorted(atoms),highlightBonds=list(env),legend=f"{task} | {feature}")
                    image_path=image_dir/f"{task}_{feature}.png";image.save(image_path);path=str(image_path)
                records.append({"task":task,"feature":feature,"role":prefix,"radius":radius,"bit":bit,"nbits_assumed":nbits,"representative_name":row.get(name_col,""),"representative_smiles":row[smiles_col],"rdkit_highlight":path,"rdkit_draw_available":int(draw_available),"rdkit_draw_error":draw_error,"manual_chemdraw_required":1});found=True;break
            if not found:records.append({"task":task,"feature":feature,"role":prefix,"radius":radius,"bit":bit,"nbits_assumed":nbits,"representative_name":"NOT FOUND","representative_smiles":"","rdkit_highlight":"","rdkit_draw_available":int(draw_available),"rdkit_draw_error":draw_error,"manual_chemdraw_required":1})
    if not records:raise FileNotFoundError("No stable Morgan features could be mapped")
    pd.DataFrame(records).to_csv(out/"FigS17_Morgan_fragment_mapping_manifest.csv",index=False,encoding="utf-8-sig")


def fig_s18_threshold(index:ResultIndex,out:Path)->None:
    tasks=["LOI","PHRR","THR","UL94_V0"];fig,axes=plt.subplots(1,4,figsize=(14.5,3.6))
    for ax,task in zip(axes,tasks):
        p=_locate_ad(index,task,"AD_threshold_sensitivity")
        if not p:add_no_data(ax,"Missing AD sensitivity");continue
        s=read_csv_auto(p);x=next((c for c in ["reliable_threshold","threshold"] if c in s),None);metric=next((c for c in ["MAE","Macro_F1","Accuracy"] if c in s),None)
        if not x or not metric:add_no_data(ax,"Columns unrecognised");continue
        groups=sorted(s.get("caution_threshold",pd.Series([np.nan])).dropna().unique())
        if len(groups):
            for i,g in enumerate(groups):sub=s[s.caution_threshold==g].sort_values(x);ax.plot(sub[x],sub[metric],marker="o",color=categorical_color(i),label=f"C={g:.2f}",markerfacecolor="white",markeredgewidth=.8)
        else:ax.plot(s[x],s[metric],marker="o",color=task_color(task),markerfacecolor="white",markeredgewidth=.8)
        ax.set_xlabel("Reliable threshold");ax.set_ylabel(metric);ax.set_title(task_label(task));clean_axes(ax,grid="y");ax.legend(frameon=False,fontsize=6)
    fig.tight_layout(pad=.9,w_pad=.75,h_pad=.75);save_figure(fig,out/"FigS18_AD_threshold_sensitivity")


def fig_s19_similarity_error(index:ResultIndex,out:Path)->None:
    tasks=["LOI","PHRR","THR","UL94_V0"];fig,axes=plt.subplots(1,4,figsize=(14.8,3.7))
    for ax,task in zip(axes,tasks):
        p=_locate_ad(index,task,"outer_predictions_with_AD")
        if not p:add_no_data(ax,"Missing AD predictions");continue
        d=read_csv_auto(p);x=numeric(d.AD_similarity).to_numpy()
        if task=="UL94_V0":
            prob_col=probability_column(d);y=numeric(d.get("sample_log_loss",pd.Series(dtype=float))).to_numpy() if "sample_log_loss" in d else (-(numeric(d.y_true)*np.log(numeric(d[prob_col]).clip(1e-8,1-1e-8))+(1-numeric(d.y_true))*np.log(1-numeric(d[prob_col]).clip(1e-8,1-1e-8)))).to_numpy();ylabel="Single-sample log loss"
        else:
            y=numeric(d.get("absolute_error",(numeric(d.y_true)-numeric(d.y_pred)).abs())).to_numpy();ylabel="Absolute error"
        mask=np.isfinite(x)&np.isfinite(y);ax.scatter(x[mask],y[mask],s=16,color=task_color(task),alpha=.45,edgecolor="white",linewidth=.15);tx,ty=robust_lowess_like(x[mask],y[mask]);
        if len(tx):ax.plot(tx,ty,color=PALETTE['red_dark'],lw=1.8)
        ax.axvline(.5,ls="--",lw=.8,color=PALETTE['gray']);ax.axvline(.7,ls="--",lw=.8,color=PALETTE['gray']);ax.set_xlabel("Max Tanimoto similarity");ax.set_ylabel(ylabel);ax.set_title(task_label(task));clean_axes(ax)
        if task=="THR":ax.text(.03,.97,"No monotonic trend is retained if observed",transform=ax.transAxes,va="top",fontsize=6.5)
    fig.tight_layout(pad=.9,w_pad=.75,h_pad=.75);save_figure(fig,out/"FigS19_similarity_error_all_core_tasks")


def fig_s20_seen_novel(index:ResultIndex,out:Path)->None:
    tasks=["LOI","PHRR","THR","UL94_V0"];fig,axes=plt.subplots(1,4,figsize=(14.2,3.6));rows=[]
    for ax,task in zip(axes,tasks):
        p=_locate_ad(index,task,"outer_predictions_with_AD")
        if not p:add_no_data(ax,"Missing AD predictions");continue
        d=read_csv_auto(p);group=next((c for c in ["scaffold_novelty","scaffold_status"] if c in d),None)
        if not group:add_no_data(ax,"Scaffold novelty column missing");continue
        if task=="UL94_V0":
            pred=numeric(d.y_pred).astype(int);true=numeric(d.y_true).astype(int);d["metric_value"]=(pred==true).astype(float);ylabel="Accuracy"
        else:d["metric_value"]=(numeric(d.y_true)-numeric(d.y_pred)).abs();ylabel="MAE"
        cats=list(d[group].dropna().astype(str).unique());vals=[]
        for cat in cats:
            sub=d[d[group].astype(str)==cat];fold_values=sub.groupby("outer_fold").metric_value.mean() if "outer_fold" in sub else pd.Series([sub.metric_value.mean()]);vals.append(fold_values.to_numpy());rows.extend({"task":task,"scaffold_group":cat,"outer_fold":idx,"metric":v,"n_samples":len(sub)} for idx,v in fold_values.items())
        bp=ax.boxplot(vals,tick_labels=[f"{c}\n(n={int((d[group].astype(str)==c).sum())})" for c in cats],patch_artist=True); [patch.set_facecolor([PALETTE["blue"],PALETTE["orange"],PALETTE["purple"]][i%3]) for i,patch in enumerate(bp["boxes"])]; ax.set_ylabel(ylabel);ax.set_title(task_label(task));clean_axes(ax,grid="y")
    fig.tight_layout(pad=.9,w_pad=.75,h_pad=.75);save_figure(fig,out/"FigS20_seen_vs_novel_scaffold_performance");pd.DataFrame(rows).to_csv(out/"FigS20_scaffold_fold_metrics.csv",index=False,encoding="utf-8-sig")


def fig_s21_outer_errors(index:ResultIndex,out:Path)->None:
    tasks=["LOI","PHRR","THR","UL94_V0"];fig,axes=plt.subplots(2,2,figsize=(11.5,8.0));tables=[]
    for ax,task in zip(axes.ravel(),tasks):
        p=index.locate_any([f"{task}_top_20_error_cases.csv",f"{task}_outer_predictions_with_errors.csv"],prefer=["outer_error_analysis",task],optional=True)
        if not p:add_no_data(ax,"Run run.py error-analysis");continue
        d=read_csv_auto(p)
        if task=="UL94_V0":metric=next((c for c in ["sample_log_loss","classification_margin"] if c in d),None)
        else:metric=next((c for c in ["abs_error","absolute_error"] if c in d),None)
        if not metric:add_no_data(ax,"Error column missing");continue
        top=d.sort_values(metric,ascending=False).head(20).copy();top["sample_label"]=[f"Sample {i+1}" for i in range(len(top))];ax.barh(np.arange(len(top)),numeric(top[metric])[::-1],color=task_color(task),alpha=.88);ax.set_yticks(np.arange(len(top)),top.sample_label[::-1]);ax.set_xlabel(metric.replace("_"," "));ax.set_title(task_label(task));clean_axes(ax,grid="x");top["task"]=task;tables.append(top)
    fig.tight_layout(pad=.9,w_pad=.75,h_pad=.75);save_figure(fig,out/"FigS21_top_outer_error_cases")
    if tables:pd.concat(tables,ignore_index=True).to_csv(out/"FigS21_top_error_case_details.csv",index=False,encoding="utf-8-sig")


def fig_s22_ul94_folds(index:ResultIndex,out:Path)->None:
    p=locate_task_file(index,"UL94_V0","outer_predictions",use_bde=False);d=read_csv_auto(p);prob_col=probability_column(d)
    folds=sorted(numeric(d.outer_fold).dropna().astype(int).unique()) if "outer_fold" in d else [1];fig,axes=plt.subplots(2,2,figsize=(9.5,7.8))
    # Pooled fold curves shown as light lines plus mean interpolation.
    grid=np.linspace(0,1,101);roc_interp=[];pr_interp=[];cal_rows=[];recall_rows=[]
    for fold in folds:
        sub=d[numeric(d.outer_fold).astype("Int64")==fold] if "outer_fold" in d else d;y=numeric(sub.y_true).astype(int).to_numpy();pred=numeric(sub.y_pred).astype(int).to_numpy();prob=numeric(sub[prob_col]).clip(0,1).to_numpy()
        cm=np.array([[((y==0)&(pred==0)).sum(),((y==0)&(pred==1)).sum()],[((y==1)&(pred==0)).sum(),((y==1)&(pred==1)).sum()]])
        recall_rows.append({"outer_fold":fold,"non_V0_recall":cm[0,0]/cm[0].sum() if cm[0].sum() else np.nan,"V0_recall":cm[1,1]/cm[1].sum() if cm[1].sum() else np.nan})
        fpr,tpr,_=roc_curve(y,prob);prec,rec,_=precision_recall_curve(y,prob);axes[0,0].plot(fpr,tpr,color=PALETTE['blue'],alpha=.25);axes[0,1].plot(rec,prec,color=PALETTE['red'],alpha=.22);roc_interp.append(np.interp(grid,fpr,tpr));pr_interp.append(np.interp(grid,rec[::-1],prec[::-1]))
        bins=pd.qcut(prob,q=min(5,len(np.unique(prob))),duplicates="drop");cal=pd.DataFrame({"prob":prob,"y":y,"bin":bins}).groupby("bin",observed=True).agg(mean_prob=("prob","mean"),frac=("y","mean"));axes[1,0].plot(cal.mean_prob,cal.frac,marker="o",alpha=.35,color=PALETTE['green']);cal_rows.extend({"outer_fold":fold,"mean_probability":r.mean_prob,"fraction_positive":r.frac} for _,r in cal.iterrows())
    axes[0,0].plot(grid,np.mean(roc_interp,axis=0),color=PALETTE['blue_dark'],lw=2,label="Mean");axes[0,0].plot([0,1],[0,1],"--",color=PALETTE['gray']);axes[0,0].set_xlabel("FPR");axes[0,0].set_ylabel("TPR");axes[0,0].legend(frameon=False)
    axes[0,1].plot(grid,np.mean(pr_interp,axis=0),color=PALETTE['red_dark'],lw=2,label="Mean");axes[0,1].set_xlabel("Recall");axes[0,1].set_ylabel("Precision");axes[0,1].legend(frameon=False)
    axes[1,0].plot([0,1],[0,1],"--",color=PALETTE['gray']);axes[1,0].set_xlabel("Mean predicted probability");axes[1,0].set_ylabel("Observed V-0 fraction")
    recdf=pd.DataFrame(recall_rows);x=np.arange(len(recdf));axes[1,1].bar(x-.18,recdf.non_V0_recall,.36,color=PALETTE['blue'],label="Non-V-0");axes[1,1].bar(x+.18,recdf.V0_recall,.36,color=PALETTE['red'],label="V-0");axes[1,1].set_xticks(x,[f"Fold {i}" for i in recdf.outer_fold]);axes[1,1].set_ylabel("Recall");axes[1,1].legend(frameon=False)
    for i,ax in enumerate(axes.ravel()):clean_axes(ax);panel_label(ax,f"({chr(97+i)})")
    fig.tight_layout(pad=.9,w_pad=.75,h_pad=.75);save_figure(fig,out/"FigS12_UL94_outer_fold_diagnostics");recdf.to_csv(out/"FigS12_class_recall_by_fold.csv",index=False,encoding="utf-8-sig");pd.DataFrame(cal_rows).to_csv(out/"FigS12_calibration_by_fold.csv",index=False,encoding="utf-8-sig")


def _candidate_file(index: ResultIndex) -> Path | None:
    path = index.locate(
        "candidate_best_per_molecule.csv",
        prefer=["06_reversedesign", "combined_flux50"],
        avoid=["final_flux50", "sensitivity_flux35", "smoke", "2x2"],
        optional=True,
        label="supplementary combined candidates",
    )
    if path:
        return path
    return index.locate_any(
        ["candidate_predictions.csv", "ranked_candidates.csv", "pareto_candidates.csv", "candidate_ranking.csv"],
        prefer=["06_reversedesign", "candidate", "pareto"],
        avoid=["sensitivity_flux35"],
        optional=True,
    )


def fig_s23_s25_candidates(index: ResultIndex, out: Path) -> None:
    p = _candidate_file(index)
    if not p:
        raise FileNotFoundError("Candidate screening results are not available")
    d = read_csv_auto(p)

    # S23: combined 50 kW/m² screening funnel.
    funnel_path = index.locate(
        "candidate_screening_funnel.csv",
        prefer=["06_reversedesign", "combined_flux50"],
        avoid=["final_flux50", "sensitivity_flux35"],
        optional=True,
        label="S23 combined screening funnel",
    )
    if funnel_path:
        funnel = read_csv_auto(funnel_path)
        stages = list(zip(funnel["stage"].astype(str), numeric(funnel["count"]).fillna(0).astype(int)))
    else:
        stages = [("Original library", len(d))]
        stage_cols = [
            ("Contains DOPO", "contains_DOPO"),
            ("Valid SMILES", "smiles_valid"),
            ("Candidate feasible for screening", "candidate_feasible"),
            ("In domain", "in_domain"),
            ("Pareto candidate", "pareto_flag"),
        ]
        for label, col in stage_cols:
            if col in d:
                stages.append((label, int(numeric(d[col]).fillna(0).astype(bool).sum())))
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    labels = [x[0] for x in stages]
    vals = [x[1] for x in stages]
    widths = np.linspace(.9, .35, len(vals))
    for i, (label, v, w) in enumerate(zip(labels, vals, widths)):
        ax.barh(i, v, height=.7, color=SERIES_COLORS[i % len(SERIES_COLORS)], alpha=float(w))
        ax.text(v/2, i, f"{label}: {v}", ha="center", va="center", fontsize=7.2)
    ax.invert_yaxis()
    ax.set_xlabel("Candidate molecules")
    clean_axes(ax, grid="x")
    save_figure(fig, out / "FigS23_candidate_screening_funnel")
    if funnel_path:
        funnel.to_csv(out / "FigS23_funnel_data_used.csv", index=False, encoding="utf-8-sig")

    # S24: combined training/design/PubChem chemical space.
    space_path = index.locate(
        "candidate_chemical_space.csv",
        prefer=["06_reversedesign", "combined_flux50"],
        avoid=["final_flux50", "sensitivity_flux35"],
        optional=True,
        label="S24 combined chemical space",
    )
    all_path = index.locate(
        "candidate_predictions_all_formulations.csv",
        prefer=["06_reversedesign", "combined_flux50"],
        avoid=["final_flux50", "sensitivity_flux35"],
        optional=True,
        label="S24 combined source map",
    )
    space = read_csv_auto(space_path) if space_path else d.copy()
    all_df = read_csv_auto(all_path) if all_path else d
    x = next((c for c in ["PC1", "UMAP1", "PCA1"] if c in space), None)
    y = next((c for c in ["PC2", "UMAP2", "PCA2"] if c in space), None)
    if x and y:
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
        cats = pd.Series("Candidate", index=space.index, dtype=object)
        cats.loc[status.str.lower().eq("training")] = "Training"
        cats.loc[~status.str.lower().eq("training") & source.eq("designed")] = "Designed"
        cats.loc[~status.str.lower().eq("training") & source.eq("pubchem")] = "PubChem"
        cmap = {"Training": PALETTE["gray"], "Designed": PALETTE["purple"], "PubChem": PALETTE["green"]}
        fig, ax = plt.subplots(figsize=(6.4, 5.1))
        for i, c in enumerate([z for z in ["Training", "Designed", "PubChem", "Candidate"] if z in cats.unique()]):
            s = cats.eq(c)
            ax.scatter(numeric(space.loc[s, x]), numeric(space.loc[s, y]), s=22, color=cmap.get(str(c), SERIES_COLORS[i % len(SERIES_COLORS)]), alpha=.62, label=c, edgecolor="white", linewidth=.25)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.legend(frameon=False)
        clean_axes(ax)
        save_figure(fig, out / "FigS24_training_candidate_chemical_space")
        space.assign(Display_group=cats).to_csv(out / "FigS24_chemical_space_data_used.csv", index=False, encoding="utf-8-sig")

    # S25: final combined priority-candidate heatmap.
    final_path = index.locate(
        "final_priority_candidates_combined.csv",
        prefer=["06_reversedesign", "final_priority_combined"],
        avoid=["final_priority/"],
        optional=True,
        label="S25 combined final priority",
    )
    heat = read_csv_auto(final_path) if final_path else d.copy()
    if final_path:
        metrics = [c for c in ["LOI_pred_50", "V0_probability_50", "PHRR_pred_50", "THR_pred_50", "Tg_pred_50", "TS_pred_50"] if c in heat]
    else:
        metrics = [c for c in ["LOI_pred", "V0_probability", "PHRR_pred", "THR_pred", "Tg_pred", "TS_pred"] if c in heat]
    if metrics:
        a = heat[metrics].apply(pd.to_numeric, errors="coerce")
        z = (a-a.mean()) / a.std(ddof=0).replace(0, np.nan)
        n = min(20 if final_path else 50, len(z))
        labels = heat.get("Candidate_Name", heat.get("Candidate_ID", pd.Series(index=heat.index, dtype=object))).astype(str).head(n)
        fig, ax = plt.subplots(figsize=(10, max(5, n*.24)))
        im = ax.imshow(z.head(n).fillna(0), aspect="auto", cmap=DIVERGING_CMAP)
        xlabels = [c.replace("_pred_50", "").replace("_50", "").replace("V0_probability", "V-0 prob.") for c in metrics]
        ax.set_xticks(range(len(metrics)), xlabels, rotation=35, ha="right")
        ax.set_yticks(range(n), [s[:28] for s in labels])
        fig.colorbar(im, ax=ax, label="Standardised value")
        save_figure(fig, out / "FigS25_candidate_multiobjective_heatmap")
        heat.head(n).to_csv(out / "FigS25_candidate_data_used.csv", index=False, encoding="utf-8-sig")



def fig_s2_compact_dataset(df: pd.DataFrame, out: Path) -> None:
    """Compact S2: data completeness, task sample sizes and target summaries."""
    fig = plt.figure(figsize=(13.2, 8.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 1.0], wspace=0.28, hspace=0.28)

    ax = fig.add_subplot(gs[:, 0])
    fields = [("LOI", "LOI"), ("UL-94", "UL94"), ("PHRR", "PHRR_kw_㎡"), ("THR", "THR_MJ_㎡"), ("Tg", "Tg_℃"), ("Char yield", "Char_yield_％_700C"), ("TS", "TS_MPa"), ("FS", "FS_MPa"), ("LOI thickness", "LOI_Thickness_mm"), ("UL-94 thickness", "UL94_Thickness_mm"), ("Cone flux", "Cone_flux_kW_m2"), ("Cone thickness", "Cone_Thickness_mm"), ("Cure temperature", "Cure_Temp_Max"), ("EP baseline LOI", "EP_matrix_LOI"), ("EP baseline PHRR", "EP_matrix_PHRR"), ("EP baseline THR", "EP_matrix_THR"), ("EP baseline Tg", "EP_matrix_Tg"), ("EP baseline char", "EP_matrix_CY"), ("EP baseline TS", "EP_matrix_TS"), ("EP baseline FS", "EP_matrix_FS")]
    rows = [{"field": label, "missing_rate_percent": float(df[col].isna().mean() * 100)} for label, col in fields if col in df.columns]
    tab = pd.DataFrame(rows).sort_values("missing_rate_percent")
    ax.barh(np.arange(len(tab)), tab["missing_rate_percent"], color=PALETTE["orange"], alpha=0.88)
    ax.set_yticks(np.arange(len(tab)), tab["field"])
    ax.set_xlim(0, 105)
    ax.set_xlabel("Missing rate (%)")
    ax.set_title("Key-field completeness")
    clean_axes(ax, grid="x")
    for i, v in enumerate(tab["missing_rate_percent"]):
        ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=7)
    panel_label(ax, "(a)")

    ax = fig.add_subplot(gs[0, 1])
    tasks = ["LOI", "PHRR", "THR", "UL94_V0", "Tg", "TS_MPa"]
    rows = []
    for task in tasks:
        col = TARGET_COLUMNS[task]
        s = df[col] if col in df.columns else pd.Series(dtype=float)
        rows.append({"task": task_label(task), "n": int(s.notna().sum())})
    tmp = pd.DataFrame(rows)
    ax.bar(np.arange(len(tmp)), tmp["n"], color=[task_color(t) for t in tasks], alpha=0.9)
    ax.set_xticks(np.arange(len(tmp)), tmp["task"], rotation=30, ha="right")
    ax.set_ylabel("Valid records (n)")
    ax.set_title("Principal-task sample sizes")
    clean_axes(ax, grid="y")
    for i, v in enumerate(tmp["n"]):
        ax.text(i, v + max(tmp["n"]) * 0.02, str(int(v)), ha="center", fontsize=7)
    panel_label(ax, "(b)")

    ax = fig.add_subplot(gs[1, 1])
    summary_rows = []
    for task in ["LOI", "PHRR", "THR", "Tg", "TS_MPa"]:
        col = TARGET_COLUMNS[task]
        vals = numeric(df[col]).dropna() if col in df.columns else pd.Series(dtype=float)
        if len(vals):
            summary_rows.append({"task": task_label(task), "median": float(vals.median())})
    if TARGET_COLUMNS.get("UL94_V0") in df.columns:
        ul = normalise_ul94(df[TARGET_COLUMNS["UL94_V0"]]).dropna()
        summary_rows.append({"task": "UL-94 V-0 rate", "median": float(ul.mean() * 100)})
    summ = pd.DataFrame(summary_rows)
    ax.barh(np.arange(len(summ)), summ["median"], color=PALETTE["purple"], alpha=0.85)
    ax.set_yticks(np.arange(len(summ)), summ["task"])
    ax.set_xlabel("Median value / V-0 rate (%)")
    ax.set_title("Compact target-distribution summary")
    clean_axes(ax, grid="x")
    for i, v in enumerate(summ["median"]):
        ax.text(v, i, f"  {v:.2f}", va="center", fontsize=7)
    panel_label(ax, "(c)")

    save_figure(fig, out / "FigS2_dataset_completeness_and_target_summary")
    tab.to_csv(out / "FigS2_missingness_data.csv", index=False, encoding="utf-8-sig")


def fig_s3_compact_chemical_space(df: pd.DataFrame, out: Path) -> None:
    """Compact S3: chemical-space embedding plus scaffold coverage."""
    tab, matrix = _unique_molecules(df)
    if len(tab) < 3:
        raise ValueError("Too few valid unique molecules")
    pca = PCA(n_components=min(20, matrix.shape[0] - 1, matrix.shape[1]), random_state=42).fit_transform(matrix)
    coords = pca[:, :2]
    tab["PC1"], tab["PC2"] = coords[:, 0], coords[:, 1]
    tab["scaffold"] = tab["SMILES_main"].map(murcko_scaffold)
    counts = tab["scaffold"].dropna().value_counts()
    top = counts.head(12).sort_values()
    cum = counts.cumsum() / counts.sum() * 100 if len(counts) else pd.Series(dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.0))
    axes[0, 0].scatter(tab["PC1"], tab["PC2"], s=24, color=PALETTE["blue"], alpha=0.72)
    axes[0, 0].set_xlabel("PC1"); axes[0, 0].set_ylabel("PC2"); axes[0, 0].set_title("Unique main flame retardants")
    cats = tab.get("Synergy_type", pd.Series("Unknown", index=tab.index)).fillna("Unknown").astype(str)
    top_cats = cats.value_counts().head(7).index
    cats = cats.where(cats.isin(top_cats), "Other")
    for i, c in enumerate(cats.value_counts().index):
        s = cats.eq(c)
        axes[0, 1].scatter(tab.loc[s, "PC1"], tab.loc[s, "PC2"], s=24, color=categorical_color(i), alpha=0.72, label=c)
    axes[0, 1].set_xlabel("PC1"); axes[0, 1].set_ylabel("PC2"); axes[0, 1].set_title("Chemical space by synergy type"); axes[0, 1].legend(frameon=False, fontsize=6, loc="best")
    axes[1, 0].barh(np.arange(len(top)), top.values, color=PALETTE["teal"], alpha=0.9)
    axes[1, 0].set_yticks(np.arange(len(top)), [f"Scaffold {counts.index.get_loc(x)+1}" for x in top.index])
    axes[1, 0].set_xlabel("Record count"); axes[1, 0].set_title("Most frequent Murcko scaffolds")
    axes[1, 1].plot(np.arange(1, len(cum) + 1), cum.values, marker="o", color=PALETTE["red_dark"], lw=1.8)
    axes[1, 1].axhline(80, ls="--", color=PALETTE["gray"])
    axes[1, 1].set_xlabel("Scaffolds ranked by frequency"); axes[1, 1].set_ylabel("Cumulative coverage (%)"); axes[1, 1].set_title("Scaffold coverage")
    for i, ax in enumerate(axes.ravel()):
        clean_axes(ax, grid="y" if i in {2, 3} else None)
        panel_label(ax, f"({chr(97+i)})")
    fig.tight_layout(pad=.9, w_pad=.75, h_pad=.75)
    save_figure(fig, out / "FigS3_chemical_space_and_scaffold_diversity")
    tab.to_csv(out / "FigS3_chemical_space_coordinates.csv", index=False, encoding="utf-8-sig")


def fig_s5_compact_stability(index: ResultIndex, out: Path) -> None:
    """Compact S5: outer-fold stability plus split sensitivity."""
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.6), gridspec_kw={"width_ratios": [1.15, 1.0]})
    data = _all_fold_rows(index)
    absolute = [t for t in ["LOI", "PHRR", "THR", "Tg", "TS_MPa"] if not data.empty and t in data.task.unique()]
    if absolute or (not data.empty and "UL94_V0" in data.task.unique()):
        for i, t in enumerate(absolute):
            vals = numeric(data.loc[data.task == t, "score"]).dropna().to_numpy()
            axes[0].scatter(np.full(len(vals), i) + jitter(len(vals), .05, 810 + i), vals, color=task_color(t), s=25, alpha=.65)
            if len(vals):
                axes[0].errorbar(i, np.mean(vals), yerr=np.std(vals, ddof=1) if len(vals) > 1 else 0, marker="D", color=task_color(t), markerfacecolor="white", markeredgewidth=.9, capsize=3)
        labels = [task_label(t) for t in absolute]
        if not data.empty and "UL94_V0" in data.task.unique():
            i = len(absolute)
            vals = numeric(data.loc[data.task == "UL94_V0", "score"]).dropna().to_numpy()
            axes[0].scatter(np.full(len(vals), i) + jitter(len(vals), .05, 899), vals, color=task_color("UL94_V0"), s=25, alpha=.65)
            axes[0].errorbar(i, np.mean(vals), yerr=np.std(vals, ddof=1) if len(vals) > 1 else 0, marker="D", color=task_color("UL94_V0"), markerfacecolor="white", markeredgewidth=.9, capsize=3)
            labels += [task_label("UL94_V0")]
        axes[0].set_xticks(range(len(labels)), labels, rotation=28, ha="right")
        axes[0].axhline(0, ls="--", lw=.8, color=PALETTE["gray"]); axes[0].set_ylabel("Outer-fold score ($R^2$ or Macro-F1)"); axes[0].set_title("Five-fold stability"); clean_axes(axes[0], grid="y")
    else:
        add_no_data(axes[0], "Missing outer-fold stability results")
    panel_label(axes[0], "(a)")
    tasks = ["LOI", "PHRR", "THR", "UL94_V0"]
    records = []
    for task in tasks:
        for p in index.all_named(f"{task}_outer_fold_metrics.csv"):
            txt = p.as_posix().lower()
            if not any(token in txt for token in ["final_grouping_sensitivity_5x5", "three_split_validation"]):
                continue
            split = next((s for s in ["molecule", "scaffold", "reference"] if f"/{s}/" in txt or f"\\{s}\\" in txt), None)
            if not split or "smoke" in txt: continue
            f = read_csv_auto(p); metric = "outer_Macro_F1" if task == "UL94_V0" else "outer_R2"
            if metric not in f: continue
            for _, r in f.iterrows():
                records.append({"task": task, "split": split, "score": r[metric]})
    split_df = pd.DataFrame(records)
    if split_df.empty:
        add_no_data(axes[1], "Missing grouping-sensitivity results")
    else:
        summary = split_df.groupby(["task", "split"], as_index=False)["score"].mean()
        order = ["molecule", "scaffold", "reference"]; width = 0.18; x = np.arange(len(tasks))
        for j, split in enumerate(order):
            vals = []
            for task in tasks:
                sub = summary[(summary.task == task) & (summary.split == split)]
                vals.append(float(sub["score"].iloc[0]) if not sub.empty else np.nan)
            axes[1].bar(x + (j - 1) * width, vals, width=width, color=SPLIT_COLORS[split], alpha=.85, label=split.title())
        axes[1].set_xticks(x, [task_label(t) for t in tasks], rotation=28, ha="right")
        axes[1].set_ylabel("Mean outer-fold score"); axes[1].set_title("Grouping sensitivity"); axes[1].legend(frameon=False, fontsize=7); clean_axes(axes[1], grid="y")
    panel_label(axes[1], "(b)")
    fig.tight_layout(pad=.9, w_pad=.75, h_pad=.75)
    save_figure(fig, out / "FigS5_outer_fold_stability_and_grouping_sensitivity")


def fig_s7_compact_bde(index: ResultIndex, bde_path: Path, out: Path) -> None:
    """Compact S7: formal P-C/P-N BDE subset composition and diagnostics."""
    df = read_csv_auto(bde_path)
    bond = next((c for c in ["Bond_Type", "bond_type"] if c in df), None)
    target = next((c for c in ["BDE_kJ_mol", "BDE (kJ/mol)", "BDE_kJ/mol", "BDE"] if c in df), None)
    if target is None:
        target = next((c for c in df.columns if "BDE" in c.upper() and "pred" not in c.lower()), None)

    # Formal BDE modelling is restricted to the P-C/P-N subset (n=239).
    formal_df = df.copy()
    if bond:
        norm_bond = (formal_df[bond].astype(str).str.strip().str.upper()
                     .str.replace("–", "-", regex=False)
                     .str.replace("—", "-", regex=False)
                     .str.replace("−", "-", regex=False))
        formal_df = formal_df[norm_bond.isin(["P-C", "P-N"])].copy()
        formal_df["_formal_bond"] = norm_bond.loc[formal_df.index]

    pred_path = index.locate_any(
        ["BDE_selected_config_all_predictions.csv", "BDE_all_predictions.csv"],
        prefer=["07_bde", "selected_config"], avoid=["smoke"], optional=True,
    )
    d = read_csv_auto(pred_path) if pred_path else pd.DataFrame()
    true_col = next((c for c in ["BDE_true_kJ_mol", "y_true", "BDE_true"] if c in d), None)
    pred_col = next((c for c in ["BDE_pred_kJ_mol", "y_pred", "BDE_pred"] if c in d), None)
    if true_col and pred_col:
        d = d.copy()
        d["abs_error_kJ_mol"] = (numeric(d[pred_col]) - numeric(d[true_col])).abs()

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2))

    # (a) Show only the two bond classes actually used in formal BDE modelling.
    if bond and "_formal_bond" in formal_df:
        vc = formal_df["_formal_bond"].value_counts().reindex(["P-C", "P-N"], fill_value=0)
        axes[0, 0].bar(vc.index.astype(str), vc.values, color=[PALETTE["blue"], PALETTE["orange"]], alpha=.9)
        axes[0, 0].set_ylabel("Unique BDE samples")
        axes[0, 0].set_title("Formal P-C/P-N BDE subset")
        ymax = max(float(vc.max()), 1.0)
        for i, v in enumerate(vc.values):
            axes[0, 0].text(i, v + ymax * .025, str(int(v)), ha="center", fontsize=8)
        pd.DataFrame({"Bond_Type": vc.index, "unique_samples": vc.values}).to_csv(
            out / "FigS7_formal_PC_PN_counts.csv", index=False, encoding="utf-8-sig"
        )
    else:
        add_no_data(axes[0, 0], "Bond type unavailable")

    # (b) Distribution is also restricted to the formal P-C/P-N modelling subset.
    if target and target in formal_df:
        vals = numeric(formal_df[target]).dropna()
        axes[0, 1].hist(vals, bins="auto", color=PALETTE["gold"], edgecolor="white", linewidth=.5, alpha=.95)
        axes[0, 1].set_xlabel("BDE (kJ mol$^{-1}$)")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("P-C/P-N experimental BDE values")
    else:
        add_no_data(axes[0, 1], "BDE target unavailable")

    if not d.empty and true_col and pred_col:
        axes[1, 0].scatter(numeric(d[true_col]), numeric(d[pred_col]), s=20, color=PALETTE["green"], alpha=.6)
        lo = float(np.nanmin([numeric(d[true_col]).min(), numeric(d[pred_col]).min()]))
        hi = float(np.nanmax([numeric(d[true_col]).max(), numeric(d[pred_col]).max()]))
        axes[1, 0].plot([lo, hi], [lo, hi], "--", color=PALETTE["gray"])
        axes[1, 0].set_xlabel("Observed BDE (kJ mol$^{-1}$)")
        axes[1, 0].set_ylabel("Predicted BDE (kJ mol$^{-1}$)")
        axes[1, 0].set_title("Repeated-CV prediction diagnostics")
        axes[1, 1].hist(numeric(d["abs_error_kJ_mol"]).dropna(), bins="auto", color=PALETTE["orange"], edgecolor="white", linewidth=.5, alpha=.95)
        axes[1, 1].set_xlabel("Absolute error (kJ mol$^{-1}$)")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].set_title("Prediction-error distribution")
    else:
        add_no_data(axes[1, 0], "Prediction file missing")
        add_no_data(axes[1, 1], "Prediction file missing")

    for i, ax in enumerate(axes.ravel()):
        clean_axes(ax)
        panel_label(ax, f"({chr(97+i)})")
    fig.tight_layout(pad=.9, w_pad=.75, h_pad=.75)
    save_figure(fig, out / "FigS7_BDE_PC_PN_dataset_and_diagnostics")

def fig_s9_compact_shap_stability(index: ResultIndex, out: Path) -> None:
    """Compact S9: SHAP cross-fold rank stability with dedicated colorbar lane."""
    tasks = ["LOI", "PHRR", "THR", "UL94_V0", "Tg", "TS_MPa"]
    fig = plt.figure(figsize=(13.6, 8.6))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.06], wspace=0.42, hspace=0.34)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    cax = fig.add_subplot(gs[:, 3])
    rows = []; im = None
    for ax, task in zip(axes, tasks):
        p = index.locate(f"{task}_SHAP_rank_spearman.csv", prefer=["05_shap", task], avoid=["smoke"], optional=True)
        if not p: add_no_data(ax, "Missing rank correlation"); continue
        c = read_csv_auto(p, index_col=0); arr = c.apply(pd.to_numeric, errors="coerce").to_numpy(); im = ax.imshow(arr, vmin=0, vmax=1, cmap=PURPLE_CMAP)
        ax.set_xticks(range(len(c.columns)), c.columns); ax.set_yticks(range(len(c.index)), c.index); ax.set_title(task_label(task)); clean_axes(ax, boxed=True)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if np.isfinite(arr[i, j]): ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if abs(arr[i, j]) > .55 else PALETTE["black"])
        upper = arr[np.triu_indices_from(arr, k=1)]; rows.append({"task": task, "mean_pairwise_spearman": np.nanmean(upper), "min": np.nanmin(upper), "max": np.nanmax(upper)})
    if im is not None: fig.colorbar(im, cax=cax, label="Spearman correlation")
    else: cax.axis("off")
    for i, ax in enumerate(axes): panel_label(ax, f"({chr(97+i)})")
    save_figure(fig, out / "FigS9_SHAP_cross_fold_rank_stability", apply_layout=False)
    pd.DataFrame(rows).to_csv(out / "FigS9_SHAP_stability_summary.csv", index=False, encoding="utf-8-sig")


def fig_s10_compact_ad(index: ResultIndex, out: Path) -> None:
    """Compact S10: applicability-domain diagnostics."""
    tasks = ["LOI", "PHRR", "THR", "UL94_V0"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8))
    for ax, task in zip(axes.ravel(), tasks):
        p = _locate_ad(index, task, "outer_predictions_with_AD")
        if not p: add_no_data(ax, "Missing AD predictions"); continue
        d = read_csv_auto(p); x = numeric(d.AD_similarity).to_numpy()
        if task == "UL94_V0":
            prob_col = probability_column(d)
            y = numeric(d.get("sample_log_loss", pd.Series(dtype=float))).to_numpy() if "sample_log_loss" in d else (-(numeric(d.y_true) * np.log(numeric(d[prob_col]).clip(1e-8, 1 - 1e-8)) + (1 - numeric(d.y_true)) * np.log(1 - numeric(d[prob_col]).clip(1e-8, 1 - 1e-8)))).to_numpy()
            ylabel = "Single-sample log loss"
        else:
            y = numeric(d.get("absolute_error", (numeric(d.y_true) - numeric(d.y_pred)).abs())).to_numpy(); ylabel = "Absolute error"
        mask = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[mask], y[mask], s=16, color=task_color(task), alpha=.45, edgecolor="white", linewidth=.15)
        tx, ty = robust_lowess_like(x[mask], y[mask])
        if len(tx): ax.plot(tx, ty, color=PALETTE["red_dark"], lw=1.8)
        ax.axvline(.5, ls="--", lw=.8, color=PALETTE["gray"]); ax.axvline(.7, ls="--", lw=.8, color=PALETTE["gray"]); ax.set_xlabel("Max Tanimoto similarity"); ax.set_ylabel(ylabel); ax.set_title(task_label(task)); clean_axes(ax)
    for i, ax in enumerate(axes.ravel()): panel_label(ax, f"({chr(97+i)})")
    fig.tight_layout(pad=.9, w_pad=.75, h_pad=.75)
    save_figure(fig, out / "FigS10_applicability_domain_diagnostics")


def fig_s11_compact_candidates(index: ResultIndex, out: Path) -> None:
    """Compact S11: complete cross-scenario screening funnel.

    The plotted counts are generated from the same function used for
    Supplementary Table S6A, so the figure and table cannot silently diverge.
    """
    # Local import avoids duplicating the cross-scenario counting rules and
    # keeps Fig. S11 numerically locked to Table S6A.
    from build_paper_tables import table_s6a_screening_summary

    summary = table_s6a_screening_summary(index).copy()
    if summary.empty:
        raise FileNotFoundError("Candidate-screening summary is not available")

    # Table S6A intentionally uses manuscript-facing column labels
    # ("Screening stage", "Candidates (n)").  Accept the legacy internal
    # labels as well so S11 and Table S6A remain generated from the same data.
    stage_col = next((c for c in ["Screening stage", "screening_stage", "stage"] if c in summary.columns), None)
    count_col = next((c for c in ["Candidates (n)", "candidate_count", "count"] if c in summary.columns), None)
    if stage_col is None or count_col is None:
        raise KeyError(f"Unexpected Table S6A columns: {list(summary.columns)}")
    labels = summary[stage_col].astype(str).tolist()
    vals = numeric(summary[count_col]).fillna(0).astype(int).tolist()
    pretty_labels = [
        x.replace("m^-2", "m$^{-2}$").replace("cross-flux", "cross-scenario")
        for x in labels
    ]

    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    xmax = max(vals) if vals else 1
    alphas = np.linspace(.92, .52, len(vals))
    for i, (label, v, alpha) in enumerate(zip(pretty_labels, vals, alphas)):
        ax.barh(
            i, v, height=.66,
            color=SERIES_COLORS[i % len(SERIES_COLORS)],
            alpha=float(alpha),
        )
        if v >= 0.42 * xmax:
            ax.text(v / 2, i, f"{label}: {v}", ha="center", va="center", fontsize=8.1)
        else:
            ax.text(v + 0.015 * xmax, i, f"{label}: {v}", ha="left", va="center", fontsize=8.1)

    ax.invert_yaxis()
    ax.set_xlim(0, xmax * 1.20)
    ax.set_yticks([])
    ax.set_xlabel("Candidate molecules")
    ax.set_title("Candidate-screening funnel: formal pool to final prioritization")
    clean_axes(ax, grid="x")
    fig.tight_layout(pad=.9)
    save_figure(fig, out / "FigS11_candidate_screening_funnel")

    summary[[stage_col, count_col]].rename(columns={stage_col: "stage", count_col: "count"}).to_csv(
        out / "FigS11_funnel_data_used.csv", index=False, encoding="utf-8-sig"
    )

def _clean_legacy_supplementary_outputs(out: Path) -> None:
    """Remove stale numbered supplementary exports before frozen-SI regeneration.

    This prevents obsolete S12/S13 and legacy full-SI filenames from being
    mistaken for the frozen S2-S11 outputs. FigS1 is intentionally untouched.
    """
    import re as _re
    if not out.exists():
        return
    for path in out.iterdir():
        if not path.is_file():
            continue
        m = _re.match(r"FigS(\d+)(?:_|\.)", path.name, flags=_re.I)
        if not m:
            continue
        n = int(m.group(1))
        if 2 <= n <= 25 and path.suffix.lower() in {".pdf", ".png", ".svg", ".csv"}:
            path.unlink(missing_ok=True)


def main()->None:
    apply_supplementary_style()
    args=parse_args();args.output.mkdir(parents=True,exist_ok=True)
    _clean_legacy_supplementary_outputs(args.output)
    plt.rcParams["savefig.dpi"] = args.dpi
    plt.rcParams["figure.dpi"] = min(max(220, args.dpi // 2), 450)
    df=read_csv_auto(args.data);index=ResultIndex(args.results_root);manifest=[]
    jobs=[
        ("S2", "Dataset completeness and target distributions.", fig_s2_compact_dataset, (df,args.output)),
        ("S3", "Chemical-space and scaffold diversity of the DOPO/EP dataset.", fig_s3_compact_chemical_space, (df,args.output)),
        ("S4", "Frozen task-specific configurations and model-selection frequencies across the FINAL outer folds.", fig_s4_frozen_configs, (index,args.output)),
        ("S5", "Dummy baselines and Y-scrambling null tests.", fig_s5_null_tests, (index,args.output)),
        ("S6", "Learning curves as a function of independent training-group size.", fig_s6_learning_curves, (index,args.output)),
        ("S7", "Formal P-C/P-N BDE dataset composition and prediction diagnostics.", fig_s7_compact_bde, (index,args.bde_data,args.output)),
        ("S8", "SHAP beeswarm plots for the core prediction tasks.", fig_s8_shap, (index,args.output)),
        ("S9", "Cross-fold stability of SHAP feature rankings for the core prediction tasks.", fig_s9_compact_shap_stability, (index,args.output)),
        ("S10", "Supplementary applicability-domain diagnostics for the core prediction tasks.", fig_s10_compact_ad, (index,args.output)),
        ("S11", "Candidate-screening funnel from the formal screening library to the final cross-scenario priority set.", fig_s11_compact_candidates, (index,args.output)),
    ]
    for name,caption,func,func_args in jobs:
        try:
            func(*func_args)
            manifest.append({"item":name,"caption":caption,"status":"generated","error":""})
            print(f"[OK] {name}")
        except Exception as exc:
            manifest.append({"item":name,"caption":caption,"status":"skipped","error":f"{type(exc).__name__}: {exc}"})
            print(f"[SKIP] {name}: {exc}")
    manifest.insert(0,{"item":"S1","caption":"Verified literature inclusion and data-extraction workflow.","status":"manual","error":"Use the real literature-search log; initial counts must not be estimated."})
    pd.DataFrame(manifest).to_csv(args.output/"supplementary_figure_manifest.csv",index=False,encoding="utf-8-sig")
    index.save_manifest(args.output/"selected_input_files.csv")
    write_json(args.output/"run_config.json",vars(args))
    print(f"\n[DONE] Frozen supplementary figures S2-S11: {args.output}")


if __name__=="__main__":main()
