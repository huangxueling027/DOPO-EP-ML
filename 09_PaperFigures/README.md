# 论文图、补充图和表格模块

所有数值图表只读取冻结数据库或正式结果 CSV；不手工填指标，不用训练集结果替代 outer-test 结果。

## 1. 统一风格

- 马卡龙色系，语义配色统一；
- 输出 `PDF + SVG + 600 dpi PNG`；
- 回归主指标：R² / RMSE / MAE；
- UL-94 主指标：Macro-F1，并补充 Accuracy、Balanced Accuracy、ROC-AUC；
- 外层折负 R² 原样保留；
- 图 3 只读取 5 个外层测试折汇总预测，并自动标注 Model / View / K。

## 2. 脚本

| 脚本 | 作用 |
|---|---|
| `audit_paper_inputs.py` | 审计每张图/表的输入、5折完整性和真实选用路径 |
| `make_main_figures.py` | 生成正文 Fig2–Fig10 |
| `make_supplementary_figures.py` | 生成 compact FigS2–FigS13；FigS1 单独保留 |
| `build_main_text_tables.py` | 生成正文 Table1–Table5 |
| `build_paper_tables.py` | 生成 Supplementary TableS1–TableS11 |
| `run_all_paper_outputs.py` | 一次完成审计、图和表 |
| `paper_utils.py` | 结果定位、分子标准化、绘图公共函数 |

### 2026-08-17 表格修正

- TableS5：`source_file` 不再参与科学结果去重，优先保留正式 `scientific_validation/final_*` 来源；
- TableS7：优先读取 `*_SHAP_stable_features_filtered.csv`，仅保留 High/Medium 稳定特征；
- TableS9：combined 50 kW/m² 全配方预测；
- TableS11：combined 35/50 最终 20 priority candidates。

## 3. 最短运行方式

建议先把旧的三个 `09_*` 输出目录移到项目外备份，再运行：

```bash
python -u run.py validate
python -u run.py paper-audit
python -u run.py paper-all
```

输出：

```text
results/09_PaperFigures/
├─ audit/
├─ main/
└─ supplementary/
results/09_PaperTables/
results/09_MainTextTables/
```

## 4. 必查 manifest

- `audit/paper_input_audit.csv`
- `audit/outer_prediction_integrity.csv`
- `main/main_figure_manifest.csv`
- `supplementary/supplementary_figure_manifest.csv`
- `09_PaperTables/paper_table_manifest.csv`
- `09_MainTextTables/main_text_table_manifest.csv`
- 各目录 `selected_input_files.csv`

最终应登记：Table1–5、TableS1–S11、Fig2–10、FigS2–S25。

## 5. 正文图

- Fig1：PowerPoint + ChemDraw 手工研究流程；
- Fig2：数据集组成与化学空间；
- Fig3：LOI、PHRR、THR、Tg、TS outer predicted vs observed；
- Fig4：外层折稳定性，UL-94 Macro-F1 独立子图，并含三 split 敏感性；
- Fig5：UL-94 混淆矩阵、ROC、PR、校准；
- Fig6：信息来源消融；
- Fig7：BDE 独立模型 + with/without-BDE 同折消融；
- Fig8：SHAP；
- Fig9：适用域；
- Fig10：**combined virtual screening**。

Fig10 最终输入应来自：

```text
results/06_ReverseDesign/combined_flux50/
results/06_ReverseDesign/final_priority_combined/
```

不要误读 designed-only `final_flux50/` 作为最终 Fig10。

## 6. 补充图/表

- FigS1：真实文献检索流程图，初始检索数不能估算；
- FigS2–FigS13：Python/现有正式结果直接生成；
- Table S1–S7：自动生成；详细逐折/逐样本/逐候选结果保留为未编号 `Data_*.csv`；
- Table S10 为最终 20 个跨热流场景优先候选，不包含后续实验模板。

## 7. 仍需人工完成

1. Fig1；
2. FigS1；
3. Word/WPS 中优先使用 PDF/SVG，不修改坐标、指标或负 R²。

本课题不生成 prospective experimental validation template，也不要求候选后续实验。


## FINAL paper-source rule
Paper-facing scripts must prefer/verify the `FINAL_*_fixed_baseline_inclusive_5x5` protocol and must not silently fall back to curated, modified-only, V4, smoke, or 2x2 results.

## FINAL audit/path corrections (2026-09)

The manuscript-facing code now uses the frozen FINAL result tree directly:

- core: `results/scientific_validation/FINAL_core_fixed_baseline_inclusive_5x5`
- auxiliary: `results/scientific_validation/FINAL_aux_fixed_baseline_inclusive_5x5`
- grouping sensitivity: `results/scientific_validation/FINAL_grouping_sensitivity_5x5`
- information-source ablation: `results/scientific_validation/FINAL_information_source_ablation_5x5`
- BDE paired ablation: `results/scientific_validation/FINAL_BDE_paired_ablation_5x5`
- SHAP: `results/05_Shap/FINAL_*_fixed_baseline_inclusive_5x5`
- AD: `results/06_ApplicabilityDomain/FINAL_fixed_5x5`
- screening: `results/06_ReverseDesign/FINAL_fixed`

`Fig. 6` uses three FINAL information scopes (`Combined`, `Conditions only`, `Molecular only`). The with/without-BDE comparison is intentionally kept in `Fig. 7` rather than duplicated in Fig. 6.

`Fig. S7` does **not** require rerunning historical K/view scans. If the old five-seed development `*_model_comparison.csv` files are unavailable, it reports the frozen view/K and the grouped-inner-CV model-selection frequency across the five FINAL outer folds.

Copied nested summaries under SHAP/AD folders are never allowed to outrank the canonical files under `results/scientific_validation/` for manuscript performance figures/tables.


## Compact SI V3 (2026-09)

- Numbered SI tables are reduced to **S1-S7**:
  - S1 compact database dictionary;
  - S2A frozen task configurations + S2B candidate model/preprocessing space;
  - S3 formal strict 5x5 performance;
  - S4 formal P-C/P-N BDE error summary;
  - S5 compact stable SHAP features;
  - S6A screening counts + S6B final 20 candidates;
  - S7A external performance + S7B structural/AD audit.
- Detailed rows are preserved as unnumbered `Data_*.csv` files, not embedded as separate SI tables.
- Fig. S8 uses only the **formal P-C/P-N subset** in the BDE-composition panel (P-C=229, P-N=10 for the frozen dataset); rare P-O/P-S/P-H records are not plotted because they were not used in formal BDE model evaluation.
- No FINAL model, SHAP, AD, BDE model, external-validation model, or screening model is retrained by these presentation-layer changes.
