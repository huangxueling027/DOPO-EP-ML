# 05_Shap：跨外层折 SHAP 稳定性

正式解释基于严格 nested grouped validation 的**外层测试折**，不是开发阶段单次划分。

## 1. 核心任务

```bash
python -u run.py shap --tasks LOI,PHRR,THR,UL94_V0 --results results/05_Shap/FINAL_core_fixed_baseline_inclusive_5x5
```

`run_shap_stability.py` 现在会依次完成：

```text
严格 5×5 SHAP stability
→ 稳定性图
→ High/Medium child features
→ parent-variable aggregation
```

规范输出：

```text
results/05_Shap/FINAL_core_fixed_baseline_inclusive_5x5/
results/05_Shap/FINAL_core_fixed_baseline_inclusive_5x5/figures/
results/05_Shap/core_features/
```

默认 Top-N = 15。

## 2. Tg / TS 辅助任务

```bash
python -u 05_Shap/run_step3_tg_ts_shap.py
```

规范输出：

```text
results/05_Shap/FINAL_aux_fixed_baseline_inclusive_5x5/
results/05_Shap/auxiliary_figures/
results/05_Shap/auxiliary_features/
```

如正式 SHAP CSV 已存在，只需要重画/重建表格：

```bash
python -u 05_Shap/run_step3_tg_ts_shap.py --skip-evaluation
```

## 3. 论文解释规则

- LOI / PHRR / THR / UL94：正文优先讨论 High/Medium 稳定父变量；
- Tg / TS：若折间 Spearman 较低，只讨论重复出现的特征组或父变量；
- Char_yield / FS：不做正式 SHAP 机理结论；
- SHAP 表示模型在当前数据分布中的统计贡献，不直接等同于因果机制；
- Morgan 位点的投稿版结构图仍建议用 ChemDraw 统一重画。

论文表 S7 只应汇总 High/Medium 稳定特征。


## FINAL protocol
Formal SHAP uses `fixed + baseline_inclusive + molecule + formulation + without_BDE + 5x5`. The view/K are predefined; only the predictive model is selected by grouped inner CV within each outer-training fold.
