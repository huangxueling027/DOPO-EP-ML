# 06_ReverseDesign：DOPO 专属候选结构库与 combined 虚拟筛选

当前流程分为 **designed-only 基线** 和 **designed + PubChem combined 最终筛选**。

```text
训练集DOPO种子 / designed候选
→ RDKit标准化、去重、DOPO核心检查
→ designed-only 50/35筛选（基线）

PubChem原始候选
→ 标准化
→ 与训练集/designed去重
→ 公共候选AD门控
→ combined candidate master
→ combined 50/35六任务预测
→ AD + Pareto
→ 35/50稳定优先级
→ Fig10、FigS23–S25、Table5、TableS9、TableS11
```

## 重要原则

1. 分子表、配方表、预测表分开。
2. 候选预测只调用冻结模型，不在候选上重新训练/调参。
3. designed 新结构必须有明确 SMILES；代码不随机制造化学结构。
4. PubChem 中存在某个结构，不等于已实验验证可合成。
5. PubChem 的 `Preparation_Method` 是结构驱动的**筛选场景假设**，不是实验事实。
6. 最终论文候选以 combined 结果为主，designed-only 结果保留作为基线/来源比较。

## 1. designed 候选入口

人工编辑：

```text
06_ReverseDesign/data/manual_candidates_template.csv
```

不要在该文件填写 LOI、PHRR、添加量或模型预测值。

建立 designed 候选库：

```bash
python -u run.py candidate-init
python -u run.py candidate-build
```

主要输出：

```text
results/06_ReverseDesign/candidate_molecule_master.csv
results/06_ReverseDesign/candidate_exclusion_report.csv
```

## 2. designed-only 50/35 基线

50 kW/m²：

```bash
python -u run.py candidate-formulations   --cone-fluxes 50 --max-loading 30   --output results/06_ReverseDesign/final_flux50/candidate_formulation_grid.csv
python -u run.py candidate-screen   --formulations results/06_ReverseDesign/final_flux50/candidate_formulation_grid.csv   --output-dir results/06_ReverseDesign/final_flux50
```

35 kW/m²：

```bash
python -u run.py candidate-formulations   --cone-fluxes 35 --max-loading 30   --output results/06_ReverseDesign/sensitivity_flux35/candidate_formulation_grid.csv
python -u run.py candidate-screen   --formulations results/06_ReverseDesign/sensitivity_flux35/candidate_formulation_grid.csv   --output-dir results/06_ReverseDesign/sensitivity_flux35
```

这些结果不删除，但不再作为论文最终候选池。

## 3. PubChem 公共候选

```bash
python -u 06_ReverseDesign/public_database/standardize_pubchem_candidates.py
python -u 06_ReverseDesign/public_database/build_public_candidate_pool.py
```

关键文件：

```text
06_ReverseDesign/public_database/data/02_pubchem_standardized/pubchem_standardized_eligible.csv
06_ReverseDesign/public_database/data/03_public_candidates/pubchem_public_candidates_for_prediction.csv
```

## 4. 构建 combined master

```bash
python -u 06_ReverseDesign/build_combined_candidate_master.py
```

输出：

```text
results/06_ReverseDesign/combined/candidate_molecule_master_combined.csv
results/06_ReverseDesign/combined/combined_candidate_master_audit.csv
results/06_ReverseDesign/combined/combined_candidate_source_summary.csv
```

当前审计：base master 558 行（420 designed + 138 training seed），PubChem 正式候选 54 行，combined master 612 行；正式 screening candidates 为 409（355 designed + 54 PubChem）。

## 5. combined 50/35 正式筛选

`run.py candidate-formulations` 当前没有 `--master` 参数，所以 combined 配方网格直接调用 `build_formulation_grid.py`。

50 kW/m²：

```bash
python -u 06_ReverseDesign/build_formulation_grid.py   --master results/06_ReverseDesign/combined/candidate_molecule_master_combined.csv   --cone-fluxes 50 --max-loading 30   --output results/06_ReverseDesign/combined_flux50/candidate_formulation_grid.csv
python -u run.py candidate-screen   --formulations results/06_ReverseDesign/combined_flux50/candidate_formulation_grid.csv   --output-dir results/06_ReverseDesign/combined_flux50
```

35 kW/m²：

```bash
python -u 06_ReverseDesign/build_formulation_grid.py   --master results/06_ReverseDesign/combined/candidate_molecule_master_combined.csv   --cone-fluxes 35 --max-loading 30   --output results/06_ReverseDesign/combined_flux35/candidate_formulation_grid.csv
python -u run.py candidate-screen   --formulations results/06_ReverseDesign/combined_flux35/candidate_formulation_grid.csv   --output-dir results/06_ReverseDesign/combined_flux35
```

六任务：LOI、UL-94 V-0 probability、PHRR、THR、Tg、TS。

适用域默认：Tanimoto ≥0.70 In_domain；0.50–0.70 Caution；<0.50 Extrapolation。

当前两种热流均为：

```text
409 candidates
→ 213 at least one In-domain/formally eligible formulation
→ 86 Pareto candidates
```

## 6. 35/50 稳定最终候选

最终论文读取：

```text
results/06_ReverseDesign/final_priority_combined/final_priority_candidates_combined.csv
results/06_ReverseDesign/final_priority_combined/final_top10_candidates_combined.csv
results/06_ReverseDesign/final_priority_combined/flux35_50_rank_stability_combined.csv
results/06_ReverseDesign/final_priority_combined/flux35_50_stability_summary_combined.csv
results/06_ReverseDesign/final_priority_combined/candidate_source_comparison.csv
```

当前：Top10 10/10、Top20 20/20、Spearman≈0.998、Pareto overlap 86/86、最终 20 个 priority candidates。

来源比较：designed 有 20 个进入 final priority；PubChem 有 9 个进入两热流共同 Pareto，但没有进入最终 Top20。

> 当前项目没有在 `run.py` 中暴露专门重建 `final_priority_combined/` 的子命令。补齐该入口以前，不要删除这组冻结结果。

## 7. 生成论文图表

```bash
python -u run.py paper-audit
python -u run.py paper-all
```

最终：Fig10 / FigS23–S25 / Table5 / TableS9 / TableS11 使用 combined 结果。

## 8. legacy

`legacy/` 只用于版本追溯，不参与正式流程。`candidate_applicability_domain.py` 和 `rank_candidates.py` 可保留为单步诊断工具。

## 7. PubChem → combined 的完整可复现链

本目录现在补齐两个原先缺失的生成步骤：

- `public_database/prepare_public_candidates_for_prediction.py`：
  从 `pubchem_public_AD.csv` 中严格筛出 `Applicability_domain == In_domain` 的公共候选，并生成 `pubchem_public_candidates_for_prediction.csv`；
- `build_flux_stable_priority.py`：
  合并 combined 50 / 35 kW/m² 的候选排名，计算 Spearman、Top-k overlap、Pareto overlap，并生成最终 20 个稳定候选。

如果已有正式上游 5×5 模型 bundle，可直接运行整个 combined 流程：

```bash
python -u 06_ReverseDesign/run_combined_screening.py
```

该脚本依次执行：

```text
PubChem raw CSV
→ standardize
→ deduplicate against training/designed
→ public AD
→ keep In_domain public candidates
→ build combined master
→ combined 50 kW/m² formulation + screening
→ combined 35 kW/m² formulation + screening
→ 35/50 stable final priority
```

默认最终输出：

```text
results/06_ReverseDesign/combined/
results/06_ReverseDesign/combined_flux50/
results/06_ReverseDesign/combined_flux35/
results/06_ReverseDesign/final_priority_combined/
```

注意：这个流程不会联网重新抓取 PubChem；它从已经冻结在
`06_ReverseDesign/public_database/data/01_pubchem_raw/` 的原始检索文件开始重建。


## FINAL model source
All candidate predictions must use `config/final_model_manifest.json`. No curated/V4/smoke fallback is allowed. Main deployment bundles are baseline-inclusive fixed-view/K and without BDE.
