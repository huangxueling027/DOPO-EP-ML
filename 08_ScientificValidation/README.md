# 08_ScientificValidation：严格科学验证

正式论文性能统一来自**fixed inner-CV 选择 + 5×5 nested grouped cross-validation**。
外层测试折只用于泛化评价；不能根据 outer-test 结果重新选择 view、K 或模型池。

## 1. 当前正式结果目录

```text
results/scientific_validation/
├─ FINAL_core_fixed_baseline_inclusive_5x5/             # LOI / PHRR / THR / UL94，with+without BDE
├─ FINAL_aux_fixed_baseline_inclusive_5x5/        # Tg / TS，with+without BDE
├─ FINAL_exploratory_fixed_baseline_inclusive_5x5/   # Char / FS
├─ FINAL_Delta_fixed_5x5/         # Delta 四任务
├─ core_three_split_validation/         # molecule/scaffold/reference
├─ information_source_ablation/
├─ null_tests/
├─ learning_curves/
└─ outer_error_analysis/
```

模型适用域结果保存于：

```text
results/06_ApplicabilityDomain/core_fixed_5x5/
```

## 2. 主验证入口

统一底层入口：

```text
08_ScientificValidation/run_scientific_evaluation.py
```

如果确实需要从零重跑四组正式结果，可执行：

```bash
python -u 08_ScientificValidation/run_all_scientific.py
```

先只查看将要执行的四组命令：

```bash
python -u 08_ScientificValidation/run_all_scientific.py --dry-run
```

该脚本现在只使用 `selection-scope=fixed`，并按核心/辅助/探索/Delta 分开输出。
**如果当前正式结果已经存在且通过 validate，不需要为了论文排版重新运行。**

## 3. 论文增强验证

根目录统一命令：

```bash
python -u run.py split-validation
python -u run.py source-ablation
python -u run.py null-tests
python -u run.py learning-curves
python -u run.py error-analysis
python -u run.py ad
```

用途：

- `run_split_validation.py`：molecule / scaffold / reference 分组敏感性；
- `run_information_source_ablation.py`：conditions-only / molecular-only / combined / combined+BDE；
- `run_null_tests.py`：Dummy + Y-scrambling；
- `run_learning_curves.py`：分子分组学习曲线；
- `analyze_outer_errors.py`：外层测试误差案例；
- `evaluate_applicability_domain.py`：结构相似度 AD 与阈值敏感性。

## 4. 其他脚本定位

- `run_smoke_tests.py`：2×2 快速流程检查，结果属于 smoke，确认后删除；
- `run_screening_mode_ablation.py`：formulation vs molecular-first 诊断，不替代主结果；
- `run_tabpfn_comparison_scientific.py`：可选文献/算法比较，需要 optional requirements；
- `run_data_standardization.py`：只生成标准化审计，不覆盖冻结原始数据；
- `validate_zero_loading_and_delta.py`：Neat EP / Delta 规则检查。

## 5. 论文使用原则

- 核心主结果：LOI、PHRR、THR、UL94；
- 辅助：Tg、TS；
- 探索：Char_yield、FS；
- Delta：补充分析；
- BDE：严格同折消融 + 独立 BDE 模型，不把单次提升当作普遍增益；
- learning curves / null tests：科学可信度补充，不与正式 nested 结果混为同一种评价。


## Frozen FINAL protocol
Primary absolute-property evaluation is `fixed + baseline_inclusive + molecule + formulation + without_BDE + 5x5`. Task view/K are predefined from the V4 development scans; grouped inner CV selects the predictive model only. Modified-only is sensitivity analysis, not the primary manuscript model.
