# 04_Delta

Delta 任务开发入口：

- `Delta_LOI/run_Delta_LOI.py`
- `Delta_PHRR/run_Delta_PHRR.py`
- `Delta_THR/run_Delta_THR.py`
- `Delta_CY/run_Delta_CY.py`

这些脚本仅作为**开发阶段任务入口**，用于复现与诊断；论文正式 Delta 性能统一由 `08_ScientificValidation/` 生成。

## Delta 的最终定义

Delta 任务用于分析 DOPO 改性配方相对于匹配 Neat EP 基线的性能变化，属于**补充/探索分析**，不替代绝对性能模型。

最终规则固定为：

- Neat EP 用于构建/匹配 baseline；
- `Loading_total_FR = 0` 的 Neat EP 行**不作为 Delta target sample**；
- Delta 正式评价仅针对具有有效 Delta 目标的改性配方；
- 正式评价使用 molecule-grouped 5×5 nested CV；
- view/K 使用开发阶段预先确定的 fixed 配置；
- 主模型使用 without BDE。

即：

```text
Neat EP = baseline source
Neat EP != Delta prediction target
```

论文正式 Delta 结果统一保存到：

```text
results/scientific_validation/FINAL_Delta_fixed_5x5/
```

如果该正式结果已经存在并通过项目审计，论文整理阶段不需要重新运行本目录中的开发入口。
