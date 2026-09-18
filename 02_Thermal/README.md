# 02_Thermal

热性能任务开发入口：

- `Tg/run_Tg.py`
- `Char/run_Char_yield.py`

定位：

- **Tg**：辅助任务；
- **Char_yield**：探索性任务。

这里的脚本用于开发阶段 repeated-group-holdout。论文正式结果统一读取 `08_ScientificValidation/` 的严格 5×5 nested grouped validation，不用开发阶段最高分替代正式外层结果。

如果正式结果已经存在，论文整理阶段不需要重跑本目录。
