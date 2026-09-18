# 03_Mechanical

力学性能任务开发入口：

- `TS/run_TS_MPa.py`
- `FS/run_FS_MPa.py`

定位：

- **TS_MPa**：辅助任务；
- **FS_MPa**：探索性任务。

这里的任务脚本用于开发阶段 repeated-group-holdout。论文正式结果来自 `08_ScientificValidation/` 下的严格 5×5 nested grouped validation。

如果正式结果已经冻结，论文整理阶段不需要重跑本目录。
