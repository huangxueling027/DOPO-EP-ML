# 01_FlameRetardancy

本目录只保留阻燃核心任务的**开发阶段任务入口**：

- `LOI/run_LOI.py`
- `PHRR/run_PHRR.py`
- `THR/run_THR.py`
- `UL94/run_UL94.py`

这些入口调用冻结的 `config/task_config.json` 和 `common/` 共享管道，主要用于 repeated-group-holdout 开发诊断。

**论文正式性能不从本目录的开发结果中选取。** 正文主结果统一来自 `08_ScientificValidation/` 下的严格 5×5 nested grouped validation。

推荐用户从项目根目录运行：

```bash
python -u run.py main --tasks LOI,PHRR,THR,UL94_V0
```

若当前正式结果已经冻结，仅做论文整理时，不需要重新运行本目录。
