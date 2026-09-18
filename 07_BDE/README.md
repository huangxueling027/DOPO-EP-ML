# Independent BDE model

- `BDE.py`：正式P–C/P–N BDE模型和下游BDE特征生成。
- `BDE_opt.py`：可选优化搜索；只有25次重复评价均值稳定高于主脚本时才替换主结果。
- `data/BDE.csv`：原始训练数据。
- `data/BDE_pc_pn.csv`：P–C/P–N建模子集。
- `data/BDE_excluded.csv`：排除样本。
- `template/predict.csv`：预测模板。

运行：

```bash
python -u run.py bde-model
```

BDE主任务结果应独立报告。将BDE加入LOI/PHRR等模型时，必须使用严格with/without消融，不能只展示单次提升。


## V5 FINAL paper policy
The independent BDE model is retained as-is. BDE is **not** a main-property input in the FINAL models. The manuscript paired with/without-BDE ablation must be generated through `08_ScientificValidation/run_bde_comparison_scientific.py` using `fixed + baseline_inclusive + molecule + formulation + 5x5`; the without-BDE branch remains the primary model.
