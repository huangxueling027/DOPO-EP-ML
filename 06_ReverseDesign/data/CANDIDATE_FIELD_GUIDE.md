# manual_candidates_template.csv 填写说明

## 推荐编号

- `CAND_PN_001`：P-N候选
- `CAND_PSI_001`：P-Si候选
- `CAND_PB_001`：P-B候选
- `CAND_BIS_001`：双DOPO候选
- `LIT_001`：文献候选

## Source_Type

- `literature_candidate`：有文献结构，但未纳入训练数据；
- `designed`：在已知种子基础上自行设计；
- 不要写 `training_seed`，该类别由程序自动生成。

## Synthesis_Level

- A：文献已报道、商业可得或1–2步路线明确；
- B：预计2–3步可合成，原料和反应类型合理；
- C：路线复杂、稳定性或化合价存疑，仅保留记录，不进入正式排序。

## SMILES

1. 在ChemDraw中画完整结构；
2. 检查化合价；
3. 复制为SMILES；
4. 一行只放一个单一分子；
5. 不要包含溶剂、催化剂、离子对或多个点号组分。
