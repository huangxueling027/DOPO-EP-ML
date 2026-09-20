# Frozen table alignment for manuscript v1.0.2

This folder is aligned to the current `Manuscript` and `Supplementary Information`.
The changes below are presentation/documentation changes only; no model is retrained.

## Main-text tables

`build_main_text_tables.py` now generates exactly three manuscript tables:

1. **Table 1.** Outer-test performance under strict 5×5 nested grouped cross-validation.
2. **Table 2.** Grouping robustness, BDE ablation, and cross-fold SHAP stability of the core flame-retardancy models.
3. **Table 3.** Top 10 cross-scenario priority DOPO-derived candidates identified by multi-objective virtual screening.

Legacy five-table outputs are deleted before regeneration.

## Supplementary tables

`build_paper_tables.py` now generates exactly the numbered table set used in the SI:

- Table S1(A): dataset size, structural coverage, and target distributions;
- Table S1(B): database variables and raw-field availability;
- Table S2(A): frozen task-specific configurations;
- Table S2(B): candidate models and preprocessing;
- Table S3: strict 5×5 grouped nested-validation performance;
- Table S4: formal P–C/P–N BDE error summary;
- Table S5: stable SHAP features;
- Table S6(A): screening-stage summary;
- Table S6(B): final 19 cross-flux priority candidates;
- Table S7(A): external-validation performance;
- Table S7(B): external molecular/AD audit.

Key consistency locks:

- Tg description is **measured by dynamic mechanical analysis (DMA)**.
- Table S6(B) is explicitly the final **19** candidates.
- External validation references are frozen to **[43]–[48]** for MFD, MBFAP, SPDO, VH-DOPO, VPAA-DOPO and DMM.
- `Data_all_BDE_prediction_errors.csv` remains the full machine-readable repeated-OOF BDE table used to support Table S4.
- Detailed machine-readable `Data_*.csv` files remain unnumbered and do not create extra SI table numbers.

## Recommended rerun

From the project root:

```bash
D:/Anaconda/python.exe -u 09_PaperFigures/build_paper_tables.py
D:/Anaconda/python.exe -u 09_PaperFigures/build_main_text_tables.py
D:/Anaconda/python.exe -u 09_PaperFigures/make_paper_figures.py --dpi 900
```

Or use the project wrapper if already configured:

```bash
D:/Anaconda/python.exe -u run.py paper-all
```

After regeneration, check:

- `results/09_PaperTables/paper_table_manifest.csv`
- `results/09_MainTextTables/main_text_table_manifest.csv`
- `results/09_PaperFigures/supplementary/supplementary_figure_manifest.csv`

The numbered outputs should expose only **Table 1–3**, **Table S1(A-B)–S7(A-B)**, and **Fig. S2–S11**.
