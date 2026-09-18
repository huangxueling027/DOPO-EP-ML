# ML_DOPO_Project

Reproducible machine-learning workflow for **DOPO-derived flame-retardant epoxy systems**, including literature-derived data standardization, grouped nested validation, model interpretation, applicability-domain analysis, independent literature-based external validation, and multi-objective virtual screening.

## Study data

- 599 standardized experimental records
- 130 published studies
- 138 unique main flame retardants
- 143 main flame-retardant/synergist combinations
- 117 Murcko scaffolds

Original publisher PDFs are not redistributed. The frozen machine-readable dataset and source-traceability fields are provided in `data/`.

## Formal manuscript protocol

The manuscript-facing models use the frozen protocol `FINAL_FIXED_BASELINE_INCLUSIVE_5X5`:

- 5 outer folds × 5 inner folds
- molecule-grouped nested cross-validation
- baseline-inclusive absolute-property modeling
- Neat-EP leakage-protection masking
- predefined task-specific feature view/K
- predictive-model selection only within grouped inner CV
- primary models without BDE features

### Frozen task configurations

| Task | Feature view | K | Role |
|---|---|---:|---|
| LOI | compact | 260 | Core |
| PHRR | compact | 330 | Core |
| THR | descriptors | 30 | Core |
| UL94_V0 | morgan_r3 | ALL | Core |
| Tg | morgan_r3 | 270 | Auxiliary |
| TS_MPa | compact | 140 | Auxiliary |
| Char_yield | full_interaction | 100 | Exploratory / SI |
| FS_MPa | morgan_r3 | 240 | Exploratory / SI |

## Main grouped-CV results

| Task | Primary metric | Result |
|---|---|---|
| LOI | R² | 0.716 ± 0.036 |
| PHRR | R² | 0.604 ± 0.042 |
| THR | R² | 0.483 ± 0.133 |
| UL-94 V-0 | Macro-F1 | 0.779 ± 0.031 |
| Tg | R² | 0.521 ± 0.097 |
| TS | R² | 0.462 ± 0.089 |

Full manuscript-facing tables and source data are provided under `results/09_MainTextTables/` and `results/09_PaperTables/`.

## Virtual screening

The frozen screening workflow evaluated 409 candidate molecules. Under joint applicability-domain constraints, 213 candidates were eligible under both 35 and 50 kW m⁻² scenarios, and 19 cross-flux robust priority candidates were retained.

## Repository structure

```text
ML_DOPO_Project/
├── data/
├── config/
├── common/
├── 01_FlameRetardancy/
├── 02_Thermal/
├── 03_Mechanical/
├── 04_Delta/
├── 05_Shap/
├── 06_ReverseDesign/
├── 07_BDE/
├── 08_ScientificValidation/
├── 09_PaperFigures/
├── 10_ExternalValidation/
├── results/
├── tools/
├── run.py
├── requirements.txt
└── PROJECT_RULES_LOCK.json
```

## Installation

A clean Python environment is recommended.

```bash
pip install -r requirements.txt
```

## Validate the frozen release

```bash
python -u run.py validate
```

The validation command checks locked data hashes, row counts, molecular identity rules, syntax, and project-level consistency.

## Main reproduction commands

```bash
# Formal models (only rerun when genuinely required)
python -u run.py final-models

# Robustness and supplementary validation
python -u run.py final-robustness

# SHAP + applicability-domain analysis
python -u run.py final-interpretation

# Designed + public-database virtual screening
python -u run.py final-screening

# Independent literature-based external validation
python -u run.py external

# Audit / regenerate manuscript tables and figures
python -u run.py paper-audit
python -u run.py paper-all
```

The completed FINAL result directories are already provided for manuscript traceability. Do not rerun them merely because the repository was reorganized.

## Result storage policy

For a practical GitHub release, large fold-level model binaries are omitted. The repository retains the small final frozen model bundles needed by downstream screening/external-validation workflows, together with manuscript-facing predictions, metrics, SHAP/AD outputs, screening tables, and figure source data. See `results/README.md`.

## Data integrity

The frozen data SHA256 hashes are stored in `PROJECT_RULES_LOCK.json`. Do not edit and resave the locked CSV files unless intentionally creating a new data release.

## Citation

Citation details will be added after publication. Until then, please cite the associated manuscript and the original literature sources represented in the curated dataset when reusing the data.

## License

No open-source license has been assigned in this preparation package yet. Select and add an appropriate license before making the repository public.
