# DOPO-EP-ML

Reproducible machine-learning workflow for **DOPO-derived flame-retardant epoxy systems**, covering literature-derived data standardization, structure–formulation modeling, grouped nested validation, model interpretation, applicability-domain analysis, independent literature-based external validation, and multi-objective virtual screening.

## Study data

The curated dataset contains:

* 599 standardized experimental records
* 130 published studies
* 138 unique main flame retardants
* 143 standardized main flame-retardant/synergist combinations
* 117 Murcko scaffolds

Original publisher PDFs are not redistributed. The frozen machine-readable dataset and source-traceability fields are provided in `data/`.

## Formal manuscript protocol

The manuscript-facing models use the frozen protocol:

`FINAL_FIXED_BASELINE_INCLUSIVE_5X5`

The formal workflow uses:

* 5 outer folds × 5 inner folds
* molecule-grouped nested cross-validation
* baseline-inclusive absolute-property modeling
* Neat-EP leakage-protection masking
* predefined task-specific feature view and K
* predictive-model selection only within grouped inner cross-validation
* primary models without BDE features

### Frozen task configurations

| Task       | Feature view     |   K | Role             |
| ---------- | ---------------- | --: | ---------------- |
| LOI        | compact          | 260 | Core             |
| PHRR       | compact          | 330 | Core             |
| THR        | descriptors      |  30 | Core             |
| UL94_V0    | morgan_r3        | ALL | Core             |
| Tg         | morgan_r3        | 270 | Auxiliary        |
| TS_MPa     | compact          | 140 | Auxiliary        |
| Char_yield | full_interaction | 100 | Exploratory / SI |
| FS_MPa     | morgan_r3        | 240 | Exploratory / SI |

## Main grouped-CV results

| Task      | Primary metric |        Result |
| --------- | -------------- | ------------: |
| LOI       | R²             | 0.716 ± 0.036 |
| PHRR      | R²             | 0.604 ± 0.042 |
| THR       | R²             | 0.483 ± 0.133 |
| UL-94 V-0 | Macro-F1       | 0.779 ± 0.031 |
| Tg        | R²             | 0.521 ± 0.097 |
| TS        | R²             | 0.462 ± 0.089 |

Values are reported as mean ± standard deviation across the outer test folds of the grouped nested cross-validation procedure.

Full manuscript-facing tables and source data are provided under `results/09_MainTextTables/` and `results/09_PaperTables/`.

## Virtual screening

The frozen screening workflow evaluated 409 candidate molecules. Under joint applicability-domain constraints, 213 candidates were eligible under both 35 and 50 kW m⁻² scenarios, and 19 cross-flux robust priority candidates were retained.

## Repository structure

```text
DOPO-EP-ML/
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

Before running individual analyses, validate the frozen project state:

```bash
python -u run.py validate
```

The validation command checks locked data hashes, row counts, molecular-identity rules, Python syntax, and project-level consistency.

## Main reproduction commands

```bash
# Formal manuscript models
# Rerun only when regeneration is genuinely required.
python -u run.py final-models

# Robustness and supplementary validation
python -u run.py final-robustness

# SHAP and applicability-domain analyses
python -u run.py final-interpretation

# Designed and public-database virtual screening
python -u run.py final-screening

# Independent literature-based external validation
python -u run.py external

# Audit manuscript-facing outputs
python -u run.py paper-audit

# Regenerate manuscript tables and figures
python -u run.py paper-all
```

Completed FINAL result directories are retained for manuscript traceability. They should not be rerun merely because the repository has been reorganized.

## Result storage policy

Large fold-level model binaries are omitted from the release where they are not required for reproducibility.

The repository retains the frozen model bundles required by downstream screening and external-validation workflows, together with manuscript-facing predictions, evaluation metrics, SHAP and applicability-domain outputs, screening tables, and figure source data.

See `results/README.md` for details.

## Data integrity

SHA-256 hashes for the frozen datasets are stored in `PROJECT_RULES_LOCK.json`.

Do not edit and resave the locked CSV files unless intentionally creating a new data release.

Run:

```bash
python -u run.py validate
```

after cloning the repository or before reproducing manuscript analyses.

## Citation

Citation information will be updated after publication.

Until then, users reusing the curated data should cite the associated manuscript and the original literature sources represented in the dataset.

## License

Source code in this repository is released under the MIT License unless otherwise stated.

The literature-derived datasets, published experimental measurements, source-traceability information, and other third-party materials are not automatically covered by the software license and remain subject to their original sources and applicable terms.
