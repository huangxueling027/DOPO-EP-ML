# Results included in the GitHub release

This repository keeps manuscript-facing summary tables, outer predictions, downstream analysis outputs, figure source data, and the small final frozen model bundles required by screening/external-validation workflows.

To keep the repository lightweight, large fold-level `.joblib` model binaries from grouped-CV sensitivity, BDE-ablation, and SHAP analyses are intentionally omitted. These binaries are reproducible from the frozen code/data workflow and are not required to inspect the reported manuscript metrics.

Included result families:

- `scientific_validation/FINAL_core_fixed_baseline_inclusive_5x5`
- `scientific_validation/FINAL_aux_fixed_baseline_inclusive_5x5`
- `scientific_validation/FINAL_exploratory_fixed_baseline_inclusive_5x5`
- `scientific_validation/FINAL_Delta_fixed_5x5`
- `scientific_validation/FINAL_grouping_sensitivity_5x5`
- `scientific_validation/FINAL_BDE_paired_ablation_5x5`
- aggregate `FINAL_information_source_ablation_5x5` tables
- FINAL SHAP, applicability-domain, reverse-design, external-validation, paper-table, and main-figure outputs

The `config/final_model_manifest.json` file defines the frozen manuscript-facing result paths.
