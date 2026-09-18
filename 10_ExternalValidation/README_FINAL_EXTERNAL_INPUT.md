# FINAL external-validation input fix

The FINAL runner accepts either the audited Excel workbook or the already audited/prepared external CSV.

Preferred input:
`data/external_validation/DOPO_EP_External_Validation_Final_6_SMILES_Audited.xlsx`

Automatic fallback included in this patch:
`data/external_validation/External_Validation_V4_ready.csv`

The `V4_ready` filename describes column-schema compatibility only. No V4 model predictions are reused. The runner still loads only the frozen FINAL V5 bundles through `config/final_model_manifest.json` / `common/final_result_paths.py`, and no model is rebuilt from external data.

The prepared CSV contains 26 literature rows spanning 6 external flame retardants. It is copied into the frozen training schema, receives stable external IDs where needed, and common target aliases are normalized without recalculating experimental values.
