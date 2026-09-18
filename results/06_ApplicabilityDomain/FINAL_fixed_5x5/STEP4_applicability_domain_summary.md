# STEP 4 Applicability-domain summary

Morgan radius: 3; bits: 2048
Prespecified zones: In-domain >= 0.70; Caution >= 0.50; otherwise Extrapolation.

The threshold-support label is descriptive. Final paper wording should also inspect sample counts, fold-level metrics, and threshold-sensitivity tables.

## LOI
- In-domain / caution / extrapolation: 143 / 320 / 136
- New-scaffold fraction: 0.755
- Threshold assessment: Supported — MAE increases monotonically from in-domain to extrapolation

## PHRR
- In-domain / caution / extrapolation: 113 / 201 / 92
- New-scaffold fraction: 0.865
- Threshold assessment: Supported — MAE increases monotonically from in-domain to extrapolation

## THR
- In-domain / caution / extrapolation: 97 / 198 / 102
- New-scaffold fraction: 0.829
- Threshold assessment: Not_supported — Extrapolation MAE does not exceed in-domain MAE

## UL94_V0
- In-domain / caution / extrapolation: 149 / 303 / 147
- New-scaffold fraction: 0.776
- Threshold assessment: Supported — Accuracy decreases monotonically from in-domain to extrapolation
