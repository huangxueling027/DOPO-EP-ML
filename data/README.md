# Data

This directory contains the frozen machine-readable data release used by the manuscript workflow.

- `DOPO_EP_new.csv`: standardized DOPO-derived flame-retardant/epoxy records.
- `DOPO_EP_new_with_BDE.csv`: the same data release with auxiliary BDE-related fields.
- `external_validation/External_Validation_V4_ready.csv`: independent literature-based external-validation input retained under its historical filename for compatibility with the frozen code.

The main data release contains 599 standardized experimental records. Dataset integrity is checked against the SHA256 values stored in `PROJECT_RULES_LOCK.json`. Original publisher PDFs are not redistributed in this repository. Literature-source fields in the dataset should be retained for traceability and the original publications should be cited when the data are reused.
