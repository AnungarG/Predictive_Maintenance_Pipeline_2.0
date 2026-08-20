# NLNG Predictive Maintenance — Predictive Maintenance Pipeline 2.0

This repository separates the experimental research record from the reusable analytical implementation and deployment layer.

## Structure

- `notebooks/` — research notebook and experimental record
- `src/predictive_maintenance/` — reusable Stage 1–6 implementation
- `scripts/run_predictive_maintenance_pipeline.py` — one-command execution
- `data/04_corrected_pipeline/` — frozen analytical outputs from the validated experiment (kept outside this clean refactor package)
- `models/` — deployment model artefacts
- `idss/backend/` — FastAPI model serving
- `idss/dashboard/` — Streamlit dashboard

## Reproducibility principle

The reusable implementation is refactored from the validated research notebook. The frozen Stage 1–6 results remain the benchmark for reproducibility validation.
