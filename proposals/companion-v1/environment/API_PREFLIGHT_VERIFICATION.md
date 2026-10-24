# Modeling Engine API Preflight Verification Report

**Document Version:** 1.1  
**Target Specification:** `docs/HTA_IMPLEMENTATION_SPEC_v1.2.md` (§5 G0, §15.3, Rule A12)  
**Evaluated Script:** `proposals/companion-v1/environment/api_preflight.py`  
**Execution Timestamp:** 2026-10-24T20:43:25+07:00  
**Overall Status:** `BLOCKED_DEPENDENCIES_OR_CHECKPOINT_UNVERIFIED` (Zero Fabricated Passes)

---

## 1. Executive Summary

In accordance with Gate G0 preflight integrity rules (§15.3, Rule A12), modeling dependencies (`tabicl==2.2.0`, `catboost==1.2.5`, `ngboost==0.5.1`) and pre-trained TabICLv2 checkpoints must NOT be installed, downloaded, or certified autonomously by the coding agent without human authorization.

The executable candidate probe `api_preflight.py` was executed in the active Python 3.12.2 environment. It verified that all assertion gates correctly fail-closed to `BLOCKED` rather than manufacturing false passes.

---

## 2. Component-by-Component Probe Results

| Component | Target Assertions (§15.3) | Observed Probe Status | Output Evidence / Error | Preflight Disposition |
| :--- | :--- | :---: | :--- | :---: |
| **TabICLRegressor** | Pinned import, `fit(X, y)`, `output_type="raw_quantiles"`, shape `(N, 999)`, valid strictly increasing alpha levels vector `alphas_` in (0, 1), non-crossing quantiles | `BLOCKED_IMPORT_ERROR` | `No module named 'tabicl'` | **BLOCKED** |
| **TabICLClassifier** | Pinned import, `fit(X, y_bin)`, `predict_proba(X)` shape `(N, 2)`, valid `classes_ == [0, 1]`, rows sum to 1.0 | `BLOCKED_IMPORT_ERROR` | `No module named 'tabicl'` | **BLOCKED** |
| **CatBoostMultiQuantile** | Loss function `MultiQuantile:alpha=...` 99 levels, output shape `(N, 99)`, finite values, monotonic quantiles | `BLOCKED_IMPORT_ERROR` | `No module named 'catboost'` | **BLOCKED** |
| **NGBoost** | Import `Normal`, `Laplace`, `TFixedDf`, distribution object has `.cdf()` and `.sf()`, supported early stopping API | `BLOCKED_IMPORT_ERROR` | `No module named 'ngboost'` | **BLOCKED** |
| **TabICLv2 Checkpoint** | Pinned file exists, computed SHA-256 matches human-approved cryptographic checksum | `BLOCKED_APPROVED_CHECKPOINT_OR_HASH_MISSING` | Checkpoint path not supplied; checksum unpinned | **BLOCKED** |
| **HardwareRuntime** | GPU / CUDA device detection, total VRAM, executable memory feasibility under registered TabICLv2 configuration | `BLOCKED_INFERENCE_MEMORY_UNVERIFIED` | GPU detected (`NVIDIA GeForce RTX 3060`, 12.0 GB VRAM); memory feasibility unverified until inference execution | **BLOCKED** |

---

## 3. Strict Preflight Invariants Enforced

1. **Anti-Fabrication Invariant:** Overall status is mathematically constrained:
   $$\text{Overall Status} = \text{ALL\_PROBES\_PASSED} \iff \forall c \in \text{Components}, \text{status}(c) == \text{PASSED}$$
   Since five components are `BLOCKED_IMPORT_ERROR` or `BLOCKED_CHECKPOINT`, the overall status evaluated to:
   `BLOCKED_DEPENDENCIES_OR_CHECKPOINT_UNVERIFIED`
2. **Quantile Grid Integrity:** The probe explicitly checks `alphas_` on `TabICLRegressor` and verifies that quantile levels are strictly increasing floats in $(0, 1)$. It refuses to infer alphas merely from column width.
3. **Hardware Honesty:** The RTX 3060 GPU with 12.0 GB VRAM is detected, but its feasibility for the full TabICLv2 model is explicitly marked as `BLOCKED_INFERENCE_MEMORY_UNVERIFIED` until an actual inference run is executed with the approved checkpoint.
