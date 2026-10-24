# API Preflight Status Report

**Document:** `proposals/companion-v1/environment/API_PREFLIGHT_STATUS.md`  
**Date:** 2026-10-24  
**Author:** Implementation Agent (G0 Remediation)  
**Governance Specification:** `docs/HTA_IMPLEMENTATION_SPEC_v1.2.md` (§5 G0, §15.3, Rule A12)  
**Status:** PROBE_READY_EXECUTION_BLOCKED  

---

## 1. Executive Summary

In strict accordance with Rule A12:
> "Package upgrades, undocumented checkpoint substitutions, and unpinned dependencies require human approval. Do not download unapproved model checkpoints."

The automated preflight probe script `proposals/companion-v1/environment/api_preflight.py` has been fully implemented, but its live execution is **held as blocked** pending human approval of the pinned dependency manifest (`requirements.candidate.txt` and `protocol.yaml`).

No mock or fabricated PASS results have been generated.

---

## 2. API Test Matrix Specification (§15.3)

Once approved packages are installed into `.venv`, the probe script will execute on synthetic data and record results directly to `outputs/G0/api_preflight.json`:

| Component | Target Verifications | Pass Criterion | Failure Action |
|---|---|---|---|
| **TabICLRegressor** | `fit`, `predict(output_type="raw_quantiles")`, alpha levels, monotonicity | Exact output shape, 999 quantiles, non-crossing | STOP (Audit shape; request version fix) |
| **TabICLClassifier** | `fit(y_binary)`, `predict_proba` | `classes_ == [0, 1]`, probabilities sum to 1.0 | STOP |
| **CatBoost MultiQuantile** | `loss_function="MultiQuantile:alpha=..."` | Output shape $(N, 99)$, valid prediction ordering | STOP |
| **NGBoost Distributions** | `Normal`, `Laplace`, `TFixedDf` | Distributions expose `.sf()`, `.cdf()`, and early stopping | STOP (Propose approved distribution subset) |
| **Checkpoint Integrity** | Checkpoint file SHA-256 hash | Exact match to human-signed checksum | STOP (Do not substitute weights) |
| **GPU / Memory Feasibility** | CUDA allocation on RTX 3060 | Peak memory within 12 GB VRAM | STOP (Propose CPU fallback / offload) |

---

## 3. Human Approval Actions Required

1. Review and approve candidate dependencies in `proposals/companion-v1/environment/requirements.candidate.txt`.
2. Authorize installation into `eswa-hta/.venv/`.
3. Provide approved TabICLv2 model checkpoint path/URL and authoritative SHA-256 checksum.
4. Execute `python proposals/companion-v1/environment/api_preflight.py` to produce final verified `outputs/G0/api_preflight.json`.
