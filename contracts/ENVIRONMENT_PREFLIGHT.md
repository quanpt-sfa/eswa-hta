# Environment and Checkpoint Preflight Report

**Document ID:** `contracts/ENVIRONMENT_PREFLIGHT.md`  
**Repository:** `quanpt-sfa/eswa-hta`  
**Date:** 2026-10-24  
**Author:** Autonomous Implementation Agent (G0 Remediation)  
**Governance Specification:** `docs/HTA_IMPLEMENTATION_SPEC_v1.2.md` (§0.1 Rule A12, §5 G0, §15.3, §16.2)  
**Blocker ID:** `OQ-003`  
**Disposition:** READY_FOR_HUMAN (System runtime profiled; `.venv` initialized; awaiting pinned dependency approval)

---

## 1. System Runtime and Hardware Diagnostics

Diagnostics were executed independently via Python and PowerShell probes on the local host machine:

| Component | Diagnostic Output | Specification Status |
|---|---|---|
| **Operating System** | Windows 11 Pro 64-bit (`10.0.26300-SP0`) | Compatible |
| **System Python** | Python 3.12.2 (`tags/v3.12.2:6abddd9, Feb 6 2024`) | Compatible ($\ge 3.10$) |
| **PyTorch Version** | `2.6.0+cu118` | Compatible with CUDA 11.8 |
| **CUDA Available** | `True` | GPU Acceleration Active |
| **GPU Hardware** | NVIDIA GeForce RTX 3060 | Compatible Ampere Architecture |
| **VRAM Total** | 12.00 GB Dedicated GDDR6 | GPU detected; memory feasibility of the registered TabICLv2 configuration remains unverified until an executable inference and memory preflight passes. |
| **CUDA Archs Supported** | `sm_37`, `sm_50`, `sm_60`, `sm_61`, `sm_70`, `sm_75`, `sm_80`, `sm_86`, `sm_90` | `sm_86` matches RTX 3060 |
| **Repository Disk Space** | Drive `D:\` — Free: 116.77 GB / Total: 1863.01 GB | Sufficient ($> 50 \text{ GB}$ required) |
| **System Drive Disk Space** | Drive `C:\` — Free: 29.26 GB / Total: 475.16 GB | Sufficient |

---

## 2. Base Environment Package Audit

Inspection of packages in the active environment revealed:
- `scipy`: `1.14.1` (PRESENT)
- `pandas`: `2.3.3` (PRESENT)
- `pyarrow`: `23.0.1` (PRESENT)
- `pytest`: `8.4.2` (PRESENT)
- `tabicl`: **NOT INSTALLED**
- `catboost`: **NOT INSTALLED**
- `ngboost`: **NOT INSTALLED**

Check of local model caches:
- `~/.cache/huggingface/hub`: No TabICL checkpoints present.
- `~/.cache/torch`: Directory absent.
- `~/.tabicl`: Directory absent.

---

## 3. Dedicated Project Virtual Environment (`.venv`)

In accordance with G0 specifications and to avoid polluting global Python libraries:
- A clean, project-isolated virtual environment was created at `eswa-hta/.venv/`.
- Python interpreter path: `eswa-hta/.venv/Scripts/python.exe`.
- `.gitignore` rule confirmed: `.venv/` is strictly ignored by Git.

---

## 4. Pending Dependency Pinning & Checkpoint Approvals

Per Rule A12:
> "Do not call external web APIs except to download the human-approved, checksum-pinned TabICL checkpoints (or use the human-provided local `model_path`). Package upgrades, undocumented checkpoint substitutions, and unpinned dependencies require human approval."

Installation is currently **held closed** until the human research lead provides:
1. Pinned package manifest in `config/protocol.yaml`:
   - Exact pinned versions for `tabicl` (candidate: `2.2.0`), `catboost`, and `ngboost`.
2. Checkpoint source & integrity lock:
   - Approved HuggingFace repository ID or local file path for TabICLv2 weights.
   - Authoritative SHA-256 hash for TabICL checkpoint file.

---

## 5. Contractual API Preflight Test Matrix (§15.3)

Once pinned packages are installed in `.venv`, the agent will execute the automated preflight probe script to record `outputs/G0/api_preflight.json` and generate `environment_lock.txt`. The test matrix encompasses:

### 5.1 TabICLv2 Regressor Probe
- Instantiate `TabICLRegressor(n_estimators=1, random_state=42)`.
- Fit on toy synthetic data `(X_toy, y_toy)`.
- Execute `r.predict(X_toy[:5], output_type="raw_quantiles")`.
- Assert and record:
  - Exact returned tensor/array shape.
  - Whether 999 raw quantiles are exposed.
  - Quantile level vector ($\alpha$) and monotonicity of alpha ordering.
  - Device placement (GPU vs CPU).

### 5.2 TabICLv2 Classifier Probe
- Instantiate `TabICLClassifier(n_estimators=1, random_state=42)`.
- Fit on binary target `is_zero_adj`.
- Execute `predict_proba(X_toy[:5])`.
- Assert `classes_` attribute equals `[0, 1]` and class 1 probabilities correspond to positive adjustments.

### 5.3 CatBoost MultiQuantile Parameter Probe
- Instantiate `CatBoostRegressor(loss_function='MultiQuantile:alpha=0.01,0.02,...,0.99', random_seed=42)`.
- Fit and predict on toy data.
- Assert output shape equals `(N, 99)`.
- Confirm categorical feature NaN tolerance and GPU CUDA execution.

### 5.4 NGBoost Distribution Families Probe
- Test fitting with distributions: `Normal`, `Laplace`, and `TFixedDf`.
- Verify `model.pred_dist(X).dist` supports `.cdf()`, `.sf()`, and `.logpdf()`.
- Validate early-stopping API behavior.

### 5.5 Checkpoint Identity & Environment Lock
- Record SHA-256 hash of every checkpoint file loaded during inference.
- Generate `environment_lock.txt` containing `pip freeze`, Python build metadata, GPU compute capability, and checkpoint hashes.

---

## 6. Current Disposition

- **Hardware & CUDA Capability:** GPU detected (RTX 3060, 12GB VRAM, CUDA 11.8); memory feasibility of the registered TabICLv2 configuration remains unverified until an executable inference and memory preflight passes.
- **Project Isolation:** PASSED (`.venv` initialized).
- **Library Installation:** BLOCKED pending human-approved pinned manifest (`protocol.yaml`).
- **Checkpoint Validation:** BLOCKED pending human-approved checkpoint SHA-256.
