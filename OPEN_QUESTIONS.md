# Open Questions Register (G0 Remediation)

**Repository:** `quanpt-sfa/eswa-hta`  
**Governance Specification:** `docs/HTA_IMPLEMENTATION_SPEC_v1.2.md` (§0.3, §15, §16)  
**Last Updated:** 2026-10-24  

---

### OQ-001: Missing Companion Input Bundle (§15.1, Rule A3, Rule H9)

| Field | Content |
|---|---|
| **ID** | `OQ-001` |
| **Root cause** | Specification `HTA_IMPLEMENTATION_SPEC_v1.2.md` is an execution contract, not a complete code bundle. Required companion artifacts (`config/protocol.yaml`, `config/columns.yaml`, `reference/hta_ref.py`, `reference/test_hta_ref.py`, `fixtures/golden_cases.csv`, `fixtures/golden_expectations.csv`, `fixtures/fixture_manifest.sha256`, and `contracts/gate_acceptance_manifest.yaml`) are physically absent from the repository. |
| **Agent remediation** | Audited repository and sibling research folders; verified no companion files exist. Documented complete mathematical contracts in `contracts/COMPANION_BUNDLE_HANDOFF.md`. Constructed full candidate companion bundle in `proposals/companion-v1/` and `proposals/human-staging/` (64/64 unit and adversarial tests passed across all modules). Remediated all four technical defects (API preflight CLI/levels, $c_{k050}$ feature authorization, fold-safe binning, master dataset collision safety). Drafted formal protocol amendment `docs/PROPOSED_AMENDMENT_FOLD_SAFE_BINNING.md`. |
| **Remaining blocker** | Human review and formal sign-off on: (1) candidate artifacts promotion to production paths (`reference/`, `fixtures/`, `config/`, `contracts/`); (2) proposed fold-safe binning amendment (`docs/PROPOSED_AMENDMENT_FOLD_SAFE_BINNING.md`); (3) pinned dependency and checkpoint verification. |
| **Required human action** | Review `proposals/companion-v1/HUMAN_REVIEW_CHECKLIST.md` and `docs/PROPOSED_AMENDMENT_FOLD_SAFE_BINNING.md`, approve amendment, and promote candidate files to authoritative paths. |
| **Required verification** | `pytest -q reference/test_hta_ref.py` passes; byte-identical copy `src/hta/ref.py` matches `reference/hta_ref.py`; file hashes match `fixture_manifest.sha256`. |
| **Current state** | **READY_FOR_HUMAN** |

---

### OQ-002: Holdout Data Physical Isolation and Access Control (§4.3, Rule A5, Rule H8)

| Field | Content |
|---|---|
| **ID** | `OQ-002` |
| **Root cause** | The G0 QA report previously claimed holdout isolation had passed based merely on workspace emptiness. However, unredacted full-period research datasets (`vn_pipeline_final_firm_year_2015_2025.parquet`) reside in other local directories under the same Windows user account (`quanp`). Merely leaving `data/locked/` empty in the workspace does not provide physical isolation against autonomous read operations. |
| **Agent remediation** | Retracted unsupported PASS claim in `gates/G0_QA_report.md` (updated to `NOT VERIFIED — STOP_BLOCKER`) and logged in `DECISIONS.log`. Authored `contracts/HUMAN_HOLDOUT_STAGING_RUNBOOK.md` and `contracts/staged_holdout_receipt.template.md`. Developed standalone human staging tool in `proposals/human-staging/` (`stage_data.py`, `test_staging.py` [2 passed], `README.md`, `STAGING_SECURITY_CHECKLIST.md`). |
| **Remaining blocker** | Human data steward has not yet executed physical isolation of master raw datasets, nor staged the separated `canonical_dev_source.parquet` and `test_features_candidate.parquet`. |
| **Required human action** | 1. Isolate master research datasets into an inaccessible vault/container.<br>2. Run `stage_data.py` on master dataset to generate `data/source/canonical_dev_source.parquet` and `data/source/test_features_candidate.parquet`.<br>3. Seal 2023–2025 outcomes in external vault.<br>4. Complete and sign `contracts/staged_holdout_receipt.md` with SHA-256 hashes. |
| **Required verification** | Access-denied probe against master raw replica passes under agent identity; hash check against `contracts/input_manifest.sha256`; zero post-audit target columns in `test_features_candidate.parquet`. |
| **Current state** | **READY_FOR_HUMAN** |

---

### OQ-003: Pinned Dependencies and Checkpoint Preflight (§15.3, Rule A12)

| Field | Content |
|---|---|
| **ID** | `OQ-003` |
| **Root cause** | Key modeling packages (`tabicl`, `catboost`, `ngboost`) and pre-trained TabICLv2 checkpoints are not installed or verified in the project environment. Pinned dependency versions and checkpoint hashes have not been formally declared in `protocol.yaml`. |
| **Agent remediation** | Profiled hardware (NVIDIA RTX 3060 12GB VRAM, CUDA 11.8 active, PyTorch 2.6.0, Python 3.12.2, 116 GB disk space). Created isolated project virtual environment at `eswa-hta/.venv/`. Retracted unsupported memory feasibility claims in `contracts/ENVIRONMENT_PREFLIGHT.md`. Created candidate requirements `requirements.candidate.txt` and automated API probe `proposals/companion-v1/environment/api_preflight.py`. |
| **Remaining blocker** | Pinned dependency manifest and approved TabICL checkpoint source/checksum have not been signed off by the human lead. Installation cannot proceed under Rule A12 without approval. |
| **Required human action** | Approve candidate requirements in `proposals/companion-v1/environment/requirements.candidate.txt`, authorize `.venv` installation, and provide approved TabICLv2 checkpoint SHA-256 in `config/protocol.yaml`. |
| **Required verification** | Successful installation into `.venv`; execution of `api_preflight.py` verifying 999 raw quantiles, monotonic alpha ordering, CatBoost MultiQuantile, and NGBoost distributions; generation of `environment_lock.txt`. |
| **Current state** | **READY_FOR_HUMAN** |

---

### OQ-004: Public GitHub Repository Data Exposure Risk and Remote Sync Policy

| Field | Content |
|---|---|
| **ID** | `OQ-004` |
| **Root cause** | The repository `https://github.com/quanpt-sfa/eswa-hta.git` is public. Any accidental commit or push of proprietary financial statements, firm identifiers, or model predictions would violate privacy, data licensing, and research blinding rules. |
| **Agent remediation** | Configured multi-layer exclusion rules in `.gitignore` protecting all raw/processed/locked data, binary formats (`*.parquet`, `*.pt`, `*.ckpt`), output predictions, logs, and secrets, while preserving essential code and synthetic fixtures. Verified via `git ls-files` and `git check-ignore -v`. Documented in `contracts/REPOSITORY_DATA_PROTECTION_AUDIT.md`. |
| **Remaining blocker** | Autonomous agent is prohibited from changing GitHub remote settings or executing unapproved `git push` operations. Server-side protections are unverified. |
| **Required human action** | 1. Confirm whether repository visibility should remain public or be transitioned to private.<br>2. Configure GitHub branch protection rules on `main` to prevent unreviewed pushes.<br>3. Confirm policy requiring human review before any `git push`. |
| **Required verification** | Review repository visibility and settings on GitHub; confirm pre-push audit checklist. |
| **Current state** | **READY_FOR_HUMAN** |
