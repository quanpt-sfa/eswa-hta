# G0 QA report
Date / commit: 2026-10-24 / initial
pytest summary (paste verbatim): N/A (execution blocked; missing companion reference test suite tests/test_g0.py and reference/test_hta_ref.py)
Outputs written:
- Repository layout directories initialized per §3
- DECISIONS.log initialized
- OPEN_QUESTIONS.md initialized (OQ-001, OQ-002, OQ-003)
Reconciliation / key numbers: N/A
Open questions raised: OQ-001, OQ-002, OQ-003
Decisions logged: [2026-10-24T18:21:00+07:00 G0 Layout Init], [2026-10-24T18:21:00+07:00 G0 Fail Closed on Missing Bundle]
Deviations from spec: none
Status requested: STOP_BLOCKER
Gate acceptance manifest path / hash: contracts/gate_acceptance_manifest.yaml (MISSING)
Executed command and exit code:
- Directory creation: Exit 0
- python --version: Exit 0 (Python 3.12.2)
- CUDA check: Exit 0 (CUDA available: True, Device: NVIDIA GeForce RTX 3060)
- pip index versions tabicl: Exit 0 (tabicl 2.2.0 available)
Golden fixture tests passed / failed: BLOCKED (fixtures/golden_cases.csv, fixtures/golden_expectations.csv, and reference/test_hta_ref.py are missing)
Protected test access audit: NOT VERIFIED — STOP_BLOCKER
  - Context & Evidence: The local ESWA workspace does not currently contain a released test-outcome file. However, an unredacted source dataset (e.g., in other local research directories such as dissertation data) may remain accessible under the same Windows user account.
  - Vulnerability: Merely having empty workspace directories or absent local files does not demonstrate effective access control (per Rule A5 and §4.3).
  - Required Remediation: A human-controlled staging and permission-verification procedure is mandatory before isolation can be claimed.

Status categorization of G0 preflight checks:
- VERIFIED:
  - Directory skeleton initialized per §3: Exit 0
  - Python runtime: Python 3.12.2 detected: Exit 0
  - CUDA GPU device: NVIDIA GeForce RTX 3060 (12GB VRAM), CUDA available: True: Exit 0
- REPORTED BUT NOT INDEPENDENTLY VERIFIED:
  - tabicl==2.2.0 reported as available on PyPI index (uninstalled in project scope; API and checkpoints unverified)
- BLOCKED:
  - Companion input bundle (§15.1, OQ-001): protocol.yaml, columns.yaml, hta_ref.py, test_hta_ref.py, gate_acceptance_manifest.yaml are MISSING
  - Golden fixtures (§15.2, OQ-001): golden_cases.csv, golden_expectations.csv, fixture_manifest.sha256 are MISSING
  - Protected test access isolation (§4.3, OQ-002): NOT VERIFIED — STOP_BLOCKER
  - Pinned dependencies & TabICL checkpoint (§15.3, OQ-003): BLOCKED pending bundle approval and pinned lockfile
- NOT EXECUTED:
  - pytest test_g0.py and test_hta_ref.py (blocked by missing reference test files)
  - TabICL/CatBoost/NGBoost API preflight matrix (§15.3, outputs/G0/api_preflight.json)
  - Bitwise comparison of reference implementation (`src/hta/ref.py` vs `reference/hta_ref.py`)
  - Fixture and input SHA-256 verification (manifests missing)

Human approval: PENDING (agent cannot self-approve; G0 remains STOP_BLOCKER)

---

## Remediation Audit Trail — Commit 742dbb5 & Candidate Companion Bundle

**Audit Date:** 2026-10-24  
**Starting Commit:** `742dbb580db1dfabec990a7fe676352937441e94`  
**Overall Gate Status:** `STOP_BLOCKER` (Unresolved human dependencies; holdout isolation unverified)

### 1. Modified and Created Artifacts
- **Governance & Security Contracts Modified/Created:**
  - `gates/G0_QA_report.md`: Retracted unsupported holdout isolation PASS; added structured classification.
  - `.gitignore`: Hardened against raw/processed data, binary tables (`*.parquet`), checkpoints (`*.pt`, `*.ckpt`), logs, and secrets.
  - `contracts/REPOSITORY_DATA_PROTECTION_AUDIT.md`: Confirmed zero sensitive files in Git history; verified 17 path exclusion rules.
  - `contracts/COMPANION_BUNDLE_HANDOFF.md`: Corrected T-Z02 ($y=0$); added T-Z03 ($y=1$); documented reference function mathematical contracts.
  - `contracts/HUMAN_HOLDOUT_STAGING_RUNBOOK.md` & `contracts/staged_holdout_receipt.template.md`: Defined vault/container/ACL isolation architectures for human steward.
  - `contracts/ENVIRONMENT_PREFLIGHT.md`: Retracted unverified RTX 3060 TabICLv2 feasibility guarantees; documented API preflight test matrix.
  - `OPEN_QUESTIONS.md` & `DECISIONS.log`: Restructured blocker register and logged all remediation actions.
- **Candidate Companion Bundle Artifacts (`proposals/companion-v1/`):**
  - Fixtures: `fixtures/golden_cases.csv`, `fixtures/golden_expectations.csv`, `GOLDEN_CASE_DERIVATIONS.md`, `fixture_validation_report.md`
  - Reference: `reference/hta_ref.py`, `reference/test_hta_ref.py`, `reference/API_CONTRACT.md`, `reference/REFERENCE_QA_REPORT.md`
  - Configurations: `config/protocol.yaml`, `config/columns.yaml`, `contracts/gate_acceptance_manifest.yaml`, `CONFIG_VALIDATION_REPORT.md`

### 2. Corrected Mathematical Contracts
- **Case T-Z02 (Subthreshold Adjustment):** Corrected expected label from $y=1$ to $y=0$ ($|\Delta|=10^{-10} < 0.05 \times 100 = 5.0$).
- **Case T-Z03 (Genuine Near-Zero Exceedance):** Added test with $PAT_{pre}=10^{-9}, PAT_{post}=1.1 \times 10^{-9}, k=0.05$ where $|\Delta|=10^{-10} \ge 5 \times 10^{-11} \implies y=1$.
- **Post-Denominator Asymmetry:** Algebraically derived and tested cut magnitude reversal between positive and negative earnings ($PAT_{pre} \gtrless 0$).

### 3. Verification Test Evidence
- `python -m pytest proposals/companion-v1/fixtures/test_fixture_schema.py -v`: **5 passed in 2.24s (Exit Code: 0)**
- `python -m pytest proposals/companion-v1/reference/test_hta_ref.py -v`: **12 passed in 3.98s (Exit Code: 0)**
- `python -c "import yaml; ..."`: **3 YAML configs parsed successfully without error (Exit Code: 0)**

### 4. Human-Owned Blockers and Pending Approvals
- **OQ-001:** Companion bundle promotion and independent human certification.
- **OQ-002:** Holdout data physical isolation on local host and steward staging.
- **OQ-003:** Pinned dependency versions and approved TabICLv2 checkpoint SHA-256.
- **OQ-004:** Public GitHub repository visibility and push authorizations.

**Final Disposition:** `STOP_BLOCKER` (Holdout isolation NOT VERIFIED; companion bundle remains PENDING_INDEPENDENT_REVIEW).
