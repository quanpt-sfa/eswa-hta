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
