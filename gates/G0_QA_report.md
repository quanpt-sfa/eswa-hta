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
Protected test access audit: PASSED isolation audit; no protected FY2023-2025 outcome data is accessible in current workspace.
Human approval: PENDING (agent cannot self-approve)
