# Human Review & Promotion Checklist

**Document:** `proposals/companion-v1/HUMAN_REVIEW_CHECKLIST.md`  
**Target:** Human Research Lead / Independent Reviewer  
**Governance:** `docs/HTA_IMPLEMENTATION_SPEC_v1.2.md` (§0.1 Rule A3, §15.1, §16.1)  

---

## 1. Candidate Bundle Promotion Steps

Once you have reviewed and approved the candidate artifacts in `proposals/companion-v1/`, perform the following promotion actions to activate Gate G0 rerun:

### Step 1: Promote Configurations (`config/`)
1. Review `proposals/companion-v1/config/protocol.yaml`:
   - Set `pilot_gate_G6.noninferiority_margin` (e.g., 0.02) and `confirmed_by_human: true`.
   - Copy to `config/protocol.yaml`.
2. Review `proposals/companion-v1/config/columns.yaml`:
   - Confirm monetary unit multiplier in `currency_and_unit.unit_multiplier` (H2).
   - Copy to `config/columns.yaml`.

### Step 2: Promote Gate Manifest (`contracts/`)
- Copy `proposals/companion-v1/contracts/gate_acceptance_manifest.yaml` to `contracts/gate_acceptance_manifest.yaml`.

### Step 3: Promote Reference Implementation (`reference/`)
1. Review `proposals/companion-v1/reference/hta_ref.py` and `reference/test_hta_ref.py`.
2. Copy `proposals/companion-v1/reference/hta_ref.py` to `reference/hta_ref.py`.
3. Copy `proposals/companion-v1/reference/test_hta_ref.py` to `reference/test_hta_ref.py`.
4. (The agent will then copy `reference/hta_ref.py` $\to$ `src/hta/ref.py` byte-for-byte under Rule A3).

### Step 4: Promote Golden Fixtures (`fixtures/`)
1. Copy `proposals/companion-v1/fixtures/golden_cases.csv` to `fixtures/golden_cases.csv`.
2. Copy `proposals/companion-v1/fixtures/golden_expectations.csv` to `fixtures/golden_expectations.csv`.
3. Generate and sign `fixtures/fixture_manifest.sha256`:
   ```powershell
   Get-FileHash reference/hta_ref.py, reference/test_hta_ref.py, fixtures/golden_cases.csv, fixtures/golden_expectations.csv -Algorithm SHA256
   ```

### Step 5: Execute Holdout Staging & Isolation
1. Follow `proposals/human-staging/README.md` to run `stage_data.py` on your master dataset.
2. Deposit `canonical_dev_source.parquet` into `data/source/`.
3. Deposit `test_features_candidate.parquet` into `data/source/`.
4. Store `test_outcomes_sealed.parquet` in your offline vault.
5. Sign `contracts/staged_holdout_receipt.md`.

### Step 6: Install Pinned Dependencies and Run API Preflight
1. Approve `proposals/companion-v1/environment/requirements.candidate.txt`.
2. Install into `.venv`:
   ```powershell
   .venv\Scripts\pip install -r proposals/companion-v1/environment/requirements.candidate.txt
   ```
3. Run API probe:
   ```powershell
   .venv\Scripts\python proposals/companion-v1/environment/api_preflight.py
   ```
4. Verify `outputs/G0/api_preflight.json`.

---

## 2. Gate G0 Rerun Authorization

Once Steps 1–6 are verified:
- Run: `pytest -q reference/ tests/test_g0.py`
- Verify QA report.
- The human lead signs and creates `gates/G0_APPROVED.txt`.
