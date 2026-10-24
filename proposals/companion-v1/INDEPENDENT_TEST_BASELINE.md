# Independent Candidate Test Baseline Report

**Execution Timestamp:** 2026-10-24T20:23:30+07:00  
**Git Branch:** `review/g0-independent-audit-fixes`  
**Base Commit:** `bad24bf`  
**Execution Environment:** Python 3.12.2 (`C:\Users\quanp\AppData\Local\Programs\Python\Python312\python.exe`), Pytest 8.4.2, Windows 11.  
**Dependency Status:** 
- Core testing dependencies available in system environment: `pytest 8.4.2`, `numpy 2.2.3`, `pandas 2.2.3`, `scipy 1.15.2`, `scikit-learn 1.6.1`, `pyyaml 6.0.2`.
- Project isolated virtual environment (`.venv`): Currently contains only `pip 24.0`; `pytest` was not installed in `.venv`. Tests were run using the host's approved Python 3.12.2 toolchain without installing unapproved modeling dependencies.
- Modeling dependencies (`tabicl==2.2.0`, `catboost==1.2.5`, `ngboost==0.5.1`) and pre-trained checkpoints remain uninstalled and unapproved as mandated by G0 preflight constraints.

---

## 1. Test Suite Execution Summary

| Command | Exit Code | Collected | Passed | Failed | Skipped | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `python -m compileall -q proposals/` | 0 | — | — | — | — | **PASS** (Bytecode syntax valid across all files) |
| `python -m pytest proposals/ -v` | 0 | 26 | 26 | 0 | 0 | **PASS** (Full candidate test suite) |
| `python -m pytest proposals/companion-v1/reference/test_hta_ref.py -v` | 0 | 15 | 15 | 0 | 0 | **PASS** (Candidate reference functions) |
| `python -m pytest proposals/human-staging/test_staging.py -v` | 0 | 4 | 4 | 0 | 0 | **PASS** (Candidate staging positive-selection security) |
| `python -m pytest proposals/companion-v1/config/test_protocol_contract.py -v` | 0 | 2 | 2 | 0 | 0 | **PASS** (Candidate config semantic invariant tests) |
| `python -m pytest proposals/companion-v1/fixtures/test_fixture_schema.py -v` | 0 | 5 | 5 | 0 | 0 | **PASS** (CSV schema, field counts, and tolerances) |

---

## 2. Test-by-Test Execution Inventory

### A. Configuration Contracts (`test_protocol_contract.py`)
- `test_model_id_meanings_are_not_swapped`: **PASSED** (Confirms B1=CatBoost X1, B2=TabICL X2, B3a=CatBoost PAT5 X2, B4=NGBoost, B5=CatBoost MultiQuantile, B6/M0=TabICL Regressor, B7=Hurdle-CatBoost, B8/M1=Hurdle-TabICLv2).
- `test_outcome_config_matches_specification_contract`: **PASSED** (Confirms `protocol.yaml` contains `outcome.zero_tol = 0` and `outcome.thresholds = [0.05, 0.075, 0.10, 0.15, 0.20]`).

### B. Fixture Schemas and Tolerances (`test_fixture_schema.py`)
- `test_csv_file_existence`: **PASSED** (Both `golden_cases.csv` and `golden_expectations.csv` exist).
- `test_csv_field_counts_and_headers`: **PASSED** (No uneven commas; exactly 11 columns for inputs, 9 columns for expectations).
- `test_matching_test_ids`: **PASSED** (All 20 test IDs match 1-to-1).
- `test_near_zero_mathematical_correction`: **PASSED** (T-Z02 y=0, T-Z03 y=1 correctly distinguished).
- `test_asymmetric_post_denominator_cuts`: **PASSED** (Negative PATpre cuts: -4.7619% / +5.2632%; Positive PATpre cuts: -5.2632% / +4.7619%).

### C. Reference Function Implementations (`test_hta_ref.py`)
- `test_build_labels_golden_cases`: **PASSED** (Exercises `build_labels` against all 20 golden cases).
- `test_build_labels_monotonic_nesting`: **PASSED** (Tests nested threshold monotonicity $Y_{0.05} \ge Y_{0.10} \ge Y_{0.20}$).
- `test_exceedance_prob_analytic_normal`: **PASSED** (Compares analytical Normal tail probability with tolerance).
- `test_exceedance_prob_monotonic_thresholds`: **PASSED** (Monotonic tail probability reduction).
- `test_hurdle_exceedance_formula`: **PASSED** (Hurdle mixture $(1-\pi_0)P(|D|\ge c\mid D\neq 0)$).
- `test_scipy_exceedance`: **PASSED** (Scipy-based exceedance calculation).
- `test_asymmetric_post_cuts`: **PASSED** (Post-denominator asymmetric cuts algebra).
- `test_temporal_expanding_folds`: **PASSED** (Expanding window folds 2015-2022).
- `test_paired_firm_cluster_bootstrap`: **PASSED** (Firm-level cluster bootstrap difference and SE).
- `test_check_feature_leakage`: **PASSED** (Point-in-time publication date vs earnings announcement cutoff).
- `test_target_transform_bijectivity`: **PASSED** (Signed log/hyperbolic arc-sine bijectivity).
- `test_crps_score`: **PASSED** (CRPS calculation).
- `test_average_precision_ties_invariant_to_row_order`: **PASSED** (scikit-learn AP with tied scores).
- `test_unknown_publication_date_is_not_safe`: **PASSED** (NaT/missing publication date treated as unsafe).
- `test_near_zero_quintiles_reject_duplicated_edges`: **PASSED** (Duplicate bin edges rejected).

### D. Synthetic Data Staging (`test_staging.py`)
- `test_holdout_staging_execution`: **PASSED** (Synthetic pipeline staging 2015-2022 dev vs 2023-2025 test).
- `test_sanitization_audit_catches_leakage`: **PASSED** (Rejects unapproved leakage columns).
- `test_staging_rejects_missing_human_allowlist`: **PASSED** (Fails closed when human allowlist is absent).
- `test_staging_rejects_source_specific_audited_columns`: **PASSED** (Rejects `fs_aud_*` and `audit_adj_*`).

---

## 3. Unexecuted / Blocked Checks
The following checks remain intentionally unexecuted and blocked:
1. **Model Checkpoint & Inference Preflight (`api_preflight.py`):** Blocked because `tabicl`, `catboost`, and `ngboost` packages are not yet approved/installed, and the pre-trained TabICLv2 checkpoint SHA-256 has not been verified.
2. **Production Gate Preflight (`tests/test_g0.py`):** Blocked until candidate reference functions and fixtures are independently approved and promoted from `proposals/companion-v1/` to production paths `reference/` and `fixtures/`.
3. **Physical Holdout Staging on Real Research Data:** Staging tool was verified solely on synthetic data; real research data execution is restricted to trusted human stewards.
