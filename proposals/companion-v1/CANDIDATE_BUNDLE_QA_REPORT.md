# Candidate Companion Bundle QA Report

**Document:** `proposals/companion-v1/CANDIDATE_BUNDLE_QA_REPORT.md`  
**Date:** 2026-10-24  
**Author:** Implementation Agent (G0 Remediation)  
**Target Proposal:** `proposals/companion-v1/` and `proposals/human-staging/`  
**Starting Commit:** `742dbb580db1dfabec990a7fe676352937441e94`  
**Status:** CANDIDATE_BUNDLE_READY_FOR_HUMAN_REVIEW  

---

## 1. Test Suite Summary

- **Total Unit & Integration Tests Run:** 19
- **Passed:** 19
- **Failed:** 0
- **Skipped:** 0
- **Overall Exit Code:** 0

### Test Execution Command
```bash
python -m pytest proposals/ -v
```

### Breakdown by Module
1. `proposals/companion-v1/fixtures/test_fixture_schema.py`:
   - `test_csv_file_existence`: **PASS**
   - `test_csv_field_counts_and_headers`: **PASS** (11 columns in cases, 9 columns in expectations)
   - `test_matching_test_ids`: **PASS** (20 identical keys)
   - `test_near_zero_mathematical_correction`: **PASS** (T-Z02 $y=0$, T-Z03 $y=1$)
   - `test_asymmetric_post_denominator_cuts`: **PASS** (T-P01, T-P02 algebraic verification)
2. `proposals/companion-v1/reference/test_hta_ref.py`:
   - `test_build_labels_golden_cases`: **PASS** (14 edge cases)
   - `test_build_labels_monotonic_nesting`: **PASS** ($y_{200} \le y_{150} \le y_{100} \le y_{075} \le y_{050}$)
   - `test_exceedance_prob_analytic_normal`: **PASS** (Matches standard normal CDF)
   - `test_exceedance_prob_monotonic_thresholds`: **PASS** (Threshold monotonicity)
   - `test_hurdle_exceedance_formula`: **PASS** ($(1-\pi_0) \cdot P_{\text{cond}}$)
   - `test_scipy_exceedance`: **PASS** (Normal and Laplace analytic distributions)
   - `test_asymmetric_post_cuts`: **PASS** (Reversed cut magnitudes for negative earnings)
   - `test_temporal_expanding_folds`: **PASS** (F1–F4 temporal precedence)
   - `test_paired_firm_cluster_bootstrap`: **PASS** (Resampling by firm cluster)
   - `test_check_feature_leakage`: **PASS** (Publication timestamp leakage check)
   - `test_target_transform_bijectivity`: **PASS** ($asinh / sinh$ bijection)
   - `test_crps_score`: **PASS** (Pinball loss integration)
3. `proposals/human-staging/test_staging.py`:
   - `test_holdout_staging_execution`: **PASS** (Physical split, sanitization, hashing, receipt)
   - `test_sanitization_audit_catches_leakage`: **PASS** (Forbidden outcome columns caught)

---

## 2. Configuration & Manifest Syntax Validation

- **Command:** `python -c "import yaml; ..."`
- **Files Checked:**
  - `proposals/companion-v1/config/protocol.yaml`: **PASS (Valid YAML, 11 sections)**
  - `proposals/companion-v1/config/columns.yaml`: **PASS (Valid YAML, 9 sections)**
  - `proposals/companion-v1/contracts/gate_acceptance_manifest.yaml`: **PASS (Valid YAML, G0–G12)**
- **Exit Code:** `0`

---

## 3. Security and Repository Invariant Checks

- **Approval/Unlock Files Audit:** `Get-ChildItem -Recurse -File | Where-Object { $_.Name -match "APPROVED" -or $_.Name -match "UNLOCK" }` $\implies$ **ZERO files found (PASS)**.
- **Git Tracking Cleanliness:** `git ls-files` $\implies$ **Zero raw datasets, zero parquet files, zero binary model weights tracked (PASS)**.
- **Ignore Rules Effectiveness:** `.venv/`, `*.parquet`, `outputs/`, `logs/`, `checkpoints/` confirmed ignored via `git check-ignore` **(PASS)**.
