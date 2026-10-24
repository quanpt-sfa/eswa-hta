# Candidate Companion Bundle QA Report

**Document:** `proposals/companion-v1/CANDIDATE_BUNDLE_QA_REPORT.md`  
**Date:** 2026-10-24  
**Author:** Implementation & Verification Agent (Review Branch Audit & Interface Repair)  
**Review Branch:** `review/g0-independent-audit-fixes`  
**Base Commit:** `bad24bf`  
**Target Proposal:** `proposals/companion-v1/` and `proposals/human-staging/`  
**Status:** `CANDIDATE_BUNDLE_VERIFIED__PENDING_HUMAN_CERTIFICATION`  

---

## 1. Test Suite Summary (Executed Post-Audit Baseline)

- **Total Unit & Integration Tests Run:** 42
- **Passed:** 42
- **Failed:** 0
- **Skipped:** 0
- **Overall Exit Code:** 0

### Test Execution Command
```bash
python -m pytest proposals/ -v
```

### Breakdown by Module
1. **Candidate Configuration & Manifest Contracts (`proposals/companion-v1/config/test_protocol_contract.py`):**
   - `test_model_id_meanings_are_not_swapped`: **PASS** (Asserts B1..B8 algorithms, M0/B6 and M1/B8 aliases, and B3a primary benchmark)
   - `test_outcome_config_matches_specification_contract`: **PASS** (Asserts primary $k=0.05$, $zero\_tol=0.0$, thresholds $[0.05, 0.075, 0.10, 0.15, 0.20]$, and test years $2023-2025$)
   - `test_gate_dependencies_and_manifest_execution_order`: **PASS** (Asserts gate prerequisite chaining $G_4 \to G_5 \to G_6 \to G_7 \to G_8/G_9 \to G_{10}$, pilot models in G5, hurdle models in G7, dev experiments in G9, and locked test evaluation in G10)

2. **Fixture Schemas and Mathematical Tolerances (`proposals/companion-v1/fixtures/test_fixture_schema.py`):**
   - `test_csv_file_existence`: **PASS**
   - `test_csv_field_counts_and_headers`: **PASS** (Exact 11 columns in `golden_cases.csv`, exact 9 columns in `golden_expectations.csv`)
   - `test_matching_test_ids`: **PASS** (All 20 test IDs match 1-to-1)
   - `test_near_zero_mathematical_correction`: **PASS** (T-Z02 $y=0$, T-Z03 $y=1$)
   - `test_asymmetric_post_denominator_cuts`: **PASS** (T-P01, T-P02 algebraic reversal verified)

3. **Candidate Reference Functions & Golden Fixtures (`proposals/companion-v1/reference/test_hta_ref.py`):**
   - `test_build_labels_golden_cases`: **PASS** (14 edge cases in `build_labels`)
   - `test_build_labels_monotonic_nesting`: **PASS** ($y_{200} \le y_{150} \le y_{100} \le y_{075} \le y_{050}$)
   - `test_exceedance_prob_analytic_normal`: **PASS** (Matches standard normal analytic CDF)
   - `test_exceedance_prob_monotonic_thresholds`: **PASS** (Threshold monotonicity)
   - `test_hurdle_exceedance_formula`: **PASS** ($(1-\pi_0) \cdot P_{\text{cond}}$)
   - `test_scipy_exceedance`: **PASS** (Normal and Laplace analytic distributions)
   - `test_asymmetric_post_cuts`: **PASS** (Reversed cut magnitudes for negative pre-audit earnings)
   - `test_post_denominator_cuts_d_scale`: **PASS** (Normalized D-scale cuts with assets)
   - `test_two_sided_prob_symmetric_and_asymmetric`: **PASS** (Symmetric and asymmetric cutoff probability calculation)
   - `test_expand_thresholds`: **PASS** (Long-format threshold expansion, sample weights summing to 1.0, row identity preservation)
   - `test_near_zero_edges_and_assign_bins`: **PASS** (Separate edge fitting on development and out-of-sample bin assignment)
   - `test_near_zero_quintiles_reject_duplicated_edges`: **PASS** (Rejects tied quantile edges)
   - `test_temporal_expanding_folds`: **PASS** (F1–F4 temporal precedence)
   - `test_assert_fold_integrity`: **PASS** (Asserts non-empty folds, no train/val index overlap, strict temporal precedence, zero firm-year leaks)
   - `test_check_feature_leakage`: **PASS** (Publication timestamp leakage check)
   - `test_unknown_publication_date_is_not_safe`: **PASS** (Missing publication dates treated as unsafe)
   - `test_scan_feature_names`: **PASS** (Scans for forbidden post-audit tokens and validates allowlist)
   - `test_numeric_leak_check`: **PASS** (Calculates correlations with outcomes and flags suspicious predictors)
   - `test_average_precision_ties_invariant_to_row_order`: **PASS** (scikit-learn AP score tie invariance)
   - `test_precision_recall_at_k`: **PASS** (Precision@K and Recall@K at screening budget fractions)
   - `test_paired_firm_cluster_bootstrap`: **PASS** (Firm-cluster resampling preserving temporal panels)
   - `test_target_transform_bijectivity`: **PASS** ($asinh / sinh$ round-trip bijection)
   - `test_crps_score`: **PASS** (Trapezoidal pinball loss integration)
   - `test_all_golden_fixtures_from_csv`: **PASS** (Direct programmatic verification of all 20 CSV fixtures)

4. **Human Staging Tool Security & Sanitization (`proposals/human-staging/test_staging.py`):**
   - `test_holdout_staging_execution`: **PASS** (Physical split, positive selection, hashing, receipt)
   - `test_sanitization_audit_catches_leakage`: **PASS** (Forbidden outcome columns caught)
   - `test_staging_rejects_missing_human_allowlist`: **PASS** (Fails closed on missing allowlist)
   - `test_staging_rejects_source_specific_audited_columns`: **PASS** (Catches `fs_aud_*` and `audit_adj_*`)
   - `test_staging_rejects_allowlist_with_pat_post_or_audit_adj`: **PASS** (Rejects allowlist containing outcome columns)
   - `test_staging_rejects_missing_required_identifiers_in_allowlist`: **PASS** (Rejects allowlist omitting required keys)
   - `test_staging_rejects_duplicate_primary_keys`: **PASS** (Rejects duplicate firm-year-scope keys)
   - `test_staging_rejects_unexpected_fiscal_years`: **PASS** (Rejects out-of-horizon fiscal years)
   - `test_staging_rejects_overlapping_output_paths`: **PASS** (Rejects path collisions)
   - `test_staging_rejects_missing_required_target_inputs`: **PASS** (Rejects master datasets lacking target columns)

---

## 2. API Preflight Probe Verification (`outputs/G0/api_preflight.json`)

- **Execution Command:** `python proposals/companion-v1/environment/api_preflight.py`
- **Result:** `BLOCKED_DEPENDENCIES_OR_CHECKPOINT_UNVERIFIED`
- **Verified Invariants:**
  - TabICL, CatBoost, NGBoost imports fail closed to `BLOCKED_IMPORT_ERROR`.
  - Checkpoint hash probe fails closed to `BLOCKED_APPROVED_CHECKPOINT_OR_HASH_MISSING`.
  - Hardware runtime probe detects RTX 3060 (12.0 GB VRAM) but marks feasibility `BLOCKED_INFERENCE_MEMORY_UNVERIFIED`.
  - Overall status CANNOT become `ALL_PROBES_PASSED` while any component is unverified. Zero fabricated passes.

---

## 3. Repository Protection and Isolation Audits

- **Approval/Unlock Files Audit:** `Get-ChildItem -Recurse -File | Where-Object { $_.Name -match "APPROVED" -or $_.Name -match "UNLOCK" }` $\implies$ **ZERO files found (PASS)**.
- **Git Tracking Cleanliness:** `git ls-files` $\implies$ **Zero raw datasets, zero parquet files, zero binary model weights tracked (PASS)**.
- **Ignore Rules Effectiveness:** `.venv/`, `*.parquet`, `outputs/`, `logs/`, `checkpoints/` confirmed ignored via `git check-ignore` **(PASS)**.
- **Final Test Boundary:** No FY2023–2025 audited outcomes were read, processed, or evaluated.
