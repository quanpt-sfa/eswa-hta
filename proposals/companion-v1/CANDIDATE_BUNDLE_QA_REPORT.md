# Candidate Companion Bundle QA Report
======================================

**Document:** `proposals/companion-v1/CANDIDATE_BUNDLE_QA_REPORT.md`  
**Date:** 2026-10-24  
**Author:** Coding & QA Agent (Final Targeted Repair — False-PASS Elimination)  
**Working Branch:** `fix/final-false-pass-repair`  
**Base Commit:** `01554335c1b3e083c953f58db7eced63b1e758ed`  
**Target Proposal:** `proposals/companion-v1/` and `proposals/human-staging/`  
**Status:** `CANDIDATE_QA_PASSED_PENDING_EXTERNAL_APPROVAL` (130/130 Unit & Adversarial Tests PASSED; G0 Preflight STOP_BLOCKER)

---

## 1. Test Suite Summary (Executed Full Suite)

- **Previous Baseline:** 70 passed
- **New Tests / Assertions Added:** +60 tests (+145 assertions)
  - `test_api_preflight.py`: +46 tests (expanded from 16 to 62 passed, covering mock execution blocking, two-model checkpoint identity via `model_path_` and hash, strict workload schema without arbitrary defaults, GPU peak VRAM recording, and aggregation invariants)
  - `test_hta_ref.py`: +14 tests (expanded from 26 to 40 passed, covering separation of declared provenance from verified source lineage, $c_{k050}$ deterministic validator with independent recomputation, units/scope/stage validation, historical lag lineage verification, and boundary-token scanner)
  - `test_staging.py`: 20 passed
  - `test_protocol_contract.py`: 3 passed
  - `test_fixture_schema.py`: 5 passed
- **Final Collected Tests:** 130
- **Passed:** 130
- **Failed:** 0
- **Skipped:** 0
- **Overall Exit Code:** 0

### Full Test Execution Command
```bash
python -m pytest proposals/ -v
```

### Breakdown by Module
1. **Candidate Configuration & Manifest Contracts (`proposals/companion-v1/config/test_protocol_contract.py`):**
   - `test_model_id_meanings_are_not_swapped`: **PASS** (Asserts B1..B8 algorithms, M0/B6 and M1/B8 aliases, and B3a primary benchmark)
   - `test_outcome_config_matches_specification_contract`: **PASS** (Asserts primary $k=0.05$, $zero\_tol=0.0$, thresholds $[0.05, 0.075, 0.10, 0.15, 0.20]$, and test years $2023-2025$)
   - `test_gate_dependencies_and_manifest_execution_order`: **PASS** (Asserts gate prerequisite chaining $G_4 \to G_5 \to G_6 \to G_7 \to G_8/G_9 \to G_{10}$, pilot models in G5, hurdle models in G7, dev experiments in G9, and locked test evaluation in G10)

2. **API Preflight Verification Suite (`proposals/companion-v1/environment/test_api_preflight.py`):**
   - `test_1_all_components_valid_overall_pass`: **PASS** (Legitimate path to `ALL_PROBES_PASSED` when dependencies, checkpoint, and 3 GPU levels pass)
   - `test_2_missing_checkpoint_blocked`: **PASS** (Missing checkpoint fails closed to `BLOCKED_DEPENDENCIES_OR_CHECKPOINT_UNVERIFIED`)
   - `test_3_incorrect_checkpoint_hash_fail`: **PASS** (Hash mismatch reports `FAILED_CHECKPOINT_HASH` / `PROBES_FAILED`)
   - `test_4_cuda_detected_but_inference_untested_blocked`: **PASS** (Level 1 detected + Level 2 untested produces `BLOCKED`)
   - `test_5_incorrect_quantile_shape_fail`: **PASS** (Wrong quantile shape flags `FAILED_GEOMETRY_OR_ALPHA_GRID` / `PROBES_FAILED`)
   - `test_6_missing_official_alpha_levels_never_pass`: **PASS** (Missing `alphas_` never reaches PASS)
   - `test_7_missing_required_ngboost_distribution_fail`: **PASS** (Missing `TFixedDf` flags `FAILED_REGISTERED_DISTRIBUTION_SUPPORT` / `PROBES_FAILED`)
   - `test_8_synthetic_smoke_passes_but_registered_workload_unverified_blocked`: **PASS** (Smoke test pass + unverified workload config produces `BLOCKED_WORKLOAD_UNVERIFIED`)
   - `test_9_fail_plus_blocked_preserves_failure_not_concealed`: **PASS** (Component failure is preserved and reported as `PROBES_FAILED`, not masked by blockers)
   - `test_diagnostic_only_mode_cannot_pass`: **PASS** (`--diagnostic-only` produces `DIAGNOSTIC_ONLY_PREFLIGHT_BLOCKED`)

3. **Fixture Schemas and Mathematical Tolerances (`proposals/companion-v1/fixtures/test_fixture_schema.py`):**
   - `test_csv_file_existence`: **PASS**
   - `test_csv_field_counts_and_headers`: **PASS** (Exact 11 columns in `golden_cases.csv`, exact 9 columns in `golden_expectations.csv`)
   - `test_matching_test_ids`: **PASS** (All 20 test IDs match 1-to-1)
   - `test_near_zero_mathematical_correction`: **PASS** (T-Z02 $y=0$, T-Z03 $y=1$)
   - `test_asymmetric_post_denominator_cuts`: **PASS** (T-P01, T-P02 algebraic reversal verified)

4. **Candidate Reference Functions & Golden Fixtures (`proposals/companion-v1/reference/test_hta_ref.py`):**
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
   - `test_x2_leakage_scanner_and_provenance_contract`: **PASS** (Authorizes legitimate $c_{k050}$, flags unapproved $c_k*$ such as $c_{k075}$, rejects forged $c_{k050}$ with audited inputs, permits pre-cutoff lags, and rejects post-cutoff / unverified features)
   - `test_near_zero_temporal_fold_isolation_and_final_edges`: **PASS** (F1–F4 train-fold edge isolation: altering validation/future $q$ does not alter train edges; final edges use 2015–2022 only and are invariant to test $q$)

5. **Human Staging Tool Security & Sanitization (`proposals/human-staging/test_staging.py`):**
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
   - `test_staging_adversarial_out_dev_equals_master`: **PASS** (Rejects `--out-dev` collision with master; master untouched)
   - `test_staging_adversarial_out_test_features_equals_master`: **PASS** (Rejects `--out-test-features` collision with master; master untouched)
   - `test_staging_adversarial_out_sealed_equals_master`: **PASS** (Rejects `--out-sealed` collision with master; master untouched)
   - `test_staging_adversarial_receipt_equals_master`: **PASS** (Rejects `--receipt` collision with master; master untouched)
   - `test_staging_adversarial_manifest_equals_master`: **PASS** (Rejects `--manifest` collision with master; master untouched)
   - `test_staging_adversarial_distinct_path_strings_resolving_to_same_destination`: **PASS** (Rejects path aliases resolving to same physical file)
   - `test_staging_adversarial_existing_output_without_overwrite_force`: **PASS** (Rejects overwrite without `--force`; succeeds when authorized)
   - `test_staging_adversarial_vault_inside_release_directory`: **PASS** (Rejects placing sealed vault inside release directory)
   - `test_staging_failure_occurs_before_any_source_modification_and_leaves_master_intact`: **PASS** (Ensures failure occurs before any disk writes; master SHA-256 byte-identical)
   - `test_staging_receipt_does_not_claim_os_isolation_without_evidence`: **PASS** (Asserts receipt reports access isolation unverified; permission denial check verified)

---

## 2. API Preflight Probe Verification (`outputs/G0/api_preflight.json`)

- **Execution Command:** `python proposals/companion-v1/environment/api_preflight.py --diagnostic-only`
- **Result:** `DIAGNOSTIC_ONLY_PREFLIGHT_BLOCKED`
- **Recorded Host Environment:**
  - Python: `3.12.2` (AMD64)
  - PyTorch: `2.6.0+cu118` (CUDA 11.8)
  - GPU: `NVIDIA GeForce RTX 3060` (12.0 GB VRAM)
  - Peak Smoke Allocation: `8.13 MB` on `cuda:0`
- **Verified Invariants:**
  - Level 1 (Hardware Detection): `PASSED` (Device detected).
  - Level 2 (Inference Smoke Test): `PASSED` (Executable tensor allocation on GPU).
  - Level 3 (Registered Workload): `BLOCKED_WORKLOAD_UNVERIFIED` (Preapproved workload config missing; toy fit rejected as insufficient).
  - Packages & Checkpoints: `BLOCKED_IMPORT_ERROR` and `BLOCKED_APPROVED_CHECKPOINT_OR_HASH_MISSING`.
  - Zero fabricated passes.

---

## 3. Four Confirmed Defect Resolutions

1. **Defect 1 (API Preflight):**
   - Root cause: Hardware runtime unconditionally assigned `BLOCKED_INFERENCE_MEMORY_UNVERIFIED`; entrypoint lacked CLI options.
   - Resolution: Added CLI arguments (`--checkpoint`, `--expected-sha256`, `--output`, `--diagnostic-only`, `--workload-config`), 3 GPU verification levels, failure preservation, and 10 mock-based unit tests.
2. **Defect 2 (X2 Leakage Scanner):**
   - Root cause: `scan_feature_names()` flagged all `c_k*` fields, conflicting with authorized $c_{k050} = 0.05 \frac{|PAT_{pre}|}{Assets_{pre}}$.
   - Resolution: Authorized $c_{k050}$ while strictly controlling other $c_k*$ (e.g. $c_{k075}$) and created `verify_feature_provenance()` to validate source columns, formula, and timestamps.
3. **Defect 3 (Near-Zero Binning Preprocessing Leakage):**
   - Root cause: Fitting bin edges on all development years (2015–2022) before cross-validation leaked validation-year distributions into historical training folds F1–F4.
   - Resolution: Implemented `fit_fold_near_zero_bins` (train-fold fitting only) and `fit_final_near_zero_bins` (2015–2022 dev-only for final model), added 10 regression assertions, and drafted amendment `docs/PROPOSED_AMENDMENT_FOLD_SAFE_BINNING.md`.
4. **Defect 4 (Master Dataset Protection in Human Staging):**
   - Root cause: Path validation checked output collisions among themselves but omitted checking against `--master`, risking accidental source overwrite.
   - Resolution: Validated all paths against master, prohibited vault inside release folders, implemented atomic temp writes with rollback, and added 10 adversarial tests verifying master SHA-256 remains byte-identical.

---

## 4. Confirmed Implementation Defect Repairs (Defect A & Defect B)

1. **Defect A: Incorrect registered-workload GPU verification (`api_preflight.py`):**
   - **Original defect:** Level 3 evaluated memory feasibility using peak memory measured from a trivial 20x5 PyTorch tensor operation (`t_dummy = torch.randn((20, 5))`) executed during Level 2, rather than executing the actual registered TabICLv2 configuration or measuring real model peak memory. Missing CLI execution flags prevented driving the registered workload verification path.
   - **Corrected implementation:**
     - Level 2 strictly verifies basic CUDA functionality and compute readiness only.
     - Level 3 requires execution of the actual registered TabICLv2 configuration (`context_size`, `n_features`, `n_estimators`, `fit`, and `predict` with `output_type="raw_quantiles"`).
     - GPU peak memory is measured directly from the model execution using `torch.cuda.max_memory_allocated(0)` following `reset_peak_memory_stats(0)`.
     - Checkpoint identity (`checkpoint_path` and verified SHA-256) is recorded and linked to the loaded model.
     - Missing dependencies or unverified checkpoints report `BLOCKED_WORKLOAD_UNVERIFIED`.
     - CUDA OOM (`FAILED_CUDA_OOM`), geometry/alpha monotonicity violations (`FAILED_GEOMETRY_OR_ALPHA_GRID`), memory limit exceedance (`FAILED_EXCEEDS_MEMORY_LIMIT`), and checkpoint checksum mismatches (`FAILED_CHECKPOINT_HASH`) fail closed to `PROBES_FAILED`.
     - Mocked model executions validate orchestration but are explicitly tagged with `evidence_type: "MOCK_ORCHESTRATION_ONLY"` and `mocked: True`, preventing mock results from being misrepresented as real G0 GPU evidence.
     - CLI arguments `--run-workload-test`, `--no-inference-smoke`, and truthful exit codes (0 for PASS, 1 for FAIL, 2 for BLOCKED) enable automated execution and gate verification.
   - **Regression tests:** Added 6 new unit tests in `test_api_preflight.py` covering GPU OOM, memory limits, output geometry/monotonicity, checkpoint linkage, mock tagging, and CLI exit codes.

2. **Defect B: Incorrect source-column substring validation (`hta_ref.py:verify_feature_provenance`):**
   - **Original defect:** `verify_feature_provenance()` included `"d"` as a forbidden substring in `forbidden_sources = ["pat_post", "fs_aud", "audit_adj", "delta", "d", "y_k", "lnst_sau_kiem_toan"]`. Because `"d" in sc` evaluated as a character substring check, legitimate pre-audit financial predictors containing the letter 'd' (e.g. `fixed_assets_pre`, `dividend_pre`, `debt_pre`, `depreciation_pre`, `rnd_pre`) were falsely rejected with `FAILED_LEAKAGE_AUDITED_INPUT`.
   - **Corrected implementation:**
     - Replaced substring matching with controlled semantic token classification: exact forbidden tokens (`forbidden_exact_sources = {"d", "delta", "pat_post", "is_zero_adj", ...}`), controlled prefixes (`"fs_aud_", "audit_adj_", "audit_adjustment_", "post_audit_", "delta_", "d_"`), and controlled suffixes (`"_post", "_aud", "_audited"`).
     - Ensured legitimate pre-audit predictors (`fixed_assets_pre`, `dividend_pre`, `debt_pre`) are recognized as safe and return `PASSED`.
     - Ensured declared metadata alone cannot certify an outcome variable: features matching outcome names/prefixes/suffixes fail closed regardless of declared metadata unless approved as historical lags.
     - Enforced that missing or unparseable prediction cutoffs fail closed (`BLOCKED_INVALID_CUTOFF`), and publication dates after cutoff fail with `FAILED_TEMPORAL_LEAKAGE`.
     - Strictly verified $c_{k050}$ lineage: requires `temporal_status: "t_pre"`, approved pre-audit sources (`pat_pre`, `total_assets_pre`), and verified pre-audit formula.
   - **Regression tests:** Added 4 regression assertion suites in `test_hta_ref.py` explicitly asserting that `fixed_assets_pre`, `dividend_pre`, `debt_pre` pass, outcome variables fail closed, missing cutoffs fail closed, and $c_{k075}$ is rejected.

---

## 5. Final Targeted Repair — False-PASS Vulnerability Elimination (Patches 1 to 7)

### 5.1 Patch 1: Mock Execution Invariants and Overall Aggregation Rules
- **Defective behavior:** Previous aggregation allowed `all(status == "PASSED")` to assign overall `ALL_PROBES_PASSED` even when models were mocked, running on synthetic runners, or lacking verified artifact identity.
- **Remediation:**
  - Mock execution can only ever attain `MOCK_ORCHESTRATION_PASSED` with `evidence_type: "MOCK_ORCHESTRATION_ONLY"` and `mocked: True`.
  - Overall aggregation (`aggregate_overall_status` in `api_preflight.py`) validates deep evidence fields (`mocked is False`, `evidence_type == "REAL_EXECUTION"`, `loaded_checkpoint_verified is True`, and `workload_configuration_verified is True`).
  - Rule: Any component `FAILED` -> `PROBES_FAILED`. Any required component `BLOCKED` or using mocks or unverified checkpoints -> `BLOCKED`. Inconsistent evidence claims -> `BLOCKED_INCONSISTENT_EVIDENCE`.
  - Logical invariant strictly maintained: `report["status"] == "ALL_PROBES_PASSED"` implies real, non-mocked execution of all mandatory engines, verified checkpoints, and verified registered GPU workload.

### 5.2 Patch 2: Independent Dual-Checkpoint Verification & `model_path_` Introspection
- **Defective behavior:** A single checkpoint was self-declared as loaded by TabICL without introspecting post-fit attributes, and was shared between the Regressor and Classifier.
- **Remediation:**
  - Separate CLI arguments for both M1/B8 model components: `--regressor-checkpoint`, `--regressor-sha256`, `--classifier-checkpoint`, `--classifier-sha256`. Older single-checkpoint flags are explicitly deprecated for the regressor only with a warning.
  - Implemented post-fit inspection of `model_path_` and `device_` against official TabICLv2 specifications.
  - Verification pattern: checks `getattr(model, "model_path_", None)`, resolves both paths with `resolve(strict=True)`, computes SHA-256 of the actual loaded file on disk, and verifies matching runtime device (`device_`).
  - If either model's checkpoint is missing, unverified, or mismatched, preflight fails closed.

### 5.3 Patch 3: Removal of Arbitrary Workload Defaults
- **Defective behavior:** Unspecified workload configuration fell back to arbitrary defaults (`max_memory_mb=11000.0`, `context_size=1000`, `n_features=20`, `n_estimators=10`).
- **Remediation:**
  - Implemented strict schema validation via `REQUIRED_WORKLOAD_FIELDS`: requires `context_size`, `n_features`, `n_estimators`, `prediction_batch_size`, `device`, `use_amp`, `offload_mode`, `max_memory_mb`.
  - Missing any required field produces `BLOCKED_WORKLOAD_CONFIG_INCOMPLETE` with zero manufactured defaults.
  - Level 3 records actual parameters used, output shapes, alpha grid, peak allocated and reserved VRAM, exceptions, and loaded artifact identity.

### 5.4 Patch 4: Separation of Declared Provenance from Verified Source Lineage
- **Defective behavior:** `verify_feature_provenance()` returned `status: "PASSED"` solely based on user-supplied metadata dictionaries, equating metadata syntax with empirical lineage proof. Also, `scan_feature_names()` allowed substring bypass for allowlisted terms and failed to detect boundary tokens like `pat_post_over_assets`.
- **Remediation:**
  - `verify_feature_provenance()` now returns the final release decision and NEVER returns `"PASSED"`.
  - Authorized release status is strictly `VERIFIED_SOURCE_LINEAGE`, guarded by `feature_release_authorized(result)`.
  - Metadata-only records return `BLOCKED_SOURCE_LINEAGE_UNVERIFIED`, with detailed subchecks `name_screen: "NAME_SCREEN_PASSED"` and `declared_provenance: "DECLARED_PROVENANCE_CONSISTENT"`.
  - Trusted deterministic validator for $c_{k050}$: requires both `pat_pre` and `total_assets_pre`, resolves columns through human-approved `columns.yaml`, validates monetary units, statement scope (`consolidated`), and reporting stage (`pre_audit`), checks publication timestamps $\le$ cutoff, rejects non-positive assets as NaN, and independently recomputes $0.05 \cdot \frac{|PAT_{pre}|}{Assets_{pre}}$ against stored values.
  - Historical lags require verified `lag_lineage` (matching issuer, exact fiscal-year lag offset, approved source column, availability date $\le$ cutoff).
  - Fixed `scan_feature_names()`: exact allowlist matching, boundary-token regex for forbidden targets, and removal of defective `d_` prefix to preserve valid features like `d_roa`.

### 5.5 Patches 5 & 6: Adversarial & Falsification Test Coverage
- **`test_api_preflight.py` (62 tests):**
  - Asserts that mock execution produces `BLOCKED`, never `ALL_PROBES_PASSED`.
  - Checks missing/wrong `model_path_`, file hash mismatches, missing classifier checkpoint, disabled auto-download, missing workload schema fields, GPU OOM, and CLI exit codes.
- **`test_hta_ref.py` (40 tests):**
  - Added full falsification test suite for $c_{k050}$: valid computation, invalid numerator ($PAT_{post}$ substitution caught and flagged as audited leakage), forged formula description with incorrect executed values, missing denominator, missing source records, unapproved caller aliases, publication date after cutoff, inconsistent monetary units, inconsistent statement scope, non-positive assets handling, row-level failure preservation, passthrough feature verification, historical lag lineage verification, boundary-token name scan, and `d_roa` preservation.

### 5.6 Patch 7: Complete PASS Audit Across Affected Codebases
- Audited every `"status": "PASSED"` in `api_preflight.py` and `hta_ref.py`:
  - In `hta_ref.py`, `"PASSED"` was completely eliminated from `verify_feature_provenance()`. Only `VERIFIED_SOURCE_LINEAGE` can authorize release.
  - In `api_preflight.py`, individual component `PASSED` requires verified real execution (`mocked is False`, `evidence_type == "REAL_EXECUTION"`, `loaded_checkpoint_verified is True`). Overall `ALL_PROBES_PASSED` requires real execution of all mandatory engines, verified checkpoints, and verified registered workload.

---

## 6. Current Dispositions

- **Engineering Code Correctness:** `ENGINEERING_REPAIR_VERIFIED`
  - 130/130 candidate tests passed across all modules.
  - Python compileall succeeded without syntax errors or warnings.
  - Zero fabricated passes; mock-only execution strictly bounded to orchestration testing.
- **Real-World Gate G0 Readiness:** `G0_STOP_BLOCKER`
  - Preflight on actual host remains blocked due to missing external dependencies (`tabicl`, `catboost`, `ngboost`), missing approved model checkpoints (`TabICLRegressor`, `TabICLClassifier`), and unapproved physical column mapping in `columns.yaml`.
  - In accordance with specification rules, no candidate artifacts have been promoted to root paths, and no autonomous approval has been claimed.
