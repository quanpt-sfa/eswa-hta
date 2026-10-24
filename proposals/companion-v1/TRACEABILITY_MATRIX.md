# Specification Traceability & Consistency Matrix

**Document:** `proposals/companion-v1/TRACEABILITY_MATRIX.md`  
**Date:** 2026-10-24  
**Author:** Implementation Agent (G0 Remediation & Independent Audit Repair)  
**Governance Specification:** `docs/HTA_IMPLEMENTATION_SPEC_v1.2.md`  
**Status:** `CANDIDATE_PROPOSAL_AUDITED` (All 15 consistency dimensions verified, 42/42 candidate unit tests passed)

---

## 1. Fifteen-Point Internal Consistency Review

| # | Consistency Audit Dimension | Audit Finding & Implementation Evidence | Status |
|---|---|---|---|
| 1 | **Function Name Consistency** | Exact canonical function names aligned across `docs/`, `API_CONTRACT.md`, and `hta_ref.py`. Canonical aliases provide dual compatibility (`ap`, `post_denominator_cuts`, `two_sided_prob`, `paired_cluster_bootstrap`, `crps_from_quantiles`). | **VERIFIED** |
| 2 | **Mathematical Expectations** | T-Z02 corrected to $y=0$; T-Z03 added ($y=1$); analytical proofs in `GOLDEN_CASE_DERIVATIONS.md`; all 20 golden cases executed and passing. | **VERIFIED** |
| 3 | **CSV Field-Count Consistency** | Exact field counts verified by automated tests: 11 fields for `golden_cases.csv`, 9 fields for `golden_expectations.csv`. | **VERIFIED** |
| 4 | **YAML Key Spec Compliance** | `protocol.yaml`, `columns.yaml`, and `gate_acceptance_manifest.yaml` keys mapped strictly to v1.2 sections. Model registry B1–B8 invariant confirmed. | **VERIFIED** |
| 5 | **Gate Dependency Acyclicity** | Strict linear ordering $G_0 \to G_1 \to \dots \to G_{12}$ without circular dependencies in `gate_acceptance_manifest.yaml`. | **VERIFIED** |
| 6 | **Completeness for Later Gates** | All required functions for G1–G12 (`build_labels`, `expand_thresholds`, `exceedance_prob`, `hurdle_exceedance`, `post_denominator_cuts`, `two_sided_prob`, `temporal_expanding_folds`, `assert_fold_integrity`, `paired_cluster_bootstrap`, `near_zero_edges`, `assign_bins`, `scan_feature_names`, `numeric_leak_check`, `precision_recall_at_k`, `ap`, `crps_from_quantiles`, `target_transform`) implemented. | **VERIFIED** |
| 7 | **Independent Test Logic** | Tests validate against analytic continuous distributions, exact fractions, and boundary conditions, never simply re-calling the tested function. | **VERIFIED** |
| 8 | **Zero & Near-Zero Handling** | Exact-zero convention ($\Delta=0 \implies is\_zero\_adj=1$) and near-zero earnings threshold exceedance ($T-Z01$, $T-Z02$, $T-Z03$) verified. | **VERIFIED** |
| 9 | **Negative Pre-Audit Earnings** | $PAT_{pre} < 0$ cases ($T-L07$, $T-L08$, $T-L09$) verified; asymmetric cut magnitude reversal ($T-P02$) algebraically proven. | **VERIFIED** |
| 10 | **Temporal Leakage Prevention** | Point-in-time guards `check_feature_leakage`, `scan_feature_names`, and `numeric_leak_check` fail closed on missing dates or forbidden names. | **VERIFIED** |
| 11 | **Expanding Window Splitting** | `temporal_expanding_folds` anchors train at 2015; `assert_fold_integrity` enforces non-overlap, strictly increasing validation years, and zero firm-year leakage. | **VERIFIED** |
| 12 | **Quantile-to-CDF Interpolation** | Linear interpolation between quantiles with monotonic sorting enforcement and conservative tail clamping to $[0, 1]$. | **VERIFIED** |
| 13 | **Firm-Cluster Bootstrap Unit** | `paired_cluster_bootstrap` resamples unique firm clusters with replacement, preserving all within-firm temporal dependencies. | **VERIFIED** |
| 14 | **API Assumption Boundaries** | TabICL 999 raw quantiles, CatBoost MultiQuantile, and NGBoost distributions marked provisional until live API probe runs; fails closed without fabricating passes. | **VERIFIED** |
| 15 | **Test Outcome Holdout Isolation** | G0–G9 manifests strictly forbid test outcome access; `HUMAN_UNLOCK_G10.txt` required before G10 evaluation; staging enforces positive selection. | **VERIFIED** |

---

## 2. Requirement Traceability Matrix

| Spec Section | Requirement Description | Candidate Module | Test ID / Name | Executed Test Result | Human Approval Status |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **§2 D01–D04, §5 G1b** | Label construction, eligibility, delta, D | `hta_ref.py:build_labels` | `test_build_labels_golden_cases`, `test_all_golden_fixtures_from_csv` | **PASS (20/20)** | PENDING_REVIEW |
| **§2 D04–D05** | Negative earnings & sign crossing | `hta_ref.py:build_labels` | `T-L05`, `T-L07..09` | **PASS** | PENDING_REVIEW |
| **§2 D07** | Exact zero adjustment convention | `hta_ref.py:build_labels` | `T-Z01` | **PASS** | PENDING_REVIEW |
| **§15.2 (Fix)** | Subthreshold small adjustment ($y=0$) | `hta_ref.py:build_labels` | `T-Z02` | **PASS** | PENDING_REVIEW |
| **§15.2 (New)** | Near-zero earnings exceedance ($y=1$) | `hta_ref.py:build_labels` | `T-Z03` | **PASS** | PENDING_REVIEW |
| **§2 D04** | Asset positivity constraint | `hta_ref.py:build_labels` | `T-M01`, `T-M02` | **PASS** | PENDING_REVIEW |
| **§2 D03, D06** | Multi-threshold monotonic nesting | `hta_ref.py:build_labels` | `test_build_labels_monotonic_nesting` | **PASS** | PENDING_REVIEW |
| **§2 D06, §6.2, §8.3** | Long-format threshold expansion | `hta_ref.py:expand_thresholds` | `test_expand_thresholds` | **PASS** | PENDING_REVIEW |
| **§2 D13–D15** | Quantile to exceedance probability | `hta_ref.py:exceedance_prob` | `T-C01`, `test_exceedance_prob_analytic_normal` | **PASS** | PENDING_REVIEW |
| **§2 D17** | Hurdle probability mixture | `hta_ref.py:hurdle_exceedance` | `T-H01`, `test_hurdle_exceedance_formula` | **PASS** | PENDING_REVIEW |
| **§2 D16** | Analytic Scipy distribution exceedance | `hta_ref.py:scipy_exceedance` | `T-C01`, `T-C02`, `test_scipy_exceedance` | **PASS** | PENDING_REVIEW |
| **§15.2, §9.3 R1** | Post-denominator asymmetric cuts | `hta_ref.py:post_denominator_cuts` | `T-P01`, `T-P02`, `test_asymmetric_post_cuts`, `test_post_denominator_cuts_d_scale` | **PASS** | PENDING_REVIEW |
| **§15.2 T-C01, §9.3 R1** | Two-sided probability calculation | `hta_ref.py:two_sided_prob` | `test_two_sided_prob_symmetric_and_asymmetric` | **PASS** | PENDING_REVIEW |
| **§2 D10, §5 G4** | Expanding window temporal folds (F1–F4)| `hta_ref.py:temporal_expanding_folds` | `test_temporal_expanding_folds` | **PASS** | PENDING_REVIEW |
| **§2 D10, §5 G4** | Fold integrity assertion | `hta_ref.py:assert_fold_integrity` | `test_assert_fold_integrity` | **PASS** | PENDING_REVIEW |
| **§2 D20, §15.2 T-B01** | Paired firm-cluster bootstrap | `hta_ref.py:paired_cluster_bootstrap` | `test_paired_firm_cluster_bootstrap` | **PASS** | PENDING_REVIEW |
| **§2 D22, §5 G4** | Near-zero edges & bin assignment | `hta_ref.py:near_zero_edges`, `assign_bins` | `test_near_zero_edges_and_assign_bins`, `test_near_zero_quintiles_reject_duplicated_edges` | **PASS** | PENDING_REVIEW |
| **§2 D20, §9.1** | Average Precision with score-ties | `hta_ref.py:ap` | `test_average_precision_ties_invariant_to_row_order` | **PASS** | PENDING_REVIEW |
| **§2 D21, §9.1** | Precision@K and Recall@K | `hta_ref.py:precision_recall_at_k` | `test_precision_recall_at_k` | **PASS** | PENDING_REVIEW |
| **§2 D08–D09, §5 G2** | Feature publication cutoff guard | `hta_ref.py:check_feature_leakage` | `test_check_feature_leakage`, `test_unknown_publication_date_is_not_safe` | **PASS** | PENDING_REVIEW |
| **§2 D08, §5 G2** | Feature name token leakage scan | `hta_ref.py:scan_feature_names` | `test_scan_feature_names` | **PASS** | PENDING_REVIEW |
| **§5 G2** | Numeric leakage diagnostics | `hta_ref.py:numeric_leak_check` | `test_numeric_leak_check` | **PASS** | PENDING_REVIEW |
| **§2 D18** | Target transform asinh / sinh | `hta_ref.py:target_transform` | `test_target_transform_bijectivity` | **PASS** | PENDING_REVIEW |
| **§14 Rule 8, §15.2** | Distributional scoring (CRPS) | `hta_ref.py:crps_from_quantiles` | `test_crps_score` | **PASS** | PENDING_REVIEW |
| **§8.3, §15.1** | Protocol model registry (B1–B8) | `config/protocol.yaml` | `test_model_id_meanings_are_not_swapped` | **PASS** | PENDING_REVIEW |
| **§2 D07, §15.1** | Protocol outcome contract | `config/protocol.yaml` | `test_outcome_config_matches_specification_contract` | **PASS** | PENDING_REVIEW |
| **§16.1** | Gate manifest execution order | `contracts/gate_acceptance_manifest.yaml` | `test_gate_dependencies_and_manifest_execution_order` | **PASS** | PENDING_REVIEW |
| **§4.3, Rule A5** | Physical holdout staging & sanitization | `proposals/human-staging/stage_data.py` | `test_staging.py` (10 tests) | **PASS (10/10)** | PENDING_REVIEW |
| **§5 G0, §15.3** | Automated API preflight probe | `proposals/.../api_preflight.py` | Syntax & execution probe | **BLOCKED (Uninstalled)** | PENDING_REVIEW |
