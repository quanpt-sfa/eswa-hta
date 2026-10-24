# Specification Traceability & Consistency Matrix

**Document:** `proposals/companion-v1/TRACEABILITY_MATRIX.md`  
**Date:** 2026-10-24  
**Author:** Implementation Agent (G0 Remediation)  
**Governance Specification:** `docs/HTA_IMPLEMENTATION_SPEC_v1.2.md`  
**Status:** CANDIDATE_PROPOSAL_AUDITED (All 15 consistency checks completed)  

---

## 1. Fifteen-Point Internal Consistency Review

| # | Consistency Audit Dimension | Audit Finding & Implementation Evidence | Status |
|---|---|---|---|
| 1 | **Function Name Consistency** | Exact canonical function names aligned across `docs/`, `contracts/COMPANION_BUNDLE_HANDOFF.md`, and `proposals/companion-v1/reference/hta_ref.py`. | **VERIFIED** |
| 2 | **Mathematical Expectations** | T-Z02 corrected to $y=0$; T-Z03 added ($y=1$); analytical proofs in `GOLDEN_CASE_DERIVATIONS.md`. | **VERIFIED** |
| 3 | **CSV Field-Count Consistency** | Exact field counts verified by automated tests: 11 fields for `golden_cases.csv`, 9 fields for `golden_expectations.csv`. | **VERIFIED** |
| 4 | **YAML Key Spec Compliance** | `protocol.yaml`, `columns.yaml`, and `gate_acceptance_manifest.yaml` keys mapped strictly to v1.2 sections. | **VERIFIED** |
| 5 | **Gate Dependency Acyclicity** | Strict linear ordering $G_0 \to G_1 \to \dots \to G_{12}$ without circular dependencies in `gate_acceptance_manifest.yaml`. | **VERIFIED** |
| 6 | **Completeness for Later Gates** | All required functions for G1–G12 (`build_labels`, `exceedance_prob`, `hurdle_exceedance`, `asymmetric_post_cuts`, `temporal_expanding_folds`, `paired_firm_cluster_bootstrap`, `near_zero_bins`, `evaluate_predictions`, `check_feature_leakage`, `target_transform`, `crps_score`) implemented. | **VERIFIED** |
| 7 | **Independent Test Logic** | Tests validate against analytic continuous distributions, exact fractions, and boundary conditions, never simply re-calling the tested function. | **VERIFIED** |
| 8 | **Zero & Near-Zero Handling** | Exact-zero convention ($\Delta=0 \implies is\_zero\_adj=1$) and near-zero earnings threshold exceedance ($T-Z01$, $T-Z02$, $T-Z03$) verified. | **VERIFIED** |
| 9 | **Negative Pre-Audit Earnings** | $PAT_{pre} < 0$ cases ($T-L07$, $T-L08$, $T-L09$) verified; asymmetric cut magnitude reversal ($T-P02$) algebraically proven. | **VERIFIED** |
| 10 | **Temporal Leakage Prevention** | Point-in-time guard `check_feature_leakage` asserts publication date $\le$ prediction cutoff date. | **VERIFIED** |
| 11 | **Expanding Window Splitting** | `temporal_expanding_folds` anchors train at 2015, expands window, and strictly precedes validation year. | **VERIFIED** |
| 12 | **Quantile-to-CDF Interpolation** | Linear interpolation between quantiles with monotonic sorting enforcement and conservative tail clamping to $[0, 1]$. | **VERIFIED** |
| 13 | **Firm-Cluster Bootstrap Unit** | `paired_firm_cluster_bootstrap` resamples unique firm clusters with replacement, preserving all within-firm temporal dependencies. | **VERIFIED** |
| 14 | **API Assumption Boundaries** | TabICL 999 raw quantiles, CatBoost MultiQuantile, and NGBoost distributions marked provisional until live API probe runs. | **VERIFIED** |
| 15 | **Test Outcome Holdout Isolation** | G0–G9 manifests strictly forbid test outcome access; `HUMAN_UNLOCK_G10.txt` required before G10 evaluation. | **VERIFIED** |

---

## 2. Requirement Traceability Matrix

| Spec Section | Requirement Description | Candidate Module | Test ID | Executed Test Result | Human Approval Status |
|---|---|---|---|---|---|
| **§2 D01–D04, §5 G1b** | Label construction, eligibility, delta, D | `hta_ref.py:build_labels` | `T-L01` to `T-L06` | **PASS (5 passed)** | PENDING_REVIEW |
| **§2 D04–D05** | Negative earnings & sign crossing | `hta_ref.py:build_labels` | `T-L05`, `T-L07..09` | **PASS** | PENDING_REVIEW |
| **§2 D07** | Exact zero adjustment convention | `hta_ref.py:build_labels` | `T-Z01` | **PASS** | PENDING_REVIEW |
| **§15.2 (Fix)** | Subthreshold small adjustment ($y=0$) | `hta_ref.py:build_labels` | `T-Z02` | **PASS** | PENDING_REVIEW |
| **§15.2 (New)** | Near-zero earnings exceedance ($y=1$) | `hta_ref.py:build_labels` | `T-Z03` | **PASS** | PENDING_REVIEW |
| **§2 D04** | Asset positivity constraint | `hta_ref.py:build_labels` | `T-M01`, `T-M02` | **PASS** | PENDING_REVIEW |
| **§2 D03, D06** | Multi-threshold monotonic nesting | `hta_ref.py:build_labels` | `test_monotonic_nesting` | **PASS** | PENDING_REVIEW |
| **§2 D13–D15** | Quantile to exceedance probability | `hta_ref.py:exceedance_prob` | `T-C01`, `test_exceedance_prob` | **PASS** | PENDING_REVIEW |
| **§2 D17** | Hurdle probability mixture | `hta_ref.py:hurdle_exceedance` | `T-H01` | **PASS** | PENDING_REVIEW |
| **§2 D16** | Analytic Scipy distribution exceedance | `hta_ref.py:scipy_exceedance` | `T-C01`, `T-C02` | **PASS** | PENDING_REVIEW |
| **§15.2** | Post-denominator asymmetric cuts | `hta_ref.py:asymmetric_post_cuts` | `T-P01`, `T-P02` | **PASS** | PENDING_REVIEW |
| **§2 D10, §5 G4** | Expanding window temporal folds (F1–F4)| `hta_ref.py:temporal_expanding_folds` | `test_temporal_folds` | **PASS** | PENDING_REVIEW |
| **§2 D20, §15.2** | Paired firm-cluster bootstrap | `hta_ref.py:paired_firm_cluster_bootstrap` | `test_cluster_bootstrap` | **PASS** | PENDING_REVIEW |
| **§2 D22** | Near-zero quintile bins on dev set | `hta_ref.py:near_zero_bins` | Unit test probe | **PASS** | PENDING_REVIEW |
| **§2 D20, D23** | Diagnostic metrics (AP, Brier, ECE) | `hta_ref.py:evaluate_predictions` | Unit test probe | **PASS** | PENDING_REVIEW |
| **§2 D08–D09** | Feature leakage publication guard | `hta_ref.py:check_feature_leakage` | `test_check_feature_leakage` | **PASS** | PENDING_REVIEW |
| **§2 D18** | Target transform asinh / sinh | `hta_ref.py:target_transform` | `test_target_transform` | **PASS** | PENDING_REVIEW |
| **§15.3** | Distributional scoring (CRPS) | `hta_ref.py:crps_score` | `test_crps_score` | **PASS** | PENDING_REVIEW |
| **§15.1** | Research protocol configuration | `proposals/.../protocol.yaml` | `test_yaml_parsing` | **PASS (Valid YAML)** | PENDING_REVIEW |
| **§15.1, §1 H1** | Physical column mapping contract | `proposals/.../columns.yaml` | `test_yaml_parsing` | **PASS (Valid YAML)** | PENDING_REVIEW |
| **§16.1** | Gate G0–G12 acceptance contracts | `proposals/.../gate_acceptance_manifest.yaml` | `test_yaml_parsing` | **PASS (Valid YAML)** | PENDING_REVIEW |
| **§4.3, Rule A5**| Physical holdout staging & sanitization | `proposals/human-staging/stage_data.py` | `test_staging.py` | **PASS (2 passed)** | PENDING_REVIEW |
| **§5 G0, §15.3** | Automated API preflight probe | `proposals/.../api_preflight.py` | Syntax check | **READY (Execution Blocked)** | PENDING_REVIEW |
