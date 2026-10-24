# Candidate Reference Implementation QA Report

**Document:** `proposals/companion-v1/reference/REFERENCE_QA_REPORT.md`  
**Date:** 2026-10-24  
**Author:** Implementation Agent (G0 Remediation)  
**Target Module:** `proposals/companion-v1/reference/hta_ref.py`  
**Test Suite:** `proposals/companion-v1/reference/test_hta_ref.py`  
**Status:** PENDING_INDEPENDENT_REVIEW  

---

## 1. Test Execution Summary

- **Runner:** `pytest 8.4.2` on Python 3.12.2 (Windows 11)
- **Command:** `python -m pytest proposals/companion-v1/reference/test_hta_ref.py -v`
- **Result:** **12 passed, 0 failed, 0 skipped** in 3.98s
- **Exit Code:** `0`

---

## 2. Detailed Test Results Matrix

| # | Test Function | Target Interface | Mathematical / Spec Rule Verified | Result |
|---|---|---|---|---|
| 1 | `test_build_labels_golden_cases` | `build_labels` | Verified on 14 edge cases (T-L01 to T-L09, T-Z01 to T-Z03, T-M01, T-M02) including corrected T-Z02 and new T-Z03 | **PASS** |
| 2 | `test_build_labels_monotonic_nesting` | `build_labels` | Invariant $y_{0.20} \le y_{0.15} \le y_{0.10} \le y_{0.075} \le y_{0.050}$ holds on 200 random eligible observations | **PASS** |
| 3 | `test_exceedance_prob_analytic_normal` | `exceedance_prob` | Compares 999-quantile prediction against true standard normal CDF at $c=1.0$ (error $< 5 \times 10^{-4}$) | **PASS** |
| 4 | `test_exceedance_prob_monotonic_thresholds` | `exceedance_prob` | Strict monotonic decline $P(|D| \ge 0.05) \ge P(|D| \ge 0.10) \ge P(|D| \ge 0.20)$ | **PASS** |
| 5 | `test_hurdle_exceedance_formula` | `hurdle_exceedance` | Exact mixture $(1 - \pi_0) \cdot P_{\text{cond}}$ verified against analytical probability | **PASS** |
| 6 | `test_scipy_exceedance` | `scipy_exceedance` | Exact two-sided exceedance for Scipy Normal and Laplace distributions | **PASS** |
| 7 | `test_asymmetric_post_cuts` | `asymmetric_post_cuts` | Algebraic formulas verified for $PAT_{pre} = 100$ and $PAT_{pre} = -100$ | **PASS** |
| 8 | `test_temporal_expanding_folds` | `temporal_expanding_folds` | Verified 4 expanding folds (F1–F4) with strict temporal isolation ($\max(\text{train}) < \text{val}$) | **PASS** |
| 9 | `test_paired_firm_cluster_bootstrap` | `paired_firm_cluster_bootstrap` | Firm-clustered bootstrap resamples firms, preserves correlation, computes paired CI | **PASS** |
| 10 | `test_check_feature_leakage` | `check_feature_leakage` | Flags forward-looking publication dates beyond cutoff | **PASS** |
| 11 | `test_target_transform_bijectivity` | `target_transform` / `inverse_target_transform` | Exact bijectivity of $asinh / sinh$ transformations ($|D - \text{rec}| < 10^{-12}$) | **PASS** |
| 12 | `test_crps_score` | `crps_score` | Computes continuous ranked probability score via trapezoidal pinball loss integration | **PASS** |

---

## 3. Discrepancies and Clarifications Noted

1. **Near-Zero Adjustment Specification:** Case T-Z02 in handoff v1 incorrectly listed $y=1$. The candidate implementation strictly implements the mathematical threshold $0.05 \times |PAT_{pre}|$, assigning $y=0$ to $T-Z02$ ($|\Delta|=10^{-10} < 5.0$) and $y=1$ to $T-Z03$ ($|\Delta|=10^{-10} \ge 5 \times 10^{-11}$).
2. **Missing Input Tolerance:** `build_labels` gracefully handles NaNs and non-positive total assets, setting `eligible_model = False` without raising unhandled exceptions.
3. **Candidate Status:** These unit tests confirm internal consistency. Under Rule A3, this module remains a **candidate proposal** until independently certified and promoted by the human research lead.
