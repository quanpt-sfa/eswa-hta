# Fixture Validation Report

**Document:** `proposals/companion-v1/fixture_validation_report.md`  
**Date:** 2026-10-24  
**Author:** Implementation Agent (G0 Remediation)  
**Status:** PROPOSAL_VALIDATED (Candidate fixtures parsed, verified, and ready for human review)  

---

## 1. Executive Summary

A comprehensive automated audit was conducted on the candidate golden fixtures:
- `proposals/companion-v1/fixtures/golden_cases.csv` (20 rows)
- `proposals/companion-v1/fixtures/golden_expectations.csv` (20 rows)

All tests passed with zero errors:
1. **Field-Count Consistency:** Exactly 11 columns in every row of `golden_cases.csv`, exactly 9 columns in every row of `golden_expectations.csv`. Zero parsing irregularities.
2. **Key Alignment:** 100% 1-to-1 match of `test_id` across inputs and expectations.
3. **Mathematical Correction:**
   - Case **T-Z02** was corrected from $y=1$ to $y=0$ (small adjustment $|\Delta|=10^{-10}$ is below materiality threshold $0.05 \times 100 = 5.0$).
   - Case **T-Z03** was introduced to rigorously test near-zero earnings with genuine exceedance ($|\Delta|=10^{-10} \ge 0.05 \times 10^{-9} = 5 \times 10^{-11} \implies y=1$).
4. **Post-Denominator Asymmetry:** Algebraic proofs and numerical tolerances verified for both positive and negative earnings.

---

## 2. Test Execution Details

- **Test Suite:** `proposals/companion-v1/fixtures/test_fixture_schema.py`
- **Runner:** `pytest 8.4.2` on Python 3.12.2
- **Command:** `python -m pytest proposals/companion-v1/fixtures/test_fixture_schema.py -v`
- **Result:** 5 tests collected, 5 passed, 0 failed in 2.24s (Exit Code: 0).

| Test Name | Verified Property | Status |
|---|---|---|
| `test_csv_file_existence` | Files exist on disk | **PASS** |
| `test_csv_field_counts_and_headers` | Strict field counts (11 and 9) | **PASS** |
| `test_matching_test_ids` | Uniqueness and strict alignment | **PASS** |
| `test_near_zero_mathematical_correction` | T-Z02 ($y=0$) and T-Z03 ($y=1$) | **PASS** |
| `test_asymmetric_post_denominator_cuts` | Algebraic formulas for $PAT_{pre} \gtrless 0$ | **PASS** |

---

## 3. Human Review Notice

These fixtures are **candidate proposals**. Under Rule A3, they must be independently certified by the research lead before being promoted to `fixtures/golden_cases.csv` and `fixtures/golden_expectations.csv`.
