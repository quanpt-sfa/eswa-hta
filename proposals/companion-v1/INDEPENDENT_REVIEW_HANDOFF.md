# Independent Review Handoff Report — Gate G0 Candidate Remediation

**Date:** 2026-10-24  
**Author:** Implementation & Verification Agent  
**Repository:** `quanpt-sfa/eswa-hta`  
**Review Branch:** `review/g0-independent-audit-fixes`  
**Base Pull Request:** PR #1 (`review/g0-independent-audit-fixes` $\to$ `main`)  
**Status Disposition:** `CANDIDATE_ENGINEERING_QA_PASSED`  
**Overall Governance Gate G0 Status:** `STOP_BLOCKER` (Rule A1, Rule A3: Awaiting human certification and physical holdout isolation)

---

## A. Git State and Repository Integrity

* **Review Branch:** `review/g0-independent-audit-fixes`
* **Starting Commit:** `bad24bf` (*Mark old 19-pass QA result as historical and unverified after independent audit patches*)
* **Remote Tracking:** `origin/review/g0-independent-audit-fixes`
* **Working Tree State:** Clean / Uncommitted candidate proposal updates staged for local commit
* **Public Repository Protection:** Zero raw datasets, zero parquet files, zero binary model checkpoints, and zero credentials tracked.
* **Approval/Unlock Absence Check:** `Get-ChildItem -Recurse | Where-Object { $_.Name -match "APPROVED" -or $_.Name -match "UNLOCK" }` $\implies$ **0 files found (PASS)**.

---

## B. Reference API Completeness

All 16 interfaces required by specification v1.2 (§2, §6, §8, §9, §14, §15) are implemented in `proposals/companion-v1/reference/hta_ref.py`:

| Interface Required by Spec v1.2 | Candidate Implementation Path | Supporting Test ID / Test Name | Implementation & QA Status |
| :--- | :--- | :--- | :---: |
| `build_labels` | `hta_ref.py:build_labels` | `test_build_labels_golden_cases`, `test_build_labels_monotonic_nesting`, `test_all_golden_fixtures_from_csv` | **PASS (Candidate)** |
| `expand_thresholds` | `hta_ref.py:expand_thresholds` | `test_expand_thresholds` | **PASS (Candidate)** |
| `exceedance_prob` | `hta_ref.py:exceedance_prob` | `test_exceedance_prob_analytic_normal`, `test_exceedance_prob_monotonic_thresholds` | **PASS (Candidate)** |
| `hurdle_exceedance` | `hta_ref.py:hurdle_exceedance` | `test_hurdle_exceedance_formula`, `test_all_golden_fixtures_from_csv (T-H01)` | **PASS (Candidate)** |
| `scipy_exceedance` | `hta_ref.py:scipy_exceedance` | `test_scipy_exceedance`, `test_all_golden_fixtures_from_csv (T-C01, T-C02)` | **PASS (Candidate)** |
| `post_denominator_cuts` | `hta_ref.py:post_denominator_cuts` | `test_asymmetric_post_cuts`, `test_post_denominator_cuts_d_scale`, `T-P01, T-P02` | **PASS (Candidate)** |
| `two_sided_prob` | `hta_ref.py:two_sided_prob` | `test_two_sided_prob_symmetric_and_asymmetric` | **PASS (Candidate)** |
| `near_zero_edges` | `hta_ref.py:near_zero_edges` | `test_near_zero_edges_and_assign_bins`, `test_near_zero_quintiles_reject_duplicated_edges` | **PASS (Candidate)** |
| `assign_bins` | `hta_ref.py:assign_bins` | `test_near_zero_edges_and_assign_bins` | **PASS (Candidate)** |
| `assert_fold_integrity` | `hta_ref.py:assert_fold_integrity` | `test_assert_fold_integrity` | **PASS (Candidate)** |
| `scan_feature_names` | `hta_ref.py:scan_feature_names` | `test_scan_feature_names` | **PASS (Candidate)** |
| `numeric_leak_check` | `hta_ref.py:numeric_leak_check` | `test_numeric_leak_check` | **PASS (Candidate)** |
| `crps_from_quantiles` | `hta_ref.py:crps_from_quantiles` | `test_crps_score` | **PASS (Candidate)** |
| `precision_recall_at_k` | `hta_ref.py:precision_recall_at_k` | `test_precision_recall_at_k` | **PASS (Candidate)** |
| `ap` | `hta_ref.py:ap` | `test_average_precision_ties_invariant_to_row_order` | **PASS (Candidate)** |
| `paired_cluster_bootstrap` | `hta_ref.py:paired_cluster_bootstrap` | `test_paired_firm_cluster_bootstrap` | **PASS (Candidate)** |
| `TABICL_RAW_ALPHAS` | `hta_ref.py:TABICL_RAW_ALPHAS` | Verified unresolved fail-closed constant in `api_preflight.py` | **BLOCKED (Unresolved)** |

---

## C. Executed QA Summary

All local candidate test suites were executed cleanly under Python 3.12.2 (`pytest 8.4.2`):

| Test Suite | Invocation Command | Exit Code | Collected | Passed | Failed | Skipped | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Candidate Tests** | `python -m pytest proposals/ -v` | 0 | 42 | 42 | 0 | 0 | **PASS** |
| **Candidate Reference Functions** | `python -m pytest proposals/companion-v1/reference/test_hta_ref.py -v` | 0 | 24 | 24 | 0 | 0 | **PASS** |
| **Human Staging Security** | `python -m pytest proposals/human-staging/test_staging.py -v` | 0 | 10 | 10 | 0 | 0 | **PASS** |
| **Protocol & Manifest Contracts** | `python -m pytest proposals/companion-v1/config/test_protocol_contract.py -v` | 0 | 3 | 3 | 0 | 0 | **PASS** |
| **CSV Fixture Schemas & Tolerances** | `python -m pytest proposals/companion-v1/fixtures/test_fixture_schema.py -v` | 0 | 5 | 5 | 0 | 0 | **PASS** |
| **Bytecode Syntax Compilation** | `python -m compileall -q proposals/` | 0 | — | — | — | — | **PASS** |

---

## D. Mathematical Verification

1. **Near-Zero Corrections (T-Z02 vs T-Z03):**
   * **T-Z02 ($PAT_{pre}=100, PAT_{post}=100.0000000001, k=0.05$):**  
     $|\Delta| = 10^{-10} < 5.0 = 0.05 \times 100 \implies \mathbf{y = 0}$ (Subthreshold adjustment correctly labeled 0).
   * **T-Z03 ($PAT_{pre}=10^{-9}, PAT_{post}=1.1 \times 10^{-9}, k=0.05$):**  
     $|\Delta| = 10^{-10} \ge 5 \times 10^{-11} = 0.05 \times 10^{-9} \implies \mathbf{y = 1}$ (Genuine exceedance on near-zero base).
2. **Post-Denominator Asymmetric Cuts Proof:**
   * Algebraic derivation proven:
     * When $PAT_{pre} > 0$: Lower cut $\Delta \le \frac{-k}{1+k} PAT_{pre} \approx -4.7619\%$; Upper cut $\Delta \ge \frac{k}{1-k} PAT_{pre} \approx +5.2632\%$.
     * When $PAT_{pre} < 0$: Lower cut $\Delta \le \frac{-k}{1-k} |PAT_{pre}| \approx -5.2632\%$; Upper cut $\Delta \ge \frac{k}{1+k} |PAT_{pre}| \approx +4.7619\%$.
   * Magnitudes reverse for loss firms; normalized D-scale cuts scale by $1 / Assets_{pre}$. Verified in `test_post_denominator_cuts_d_scale`.
3. **Average Precision with Score Ties:**
   * Predictions with tied scores form one precision-recall step identical to scikit-learn; row order invariance proven by `test_average_precision_ties_invariant_to_row_order`.
4. **Firm-Cluster Bootstrap Alignment:**
   * Resamples permanent firm IDs; retains all firm-year observations per selected cluster; evaluates paired differences. Verified in `test_paired_firm_cluster_bootstrap`.

---

## E. Security Validity

### Synthetic Staging Red-Team Attacks
10/10 adversarial attack tests passed in `test_staging.py`:
- Rejects missing allowlist (fails closed).
- Rejects allowlist with `pat_post` or `audit_adj_*`.
- Rejects missing required primary keys (`firm_id`, `fiscal_year`, `scope`).
- Rejects duplicate primary keys in master source.
- Rejects out-of-horizon fiscal years (outside 2015–2025).
- Rejects overlapping output destinations.
- Rejects missing target inputs in sealed source.

### Real Holdout Physical Isolation Status
* **Status:** `NOT VERIFIED (STOP_BLOCKER)`
* **Operational Boundary:** Real master dissertation data was NOT accessed, read, staged, or processed by the agent.
* **Separation of Evidence:** File creation receipts are explicitly distinguished from OS-level permission-denial probes. An access-denial test under the agent's identity must pass before physical holdout isolation can be certified.

---

## F. Remaining Blockers (Human-Controlled Actions)

1. **Companion Bundle Certification (OQ-001):** Human research lead reviews `proposals/companion-v1/`, approves `protocol.yaml`, `columns.yaml`, `gate_acceptance_manifest.yaml`, and copies candidate `hta_ref.py` and `test_hta_ref.py` into production paths `reference/` and `fixtures/` with signed `fixture_manifest.sha256`.
2. **Physical Holdout Staging Execution (OQ-002):** Human data steward executes `stage_data.py` on real unredacted data, places sealed outcomes in an external vault, enforces OS-level Deny ACL / detached boundary, and completes `contracts/staged_holdout_receipt.md`.
3. **Dependency Pinning & Checkpoint Verification (OQ-003):** Human lead approves `requirements.candidate.txt`, authorizes installation into `.venv`, supplies approved TabICLv2 checkpoint SHA-256 in `protocol.yaml`, and runs `api_preflight.py` to produce `environment_lock.txt`.
4. **Public Repository Policy Sign-Off (OQ-004):** Human lead reviews branch protection rules on GitHub and approves commit/push policies.

---

## G. Final Recommendation

**Disposition:** `CANDIDATE_ENGINEERING_QA_PASSED`  
All candidate code, interfaces, configurations, fixtures, and security tests pass engineering QA cleanly without fabrications or short-cuts. The candidate bundle is completely prepared for independent mathematical review and human sign-off.

**Overall Gate G0 Disposition:** **`STOP_BLOCKER`**  
In accordance with Rules A1, A2, and A3, Gate G0 cannot be declared PASSED until the human companion bundle promotion, physical holdout isolation, and environment checkpoint locks are certified by the human lead.
