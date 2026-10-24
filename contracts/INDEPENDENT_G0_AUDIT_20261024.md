# Independent G0 Candidate Audit — 2026-10-24

**Scope:** Candidate proposal bundle only, baseline Git commit `7ea7fe038989e352c199b0397a6c588b98579314`.  
**Review branch:** `review/g0-independent-audit-fixes`.  
**Status:** **CHANGES PROPOSED — NOT APPROVED — NO G0 PASS**.

## Evidence inspected

Reviewed the repository tree, v1.2 implementation contract, candidate `hta_ref.py`, candidate unit tests and fixtures, `protocol.yaml`, `gate_acceptance_manifest.yaml`, `stage_data.py`, staging tests, and API preflight. Recomputed the tie example with independent scikit-learn: `average_precision_score([1,0],[0.5,0.5]) = 0.5`; reversed label order also gives 0.5.

**Execution limitation:** Full project `pytest` and model API/checkpoint probes have **not** been executed on this review branch. Files were inspected and edited using the connected GitHub repository; the runner that has the live project environment remains the user's local agent. Earlier "19 passed" was recorded against the older commit, and must NOT be treated as post-patch evidence.

## Confirmed defects corrected in this branch

| Severity | Defect at baseline | Remediation |
|---|---|---|
| CRITICAL | `protocol.yaml` B1–B8 identities did not match specification: B3a incorrectly MultiQuantile; B8 incorrectly a CatBoost ensemble | Aligned model registry with §8.3; B3a is threshold-informed CatBoost classifier, B8 is Hurdle-TabICLv2, M0/B6 and M1/B8 aliases corrected |
| CRITICAL | Human-only staging accepted all non-blacklisted columns; `fs_aud_*` and `audit_adj_*` could leak | Require a human-approved explicit output feature allowlist, reject audited and adjustment patterns, reject missing allowlist; add malicious-source regression tests |
| HIGH | AP implementation counted equal scores in arbitrary input order | Use `sklearn.metrics.average_precision_score`; add label-order-invariance tests at equal scores |
| HIGH | Missing publication timestamps evaluated as safe (NaT comparison returned False) | Treat missing/invalid feature publication and missing cutoffs as unsafe in both date guards; add tests |
| HIGH | Staging receipt falsely claimed `STAGED_AND_VERIFIED` without access-denial evidence | Rename to `STAGING_OUTPUTS_CREATED__ACCESS_ISOLATION_UNVERIFIED`, and explain independent OS-level check |
| HIGH | G5/G6/G9/G10 in candidate manifest contradicted v1.2 gate execution order and final-test lock | Align G5 pilot, G6 decision, G9 development comparisons and G10 two-phase locked test |
| HIGH | `api_preflight.py` marked component checks PASSED without checking shapes/classes or approved checkpoint checksum | Add actual expected-geometry and checksum comparisons, fail closed when reference alphas/checkpoint hash unavailable |
| MEDIUM | Duplicate near-zero quantile bin edges could cause undefined binning or runtime failure | Explicitly reject tied edges pending human-approved fallback |
| MEDIUM | Zero valid cluster-bootstrap samples produced misleading numerical behavior | Raise explicit error if fewer than two valid replications |
| MEDIUM | Candidate YAML outcome keys were incompatible with `protocol.outcome.zero_tol` / `protocol.outcome.thresholds` used in specification | Rename candidate outcome section/keys and add registry contract tests |

## Remaining blockers BEFORE any G0 approval or candidate promotion

1. **Do not promote candidate reference yet.** Several spec-required interfaces remain unimplemented or mismatched in this candidate: `expand_thresholds` long-format examples and weights; `near_zero_edges` / `assign_bins`; `post_denominator_cuts` on normalized D scale; `two_sided_prob` for asymmetric bounds; `assert_fold_integrity`; `scan_feature_names`; `numeric_leak_check`; `crps_from_quantiles`; `precision_recall_at_k`. They require contract-specific golden fixtures and executable tests before promotion.
2. **No test results yet for review branch.** Run `python -m pytest proposals/ -v` on a clean local checkout and attach actual results. Fix errors without mutating golden expected values to conceal defects. Run `python -m compileall -q proposals/`.
3. **No physical holdout isolation evidence.** Human steward must provide an access-denial receipt for every accessible FY2023–2025 outcome replica. `stage_data.py` is still a human-only *proposal*. An allowlist does not certify point-in-time provenance.
4. **No model API and checksum proof.** `tabicl==2.2.0`, raw quantile grid, CatBoost MQ, NGBoost families and pinned checkpoint must be smoke-tested in the approved environment. Source API incompatibility must stop G0.
5. **Candidate parameters unapproved.** G6 noninferiority margins, reconciled development counts, monetary units, source mappings and data availability remain unresolved. Do not choose these automatically.
6. **Reference and golden fixture independence not yet established.** The candidate fixtures and implementation originated from the same agent. Human or independent reviewer must verify all mathematical edge cases, approve the files, then generate the manifest hashes.

## Required independent local verification before review

1. `python -m pytest proposals/ -v`
2. `python -m compileall -q proposals/`
3. `python -m pytest proposals/companion-v1/reference/test_hta_ref.py -v`
4. `python -m pytest proposals/human-staging/test_staging.py -v`
5. Independently inspect the staged feature column allowlist and assert no unapproved physical fields enter test features.
6. After those tests, reconcile exact path/schema/function contracts in `docs/HTA_IMPLEMENTATION_SPEC_v1.2.md` with the accepted candidate files.
7. Only then request separate human promotion and G0 approval.

**Disposition: REQUIRES_FURTHER_IMPLEMENTATION_AND_TESTING.** No raw data, protected outcome, model checkpoint, approval file or G10 unlock was read or created in this review.
