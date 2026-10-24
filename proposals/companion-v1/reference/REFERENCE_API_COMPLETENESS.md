# Reference API Completeness and Reconciliation Matrix

**Document Version:** 1.1  
**Target Specification:** `docs/HTA_IMPLEMENTATION_SPEC_v1.2.md` (§15, §16, Rules A2, A3)  
**Candidate Implementation:** `proposals/companion-v1/reference/hta_ref.py`  
**Candidate Test Suite:** `proposals/companion-v1/reference/test_hta_ref.py`  
**Status:** `CANDIDATE_COMPLETE_PENDING_INDEPENDENT_REVIEW`

---

## 1. Authoritative API Reconciliation Matrix

| Specification Interface | Implementation Function / Constant | Candidate Status | Calling Signature / Contract | Missing/Contradictory Behavior Handled |
| :--- | :--- | :---: | :--- | :--- |
| `ref.build_labels` (§2 D01-D07, §5 G1b) | `build_labels` | **PASS (Candidate)** | `build_labels(df, pat_pre_col='pat_pre', pat_post_col='pat_post', total_assets_pre_col='total_assets_pre', ks=(0.05, 0.075, 0.10, 0.15, 0.20), zero_tol=0.0)` | Accepts both column names and Series/arrays; outputs canonical columns (`delta`, `D`, `q`, `is_zero_adj`, `eligible_label`, `eligible_model`, `usable_pair`, `sign_change`, `pre_zero`, `y_k***`, `c_k***`, `boundary_case`). |
| `ref.expand_thresholds` (§2 D06, §6.2, §8.3 B3b) | `expand_thresholds` | **PASS (Candidate)** | `expand_thresholds(df, ks=(0.05, 0.10, 0.20), pat_pre_col='pat_pre', total_assets_pre_col='total_assets_pre', delta_col='delta', y_col_prefix='y_k', weight_col=None, id_col=None)` | Produces long-format training data for B3b; replicates observations across $k \in \{0.05, 0.10, 0.20\}$; scales sample weight by $1 / K$; preserves firm-year fold assignment; records `source_row_id`. |
| `ref.exceedance_prob` (§2 D13-D15, §15.3) | `exceedance_prob` | **PASS (Candidate)** | `exceedance_prob(Q, alphas, c, tail_policy='conservative')` | Converts multi-quantile predictions into $P(\|D\| \ge c) = P(D \ge c) + P(D \le -c)$; enforces non-crossing; supports 'conservative' and 'optimistic' tail policies. |
| `ref.hurdle_exceedance` (§2 D17) | `hurdle_exceedance` | **PASS (Candidate)** | `hurdle_exceedance(pi_zero, Q_nonzero, alphas, c, tail_policy='conservative')` | Multiplies $(1 - \pi_0) \times P(\|D_{nonzero}\| \ge c)$; checks $\pi_0 \in [0, 1]$. |
| `ref.scipy_exceedance` (§2 D16) | `scipy_exceedance` | **PASS (Candidate)** | `scipy_exceedance(dist, c)` | Exact analytic exceedance $dist.sf(c) + dist.cdf(-c)$ for continuous distributions. |
| `ref.post_denominator_cuts` (§15.2 T-P01, T-P02, §9.3 R1) | `post_denominator_cuts` (alias `asymmetric_post_cuts`) | **PASS (Candidate)** | `post_denominator_cuts(pat_pre, k=0.05, total_assets_pre=None)` | Independent algebraic derivation for post-audit denominator cuts; handles positive and negative $PAT_{pre}$; supports asset-normalization on $D$ scale if `total_assets_pre` provided. |
| `ref.two_sided_prob` (§15.2 T-C01, §9.3 R1) | `two_sided_prob` (alias `two_sided_exceedance`) | **PASS (Candidate)** | `two_sided_prob(dist_or_cdf, c=None, lower_cutoff=None, upper_cutoff=None)` | Computes symmetric $P(\|X\| \ge c)$ or asymmetric $P(X \le lower\_cutoff) + P(X \ge upper\_cutoff)$. |
| `ref.near_zero_edges` (§2 D22, §5 G4) | `near_zero_edges` | **PASS (Candidate)** | `near_zero_edges(q, n_bins=5)` | Fits quantile edges strictly on finite development $eligible\_model$ rows; fails closed with explicit ValueError on tied edges; endpoints $[-\infty, \dots, +\infty]$. |
| `ref.assign_bins` (§2 D22, §5 G4) | `assign_bins` | **PASS (Candidate)** | `assign_bins(q, edges)` | Applies frozen precomputed quantile edges to validation/test rows without refitting. |
| `ref.near_zero_bins` (legacy wrapper) | `near_zero_bins` | **PASS (Candidate)** | `near_zero_bins(df, q_col='q', n_bins=5, eligible_col='eligible_model')` | Convenience wrapper combining `near_zero_edges` and `assign_bins`. |
| `ref.assert_fold_integrity` (§2 D10, §5 G4) | `assert_fold_integrity` | **PASS (Candidate)** | `assert_fold_integrity(folds, df, year_col='fiscal_year', firm_col='firm_id')` | Validates expanding-window folds: non-empty sets, no train/val index overlap, strict temporal ordering ($\max(train) < \min(val)$), strictly increasing validation years, no firm-year cross-contamination. |
| `ref.scan_feature_names` (§2 D08, §5 G2) | `scan_feature_names` | **PASS (Candidate)** | `scan_feature_names(feature_names, allow=None)` | Detects forbidden post-audit tokens (`aud`, `post`, `delta`, `audit_adj`, etc.); permits explicit allowlist and verified lagged items (`_lag[1-9]`). Returns `[]` when clean. |
| `ref.numeric_leak_check` (§5 G2) | `numeric_leak_check` | **PASS (Candidate)** | `numeric_leak_check(X_numeric, pat_pre, pat_post, delta, y_k050, corr_threshold=0.99)` | Diagnostics for extreme correlation ($|r| \ge 0.99$) against audit targets; outputs diagnostic dataframe without silently dropping features. |
| `ref.crps_from_quantiles` (§14 Rule 8, §15.2) | `crps_from_quantiles` (alias `crps_score`) | **PASS (Candidate)** | `crps_from_quantiles(y_true, quantiles, alphas)` | Trapezoidal numerical integration of quantile pinball loss; supports non-uniform alpha grids; evaluated on original outcome scale. |
| `ref.precision_recall_at_k` (§2 D21, §9.1) | `precision_recall_at_k` | **PASS (Candidate)** | `precision_recall_at_k(y_true, y_score, k_budget)` | Calculates Precision@K and Recall@K at screening budget $m = \lceil k\_budget \cdot n \rceil$; handles score ties stably. |
| `ref.ap` (§2 D20, §9.1) | `ap` (alias `compute_average_precision`) | **PASS (Candidate)** | `ap(y_true, y_score)` | Average Precision identical to `sklearn.metrics.average_precision_score`; score-tie invariant to input row order. |
| `ref.paired_cluster_bootstrap` (§2 D20, §15.2 T-B01) | `paired_cluster_bootstrap` (alias `paired_firm_cluster_bootstrap`) | **PASS (Candidate)** | `paired_cluster_bootstrap(df, firm_col, score_a, score_b, y_true, metric=None, metric_fn=None, n_boot=1000, seed=42, alpha=0.05)` | Resamples permanent firm IDs with replacement; retains all firm-year observations per selected cluster; evaluates paired metric differences; handles single-class resamples; raises error if $< 2$ valid replications. |
| `ref.check_feature_leakage` (§2 D08-D09, §15.2 T-T01) | `check_feature_leakage` | **PASS (Candidate)** | `check_feature_leakage(feature_pub_dates, prediction_cutoffs)` | Fails closed on missing timestamps; flags feature date $>$ prediction cutoff as True (leakage). |
| `ref.check_point_in_time` (§2 D08-D09) | `check_point_in_time` | **PASS (Candidate)** | `check_point_in_time(df, date_cols, cutoff_col)` | Batch point-in-time leakage check across multiple date columns. |
| `ref.target_transform` / `inverse_target_transform` (§2 D18) | `target_transform`, `inverse_target_transform` | **PASS (Candidate)** | `target_transform(D, s=1.0)`, `inverse_target_transform(Z, s=1.0)` | Bijective signed arcsinh / sinh transformation for optional sensitivity analysis. |
| `ref.TABICL_RAW_ALPHAS` (§2 D13, §15.3) | `TABICL_RAW_ALPHAS` | **UNRESOLVED (Blocked)** | `TABICL_RAW_ALPHAS = None` | Pinned TabICLv2 version and checkpoint alpha levels must be verified via approved G0 executable preflight before resolving to a fixed float array. Fails closed. |

---

## 2. Summary Status

- **Total Specification Interfaces Evaluated:** 21
- **Candidate Implementations Complete & Testable:** 20 (95.2%)
- **Unresolved / Externally Blocked:** 1 (`TABICL_RAW_ALPHAS`, pending human dependency installation and checkpoint verification)
- **Failed Interfaces:** 0 (0.0%)

*Note: All interfaces are candidate proposals located under `proposals/companion-v1/reference/` and require independent human review and sign-off before being promoted to `reference/`.*
