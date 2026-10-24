# Companion Bundle Human Handoff Specification

**Repository:** `quanpt-sfa/eswa-hta`  
**Date:** 2026-10-24  
**Author:** Autonomous Implementation Agent (G0 Remediation)  
**Governance Specification:** `docs/HTA_IMPLEMENTATION_SPEC_v1.2.md` (§0.1 Rule A3, §4.3, §15, §16)  
**Blocker ID:** `OQ-001`  
**Disposition:** READY_FOR_HUMAN (All files audited; precise schemas and handoff contracts defined)

---

## 1. Executive Summary and File Inventory

Under §15.1 of the implementation specification, `HTA_IMPLEMENTATION_SPEC_v1.2.md` is an execution specification, **not a self-contained runnable bundle**. G0 cannot pass without an independently verified, human-approved companion bundle.

An exhaustive search across the local workspace and authorized research directories confirmed that **no companion bundle files currently exist**. 

### Status Classification Table

| # | Artifact | Expected Owner | Status | Exact Destination Path | Required Approval |
|---|---|---|---|---|---|
| 1 | `protocol.yaml` | Human Research Lead | **MISSING** | `config/protocol.yaml` | Human sign-off on seeds, grids, reconciliation targets |
| 2 | `columns.yaml` | Human Lead / Steward | **MISSING** | `config/columns.yaml` | Human approval of physical mapping & availability dates (H1–H3) |
| 3 | `hta_ref.py` | Independent Author / Human | **MISSING** | `reference/hta_ref.py` | Human certification; byte-for-byte copied to `src/hta/ref.py` |
| 4 | `test_hta_ref.py` | Independent Author / Human | **MISSING** | `reference/test_hta_ref.py` | Human certification; runnable via `pytest` |
| 5 | `golden_cases.csv` | Independent Reviewer | **MISSING** | `fixtures/golden_cases.csv` | Human sign-off on input edge cases |
| 6 | `golden_expectations.csv` | Independent Reviewer | **MISSING** | `fixtures/golden_expectations.csv` | Human sign-off on expected values and numeric tolerances |
| 7 | `fixture_manifest.sha256` | Human Lead | **MISSING** | `fixtures/fixture_manifest.sha256` | Checksum manifest covering files 3, 4, 5, 6 |
| 8 | `gate_acceptance_manifest.yaml` | Human Lead | **MISSING** | `contracts/gate_acceptance_manifest.yaml` | Formal contract governing each gate G0–G12 |

*Note: In accordance with Rules A1, A2, and A3, the coding agent is strictly prohibited from inventing reference implementations, fabricating golden expected outputs, or self-approving manifests.*

---

## 2. Mathematical Contracts for Reference Implementation (`reference/hta_ref.py`)

Per Rule A3, the functions in `reference/hta_ref.py` must be imported directly by production code and never re-implemented in `src/hta/`. The supplied file must implement and export the following functions conforming strictly to the mathematical specifications of protocol v1.2:

### 2.1 Label Construction: `build_labels(df, pat_pre_col, pat_post_col, assets_pre_col, ks, zero_tol=0)`
- **Mathematical Specification:**
  - $\Delta = \text{PAT}_{\text{post}} - \text{PAT}_{\text{pre}}$
  - $D = \Delta / \text{total\_assets}_{\text{pre}}$
  - Primary zero indicator: $\text{is\_zero\_adj} = (\Delta == 0)$ when `zero_tol == 0`.
  - Eligibility rules:
    - $\text{eligible\_label} \iff (\text{PAT}_{\text{pre}} \neq 0) \land (\text{PAT}_{\text{post}} \text{ present}) \land (\text{PAT}_{\text{pre}} \cdot \text{PAT}_{\text{post}} \ge 0)$. Note: $\text{PAT}_{\text{post}} = 0$ is eligible; sign crossing ($\text{PAT}_{\text{pre}} \cdot \text{PAT}_{\text{post}} < 0$) is ineligible.
    - $\text{eligible\_model} \iff \text{eligible\_label} \land (\text{total\_assets}_{\text{pre}} > 0)$.
  - Multi-threshold binary indicators:
    - For each $k \in \{0.05, 0.075, 0.10, 0.15, 0.20\}$:
      $$y_k = \mathbf{1}_{\{|\Delta| \ge k \cdot |\text{PAT}_{\text{pre}}|\}}$$
    - Threshold on $D$-scale: $c_k = k \cdot |\text{PAT}_{\text{pre}}| / \text{total\_assets}_{\text{pre}}$.
  - Invariants: Monotonic nesting $y_{0.20} \le y_{0.15} \le y_{0.10} \le y_{0.075} \le y_{0.050}$ for all eligible rows.
  - Boundary case flag: $\text{boundary\_case} \iff \text{eligible\_label} \land \exists k: \left| \frac{|\Delta|}{|\text{PAT}_{\text{pre}}|} - k \right| < 10^{-9}$.

### 2.2 Hurdle Mixture Exceedance: `hurdle_exceedance(pi_zero, Q_nonzero, alphas, c, tail_policy="conservative")`
- **Mathematical Specification:**
  - Let $\pi_0 = P(\text{is\_zero\_adj} = 1)$ be the zero-class probability (from TabICL/CatBoost classifier).
  - Let $Q(\alpha)$ be the predicted quantiles of $D$ conditional on $D \neq 0$.
  - Two-sided exceedance on $D$-scale with threshold $c > 0$:
    $$P(|D| \ge c \mid D \neq 0) = P(D \ge c \mid D \neq 0) + P(D \le -c \mid D \neq 0)$$
  - Hurdle combination:
    $$P(|D| \ge c) = (1 - \pi_0) \cdot P(|D| \ge c \mid D \neq 0)$$
  - Tail policy: Conservative interpolation between quantile grid points; optimistic sensitivity fallback.

### 2.3 Analytic Distribution Exceedance: `scipy_exceedance(dist, c)`
- For NGBoost output distribution objects (Normal, Laplace, TFixedDf):
  $$P(|D| \ge c) = \text{dist.sf}(c) + \text{dist.cdf}(-c)$$
  Checked against analytical tolerances ($\le 10^{-10}$).

### 2.4 Asymmetric Post-Denominator Cuts: `asymmetric_post_cuts(pat_pre, k)`
- When adjusting for post-audit denominator scale:
  - For $\text{PAT}_{\text{pre}} > 0$: lower cut $\Delta \le \frac{-k}{1+k} \text{PAT}_{\text{pre}}$; upper cut $\Delta \ge \frac{k}{1-k} \text{PAT}_{\text{pre}}$.
  - For $\text{PAT}_{\text{pre}} < 0$: signs and cut magnitudes reverse.

### 2.5 Temporal Folds: `temporal_expanding_folds(df, year_col, start_year=2015, dev_end_year=2022)`
- Folds F1–F4 with expanding train window (F1: train 2015–2018, val 2019; ... F4: train 2015–2021, val 2022).
- Strict temporal isolation; validation years strictly post-date training years.

### 2.6 Paired Firm-Cluster Bootstrap: `paired_firm_cluster_bootstrap(df, firm_id_col, score_m1, score_b3a, y_true, n_boot, seed)`
- Resampling unit is **firm**, not firm-year (preserves temporal correlation within firm clusters).
- Computes paired $\Delta \text{AP} = \text{AP}(\text{M1}) - \text{AP}(\text{B3a})$ and paired 95% bootstrap confidence intervals.

### 2.7 Leakage Guards: `audit_feature_leakage(features_df, metadata_df, cutoff_col)`
- Invariant check: Feature publication timestamp $\le$ prediction timestamp.

---

## 3. Synthetic Golden Fixtures Contract (`fixtures/`)

Golden fixtures test calculations independently of model code. The human reviewer must provide two CSV files and a SHA-256 checksum manifest:

### 3.1 `fixtures/golden_cases.csv` Schema
```csv
test_id,pat_pre,pat_post,total_assets_pre,k,dist_family,dist_param1,dist_param2,c,pi_zero
T-L01,100,104,1000,0.05,,,,
T-L02,100,105,1000,0.05,,,,
T-L03,100,95,1000,0.05,,,,
T-L04,0,10,1000,0.05,,,,
T-L05,100,-100,1000,0.05,,,,
T-L06,100,0,1000,0.05,,,,
T-Z01,100,100,1000,0.05,,,,
T-Z02,100,100.0000000001,1000,0.05,,,,
T-Z03,1e-9,1.1e-9,1000,0.05,,,,
T-C01,,,,,normal,0,1,1.0,0.0
T-H01,,,,,normal,0,1,1.0,0.4
T-P01,100,,1000,0.05,,,,
T-P02,-100,,1000,0.05,,,,
```

### 3.2 `fixtures/golden_expectations.csv` Schema
```csv
test_id,eligible_label,eligible_model,is_zero_adj,y_k,expected_prob,lower_delta_cut,upper_delta_cut,tolerance
T-L01,True,True,0,0,,,0.0
T-L02,True,True,0,1,,,0.0
T-L03,True,True,0,1,,,0.0
T-L04,False,False,,,,0.0
T-L05,False,False,,,,0.0
T-L06,True,True,0,1,,,0.0
T-Z01,True,True,1,0,,,0.0
T-Z02,True,True,0,0,,,0.0
T-Z03,True,True,0,1,,,0.0
T-C01,,,,0.31731050786291415,,,1e-10
T-H01,,,,0.19038630471774849,,,1e-10
T-P01,,,,,-4.76190476190476,5.26315789473684,1e-8
T-P02,,,,,-5.26315789473684,4.76190476190476,1e-8
```

### 3.3 `fixtures/fixture_manifest.sha256`
Must contain verified SHA-256 hashes of:
- `reference/hta_ref.py`
- `reference/test_hta_ref.py`
- `fixtures/golden_cases.csv`
- `fixtures/golden_expectations.csv`

---

## 4. Configuration Template Contracts (`config/`)

### 4.1 `config/protocol.yaml` Required Sections
- `project`: name, version (`1.2`), primary_seed (`42`), bootstrap_seeds, bootstrap_reps.
- `sample`: dev_years (`[2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]`), test_years (`[2023, 2024, 2025]`).
- `outcome`:
  - `primary_k`: `0.05`
  - `thresholds`: `[0.05, 0.075, 0.10, 0.15, 0.20]`
  - `zero_tol`: `0.0` (frozen before G1)
- `models`: CatBoost grid configurations, NGBoost candidate distributions, TabICL parameters.
- `pilot_gate_G6`: acceptance margins, minimum support rules, NO-GO thresholds.
- `reconciliation_targets`: expected row counts and positive counts by development year.

### 4.2 `config/columns.yaml` Structure (Pre-G1 Proposal / Approval)
- `keys`: `firm_id`, `fiscal_year`, `scope`, `ticker`, `exchange`, `industry`.
- `outcome_inputs`: `pat_pre`, `pat_post`, `total_assets_pre`.
- `currency_and_unit`: monetary scale factor.
- `availability`: explicit publication date columns or as-of availability flags for point-in-time enforcement.
- `approved_by_human`: boolean sign-off flag.

---

## 5. Required Human Actions for OQ-001 Clearance

To clear Blocker OQ-001, the human research lead must:
1. Deposit `reference/hta_ref.py` and `reference/test_hta_ref.py` conforming to Section 2 into `reference/`.
2. Deposit `fixtures/golden_cases.csv` and `fixtures/golden_expectations.csv` into `fixtures/`.
3. Generate and commit `fixtures/fixture_manifest.sha256`.
4. Deposit `config/protocol.yaml` and `contracts/gate_acceptance_manifest.yaml`.
5. Once these files are placed, the agent will execute the byte-for-byte copy to `src/hta/ref.py`, verify checksums against `fixture_manifest.sha256`, run `pytest -q reference/`, and proceed with G0 verification.
