# Protocol Amendment Proposal: Fold-Safe Near-Zero Binning

**Document ID:** AMEND-2026-001  
**Target Specification:** `docs/HTA_IMPLEMENTATION_SPEC_v1.2.md`  
**Target Sections:** §2 (D22), §6.2, §6.3, §8.1 (G4 Splits and Preprocessing), §8.2–8.3 (G5 Modeling and Evaluation)  
**Status:** PROPOSED_AMENDMENT (Pending Human Review and Sign-off)  
**Governance Invariant:** Rule A10 (Specification Clarification and Leakage Prevention)

---

## 1. Executive Summary & Motivation

In the HTA research protocol, the $X_2$ predictor set augments $X_1$ with unaudited pre-audit earnings intensity $q = \frac{|PAT_{pre}|}{Assets_{pre}}$, the primary threshold scale $c_{k050} = 0.05 \cdot q$, and near-zero earnings bin indicators $nz\_bin$.

The current text of §6.2 and §8.1 (G4) specifies computing quantile bin edges from all development observations 2015–2022 prior to cross-validation:
> *"Compute `edges = ref.near_zero_edges(dev.q[dev.eligible_model])`; add `nz_bin` to X2 for all rows; save `outputs/G4/near_zero_edges.json`."*

However, the experimental protocol (§8.1, §8.2, §15.1) evaluates model selection and out-of-fold calibration using expanding-window temporal folds:
- **Fold F1:** Training 2015–2018 $\rightarrow$ Validation 2019
- **Fold F2:** Training 2015–2019 $\rightarrow$ Validation 2020
- **Fold F3:** Training 2015–2020 $\rightarrow$ Validation 2021
- **Fold F4:** Training 2015–2021 $\rightarrow$ Validation 2022

If global development bin edges are computed across 2015–2022 before cross-validation, information from the validation years (e.g., FY2019 distribution for F1, FY2020 for F2, etc.) directly influences the feature definition and quantile boundaries of training observations. This constitutes **temporal distribution leakage** during cross-validation, directly contradicting §6.3:
> *"Nothing is fitted on validation or test rows."*

This proposed amendment resolves the contradiction by standardizing **fold-specific fitting** for expanding development folds F1–F4, while preserving **full-development fitting** (2015–2022) strictly for the frozen final models evaluated on FY2023–2025 holdout test data.

---

## 2. Specific Clause Amendments

### 2.1 Clause §2 D22 (Near-Zero Bins)

#### Before:
> **D22 — Near-zero earnings bins:**
> Divide development observations of $q$ into quintiles using `ref.near_zero_edges(dev.q[dev.eligible_model], n_bins=5)` to produce 5 bins. Assign bin indices to all rows via `ref.assign_bins(q, edges)`. Frozen edges are stored in `outputs/G4/near_zero_edges.json`.

#### After:
> **D22 — Fold-isolated near-zero earnings bins:**
> 1. **Cross-validation folds (F1–F4):** For each expanding development fold $f \in \{1, 2, 3, 4\}$, bin edges $edges_f$ are computed exclusively from the eligible training fold observations:
>    $$\text{edges}_f = \text{ref.near\_zero\_edges}(q_{\text{train}, f}[\text{eligible\_model}], n\_bins=5)$$
>    Bin indices for both training and validation folds are assigned using frozen $edges_f$:
>    $$\text{bins}_{\text{train}, f} = \text{ref.assign\_bins}(q_{\text{train}, f}, \text{edges}_f), \quad \text{bins}_{\text{val}, f} = \text{ref.assign\_bins}(q_{\text{val}, f}, \text{edges}_f)$$
>    Validation rows are strictly excluded when fitting $edges_f$.
> 2. **Final frozen model (for FY2023–2025 inference):** After cross-validation model selection is frozen, final bin edges $edges_{\text{final}}$ are fitted on all authorized development observations 2015–2022:
>    $$\text{edges}_{\text{final}} = \text{ref.near\_zero\_edges}(q_{\text{dev\_2015\_2022}}[\text{eligible\_model}], n\_bins=5)$$
>    These final edges are saved to `outputs/G4/near_zero_edges_final.json` and applied to test features FY2023–2025 without re-estimation.

---

### 2.2 Clause §6.2 (Feature Set X2 Definition)

#### Before:
> `nz_bin` = `ref.assign_bins(q, edges)` with `edges = ref.near_zero_edges(q[dev eligible_model])` **computed in G4** from development rows only. G2 writes X2 without `nz_bin`; G4 adds it to development and test features using frozen edges. Verify `q` and threshold values use unaudited PAT and assets only.

#### After:
> `nz_bin` = `ref.assign_bins(q, edges)`:
> - In development cross-validation (G5, G7, G9), `nz_bin` is generated per-fold via `ref.fit_fold_near_zero_bins(train_df, val_df)` where edges are fitted strictly on fold-training eligible rows.
> - In final-model deployment (G10), `nz_bin` is generated via `ref.fit_final_near_zero_bins(dev_df, test_df)` where edges are fitted on full authorized development rows (2015–2022) and applied to FY2023–2025 candidate test features.
> - Test-period distributions (FY2023–2025) must never be used to fit, alter, or select bin edges.

---

### 2.3 Clause §8.1 G4 (Splits, Preprocessing, and Lockout)

#### Before:
> 2. Compute `edges = ref.near_zero_edges(dev.q[dev.eligible_model])`; add `nz_bin` to X2 for all rows; save `outputs/G4/near_zero_edges.json`.

#### After:
> 2. Generate fold-specific near-zero bin edges for development folds F1–F4 using `ref.fit_fold_near_zero_bins` and save to `outputs/G4/near_zero_edges_folds.json`. In addition, compute full-development bin edges (2015–2022) for final model training using `ref.near_zero_edges` and save to `outputs/G4/near_zero_edges_final.json`. Both artifacts are cryptographically hashed and verified before baseline model fitting.

---

### 2.4 Clause §8.1 Acceptance Criteria

#### Before:
> `nz_bin` edges recompute from development rows only.

#### After:
> Fold-specific `nz_bin` edges recompute from each fold's training observations only; validation observations must have zero mathematical influence on fold edges. Full-development bin edges recompute from authorized FY2015–2022 development rows only; candidate test features FY2023–2025 must have zero mathematical influence on final bin edges.

---

## 3. Mathematical & Empirical Invariants

1. **Invariance to Future Distributions in Cross-Validation:**  
   Altering $q$ values in FY2019 (the validation year of Fold F1) or FY2020–2022 produces **zero change** in the bin edges $edges_{F1}$.
2. **Invariance to Holdout Test Distributions:**  
   Altering $q$ values in FY2023–2025 produces **zero change** in the final model bin edges $edges_{\text{final}}$.
3. **Discrete Edge Policy on Ties:**  
   If tied quantiles occur in any fold or the final development dataset, the algorithm raises `ValueError` requiring registered discrete fallback rather than silently altering bin counts.

---

## 4. Implementation Status in Candidate Bundle

The candidate reference implementation in `proposals/companion-v1/reference/hta_ref.py` already includes the necessary fold-isolated functions:
- `fit_fold_near_zero_bins(train_df, val_df, q_col="q", n_bins=5, eligible_col="eligible_model")`
- `fit_final_near_zero_bins(dev_df, test_df, q_col="q", n_bins=5, eligible_col="eligible_model", dev_max_year=2022)`

Verified by automated regression tests in:
`proposals/companion-v1/reference/test_hta_ref.py::test_near_zero_temporal_fold_isolation_and_final_edges` (PASSED).

---

## 5. Human Sign-Off Requirement

This amendment proposal does not alter registered research hypotheses, model IDs, evaluation metrics, or sample periods. Human approval of this document formally authorizes promoting the candidate fold-isolated implementation to production `reference/hta_ref.py` at Gate G0 closure.
