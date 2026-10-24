# Candidate Reference API Contract

**Document:** `proposals/companion-v1/reference/API_CONTRACT.md`  
**Module:** `proposals/companion-v1/reference/hta_ref.py`  
**Governing Specification:** `docs/HTA_IMPLEMENTATION_SPEC_v1.2.md` (§0.1 Rule A3, §15.1, §15.2, §15.3)  
**Status:** PENDING_INDEPENDENT_REVIEW  

---

## 1. Overview

This document specifies the exact public interfaces, signatures, mathematical definitions, input/output types, and boundary behaviors implemented in `hta_ref.py`. Under Rule A3, downstream pipeline code across Gates G1–G12 must import these functions directly and never re-implement them.

---

## 2. Public API Specifications

### 2.1 `build_labels`
```python
def build_labels(
    df: pd.DataFrame,
    pat_pre_col: str = "pat_pre",
    pat_post_col: str = "pat_post",
    total_assets_pre_col: str = "total_assets_pre",
    ks: Sequence[float] = (0.05, 0.075, 0.10, 0.15, 0.20),
    zero_tol: float = 0.0
) -> pd.DataFrame
```
- **Inputs:** Dataframe with pre-audit and post-audit earnings and total assets.
- **Outputs:** Dataframe containing:
  - `delta` ($PAT_{post} - PAT_{pre}$)
  - `D` ($\Delta / Assets_{pre}$)
  - `q` ($|PAT_{pre}| / Assets_{pre}$)
  - `is_zero_adj` ($|\Delta| \le zero\_tol$)
  - `eligible_label` (both PAT present, $PAT_{pre} \neq 0$, $PAT_{pre} \cdot PAT_{post} \ge 0$)
  - `eligible_model` (`eligible_label` and $Assets_{pre} > 0$)
  - `usable_pair`, `pre_zero`, `sign_change`
  - Multi-threshold indicators `y_k050`, `y_k075`, `y_k100`, `y_k150`, `y_k200`
  - Normalized thresholds `c_k050`, `c_k075`, `c_k100`, `c_k150`, `c_k200`
  - `boundary_case` (flagging $| |\Delta|/|PAT_{pre}| - k | < 10^{-9}$)
- **Boundary Handling:**
  - $PAT_{pre} = 0 \implies eligible\_label = False$.
  - $PAT_{pre} \cdot PAT_{post} < 0 \implies eligible\_label = False$.
  - $Assets_{pre} \le 0 \implies eligible\_model = False$.

---

### 2.2 `exceedance_prob`
```python
def exceedance_prob(
    Q: np.ndarray,
    alphas: Sequence[float],
    c: Union[float, np.ndarray],
    tail_policy: str = "conservative"
) -> np.ndarray
```
- **Mathematical Contract:** Converts predicted quantiles $Q(\alpha)$ to $P(|D| \ge c) = P(D \ge c) + P(D \le -c)$.
- **Inputs:** `Q` of shape $(N, M)$, strictly increasing `alphas` of shape $(M,)$, threshold $c \ge 0$.
- **Outputs:** Array of shape $(N,)$ with probabilities bounded in $[0, 1]$.
- **Tail Policy:**
  - `'conservative'` (Default): If $c > Q_{\max}$, $P(D \ge c) = 0$; if $-c < Q_{\min}$, $P(D \le -c) = 0$.
  - `'optimistic'`: Linear tail extrapolation based on outer quantile slopes.

---

### 2.3 `hurdle_exceedance`
```python
def hurdle_exceedance(
    pi_zero: Union[float, np.ndarray],
    Q_nonzero: np.ndarray,
    alphas: Sequence[float],
    c: Union[float, np.ndarray],
    tail_policy: str = "conservative"
) -> np.ndarray
```
- **Formula:** $P(|D| \ge c) = (1 - \pi_0) \cdot P(|D_{nonzero}| \ge c)$.
- **Validation:** Enforces $\pi_0 \in [0, 1]$.

---

### 2.4 `scipy_exceedance`
```python
def scipy_exceedance(dist, c: Union[float, np.ndarray]) -> np.ndarray
```
- **Formula:** $\text{dist.sf}(c) + \text{dist.cdf}(-c)$.
- **Usage:** Evaluates NGBoost analytic predictive distributions (Normal, Laplace, Student-t).

---

### 2.5 `asymmetric_post_cuts`
```python
def asymmetric_post_cuts(
    pat_pre: Union[float, np.ndarray],
    k: float = 0.05
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]
```
- **Formulas:**
  - $PAT_{pre} > 0$: $\Delta_{\text{lower}} = -\frac{k}{1+k} PAT_{pre}$, $\Delta_{\text{upper}} = +\frac{k}{1-k} PAT_{pre}$.
  - $PAT_{pre} < 0$: $\Delta_{\text{lower}} = -\frac{k}{1-k} |PAT_{pre}|$, $\Delta_{\text{upper}} = +\frac{k}{1+k} |PAT_{pre}|$.

---

### 2.6 `temporal_expanding_folds`
```python
def temporal_expanding_folds(
    df: pd.DataFrame,
    year_col: str = "fiscal_year",
    start_year: int = 2015,
    dev_end_year: int = 2022
) -> List[Tuple[np.ndarray, np.ndarray]]
```
- **Folds Generated:** F1 (Train 2015–2018, Val 2019) through F4 (Train 2015–2021, Val 2022).
- **Invariant:** Strict temporal precedence ($\max(\text{train\_year}) < \text{val\_year}$).

---

### 2.7 `paired_firm_cluster_bootstrap`
```python
def paired_firm_cluster_bootstrap(
    df: pd.DataFrame,
    firm_col: str,
    score_a: Sequence[float],
    score_b: Sequence[float],
    y_true: Sequence[int],
    metric_fn,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05
) -> Dict[str, float]
```
- **Clustering:** Samples unique firms with replacement; preserves all firm-year observations for sampled firms.
- **Returns:** Point estimate, standard error, percentile confidence interval $[CI_{\text{lower}}, CI_{\text{upper}}]$.

---

### 2.8 Diagnostic and Guard Functions
- `near_zero_bins(df, q_col, n_bins=5, eligible_col='eligible_model')`: Quintile binning on development set.
- `evaluate_predictions(y_true, y_prob)`: Average Precision, Brier Score, ECE, LogLoss.
- `check_feature_leakage(feature_pub_dates, prediction_cutoffs)`: Point-in-time publication date verification.
- `target_transform(D, s)` and `inverse_target_transform(Z, s)`: Bijective $asinh / sinh$ transformations.
- `crps_score(y_true, Q, alphas)`: Continuous Ranked Probability Score via trapezoidal pinball loss integration.
