"""
HTA Reference Implementation (Candidate v1.0)
=============================================
Repository: quanpt-sfa/eswa-hta
Governance: docs/HTA_IMPLEMENTATION_SPEC_v1.2.md (§15.1, Rule A3)
Status: PENDING_INDEPENDENT_REVIEW (Candidate Proposal for Companion Bundle)

This module provides candidate mathematical implementations for the HTA-TabICL
research pipeline, including:
  1. Label construction and materiality thresholds (build_labels)
  2. Multi-quantile probability exceedance (exceedance_prob)
  3. Hurdle mixture exceedance (hurdle_exceedance)
  4. Analytic distribution exceedance (scipy_exceedance)
  5. Post-audit denominator asymmetric cuts (asymmetric_post_cuts)
  6. Two-sided exceedance probability (two_sided_exceedance)
  7. Multi-threshold expansion (threshold_expansion)
  8. Expanding-window temporal folds (temporal_expanding_folds)
  9. Paired firm-cluster bootstrap (paired_firm_cluster_bootstrap)
 10. Near-zero earnings binning (near_zero_bins)
 11. Distributional and classification metrics (evaluate_predictions)
 12. Feature leakage and point-in-time guards (check_feature_leakage, check_point_in_time)
 13. Target transformations (target_transform, inverse_target_transform)
 14. Distributional loss / CRPS (crps_score)

Under Rule A3, this candidate implementation must be independently audited,
verified against golden fixtures, and signed off before being promoted to
the authoritative production location `reference/hta_ref.py`.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


# ==============================================================================
# 1. Label Construction and Materiality Invariants (§2 D01-D07, §5 G1b)
# ==============================================================================

def build_labels(
    df: pd.DataFrame,
    pat_pre_col: str = "pat_pre",
    pat_post_col: str = "pat_post",
    total_assets_pre_col: str = "total_assets_pre",
    ks: Sequence[float] = (0.05, 0.075, 0.10, 0.15, 0.20),
    zero_tol: float = 0.0
) -> pd.DataFrame:
    """
    Constructs audit adjustment outcomes, eligibility indicators, and multi-threshold
    labels according to §2 (D01-D07) and §5 (G1b).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing pre-audit, post-audit PAT, and pre-audit assets.
    pat_pre_col : str
        Column name for unaudited pre-audit profit after tax (PAT_pre).
    pat_post_col : str
        Column name for audited post-audit profit after tax (PAT_post).
    total_assets_pre_col : str
        Column name for pre-audit total assets.
    ks : Sequence[float]
        Registered materiality thresholds (default: 0.05, 0.075, 0.10, 0.15, 0.20).
    zero_tol : float
        Zero adjustment tolerance. Default is 0.0 (frozen before G1 per D07).

    Returns
    -------
    pd.DataFrame
        A dataframe with canonical outcome and label columns:
        - delta: PAT_post - PAT_pre
        - D: delta / total_assets_pre
        - is_zero_adj: (|delta| <= zero_tol)
        - eligible_label: True if both PAT present, PAT_pre != 0, and PAT_pre * PAT_post >= 0
        - eligible_model: eligible_label and total_assets_pre > 0
        - usable_pair: both PAT present
        - sign_change: PAT_pre * PAT_post < 0
        - pre_zero: PAT_pre == 0
        - y_k***: binary exceedance indicator (|delta| >= k * |PAT_pre|) for each k in ks
        - c_k***: normalized threshold (k * |PAT_pre| / total_assets_pre)
        - boundary_case: True if eligible_label and ||delta|/|PAT_pre| - k| < 1e-9 for any k
    """
    out = pd.DataFrame(index=df.index)
    pat_pre = df[pat_pre_col].astype(float)
    pat_post = df[pat_post_col].astype(float)
    assets = df[total_assets_pre_col].astype(float)

    # Basic adjustments
    delta = pat_post - pat_pre
    out["delta"] = delta
    
    # Scale D by total assets (avoid division by non-positive or zero assets)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["D"] = np.where(assets > 0, delta / assets, np.nan)
        out["q"] = np.where(assets > 0, np.abs(pat_pre) / assets, np.nan)

    # Zero adjustment indicator
    out["is_zero_adj"] = np.abs(delta) <= zero_tol

    # Special case audits
    out["usable_pair"] = pat_pre.notna() & pat_post.notna()
    out["pre_zero"] = out["usable_pair"] & (pat_pre == 0.0)
    out["sign_change"] = out["usable_pair"] & (pat_pre * pat_post < 0.0)

    # Primary eligibility
    eligible_label = (
        out["usable_pair"]
        & (~out["pre_zero"])
        & (~out["sign_change"])
    )
    out["eligible_label"] = eligible_label
    out["eligible_model"] = eligible_label & (assets > 0.0)

    # Threshold labels and thresholds on D scale
    boundary_flags = np.zeros(len(df), dtype=bool)

    abs_delta = np.abs(delta)
    abs_pat_pre = np.abs(pat_pre)

    for k in ks:
        tag = f"{int(round(k * 1000)):03d}"
        col_y = f"y_k{tag}"
        col_c = f"c_k{tag}"

        threshold_val = k * abs_pat_pre
        
        # Label is 1 if |delta| >= k * |PAT_pre|, else 0 on eligible rows; NaN otherwise
        y_val = np.where(eligible_label, (abs_delta >= threshold_val).astype(int), np.nan)
        out[col_y] = y_val

        # Normalized threshold c_k = k * |PAT_pre| / assets
        with np.errstate(divide="ignore", invalid="ignore"):
            c_val = np.where(assets > 0, threshold_val / assets, np.nan)
        out[col_c] = c_val

        # Check boundary proximity: ||delta|/|PAT_pre| - k| < 1e-9
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_ratio = np.where(abs_pat_pre > 0, abs_delta / abs_pat_pre, np.nan)
            near_boundary = np.abs(rel_ratio - k) < 1e-9
            boundary_flags = boundary_flags | (eligible_label & np.nan_to_num(near_boundary, nan=False))

    out["boundary_case"] = boundary_flags
    return out


# ==============================================================================
# 2. Probability Exceedance from Quantile Predictions (§2 D13-D15, §15.3)
# ==============================================================================

def exceedance_prob(
    Q: np.ndarray,
    alphas: Sequence[float],
    c: Union[float, np.ndarray],
    tail_policy: str = "conservative"
) -> np.ndarray:
    """
    Converts multi-quantile predictions Q(alpha) into two-sided exceedance
    probabilities P(|D| >= c) = P(D >= c) + P(D <= -c).

    Parameters
    ----------
    Q : np.ndarray
        Predicted quantiles of shape (N, M) or (M,), where M = len(alphas).
    alphas : Sequence[float]
        Quantile levels strictly increasing in (0, 1).
    c : Union[float, np.ndarray]
        Positive threshold on D-scale. Scalar or array of shape (N,).
    tail_policy : str
        Policy for thresholds exceeding the quantile grid:
        - 'conservative': P(D >= c) = 0 if c > Q_max; P(D <= -c) = 0 if -c < Q_min.
        - 'optimistic': Linear tail extrapolation using slope of outer quantile pairs.

    Returns
    -------
    np.ndarray
        Array of exceedance probabilities in [0, 1] of shape (N,).
    """
    Q_arr = np.asarray(Q, dtype=float)
    alphas_arr = np.asarray(alphas, dtype=float)

    if Q_arr.ndim == 1:
        Q_arr = Q_arr.reshape(1, -1)
    
    N, M = Q_arr.shape
    if M != len(alphas_arr):
        raise ValueError(f"Quantile count mismatch: Q has {M} cols, alphas has {len(alphas_arr)}")
    
    if np.any(np.diff(alphas_arr) <= 0):
        raise ValueError("alphas must be strictly increasing")

    c_arr = np.broadcast_to(np.asarray(c, dtype=float), (N,))
    if np.any(c_arr < 0):
        raise ValueError("Threshold c must be non-negative")

    probs = np.zeros(N, dtype=float)

    for i in range(N):
        q_row = Q_arr[i]
        c_val = c_arr[i]

        # Enforce non-crossing sorting on row
        if np.any(np.diff(q_row) < 0):
            q_row = np.sort(q_row)

        q_min = q_row[0]
        q_max = q_row[-1]
        a_min = alphas_arr[0]
        a_max = alphas_arr[-1]

        # 1. Lower tail: P(D <= -c) = F(-c)
        target_lower = -c_val
        if target_lower < q_min:
            if tail_policy == "conservative":
                p_lower = 0.0
            else:
                # Linear extrapolation
                slope = (q_row[1] - q_row[0]) / (alphas_arr[1] - alphas_arr[0])
                p_lower = max(0.0, a_min + (target_lower - q_min) / max(slope, 1e-12))
        elif target_lower >= q_max:
            p_lower = a_max
        else:
            p_lower = float(np.interp(target_lower, q_row, alphas_arr))

        # 2. Upper tail: P(D >= c) = 1 - F(c)
        target_upper = c_val
        if target_upper > q_max:
            if tail_policy == "conservative":
                p_upper = 0.0
            else:
                slope = (q_row[-1] - q_row[-2]) / (alphas_arr[-1] - alphas_arr[-2])
                f_upper = min(1.0, a_max + (target_upper - q_max) / max(slope, 1e-12))
                p_upper = 1.0 - f_upper
        elif target_upper <= q_min:
            p_upper = 1.0 - a_min
        else:
            f_upper = float(np.interp(target_upper, q_row, alphas_arr))
            p_upper = 1.0 - f_upper

        total_p = np.clip(p_lower + p_upper, 0.0, 1.0)
        probs[i] = total_p

    return probs


# ==============================================================================
# 3. Hurdle Mixture Exceedance (§2 D17)
# ==============================================================================

def hurdle_exceedance(
    pi_zero: Union[float, np.ndarray],
    Q_nonzero: np.ndarray,
    alphas: Sequence[float],
    c: Union[float, np.ndarray],
    tail_policy: str = "conservative"
) -> np.ndarray:
    """
    Computes hurdle exceedance probability combining zero probability pi_zero
    and conditional non-zero exceedance:
      P(|D| >= c) = (1 - pi_zero) * P(|D_nonzero| >= c)

    Parameters
    ----------
    pi_zero : Union[float, np.ndarray]
        Probability of zero adjustment P(is_zero_adj == 1).
    Q_nonzero : np.ndarray
        Quantile predictions for non-zero adjustments.
    alphas : Sequence[float]
        Quantile levels.
    c : Union[float, np.ndarray]
        Exceedance threshold.
    tail_policy : str
        Tail interpolation policy ('conservative' or 'optimistic').

    Returns
    -------
    np.ndarray
        Hurdle exceedance probabilities.
    """
    cond_prob = exceedance_prob(Q_nonzero, alphas, c, tail_policy=tail_policy)
    pi_zero_arr = np.broadcast_to(np.asarray(pi_zero, dtype=float), cond_prob.shape)
    
    if np.any((pi_zero_arr < 0.0) | (pi_zero_arr > 1.0)):
        raise ValueError("pi_zero must be in [0, 1]")

    return (1.0 - pi_zero_arr) * cond_prob


# ==============================================================================
# 4. Analytic Distribution Exceedance (NGBoost) (§2 D16)
# ==============================================================================

def scipy_exceedance(dist, c: Union[float, np.ndarray]) -> np.ndarray:
    """
    Computes exact analytic two-sided exceedance probability for Scipy/NGBoost
    continuous distributions:
      P(|D| >= c) = dist.sf(c) + dist.cdf(-c)
    """
    c_arr = np.asarray(c, dtype=float)
    if np.any(c_arr < 0):
        raise ValueError("Threshold c must be non-negative")
    
    sf_val = np.asarray(dist.sf(c_arr), dtype=float)
    cdf_val = np.asarray(dist.cdf(-c_arr), dtype=float)
    return np.clip(sf_val + cdf_val, 0.0, 1.0)


# ==============================================================================
# 5. Asymmetric Post-Denominator Cuts (§15.2 T-P01, T-P02)
# ==============================================================================

def asymmetric_post_cuts(
    pat_pre: Union[float, np.ndarray],
    k: float = 0.05
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Computes asymmetric adjustment cuts on Delta scale when the materiality
    threshold is defined relative to post-audit earnings |Delta / PAT_post| >= k.

    Derivation:
      Case 1: PAT_pre > 0
        lower_cut = -k / (1 + k) * PAT_pre
        upper_cut = +k / (1 - k) * PAT_pre
      Case 2: PAT_pre < 0
        lower_cut = -k / (1 - k) * |PAT_pre|
        upper_cut = +k / (1 + k) * |PAT_pre|

    Parameters
    ----------
    pat_pre : Union[float, np.ndarray]
        Pre-audit earnings.
    k : float
        Materiality threshold fraction (0 < k < 1).

    Returns
    -------
    Tuple[lower_cut, upper_cut]
        The lower and upper Delta cut values.
    """
    if not (0.0 < k < 1.0):
        raise ValueError(f"k must be in (0, 1), got {k}")

    pat_arr = np.asarray(pat_pre, dtype=float)
    is_scalar = (pat_arr.ndim == 0)

    lower = np.zeros_like(pat_arr)
    upper = np.zeros_like(pat_arr)

    pos_mask = pat_arr > 0
    neg_mask = pat_arr < 0
    zero_mask = pat_arr == 0

    # Positive earnings
    lower[pos_mask] = -k / (1.0 + k) * pat_arr[pos_mask]
    upper[pos_mask] = +k / (1.0 - k) * pat_arr[pos_mask]

    # Negative earnings (magnitudes reversed)
    abs_neg = np.abs(pat_arr[neg_mask])
    lower[neg_mask] = -k / (1.0 - k) * abs_neg
    upper[neg_mask] = +k / (1.0 + k) * abs_neg

    # Zero pre-audit earnings
    lower[zero_mask] = np.nan
    upper[zero_mask] = np.nan

    if is_scalar:
        return float(lower), float(upper)
    return lower, upper


# ==============================================================================
# 6. Two-Sided Exceedance Function
# ==============================================================================

def two_sided_exceedance(cdf_func, c: float) -> float:
    """
    Computes P(|X| >= c) given an arbitrary continuous CDF function F(x):
      P(|X| >= c) = F(-c) + (1 - F(c))
    """
    if c < 0:
        raise ValueError("c must be non-negative")
    return float(cdf_func(-c) + (1.0 - cdf_func(c)))


# ==============================================================================
# 7. Multi-Threshold Expansion (§2 D06)
# ==============================================================================

def threshold_expansion(
    df: pd.DataFrame,
    pat_pre_col: str = "pat_pre",
    total_assets_pre_col: str = "total_assets_pre",
    ks: Sequence[float] = (0.05, 0.075, 0.10, 0.15, 0.20)
) -> pd.DataFrame:
    """
    Computes the D-scale thresholds c_k = k * |PAT_pre| / total_assets_pre
    for each observation and each registered k.
    """
    res = pd.DataFrame(index=df.index)
    pat_abs = df[pat_pre_col].abs().astype(float)
    assets = df[total_assets_pre_col].astype(float)

    for k in ks:
        tag = f"{int(round(k * 1000)):03d}"
        with np.errstate(divide="ignore", invalid="ignore"):
            res[f"c_k{tag}"] = np.where(assets > 0, (k * pat_abs) / assets, np.nan)
    return res


# ==============================================================================
# 8. Expanding-Window Temporal Folds (§2 D10, §5 G4)
# ==============================================================================

def temporal_expanding_folds(
    df: pd.DataFrame,
    year_col: str = "fiscal_year",
    start_year: int = 2015,
    dev_end_year: int = 2022
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Constructs folds F1-F4 using expanding window training:
      - Fold F1: Train 2015-2018, Val 2019
      - Fold F2: Train 2015-2019, Val 2020
      - Fold F3: Train 2015-2020, Val 2021
      - Fold F4: Train 2015-2021, Val 2022

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        List of (train_indices, val_indices).
    """
    years = df[year_col].to_numpy()
    folds = []
    
    val_years = [2019, 2020, 2021, 2022]
    for vy in val_years:
        if vy > dev_end_year:
            break
        train_idx = np.where((years >= start_year) & (years < vy))[0]
        val_idx = np.where(years == vy)[0]
        folds.append((train_idx, val_idx))

    return folds


# ==============================================================================
# 9. Paired Firm-Cluster Bootstrap (§2 D20, §15.2 T-B01)
# ==============================================================================

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
) -> Dict[str, float]:
    """
    Executes paired firm-cluster bootstrap testing between model A and model B.
    Resamples FIRMS with replacement, retaining all firm-year observations for
    selected firms to preserve within-cluster temporal correlation.
    """
    rng = np.random.default_rng(seed)
    
    firm_ids = df[firm_col].to_numpy()
    unique_firms = np.unique(firm_ids)
    n_firms = len(unique_firms)

    # Pre-index row positions by firm
    firm_to_idx = {f: np.where(firm_ids == f)[0] for f in unique_firms}

    y_arr = np.asarray(y_true, dtype=int)
    s_a = np.asarray(score_a, dtype=float)
    s_b = np.asarray(score_b, dtype=float)

    point_a = metric_fn(y_arr, s_a)
    point_b = metric_fn(y_arr, s_b)
    point_diff = point_a - point_b

    diffs = np.zeros(n_boot, dtype=float)

    for b in range(n_boot):
        boot_firms = rng.choice(unique_firms, size=n_firms, replace=True)
        boot_indices = np.concatenate([firm_to_idx[f] for f in boot_firms])

        y_boot = y_arr[boot_indices]
        # Skip if single-class resample
        if len(np.unique(y_boot)) < 2:
            diffs[b] = np.nan
            continue

        ma = metric_fn(y_boot, s_a[boot_indices])
        mb = metric_fn(y_boot, s_b[boot_indices])
        diffs[b] = ma - mb

    valid_diffs = diffs[~np.isnan(diffs)]
    ci_lower = float(np.percentile(valid_diffs, 100 * (alpha / 2)))
    ci_upper = float(np.percentile(valid_diffs, 100 * (1 - alpha / 2)))

    return {
        "point_diff": float(point_diff),
        "point_a": float(point_a),
        "point_b": float(point_b),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "se": float(np.std(valid_diffs, ddof=1)),
        "n_valid_bootstraps": int(len(valid_diffs))
    }


# ==============================================================================
# 10. Near-Zero Earnings Bins (§2 D22)
# ==============================================================================

def near_zero_bins(
    df: pd.DataFrame,
    q_col: str = "q",
    n_bins: int = 5,
    eligible_col: str = "eligible_model"
) -> Tuple[pd.Series, np.ndarray]:
    """
    Computes quintiles of q = |PAT_pre| / assets on development eligible_model
    rows with fixed edges, returning assigned bin IDs [0..n_bins-1] and edges.
    """
    eligible_mask = df[eligible_col].astype(bool)
    dev_q = df.loc[eligible_mask, q_col].dropna()

    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.percentile(dev_q, quantiles * 100)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    bin_ids = pd.cut(df[q_col], bins=bin_edges, labels=False, include_lowest=True)
    return bin_ids, bin_edges


# ==============================================================================
# 11. Prediction Evaluation Metrics (§2 D20, D23)
# ==============================================================================

def compute_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes exact Average Precision (area under PR curve)."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    if len(np.unique(y_true)) < 2:
        return np.nan

    order = np.argsort(-y_score)
    y_ordered = y_true[order]
    
    tp = np.cumsum(y_ordered == 1)
    fp = np.cumsum(y_ordered == 0)
    recalls = tp / np.sum(y_true == 1)
    precisions = tp / (tp + fp)

    # Average precision definition: sum (R_k - R_{k-1}) * P_k
    r_prev = np.concatenate(([0.0], recalls[:-1]))
    ap = float(np.sum((recalls - r_prev) * precisions))
    return ap


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared probability error."""
    return float(np.mean((np.asarray(y_true) - np.asarray(y_prob)) ** 2))


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error with uniform binning."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        bin_mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i+1] if i < n_bins - 1 else y_prob <= bin_edges[i+1])
        bin_size = np.sum(bin_mask)
        if bin_size > 0:
            acc = np.mean(y_true[bin_mask])
            conf = np.mean(y_prob[bin_mask])
            ece += (bin_size / n) * np.abs(acc - conf)

    return float(ece)


def evaluate_predictions(y_true: Sequence[int], y_prob: Sequence[float]) -> Dict[str, float]:
    """Computes full diagnostic evaluation metrics."""
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_prob, dtype=float)

    return {
        "average_precision": compute_average_precision(yt, yp),
        "brier_score": compute_brier_score(yt, yp),
        "ece": compute_ece(yt, yp),
        "log_loss": float(stats.log_loss(yt, np.clip(yp, 1e-15, 1 - 1e-15))) if hasattr(stats, "log_loss") else float(-np.mean(yt * np.log(np.clip(yp, 1e-15, 1)) + (1 - yt) * np.log(np.clip(1 - yp, 1e-15, 1))))
    }


# ==============================================================================
# 12. Feature Leakage & Point-in-Time Guards (§2 D08-D09, §15.2 T-T01)
# ==============================================================================

def check_feature_leakage(
    feature_pub_dates: Sequence[Union[str, pd.Timestamp]],
    prediction_cutoffs: Sequence[Union[str, pd.Timestamp]]
) -> pd.Series:
    """
    Checks that every feature's release date is not later than the prediction cutoff.
    Returns boolean Series where True indicates a LEAKAGE violation.
    """
    f_dates = pd.to_datetime(feature_pub_dates)
    cutoffs = pd.to_datetime(prediction_cutoffs)

    # Violation if feature publication is strictly after prediction timestamp
    leakage_flag = f_dates > cutoffs
    return pd.Series(leakage_flag, name="leakage")


def check_point_in_time(
    df: pd.DataFrame,
    date_cols: Sequence[str],
    cutoff_col: str
) -> pd.DataFrame:
    """Audits multiple date columns against cutoff; returns violation flags."""
    res = pd.DataFrame(index=df.index)
    cutoff = pd.to_datetime(df[cutoff_col])
    for col in date_cols:
        d_val = pd.to_datetime(df[col])
        res[f"leakage_{col}"] = d_val > cutoff
    return res


# ==============================================================================
# 13. Target Transformations (§2 D18)
# ==============================================================================

def target_transform(D: np.ndarray, s: float = 1.0) -> np.ndarray:
    """Hyperbolic arcsine transform asinh(D / s)."""
    return np.arcsinh(np.asarray(D, dtype=float) / s)


def inverse_target_transform(Z: np.ndarray, s: float = 1.0) -> np.ndarray:
    """Inverse hyperbolic arcsine transform s * sinh(Z)."""
    return s * np.sinh(np.asarray(Z, dtype=float))


# ==============================================================================
# 14. Distributional Pinball Loss & CRPS
# ==============================================================================

def pinball_loss(y: float, q: float, alpha: float) -> float:
    """Quantile pinball loss: max(alpha * (y - q), (1 - alpha) * (q - y))."""
    err = y - q
    return alpha * err if err >= 0 else (alpha - 1.0) * err


def crps_score(y_true: Sequence[float], Q: np.ndarray, alphas: Sequence[float]) -> float:
    """Approximates continuous ranked probability score via pinball loss integration."""
    y_arr = np.asarray(y_true, dtype=float)
    Q_arr = np.asarray(Q, dtype=float)
    alphas_arr = np.asarray(alphas, dtype=float)

    N, M = Q_arr.shape
    total_crps = 0.0

    # Trapezoidal weights for quantile levels
    weights = np.zeros(M)
    weights[0] = 0.5 * (alphas_arr[1] - alphas_arr[0])
    weights[-1] = 0.5 * (alphas_arr[-1] - alphas_arr[-2])
    for j in range(1, M - 1):
        weights[j] = 0.5 * (alphas_arr[j+1] - alphas_arr[j-1])

    for i in range(N):
        y_i = y_arr[i]
        sample_loss = 0.0
        for j in range(M):
            sample_loss += weights[j] * pinball_loss(y_i, Q_arr[i, j], alphas_arr[j])
        total_crps += 2.0 * sample_loss

    return float(total_crps / N)
