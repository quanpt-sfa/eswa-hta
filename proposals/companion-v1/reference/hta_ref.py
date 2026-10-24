"""
HTA Reference Implementation (Candidate v1.1)
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
  5. Post-audit denominator asymmetric cuts (post_denominator_cuts / asymmetric_post_cuts)
  6. Two-sided exceedance probability (two_sided_prob / two_sided_exceedance)
  7. Multi-threshold expansion (expand_thresholds / threshold_expansion)
  8. Expanding-window temporal folds (temporal_expanding_folds)
  9. Paired firm-cluster bootstrap (paired_cluster_bootstrap / paired_firm_cluster_bootstrap)
 10. Near-zero earnings binning (near_zero_edges, assign_bins, near_zero_bins)
 11. Prediction evaluation metrics (ap, compute_average_precision, compute_brier_score, compute_ece, precision_recall_at_k, evaluate_predictions)
 12. Feature leakage and point-in-time guards (check_feature_leakage, check_point_in_time, scan_feature_names, numeric_leak_check)
 13. Fold integrity validation (assert_fold_integrity)
 14. Target transformations (target_transform, inverse_target_transform)
 15. Distributional loss and CRPS (crps_from_quantiles / crps_score, pinball_loss)

Under Rule A3, this candidate implementation must be independently audited,
verified against golden fixtures, and signed off before being promoted to
the authoritative production location `reference/hta_ref.py`.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


# Unresolved official TabICL alpha vector until G0 API preflight passes (§2 D13, §15.3)
TABICL_RAW_ALPHAS: Optional[Sequence[float]] = None

DEFAULT_LEAKAGE_ALLOWLIST = [
    "y_k050_lag1",
    "zero_adj_lag1",
    "D_lag1",
    "n_pos_past",
    "n_obs_past",
]


# ==============================================================================
# 1. Label Construction and Materiality Invariants (§2 D01-D07, §5 G1b)
# ==============================================================================

def build_labels(
    df: pd.DataFrame,
    pat_pre_col: Union[str, Sequence[float], pd.Series] = "pat_pre",
    pat_post_col: Union[str, Sequence[float], pd.Series] = "pat_post",
    total_assets_pre_col: Union[str, Sequence[float], pd.Series] = "total_assets_pre",
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
    pat_pre_col : Union[str, Sequence[float], pd.Series]
        Column name or Series for unaudited pre-audit profit after tax (PAT_pre).
    pat_post_col : Union[str, Sequence[float], pd.Series]
        Column name or Series for audited post-audit profit after tax (PAT_post).
    total_assets_pre_col : Union[str, Sequence[float], pd.Series]
        Column name or Series for pre-audit total assets.
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
        - q: |PAT_pre| / total_assets_pre
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

    if isinstance(pat_pre_col, str):
        pat_pre = df[pat_pre_col].astype(float)
    else:
        pat_pre = pd.Series(pat_pre_col, index=df.index, dtype=float)

    if isinstance(pat_post_col, str):
        pat_post = df[pat_post_col].astype(float)
    else:
        pat_post = pd.Series(pat_post_col, index=df.index, dtype=float)

    if isinstance(total_assets_pre_col, str):
        assets = df[total_assets_pre_col].astype(float)
    else:
        assets = pd.Series(total_assets_pre_col, index=df.index, dtype=float)

    # Basic adjustments
    delta = pat_post - pat_pre
    out["delta"] = delta
    
    # Scale D and q by total assets (avoid division by non-positive or zero assets)
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
# 5. Post-Audit Denominator Asymmetric Cuts (§15.2 T-P01, T-P02, §9.3 R1)
# ==============================================================================

def post_denominator_cuts(
    pat_pre: Union[float, np.ndarray],
    k: float = 0.05,
    total_assets_pre: Optional[Union[float, np.ndarray]] = None
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Computes asymmetric adjustment cuts on Delta scale (or D scale) when the
    materiality threshold is defined relative to post-audit earnings |Delta / PAT_post| >= k.

    Algebraic Derivation:
      Case 1: PAT_pre > 0
        Delta_lower = -k / (1 + k) * PAT_pre
        Delta_upper = +k / (1 - k) * PAT_pre
      Case 2: PAT_pre < 0 (Pre-audit loss)
        Delta_lower = -k / (1 - k) * |PAT_pre|
        Delta_upper = +k / (1 + k) * |PAT_pre|

    If total_assets_pre is supplied, cuts are normalized to the D scale:
      D_lower = Delta_lower / total_assets_pre
      D_upper = Delta_upper / total_assets_pre

    Parameters
    ----------
    pat_pre : Union[float, np.ndarray]
        Pre-audit earnings.
    k : float
        Materiality threshold fraction (0 < k < 1).
    total_assets_pre : Optional[Union[float, np.ndarray]]
        Pre-audit total assets. If provided, returns cuts normalized by assets.

    Returns
    -------
    Tuple[lower_cut, upper_cut]
        The lower and upper cut values.
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

    if total_assets_pre is not None:
        assets_arr = np.asarray(total_assets_pre, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            lower = np.where(assets_arr > 0, lower / assets_arr, np.nan)
            upper = np.where(assets_arr > 0, upper / assets_arr, np.nan)

    if is_scalar:
        return float(lower), float(upper)
    return lower, upper

# Canonical alias
asymmetric_post_cuts = post_denominator_cuts


# ==============================================================================
# 6. Two-Sided Exceedance Function (§15.2 T-C01, §9.3 R1)
# ==============================================================================

def two_sided_prob(
    dist_or_cdf,
    c: Optional[float] = None,
    lower_cutoff: Optional[float] = None,
    upper_cutoff: Optional[float] = None
) -> float:
    """
    Computes two-sided exceedance probability for symmetric or asymmetric cutoffs.

    If c is provided (symmetric threshold on absolute value):
      P(|X| >= c) = F(-c) + (1 - F(c))
    If lower_cutoff and upper_cutoff are provided (asymmetric):
      P(X <= lower_cutoff or X >= upper_cutoff) = F(lower_cutoff) + (1 - F(upper_cutoff))

    Parameters
    ----------
    dist_or_cdf : callable or distribution object with .cdf() and optionally .sf()
    c : Optional[float]
    lower_cutoff : Optional[float]
    upper_cutoff : Optional[float]

    Returns
    -------
    float
        Probability in [0, 1].
    """
    if c is not None:
        if c < 0:
            raise ValueError("Symmetric cutoff c must be non-negative")
        lower_cut = -float(c)
        upper_cut = float(c)
    elif lower_cutoff is not None and upper_cutoff is not None:
        lower_cut = float(lower_cutoff)
        upper_cut = float(upper_cutoff)
    else:
        raise ValueError("Must provide either symmetric 'c' or both 'lower_cutoff' and 'upper_cutoff'")

    if hasattr(dist_or_cdf, "cdf"):
        cdf_fn = dist_or_cdf.cdf
        sf_fn = getattr(dist_or_cdf, "sf", lambda x: 1.0 - cdf_fn(x))
        p_lower = float(cdf_fn(lower_cut))
        p_upper = float(sf_fn(upper_cut))
    elif callable(dist_or_cdf):
        p_lower = float(dist_or_cdf(lower_cut))
        p_upper = float(1.0 - dist_or_cdf(upper_cut))
    else:
        raise TypeError("dist_or_cdf must be a callable or distribution object with .cdf()")

    return float(np.clip(p_lower + p_upper, 0.0, 1.0))

# Canonical alias
two_sided_exceedance = two_sided_prob


# ==============================================================================
# 7. Multi-Threshold Expansion (§2 D06, §6.2, §8.3 B3b)
# ==============================================================================

def expand_thresholds(
    df: pd.DataFrame,
    ks: Sequence[float] = (0.05, 0.10, 0.20),
    pat_pre_col: str = "pat_pre",
    total_assets_pre_col: str = "total_assets_pre",
    delta_col: Optional[str] = "delta",
    y_col_prefix: Optional[str] = "y_k",
    weight_col: Optional[str] = None,
    id_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Expands training observations into long-format across multiple materiality
    thresholds k for threshold-conditioned models (B3b) per §2 D06, §6.2, §8.3.

    Each original firm-year observation is replicated len(ks) times:
      - Threshold k
      - Observation-specific c_k = k * |PAT_pre| / total_assets_pre
      - Binary label y at threshold k: 1 if |delta| >= k * |PAT_pre|, else 0
      - Sample weight: original_weight / len(ks), controlling combined training influence
      - Source row identifier (source_row_id)
      - Retains original firm-year identifiers and fold assignment

    Evaluation thresholds (0.075, 0.15) remain held-out per registered protocol.

    Parameters
    ----------
    df : pd.DataFrame
        Original training dataframe.
    ks : Sequence[float]
        Registered training thresholds (default: 0.05, 0.10, 0.20).
    pat_pre_col : str
    total_assets_pre_col : str
    delta_col : Optional[str]
    y_col_prefix : Optional[str]
    weight_col : Optional[str]
    id_col : Optional[str]

    Returns
    -------
    pd.DataFrame
        Long-format dataframe containing expanded observations.
    """
    if len(df) == 0:
        return pd.DataFrame()

    n_thresholds = len(ks)
    if n_thresholds == 0:
        raise ValueError("Thresholds sequence ks cannot be empty")

    pat_abs = df[pat_pre_col].abs().astype(float)
    assets = df[total_assets_pre_col].astype(float)

    # Base weight per observation
    if weight_col and weight_col in df.columns:
        base_weights = df[weight_col].astype(float)
    else:
        base_weights = pd.Series(1.0, index=df.index, dtype=float)

    # Base row identifier
    if id_col and id_col in df.columns:
        row_ids = df[id_col]
    else:
        row_ids = pd.Series(df.index, index=df.index)

    chunks = []
    for k in ks:
        chunk = df.copy()
        tag = f"{int(round(k * 1000)):03d}"
        
        chunk["k"] = float(k)
        with np.errstate(divide="ignore", invalid="ignore"):
            chunk["c_k"] = np.where(assets > 0, (k * pat_abs) / assets, np.nan)

        # Label resolution
        col_y_name = f"{y_col_prefix}{tag}"
        if col_y_name in df.columns:
            chunk["y"] = df[col_y_name].astype(int)
        elif delta_col and delta_col in df.columns:
            abs_delta = df[delta_col].abs().astype(float)
            chunk["y"] = (abs_delta >= (k * pat_abs)).astype(int)
        else:
            chunk["y"] = np.nan

        # Controlled weight
        chunk["sample_weight"] = base_weights / n_thresholds
        chunk["source_row_id"] = row_ids
        chunks.append(chunk)

    long_df = pd.concat(chunks, axis=0, ignore_index=True)
    return long_df


def threshold_expansion(
    df: pd.DataFrame,
    pat_pre_col: str = "pat_pre",
    total_assets_pre_col: str = "total_assets_pre",
    ks: Sequence[float] = (0.05, 0.075, 0.10, 0.15, 0.20)
) -> pd.DataFrame:
    """
    Wide-format threshold expansion: computes D-scale thresholds c_k = k * |PAT_pre| / assets.
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


def assert_fold_integrity(
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    df: pd.DataFrame,
    year_col: str = "fiscal_year",
    firm_col: Optional[str] = "firm_id"
) -> bool:
    """
    Verifies temporal expanding-window fold integrity (§2 D10, §5 G4):
      1. Exactly 4 development folds (F1-F4) or specified sequence.
      2. Non-empty train and validation indices in every fold.
      3. No index overlap between train and validation within any fold.
      4. Strict temporal precedence: max(train_year) < min(val_year).
      5. Strictly increasing validation years across successive folds.
      6. No validation firm-year observation leaks into training fold.

    Raises
    ------
    AssertionError
        If any fold integrity invariant is violated.

    Returns
    -------
    bool
        True if all invariants hold.
    """
    if len(folds) == 0:
        raise AssertionError("Folds sequence is empty")

    years = df[year_col].to_numpy()
    prev_val_year = -float("inf")

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        fold_num = fold_idx + 1
        if len(train_idx) == 0:
            raise AssertionError(f"Fold F{fold_num} has empty train set")
        if len(val_idx) == 0:
            raise AssertionError(f"Fold F{fold_num} has empty validation set")

        # No index overlap
        overlap = np.intersect1d(train_idx, val_idx)
        if len(overlap) > 0:
            raise AssertionError(f"Fold F{fold_num} has {len(overlap)} overlapping indices between train and val")

        train_years = years[train_idx]
        val_years = years[val_idx]

        max_train_year = train_years.max()
        min_val_year = val_years.min()
        max_val_year = val_years.max()

        if min_val_year != max_val_year:
            raise AssertionError(f"Fold F{fold_num} validation set spans multiple years: {min_val_year} to {max_val_year}")

        if max_train_year >= min_val_year:
            raise AssertionError(
                f"Fold F{fold_num} temporal violation: max train year ({max_train_year}) >= "
                f"min validation year ({min_val_year})"
            )

        if min_val_year <= prev_val_year:
            raise AssertionError(
                f"Fold F{fold_num} validation year ({min_val_year}) is not strictly greater than "
                f"previous fold validation year ({prev_val_year})"
            )
        prev_val_year = min_val_year

        # Firm-year leak check if firm_col present
        if firm_col and firm_col in df.columns:
            firms = df[firm_col].to_numpy()
            train_keys = set(zip(firms[train_idx], train_years))
            val_keys = set(zip(firms[val_idx], val_years))
            key_overlap = train_keys.intersection(val_keys)
            if key_overlap:
                raise AssertionError(f"Fold F{fold_num} has leaking firm-year keys: {key_overlap}")

    return True


# ==============================================================================
# 9. Paired Firm-Cluster Bootstrap (§2 D20, §15.2 T-B01)
# ==============================================================================

def paired_firm_cluster_bootstrap(
    df: pd.DataFrame,
    firm_col: str,
    score_a: Sequence[float],
    score_b: Sequence[float],
    y_true: Sequence[int],
    metric_fn=None,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Executes paired firm-cluster bootstrap testing between model A and model B.
    Resamples FIRMS with replacement, retaining all firm-year observations for
    selected firms to preserve within-cluster temporal correlation.
    """
    if metric_fn is None:
        metric_fn = compute_average_precision

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

    valid_diffs = diffs[np.isfinite(diffs)]
    if len(valid_diffs) < 2:
        raise ValueError("Insufficient valid firm-cluster bootstrap replications")

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


def paired_cluster_bootstrap(
    df: pd.DataFrame,
    firm_col: str,
    score_a: Sequence[float],
    score_b: Sequence[float],
    y_true: Sequence[int],
    metric=None,
    metric_fn=None,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05
) -> Dict[str, float]:
    """Canonical alias matching specification calling contract."""
    fn = metric if metric is not None else metric_fn
    return paired_firm_cluster_bootstrap(
        df=df,
        firm_col=firm_col,
        score_a=score_a,
        score_b=score_b,
        y_true=y_true,
        metric_fn=fn,
        n_boot=n_boot,
        seed=seed,
        alpha=alpha
    )


# ==============================================================================
# 10. Near-Zero Earnings Bins (§2 D22, §5 G4)
# ==============================================================================

def near_zero_edges(
    q: Union[Sequence[float], np.ndarray, pd.Series],
    n_bins: int = 5
) -> np.ndarray:
    """
    Computes quantile bin edges on development eligible_model observations
    of q = |PAT_pre| / assets (§2 D22, §5 G4).

    Parameters
    ----------
    q : Union[Sequence[float], np.ndarray, pd.Series]
        Development eligible_model values of q.
    n_bins : int
        Number of quantile bins (default: 5 quintiles).

    Returns
    -------
    np.ndarray
        Array of length n_bins + 1 containing bin edges starting at -inf and ending at +inf.
    """
    q_arr = np.asarray(q, dtype=float)
    q_valid = q_arr[np.isfinite(q_arr)]
    if len(q_valid) == 0 or n_bins < 2:
        raise ValueError("near_zero_edges requires finite, nonempty data and n_bins >= 2")

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = np.percentile(q_valid, quantiles * 100)
    if np.any(np.diff(bin_edges) <= 0):
        raise ValueError(
            f"Quantile bin edges are tied ({bin_edges}); "
            "request a pre-registered binning fallback or discrete edge policy"
        )
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    return bin_edges


def assign_bins(
    q: Union[Sequence[float], np.ndarray, pd.Series],
    edges: np.ndarray
) -> pd.Series:
    """
    Assigns near-zero bin indices [0..len(edges)-2] using frozen edges (§2 D22).
    Does NOT recompute quantile edges on test or validation data.
    """
    edges_arr = np.asarray(edges, dtype=float)
    if len(edges_arr) < 3 or np.any(np.diff(edges_arr) <= 0):
        raise ValueError("Invalid edges array: must have at least 3 strictly increasing elements")

    q_series = pd.Series(q) if not isinstance(q, pd.Series) else q
    bins = pd.cut(q_series, bins=edges_arr, labels=False, include_lowest=True)
    return bins


def near_zero_bins(
    df: pd.DataFrame,
    q_col: str = "q",
    n_bins: int = 5,
    eligible_col: str = "eligible_model"
) -> Tuple[pd.Series, np.ndarray]:
    """
    Legacy wrapper: computes quintiles of q on eligible development rows
    and assigns bins to all rows in df.
    """
    eligible_mask = df[eligible_col].fillna(False).astype(bool)
    dev_q = df.loc[eligible_mask, q_col].dropna()
    edges = near_zero_edges(dev_q, n_bins=n_bins)
    bin_ids = assign_bins(df[q_col], edges)
    return bin_ids, edges


# ==============================================================================
# 11. Prediction Evaluation Metrics (§2 D20, D23, §9.1)
# ==============================================================================

def compute_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Average Precision with score-tie handling identical to sklearn.metrics.average_precision_score.
    All examples with identical prediction scores form one precision-recall step;
    arbitrary input row order does not alter the result.
    """
    from sklearn.metrics import average_precision_score

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.ndim != 1 or y_score.ndim != 1 or len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must be one-dimensional with equal length")
    if not len(y_true) or not np.isfinite(y_score).all():
        raise ValueError("AP requires nonempty finite score arrays")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("AP requires binary 0/1 labels")
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))

# Canonical alias
ap = compute_average_precision


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


def precision_recall_at_k(
    y_true: Sequence[int],
    y_score: Sequence[float],
    k_budget: float
) -> Tuple[float, float]:
    """
    Computes Precision@K and Recall@K at screening budget fraction k_budget (§2 D21, §9.1).

    Parameters
    ----------
    y_true : Sequence[int]
        Binary ground-truth labels {0, 1}.
    y_score : Sequence[float]
        Predicted risk probabilities or scores.
    k_budget : float
        Screening budget fraction in (0, 1], e.g. 0.05 or 0.10.

    Returns
    -------
    Tuple[precision, recall]
    """
    y_t = np.asarray(y_true, dtype=int)
    y_s = np.asarray(y_score, dtype=float)

    if len(y_t) != len(y_s):
        raise ValueError("y_true and y_score must have equal length")
    if not (0.0 < k_budget <= 1.0):
        raise ValueError(f"k_budget must be in (0, 1], got {k_budget}")

    n = len(y_t)
    if n == 0:
        return 0.0, 0.0

    m = int(math.ceil(k_budget * n))
    m = min(m, n)

    # Sort descending by score stably
    order = np.argsort(-y_s, kind="stable")
    top_indices = order[:m]

    n_pos = int(np.sum(y_t == 1))
    tp = int(np.sum(y_t[top_indices] == 1))

    precision = tp / m if m > 0 else 0.0
    recall = tp / n_pos if n_pos > 0 else 0.0

    return float(precision), float(recall)


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
# 12. Feature Leakage & Point-in-Time Guards (§2 D08-D09, §5 G2, §15.2 T-T01)
# ==============================================================================

def check_feature_leakage(
    feature_pub_dates: Sequence[Union[str, pd.Timestamp]],
    prediction_cutoffs: Sequence[Union[str, pd.Timestamp]]
) -> pd.Series:
    """Fail closed: missing/invalid timestamps are unverified, not safe."""
    f_dates = pd.to_datetime(feature_pub_dates, errors="coerce", utc=True)
    cutoffs = pd.to_datetime(prediction_cutoffs, errors="coerce", utc=True)
    if len(f_dates) != len(cutoffs):
        raise ValueError("Publication dates and cutoffs must have equal length")
    flags = f_dates.isna() | cutoffs.isna() | (f_dates > cutoffs)
    return pd.Series(np.asarray(flags, dtype=bool), name="leakage")


def check_point_in_time(
    df: pd.DataFrame,
    date_cols: Sequence[str],
    cutoff_col: str
) -> pd.DataFrame:
    """Return True for an unavailable, missing, or future-dated feature."""
    res = pd.DataFrame(index=df.index)
    cutoff = pd.to_datetime(df[cutoff_col], errors="coerce", utc=True)
    for col in date_cols:
        date = pd.to_datetime(df[col], errors="coerce", utc=True)
        res[f"leakage_{col}"] = date.isna() | cutoff.isna() | (date > cutoff)
    return res


def scan_feature_names(
    feature_names: Sequence[str],
    allow: Optional[Sequence[str]] = None
) -> List[str]:
    """
    Scans candidate feature column names for potential post-audit leakage (§2 D08, §5 G2).

    Rules:
      - Forbidden outcome variables and substrings: 'aud', 'post', 'delta', 'pat_post', 'audit_adj', 'is_zero_adj', 'eligible_'
      - Allowed: explicit entries in allowlist, or verified lagged variables (e.g. ending in _lag1, _lag2).

    Returns
    -------
    List[str]
        List of flagged feature names. Returns [] if all features are clean.
    """
    allowlist = set(DEFAULT_LEAKAGE_ALLOWLIST if allow is None else allow)
    forbidden_exact = {
        "pat_post", "delta", "d", "is_zero_adj", "eligible_label", "eligible_model",
        "usable_pair", "sign_change", "boundary_case"
    }
    forbidden_substrings = ["audit_adj", "fs_aud", "post_audit", "audited"]

    flagged = []
    for col in feature_names:
        c_lower = str(col).lower()
        if col in allowlist or c_lower in allowlist:
            continue
        # Verified lagged items in allowlist
        if any(c_lower.endswith(f"_lag{i}") for i in range(1, 10)):
            if c_lower in allowlist or any(allowed_pat in c_lower for allowed_pat in allowlist):
                continue
        # Exact forbidden outcomes
        if c_lower in forbidden_exact or c_lower.startswith("y_k") or c_lower.startswith("c_k"):
            flagged.append(col)
            continue
        # Forbidden substrings (unless lagged and allowed)
        if any(sub in c_lower for sub in forbidden_substrings) or c_lower.endswith("_aud") or c_lower.endswith("_post"):
            flagged.append(col)
            continue

    return flagged


def numeric_leak_check(
    X_numeric: pd.DataFrame,
    pat_pre: pd.Series,
    pat_post: pd.Series,
    delta: pd.Series,
    y_k050: pd.Series,
    corr_threshold: float = 0.99
) -> pd.DataFrame:
    """
    Development-only numeric leakage diagnostics (§5 G2).
    Calculates Pearson correlation between each numeric predictor and outcome targets.
    Flags features with near-perfect correlation (|r| >= corr_threshold).
    Returns diagnostic DataFrame: [feature, target, correlation, suspicious].
    Does NOT automatically remove features.
    """
    targets = {
        "pat_post": pd.to_numeric(pat_post, errors="coerce"),
        "delta": pd.to_numeric(delta, errors="coerce"),
        "y_k050": pd.to_numeric(y_k050, errors="coerce"),
    }

    records = []
    for col in X_numeric.columns:
        s = pd.to_numeric(X_numeric[col], errors="coerce")
        if s.std(ddof=0) == 0 or s.isna().all():
            continue
        for tgt_name, tgt_s in targets.items():
            valid_mask = s.notna() & tgt_s.notna()
            if valid_mask.sum() < 5 or tgt_s[valid_mask].std(ddof=0) == 0:
                continue
            r = float(np.corrcoef(s[valid_mask], tgt_s[valid_mask])[0, 1])
            is_suspicious = abs(r) >= corr_threshold
            records.append({
                "feature": col,
                "target": tgt_name,
                "correlation": r,
                "suspicious": is_suspicious
            })

    res_df = pd.DataFrame(records)
    if res_df.empty:
        return pd.DataFrame(columns=["feature", "target", "correlation", "suspicious"])
    return res_df.sort_values(by="correlation", key=abs, ascending=False).reset_index(drop=True)


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
# 14. Distributional Pinball Loss & CRPS (§14 Rule 8, §15.2)
# ==============================================================================

def pinball_loss(y: float, q: float, alpha: float) -> float:
    """Quantile pinball loss: max(alpha * (y - q), (1 - alpha) * (q - y))."""
    err = y - q
    return alpha * err if err >= 0 else (alpha - 1.0) * err


def crps_from_quantiles(
    y_true: Sequence[float],
    quantiles: np.ndarray,
    alphas: Sequence[float]
) -> float:
    """
    Computes Continuous Ranked Probability Score (CRPS) from predicted quantiles
    via numerical trapezoidal pinball loss integration (§14 Rule 8, §15.2).

    Evaluated on the original continuous outcome scale.

    Parameters
    ----------
    y_true : Sequence[float]
        Observed continuous outcome.
    quantiles : np.ndarray
        Array of shape (N, M) containing predicted quantiles.
    alphas : Sequence[float]
        Quantile levels strictly increasing in (0, 1).

    Returns
    -------
    float
        Mean CRPS across all observations.
    """
    y_arr = np.asarray(y_true, dtype=float)
    Q_arr = np.asarray(quantiles, dtype=float)
    alphas_arr = np.asarray(alphas, dtype=float)

    if Q_arr.ndim == 1:
        Q_arr = Q_arr.reshape(1, -1)

    N, M = Q_arr.shape
    if M != len(alphas_arr):
        raise ValueError(f"Quantile count mismatch: Q has {M} cols, alphas has {len(alphas_arr)}")
    if np.any(np.diff(alphas_arr) <= 0):
        raise ValueError("alphas must be strictly increasing")
    if len(y_arr) != N:
        raise ValueError(f"Length mismatch: y_true has {len(y_arr)} elements, quantiles has {N} rows")

    # Enforce non-crossing sorting per row
    for i in range(N):
        if np.any(np.diff(Q_arr[i]) < 0):
            Q_arr[i] = np.sort(Q_arr[i])

    # Trapezoidal weights for non-uniform quantile grid
    weights = np.zeros(M)
    weights[0] = 0.5 * (alphas_arr[1] - alphas_arr[0])
    weights[-1] = 0.5 * (alphas_arr[-1] - alphas_arr[-2])
    for j in range(1, M - 1):
        weights[j] = 0.5 * (alphas_arr[j+1] - alphas_arr[j-1])

    total_crps = 0.0
    for i in range(N):
        y_i = y_arr[i]
        sample_loss = 0.0
        for j in range(M):
            sample_loss += weights[j] * pinball_loss(y_i, Q_arr[i, j], alphas_arr[j])
        total_crps += 2.0 * sample_loss

    return float(total_crps / N)

# Canonical alias
crps_score = crps_from_quantiles
