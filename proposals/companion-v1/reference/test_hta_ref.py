"""
Candidate Unit Test Suite for HTA Reference Implementation
==========================================================
Repository: quanpt-sfa/eswa-hta
Governance: docs/HTA_IMPLEMENTATION_SPEC_v1.2.md (§15.1, §15.2, Rule A3)
Status: PENDING_INDEPENDENT_REVIEW

Tests all functions in `proposals/companion-v1/reference/hta_ref.py`
against analytical derivations, invariant properties, and golden fixtures.
"""

import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy import stats

ref_dir = Path(__file__).resolve().parent
if str(ref_dir) not in sys.path:
    sys.path.insert(0, str(ref_dir))

import hta_ref


# ==============================================================================
# 1. Label Construction Tests (§2 D01-D07, §15.2)
# ==============================================================================

def test_build_labels_golden_cases():
    """Verify build_labels on synthetic golden cases."""
    df = pd.DataFrame({
        "pat_pre": [100.0, 100.0, 100.0, 0.0, 100.0, 100.0, -100.0, -100.0, -100.0, 100.0, 100.0, 1e-9, 100.0, 100.0],
        "pat_post": [104.0, 105.0, 95.0, 10.0, -100.0, 0.0, -105.0, -95.0, -96.0, 100.0, 100.0000000001, 1.1e-9, 105.0, 105.0],
        "total_assets_pre": [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 0.0, -500.0]
    }, index=[
        "T-L01", "T-L02", "T-L03", "T-L04", "T-L05", "T-L06", "T-L07", "T-L08", "T-L09",
        "T-Z01", "T-Z02", "T-Z03", "T-M01", "T-M02"
    ])

    res = hta_ref.build_labels(df, zero_tol=0.0)

    # T-L01: Subthreshold (+4 < 5)
    assert res.loc["T-L01", "eligible_label"] == True
    assert res.loc["T-L01", "eligible_model"] == True
    assert res.loc["T-L01", "y_k050"] == 0
    assert res.loc["T-L01", "is_zero_adj"] == False

    # T-L02: Inclusive threshold (+5 >= 5)
    assert res.loc["T-L02", "eligible_label"] == True
    assert res.loc["T-L02", "y_k050"] == 1

    # T-L03: Negative adjustment at threshold (-5 -> |-5| >= 5)
    assert res.loc["T-L03", "eligible_label"] == True
    assert res.loc["T-L03", "y_k050"] == 1

    # T-L04: Zero pre-audit PAT (denominator zero)
    assert res.loc["T-L04", "eligible_label"] == False
    assert res.loc["T-L04", "pre_zero"] == True
    assert pd.isna(res.loc["T-L04", "y_k050"])

    # T-L05: Sign change
    assert res.loc["T-L05", "eligible_label"] == False
    assert res.loc["T-L05", "sign_change"] == True
    assert pd.isna(res.loc["T-L05", "y_k050"])

    # T-L06: Post-audit PAT zero (allowed and eligible)
    assert res.loc["T-L06", "eligible_label"] == True
    assert res.loc["T-L06", "y_k050"] == 1

    # T-L07 & T-L08: Negative pre-audit earnings
    assert res.loc["T-L07", "eligible_label"] == True
    assert res.loc["T-L07", "y_k050"] == 1
    assert res.loc["T-L08", "eligible_label"] == True
    assert res.loc["T-L08", "y_k050"] == 1
    assert res.loc["T-L09", "eligible_label"] == True
    assert res.loc["T-L09", "y_k050"] == 0

    # T-Z01: Exact zero adjustment
    assert res.loc["T-Z01", "is_zero_adj"] == True
    assert res.loc["T-Z01", "y_k050"] == 0

    # T-Z02: Corrected near-zero subthreshold adjustment
    assert res.loc["T-Z02", "is_zero_adj"] == False
    assert res.loc["T-Z02", "y_k050"] == 0  # 1e-10 < 5.0

    # T-Z03: Near-zero earnings with genuine exceedance
    assert res.loc["T-Z03", "eligible_label"] == True
    assert res.loc["T-Z03", "is_zero_adj"] == False
    assert res.loc["T-Z03", "y_k050"] == 1  # 1e-10 >= 0.05 * 1e-9 = 5e-11

    # T-M01 & T-M02: Asset filtering
    assert res.loc["T-M01", "eligible_label"] == True
    assert res.loc["T-M01", "eligible_model"] == False
    assert res.loc["T-M02", "eligible_label"] == True
    assert res.loc["T-M02", "eligible_model"] == False


def test_build_labels_monotonic_nesting():
    """Verify invariant: y_k200 <= y_k150 <= y_k100 <= y_k075 <= y_k050."""
    rng = np.random.default_rng(42)
    N = 200
    pat_pre = rng.normal(loc=100.0, scale=30.0, size=N)
    pat_post = pat_pre + rng.normal(loc=0.0, scale=15.0, size=N)
    assets = rng.uniform(500.0, 5000.0, size=N)

    df = pd.DataFrame({"pat_pre": pat_pre, "pat_post": pat_post, "total_assets_pre": assets})
    res = hta_ref.build_labels(df)
    eligible = res[res["eligible_label"]]

    y050 = eligible["y_k050"].to_numpy()
    y075 = eligible["y_k075"].to_numpy()
    y100 = eligible["y_k100"].to_numpy()
    y150 = eligible["y_k150"].to_numpy()
    y200 = eligible["y_k200"].to_numpy()

    assert np.all(y200 <= y150), "y200 <= y150 violated"
    assert np.all(y150 <= y100), "y150 <= y100 violated"
    assert np.all(y100 <= y075), "y100 <= y075 violated"
    assert np.all(y075 <= y050), "y075 <= y050 violated"


# ==============================================================================
# 2. Exceedance Probability Tests (§2 D13-D15, §15.2 T-C01, T-C02)
# ==============================================================================

def test_exceedance_prob_analytic_normal():
    """Test exceedance_prob against exact standard normal CDF (T-C01)."""
    alphas = np.linspace(0.001, 0.999, 999)
    Q = stats.norm.ppf(alphas)

    prob = hta_ref.exceedance_prob(Q, alphas, c=1.0, tail_policy="conservative")
    expected_prob = 2.0 * (1.0 - stats.norm.cdf(1.0))  # 0.31731050786291415

    assert math.isclose(float(prob[0]), expected_prob, abs_tol=5e-4)


def test_exceedance_prob_monotonic_thresholds():
    """Higher threshold must produce lower or equal exceedance probability."""
    alphas = np.linspace(0.01, 0.99, 99)
    Q = stats.norm.ppf(alphas)

    p05 = hta_ref.exceedance_prob(Q, alphas, c=0.05)[0]
    p10 = hta_ref.exceedance_prob(Q, alphas, c=0.10)[0]
    p20 = hta_ref.exceedance_prob(Q, alphas, c=0.20)[0]

    assert p05 >= p10 >= p20, f"Monotonicity failed: {p05} >= {p10} >= {p20}"


# ==============================================================================
# 3. Hurdle Mixture Exceedance Tests (§2 D17, §15.2 T-H01)
# ==============================================================================

def test_hurdle_exceedance_formula():
    """Test hurdle mixture P(|D| >= c) = (1 - pi_0) * P(|D_nonzero| >= c)."""
    alphas = np.linspace(0.001, 0.999, 999)
    Q = stats.norm.ppf(alphas)

    pi_zero = 0.4
    hurdle_p = hta_ref.hurdle_exceedance(pi_zero, Q, alphas, c=1.0)
    cond_p = hta_ref.exceedance_prob(Q, alphas, c=1.0)

    assert math.isclose(float(hurdle_p[0]), 0.6 * float(cond_p[0]), rel_tol=1e-12)


# ==============================================================================
# 4. Scipy / NGBoost Exceedance Tests (§2 D16, §15.2 T-C01, T-C02)
# ==============================================================================

def test_scipy_exceedance():
    norm_dist = stats.norm(loc=0, scale=1)
    laplace_dist = stats.laplace(loc=0, scale=1)

    p_norm = hta_ref.scipy_exceedance(norm_dist, c=1.0)
    p_laplace = hta_ref.scipy_exceedance(laplace_dist, c=1.0)

    expected_norm = 2.0 * (1.0 - stats.norm.cdf(1.0))
    expected_laplace = math.exp(-1.0)

    assert math.isclose(float(p_norm), expected_norm, abs_tol=1e-12)
    assert math.isclose(float(p_laplace), expected_laplace, abs_tol=1e-12)


# ==============================================================================
# 5. Asymmetric Post-Denominator Cuts (§15.2 T-P01, T-P02)
# ==============================================================================

def test_asymmetric_post_cuts():
    # Positive PAT_pre (T-P01)
    lower_pos, upper_pos = hta_ref.post_denominator_cuts(100.0, k=0.05)
    assert math.isclose(lower_pos, -4.761904761904762, abs_tol=1e-10)
    assert math.isclose(upper_pos, 5.263157894736842, abs_tol=1e-10)

    # Negative PAT_pre (T-P02) - magnitudes reversed
    lower_neg, upper_neg = hta_ref.post_denominator_cuts(-100.0, k=0.05)
    assert math.isclose(lower_neg, -5.263157894736842, abs_tol=1e-10)
    assert math.isclose(upper_neg, 4.761904761904762, abs_tol=1e-10)


def test_post_denominator_cuts_d_scale():
    """Verify post_denominator_cuts normalized by assets to D scale."""
    lower_d, upper_d = hta_ref.post_denominator_cuts(100.0, k=0.05, total_assets_pre=1000.0)
    assert math.isclose(lower_d, -0.004761904761904762, abs_tol=1e-10)
    assert math.isclose(upper_d, 0.005263157894736842, abs_tol=1e-10)


# ==============================================================================
# 6. Two-Sided Probability Calculation (§15.2 T-C01, §9.3 R1)
# ==============================================================================

def test_two_sided_prob_symmetric_and_asymmetric():
    norm_dist = stats.norm(loc=0, scale=1)

    # Symmetric c = 1.0
    p_sym = hta_ref.two_sided_prob(norm_dist, c=1.0)
    expected_sym = 2.0 * (1.0 - stats.norm.cdf(1.0))
    assert math.isclose(p_sym, expected_sym, abs_tol=1e-12)

    # Asymmetric cutoffs: lower=-1.0, upper=+2.0
    p_asym = hta_ref.two_sided_prob(norm_dist, lower_cutoff=-1.0, upper_cutoff=2.0)
    expected_asym = stats.norm.cdf(-1.0) + (1.0 - stats.norm.cdf(2.0))
    assert math.isclose(p_asym, expected_asym, abs_tol=1e-12)

    # Error handling
    with pytest.raises(ValueError):
        hta_ref.two_sided_prob(norm_dist, c=-1.0)
    with pytest.raises(ValueError):
        hta_ref.two_sided_prob(norm_dist)


# ==============================================================================
# 7. Threshold Expansion Tests (§2 D06, §6.2, §8.3 B3b)
# ==============================================================================

def test_expand_thresholds():
    df = pd.DataFrame({
        "firm_id": ["F1", "F2"],
        "fiscal_year": [2018, 2018],
        "pat_pre": [100.0, 200.0],
        "total_assets_pre": [1000.0, 2000.0],
        "delta": [6.0, 15.0],
        "feature_1": [1.5, 2.5]
    })

    long_df = hta_ref.expand_thresholds(df, ks=(0.05, 0.10, 0.20))
    assert len(long_df) == 6  # 2 obs * 3 thresholds

    # Invariants
    assert set(long_df["k"].unique()) == {0.05, 0.10, 0.20}
    assert "c_k" in long_df.columns
    assert "y" in long_df.columns
    assert "sample_weight" in long_df.columns
    assert "source_row_id" in long_df.columns

    # Combined weights per original observation must sum to 1.0
    for orig_idx in df.index:
        obs_weights = long_df[long_df["source_row_id"] == orig_idx]["sample_weight"]
        assert math.isclose(obs_weights.sum(), 1.0, rel_tol=1e-12)

    # Check labels for F1 (delta=6, pat_pre=100 -> delta/pat_pre = 0.06):
    # at k=0.05: y=1 (0.06 >= 0.05)
    # at k=0.10: y=0 (0.06 < 0.10)
    # at k=0.20: y=0 (0.06 < 0.20)
    f1_rows = long_df[long_df["firm_id"] == "F1"].sort_values("k")
    assert list(f1_rows["y"]) == [1, 0, 0]


# ==============================================================================
# 8. Near-Zero Edges and Bin Assignment (§2 D22, §5 G4)
# ==============================================================================

def test_near_zero_edges_and_assign_bins():
    q_dev = np.linspace(0.01, 0.50, 100)
    edges = hta_ref.near_zero_edges(q_dev, n_bins=5)
    assert len(edges) == 6
    assert edges[0] == -np.inf
    assert edges[-1] == np.inf
    assert np.all(np.diff(edges) > 0)

    # Assign bins to new out-of-sample data reusing edges
    q_test = pd.Series([-10.0, 0.02, 0.25, 0.49, 100.0])
    bins = hta_ref.assign_bins(q_test, edges)
    assert list(bins) == [0, 0, 2, 4, 4]


def test_near_zero_quintiles_reject_duplicated_edges():
    q_tied = [0.01] * 25
    with pytest.raises(ValueError, match="tied"):
        hta_ref.near_zero_edges(q_tied, n_bins=5)


# ==============================================================================
# 9. Temporal Folds and Fold Integrity Tests (§2 D10, §5 G4)
# ==============================================================================

def test_temporal_expanding_folds():
    df = pd.DataFrame({
        "fiscal_year": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
    })
    folds = hta_ref.temporal_expanding_folds(df, year_col="fiscal_year", start_year=2015, dev_end_year=2022)
    assert len(folds) == 4, "Must generate exactly 4 development folds (F1-F4)"

    val_years = [2019, 2020, 2021, 2022]
    for idx, (train_idx, val_idx) in enumerate(folds):
        train_years = df.loc[train_idx, "fiscal_year"]
        val_year = df.loc[val_idx, "fiscal_year"].iloc[0]

        assert val_year == val_years[idx]
        assert train_years.max() < val_year, "Strict temporal precedence violated"
        assert train_years.min() == 2015, "Expanding window must anchor at start year"


def test_assert_fold_integrity():
    df = pd.DataFrame({
        "firm_id": ["F1", "F2", "F1", "F2", "F1", "F2", "F1", "F2"],
        "fiscal_year": [2015, 2015, 2016, 2016, 2017, 2017, 2018, 2018]
    })
    # Valid expanding fold
    valid_folds = [
        (np.array([0, 1]), np.array([2, 3])),
        (np.array([0, 1, 2, 3]), np.array([4, 5])),
        (np.array([0, 1, 2, 3, 4, 5]), np.array([6, 7]))
    ]
    assert hta_ref.assert_fold_integrity(valid_folds, df) == True

    # Leaking fold (future train year)
    invalid_temporal_folds = [
        (np.array([2, 3, 4, 5]), np.array([0, 1]))
    ]
    with pytest.raises(AssertionError, match="temporal violation"):
        hta_ref.assert_fold_integrity(invalid_temporal_folds, df)

    # Overlapping indices fold
    invalid_overlap_folds = [
        (np.array([0, 1, 2]), np.array([2, 3]))
    ]
    with pytest.raises(AssertionError, match="overlapping"):
        hta_ref.assert_fold_integrity(invalid_overlap_folds, df)


# ==============================================================================
# 10. Feature Leakage Guard Tests (§2 D08-D09, §5 G2, §15.2 T-T01)
# ==============================================================================

def test_check_feature_leakage():
    pub_dates = ["2020-03-01", "2020-04-15", "2020-05-01"]
    cutoffs = ["2020-03-15", "2020-04-15", "2020-04-30"]

    flags = hta_ref.check_feature_leakage(pub_dates, cutoffs)
    assert flags.iloc[0] == False  # Released before cutoff
    assert flags.iloc[1] == False  # Same day (permitted)
    assert flags.iloc[2] == True   # Released after cutoff (LEAKAGE)


def test_unknown_publication_date_is_not_safe():
    flags = hta_ref.check_feature_leakage(
        ["2020-03-01", None, "2020-05-01", "2020-01-01"],
        ["2020-04-01", "2020-04-01", None, "2020-02-01"],
    )
    assert list(flags) == [False, True, True, False]
    d = pd.DataFrame({
        "source_date": [None, "2020-04-05"],
        "cutoff": ["2020-04-01", "2020-04-01"],
    })
    assert list(hta_ref.check_point_in_time(d, ["source_date"], "cutoff")["leakage_source_date"]) == [True, True]


def test_scan_feature_names():
    clean_cols = ["sales_growth", "leverage", "roa", "y_k050_lag1", "zero_adj_lag1", "D_lag1"]
    assert hta_ref.scan_feature_names(clean_cols) == []

    leaky_cols = ["sales_growth", "pat_post", "fs_aud_revenue", "audit_adj_tax", "y_k050"]
    flagged = hta_ref.scan_feature_names(leaky_cols)
    assert set(flagged) == {"pat_post", "fs_aud_revenue", "audit_adj_tax", "y_k050"}


def test_numeric_leak_check():
    rng = np.random.default_rng(42)
    N = 50
    pat_post = pd.Series(rng.normal(100, 20, N))
    delta = pd.Series(rng.normal(5, 2, N))
    y_k050 = pd.Series((delta > 5).astype(int))
    pat_pre = pat_post - delta

    X_numeric = pd.DataFrame({
        "safe_feature": rng.normal(0, 1, N),
        "leaking_pat_post": pat_post + 1e-12,  # Perfectly correlated
        "safe_feature_2": rng.uniform(10, 20, N)
    })

    report = hta_ref.numeric_leak_check(X_numeric, pat_pre, pat_post, delta, y_k050, corr_threshold=0.99)
    assert not report.empty
    suspicious = report[report["suspicious"]]
    assert len(suspicious) == 1
    assert suspicious.iloc[0]["feature"] == "leaking_pat_post"
    assert math.isclose(suspicious.iloc[0]["correlation"], 1.0, abs_tol=1e-6)


# ==============================================================================
# 11. Prediction Evaluation Metrics (§2 D20, D23, §9.1)
# ==============================================================================

def test_average_precision_ties_invariant_to_row_order():
    scores = np.array([0.5, 0.5])
    assert hta_ref.ap([1, 0], scores) == pytest.approx(0.5)
    assert hta_ref.ap([0, 1], scores) == pytest.approx(0.5)


def test_precision_recall_at_k():
    y_true = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    y_score = [0.95, 0.90, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05]

    # At budget 0.20 (top 2 of 10)
    p20, r20 = hta_ref.precision_recall_at_k(y_true, y_score, k_budget=0.20)
    assert p20 == 1.0
    assert r20 == 1.0

    # At budget 0.10 (top 1 of 10)
    p10, r10 = hta_ref.precision_recall_at_k(y_true, y_score, k_budget=0.10)
    assert p10 == 1.0
    assert r10 == 0.5

    # At budget 0.50 (top 5 of 10)
    p50, r50 = hta_ref.precision_recall_at_k(y_true, y_score, k_budget=0.50)
    assert math.isclose(p50, 2.0 / 5.0)
    assert r50 == 1.0


# ==============================================================================
# 12. Paired Cluster Bootstrap Tests (§2 D20, §15.2 T-B01)
# ==============================================================================

def test_paired_firm_cluster_bootstrap():
    df = pd.DataFrame({
        "firm_id": ["F1", "F1", "F2", "F2", "F3", "F3", "F4", "F4"],
        "y": [0, 1, 0, 0, 1, 1, 0, 1],
        "score_a": [0.1, 0.9, 0.2, 0.1, 0.8, 0.85, 0.3, 0.7],
        "score_b": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    })

    res = hta_ref.paired_cluster_bootstrap(
        df,
        firm_col="firm_id",
        score_a=df["score_a"],
        score_b=df["score_b"],
        y_true=df["y"],
        metric=hta_ref.ap,
        n_boot=100,
        seed=42
    )

    assert res["point_diff"] > 0
    assert "ci_lower" in res and "ci_upper" in res
    assert res["n_valid_bootstraps"] > 0


# ==============================================================================
# 13. Target Transformation Bijectivity Tests (§2 D18)
# ==============================================================================

def test_target_transform_bijectivity():
    D = np.array([-1.5, -0.5, 0.0, 0.05, 1.0, 10.0])
    s = 0.5

    Z = hta_ref.target_transform(D, s=s)
    D_rec = hta_ref.inverse_target_transform(Z, s=s)

    assert np.allclose(D, D_rec, rtol=1e-12, atol=1e-12), "asinh/sinh round-trip failed"


# ==============================================================================
# 14. CRPS and Quantile Loss Tests (§14 Rule 8, §15.2)
# ==============================================================================

def test_crps_score():
    alphas = np.linspace(0.1, 0.9, 9)
    N = 100
    y_true = np.zeros(N)
    Q = np.tile(stats.norm.ppf(alphas), (N, 1))

    crps = hta_ref.crps_from_quantiles(y_true, Q, alphas)
    assert crps > 0.0, "CRPS must be positive"
    assert math.isfinite(crps)


# ==============================================================================
# 15. Comprehensive Golden Fixture Execution (§15.2)
# ==============================================================================

def test_all_golden_fixtures_from_csv():
    """
    Executes and asserts all 20 golden cases from CSV fixtures against reference functions.
    """
    fixtures_dir = ref_dir.parent / "fixtures"
    cases_df = pd.read_csv(fixtures_dir / "golden_cases.csv").set_index("test_id")
    exp_df = pd.read_csv(fixtures_dir / "golden_expectations.csv").set_index("test_id")

    assert len(cases_df) == 20
    assert len(exp_df) == 20

    for test_id in cases_df.index:
        case = cases_df.loc[test_id]
        exp = exp_df.loc[test_id]
        tol = float(exp["tolerance"])

        if test_id.startswith(("T-L", "T-Z", "T-M")):
            row_df = pd.DataFrame({
                "pat_pre": [float(case["pat_pre"])],
                "pat_post": [float(case["pat_post"])],
                "total_assets_pre": [float(case["total_assets_pre"])]
            }, index=[test_id])
            res = hta_ref.build_labels(row_df, zero_tol=0.0)

            assert res.loc[test_id, "eligible_label"] == (exp["eligible_label"] == "True" or exp["eligible_label"] == True)
            assert res.loc[test_id, "eligible_model"] == (exp["eligible_model"] == "True" or exp["eligible_model"] == True)

            if pd.notna(exp["is_zero_adj"]):
                assert res.loc[test_id, "is_zero_adj"] == (exp["is_zero_adj"] == "True" or exp["is_zero_adj"] == True)

            if pd.notna(exp["y_k"]):
                assert res.loc[test_id, "y_k050"] == int(float(exp["y_k"]))

        elif test_id == "T-C01":
            p = hta_ref.scipy_exceedance(stats.norm(0, 1), c=float(case["c"]))
            assert math.isclose(float(p), float(exp["expected_prob"]), abs_tol=max(tol, 1e-10))

        elif test_id == "T-C02":
            p = hta_ref.scipy_exceedance(stats.laplace(0, 1), c=float(case["c"]))
            assert math.isclose(float(p), float(exp["expected_prob"]), abs_tol=max(tol, 1e-10))

        elif test_id == "T-H01":
            alphas = np.linspace(0.001, 0.999, 999)
            Q = stats.norm.ppf(alphas)
            p = hta_ref.hurdle_exceedance(float(case["pi_zero"]), Q, alphas, c=float(case["c"]))
            assert math.isclose(float(p[0]), float(exp["expected_prob"]), abs_tol=max(tol, 5e-4))

        elif test_id in ("T-P01", "T-P02"):
            low, up = hta_ref.post_denominator_cuts(float(case["pat_pre"]), k=float(case["k"]))
            assert math.isclose(low, float(exp["lower_delta_cut"]), abs_tol=max(tol, 1e-8))
            assert math.isclose(up, float(exp["upper_delta_cut"]), abs_tol=max(tol, 1e-8))

        elif test_id == "T-N01":
            p = hta_ref.scipy_exceedance(stats.norm(0, 1), c=float(case["c"]))
            assert math.isclose(float(p), float(exp["expected_prob"]), abs_tol=max(tol, 1e-10))
