"""
Unit test for validating golden fixture schema, field counts, and mathematical integrity.
"""

import csv
from pathlib import Path
import pytest
import math
from scipy import stats

fixtures_dir = Path(__file__).resolve().parent

cases_path = fixtures_dir / "golden_cases.csv"
exp_path = fixtures_dir / "golden_expectations.csv"

def test_csv_file_existence():
    assert cases_path.exists(), "golden_cases.csv must exist"
    assert exp_path.exists(), "golden_expectations.csv must exist"

def test_csv_field_counts_and_headers():
    with open(cases_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert len(header) == 11, f"Expected 11 columns in cases header, got {len(header)}"
        assert header[0] == "test_id"
        cases_rows = list(reader)
        for i, row in enumerate(cases_rows, start=2):
            assert len(row) == 11, f"Row {i} in golden_cases.csv has {len(row)} fields instead of 11"

    with open(exp_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert len(header) == 9, f"Expected 9 columns in expectations header, got {len(header)}"
        assert header[0] == "test_id"
        exp_rows = list(reader)
        for i, row in enumerate(exp_rows, start=2):
            assert len(row) == 9, f"Row {i} in golden_expectations.csv has {len(row)} fields instead of 9"

def test_matching_test_ids():
    with open(cases_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        case_ids = [r["test_id"] for r in reader]

    with open(exp_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        exp_ids = [r["test_id"] for r in reader]

    assert case_ids == exp_ids, "Test IDs must be identical and in matching order"
    assert len(case_ids) == len(set(case_ids)), "Test IDs must be unique"

def test_near_zero_mathematical_correction():
    """Verify T-Z02 (false exceedance) and T-Z03 (genuine exceedance)"""
    with open(cases_path, "r", encoding="utf-8") as f:
        cases = {r["test_id"]: r for r in csv.DictReader(f)}
    with open(exp_path, "r", encoding="utf-8") as f:
        exps = {r["test_id"]: r for r in csv.DictReader(f)}

    # T-Z02: PAT_pre=100, PAT_post=100.0000000001, k=0.05
    # |delta| = 1e-10, k * |PAT_pre| = 5.0 -> delta < threshold -> y=0
    c_z02 = cases["T-Z02"]
    e_z02 = exps["T-Z02"]
    pat_pre = float(c_z02["pat_pre"])
    pat_post = float(c_z02["pat_post"])
    k = float(c_z02["k"])
    delta = abs(pat_post - pat_pre)
    assert delta < k * abs(pat_pre), "T-Z02 delta must be strictly below threshold"
    assert e_z02["y_k"] == "0", "T-Z02 expected y_k must be 0 (corrected)"
    assert e_z02["is_zero_adj"] == "False", "T-Z02 delta is 1e-10 != 0, so is_zero_adj is False"

    # T-Z03: PAT_pre=1e-9, PAT_post=1.1e-9, k=0.05
    # |delta| = 1e-10, k * |PAT_pre| = 5e-11 -> delta >= threshold -> y=1
    c_z03 = cases["T-Z03"]
    e_z03 = exps["T-Z03"]
    pat_pre_3 = float(c_z03["pat_pre"])
    pat_post_3 = float(c_z03["pat_post"])
    k_3 = float(c_z03["k"])
    delta_3 = abs(pat_post_3 - pat_pre_3)
    assert delta_3 >= k_3 * abs(pat_pre_3), "T-Z03 delta must exceed threshold"
    assert e_z03["y_k"] == "1", "T-Z03 expected y_k must be 1"
    assert e_z03["eligible_label"] == "True", "T-Z03 must be eligible"
    assert e_z03["is_zero_adj"] == "False", "T-Z03 is nonzero adjustment"

def test_asymmetric_post_denominator_cuts():
    """Independent algebraic check of post-denominator asymmetric cuts"""
    # For positive PAT_pre=100, k=0.05:
    # Lower cut: delta <= -k / (1 + k) * PAT_pre = -4.761904761904762
    # Upper cut: delta >= k / (1 - k) * PAT_pre = +5.263157894736842
    # Check: at delta = 5.263157894736842:
    # PAT_post = 105.263157894736842
    # delta / PAT_post = 5.263157894736842 / 105.263157894736842 = 0.05 exactly!
    upper_pos = 100 * 0.05 / 0.95
    assert math.isclose(upper_pos / (100 + upper_pos), 0.05, rel_tol=1e-12)

    # Check lower cut: delta = -4.761904761904762:
    # PAT_post = 95.23809523809524
    # |delta| / PAT_post = 4.761904761904762 / 95.23809523809524 = 0.05 exactly!
    lower_pos = -100 * 0.05 / 1.05
    assert math.isclose(abs(lower_pos) / (100 + lower_pos), 0.05, rel_tol=1e-12)

    # For negative PAT_pre = -100, k = 0.05:
    # Lower cut: delta <= -100 * 0.05 / 0.95 = -5.263157894736842
    # At lower cut: PAT_post = -105.263157894736842
    # |delta| / |PAT_post| = 5.263157894736842 / 105.263157894736842 = 0.05 exactly!
    lower_neg = -100 * 0.05 / 0.95
    assert math.isclose(abs(lower_neg) / abs(-100 + lower_neg), 0.05, rel_tol=1e-12)

    # Upper cut: delta >= 100 * 0.05 / 1.05 = +4.761904761904762
    # At upper cut: PAT_post = -95.23809523809524
    # |delta| / |PAT_post| = 4.761904761904762 / 95.23809523809524 = 0.05 exactly!
    upper_neg = 100 * 0.05 / 1.05
    assert math.isclose(abs(upper_neg) / abs(-100 + upper_neg), 0.05, rel_tol=1e-12)
