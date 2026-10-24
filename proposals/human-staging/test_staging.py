"""
Unit Test Suite for Human-Only Holdout Staging Tool
===================================================
Governance: docs/HTA_IMPLEMENTATION_SPEC_v1.2.md (§4.3, Rules A5, H8)
Tests the staging tool exclusively on synthetic toy data in a temporary directory.
"""

import sys
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

staging_dir = Path(__file__).resolve().parent
if str(staging_dir) not in sys.path:
    sys.path.insert(0, str(staging_dir))

from stage_data import (
    stage_holdout_data,
    audit_test_features_sanitization,
    FORBIDDEN_OUTCOME_COLUMNS
)


@pytest.fixture
def synthetic_master_dataset():
    """Generates synthetic panel dataset covering 2015-2025."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        n_firms = 10
        years = list(range(2015, 2026))
        
        rows = []
        for f in range(n_firms):
            for y in years:
                rows.append({
                    "firm_id": f"F{f:03d}",
                    "fiscal_year": y,
                    "scope": "consolidated",
                    "pat_pre": 100.0 + f * 5.0,
                    "pat_post": 105.0 + f * 5.0,
                    "total_assets_pre": 1000.0 + f * 50.0,
                    "delta": 5.0,
                    "D": 0.005,
                    "is_zero_adj": False,
                    "y_k050": 1,
                    "feature_sales_growth": 0.08,
                    "feature_leverage": 0.45
                })
        
        df = pd.DataFrame(rows)
        master_file = tmp_path / "synthetic_master.parquet"
        df.to_parquet(master_file, index=False)
        yield master_file, tmp_path


def test_holdout_staging_execution(synthetic_master_dataset):
    master_file, tmp_path = synthetic_master_dataset

    out_dev = tmp_path / "data" / "source" / "canonical_dev_source.parquet"
    out_test_features = tmp_path / "data" / "source" / "test_features_candidate.parquet"
    out_sealed = tmp_path / "vault" / "test_outcomes_sealed.parquet"
    receipt = tmp_path / "contracts" / "staged_holdout_receipt.md"
    manifest = tmp_path / "contracts" / "input_manifest.sha256"

    dev_hash, test_hash, sealed_hash = stage_holdout_data(
        master_path=master_file,
        output_dev_path=out_dev,
        output_test_features_path=out_test_features,
        output_sealed_outcomes_path=out_sealed,
        receipt_path=receipt,
        manifest_path=manifest,
        dev_max_year=2022,
        test_min_year=2023
    )

    # 1. Assert file creations
    assert out_dev.exists()
    assert out_test_features.exists()
    assert out_sealed.exists()
    assert receipt.exists()
    assert manifest.exists()

    # 2. Check development sample boundaries
    dev_df = pd.read_parquet(out_dev)
    assert dev_df["fiscal_year"].max() == 2022
    assert dev_df["fiscal_year"].min() == 2015
    assert "pat_post" in dev_df.columns

    # 3. Check candidate test features sanitization
    test_feat_df = pd.read_parquet(out_test_features)
    assert test_feat_df["fiscal_year"].min() == 2023
    assert test_feat_df["fiscal_year"].max() == 2025
    
    # Assert zero forbidden outcome columns
    violations = audit_test_features_sanitization(test_feat_df)
    assert len(violations) == 0, f"Violations found: {violations}"
    assert "pat_post" not in test_feat_df.columns
    assert "delta" not in test_feat_df.columns
    assert "y_k050" not in test_feat_df.columns

    # 4. Check sealed test outcomes
    sealed_df = pd.read_parquet(out_sealed)
    assert sealed_df["fiscal_year"].min() == 2023
    assert "pat_post" in sealed_df.columns
    assert len(sealed_df) == len(test_feat_df)

    # 5. Check manifest contents
    manifest_text = manifest.read_text(encoding="utf-8")
    assert dev_hash in manifest_text
    assert test_hash in manifest_text


def test_sanitization_audit_catches_leakage():
    leaky_df = pd.DataFrame({
        "firm_id": ["F001"],
        "fiscal_year": [2023],
        "scope": ["consolidated"],
        "pat_pre": [100.0],
        "pat_post": [105.0],   # FORBIDDEN
        "y_k050": [1]          # FORBIDDEN
    })
    violations = audit_test_features_sanitization(leaky_df)
    assert "pat_post" in violations
    assert "y_k050" in violations
