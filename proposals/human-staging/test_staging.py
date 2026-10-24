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
    verify_agent_access_denial,
    compute_sha256,
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
        test_min_year=2023,
        approved_test_feature_columns=[
            "firm_id", "fiscal_year", "scope", "pat_pre", "total_assets_pre",
            "feature_sales_growth", "feature_leverage",
        ],
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

    # 6. Check receipt structure
    receipt_text = receipt.read_text(encoding="utf-8")
    assert "STAGING_OUTPUTS_CREATED__ACCESS_ISOLATION_UNVERIFIED" in receipt_text
    assert "Positive-Selection Feature Allowlist" in receipt_text


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


def test_staging_rejects_missing_human_allowlist(synthetic_master_dataset):
    master, folder = synthetic_master_dataset
    with pytest.raises(ValueError, match="allowlist"):
        stage_holdout_data(
            master_path=master,
            output_dev_path=folder / "out" / "dev.parquet",
            output_test_features_path=folder / "out" / "test.parquet",
            output_sealed_outcomes_path=folder / "vault" / "sealed.parquet",
            receipt_path=folder / "receipt.md",
            manifest_path=folder / "manifest.sha256",
        )
    assert not (folder / "out" / "test.parquet").exists()


def test_staging_rejects_source_specific_audited_columns(synthetic_master_dataset):
    master, folder = synthetic_master_dataset
    df = pd.read_parquet(master)
    df["fs_aud_profit_after_tax"] = df["pat_post"]
    df["audit_adj_revenue"] = 123.0
    df.to_parquet(master, index=False)
    assert set(audit_test_features_sanitization(df)) >= {
        "fs_aud_profit_after_tax", "audit_adj_revenue"
    }
    with pytest.raises(ValueError, match="Forbidden"):
        stage_holdout_data(
            master_path=master,
            output_dev_path=folder / "out" / "dev.parquet",
            output_test_features_path=folder / "out" / "test.parquet",
            output_sealed_outcomes_path=folder / "vault" / "sealed.parquet",
            receipt_path=folder / "receipt.md",
            manifest_path=folder / "manifest.sha256",
            approved_test_feature_columns=[
                "firm_id", "fiscal_year", "scope", "pat_pre", "total_assets_pre",
                "fs_aud_profit_after_tax",
            ],
        )
    assert not (folder / "out" / "test.parquet").exists()


def test_staging_rejects_allowlist_with_pat_post_or_audit_adj(synthetic_master_dataset):
    master, folder = synthetic_master_dataset
    with pytest.raises(ValueError, match="Forbidden"):
        stage_holdout_data(
            master_path=master,
            output_dev_path=folder / "out" / "dev.parquet",
            output_test_features_path=folder / "out" / "test.parquet",
            output_sealed_outcomes_path=folder / "vault" / "sealed.parquet",
            receipt_path=folder / "receipt.md",
            manifest_path=folder / "manifest.sha256",
            approved_test_feature_columns=[
                "firm_id", "fiscal_year", "scope", "pat_pre", "total_assets_pre",
                "pat_post",
            ],
        )


def test_staging_rejects_missing_required_identifiers_in_allowlist(synthetic_master_dataset):
    master, folder = synthetic_master_dataset
    with pytest.raises(ValueError, match="required keys"):
        stage_holdout_data(
            master_path=master,
            output_dev_path=folder / "out" / "dev.parquet",
            output_test_features_path=folder / "out" / "test.parquet",
            output_sealed_outcomes_path=folder / "vault" / "sealed.parquet",
            receipt_path=folder / "receipt.md",
            manifest_path=folder / "manifest.sha256",
            approved_test_feature_columns=[
                "firm_id", "fiscal_year", "feature_sales_growth"  # Missing 'scope'
            ],
        )


def test_staging_rejects_duplicate_primary_keys(synthetic_master_dataset):
    master, folder = synthetic_master_dataset
    df = pd.read_parquet(master)
    # Duplicate first row
    dup_df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    dup_file = folder / "dup_master.parquet"
    dup_df.to_parquet(dup_file, index=False)

    with pytest.raises(ValueError, match="Duplicate primary key"):
        stage_holdout_data(
            master_path=dup_file,
            output_dev_path=folder / "out" / "dev.parquet",
            output_test_features_path=folder / "out" / "test.parquet",
            output_sealed_outcomes_path=folder / "vault" / "sealed.parquet",
            receipt_path=folder / "receipt.md",
            manifest_path=folder / "manifest.sha256",
            approved_test_feature_columns=[
                "firm_id", "fiscal_year", "scope", "pat_pre", "total_assets_pre"
            ],
        )


def test_staging_rejects_unexpected_fiscal_years(synthetic_master_dataset):
    master, folder = synthetic_master_dataset
    df = pd.read_parquet(master)
    df.loc[0, "fiscal_year"] = 2010  # Pre-2015 out of bounds
    out_of_bounds_file = folder / "out_of_bounds.parquet"
    df.to_parquet(out_of_bounds_file, index=False)

    with pytest.raises(ValueError, match="Unexpected fiscal year range"):
        stage_holdout_data(
            master_path=out_of_bounds_file,
            output_dev_path=folder / "out" / "dev.parquet",
            output_test_features_path=folder / "out" / "test.parquet",
            output_sealed_outcomes_path=folder / "vault" / "sealed.parquet",
            receipt_path=folder / "receipt.md",
            manifest_path=folder / "manifest.sha256",
            approved_test_feature_columns=[
                "firm_id", "fiscal_year", "scope", "pat_pre", "total_assets_pre"
            ],
        )


def test_staging_rejects_overlapping_output_paths(synthetic_master_dataset):
    master, folder = synthetic_master_dataset
    shared_path = folder / "out" / "same_file.parquet"
    with pytest.raises(ValueError, match="must all be distinct"):
        stage_holdout_data(
            master_path=master,
            output_dev_path=folder / "out" / "dev.parquet",
            output_test_features_path=shared_path,
            output_sealed_outcomes_path=shared_path,  # OVERLAP
            receipt_path=folder / "receipt.md",
            manifest_path=folder / "manifest.sha256",
            approved_test_feature_columns=[
                "firm_id", "fiscal_year", "scope", "pat_pre", "total_assets_pre"
            ],
        )


def test_staging_rejects_missing_required_target_inputs(synthetic_master_dataset):
    master, folder = synthetic_master_dataset
    df = pd.read_parquet(master)
    df_missing = df.drop(columns=["pat_post"])
    missing_file = folder / "missing_target.parquet"
    df_missing.to_parquet(missing_file, index=False)

    with pytest.raises(KeyError, match="Required target input 'pat_post' missing"):
        stage_holdout_data(
            master_path=missing_file,
            output_dev_path=folder / "out" / "dev.parquet",
            output_test_features_path=folder / "out" / "test.parquet",
            output_sealed_outcomes_path=folder / "vault" / "sealed.parquet",
            receipt_path=folder / "receipt.md",
            manifest_path=folder / "manifest.sha256",
            approved_test_feature_columns=[
                "firm_id", "fiscal_year", "scope", "pat_pre", "total_assets_pre"
            ],
        )


# ==============================================================================
# Adversarial Security Tests: Master Dataset Protection & Write Safety (D4.1 - D4.12)
# ==============================================================================

def test_staging_adversarial_out_dev_equals_master(synthetic_master_dataset):
    master, folder = synthetic_master_dataset
    sha_before = compute_sha256(master)

    with pytest.raises(ValueError, match="equals master input dataset path"):
        stage_holdout_data(
            master_path=master,
            output_dev_path=master,  # COLLISION WITH MASTER
            output_test_features_path=folder / "out" / "test.parquet",
            output_sealed_outcomes_path=folder / "vault" / "sealed.parquet",
            receipt_path=folder / "receipt.md",
            manifest_path=folder / "manifest.sha256",
            approved_test_feature_columns=["firm_id", "fiscal_year", "scope", "pat_pre", "total_assets_pre"],
        )

    # Master file must remain byte-identical
    assert compute_sha256(master) == sha_before


def test_staging_adversarial_out_test_features_equals_master(synthetic_master_dataset):
    master, folder = synthetic_master_dataset
    sha_before = compute_sha256(master)

    with pytest.raises(ValueError, match="equals master input dataset path"):
        stage_holdout_data(
            master_path=master,
            output_dev_path=folder / "out" / "dev.parquet",
            output_test_features_path=master,  # COLLISION WITH MASTER
            output_sealed_outcomes_path=folder / "vault" / "sealed.parquet",
            receipt_path=folder / "receipt.md",
            manifest_path=folder / "manifest.sha256",
            approved_test_feature_columns=["firm_id", "fiscal_year", "scope", "pat_pre", "total_assets_pre"],
        )

    assert compute_sha256(master) == sha_before


def test_staging_adversarial_out_sealed_equals_master(synthetic_master_dataset):
    master, folder = synthetic_master_dataset
    sha_before = compute_sha256(master)

    with pytest.raises(ValueError, match="equals master input dataset path"):
        stage_holdout_data(
            master_path=master,
            output_dev_path=folder / "out" / "dev.parquet",
            output_test_features_path=folder / "out" / "test.parquet",
            output_sealed_outcomes_path=master,  # COLLISION WITH MASTER
            receipt_path=folder / "receipt.md",
            manifest_path=folder / "manifest.sha256",
            approved_test_feature_columns=["firm_id", "fiscal_year", "scope", "pat_pre", "total_assets_pre"],
        )

    assert compute_sha256(master) == sha_before


def test_staging_adversarial_receipt_equals_master(synthetic_master_dataset):
    master, folder = synthetic_master_dataset
    sha_before = compute_sha256(master)

    with pytest.raises(ValueError, match="equals master input dataset path"):
        stage_holdout_data(
            master_path=master,
            output_dev_path=folder / "out" / "dev.parquet",
            output_test_features_path=folder / "out" / "test.parquet",
            output_sealed_outcomes_path=folder / "vault" / "sealed.parquet",
            receipt_path=master,  # COLLISION WITH MASTER
            manifest_path=folder / "manifest.sha256",
            approved_test_feature_columns=["firm_id", "fiscal_year", "scope", "pat_pre", "total_assets_pre"],
        )

    assert compute_sha256(master) == sha_before


def test_staging_adversarial_manifest_equals_master(synthetic_master_dataset):
    master, folder = synthetic_master_dataset
    sha_before = compute_sha256(master)

    with pytest.raises(ValueError, match="equals master input dataset path"):
        stage_holdout_data(
            master_path=master,
            output_dev_path=folder / "out" / "dev.parquet",
            output_test_features_path=folder / "out" / "test.parquet",
            output_sealed_outcomes_path=folder / "vault" / "sealed.parquet",
            receipt_path=folder / "receipt.md",
            manifest_path=master,  # COLLISION WITH MASTER
            approved_test_feature_columns=["firm_id", "fiscal_year", "scope", "pat_pre", "total_assets_pre"],
        )

    assert compute_sha256(master) == sha_before


def test_staging_adversarial_distinct_path_strings_resolving_to_same_destination(synthetic_master_dataset):
    master, folder = synthetic_master_dataset
    sha_before = compute_sha256(master)

    p1 = folder / "out" / "dev.parquet"
    # p2 uses dot-segments to resolve to the exact same file
    p2 = folder / "out" / "subdir" / ".." / "dev.parquet"

    with pytest.raises(ValueError, match="resolve to the same path|must all be distinct"):
        stage_holdout_data(
            master_path=master,
            output_dev_path=p1,
            output_test_features_path=p2,  # RESOLVES TO SAME DESTINATION
            output_sealed_outcomes_path=folder / "vault" / "sealed.parquet",
            receipt_path=folder / "receipt.md",
            manifest_path=folder / "manifest.sha256",
            approved_test_feature_columns=["firm_id", "fiscal_year", "scope", "pat_pre", "total_assets_pre"],
        )

    assert compute_sha256(master) == sha_before


def test_staging_adversarial_existing_output_without_overwrite_force(synthetic_master_dataset):
    master, folder = synthetic_master_dataset
    out_dev = folder / "out" / "dev.parquet"
    out_dev.parent.mkdir(parents=True, exist_ok=True)
    out_dev.write_bytes(b"EXISTING_STAGED_ARTIFACT")

    # Without overwrite=True, staging must raise FileExistsError
    with pytest.raises(FileExistsError, match="already exists"):
        stage_holdout_data(
            master_path=master,
            output_dev_path=out_dev,
            output_test_features_path=folder / "out" / "test.parquet",
            output_sealed_outcomes_path=folder / "vault" / "sealed.parquet",
            receipt_path=folder / "receipt.md",
            manifest_path=folder / "manifest.sha256",
            approved_test_feature_columns=["firm_id", "fiscal_year", "scope", "pat_pre", "total_assets_pre"],
            overwrite=False,
        )

    # Existing file must not have been overwritten
    assert out_dev.read_bytes() == b"EXISTING_STAGED_ARTIFACT"

    # With overwrite=True, staging proceeds
    stage_holdout_data(
        master_path=master,
        output_dev_path=out_dev,
        output_test_features_path=folder / "out" / "test.parquet",
        output_sealed_outcomes_path=folder / "vault" / "sealed.parquet",
        receipt_path=folder / "receipt.md",
        manifest_path=folder / "manifest.sha256",
        approved_test_feature_columns=["firm_id", "fiscal_year", "scope", "pat_pre", "total_assets_pre"],
        overwrite=True,
    )
    assert out_dev.read_bytes() != b"EXISTING_STAGED_ARTIFACT"


def test_staging_adversarial_vault_inside_release_directory(synthetic_master_dataset):
    master, folder = synthetic_master_dataset
    sha_before = compute_sha256(master)

    release_dir = folder / "data" / "source"
    # Placing sealed vault inside the release directory
    bad_vault = release_dir / "sealed_outcomes.parquet"

    with pytest.raises(ValueError, match="Protected vault destination"):
        stage_holdout_data(
            master_path=master,
            output_dev_path=release_dir / "dev.parquet",
            output_test_features_path=release_dir / "test.parquet",
            output_sealed_outcomes_path=bad_vault,  # FORBIDDEN: INSIDE RELEASE DIR
            receipt_path=folder / "receipt.md",
            manifest_path=folder / "manifest.sha256",
            approved_test_feature_columns=["firm_id", "fiscal_year", "scope", "pat_pre", "total_assets_pre"],
        )

    assert compute_sha256(master) == sha_before


def test_staging_failure_occurs_before_any_source_modification_and_leaves_master_intact(synthetic_master_dataset):
    master, folder = synthetic_master_dataset
    sha_before = compute_sha256(master)

    out_dev = folder / "out" / "dev.parquet"
    out_test = folder / "out" / "test.parquet"

    # Trigger failure through invalid fiscal years
    df = pd.read_parquet(master)
    df.loc[0, "fiscal_year"] = 1990
    invalid_master = folder / "invalid_master.parquet"
    df.to_parquet(invalid_master, index=False)
    invalid_sha_before = compute_sha256(invalid_master)

    with pytest.raises(ValueError, match="Unexpected fiscal year range"):
        stage_holdout_data(
            master_path=invalid_master,
            output_dev_path=out_dev,
            output_test_features_path=out_test,
            output_sealed_outcomes_path=folder / "vault" / "sealed.parquet",
            receipt_path=folder / "receipt.md",
            manifest_path=folder / "manifest.sha256",
            approved_test_feature_columns=["firm_id", "fiscal_year", "scope", "pat_pre", "total_assets_pre"],
        )

    # Invariants: master untouched, no output created, no temp files left
    assert compute_sha256(invalid_master) == invalid_sha_before
    assert not out_dev.exists()
    assert not out_test.exists()
    assert len(list(folder.glob(".tmp.*"))) == 0


def test_staging_receipt_does_not_claim_os_isolation_without_evidence(synthetic_master_dataset):
    master, folder = synthetic_master_dataset
    out_dev = folder / "data" / "source" / "canonical_dev_source.parquet"
    out_test = folder / "data" / "source" / "test_features_candidate.parquet"
    out_sealed = folder / "vault" / "test_outcomes_sealed.parquet"
    receipt = folder / "contracts" / "staged_holdout_receipt.md"
    manifest = folder / "contracts" / "input_manifest.sha256"

    stage_holdout_data(
        master_path=master,
        output_dev_path=out_dev,
        output_test_features_path=out_test,
        output_sealed_outcomes_path=out_sealed,
        receipt_path=receipt,
        manifest_path=manifest,
        approved_test_feature_columns=["firm_id", "fiscal_year", "scope", "pat_pre", "total_assets_pre"],
    )

    receipt_text = receipt.read_text(encoding="utf-8")
    # Must NOT claim pass without evidence
    assert "- [ ] 4. OS-level access isolation verified" in receipt_text
    assert "STAGING_OUTPUTS_CREATED__ACCESS_ISOLATION_UNVERIFIED" in receipt_text
    assert "Human sign-off required after independent OS-level access-denial probe" in receipt_text

    # Probe on readable file returns False (access NOT denied)
    assert verify_agent_access_denial(out_sealed) is False

