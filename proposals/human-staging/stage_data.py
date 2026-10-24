"""
Human-Only Holdout Staging Tool
================================
Governance: docs/HTA_IMPLEMENTATION_SPEC_v1.2.md (§4.3, Rules A5, H8)
Status: HUMAN_STEWARD_TOOL (NOT FOR AUTONOMOUS AGENT RUNTIME EXECUTION)

This script is designed to be executed solely by a trusted human data steward
to physically split master financial statement data into:
  1. Development source (FY2015-2022) with development outcomes.
  2. Candidate test features (FY2023-2025) with post-audit outcomes strictly redacted.
  3. Sealed test outcomes (FY2023-2025) stored in secure, agent-inaccessible storage.
  4. Cryptographic SHA-256 manifests and human staging receipts.
"""

import argparse
import hashlib
from pathlib import Path
import sys
from typing import List, Tuple
import pandas as pd


FORBIDDEN_OUTCOME_COLUMNS = [
    "pat_post",
    "lnst_sau_kiem_toan",
    "profit_after_tax_post",
    "delta",
    "D",
    "is_zero_adj",
    "eligible_label",
    "usable_pair",
    "sign_change",
    "boundary_case",
    "y_k050",
    "y_k075",
    "y_k100",
    "y_k150",
    "y_k200"
]

REQUIRED_KEYS = ["firm_id", "fiscal_year", "scope"]


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def audit_test_features_sanitization(df: pd.DataFrame) -> List[str]:
    """Returns any forbidden outcome or target-derived columns found in dataframe."""
    found = []
    cols_lower = [c.lower() for c in df.columns]
    for pattern in FORBIDDEN_OUTCOME_COLUMNS:
        pattern_lower = pattern.lower()
        for col in df.columns:
            if pattern_lower == col.lower() or col.lower().startswith("y_k") or col.lower().startswith("delta"):
                if col not in found:
                    found.append(col)
    return found


def stage_holdout_data(
    master_path: Path,
    output_dev_path: Path,
    output_test_features_path: Path,
    output_sealed_outcomes_path: Path,
    receipt_path: Path,
    manifest_path: Path,
    dev_max_year: int = 2022,
    test_min_year: int = 2023
) -> Tuple[str, str, str]:
    """
    Executes physical holdout separation and writes verification metadata.
    """
    if not master_path.exists():
        raise FileNotFoundError(f"Master dataset not found: {master_path}")

    # Load master data
    df = pd.read_parquet(master_path) if master_path.suffix == ".parquet" else pd.read_csv(master_path)

    for k in REQUIRED_KEYS:
        if k not in df.columns:
            raise KeyError(f"Required key '{k}' missing from master dataset")

    # 1. Development Split (FY <= dev_max_year)
    dev_df = df[df["fiscal_year"] <= dev_max_year].copy()
    output_dev_path.parent.mkdir(parents=True, exist_ok=True)
    dev_df.to_parquet(output_dev_path, index=False)
    dev_hash = compute_sha256(output_dev_path)

    # 2. Test Split (FY >= test_min_year)
    test_raw = df[df["fiscal_year"] >= test_min_year].copy()

    # 3. Sealed Outcomes (FY >= test_min_year)
    outcome_cols = [c for c in test_raw.columns if c in REQUIRED_KEYS or c in FORBIDDEN_OUTCOME_COLUMNS or c in ["pat_pre", "total_assets_pre"]]
    sealed_df = test_raw[outcome_cols].copy()
    output_sealed_outcomes_path.parent.mkdir(parents=True, exist_ok=True)
    sealed_df.to_parquet(output_sealed_outcomes_path, index=False)
    sealed_hash = compute_sha256(output_sealed_outcomes_path)

    # 4. Sanitized Test Features (Remove all outcomes)
    feature_cols = [c for c in test_raw.columns if c not in FORBIDDEN_OUTCOME_COLUMNS and not c.startswith("y_k")]
    test_features_df = test_raw[feature_cols].copy()

    # Redaction assertion
    violations = audit_test_features_sanitization(test_features_df)
    if violations:
        raise ValueError(f"CRITICAL LEAKAGE: Sanitized test features contain forbidden outcome columns: {violations}")

    output_test_features_path.parent.mkdir(parents=True, exist_ok=True)
    test_features_df.to_parquet(output_test_features_path, index=False)
    test_features_hash = compute_sha256(output_test_features_path)

    # 5. Write Checksum Manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(f"{dev_hash}  {output_dev_path.name}\n")
        f.write(f"{test_features_hash}  {output_test_features_path.name}\n")

    # 6. Generate Staging Receipt
    receipt_content = f"""# Staged Holdout Human Receipt
Date: {pd.Timestamp.now().isoformat()}
Steward Identity: Human Data Steward (Automated Tool Generation)
Master Source: {master_path.name}

## Staged Artifacts
1. Development Source:
   - Path: {output_dev_path}
   - Rows: {len(dev_df)}
   - Fiscal Years: {dev_df['fiscal_year'].min()} to {dev_df['fiscal_year'].max()}
   - SHA-256: {dev_hash}

2. Candidate Test Features (Outcomes Redacted):
   - Path: {output_test_features_path}
   - Rows: {len(test_features_df)}
   - Fiscal Years: {test_features_df['fiscal_year'].min()} to {test_features_df['fiscal_year'].max()}
   - Forbidden Columns Found: {len(violations)} (PASSED SANITIZATION)
   - SHA-256: {test_features_hash}

3. Sealed Test Outcomes:
   - Path: {output_sealed_outcomes_path}
   - Rows: {len(sealed_df)}
   - SHA-256: {sealed_hash}

Status: STAGED_AND_VERIFIED
"""
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(receipt_path, "w", encoding="utf-8") as f:
        f.write(receipt_content)

    return dev_hash, test_features_hash, sealed_hash


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Human Data Steward Holdout Staging Tool")
    parser.add_argument("--master", required=True, type=Path, help="Path to unredacted master dataset")
    parser.add_argument("--out-dev", required=True, type=Path, help="Destination for canonical_dev_source.parquet")
    parser.add_argument("--out-test-features", required=True, type=Path, help="Destination for test_features_candidate.parquet")
    parser.add_argument("--out-sealed", required=True, type=Path, help="Destination for sealed test outcomes")
    parser.add_argument("--receipt", required=True, type=Path, help="Destination for staged_holdout_receipt.md")
    parser.add_argument("--manifest", required=True, type=Path, help="Destination for input_manifest.sha256")

    args = parser.parse_args()
    try:
        stage_holdout_data(
            master_path=args.master,
            output_dev_path=args.out_dev,
            output_test_features_path=args.out_test_features,
            output_sealed_outcomes_path=args.out_sealed,
            receipt_path=args.receipt,
            manifest_path=args.manifest
        )
        print("Holdout staging completed successfully.")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
