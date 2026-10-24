"""
Human-Only Holdout Staging Tool
================================
Governance: docs/HTA_IMPLEMENTATION_SPEC_v1.2.md (§4.3, Rules A5, H8)
Status: HUMAN_STEWARD_TOOL (NOT FOR AUTONOMOUS AGENT RUNTIME EXECUTION)

This script is designed to be executed solely by a trusted human data steward
to physically split master financial statement data into:
  1. Development source (FY2015-2022) with development outcomes.
  2. Candidate test features (FY2023-2025) constructed strictly via positive selection
     from a human-approved feature allowlist.
  3. Sealed test outcomes (FY2023-2025) stored in secure, agent-inaccessible storage.
  4. Cryptographic SHA-256 manifests and human staging receipts distinguishing
     file creation from OS-level access isolation.
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
REQUIRED_TARGET_INPUTS = ["pat_pre", "pat_post", "total_assets_pre"]


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def audit_test_features_sanitization(df: pd.DataFrame) -> List[str]:
    """Fail-closed vocabulary scan; explicit human allowlist is also required."""
    found = []
    forbidden_exact = {name.lower() for name in FORBIDDEN_OUTCOME_COLUMNS}
    forbidden_prefixes = (
        "fs_aud_", "audit_adj_", "audit_adjustment_", "delta",
        "y_k", "post_audit_", "audited_",
    )
    forbidden_suffixes = ("_post", "_aud", "_audited")
    for col in df.columns:
        name = str(col).lower()
        if (name in forbidden_exact or name.startswith(forbidden_prefixes)
                or name.endswith(forbidden_suffixes)):
            found.append(col)
    return found


def verify_agent_access_denial(protected_path: Path) -> bool:
    """
    Verifies that the coding agent's execution identity cannot read the protected path.
    Returns True ONLY if reading raises PermissionError.
    Returns False if readable or if an unexpected state occurs.
    """
    try:
        with open(protected_path, "rb") as f:
            f.read(1)
        return False  # Isolation breach: file was readable!
    except PermissionError:
        return True   # OS-level access denied
    except Exception:
        return False


def stage_holdout_data(
    master_path: Path,
    output_dev_path: Path,
    output_test_features_path: Path,
    output_sealed_outcomes_path: Path,
    receipt_path: Path,
    manifest_path: Path,
    dev_max_year: int = 2022,
    test_min_year: int = 2023,
    approved_test_feature_columns: List[str] = None,
) -> Tuple[str, str, str]:
    """
    Executes physical holdout separation and writes verification metadata.
    Enforces positive selection, primary-key uniqueness, fiscal-year bounds,
    and distinct path requirements.
    """
    if not master_path.exists():
        raise FileNotFoundError(f"Master dataset not found: {master_path}")

    # Load master data
    df = pd.read_parquet(master_path) if master_path.suffix == ".parquet" else pd.read_csv(master_path)

    # 1. Primary Keys Check
    for k in REQUIRED_KEYS:
        if k not in df.columns:
            raise KeyError(f"Required key '{k}' missing from master dataset")

    if df.duplicated(subset=REQUIRED_KEYS).any():
        n_dups = int(df.duplicated(subset=REQUIRED_KEYS).sum())
        raise ValueError(f"Duplicate primary key ({REQUIRED_KEYS}) detected in master dataset: {n_dups} duplicate rows")

    # 2. Required Target Inputs in Master Check
    for col in REQUIRED_TARGET_INPUTS:
        if col not in df.columns:
            raise KeyError(f"Required target input '{col}' missing from master dataset")

    # 3. Fiscal Year Bounds Check
    min_year = int(df["fiscal_year"].min())
    max_year = int(df["fiscal_year"].max())
    if min_year < 2015 or max_year > 2025:
        raise ValueError(f"Unexpected fiscal year range in master dataset: [{min_year}, {max_year}], expected within [2015, 2025]")

    # 4. Human-Approved Allowlist Check (Positive Selection)
    if not approved_test_feature_columns:
        raise ValueError("Human-approved test-feature column allowlist is required")
    allowed = list(approved_test_feature_columns)
    if len(set(allowed)) != len(allowed):
        raise ValueError("Duplicate entries in approved feature allowlist")
    if not set(REQUIRED_KEYS).issubset(allowed):
        raise ValueError("Approved allowlist must include all required keys")
    missing = sorted(set(allowed) - set(df.columns))
    if missing:
        raise ValueError(f"Approved columns absent from source: {missing}")

    forbidden = audit_test_features_sanitization(pd.DataFrame(columns=allowed))
    if forbidden:
        raise ValueError(f"Forbidden output columns in approved allowlist: {forbidden}")

    # 5. Path Distinctness Checks
    p_dev = output_dev_path.resolve()
    p_test = output_test_features_path.resolve()
    p_sealed = output_sealed_outcomes_path.resolve()
    if p_sealed == p_test or p_sealed == p_dev or p_dev == p_test:
        raise ValueError("Output paths for dev, test features, and sealed outcomes must all be distinct")

    # 6. Development Split (FY <= dev_max_year)
    dev_df = df[df["fiscal_year"] <= dev_max_year].copy()
    output_dev_path.parent.mkdir(parents=True, exist_ok=True)
    dev_df.to_parquet(output_dev_path, index=False)
    dev_hash = compute_sha256(output_dev_path)

    # 7. Test Split (FY >= test_min_year)
    test_raw = df[df["fiscal_year"] >= test_min_year].copy()

    # 8. Sealed Outcomes (FY >= test_min_year)
    outcome_cols = [c for c in test_raw.columns if c in REQUIRED_KEYS or c in FORBIDDEN_OUTCOME_COLUMNS or c in REQUIRED_TARGET_INPUTS]
    sealed_df = test_raw[outcome_cols].copy()
    output_sealed_outcomes_path.parent.mkdir(parents=True, exist_ok=True)
    sealed_df.to_parquet(output_sealed_outcomes_path, index=False)
    sealed_hash = compute_sha256(output_sealed_outcomes_path)

    # 9. Sanitized Test Features (Strict positive selection from allowlist)
    test_features_df = test_raw[allowed].copy()

    # Sanitization assertion
    violations = audit_test_features_sanitization(test_features_df)
    if violations:
        raise ValueError(f"CRITICAL LEAKAGE: Sanitized test features contain forbidden outcome columns: {violations}")

    output_test_features_path.parent.mkdir(parents=True, exist_ok=True)
    test_features_df.to_parquet(output_test_features_path, index=False)
    test_features_hash = compute_sha256(output_test_features_path)

    # 10. Write Checksum Manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(f"{dev_hash}  {output_dev_path.name}\n")
        f.write(f"{test_features_hash}  {output_test_features_path.name}\n")

    # 11. Generate Staging Receipt
    receipt_content = f"""# Staged Holdout Human Receipt
Date: {pd.Timestamp.now().isoformat()}
Steward Identity: Human Data Steward (Automated Tool Generation)
Master Source: {master_path.name}

## Positive-Selection Feature Allowlist
- Human Approval: Verified against explicit allowlist ({len(allowed)} columns)
- Sanitization Scan: Passed (0 forbidden tokens found)

## Verification Checklist
- [x] 1. Files successfully created
- [x] 2. Required schema checks passed
- [x] 3. Outcome columns excluded (Positive selection allowlist enforced)
- [ ] 4. OS-level access isolation verified (Requires independent agent permission-denial probe)
- [ ] 5. Human approval completed (Requires human steward signature)

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

Status: STAGING_OUTPUTS_CREATED__ACCESS_ISOLATION_UNVERIFIED
Human sign-off required after independent OS-level access-denial probe.
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
    parser.add_argument("--allowlist", required=True, type=Path,
                        help="Human-approved UTF-8 text file: one canonical test feature column per line")

    args = parser.parse_args()
    try:
        stage_holdout_data(
            master_path=args.master,
            output_dev_path=args.out_dev,
            output_test_features_path=args.out_test_features,
            output_sealed_outcomes_path=args.out_sealed,
            receipt_path=args.receipt,
            manifest_path=args.manifest,
            approved_test_feature_columns=[
                x.strip() for x in args.allowlist.read_text(encoding="utf-8").splitlines()
                if x.strip() and not x.strip().startswith("#")
            ],
        )
        print("Holdout staging completed successfully.")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
