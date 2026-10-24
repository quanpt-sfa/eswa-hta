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
import os
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
    "y_k200",
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
    overwrite: bool = False,
) -> Tuple[str, str, str]:
    """
    Executes physical holdout separation and writes verification metadata.
    Enforces master dataset collision protection, path distinctness,
    protected vault separation, atomic writes, positive selection,
    primary-key uniqueness, and fiscal-year bounds.
    """
    if not master_path.exists():
        raise FileNotFoundError(f"Master dataset not found: {master_path}")

    # =========================================================================
    # D1. Validate Paths Before Any Read or Write
    # =========================================================================
    p_master = master_path.resolve()
    p_dev = output_dev_path.resolve()
    p_test = output_test_features_path.resolve()
    p_sealed = output_sealed_outcomes_path.resolve()
    p_receipt = receipt_path.resolve()
    p_manifest = manifest_path.resolve()

    output_map = {
        "--out-dev": p_dev,
        "--out-test-features": p_test,
        "--out-sealed": p_sealed,
        "--receipt": p_receipt,
        "--manifest": p_manifest,
    }

    # 1. No output path may equal the master path
    for name, p in output_map.items():
        if p == p_master:
            raise ValueError(f"Path collision: {name} path ({p}) equals master input dataset path ({p_master})")

    # 2. No two output files may resolve to the same destination
    items = list(output_map.items())
    for i in range(len(items)):
        name1, p1 = items[i]
        for j in range(i + 1, len(items)):
            name2, p2 = items[j]
            if p1 == p2:
                raise ValueError(
                    f"Output path collision: '{name1}' and '{name2}' resolve to the same path ({p1}); "
                    "output paths must all be distinct"
                )

    # 3. Check physical file identity via samefile where files exist
    for name, p in output_map.items():
        if p.exists() and p_master.exists():
            try:
                if p.samefile(p_master):
                    raise ValueError(f"Physical path collision: {name} ({p}) physically resolves to master dataset ({p_master})")
            except (OSError, ValueError) as e:
                if "Physical path collision" in str(e):
                    raise

    for i in range(len(items)):
        name1, p1 = items[i]
        for j in range(i + 1, len(items)):
            name2, p2 = items[j]
            if p1.exists() and p2.exists():
                try:
                    if p1.samefile(p2):
                        raise ValueError(
                            f"Physical path collision: '{name1}' ({p1}) and '{name2}' ({p2}) physically resolve to the same file; "
                            "output paths must all be distinct"
                        )
                except (OSError, ValueError) as e:
                    if "Physical path collision" in str(e):
                        raise

    # 4. Check for existing outputs (prevent silent overwrite)
    if not overwrite:
        for name, p in output_map.items():
            if p.exists():
                raise FileExistsError(
                    f"Staging output path '{name}' already exists at {p}. "
                    "Refusing to overwrite existing staged artifact without explicit overwrite/--force authorization."
                )

    # =========================================================================
    # D2. Distinguish Release and Sealed Vault Destinations
    # =========================================================================
    # Sealed outcomes must NOT be placed inside the agent-accessible release directories
    if p_sealed.parent == p_test.parent or p_sealed.parent == p_dev.parent:
        raise ValueError(
            f"Protected vault destination ({p_sealed}) cannot be located inside the release feature directory ({p_test.parent}). "
            "Sealed outcomes must be stored in an isolated protected vault."
        )

    try:
        if p_sealed.is_relative_to(p_test.parent) or p_sealed.is_relative_to(p_dev.parent):
            raise ValueError(
                f"Protected vault destination ({p_sealed}) cannot be a subdirectory of the release directory. "
                "Sealed outcomes must be stored in an isolated protected vault."
            )
    except AttributeError:
        # Python < 3.9 fallback
        pass

    # Verify master dataset integrity before read
    master_sha_before = compute_sha256(p_master)

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

    # =========================================================================
    # D3. Atomic Write Safety via Staging Temp Files
    # =========================================================================
    pid = os.getpid()
    tmp_dev = p_dev.with_name(f".tmp.{p_dev.name}.{pid}")
    tmp_test = p_test.with_name(f".tmp.{p_test.name}.{pid}")
    tmp_sealed = p_sealed.with_name(f".tmp.{p_sealed.name}.{pid}")
    tmp_manifest = p_manifest.with_name(f".tmp.{p_manifest.name}.{pid}")
    tmp_receipt = p_receipt.with_name(f".tmp.{p_receipt.name}.{pid}")

    created_tmps = []
    try:
        # Development Split (FY <= dev_max_year)
        dev_df = df[df["fiscal_year"] <= dev_max_year].copy()
        tmp_dev.parent.mkdir(parents=True, exist_ok=True)
        dev_df.to_parquet(tmp_dev, index=False)
        created_tmps.append(tmp_dev)
        dev_hash = compute_sha256(tmp_dev)

        # Test Split (FY >= test_min_year)
        test_raw = df[df["fiscal_year"] >= test_min_year].copy()

        # Sealed Outcomes (FY >= test_min_year)
        outcome_cols = [c for c in test_raw.columns if c in REQUIRED_KEYS or c in FORBIDDEN_OUTCOME_COLUMNS or c in REQUIRED_TARGET_INPUTS]
        sealed_df = test_raw[outcome_cols].copy()
        tmp_sealed.parent.mkdir(parents=True, exist_ok=True)
        sealed_df.to_parquet(tmp_sealed, index=False)
        created_tmps.append(tmp_sealed)
        sealed_hash = compute_sha256(tmp_sealed)

        # Sanitized Test Features (Strict positive selection from allowlist)
        test_features_df = test_raw[allowed].copy()
        violations = audit_test_features_sanitization(test_features_df)
        if violations:
            raise ValueError(f"CRITICAL LEAKAGE: Sanitized test features contain forbidden outcome columns: {violations}")

        tmp_test.parent.mkdir(parents=True, exist_ok=True)
        test_features_df.to_parquet(tmp_test, index=False)
        created_tmps.append(tmp_test)
        test_features_hash = compute_sha256(tmp_test)

        # Write Checksum Manifest
        tmp_manifest.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_manifest, "w", encoding="utf-8") as f:
            f.write(f"{dev_hash}  {output_dev_path.name}\n")
            f.write(f"{test_features_hash}  {output_test_features_path.name}\n")
        created_tmps.append(tmp_manifest)

        # Generate Staging Receipt
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
        tmp_receipt.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_receipt, "w", encoding="utf-8") as f:
            f.write(receipt_content)
        created_tmps.append(tmp_receipt)

        # Controlled Finalization: Atomic file rename
        os.replace(tmp_dev, p_dev)
        os.replace(tmp_test, p_test)
        os.replace(tmp_sealed, p_sealed)
        os.replace(tmp_manifest, p_manifest)
        os.replace(tmp_receipt, p_receipt)

    except Exception:
        # Cleanup temporary files on failure
        for tmp_f in created_tmps:
            try:
                if tmp_f.exists():
                    tmp_f.unlink()
            except Exception:
                pass
        raise

    # Verify master dataset remains byte-identical
    master_sha_after = compute_sha256(p_master)
    if master_sha_after != master_sha_before:
        raise RuntimeError(
            f"CRITICAL ERROR: Master dataset SHA-256 changed during staging! "
            f"Before: {master_sha_before}, After: {master_sha_after}"
        )

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
    parser.add_argument("--force", action="store_true",
                        help="Explicitly overwrite existing staging outputs")

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
            overwrite=args.force,
        )
        print("Holdout staging completed successfully.")
    except Exception as e:
        print(f"STAGING FAILED: {e}", file=sys.stderr)
        sys.exit(1)
