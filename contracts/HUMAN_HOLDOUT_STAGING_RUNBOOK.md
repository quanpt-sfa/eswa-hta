# Human Holdout Staging Runbook

**Document ID:** `contracts/HUMAN_HOLDOUT_STAGING_RUNBOOK.md`  
**Target Audience:** Trusted Human Data Steward / Principal Investigator  
**Subject:** Physical Separation and Holdout Isolation for FY2023–2025 Test Set  
**Governance Specification:** `docs/HTA_IMPLEMENTATION_SPEC_v1.2.md` (§4.3, Rules A5, A10, H8)  
**Status:** PREPARATION ARTIFACT (Does not constitute proof that isolation has passed)

---

## 1. Objective and Regulatory Context

Under Specification v1.2 §4.3 and Rule A5:
> "The final-test outcomes (FY2023–2025) must be physically separated before the coding agent starts G1. Pre-G10 code may read only development outcomes and test features, never full raw source copies or any replica containing protected test targets... Merely placing a sentinel next to a readable Parquet file is **not** isolation."

The research integrity of the out-of-time holdout (FY2023–2025) requires that an autonomous coding agent operating in `eswa-hta` cannot access, inspect, aggregate, or infer protected post-audit outcomes prior to Gate G10 unlock.

---

## 2. The Local Access Dilemma and Required Isolation Architecture

### The Threat Model
The autonomous coding agent runs in a local shell on Windows under user account `quanp`. Because this same OS account possesses full NTFS read permissions across all directories on drive `D:\`, the agent could theoretically read:
`d:\Works\dissertation\dissertation-v4\data\source\vn_pipeline_final_firm_year_2015_2025.parquet`
or any other historical research copy containing FY2023–2025 audit adjustments.

Therefore, **simply keeping `data/locked/` empty in the `eswa-hta` repo does NOT provide isolation**. Effective isolation requires preventing the agent's execution process from reading any unredacted full-period dataset anywhere on the host system.

### Recommended Isolation Architectures (Choose One)

#### Architecture A: Encrypted Offline Vault / Air-Gapped Storage (Recommended for simplicity)
1. The human data steward moves all unredacted all-years datasets (including dissertation source files) into an encrypted container (e.g., VeraCrypt volume or password-protected 7-Zip archive) or an offline external storage drive.
2. The encrypted container/external drive is unmounted/disconnected during all agent coding sessions.
3. Only the steward-split development file (`canonical_dev_source.parquet`) and test feature file (`test_features_candidate.parquet`) remain on the active filesystem.

#### Architecture B: Scoped Docker / DevContainer Sandbox
1. The coding agent executes exclusively inside a Docker container or WSL2 distribution.
2. Only the `eswa-hta` repository directory is bind-mounted into the container (`-v "d:\Works\Journals\1. Elsevier\ESWA - Experts system with applications\eswa-hta:/workspace"`).
3. The container runtime has zero network/filesystem mounts to `d:\Works\dissertation\` or any other host folder.

#### Architecture C: Windows NTFS ACL Hardening
1. Create a dedicated low-privilege Windows user (e.g., `eswa-runner`).
2. Run the IDE and agent under `eswa-runner`.
3. Apply explicit Deny ACLs to `d:\Works\dissertation` and all other non-ESWA directories for `eswa-runner`.

---

## 3. Human Steward Staging Procedure (Step-by-Step)

The trusted human data steward must execute the following operations independently of the coding agent:

### Step 1: Extract Authorized Development Dataset (FY2015–2022)
From the master financial dataset, extract rows where `fiscal_year <= 2022`:
- **Destination:** `eswa-hta/data/source/canonical_dev_source.parquet`
- **Permitted Fields:** Firm identifiers, fiscal year, statement scope, `pat_pre`, `pat_post`, `total_assets_pre`, and all pre-audit financial statement items.
- **Strict Boundary:** Must contain zero rows for fiscal years 2023, 2024, or 2025.

### Step 2: Extract Candidate Test Features (FY2023–2025)
From the master financial dataset, extract rows where `fiscal_year >= 2023`:
- **Destination:** `eswa-hta/data/source/test_features_candidate.parquet`
- **Permitted Fields:** Firm identifiers, fiscal year, statement scope, `pat_pre`, `total_assets_pre`, and verified point-in-time predictors ($X_1$).
- **FORBIDDEN FIELDS (Must be completely omitted/redacted):**
  - `pat_post`
  - `delta` ($\Delta$)
  - `D` ($\Delta / \text{assets}$)
  - `is_zero_adj`
  - `eligible_label`
  - All binary labels $y_{k050}, y_{k075}, y_{k100}, y_{k150}, y_{k200}$
  - Any post-audit disclosure dates, auditor adjustment notes, or future financial ratios.

### Step 3: Precompute Permitted Historical Lag Features
For test years 2023–2025, lagged audit information (e.g., FY2022 audited outcomes used for FY2023 predictions) is allowed **only if** the audit publication date preceded the prediction cutoff date.
- The human steward precomputes these approved historical lag features and merges them into `test_features_candidate.parquet`.
- If publication dates are unverified, historical lag features must be left as `NaN` under the strict primary feature policy (§4.2, D08–D09).

### Step 4: Seal Test Outcomes (FY2023–2025)
- Extract the true target columns for FY2023–2025 matching `row_id`, `firm_id`, `fiscal_year`, and `scope`.
- Save as `test_outcomes_sealed.parquet`.
- Store this file in the secure vault / external storage created in Section 2.
- **Do not place this file in the workspace or `data/locked/` before G10.**

### Step 5: Checksum Manifest Generation
Calculate SHA-256 hashes of the staged files:
```powershell
Get-FileHash -Algorithm SHA256 eswa-hta\data\source\canonical_dev_source.parquet
Get-FileHash -Algorithm SHA256 eswa-hta\data\source\test_features_candidate.parquet
```
Record the resulting checksums in `eswa-hta/contracts/input_manifest.sha256`.

### Step 6: Complete the Staging Receipt
Fill out the human receipt at `contracts/staged_holdout_receipt.md` using the template `contracts/staged_holdout_receipt.template.md`.

---

## 4. Verification of Access Denial (Audit Step)

Before authorizing Gate G1, the steward must execute a non-sensitive permissions check under the agent's identity:

```powershell
# Attempt to access the protected raw dissertation source
Test-Path "d:\Works\dissertation\dissertation-v4\data\source\vn_pipeline_final_firm_year_2015_2025.parquet"
```
**Expected Outcome:** `False` or `PermissionDenied`. If this returns `True`, the isolation protocol is incomplete and G0 cannot be signed off.
