# Staged Holdout Human Receipt (Template)

**Document:** `contracts/staged_holdout_receipt.md`  
**Governing Specification:** `docs/HTA_IMPLEMENTATION_SPEC_v1.2.md` (§4.3, Rules A5, H8)  
**Status:** DRAFT_TEMPLATE (To be completed and signed by Human Data Steward)

---

## 1. Staging Metadata

| Field | Steward Value |
|---|---|
| **Date of Staging** | `YYYY-MM-DD` |
| **Data Steward Name / Role** | `[Full Name, Affiliation]` |
| **Master Source Version** | `[e.g., vn_pipeline_v4_20261024]` |
| **Isolation Architecture Applied** | `[Architecture A: Vault / Architecture B: Container / Architecture C: ACL]` |
| **Access Denial Confirmed on Replicas** | `[YES / NO — Date verified]` |

---

## 2. Staged File Specifications

### 2.1 Development Source (FY2015–2022)
- **Path:** `data/source/canonical_dev_source.parquet`
- **Fiscal Years Covered:** 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022
- **Row Count:** `[Exact Integer]`
- **Unique Firms:** `[Exact Integer]`
- **Statement Scope:** Consolidated preferred; separate when consolidated unavailable (per H1 rule)
- **Monetary Unit:** `VND` (Scale: `[1 / 1,000 / 1,000,000]`)
- **SHA-256 Checksum:** `[64-character hex hash]`

### 2.2 Test Feature Source (FY2023–2025)
- **Path:** `data/source/test_features_candidate.parquet`
- **Fiscal Years Covered:** 2023, 2024, 2025
- **Row Count:** `[Exact Integer]`
- **Unique Firms:** `[Exact Integer]`
- **Outcome Redaction Confirmed:** `[YES — All post-audit PAT, deltas, labels, and targets omitted]`
- **Historical Lag Feature Release Policy:** `[Strict Point-in-time publication dates verified / NaN fallback]`
- **SHA-256 Checksum:** `[64-character hex hash]`

---

## 3. Sealed Test Outcomes (FY2023–2025)

- **Sealed Storage Location:** `[Secure vault external to agent's readable workspace]`
- **Target Unsealing Date:** Gate G10 (after model lock is signed and frozen)
- **SHA-256 Checksum (Human Records Only):** `[64-character hex hash]`

---

## 4. Human Steward Attestation

I hereby certify that:
1. The development source `canonical_dev_source.parquet` contains exclusively FY2015–2022 observations.
2. The candidate test feature source `test_features_candidate.parquet` contains no post-audit financial targets, labels, or unverified forward-looking data.
3. Protected test outcomes for FY2023–2025 have been physically isolated in secure storage inaccessible to the coding agent.
4. All unredacted local copies of the full dataset have been rendered unreadable to the agent's execution process.

**Signed:** ______________________________________  
**Date:** ________________________
