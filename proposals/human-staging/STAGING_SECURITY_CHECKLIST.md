# Data Steward Staging Security Checklist

**Document:** `proposals/human-staging/STAGING_SECURITY_CHECKLIST.md`  
**Audience:** Trusted Human Data Steward  
**Governance:** `docs/HTA_IMPLEMENTATION_SPEC_v1.2.md` (§4.3, Rules A5, H8)  

---

## Pre-Staging Checklist

- [ ] **Master Dataset Integrity:** Verify the master financial statement panel source version and SHA-256 hash.
- [ ] **Agent Isolation Boundary Active:** Confirm that the agent execution environment (IDE, terminal, or background process) cannot read master or sealed files.
  - If using Architecture A (Vault): Encrypted volume unmounted or offline USB detached.
  - If using Architecture B (Container): Container mount strictly scoped to `eswa-hta` with no host bypass.
  - If using Architecture C (OS ACLs): NTFS Deny ACL applied to dissertation/master folders for the agent user.

## Staging Execution Checklist

- [ ] Run `test_staging.py` on synthetic data to confirm script operation (`pytest proposals/human-staging/test_staging.py`).
- [ ] Execute `stage_data.py` on the master dataset.
- [ ] Inspect generated `data/source/canonical_dev_source.parquet`:
  - [ ] Covers exactly fiscal years 2015 to 2022.
  - [ ] Contains pre-audit and post-audit earnings.
- [ ] Inspect generated `data/source/test_features_candidate.parquet`:
  - [ ] Covers fiscal years 2023 to 2025.
  - [ ] Verified complete absence of `pat_post`, `delta`, `D`, `is_zero_adj`, and all `y_k*` columns.
  - [ ] Historical lag features verified against point-in-time disclosure dates.
- [ ] Confirm sealed outcomes file `test_outcomes_sealed.parquet` is stored in the secure vault outside the repository.
- [ ] Commit generated checksums into `contracts/input_manifest.sha256`.
- [ ] Sign and commit `contracts/staged_holdout_receipt.md`.

## Access-Denial Verification Checklist

- [ ] Run test read under agent identity:
  ```powershell
  # Must return False or PermissionDenied
  Test-Path "d:\Works\dissertation\dissertation-v4\data\source\vn_pipeline_final_firm_year_2015_2025.parquet"
  ```
- [ ] Authorize progression to Gate G1 only after all checklist items are verified.
