# Human-Only Holdout Staging Tools

**Directory:** `proposals/human-staging/`  
**Target User:** Trusted Human Data Steward / Principal Investigator  
**Governance Specification:** `docs/HTA_IMPLEMENTATION_SPEC_v1.2.md` (§4.3, Rules A5, H8)  
**Status:** PROPOSAL_TOOL (Pending independent human sign-off)

---

## 1. Security Warning and Purpose

> **CRITICAL RESTRICTION:**  
> The scripts in this directory (`stage_data.py`) are strictly for **human data steward execution**. The autonomous coding agent is strictly prohibited from running this script against real research datasets or accessing master financial statement copies containing FY2023–2025 audit outcomes.

This toolset automates the physical extraction, sanitization, hashing, and receipt generation required before Gate G1:
1. Splits master panel into development source (FY2015–2022) with development outcomes.
2. Generates candidate test features (FY2023–2025) with all post-audit outcomes, deltas, and labels strictly redacted.
3. Packages sealed test outcomes (FY2023–2025) into secure storage outside agent access.
4. Generates SHA-256 cryptographic manifests and staging receipts.

---

## 2. Usage Instructions for Data Steward

### Command Syntax
```bash
python proposals/human-staging/stage_data.py \
  --master "/path/to/unredacted/master_panel.parquet" \
  --out-dev "data/source/canonical_dev_source.parquet" \
  --out-test-features "data/source/test_features_candidate.parquet" \
  --out-sealed "/path/to/secure_vault/test_outcomes_sealed.parquet" \
  --receipt "contracts/staged_holdout_receipt.md" \
  --manifest "contracts/input_manifest.sha256"
```

### Verification Pipeline
Before releasing the staged files to the repository:
1. Run the synthetic verification test suite:
   ```bash
   python -m pytest proposals/human-staging/test_staging.py -v
   ```
2. Verify that the output `test_features_candidate.parquet` contains zero forbidden outcome columns (`pat_post`, `delta`, `y_k*`).
3. Disconnect or unmount the master dataset and sealed vault storage from the agent's environment.

---

## 3. Recommended Execution Environments & Isolation Hardening

### The Shared Windows OS Identity Problem
On a standard Windows workstation where both the researcher and the coding agent operate under the same user identity (`quanp`), file permissions are shared. Simply moving the master dataset outside the repository folder does not deny read access to an autonomous agent running shell commands.

### Enforcing True File Isolation

#### Recommended Approach 1: Scoped Docker Container Boundary
- Run the agent inside a Docker container where only the project repository is bind-mounted:
  ```bash
  docker run -it --rm \
    -v "d:/Works/Journals/1. Elsevier/ESWA - Experts system with applications/eswa-hta:/workspace" \
    --user 1000:1000 \
    eswa-agent-env
  ```
- **Crucial Security Requirement:** The Docker container provides an effective isolation boundary **only if** the agent cannot execute host Windows commands (e.g., via host agent bridges or elevated hypervisor sockets).

#### Recommended Approach 2: Encrypted Vault / Air-Gapped Drive (Simpler & More Robust)
- Keep master datasets and sealed test outcomes on an encrypted VeraCrypt volume or external USB SSD.
- Dismount or unplug the drive whenever the coding agent is active.
- This guarantees physical isolation regardless of OS account privileges.
