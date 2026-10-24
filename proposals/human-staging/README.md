# Human-Only Holdout Staging Tools
================================

**Directory:** `proposals/human-staging/`  
**Target User:** Trusted Human Data Steward / Principal Investigator  
**Governance Specification:** `docs/HTA_IMPLEMENTATION_SPEC_v1.2.md` (§4.3, Rules A5, H8)  
**Status:** PROPOSAL_TOOL (Hardened with Master Dataset Protection & Atomic Write Safety)

---

## 1. Security Warning and Purpose

> **CRITICAL RESTRICTION:**  
> The scripts in this directory (`stage_data.py`) are strictly for **human data steward execution**. The autonomous coding agent is strictly prohibited from running this script against real research datasets or accessing master financial statement copies containing FY2023–2025 audit outcomes.

This toolset automates the physical extraction, sanitization, hashing, atomic writing, and receipt generation required before Gate G1:
1. Validates all input and output paths to ensure **zero possibility of overwriting or corrupting the master dataset**.
2. Splits master panel into development source (FY2015–2022) with development outcomes.
3. Generates candidate test features (FY2023–2025) with all post-audit outcomes, deltas, and labels strictly redacted via positive-selection allowlisting.
4. Packages sealed test outcomes (FY2023–2025) into secure storage outside agent access.
5. Writes all outputs atomically via staging temporary files to eliminate partial writes.
6. Generates SHA-256 cryptographic manifests and staging receipts.

---

## 2. Master Dataset Protection Invariants

The staging tool enforces five layers of safety before any write operation:

1. **Path Collision Validation (D1):**
   - No output path (`--out-dev`, `--out-test-features`, `--out-sealed`, `--receipt`, `--manifest`) may equal `--master`.
   - All output destinations must be mutually distinct.
   - Distinct path strings resolving to the same physical file (via symlinks or relative dot-segments) are detected and rejected.
   - Any path collision aborts execution **before creating or modifying any file**.

2. **Protected Vault Destination Separation (D2):**
   - Sealed outcomes (`--out-sealed`) must NOT be located within release directories (`--out-dev`, `--out-test-features`, or their parent directories).
   - The sealed vault destination must be in an isolated, secure location controlled exclusively by the human steward.

3. **Silent Overwrite Prevention:**
   - Existing output artifacts will **not** be silently overwritten.
   - If an output file already exists, staging aborts with `FileExistsError` unless the human steward explicitly passes the `--force` flag.

4. **Atomic Temporary Writes (D3):**
   - Outputs are written to isolated temporary files (`.tmp.<name>.<pid>`).
   - If sanitization or schema validation fails, all temporary files are cleaned up immediately.
   - Finalization occurs via atomic replacement (`os.replace`), ensuring that existing files are never left in a corrupted or partially written state.

5. **Byte-Identical Master Verification:**
   - The tool computes the SHA-256 hash of the master input before and after execution, asserting that the raw research source remains strictly 100% byte-identical.

---

## 3. Usage Instructions for Data Steward

### Command Syntax
```bash
python proposals/human-staging/stage_data.py \
  --master "/path/to/unredacted/master_panel.parquet" \
  --out-dev "data/source/canonical_dev_source.parquet" \
  --out-test-features "data/source/test_features_candidate.parquet" \
  --out-sealed "/path/to/secure_vault/test_outcomes_sealed.parquet" \
  --receipt "contracts/staged_holdout_receipt.md" \
  --manifest "contracts/input_manifest.sha256" \
  --allowlist "/path/to/human_approved_test_feature_columns.txt" \
  [--force]
```

### Verification Pipeline
Before releasing the staged files to the repository:
1. Run the synthetic adversarial test suite:
   ```bash
   python -m pytest proposals/human-staging/test_staging.py -v
   ```
2. Review the human-owned allowlist file before execution: **one released source column per line**, explicitly covering firm keys, verified pre-audit inputs, and point-in-time-approved historical features only. An absent allowlist makes staging fail closed. Verify that the test feature dataset contains no audited/current-year or outcome-derived source columns (including `fs_aud_*` and `audit_adj_*`).
3. Disconnect or unmount the master dataset and sealed vault storage from the agent's environment. The script's receipt proves only **file creation and schema redaction**, not OS access isolation. An independent access-denial probe, approved row/column mapping, and human receipt are required before claiming isolation verified.

---

## 4. Recommended Execution Environments & Isolation Hardening

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
