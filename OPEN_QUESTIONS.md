# Open Questions Register

Format:
- **ID**: `OQ-xxx`
- **Gate**: Gate ID
- **Question**: Specific blocking question
- **Evidence**: Grounding facts from workspace and spec
- **Options**: Possible courses of action
- **Recommendation**: Agent's recommended path adhering to HTA_IMPLEMENTATION_SPEC_v1.2

---

### OQ-001: Missing Companion Input Bundle (§15.1, Rule A3, Rule H9)
- **Gate**: G0
- **Question**: Where are the companion configuration, reference implementation, test suite, golden fixtures, and gate acceptance manifests located?
- **Evidence**:
  - `docs/HTA_IMPLEMENTATION_SPEC_v1.2.md` lines 6-8 explicitly state:
    > "Required bundle: this file + `config/protocol.yaml` + `config/columns.yaml` + `reference/hta_ref.py` + `reference/test_hta_ref.py` + human-approved golden fixtures (§15). These companion files were referenced in v1.1 but have not been verified as supplied alongside this Markdown file. This specification is NOT a substitute for them. G0 must fail closed until they physically exist, pass the manifest audit, and are human-approved."
  - §15.1 lists the required artifacts whose missing-file action is mandatory `STOP`:
    - `config/protocol.yaml`
    - `config/columns.yaml`
    - `reference/hta_ref.py`
    - `reference/test_hta_ref.py`
    - `fixtures/golden_cases.csv` and `fixtures/golden_expectations.csv`
    - `fixtures/fixture_manifest.sha256`
    - `contracts/gate_acceptance_manifest.yaml`
  - Rule A3 forbids the agent from inventing or self-certifying reference code.
- **Options**:
  1. Human research lead supplies the companion bundle files and fixture manifest into the repository.
  2. Human research lead points the agent to an external repository/folder containing the approved bundle.
- **Recommendation**: Human research lead supplies the canonical companion bundle into `config/`, `reference/`, `fixtures/`, and `contracts/` along with `fixture_manifest.sha256`.

---

### OQ-002: Staging of Holdout Data and Physical Test Outcome Isolation (§4.3, Rule A5, Rule H8)
- **Gate**: G0 / G1
- **Question**: Has the human data steward physically staged the separated development source (2015–2022) and candidate test feature source (2023–2025), and isolated sealed test outcomes?
- **Evidence**:
  - Rule A5 and §4.3 state that final-test outcomes (FY2023–2025) must be physically separated before G1 starts. Pre-G10 code must not read full raw source copies or replicas containing protected test targets.
  - A staging receipt `contracts/staged_holdout_receipt.md` and input checksums `contracts/input_manifest.sha256` must exist before proceeding.
  - A panel file `vn_pipeline_final_firm_year_2015_2025.parquet` exists in `d:/Works/dissertation/dissertation-v4/data/source/`, but under §4.3 Rule 4, "Remove or deny the agent read permission to every raw or processed duplicate of test outcomes... a file-name convention and a sentinel alone do not enforce access control."
- **Options**:
  1. Data steward stages `data/source/canonical_dev_source.parquet` (2015–2022) and `data/source/test_features_candidate.parquet` (2023–2025) into `data/source/`, while keeping test outcomes strictly sealed/unreadable until G10.
  2. Data steward provides scripts and instructions for staging according to §4.3.
- **Recommendation**: Option 1. The data steward stages the separated development and test-feature files into `data/source/` and provides `contracts/staged_holdout_receipt.md`.

---

### OQ-003: Environment Setup and Model Checkpoint Verification (G0 Step 4, §15.3, Rule A12)
- **Gate**: G0
- **Question**: What virtual environment strategy and model checkpoint source should be used for TabICLv2, CatBoost, and NGBoost?
- **Evidence**:
  - System environment has Python 3.12.2 and PyTorch 2.6.0+cu118 with NVIDIA GeForce RTX 3060 (12GB VRAM).
  - `tabicl==2.2.0` is available on PyPI, but currently neither `tabicl`, `catboost`, nor `ngboost` is installed in the current environment.
  - Rule A12 states: "Do not call external web APIs except to download the human-approved, checksum-pinned TabICL checkpoints (or use the human-provided local model_path). Package upgrades, undocumented checkpoint substitutions, and unpinned dependencies require human approval."
- **Options**:
  1. Create a dedicated `.venv` in the workspace, install pinned versions (`tabicl==2.2.0`, `catboost`, `ngboost`, `scipy`, `pandas`, `pyarrow`, `pytest`), download TabICL checkpoints with human-approved checksums, and record `environment_lock.txt`.
  2. Use a shared virtual environment specified by the human lead.
- **Recommendation**: Create a dedicated project virtual environment `.venv`, install the pinned dependencies, run G0 API preflight, and generate `environment_lock.txt`.
