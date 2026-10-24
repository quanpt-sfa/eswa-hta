# Configuration and Gate Manifest Validation Report

**Document:** `proposals/companion-v1/CONFIG_VALIDATION_REPORT.md`  
**Date:** 2026-10-24  
**Author:** Implementation Agent (G0 Remediation)  
**Status:** VALIDATED_CANDIDATE (Ready for human review and sign-off)  

---

## 1. Executive Summary

All three candidate configuration files:
1. `proposals/companion-v1/config/protocol.yaml`
2. `proposals/companion-v1/config/columns.yaml`
3. `proposals/companion-v1/contracts/gate_acceptance_manifest.yaml`

have been constructed and successfully validated using `yaml.safe_load`.
- **Zero Syntax Errors:** Strict YAML syntax adherence verified.
- **Specification Fidelity:** Populated strictly according to `docs/HTA_IMPLEMENTATION_SPEC_v1.2.md` (§2 D01–D25, §15.1, §16.1).
- **Zero Fabricated Values:**
  - Pilot Gate G6 non-inferiority margins and minimum support counts are marked `UNRESOLVED_PENDING_HUMAN_CONFIRMATION`.
  - Reconciliation target numbers are marked `UNRESOLVED_PENDING_HUMAN_STAGING`.
  - Physical column mappings in `columns.yaml` are marked `CANDIDATE_PROPOSAL_UNAPPROVED` with `approved_by_human: false`.
  - Monetary unit multiplier is marked `null` pending human decision H2.

---

## 2. Structural & Schema Audit

| Artifact Path | Top-Level Sections | Key Parameter Checks | Status |
|---|---|---|---|
| `config/protocol.yaml` | `project`, `random_seeds`, `temporal_windows`, `outcome_specification`, `feature_sets`, `models`, `quantile_grids`, `tail_policies`, `screening_budgets`, `pilot_gate_G6`, `reconciliation_targets` | `primary_threshold_k: 0.05`, `zero_adjustment_tolerance: 0.0`, `dev_years: 2015-2022`, `test_years: 2023-2025`, `model_seed: 42`, `bootstrap_reps: 1000` | **PASS (Valid YAML)** |
| `config/columns.yaml` | `schema_version`, `status`, `approved_by_human`, `keys`, `outcome_inputs`, `currency_and_unit`, `point_in_time_metadata` | Strict definitions for `firm_id`, `fiscal_year`, `scope`, `pat_pre`, `pat_post`, `total_assets_pre`, `prediction_cutoff` | **PASS (Valid YAML)** |
| `contracts/gate_acceptance_manifest.yaml` | `manifest_version`, `project`, `gates` (G0 through G12) | Every gate specifies `requires_approvals`, `input_files`, `expected_outputs`, `primary_key`, `row_scope`, `qa_command`, `stop_conditions`, `unlock_policy`, `approval_owner` | **PASS (Valid YAML)** |

---

## 3. Human Actions for Promotion

1. **Review and Confirm Pilot Margins (`protocol.yaml`):** Review and populate `pilot_gate_G6.noninferiority_margin` and set `confirmed_by_human: true`.
2. **Confirm Monetary Unit (`columns.yaml`):** Set `unit_multiplier` to 1.0 (VND), 1,000, or 1,000,000 as appropriate for master source data.
3. **Approve and Promote:** Copy files to production locations:
   - `proposals/companion-v1/config/protocol.yaml` $\to$ `config/protocol.yaml`
   - `proposals/companion-v1/config/columns.yaml` $\to$ `config/columns.yaml`
   - `proposals/companion-v1/contracts/gate_acceptance_manifest.yaml` $\to$ `contracts/gate_acceptance_manifest.yaml`
