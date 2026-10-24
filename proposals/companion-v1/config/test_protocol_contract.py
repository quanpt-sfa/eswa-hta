"""
Independent regression test: candidate model registry and gate contracts
must match HTA v1.2 specification (§2, §8, §15, §16).
"""
from pathlib import Path
import yaml
import pytest


def test_model_id_meanings_are_not_swapped():
    protocol = yaml.safe_load((Path(__file__).parent / "protocol.yaml").read_text(encoding="utf-8"))
    suite = protocol["models"]["registered_suite"]
    expected = {
        "B1": ("CatBoost", "BinaryClassification"),
        "B2": ("TabICL", "BinaryClassification"),
        "B3a": ("CatBoost", "BinaryClassification"),
        "B3b": ("CatBoost", "ThresholdConditionedBinaryClassification"),
        "B4": ("NGBoost", "ParametricDistributionalRegression"),
        "B5": ("CatBoost", "MultiQuantileRegression"),
        "B6": ("TabICL", "DistributionalQuantileRegression"),
        "B7": ("CatBoostHurdle", None),
        "B8": ("TabICLHurdle", None),
    }
    for model, (family, objective) in expected.items():
        assert suite[model]["family"] == family, model
        if objective is not None:
            assert suite[model]["objective"] == objective, model
    assert suite["M0"]["alias_for"] == "B6"
    assert suite["M1"]["alias_for"] == "B8"
    assert protocol["models"]["primary_claim"]["proposed_model"] == "M1"
    assert protocol["models"]["primary_claim"]["primary_benchmark"] == "B3a"
    assert "BinaryClassification" in suite["B3a"]["objective"]


def test_outcome_config_matches_specification_contract():
    protocol = yaml.safe_load((Path(__file__).parent / "protocol.yaml").read_text(encoding="utf-8"))
    assert protocol["outcome"]["primary_k"] == 0.05
    assert protocol["outcome"]["zero_tol"] == 0.0
    assert protocol["outcome"]["thresholds"] == [0.05, 0.075, 0.10, 0.15, 0.20]
    assert protocol["temporal_windows"]["final_test_years"] == [2023, 2024, 2025]
    assert protocol["random_seeds"]["bootstrap_replications"] == 2000
    assert "Average Precision" in protocol["models"]["primary_claim"]["primary_metric"]
    assert "k=0.05" in protocol["models"]["primary_claim"]["primary_metric"]


def test_gate_dependencies_and_manifest_execution_order():
    manifest_path = Path(__file__).resolve().parent.parent / "contracts" / "gate_acceptance_manifest.yaml"
    assert manifest_path.exists(), "gate_acceptance_manifest.yaml must exist"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    gates = manifest["gates"]

    # Invariant: Gate prerequisite chaining
    assert "gates/G4_APPROVED.txt" in gates["G5"]["requires_approvals"]
    assert "gates/G5_APPROVED.txt" in gates["G6"]["requires_approvals"]
    assert "gates/G6_APPROVED.txt" in gates["G7"]["requires_approvals"]
    assert "gates/G7_APPROVED.txt" in gates["G8"]["requires_approvals"]
    assert "gates/G7_APPROVED.txt" in gates["G9"]["requires_approvals"]
    assert "gates/G9_APPROVED.txt" in gates["G10"]["requires_approvals"]

    # Invariant: Pilot models in G5 outputs
    g5_outputs = [o["path"] for o in gates["G5"]["expected_outputs"]]
    assert any("predictions_B1" in p for p in g5_outputs)
    assert any("predictions_B3a" in p for p in g5_outputs)
    assert any("predictions_B3b" in p for p in g5_outputs)
    assert any("predictions_B4" in p for p in g5_outputs)
    assert any("predictions_B6" in p for p in g5_outputs)

    # Invariant: G6 is pilot GO/NO-GO based on dev predictions
    assert "pilot_go_no_go.json" in [o["path"] for o in gates["G6"]["expected_outputs"]][0]

    # Invariant: G7 contains Hurdle models B7, B8
    g7_outputs = [o["path"] for o in gates["G7"]["expected_outputs"]]
    assert any("hurdle" in p for p in g7_outputs)

    # Invariant: G9 contains development experiments E1-E5 (NOT final test)
    g9_outputs = [o["path"] for o in gates["G9"]["expected_outputs"]]
    assert any("E1_distributional_comparison" in p for p in g9_outputs)
    assert any("E2_hurdle_ablation" in p for p in g9_outputs)
    assert any("E3_threshold_comparison" in p for p in g9_outputs)
    assert any("E4_threshold_transfer" in p for p in g9_outputs)
    assert any("E5_near_zero_subgroups" in p for p in g9_outputs)
    assert "Development out-of-fold" in gates["G9"]["row_scope"]
    assert gates["G9"]["unlock_policy"] == "PROTECTED_OUTCOMES_STRICTLY_FORBIDDEN"

    # Invariant: G0 through G9 strictly forbid protected outcomes
    for g_id in ["G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9"]:
        assert gates[g_id]["unlock_policy"] == "PROTECTED_OUTCOMES_STRICTLY_FORBIDDEN", g_id

    # Invariant: G10 unlocks test outcomes only in Phase 2
    assert gates["G10"]["unlock_policy"] == "TWO_PHASE_HUMAN_UNLOCK_AFTER_LOCKED_PREDICTIONS"
