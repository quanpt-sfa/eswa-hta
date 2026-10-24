"""Independent regression test: candidate model registry must match HTA v1.2."""
from pathlib import Path

import yaml


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
    assert protocol["models"]["primary_claim"]["primary_benchmark"] == "B3a"
    assert "BinaryClassification" in suite["B3a"]["objective"]


def test_outcome_config_matches_specification_contract():
    protocol = yaml.safe_load((Path(__file__).parent / "protocol.yaml").read_text(encoding="utf-8"))
    assert protocol["outcome"]["primary_k"] == 0.05
    assert protocol["outcome"]["zero_tol"] == 0.0
    assert protocol["outcome"]["thresholds"] == [0.05, 0.075, 0.10, 0.15, 0.20]
    assert protocol["temporal_windows"]["final_test_years"] == [2023, 2024, 2025]
    assert protocol["random_seeds"]["bootstrap_replications"] == 2000
