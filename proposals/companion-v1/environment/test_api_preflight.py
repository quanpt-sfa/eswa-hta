"""
Unit Test Suite for API Preflight Verification
==============================================
Governance: docs/HTA_IMPLEMENTATION_SPEC_v1.2.md (§5 G0, §15.3)

These tests exercise preflight ORCHESTRATION with mocks and synthetic stubs
(no downloads, no real checkpoints). By construction a mocked run can never
produce ALL_PROBES_PASSED: it proves the plumbing, not GPU workload feasibility
(see test_mock_orchestration_never_produces_overall_pass).
"""

import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

env_dir = Path(__file__).resolve().parent
if str(env_dir) not in sys.path:
    sys.path.insert(0, str(env_dir))

import api_preflight  # noqa: E402
from api_preflight import (  # noqa: E402
    REQUIRED_WORKLOAD_FIELDS,
    aggregate_overall_status,
    implementation_identity,
    probe_api_compatibility,
)

PREFLIGHT_SCRIPT = env_dir / "api_preflight.py"


# ------------------------------------------------------------------------------
# Mock TabICL estimators (explicit constructor signature = documented TabICL API)
# ------------------------------------------------------------------------------

def _make_tabicl_mock(kind, *, loaded="correct", device_attr="cuda", n_cols=999, ascending=True,
                      oom_rows=None, with_use_amp=True, mutate_file=False, other_path=None):
    class _Base:
        calls = []

        def _init(self, model_path, allow_auto_download, device, n_estimators, random_state, extra):
            type(self).calls.append({
                "model_path": model_path, "allow_auto_download": allow_auto_download,
                "device": device, "n_estimators": n_estimators, "random_state": random_state, "extra": extra,
            })
            self.model_path = model_path
            self.device = device

        def fit(self, X, y):
            if oom_rows is not None and len(X) > oom_rows:
                raise MockTorchValid.cuda.OutOfMemoryError("CUDA out of memory")
            if mutate_file:
                with open(self.model_path, "ab") as f:
                    f.write(b"TAMPERED")
            self.model_path_ = {"correct": self.model_path, "missing": None, "other": other_path}[loaded]
            self.device_ = None if device_attr == "missing" else device_attr
            if kind == "regressor":
                self.alphas_ = np.linspace(0.001, 0.999, n_cols)
            else:
                self.classes_ = np.array([0, 1])
            return self

        def _out(self, X):
            n = len(X)
            if kind == "regressor":
                base = np.linspace(-5.0, 5.0, n_cols)
                return np.tile(base if ascending else base[::-1], (n, 1))
            return np.full((n, 2), 0.5)

    if with_use_amp:
        class Mock(_Base):
            def __init__(self, model_path=None, allow_auto_download=True, device="cpu", n_estimators=8,
                         random_state=None, use_amp="auto", offload_mode="auto", disk_offload_dir=None):
                self._init(model_path, allow_auto_download, device, n_estimators, random_state,
                           {"use_amp": use_amp, "offload_mode": offload_mode})
    else:
        class Mock(_Base):
            def __init__(self, model_path=None, allow_auto_download=True, device="cpu", n_estimators=8,
                         random_state=None):
                self._init(model_path, allow_auto_download, device, n_estimators, random_state, {})

    if kind == "regressor":
        Mock.predict = lambda self, X, output_type=None: self._out(X)
        Mock.__name__ = "MockTabICLRegressor"
    else:
        Mock.predict_proba = lambda self, X: self._out(X)
        Mock.__name__ = "MockTabICLClassifier"
    return Mock


class MockCatBoostRegressorValid:
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.tile(np.linspace(-3.0, 3.0, 99), (len(X), 1))


class MockDistObject:
    def cdf(self, x):
        return 0.5

    def sf(self, x):
        return 0.5


class MockPredDist:
    def __init__(self):
        self.dist = MockDistObject()


class MockNGBRegressorValid:
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, X, y):
        return self

    def pred_dist(self, X):
        return MockPredDist()


class MockTorchValid:
    __version__ = "2.2.0"

    class version:
        cuda = "12.1"

    class cuda:
        class OutOfMemoryError(Exception):
            pass

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def get_device_name(idx):
            return "NVIDIA GeForce RTX 3060"

        @staticmethod
        def get_device_properties(idx):
            props = MagicMock()
            props.total_memory = 12 * (1024**3)
            return props

        @staticmethod
        def reset_peak_memory_stats(idx=0):
            pass

        @staticmethod
        def max_memory_allocated(idx=0):
            return int(2048 * (1024**2))  # 2048 MB

        @staticmethod
        def max_memory_reserved(idx=0):
            return int(3072 * (1024**2))  # 3072 MB

    @staticmethod
    def randn(*args, **kwargs):
        return MagicMock()

    @staticmethod
    def matmul(*args, **kwargs):
        return MagicMock()


VALID_CFG = {
    "context_size": 200,
    "n_features": 8,
    "n_estimators": 4,
    "prediction_batch_size": 10,
    "device": "cuda",
    "use_amp": False,
    "offload_mode": "auto",
    "max_memory_mb": 11000.0,
}


class Env:
    """Installs mocks into sys.modules and owns approved synthetic checkpoint files."""

    def __init__(self, monkeypatch, tmp_path):
        self.mp = monkeypatch
        self.reg_path = tmp_path / "regressor_approved.ckpt"
        self.clf_path = tmp_path / "classifier_approved.ckpt"
        self.other_path = tmp_path / "unrelated.ckpt"
        self.reg_path.write_bytes(b"REGRESSOR-ARTIFACT-A")
        self.clf_path.write_bytes(b"CLASSIFIER-ARTIFACT-B")
        self.other_path.write_bytes(b"UNRELATED-ARTIFACT-C")
        # Independent oracle: hashes computed with hashlib directly, not via api_preflight.
        import hashlib
        self.reg_sha = hashlib.sha256(b"REGRESSOR-ARTIFACT-A").hexdigest()
        self.clf_sha = hashlib.sha256(b"CLASSIFIER-ARTIFACT-B").hexdigest()
        self.install()

    def install(self, reg=None, clf=None):
        self.reg_cls = reg or _make_tabicl_mock("regressor")
        self.clf_cls = clf or _make_tabicl_mock("classifier")
        tabicl_mock = MagicMock()
        tabicl_mock.TabICLRegressor = self.reg_cls
        tabicl_mock.TabICLClassifier = self.clf_cls
        catboost_mock = MagicMock()
        catboost_mock.CatBoostRegressor = MockCatBoostRegressorValid
        ngboost_mock = MagicMock()
        ngboost_mock.NGBRegressor = MockNGBRegressorValid
        distns_mock = MagicMock()
        self.mp.setitem(sys.modules, "tabicl", tabicl_mock)
        self.mp.setitem(sys.modules, "catboost", catboost_mock)
        self.mp.setitem(sys.modules, "ngboost", ngboost_mock)
        self.mp.setitem(sys.modules, "ngboost.distns", distns_mock)
        self.mp.setitem(sys.modules, "torch", MockTorchValid)
        self.distns = distns_mock

    def run(self, **overrides):
        kwargs = dict(
            regressor_checkpoint=self.reg_path, regressor_sha256=self.reg_sha,
            classifier_checkpoint=self.clf_path, classifier_sha256=self.clf_sha,
            workload_config=dict(VALID_CFG), run_inference_smoke=True, run_workload_test=True,
        )
        kwargs.update(overrides)
        return probe_api_compatibility(**kwargs)


@pytest.fixture
def env(monkeypatch, tmp_path):
    return Env(monkeypatch, tmp_path)


def l3(report):
    return report["components"]["HardwareRuntime"]["level3_registered_workload"]


# ==============================================================================
# Mock orchestration can never produce overall PASS
# ==============================================================================

def test_mock_orchestration_never_produces_overall_pass(env):
    report = env.run()
    assert report["status"] != "ALL_PROBES_PASSED"
    assert report["status"] == "BLOCKED_DEPENDENCIES_OR_CHECKPOINT_UNVERIFIED"
    assert l3(report)["mocked"] is True
    assert l3(report)["status"] == "MOCK_ORCHESTRATION_PASSED"
    assert l3(report)["evidence_type"] == "MOCK_ORCHESTRATION_ONLY"
    assert report["contains_mock_execution"] is True
    assert report["components"]["HardwareRuntime"]["status"] == "MOCK_ORCHESTRATION_PASSED"
    for name in ("TabICLRegressor", "TabICLClassifier", "CatBoostMultiQuantile", "NGBoost"):
        assert report["components"][name]["status"] == "MOCK_ORCHESTRATION_PASSED", name
        assert report["components"][name]["mocked"] is True
    # Orchestration itself works: loaded artifacts were checked for both models
    assert l3(report)["loaded_checkpoint_verified"] is True
    assert l3(report)["registered_workload_executed"] is True


def test_mock_orchestration_uses_each_models_own_approved_path_and_disables_download(env):
    env.run()
    reg_calls, clf_calls = env.reg_cls.calls, env.clf_cls.calls
    assert reg_calls and clf_calls
    for c in reg_calls:
        assert c["model_path"] == str(env.reg_path.resolve())
        assert c["allow_auto_download"] is False
        assert c["random_state"] == 42
    for c in clf_calls:
        assert c["model_path"] == str(env.clf_path.resolve())
        assert c["allow_auto_download"] is False
    # Registered workload configuration reached the constructors verbatim
    wl = [c for c in reg_calls if c["n_estimators"] == VALID_CFG["n_estimators"]]
    assert wl and wl[0]["device"] == "cuda" and wl[0]["extra"] == {"use_amp": False, "offload_mode": "auto"}


# ==============================================================================
# Checkpoint identity
# ==============================================================================

def test_missing_checkpoints_blocked_and_no_model_constructed(env):
    report = env.run(regressor_checkpoint=None, regressor_sha256=None, classifier_checkpoint=None, classifier_sha256=None)
    assert report["components"]["RegressorCheckpoint"]["status"] == "BLOCKED_APPROVED_CHECKPOINT_OR_HASH_MISSING"
    assert report["components"]["ClassifierCheckpoint"]["status"] == "BLOCKED_APPROVED_CHECKPOINT_OR_HASH_MISSING"
    assert report["status"] == "BLOCKED_DEPENDENCIES_OR_CHECKPOINT_UNVERIFIED"
    # Nothing is constructed without an approved artifact -> auto-download cannot happen
    assert env.reg_cls.calls == [] and env.clf_cls.calls == []
    assert l3(report)["status"] == "BLOCKED_APPROVED_CHECKPOINT_UNVERIFIED"


def test_regressor_hash_mismatch_fails(env):
    report = env.run(regressor_sha256="deadbeef" * 8)
    assert report["components"]["RegressorCheckpoint"]["status"] == "FAILED_CHECKPOINT_HASH"
    assert report["status"] == "PROBES_FAILED"
    assert env.reg_cls.calls == []


def test_classifier_hash_mismatch_fails(env):
    report = env.run(classifier_sha256="deadbeef" * 8)
    assert report["components"]["ClassifierCheckpoint"]["status"] == "FAILED_CHECKPOINT_HASH"
    assert l3(report)["status"] == "FAILED_CHECKPOINT_HASH"
    assert report["status"] == "PROBES_FAILED"


def test_regressor_ok_but_classifier_checkpoint_missing_stays_blocked(env):
    report = env.run(classifier_checkpoint=None, classifier_sha256=None)
    assert report["components"]["RegressorCheckpoint"]["status"] == "FILE_HASH_VERIFIED"
    assert report["components"]["TabICLRegressor"]["status"] == "MOCK_ORCHESTRATION_PASSED"
    assert report["components"]["ClassifierCheckpoint"]["status"] == "BLOCKED_APPROVED_CHECKPOINT_OR_HASH_MISSING"
    assert report["components"]["TabICLClassifier"]["status"] == "BLOCKED_APPROVED_CHECKPOINT_UNVERIFIED"
    assert l3(report)["status"] == "BLOCKED_APPROVED_CHECKPOINT_UNVERIFIED"
    assert report["status"] == "BLOCKED_DEPENDENCIES_OR_CHECKPOINT_UNVERIFIED"
    assert env.clf_cls.calls == []


def test_deprecated_single_checkpoint_args_apply_to_regressor_only(env):
    report = probe_api_compatibility(
        checkpoint_path=env.reg_path, expected_checkpoint_sha256=env.reg_sha,
        workload_config=dict(VALID_CFG), run_workload_test=True,
    )
    assert report["deprecated_single_checkpoint_argument_used"] is True
    assert report["components"]["RegressorCheckpoint"]["status"] == "FILE_HASH_VERIFIED"
    assert report["components"]["ClassifierCheckpoint"]["status"] == "BLOCKED_APPROVED_CHECKPOINT_OR_HASH_MISSING"
    assert report["status"] == "BLOCKED_DEPENDENCIES_OR_CHECKPOINT_UNVERIFIED"


def test_loaded_model_path_missing_is_blocked(env):
    env.install(reg=_make_tabicl_mock("regressor", loaded="missing"))
    report = env.run()
    assert report["components"]["TabICLRegressor"]["status"] == "BLOCKED_MODEL_ARTIFACT_IDENTITY_UNVERIFIED"
    assert l3(report)["status"] == "BLOCKED_MODEL_ARTIFACT_IDENTITY_UNVERIFIED"
    assert report["status"] == "BLOCKED_DEPENDENCIES_OR_CHECKPOINT_UNVERIFIED"


def test_loaded_model_path_wrong_artifact_fails_even_if_unrelated_file_exists(env):
    env.install(reg=_make_tabicl_mock("regressor", loaded="other", other_path=str(env.other_path)))
    report = env.run()
    assert report["components"]["TabICLRegressor"]["status"] == "FAILED_LOADED_CHECKPOINT_MISMATCH"
    assert l3(report)["status"] == "FAILED_LOADED_CHECKPOINT_MISMATCH"
    assert report["status"] == "PROBES_FAILED"


def test_loaded_artifact_hash_mismatch_fails(env):
    # Approved path is loaded, but the bytes at that path are not the approved artifact.
    env.install(clf=_make_tabicl_mock("classifier", mutate_file=True))
    report = env.run()
    assert report["components"]["ClassifierCheckpoint"]["status"] == "FILE_HASH_VERIFIED"  # pre-load file was fine
    assert report["components"]["TabICLClassifier"]["status"] == "FAILED_LOADED_CHECKPOINT_HASH"
    assert report["status"] == "PROBES_FAILED"


def test_runtime_device_mismatch_fails_and_missing_device_blocks(env):
    env.install(reg=_make_tabicl_mock("regressor", device_attr="cpu"))
    report = env.run()
    assert report["components"]["TabICLRegressor"]["status"] == "FAILED_DEVICE_MISMATCH"
    assert report["status"] == "PROBES_FAILED"
    env.install(reg=_make_tabicl_mock("regressor", device_attr="missing"))
    report = env.run()
    assert report["components"]["TabICLRegressor"]["status"] == "BLOCKED_MODEL_DEVICE_UNVERIFIED"
    assert report["status"] == "BLOCKED_DEPENDENCIES_OR_CHECKPOINT_UNVERIFIED"


def test_installed_constructor_lacking_required_parameters_blocks_not_invented(env):
    env.install(reg=_make_tabicl_mock("regressor", with_use_amp=False))
    report = env.run()
    assert l3(report)["status"] == "BLOCKED_API_INCOMPATIBLE"
    assert report["status"] == "BLOCKED_DEPENDENCIES_OR_CHECKPOINT_UNVERIFIED"


# ==============================================================================
# Registered workload: strict schema, no defaults
# ==============================================================================

@pytest.mark.parametrize("missing", REQUIRED_WORKLOAD_FIELDS)
def test_missing_workload_field_blocks(env, missing):
    cfg = {k: v for k, v in VALID_CFG.items() if k != missing}
    report = env.run(workload_config=cfg)
    assert l3(report)["status"] == "BLOCKED_WORKLOAD_CONFIG_INCOMPLETE"
    assert missing in l3(report)["reason"]
    assert l3(report)["registered_workload_executed"] is False
    assert l3(report)["workload_configuration_verified"] is False
    assert report["status"] == "BLOCKED_DEPENDENCIES_OR_CHECKPOINT_UNVERIFIED"
    # No silent default: no workload-sized model was ever constructed
    assert all(c["n_estimators"] != 10 for c in env.reg_cls.calls + env.clf_cls.calls)


@pytest.mark.parametrize("key,value", [
    ("context_size", 0), ("context_size", True), ("n_features", -3), ("n_estimators", 2.5),
    ("prediction_batch_size", 0), ("max_memory_mb", 0), ("max_memory_mb", float("nan")),
    ("device", "cpu"), ("use_amp", "yes"), ("offload_mode", "tape"),
])
def test_invalid_workload_values_block(env, key, value):
    cfg = dict(VALID_CFG)
    cfg[key] = value
    report = env.run(workload_config=cfg)
    assert l3(report)["status"] == "BLOCKED_WORKLOAD_CONFIG_INVALID"
    assert l3(report)["registered_workload_executed"] is False
    assert report["status"] != "ALL_PROBES_PASSED"


def test_disk_offload_requires_directory(env):
    cfg = dict(VALID_CFG, offload_mode="disk")
    assert l3(env.run(workload_config=cfg))["status"] == "BLOCKED_WORKLOAD_CONFIG_INVALID"


def test_complete_config_without_execution_is_blocked(env):
    report = env.run(run_workload_test=False)
    assert l3(report)["status"] == "BLOCKED_WORKLOAD_UNVERIFIED"
    assert l3(report)["registered_workload_executed"] is False
    assert report["status"] == "BLOCKED_DEPENDENCIES_OR_CHECKPOINT_UNVERIFIED"


def test_workload_hash_in_config_must_agree_with_approved_checkpoint(env):
    report = env.run(workload_config=dict(VALID_CFG, regressor_sha256="00" * 32))
    assert l3(report)["status"] == "BLOCKED_WORKLOAD_CONFIG_INVALID"


def test_workload_records_actual_parameters_shapes_and_memory(env):
    rec = l3(env.run())
    assert rec["actual_parameters"] == VALID_CFG
    # Independent oracle: dimensions come straight from the registered config
    assert rec["synthetic_data"]["X_train_shape"] == [200, 8]
    assert rec["synthetic_data"]["X_eval_shape"] == [10, 8]
    assert rec["models"]["regressor"]["output_shape"] == [10, 999]
    assert rec["models"]["classifier"]["output_shape"] == [10, 2]
    assert rec["models"]["regressor"]["peak_vram_allocated_mb"] == 2048.0
    assert rec["models"]["regressor"]["peak_vram_reserved_mb"] == 3072.0
    assert rec["models"]["regressor"]["alphas_count"] == 999
    assert rec["models"]["regressor"]["loaded_artifact"]["loaded_sha256"] == env.reg_sha
    assert rec["models"]["classifier"]["loaded_artifact"]["loaded_sha256"] == env.clf_sha
    assert rec["memory_limit_mb"] == 11000.0 and rec["exceptions"] == []


def test_gpu_oom_in_registered_workload_fails(env):
    # Toy API probe (20 rows) survives; the registered 200-row context runs out of memory.
    env.install(reg=_make_tabicl_mock("regressor", oom_rows=100))
    report = env.run()
    assert report["components"]["TabICLRegressor"]["status"] == "MOCK_ORCHESTRATION_PASSED"
    assert l3(report)["status"] == "FAILED_CUDA_OOM"
    assert l3(report)["exceptions"]
    assert report["components"]["HardwareRuntime"]["status"] == "FAILED_HARDWARE_RUNTIME"
    assert report["status"] == "PROBES_FAILED"


def test_memory_budget_exceeded_fails(env):
    report = env.run(workload_config=dict(VALID_CFG, max_memory_mb=1000.0))  # mock reports 2048 MB
    assert l3(report)["status"] == "FAILED_EXCEEDS_MEMORY_LIMIT"
    assert report["status"] == "PROBES_FAILED"


@pytest.mark.parametrize("kwargs", [{"n_cols": 500}, {"ascending": False}])
def test_incompatible_output_geometry_fails(env, kwargs):
    env.install(reg=_make_tabicl_mock("regressor", **kwargs))
    report = env.run()
    assert report["components"]["TabICLRegressor"]["status"] == "FAILED_GEOMETRY_OR_ALPHA_GRID"
    assert report["status"] == "PROBES_FAILED"


def test_level2_is_basic_cuda_only(env):
    report = env.run(run_workload_test=False)
    lvl2 = report["components"]["HardwareRuntime"]["level2_synthetic_smoke"]
    assert lvl2["status"] == "PASSED" and lvl2["cuda_functional"] is True
    assert "peak_vram_allocated_mb" not in lvl2  # dummy tensor memory is never reused as workload evidence
    assert l3(report)["status"] == "BLOCKED_WORKLOAD_UNVERIFIED"


def test_cuda_inference_untested_blocked(env):
    report = env.run(run_inference_smoke=False)
    hw = report["components"]["HardwareRuntime"]
    assert hw["level2_synthetic_smoke"]["status"] == "BLOCKED_INFERENCE_UNTESTED"
    assert hw["status"] == "BLOCKED_INFERENCE_MEMORY_UNVERIFIED"
    assert report["status"] == "BLOCKED_DEPENDENCIES_OR_CHECKPOINT_UNVERIFIED"


# ==============================================================================
# Other components / precedence
# ---------------------------------------------------------------------------

def test_missing_ngboost_distribution_fails(env):
    del env.distns.TFixedDf
    report = env.run()
    assert report["components"]["NGBoost"]["status"] == "FAILED_REGISTERED_DISTRIBUTION_SUPPORT"
    assert report["status"] == "PROBES_FAILED"


def test_failure_not_concealed_by_blockers(env):
    report = env.run(regressor_sha256="wrong_hash_deadbeef", run_workload_test=False)
    assert report["status"] == "PROBES_FAILED"


def test_diagnostic_only_mode_cannot_pass(env):
    report = env.run(diagnostic_only=True)
    assert report["status"] == "DIAGNOSTIC_ONLY_PREFLIGHT_BLOCKED"


# ==============================================================================
# Implementation identity (not name based)
# ==============================================================================

def test_identity_magicmock_and_spoofed_names_are_mock(monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    class TabICLRegressor:  # name spoof: real-looking class name, defined outside the package
        pass

    ident, why = implementation_identity("numpy", np, TabICLRegressor)
    assert ident == "MOCK" and any("not in package" in w for w in why)
    ident, _ = implementation_identity("tabicl", MagicMock())
    assert ident == "MOCK"


def test_identity_real_unresolved_and_test_runner(monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    assert implementation_identity("numpy", np, np.ndarray)[0] == "REAL"
    fake = types.ModuleType("nopkg_xyz")
    fake.__file__ = "x.py"
    assert implementation_identity("nopkg_xyz", fake)[0] == "UNRESOLVED"
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "t")
    ident, why = implementation_identity("numpy", np, np.ndarray)
    assert ident == "MOCK" and any("test runner" in w for w in why)


def test_unresolved_tabicl_identity_blocks(env, monkeypatch):
    fake_mod = types.ModuleType("tabicl")
    fake_mod.__file__ = "tabicl/__init__.py"
    reg = _make_tabicl_mock("regressor")
    clf = _make_tabicl_mock("classifier")
    reg.__module__ = clf.__module__ = "tabicl"
    fake_mod.TabICLRegressor, fake_mod.TabICLClassifier = reg, clf
    monkeypatch.setitem(sys.modules, "tabicl", fake_mod)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    def no_dist(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", no_dist)
    report = env.run()
    assert report["components"]["TabICLRegressor"]["status"] == "BLOCKED_MODEL_IMPLEMENTATION_UNRESOLVED"
    assert l3(report)["status"] == "BLOCKED_MODEL_IMPLEMENTATION_UNRESOLVED"
    assert report["status"] == "BLOCKED_DEPENDENCIES_OR_CHECKPOINT_UNVERIFIED"


# ==============================================================================
# Aggregation never trusts status strings alone
# ==============================================================================

def _forged_all_real_report():
    ev = {"mocked": False, "evidence_type": "REAL_EXECUTION"}
    return {
        "components": {
            "RegressorCheckpoint": {"status": "FILE_HASH_VERIFIED"},
            "ClassifierCheckpoint": {"status": "FILE_HASH_VERIFIED"},
            "TabICLRegressor": {"status": "PASSED", "loaded_checkpoint_verified": True, **ev},
            "TabICLClassifier": {"status": "PASSED", "loaded_checkpoint_verified": True, **ev},
            "CatBoostMultiQuantile": {"status": "PASSED", **ev},
            "NGBoost": {"status": "PASSED", **ev},
            "HardwareRuntime": {
                "status": "PASSED", "torch_real": True,
                "level3_registered_workload": {
                    "status": "PASSED", "loaded_checkpoint_verified": True,
                    "workload_configuration_verified": True, "registered_workload_executed": True, **ev,
                },
            },
        }
    }


def test_aggregate_all_real_evidence_is_the_only_pass_route():
    assert aggregate_overall_status(_forged_all_real_report()) == "ALL_PROBES_PASSED"


@pytest.mark.parametrize("mutate", [
    lambda r: r["components"]["TabICLRegressor"].update(mocked=True),
    lambda r: r["components"]["TabICLClassifier"].update(evidence_type="MOCK_ORCHESTRATION_ONLY"),
    lambda r: r["components"]["TabICLRegressor"].update(loaded_checkpoint_verified=False),
    lambda r: r["components"]["CatBoostMultiQuantile"].update(mocked=True),
    lambda r: r["components"]["NGBoost"].pop("evidence_type"),
    lambda r: r["components"]["HardwareRuntime"].update(torch_real=False),
    lambda r: r["components"]["HardwareRuntime"]["level3_registered_workload"].update(mocked=True),
    lambda r: r["components"]["HardwareRuntime"]["level3_registered_workload"].update(loaded_checkpoint_verified=False),
    lambda r: r["components"]["HardwareRuntime"]["level3_registered_workload"].update(workload_configuration_verified=False),
    lambda r: r["components"]["HardwareRuntime"]["level3_registered_workload"].update(registered_workload_executed=False),
    lambda r: r["components"].pop("ClassifierCheckpoint"),
    lambda r: r["components"]["RegressorCheckpoint"].update(status="BLOCKED_CHECKPOINT_FILE_NOT_FOUND"),
])
def test_aggregate_rejects_pass_claims_with_inconsistent_or_missing_evidence(mutate):
    report = _forged_all_real_report()
    mutate(report)
    assert aggregate_overall_status(report) != "ALL_PROBES_PASSED"


def test_aggregate_failure_dominates_and_diagnostic_never_passes():
    report = _forged_all_real_report()
    assert aggregate_overall_status(report, diagnostic_only=True) == "DIAGNOSTIC_ONLY_PREFLIGHT_BLOCKED"
    report["components"]["NGBoost"]["status"] = "FAILED_REGISTERED_DISTRIBUTION_SUPPORT"
    assert aggregate_overall_status(report) == "PROBES_FAILED"


# ==============================================================================
# CLI
# ==============================================================================

def _cli(args, out):
    return subprocess.run(
        [sys.executable, str(PREFLIGHT_SCRIPT), "--output", str(out), *args],
        capture_output=True, text=True,
    )


def test_cli_exit_codes_and_independent_checkpoints(tmp_path):
    import hashlib
    reg = tmp_path / "reg.ckpt"
    reg.write_bytes(b"R")
    out = tmp_path / "o.json"

    assert _cli(["--diagnostic-only"], out).returncode == 2

    res = _cli(["--regressor-checkpoint", str(reg), "--regressor-sha256", "wrong"], out)
    assert res.returncode == 1
    assert json.loads(out.read_text())["components"]["RegressorCheckpoint"]["status"] == "FAILED_CHECKPOINT_HASH"

    # Deprecated single-checkpoint arguments: regressor only, with an explicit warning
    res = _cli(["--checkpoint", str(reg), "--expected-sha256", hashlib.sha256(b"R").hexdigest()], out)
    rep = json.loads(out.read_text())
    assert res.returncode == 2 and "deprecated" in res.stderr.lower()
    assert rep["components"]["RegressorCheckpoint"]["status"] == "FILE_HASH_VERIFIED"
    assert rep["components"]["ClassifierCheckpoint"]["status"] == "BLOCKED_APPROVED_CHECKPOINT_OR_HASH_MISSING"


def test_cli_workload_flags_and_unreadable_config_blocks(tmp_path):
    out = tmp_path / "o.json"
    wl = tmp_path / "wl.json"
    wl.write_text(json.dumps(VALID_CFG), encoding="utf-8")
    res = _cli(["--workload-config", str(wl), "--run-workload-test"], out)
    assert res.returncode == 2  # model packages / checkpoints are not available -> BLOCKED, never PASS
    assert json.loads(out.read_text())["status"] != "ALL_PROBES_PASSED"

    res = _cli(["--workload-config", str(tmp_path / "does_not_exist.json"), "--run-workload-test"], out)
    assert res.returncode == 2
    assert json.loads(out.read_text())["components"]["HardwareRuntime"]["level3_registered_workload"]["status"] != "PASSED"
