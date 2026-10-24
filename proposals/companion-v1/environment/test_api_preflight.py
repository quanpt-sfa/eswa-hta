"""
Unit Test Suite for API Preflight Verification
==============================================
Governance: docs/HTA_IMPLEMENTATION_SPEC_v1.2.md (§5 G0, §15.3)
Tests preflight orchestration, verification levels, fail-closed handling,
and reporting integrity using mocks and synthetic stubs (no external downloads).
"""

from pathlib import Path
import sys
import tempfile
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

env_dir = Path(__file__).resolve().parent
if str(env_dir) not in sys.path:
    sys.path.insert(0, str(env_dir))

from api_preflight import probe_api_compatibility, compute_file_sha256


class MockTabICLRegressorValid:
    def __init__(self, *args, **kwargs):
        self.alphas_ = np.linspace(0.001, 0.999, 999)

    def fit(self, X, y):
        return self

    def predict(self, X, output_type=None):
        n = len(X)
        # Returns shape (n, 999) strictly monotonic
        base = np.linspace(-5.0, 5.0, 999)
        return np.tile(base, (n, 1))


class MockTabICLRegressorBadShape:
    def __init__(self, *args, **kwargs):
        self.alphas_ = np.linspace(0.001, 0.999, 500)

    def fit(self, X, y):
        return self

    def predict(self, X, output_type=None):
        return np.zeros((len(X), 500))  # Expected 999


class MockTabICLRegressorMissingAlphas:
    def __init__(self, *args, **kwargs):
        self.alphas_ = None

    def fit(self, X, y):
        return self

    def predict(self, X, output_type=None):
        return np.zeros((len(X), 999))


class MockTabICLRegressorOOM:
    def __init__(self, *args, **kwargs):
        self.alphas_ = np.linspace(0.001, 0.999, 999)

    def fit(self, X, y):
        return self

    def predict(self, X, output_type=None):
        raise MockTorchValid.cuda.OutOfMemoryError("CUDA out of memory on registered workload")


class MockTabICLRegressorNonMonotonic:
    def __init__(self, *args, **kwargs):
        self.alphas_ = np.linspace(0.001, 0.999, 999)

    def fit(self, X, y):
        return self

    def predict(self, X, output_type=None):
        n = len(X)
        # Inverted ordering violates monotonicity
        base = np.linspace(5.0, -5.0, 999)
        return np.tile(base, (n, 1))


class MockTabICLClassifierValid:
    def __init__(self, *args, **kwargs):
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        n = len(X)
        return np.full((n, 2), 0.5)


class MockCatBoostRegressorValid:
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, X, y):
        return self

    def predict(self, X):
        n = len(X)
        base = np.linspace(-3.0, 3.0, 99)
        return np.tile(base, (n, 1))


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
            return int(2048 * (1024**2))  # 2048 MB peak

    @staticmethod
    def randn(*args, **kwargs):
        return MagicMock()

    @staticmethod
    def matmul(*args, **kwargs):
        return MagicMock()


@pytest.fixture
def mock_all_dependencies(monkeypatch):
    """Mocks all modeling libraries and hardware dependencies in sys.modules."""
    tabicl_mock = MagicMock()
    tabicl_mock.TabICLRegressor = MockTabICLRegressorValid
    tabicl_mock.TabICLClassifier = MockTabICLClassifierValid

    catboost_mock = MagicMock()
    catboost_mock.CatBoostRegressor = MockCatBoostRegressorValid

    ngboost_mock = MagicMock()
    ngboost_mock.NGBRegressor = MockNGBRegressorValid
    distns_mock = MagicMock()
    distns_mock.Normal = MagicMock()
    distns_mock.Laplace = MagicMock()
    distns_mock.TFixedDf = MagicMock()

    monkeypatch.setitem(sys.modules, "tabicl", tabicl_mock)
    monkeypatch.setitem(sys.modules, "catboost", catboost_mock)
    monkeypatch.setitem(sys.modules, "ngboost", ngboost_mock)
    monkeypatch.setitem(sys.modules, "ngboost.distns", distns_mock)
    monkeypatch.setitem(sys.modules, "torch", MockTorchValid)


@pytest.fixture
def sample_checkpoint_file():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        f.write(b"MOCK_CHECKPOINT_BINARY_DATA_FOR_PREFLIGHT_TEST")
        path = Path(f.name)
    expected_sha = compute_file_sha256(path)
    yield path, expected_sha
    if path.exists():
        path.unlink()


# ==============================================================================
# Mandatory Cases (A5.1 - A5.9)
# ==============================================================================

def test_1_all_components_valid_overall_pass(mock_all_dependencies, sample_checkpoint_file):
    ckpt_path, expected_sha = sample_checkpoint_file
    workload_cfg = {
        "n_estimators": 10,
        "context_size": 1000,
        "max_memory_mb": 11000.0,
    }

    report = probe_api_compatibility(
        checkpoint_path=ckpt_path,
        expected_checkpoint_sha256=expected_sha,
        workload_config=workload_cfg,
        run_inference_smoke=True,
        run_workload_test=True,
    )

    assert report["components"]["TabICLRegressor"]["status"] == "PASSED"
    assert report["components"]["TabICLClassifier"]["status"] == "PASSED"
    assert report["components"]["CatBoostMultiQuantile"]["status"] == "PASSED"
    assert report["components"]["NGBoost"]["status"] == "PASSED"
    assert report["components"]["Checkpoint"]["status"] == "PASSED"
    assert report["components"]["HardwareRuntime"]["status"] == "PASSED"
    assert report["components"]["HardwareRuntime"]["level1_hardware_detection"]["status"] == "PASSED"
    assert report["components"]["HardwareRuntime"]["level2_synthetic_smoke"]["status"] == "PASSED"
    assert report["components"]["HardwareRuntime"]["level3_registered_workload"]["status"] == "PASSED"
    assert report["status"] == "ALL_PROBES_PASSED"


def test_2_missing_checkpoint_blocked(mock_all_dependencies):
    report = probe_api_compatibility(
        checkpoint_path=None,
        expected_checkpoint_sha256=None,
    )
    assert report["components"]["Checkpoint"]["status"] == "BLOCKED_APPROVED_CHECKPOINT_OR_HASH_MISSING"
    assert report["status"] == "BLOCKED_DEPENDENCIES_OR_CHECKPOINT_UNVERIFIED"
    assert report["status"] != "ALL_PROBES_PASSED"


def test_3_incorrect_checkpoint_hash_fail(mock_all_dependencies, sample_checkpoint_file):
    ckpt_path, _ = sample_checkpoint_file
    report = probe_api_compatibility(
        checkpoint_path=ckpt_path,
        expected_checkpoint_sha256="deadbeef" * 8,  # Deliberate mismatch
    )
    assert report["components"]["Checkpoint"]["status"] == "FAILED_CHECKPOINT_HASH"
    assert report["status"] == "PROBES_FAILED"


def test_4_cuda_detected_but_inference_untested_blocked(mock_all_dependencies, sample_checkpoint_file):
    ckpt_path, expected_sha = sample_checkpoint_file
    report = probe_api_compatibility(
        checkpoint_path=ckpt_path,
        expected_checkpoint_sha256=expected_sha,
        run_inference_smoke=False,  # Untested inference
    )
    assert report["components"]["HardwareRuntime"]["level1_hardware_detection"]["status"] == "PASSED"
    assert report["components"]["HardwareRuntime"]["level2_synthetic_smoke"]["status"] == "BLOCKED_INFERENCE_UNTESTED"
    assert report["components"]["HardwareRuntime"]["status"] == "BLOCKED_INFERENCE_MEMORY_UNVERIFIED"
    assert report["status"] == "BLOCKED_DEPENDENCIES_OR_CHECKPOINT_UNVERIFIED"


def test_5_incorrect_quantile_shape_fail(mock_all_dependencies, monkeypatch, sample_checkpoint_file):
    ckpt_path, expected_sha = sample_checkpoint_file
    # Monkeypatch TabICLRegressor to return incorrect shape (500 instead of 999)
    tabicl_mock = sys.modules["tabicl"]
    tabicl_mock.TabICLRegressor = MockTabICLRegressorBadShape

    report = probe_api_compatibility(
        checkpoint_path=ckpt_path,
        expected_checkpoint_sha256=expected_sha,
    )
    assert report["components"]["TabICLRegressor"]["status"] == "FAILED_GEOMETRY_OR_ALPHA_GRID"
    assert report["status"] == "PROBES_FAILED"


def test_6_missing_official_alpha_levels_never_pass(mock_all_dependencies, monkeypatch, sample_checkpoint_file):
    ckpt_path, expected_sha = sample_checkpoint_file
    # Monkeypatch TabICLRegressor to have alphas_ = None
    tabicl_mock = sys.modules["tabicl"]
    tabicl_mock.TabICLRegressor = MockTabICLRegressorMissingAlphas

    report = probe_api_compatibility(
        checkpoint_path=ckpt_path,
        expected_checkpoint_sha256=expected_sha,
    )
    assert report["components"]["TabICLRegressor"]["status"] == "FAILED_GEOMETRY_OR_ALPHA_GRID"
    assert report["status"] != "ALL_PROBES_PASSED"
    assert report["status"] == "PROBES_FAILED"


def test_7_missing_required_ngboost_distribution_fail(mock_all_dependencies, monkeypatch, sample_checkpoint_file):
    ckpt_path, expected_sha = sample_checkpoint_file
    # Remove TFixedDf from distns
    distns_mock = sys.modules["ngboost.distns"]
    del distns_mock.TFixedDf

    report = probe_api_compatibility(
        checkpoint_path=ckpt_path,
        expected_checkpoint_sha256=expected_sha,
    )
    assert report["components"]["NGBoost"]["status"] == "FAILED_REGISTERED_DISTRIBUTION_SUPPORT"
    assert report["status"] == "PROBES_FAILED"


def test_8_synthetic_smoke_passes_but_registered_workload_unverified_blocked(mock_all_dependencies, sample_checkpoint_file):
    ckpt_path, expected_sha = sample_checkpoint_file
    report = probe_api_compatibility(
        checkpoint_path=ckpt_path,
        expected_checkpoint_sha256=expected_sha,
        run_inference_smoke=True,
        run_workload_test=False,  # Registered workload unverified
    )
    assert report["components"]["HardwareRuntime"]["level1_hardware_detection"]["status"] == "PASSED"
    assert report["components"]["HardwareRuntime"]["level2_synthetic_smoke"]["status"] == "PASSED"
    assert report["components"]["HardwareRuntime"]["level3_registered_workload"]["status"] == "BLOCKED_WORKLOAD_UNVERIFIED"
    assert report["components"]["HardwareRuntime"]["status"] == "BLOCKED_INFERENCE_MEMORY_UNVERIFIED"
    assert report["status"] == "BLOCKED_DEPENDENCIES_OR_CHECKPOINT_UNVERIFIED"


def test_9_fail_plus_blocked_preserves_failure_not_concealed(mock_all_dependencies, sample_checkpoint_file):
    ckpt_path, _ = sample_checkpoint_file
    # Checkpoint has invalid hash (FAIL), but workload is unverified (BLOCKED)
    report = probe_api_compatibility(
        checkpoint_path=ckpt_path,
        expected_checkpoint_sha256="wrong_hash_deadbeef",
        run_workload_test=False,
    )
    assert report["components"]["Checkpoint"]["status"] == "FAILED_CHECKPOINT_HASH"
    assert report["components"]["HardwareRuntime"]["status"] == "BLOCKED_INFERENCE_MEMORY_UNVERIFIED"
    # Invariant: Must report PROBES_FAILED, not conceal under BLOCKED!
    assert report["status"] == "PROBES_FAILED"


def test_diagnostic_only_mode_cannot_pass(mock_all_dependencies, sample_checkpoint_file):
    ckpt_path, expected_sha = sample_checkpoint_file
    report = probe_api_compatibility(
        checkpoint_path=ckpt_path,
        expected_checkpoint_sha256=expected_sha,
        diagnostic_only=True,
        run_workload_test=True,
    )
    assert report["status"] == "DIAGNOSTIC_ONLY_PREFLIGHT_BLOCKED"
    assert report["status"] != "ALL_PROBES_PASSED"


def test_level3_gpu_oom_results_in_failed_cuda_oom(mock_all_dependencies, monkeypatch, sample_checkpoint_file):
    ckpt_path, expected_sha = sample_checkpoint_file
    workload_cfg = {"n_estimators": 10, "context_size": 1000, "max_memory_mb": 11000.0}

    # Patch TabICLRegressor to raise CUDA OutOfMemoryError
    tabicl_mock = sys.modules["tabicl"]
    tabicl_mock.TabICLRegressor = MockTabICLRegressorOOM

    report = probe_api_compatibility(
        checkpoint_path=ckpt_path,
        expected_checkpoint_sha256=expected_sha,
        workload_config=workload_cfg,
        run_inference_smoke=True,
        run_workload_test=True,
    )
    assert report["components"]["HardwareRuntime"]["level3_registered_workload"]["status"] == "FAILED_CUDA_OOM"
    assert report["components"]["HardwareRuntime"]["status"] == "FAILED_HARDWARE_RUNTIME"
    assert report["status"] == "PROBES_FAILED"


def test_level3_exceeds_memory_limit_results_in_failed(mock_all_dependencies, monkeypatch, sample_checkpoint_file):
    ckpt_path, expected_sha = sample_checkpoint_file
    workload_cfg = {"n_estimators": 10, "context_size": 1000, "max_memory_mb": 1000.0}

    # MockTorchValid reports 2048 MB peak memory, which exceeds 1000 MB limit
    report = probe_api_compatibility(
        checkpoint_path=ckpt_path,
        expected_checkpoint_sha256=expected_sha,
        workload_config=workload_cfg,
        run_inference_smoke=True,
        run_workload_test=True,
    )
    assert report["components"]["HardwareRuntime"]["level3_registered_workload"]["status"] == "FAILED_EXCEEDS_MEMORY_LIMIT"
    assert report["components"]["HardwareRuntime"]["status"] == "FAILED_HARDWARE_RUNTIME"
    assert report["status"] == "PROBES_FAILED"


def test_level3_incompatible_geometry_results_in_failed(mock_all_dependencies, monkeypatch, sample_checkpoint_file):
    ckpt_path, expected_sha = sample_checkpoint_file
    workload_cfg = {"n_estimators": 10, "context_size": 1000, "max_memory_mb": 11000.0}

    # Non-monotonic predictions in registered workload
    tabicl_mock = sys.modules["tabicl"]
    tabicl_mock.TabICLRegressor = MockTabICLRegressorNonMonotonic

    report = probe_api_compatibility(
        checkpoint_path=ckpt_path,
        expected_checkpoint_sha256=expected_sha,
        workload_config=workload_cfg,
        run_inference_smoke=True,
        run_workload_test=True,
    )
    assert report["components"]["HardwareRuntime"]["level3_registered_workload"]["status"] == "FAILED_GEOMETRY_OR_ALPHA_GRID"
    assert report["components"]["HardwareRuntime"]["status"] == "FAILED_HARDWARE_RUNTIME"
    assert report["status"] == "PROBES_FAILED"


def test_level3_checkpoint_identity_and_mock_labeling(mock_all_dependencies, sample_checkpoint_file):
    ckpt_path, expected_sha = sample_checkpoint_file
    workload_cfg = {"n_estimators": 10, "context_size": 1000, "max_memory_mb": 11000.0}

    report = probe_api_compatibility(
        checkpoint_path=ckpt_path,
        expected_checkpoint_sha256=expected_sha,
        workload_config=workload_cfg,
        run_inference_smoke=True,
        run_workload_test=True,
    )
    lvl3 = report["components"]["HardwareRuntime"]["level3_registered_workload"]
    assert lvl3["status"] == "PASSED"
    # Checkpoint identity is linked to loaded artifact
    assert lvl3["loaded_checkpoint"]["checkpoint_path"] == str(ckpt_path)
    assert lvl3["loaded_checkpoint"]["checkpoint_sha256"] == expected_sha
    # Mocked model can validate orchestration but cannot constitute real G0 execution evidence
    assert lvl3["mocked"] is True
    assert lvl3["evidence_type"] == "MOCK_ORCHESTRATION_ONLY"


def test_level2_verifies_basic_cuda_smoke_only(mock_all_dependencies, sample_checkpoint_file):
    ckpt_path, expected_sha = sample_checkpoint_file
    report = probe_api_compatibility(
        checkpoint_path=ckpt_path,
        expected_checkpoint_sha256=expected_sha,
        run_inference_smoke=True,
        run_workload_test=False,
    )
    lvl2 = report["components"]["HardwareRuntime"]["level2_synthetic_smoke"]
    assert lvl2["status"] == "PASSED"
    assert lvl2["cuda_functional"] is True
    # Level 2 does not claim registered workload feasibility
    assert report["components"]["HardwareRuntime"]["level3_registered_workload"]["status"] == "BLOCKED_WORKLOAD_UNVERIFIED"


def test_cli_flags_and_truthful_exit_codes(sample_checkpoint_file, tmp_path):
    import subprocess
    ckpt_path, expected_sha = sample_checkpoint_file
    out_json = tmp_path / "preflight_cli_out.json"

    # 1. Diagnostic only -> Exit code 2 (BLOCKED)
    res_diag = subprocess.run(
        [sys.executable, str(Path("proposals/companion-v1/environment/api_preflight.py")), "--diagnostic-only", "--output", str(out_json)],
        capture_output=True,
        text=True,
    )
    assert res_diag.returncode == 2

    # 2. Checkpoint mismatch -> Exit code 1 (FAILED)
    res_fail = subprocess.run(
        [sys.executable, str(Path("proposals/companion-v1/environment/api_preflight.py")), "--checkpoint", str(ckpt_path), "--expected-sha256", "wrong_hash", "--output", str(out_json)],
        capture_output=True,
        text=True,
    )
    assert res_fail.returncode == 1

    # 3. CLI accepts --run-workload-test and --workload-config
    wl_json = tmp_path / "workload.json"
    wl_json.write_text('{"context_size": 50, "n_features": 5, "max_memory_mb": 5000.0}', encoding="utf-8")
    res_wl = subprocess.run(
        [sys.executable, str(Path("proposals/companion-v1/environment/api_preflight.py")), "--workload-config", str(wl_json), "--run-workload-test", "--output", str(out_json)],
        capture_output=True,
        text=True,
    )
    # Exits with 2 because unapproved dependencies/checkpoint are blocked in clean environment
    assert res_wl.returncode == 2
    assert out_json.exists()
