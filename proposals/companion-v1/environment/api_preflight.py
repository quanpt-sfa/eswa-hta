"""
API Compatibility Preflight Probe Script
========================================
Governance: docs/HTA_IMPLEMENTATION_SPEC_v1.2.md (§5 G0, §15.3)
Status: CANDIDATE_PROBE

Probes:
  1. TabICLRegressor : approved checkpoint, raw_quantiles geometry and alpha grid.
  2. TabICLClassifier: approved checkpoint, class mapping and predict_proba.
  3. CatBoostRegressor: MultiQuantile loss function syntax and quantile count.
  4. NGBoost: Normal, Laplace, and TFixedDf distribution availability and methods.
  5. GPU / CUDA: Level 1 detection, Level 2 basic CUDA smoke, Level 3 registered workload.
  6. Checkpoint identity: file SHA-256 AND the artifact actually loaded by the model.

False-PASS policy
-----------------
* A mocked / substituted implementation can only yield ``MOCK_ORCHESTRATION_PASSED``.
* An implementation whose identity cannot be resolved yields BLOCKED.
* ``ALL_PROBES_PASSED`` requires independently verified REAL execution of every
  mandatory component (see ``aggregate_overall_status``); the status strings of
  nested components are never trusted without matching evidence fields.
* No workload default is manufactured: the registered workload must be supplied
  completely (``REQUIRED_WORKLOAD_FIELDS``) or Level 3 is BLOCKED.
* The two M1 / B8 artifacts (TabICLRegressor, TabICLClassifier) are approved and
  verified independently; a checkpoint is never shared between them.

TabICL API (https://tabicl.readthedocs.io/en/latest/api.html): constructor
parameters ``model_path``, ``allow_auto_download``, ``device``, ``n_estimators``,
``random_state``, ``use_amp``, ``offload_mode``; fitted attributes
``model_path_`` and ``device_``. Every parameter is verified against the
installed signature before use; nothing is assumed.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import time
import types
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


REQUIRED_WORKLOAD_FIELDS = (
    "context_size",
    "n_features",
    "n_estimators",
    "prediction_batch_size",
    "device",
    "use_amp",
    "offload_mode",
    "max_memory_mb",
)
ALLOWED_DEVICES = ("cuda", "cuda:0")
ALLOWED_OFFLOAD_MODES = ("auto", "gpu", "cpu", "disk")

REQUIRED_COMPONENTS_MODELS = (
    "TabICLRegressor",
    "TabICLClassifier",
    "CatBoostMultiQuantile",
    "NGBoost",
)
REQUIRED_COMPONENTS_CHECKPOINTS = ("RegressorCheckpoint", "ClassifierCheckpoint")

_TABICL_REQUIRED_PARAMS = ("model_path", "allow_auto_download", "device", "n_estimators", "random_state")


class _PreflightBlocked(Exception):
    def __init__(self, status: str, detail: str = ""):
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


class _PreflightFailed(Exception):
    def __init__(self, status: str, detail: str = ""):
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 of model checkpoint."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# ==============================================================================
# Implementation identity (mock / substitution detection)
# ==============================================================================

def implementation_identity(package: str, module_obj: Any, cls: Any = None) -> Tuple[str, List[str]]:
    """
    Classifies an imported implementation as "REAL", "MOCK" or "UNRESOLVED".

    Not name based: the module object must be a genuine module with a file on
    disk, the class (if any) must be defined inside the package, the package
    must have installed distribution metadata, and the run must not be inside a
    test runner. Anything else is not real execution evidence.
    """
    reasons: List[str] = []
    if not isinstance(module_obj, types.ModuleType):
        reasons.append(f"'{package}' is not a genuine module object ({type(module_obj).__name__})")
    elif getattr(module_obj, "__file__", None) is None:
        reasons.append(f"'{package}' module has no file on disk")
    if cls is not None:
        cls_mod = str(getattr(cls, "__module__", ""))
        if cls_mod.split(".")[0] != package:
            reasons.append(f"class defined in '{cls_mod}', not in package '{package}'")
        if type(cls).__module__.startswith("unittest.mock"):
            reasons.append("class is a unittest.mock object")
    if os.environ.get("PYTEST_CURRENT_TEST"):
        reasons.append("executing inside a test runner")
    if reasons:
        return "MOCK", reasons
    try:
        importlib.metadata.version(package)
    except Exception as e:
        return "UNRESOLVED", [f"no installed distribution metadata for '{package}': {e}"]
    return "REAL", []


# ==============================================================================
# Checkpoint and workload validation
# ==============================================================================

def check_checkpoint_file(label: str, path: Optional[Path], expected_sha256: Optional[str]) -> Dict[str, Any]:
    """File-level checkpoint verification (necessary, not sufficient: see verify_loaded_artifact)."""
    if path is None or not expected_sha256:
        return {"status": "BLOCKED_APPROVED_CHECKPOINT_OR_HASH_MISSING", "role": label}
    p = Path(path)
    if not p.is_file():
        return {
            "status": "BLOCKED_CHECKPOINT_FILE_NOT_FOUND",
            "role": label,
            "path": str(p),
            "expected_sha256": expected_sha256,
        }
    observed = compute_file_sha256(p)
    match = observed.lower() == str(expected_sha256).lower()
    return {
        "status": "FILE_HASH_VERIFIED" if match else "FAILED_CHECKPOINT_HASH",
        "role": label,
        "path": str(p),
        "resolved_path": str(p.resolve(strict=True)),
        "sha256": observed,
        "expected_sha256": expected_sha256,
        "hash_matched": match,
    }


def validate_workload_config(cfg: Any) -> Tuple[Optional[str], str]:
    """Strict schema validation; never fills defaults. Returns (blocked_status, detail)."""
    if not isinstance(cfg, dict):
        return "BLOCKED_WORKLOAD_CONFIG_INCOMPLETE", f"missing fields: {sorted(REQUIRED_WORKLOAD_FIELDS)}"
    missing = sorted(set(REQUIRED_WORKLOAD_FIELDS) - set(cfg))
    if missing:
        return "BLOCKED_WORKLOAD_CONFIG_INCOMPLETE", f"missing fields: {missing}"

    def pos_int(v):
        return isinstance(v, int) and not isinstance(v, bool) and v > 0

    for k in ("context_size", "n_features", "n_estimators", "prediction_batch_size"):
        if not pos_int(cfg[k]):
            return "BLOCKED_WORKLOAD_CONFIG_INVALID", f"{k} must be a positive integer, got {cfg[k]!r}"
    mm = cfg["max_memory_mb"]
    if isinstance(mm, bool) or not isinstance(mm, (int, float)) or not np.isfinite(mm) or mm <= 0:
        return "BLOCKED_WORKLOAD_CONFIG_INVALID", f"max_memory_mb must be a positive number, got {mm!r}"
    if cfg["device"] not in ALLOWED_DEVICES:
        return "BLOCKED_WORKLOAD_CONFIG_INVALID", f"device must be one of {ALLOWED_DEVICES}, got {cfg['device']!r}"
    ua = cfg["use_amp"]
    if not (isinstance(ua, bool) or ua == "auto"):
        return "BLOCKED_WORKLOAD_CONFIG_INVALID", f"use_amp must be bool or 'auto', got {ua!r}"
    om = cfg["offload_mode"]
    if not (isinstance(om, bool) or om in ALLOWED_OFFLOAD_MODES):
        return "BLOCKED_WORKLOAD_CONFIG_INVALID", f"offload_mode must be bool or one of {ALLOWED_OFFLOAD_MODES}, got {om!r}"
    if om == "disk" and not isinstance(cfg.get("disk_offload_dir"), str):
        return "BLOCKED_WORKLOAD_CONFIG_INVALID", "offload_mode='disk' requires disk_offload_dir"
    return None, ""


def verify_loaded_artifact(model: Any, ckpt: Dict[str, Any], expected_device: str) -> Dict[str, Any]:
    """
    Verifies the artifact the fitted model REPORTS having loaded (``model_path_``),
    independent of the file the caller believes it supplied. Raises
    _PreflightBlocked / _PreflightFailed; returns identity only after every check.
    """
    loaded_path = getattr(model, "model_path_", None)
    if loaded_path is None:
        raise _PreflightBlocked("BLOCKED_MODEL_ARTIFACT_IDENTITY_UNVERIFIED", "model.model_path_ is missing after fit")
    try:
        actual = Path(loaded_path).resolve(strict=True)
    except (OSError, TypeError) as e:
        raise _PreflightFailed("FAILED_LOADED_CHECKPOINT_MISMATCH", f"loaded path {loaded_path!r} is not an existing file: {e}")
    approved = Path(ckpt["resolved_path"])
    if actual != approved:
        raise _PreflightFailed("FAILED_LOADED_CHECKPOINT_MISMATCH", f"loaded {actual} != approved {approved}")
    loaded_sha = compute_file_sha256(actual)
    if loaded_sha.lower() != str(ckpt["expected_sha256"]).lower():
        raise _PreflightFailed("FAILED_LOADED_CHECKPOINT_HASH", f"loaded artifact sha256 {loaded_sha}")
    device_ = getattr(model, "device_", None)
    if device_ is None:
        raise _PreflightBlocked("BLOCKED_MODEL_DEVICE_UNVERIFIED", "model.device_ is missing after fit")
    if not str(device_).startswith(str(expected_device).split(":")[0]):
        raise _PreflightFailed("FAILED_DEVICE_MISMATCH", f"model.device_={device_} expected {expected_device}")
    return {
        "loaded_model_path": str(actual),
        "loaded_sha256": loaded_sha,
        "loaded_device": str(device_),
        "approved_sha256": ckpt["expected_sha256"],
    }


def _construct_tabicl(cls: Any, ckpt: Dict[str, Any], device: str, n_estimators: int, extra: Dict[str, Any]) -> Any:
    """Constructs a TabICL estimator with the approved checkpoint and auto-download disabled."""
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError) as e:
        raise _PreflightBlocked("BLOCKED_API_INCOMPATIBLE", f"cannot inspect constructor: {e}")
    absent = [p for p in _TABICL_REQUIRED_PARAMS if p not in params]
    absent += [p for p in extra if p not in params]
    if absent:
        raise _PreflightBlocked("BLOCKED_API_INCOMPATIBLE", f"installed constructor lacks parameters {absent}")
    return cls(
        model_path=str(ckpt["resolved_path"]),
        allow_auto_download=False,
        device=device,
        n_estimators=n_estimators,
        random_state=42,
        **extra,
    )


def _fit_predict_verified(
    kind: str,
    cls: Any,
    ckpt: Dict[str, Any],
    device: str,
    n_estimators: int,
    extra: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    X_eval: np.ndarray,
    row_chunk: int,
) -> Dict[str, Any]:
    model = _construct_tabicl(cls, ckpt, device, n_estimators, extra)
    model.fit(X, y)
    loaded = verify_loaded_artifact(model, ckpt, device)
    outs = []
    for i in range(0, len(X_eval), row_chunk):
        chunk = X_eval[i:i + row_chunk]
        if kind == "regressor":
            outs.append(np.asarray(model.predict(chunk, output_type="raw_quantiles")))
        else:
            outs.append(np.asarray(model.predict_proba(chunk)))
    return {"model": model, "output": np.concatenate(outs, axis=0), "loaded": loaded}


def _regressor_geometry(model: Any, Q: np.ndarray, n_rows: int) -> Dict[str, Any]:
    alphas = getattr(model, "alphas_", None)
    shape_ok = Q.ndim == 2 and tuple(Q.shape) == (n_rows, 999)
    alpha_ok = False
    if alphas is not None:
        a = np.asarray(alphas, dtype=float)
        alpha_ok = bool(
            a.ndim == 1 and a.shape[0] == (Q.shape[1] if Q.ndim == 2 else -1)
            and np.isfinite(a).all() and np.all(np.diff(a) > 0) and np.all((a > 0) & (a < 1))
        )
    quantile_ok = bool(shape_ok and np.isfinite(Q).all() and np.all(np.diff(Q, axis=1) >= 0))
    a_arr = np.asarray(alphas, dtype=float) if alphas is not None else None
    return {
        "ok": bool(shape_ok and alpha_ok and quantile_ok),
        "output_shape": list(Q.shape),
        "expected_shape": [n_rows, 999],
        "alphas_exposed": alphas is not None,
        "alphas_count": int(a_arr.shape[0]) if a_arr is not None and a_arr.ndim == 1 else None,
        "alpha_min": float(a_arr.min()) if a_arr is not None and a_arr.size else None,
        "alpha_max": float(a_arr.max()) if a_arr is not None and a_arr.size else None,
        "monotonic_quantiles": quantile_ok,
    }


def _classifier_geometry(model: Any, P: np.ndarray, n_rows: int) -> Dict[str, Any]:
    classes = getattr(model, "classes_", None)
    classes_ok = classes is not None and set(int(x) for x in classes) == {0, 1}
    prob_ok = bool(
        P.ndim == 2 and tuple(P.shape) == (n_rows, 2) and np.isfinite(P).all()
        and np.all((P >= 0) & (P <= 1)) and np.allclose(P.sum(axis=1), 1.0)
    )
    return {
        "ok": bool(classes_ok and prob_ok),
        "classes_": [int(x) for x in classes] if classes is not None else None,
        "probs_shape": list(P.shape),
        "output_shape": list(P.shape),
        "expected_shape": [n_rows, 2],
    }


# ==============================================================================
# Aggregation (never trusts status strings alone)
# ==============================================================================

def _evidence_inconsistencies(report: Dict[str, Any]) -> List[str]:
    """Returns every place where a PASS-like claim is not backed by real, verified evidence."""
    problems: List[str] = []
    comps = report.get("components", {})
    for name in ("TabICLRegressor", "TabICLClassifier"):
        c = comps.get(name, {})
        if c.get("status") == "PASSED":
            if c.get("mocked") is not False or c.get("evidence_type") != "REAL_EXECUTION":
                problems.append(f"{name} claims PASSED without real-execution evidence")
            if c.get("loaded_checkpoint_verified") is not True:
                problems.append(f"{name} claims PASSED without verified loaded checkpoint")
    for name in ("CatBoostMultiQuantile", "NGBoost"):
        c = comps.get(name, {})
        if c.get("status") == "PASSED" and (c.get("mocked") is not False or c.get("evidence_type") != "REAL_EXECUTION"):
            problems.append(f"{name} claims PASSED without real-execution evidence")
    hw = comps.get("HardwareRuntime", {})
    l3 = hw.get("level3_registered_workload", {})
    if hw.get("status") == "PASSED" or l3.get("status") == "PASSED":
        if hw.get("torch_real") is not True:
            problems.append("HardwareRuntime claims PASSED without a real torch/CUDA implementation")
        if l3.get("mocked") is not False or l3.get("evidence_type") != "REAL_EXECUTION":
            problems.append("Level 3 claims PASSED without real-execution evidence")
        if l3.get("loaded_checkpoint_verified") is not True:
            problems.append("Level 3 claims PASSED without verified loaded checkpoints")
        if l3.get("workload_configuration_verified") is not True:
            problems.append("Level 3 claims PASSED without a verified registered workload configuration")
        if l3.get("registered_workload_executed") is not True:
            problems.append("Level 3 claims PASSED but the registered workload was not executed")
    return problems


def aggregate_overall_status(report: Dict[str, Any], diagnostic_only: bool = False) -> str:
    """
    Overall decision. Order: any FAIL -> PROBES_FAILED; diagnostic -> never PASS;
    anything unverified / mocked / inconsistent -> BLOCKED; otherwise
    ALL_PROBES_PASSED only if every mandatory component is real and verified.
    """
    comps = report.get("components", {})
    statuses = [c.get("status") for c in comps.values()]
    if any(isinstance(s, str) and ("FAIL" in s or s.startswith("ERR")) for s in statuses):
        return "PROBES_FAILED"
    if diagnostic_only:
        return "DIAGNOSTIC_ONLY_PREFLIGHT_BLOCKED"

    inconsistencies = _evidence_inconsistencies(report)
    report["evidence_audit"] = {"inconsistencies": inconsistencies}
    if inconsistencies:
        return "BLOCKED_INCONSISTENT_EVIDENCE"

    all_real = (
        all(comps.get(n, {}).get("status") == "PASSED" for n in REQUIRED_COMPONENTS_MODELS)
        and all(comps.get(n, {}).get("status") == "FILE_HASH_VERIFIED" for n in REQUIRED_COMPONENTS_CHECKPOINTS)
        and comps.get("HardwareRuntime", {}).get("status") == "PASSED"
        and comps.get("HardwareRuntime", {}).get("level3_registered_workload", {}).get("status") == "PASSED"
    )
    if not all_real:
        return "BLOCKED_DEPENDENCIES_OR_CHECKPOINT_UNVERIFIED"

    # Logical invariant (not a substitute for the evidence above).
    l3 = comps["HardwareRuntime"]["level3_registered_workload"]
    no_mock_execution = not any(comps.get(n, {}).get("mocked") for n in REQUIRED_COMPONENTS_MODELS) and l3["mocked"] is False
    if not (no_mock_execution and l3["loaded_checkpoint_verified"] and l3["registered_workload_executed"]):
        return "BLOCKED_INCONSISTENT_EVIDENCE"
    return "ALL_PROBES_PASSED"


# ==============================================================================
# Probe
# ==============================================================================

def _model_component(
    kind: str,
    ckpt: Dict[str, Any],
    workload_config: Optional[Dict[str, Any]],
    X_toy: np.ndarray,
    y_toy: np.ndarray,
) -> Dict[str, Any]:
    """Toy API probe of one TabICL estimator using ITS OWN approved checkpoint."""
    name = "TabICLRegressor" if kind == "regressor" else "TabICLClassifier"
    try:
        import tabicl
        cls = getattr(tabicl, name)
    except ImportError as e:
        return {"status": "BLOCKED_IMPORT_ERROR", "error": str(e), "mocked": None, "evidence_type": "NONE"}
    except AttributeError as e:
        return {"status": "BLOCKED_API_INCOMPATIBLE", "error": str(e), "mocked": None, "evidence_type": "NONE"}
    identity, reasons = implementation_identity("tabicl", tabicl, cls)
    base = {"implementation_identity": identity, "identity_reasons": reasons, "loaded_checkpoint_verified": False}
    if ckpt.get("status") != "FILE_HASH_VERIFIED":
        return {**base, "status": "BLOCKED_APPROVED_CHECKPOINT_UNVERIFIED", "mocked": identity != "REAL",
                "evidence_type": "NONE", "checkpoint_status": ckpt.get("status")}
    device = (workload_config or {}).get("device")
    if device not in ALLOWED_DEVICES:
        return {**base, "status": "BLOCKED_WORKLOAD_CONFIG_INCOMPLETE", "mocked": identity != "REAL",
                "evidence_type": "NONE", "error": "approved runtime device not specified"}
    try:
        res = _fit_predict_verified(kind, cls, ckpt, device, 1, {}, X_toy, y_toy, X_toy[:5], 5)
    except (_PreflightBlocked, _PreflightFailed) as e:
        return {**base, "status": e.status, "error": e.detail, "mocked": identity != "REAL", "evidence_type": "NONE"}
    except Exception as e:
        return {**base, "status": "FAILED", "error": str(e), "mocked": identity != "REAL", "evidence_type": "NONE"}
    geom = (_regressor_geometry if kind == "regressor" else _classifier_geometry)(res["model"], res["output"], 5)
    ok = geom.pop("ok")
    if not ok:
        status = "FAILED_GEOMETRY_OR_ALPHA_GRID" if kind == "regressor" else "FAILED_PROBABILITY_OR_CLASSES"
        return {**base, **geom, "status": status, "mocked": identity != "REAL", "evidence_type": "NONE",
                "loaded_artifact": res["loaded"]}
    out = {**base, **geom, "loaded_artifact": res["loaded"], "loaded_checkpoint_verified": True}
    if identity == "REAL":
        return {**out, "status": "PASSED", "mocked": False, "evidence_type": "REAL_EXECUTION"}
    if identity == "MOCK":
        return {**out, "status": "MOCK_ORCHESTRATION_PASSED", "mocked": True, "evidence_type": "MOCK_ORCHESTRATION_ONLY"}
    return {**out, "status": "BLOCKED_MODEL_IMPLEMENTATION_UNRESOLVED", "mocked": None, "evidence_type": "NONE"}


def _booster_status(identity: str, ok_status: str) -> Tuple[str, Any, str]:
    if identity == "REAL":
        return ok_status, False, "REAL_EXECUTION"
    if identity == "MOCK":
        return "MOCK_ORCHESTRATION_PASSED", True, "MOCK_ORCHESTRATION_ONLY"
    return "BLOCKED_MODEL_IMPLEMENTATION_UNRESOLVED", None, "NONE"


def probe_api_compatibility(
    regressor_checkpoint: Optional[Path] = None,
    regressor_sha256: Optional[str] = None,
    classifier_checkpoint: Optional[Path] = None,
    classifier_sha256: Optional[str] = None,
    diagnostic_only: bool = False,
    workload_config: Optional[Dict[str, Any]] = None,
    run_inference_smoke: bool = True,
    run_workload_test: bool = False,
    checkpoint_path: Optional[Path] = None,
    expected_checkpoint_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Runs automated preflight probe across all modeling engines and hardware tiers.

    ``checkpoint_path`` / ``expected_checkpoint_sha256`` are DEPRECATED aliases for
    the REGRESSOR artifact only. They are never applied to the classifier.
    """
    deprecated_used = checkpoint_path is not None or expected_checkpoint_sha256 is not None
    if regressor_checkpoint is None and checkpoint_path is not None:
        regressor_checkpoint = checkpoint_path
    if regressor_sha256 is None and expected_checkpoint_sha256 is not None:
        regressor_sha256 = expected_checkpoint_sha256

    report: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "RUNNING",
        "diagnostic_only": diagnostic_only,
        "deprecated_single_checkpoint_argument_used": deprecated_used,
        "environment": {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "installed_packages": {},
            "pytorch_version": None,
            "cuda_version": None,
            "gpu_name": None,
            "total_vram_gb": 0.0,
            "peak_vram_allocated_mb": 0.0,
        },
        "components": {},
    }

    target_pkgs = ["torch", "tabicl", "catboost", "ngboost", "scikit-learn", "numpy", "pandas", "scipy"]
    for pkg in target_pkgs:
        try:
            report["environment"]["installed_packages"][pkg] = importlib.metadata.version(pkg)
        except Exception:
            report["environment"]["installed_packages"][pkg] = "NOT_INSTALLED"

    rng = np.random.default_rng(42)
    X_toy = rng.normal(size=(20, 5))
    y_toy = X_toy[:, 0] * 2.0 + rng.normal(size=20)
    y_binary = (y_toy > 0).astype(int)

    # Checkpoint artifacts (independent per model; file-level evidence)
    reg_ckpt = check_checkpoint_file("regressor", regressor_checkpoint, regressor_sha256)
    clf_ckpt = check_checkpoint_file("classifier", classifier_checkpoint, classifier_sha256)
    report["components"]["RegressorCheckpoint"] = reg_ckpt
    report["components"]["ClassifierCheckpoint"] = clf_ckpt

    cfg_for_toy = workload_config if isinstance(workload_config, dict) else None

    # 1/2. TabICL probes with their own approved artifacts
    report["components"]["TabICLRegressor"] = _model_component("regressor", reg_ckpt, cfg_for_toy, X_toy, y_toy)
    report["components"]["TabICLClassifier"] = _model_component("classifier", clf_ckpt, cfg_for_toy, X_toy, y_binary)

    # 3. CatBoost MultiQuantile Probe
    try:
        import catboost
        from catboost import CatBoostRegressor
        identity, reasons = implementation_identity("catboost", catboost, CatBoostRegressor)
        alpha_str = ",".join(f"{a:.2f}" for a in np.linspace(0.01, 0.99, 99))
        cb = CatBoostRegressor(loss_function=f"MultiQuantile:alpha={alpha_str}", iterations=5, random_seed=42, verbose=0)
        cb.fit(X_toy, y_toy)
        cb_preds = np.asarray(cb.predict(X_toy[:5]))
        cb_geometry_ok = cb_preds.shape == (5, 99) and bool(np.isfinite(cb_preds).all())
        if cb_geometry_ok:
            st, mocked, ev = _booster_status(identity, "PASSED")
        else:
            st, mocked, ev = "FAILED_OUTPUT_GEOMETRY", identity != "REAL", "NONE"
        report["components"]["CatBoostMultiQuantile"] = {
            "status": st, "mocked": mocked, "evidence_type": ev,
            "implementation_identity": identity, "identity_reasons": reasons,
            "output_shape": list(cb_preds.shape), "expected_shape": [5, 99],
            "monotonic_quantiles": bool(np.all(np.diff(cb_preds, axis=1) >= 0)),
        }
    except ImportError as e:
        report["components"]["CatBoostMultiQuantile"] = {"status": "BLOCKED_IMPORT_ERROR", "error": str(e), "mocked": None, "evidence_type": "NONE"}
    except Exception as e:
        report["components"]["CatBoostMultiQuantile"] = {"status": "FAILED", "error": str(e), "mocked": None, "evidence_type": "NONE"}

    # 4. NGBoost Distributions Probe
    try:
        import ngboost
        from ngboost import NGBRegressor
        from ngboost.distns import Normal, Laplace
        identity, reasons = implementation_identity("ngboost", ngboost, NGBRegressor)

        ng_results = {}
        for dist_name, dist_cls in [("Normal", Normal), ("Laplace", Laplace)]:
            ng = NGBRegressor(Dist=dist_cls, n_estimators=5, random_state=42, verbose=False)
            ng.fit(X_toy, y_toy)
            preds_dist = ng.pred_dist(X_toy[:5])
            ng_results[dist_name] = {
                "fitted": True,
                "has_cdf": hasattr(preds_dist.dist, "cdf"),
                "has_sf": hasattr(preds_dist.dist, "sf"),
            }
        try:
            from ngboost.distns import TFixedDf  # noqa: F401
            ng_results["TFixedDf"] = {"available": True}
        except ImportError:
            ng_results["TFixedDf"] = {"available": False}

        tf_ok = ng_results["TFixedDf"]["available"]
        base_ok = all(ng_results[k]["has_cdf"] and ng_results[k]["has_sf"] for k in ("Normal", "Laplace"))
        if tf_ok and base_ok:
            st, mocked, ev = _booster_status(identity, "PASSED")
        else:
            st, mocked, ev = "FAILED_REGISTERED_DISTRIBUTION_SUPPORT", identity != "REAL", "NONE"
        report["components"]["NGBoost"] = {
            "status": st, "mocked": mocked, "evidence_type": ev,
            "implementation_identity": identity, "identity_reasons": reasons,
            "distributions": ng_results,
        }
    except ImportError as e:
        report["components"]["NGBoost"] = {"status": "BLOCKED_IMPORT_ERROR", "error": str(e), "mocked": None, "evidence_type": "NONE"}
    except Exception as e:
        report["components"]["NGBoost"] = {"status": "FAILED", "error": str(e), "mocked": None, "evidence_type": "NONE"}

    # 5. Hardware & workload (3 levels)
    hw_report: Dict[str, Any] = {
        "status": "BLOCKED",
        "torch_real": False,
        "level1_hardware_detection": {"status": "BLOCKED"},
        "level2_synthetic_smoke": {"status": "BLOCKED"},
        "level3_registered_workload": {
            "status": "BLOCKED_WORKLOAD_UNVERIFIED",
            "mocked": None,
            "evidence_type": "NONE",
            "loaded_checkpoint_verified": False,
            "workload_configuration_verified": False,
            "registered_workload_executed": False,
        },
    }
    try:
        import torch
        torch_identity, torch_reasons = implementation_identity("torch", torch)
        hw_report["torch_real"] = torch_identity == "REAL"
        hw_report["torch_identity"] = torch_identity
        hw_report["torch_identity_reasons"] = torch_reasons

        cuda_avail = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
        vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if cuda_avail else 0.0
        report["environment"]["pytorch_version"] = getattr(torch, "__version__", None)
        report["environment"]["cuda_version"] = getattr(torch.version, "cuda", None)
        report["environment"]["gpu_name"] = device_name
        report["environment"]["total_vram_gb"] = vram_gb

        # Level 1: hardware detection
        if cuda_avail:
            hw_report["level1_hardware_detection"] = {
                "status": "PASSED", "cuda_available": True, "device_name": device_name, "total_vram_gb": vram_gb,
            }
        else:
            hw_report["level1_hardware_detection"] = {
                "status": "BLOCKED_CUDA_UNAVAILABLE", "cuda_available": False, "device_name": "CPU", "total_vram_gb": 0.0,
            }

        # Level 2: basic CUDA functionality ONLY
        if cuda_avail and run_inference_smoke:
            try:
                t_dummy = torch.randn((20, 5), device="cuda:0")
                _ = torch.matmul(t_dummy, t_dummy.T)
                hw_report["level2_synthetic_smoke"] = {
                    "status": "PASSED", "smoke_test_device": "cuda:0", "cuda_functional": True,
                }
            except Exception as e:
                hw_report["level2_synthetic_smoke"] = {"status": "FAILED_CUDA_SMOKE", "error": str(e), "cuda_functional": False}
        else:
            hw_report["level2_synthetic_smoke"] = {
                "status": "BLOCKED_INFERENCE_UNTESTED",
                "reason": "CUDA unavailable or synthetic inference smoke test not executed",
            }

        # Level 3: registered workload (actual registered TabICL configuration)
        hw_report["level3_registered_workload"] = _level3(
            torch, torch_identity, hw_report, reg_ckpt, clf_ckpt, workload_config, run_workload_test,
        )
        peak = hw_report["level3_registered_workload"].get("peak_vram_allocated_mb")
        if isinstance(peak, (int, float)):
            report["environment"]["peak_vram_allocated_mb"] = peak

        lvl = [hw_report[k]["status"] for k in ("level1_hardware_detection", "level2_synthetic_smoke", "level3_registered_workload")]
        if any("FAIL" in s for s in lvl):
            hw_report["status"] = "FAILED_HARDWARE_RUNTIME"
        elif any("BLOCK" in s or "UNTESTED" in s or "UNVERIFIED" in s for s in lvl):
            hw_report["status"] = "BLOCKED_INFERENCE_MEMORY_UNVERIFIED"
        elif lvl[2] == "MOCK_ORCHESTRATION_PASSED" or not hw_report["torch_real"]:
            hw_report["status"] = "MOCK_ORCHESTRATION_PASSED"
        elif all(s == "PASSED" for s in lvl):
            hw_report["status"] = "PASSED"
        else:
            hw_report["status"] = "BLOCKED_INFERENCE_MEMORY_UNVERIFIED"
    except ImportError as e:
        hw_report["status"] = "BLOCKED_IMPORT_ERROR"
        hw_report["error"] = str(e)
    except Exception as e:
        hw_report["status"] = "FAILED_HARDWARE_RUNTIME"
        hw_report["error"] = str(e)

    report["components"]["HardwareRuntime"] = hw_report

    report["status"] = aggregate_overall_status(report, diagnostic_only=diagnostic_only)
    report["contains_mock_execution"] = any(
        c.get("mocked") is True
        or c.get("status") == "MOCK_ORCHESTRATION_PASSED"
        or c.get("level3_registered_workload", {}).get("mocked") is True
        for c in report["components"].values()
    )
    return report


def _level3(
    torch: Any,
    torch_identity: str,
    hw_report: Dict[str, Any],
    reg_ckpt: Dict[str, Any],
    clf_ckpt: Dict[str, Any],
    workload_config: Optional[Dict[str, Any]],
    run_workload_test: bool,
) -> Dict[str, Any]:
    base = {
        "mocked": None,
        "evidence_type": "NONE",
        "loaded_checkpoint_verified": False,
        "workload_configuration_verified": False,
        "registered_workload_executed": False,
    }

    def blocked(status: str, reason: str) -> Dict[str, Any]:
        return {**base, "status": status, "reason": reason}

    if not run_workload_test:
        return blocked("BLOCKED_WORKLOAD_UNVERIFIED", "Registered workload verification not requested (--run-workload-test).")
    cfg_status, cfg_detail = validate_workload_config(workload_config)
    if cfg_status:
        return blocked(cfg_status, cfg_detail)
    cfg = dict(workload_config)  # validated; no defaults injected
    base["workload_configuration_verified"] = True
    if hw_report["level1_hardware_detection"]["status"] != "PASSED":
        return {**base, "status": "BLOCKED_CUDA_UNAVAILABLE", "reason": "Level 3 requires CUDA execution."}
    if hw_report["level2_synthetic_smoke"]["status"] != "PASSED":
        return {**base, "status": "BLOCKED_INFERENCE_UNTESTED", "reason": "Level 2 CUDA smoke must pass first."}
    for ck in (reg_ckpt, clf_ckpt):
        if ck.get("status") == "FAILED_CHECKPOINT_HASH":
            return {**base, "status": "FAILED_CHECKPOINT_HASH", "reason": f"{ck.get('role')} checkpoint hash mismatch."}
    for ck in (reg_ckpt, clf_ckpt):
        if ck.get("status") != "FILE_HASH_VERIFIED":
            return {**base, "status": "BLOCKED_APPROVED_CHECKPOINT_UNVERIFIED",
                    "reason": f"{ck.get('role')} checkpoint unverified ({ck.get('status')}); M1 requires both."}
    for k in ("regressor_sha256", "classifier_sha256"):
        role_ck = reg_ckpt if k.startswith("regressor") else clf_ckpt
        if k in cfg and str(cfg[k]).lower() != str(role_ck["expected_sha256"]).lower():
            return {**base, "status": "BLOCKED_WORKLOAD_CONFIG_INVALID",
                    "reason": f"workload {k} disagrees with the approved checkpoint hash."}

    try:
        import tabicl
        Reg = tabicl.TabICLRegressor
        Clf = tabicl.TabICLClassifier
    except ImportError as e:
        return {**base, "status": "BLOCKED_WORKLOAD_UNVERIFIED", "reason": f"TabICL package not installed: {e}"}
    except AttributeError as e:
        return {**base, "status": "BLOCKED_API_INCOMPATIBLE", "reason": str(e)}
    ident_r, why_r = implementation_identity("tabicl", tabicl, Reg)
    ident_c, why_c = implementation_identity("tabicl", tabicl, Clf)
    if "UNRESOLVED" in (ident_r, ident_c, torch_identity):
        return {**base, "status": "BLOCKED_MODEL_IMPLEMENTATION_UNRESOLVED",
                "reason": f"identity unresolved: {why_r + why_c}"}
    mocked = "MOCK" in (ident_r, ident_c, torch_identity)

    extra = {"use_amp": cfg["use_amp"], "offload_mode": cfg["offload_mode"]}
    if "disk_offload_dir" in cfg:
        extra["disk_offload_dir"] = cfg["disk_offload_dir"]

    n_ctx, n_feat = cfg["context_size"], cfg["n_features"]
    n_eval = cfg["prediction_batch_size"]
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_ctx, n_feat))
    y_reg = X[:, 0] * 2.0 + rng.normal(size=n_ctx)
    y_bin = (y_reg > np.median(y_reg)).astype(int)
    X_eval = rng.normal(size=(n_eval, n_feat))

    oom = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
    oom_types = ((oom,) if isinstance(oom, type) else ()) + (MemoryError,)

    record: Dict[str, Any] = {
        **base,
        "mocked": mocked,
        "actual_parameters": {k: cfg[k] for k in cfg},
        "synthetic_data": {"X_train_shape": [n_ctx, n_feat], "X_eval_shape": [n_eval, n_feat], "seed": 42},
        "implementation_identity": {"regressor": ident_r, "classifier": ident_c, "torch": torch_identity},
        "evidence_type": "MOCK_ORCHESTRATION_ONLY" if mocked else "REAL_EXECUTION",
        "exceptions": [],
        "models": {},
    }
    limit = float(cfg["max_memory_mb"])
    peaks = []
    try:
        for kind, cls, ck, y in (("regressor", Reg, reg_ckpt, y_reg), ("classifier", Clf, clf_ckpt, y_bin)):
            torch.cuda.reset_peak_memory_stats(0)
            res = _fit_predict_verified(kind, cls, ck, cfg["device"], cfg["n_estimators"], extra,
                                        X, y, X_eval, cfg["prediction_batch_size"])
            alloc = round(torch.cuda.max_memory_allocated(0) / (1024**2), 2)
            reserved_fn = getattr(torch.cuda, "max_memory_reserved", None)
            reserved = round(reserved_fn(0) / (1024**2), 2) if callable(reserved_fn) else None
            peaks.append(alloc)
            geom = (_regressor_geometry if kind == "regressor" else _classifier_geometry)(res["model"], res["output"], n_eval)
            ok = geom.pop("ok")
            record["models"][kind] = {
                **geom,
                "peak_vram_allocated_mb": alloc,
                "peak_vram_reserved_mb": reserved,
                "loaded_artifact": res["loaded"],
            }
            if not ok:
                return {**record, "status": "FAILED_GEOMETRY_OR_ALPHA_GRID" if kind == "regressor" else "FAILED_PROBABILITY_OR_CLASSES",
                        "registered_workload_executed": True}
    except _PreflightBlocked as e:
        record["exceptions"].append(f"{e.status}: {e.detail}")
        return {**record, "status": e.status, "reason": e.detail}
    except _PreflightFailed as e:
        record["exceptions"].append(f"{e.status}: {e.detail}")
        return {**record, "status": e.status, "reason": e.detail, "registered_workload_executed": True}
    except oom_types as e:
        record["exceptions"].append(f"CUDA OOM: {e}")
        return {**record, "status": "FAILED_CUDA_OOM", "error": str(e), "memory_limit_mb": limit}
    except Exception as e:
        record["exceptions"].append(f"{type(e).__name__}: {e}")
        return {**record, "status": "FAILED_WORKLOAD_EXECUTION", "error": str(e)}

    peak = max(peaks)
    record.update({
        "registered_workload_executed": True,
        "loaded_checkpoint_verified": True,
        "peak_vram_allocated_mb": peak,
        "memory_limit_mb": limit,
    })
    if peak > limit:
        return {**record, "status": "FAILED_EXCEEDS_MEMORY_LIMIT"}
    return {**record, "status": "MOCK_ORCHESTRATION_PASSED" if mocked else "PASSED"}


def _exit_code(status: str) -> int:
    if status == "ALL_PROBES_PASSED":
        return 0
    if "FAIL" in status:
        return 1
    return 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API Compatibility and Environment Preflight Probe")
    parser.add_argument("--regressor-checkpoint", type=Path, default=None, help="Approved TabICLRegressor checkpoint")
    parser.add_argument("--regressor-sha256", type=str, default=None, help="Approved SHA-256 of the regressor checkpoint")
    parser.add_argument("--classifier-checkpoint", type=Path, default=None, help="Approved TabICLClassifier checkpoint")
    parser.add_argument("--classifier-sha256", type=str, default=None, help="Approved SHA-256 of the classifier checkpoint")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="DEPRECATED alias of --regressor-checkpoint (never applied to the classifier)")
    parser.add_argument("--expected-sha256", type=str, default=None,
                        help="DEPRECATED alias of --regressor-sha256 (never applied to the classifier)")
    parser.add_argument("--output", type=Path, default=Path("outputs/G0/api_preflight.json"), help="Destination path for output JSON report")
    parser.add_argument("--diagnostic-only", action="store_true", help="Diagnostic-only mode (cannot produce G0 PASS)")
    parser.add_argument("--workload-config", type=Path, default=None, help="Path to registered workload configuration (JSON)")
    parser.add_argument("--run-workload-test", action="store_true", help="Execute Level 3 registered workload verification on GPU")
    parser.add_argument("--no-inference-smoke", dest="run_inference_smoke", action="store_false", default=True,
                        help="Skip Level 2 CUDA smoke test")
    args = parser.parse_args()

    if args.checkpoint or args.expected_sha256:
        print("WARNING: --checkpoint/--expected-sha256 are deprecated and apply to the REGRESSOR only; "
              "the classifier checkpoint must be supplied explicitly.", file=sys.stderr)

    workload_cfg: Optional[Dict[str, Any]] = None
    if args.workload_config is not None:
        try:
            with open(args.workload_config, "r", encoding="utf-8") as f:
                workload_cfg = json.load(f)
        except (OSError, ValueError) as e:
            print(f"WARNING: workload config unreadable ({e}); Level 3 will be BLOCKED.", file=sys.stderr)
            workload_cfg = {}

    result = probe_api_compatibility(
        regressor_checkpoint=args.regressor_checkpoint,
        regressor_sha256=args.regressor_sha256,
        classifier_checkpoint=args.classifier_checkpoint,
        classifier_sha256=args.classifier_sha256,
        diagnostic_only=args.diagnostic_only,
        workload_config=workload_cfg,
        run_inference_smoke=args.run_inference_smoke,
        run_workload_test=args.run_workload_test,
        checkpoint_path=args.checkpoint,
        expected_checkpoint_sha256=args.expected_sha256,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    print(json.dumps(result, indent=2, default=str))
    sys.exit(_exit_code(result["status"]))
