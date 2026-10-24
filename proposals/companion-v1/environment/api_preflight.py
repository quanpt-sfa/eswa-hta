"""
API Compatibility Preflight Probe Script
========================================
Governance: docs/HTA_IMPLEMENTATION_SPEC_v1.2.md (§5 G0, §15.3)
Status: CANDIDATE_PROBE (Executes synthetic probes and hardware verification)

Probes:
  1. TabICLRegressor: constructor, fit on toy data, raw_quantiles shape and alpha grid.
  2. TabICLClassifier: constructor, fit on binary targets, predict_proba class mapping.
  3. CatBoostRegressor: MultiQuantile loss function syntax and quantile count.
  4. NGBoost: Normal, Laplace, and TFixedDf distribution availability and methods.
  5. GPU / CUDA device allocation, 3 verification levels, and peak memory usage.
  6. Checkpoint identity and SHA-256 calculation.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional

import numpy as np


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 of model checkpoint."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def probe_api_compatibility(
    checkpoint_path: Optional[Path] = None,
    expected_checkpoint_sha256: Optional[str] = None,
    diagnostic_only: bool = False,
    workload_config: Optional[Dict[str, Any]] = None,
    run_inference_smoke: bool = True,
    run_workload_test: bool = False,
) -> Dict[str, Any]:
    """Runs automated preflight probe across all modeling engines and hardware tiers."""
    report: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "RUNNING",
        "diagnostic_only": diagnostic_only,
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

    # Record installed package versions
    target_pkgs = ["torch", "tabicl", "catboost", "ngboost", "scikit-learn", "numpy", "pandas", "scipy"]
    for pkg in target_pkgs:
        try:
            report["environment"]["installed_packages"][pkg] = importlib.metadata.version(pkg)
        except Exception:
            report["environment"]["installed_packages"][pkg] = "NOT_INSTALLED"

    # Generate synthetic toy data
    rng = np.random.default_rng(42)
    X_toy = rng.normal(size=(20, 5))
    y_toy = X_toy[:, 0] * 2.0 + rng.normal(size=20)
    y_binary = (y_toy > 0).astype(int)

    # 1. TabICL Regressor Probe
    try:
        from tabicl import TabICLRegressor
        r = TabICLRegressor(n_estimators=1, random_state=42)
        r.fit(X_toy, y_toy)
        Q = r.predict(X_toy[:5], output_type="raw_quantiles")

        # Inspect alphas
        alphas = getattr(r, "alphas_", None)
        shape_ok = tuple(Q.shape) == (5, 999)
        alpha_ok = (
            alphas is not None
            and len(alphas) == Q.shape[1]
            and bool(np.isfinite(alphas).all())
            and bool(np.all(np.diff(alphas) > 0))
            and bool(np.all((np.asarray(alphas) > 0) & (np.asarray(alphas) < 1)))
        )
        quantile_ok = bool(np.isfinite(Q).all() and np.all(np.diff(Q, axis=1) >= 0))
        report["components"]["TabICLRegressor"] = {
            "status": "PASSED" if shape_ok and alpha_ok and quantile_ok else "FAILED_GEOMETRY_OR_ALPHA_GRID",
            "output_shape": list(Q.shape),
            "expected_shape": [5, 999],
            "alphas_exposed": alphas is not None,
            "alphas_count": len(alphas) if alphas is not None else None,
            "monotonic_quantiles": quantile_ok,
        }
    except ImportError as e:
        report["components"]["TabICLRegressor"] = {"status": "BLOCKED_IMPORT_ERROR", "error": str(e)}
    except Exception as e:
        report["components"]["TabICLRegressor"] = {"status": "FAILED", "error": str(e)}

    # 2. TabICL Classifier Probe
    try:
        from tabicl import TabICLClassifier
        clf = TabICLClassifier(n_estimators=1, random_state=42)
        clf.fit(X_toy, y_binary)
        probs = clf.predict_proba(X_toy[:5])
        classes = getattr(clf, "classes_", None)

        classes_ok = classes is not None and set(int(x) for x in classes) == {0, 1}
        prob_ok = (
            probs.shape == (5, 2)
            and bool(np.isfinite(probs).all())
            and bool(np.all((probs >= 0) & (probs <= 1)))
            and bool(np.allclose(np.sum(probs, axis=1), 1.0))
        )
        report["components"]["TabICLClassifier"] = {
            "status": "PASSED" if classes_ok and prob_ok else "FAILED_PROBABILITY_OR_CLASSES",
            "classes_": list(classes) if classes is not None else None,
            "probs_shape": list(probs.shape),
            "sums_to_one": bool(np.allclose(np.sum(probs, axis=1), 1.0)),
        }
    except ImportError as e:
        report["components"]["TabICLClassifier"] = {"status": "BLOCKED_IMPORT_ERROR", "error": str(e)}
    except Exception as e:
        report["components"]["TabICLClassifier"] = {"status": "FAILED", "error": str(e)}

    # 3. CatBoost MultiQuantile Probe
    try:
        from catboost import CatBoostRegressor
        alpha_str = ",".join(f"{a:.2f}" for a in np.linspace(0.01, 0.99, 99))
        cb = CatBoostRegressor(loss_function=f"MultiQuantile:alpha={alpha_str}", iterations=5, random_seed=42, verbose=0)
        cb.fit(X_toy, y_toy)
        cb_preds = cb.predict(X_toy[:5])

        cb_geometry_ok = cb_preds.shape == (5, 99) and bool(np.isfinite(cb_preds).all())
        report["components"]["CatBoostMultiQuantile"] = {
            "status": "PASSED" if cb_geometry_ok else "FAILED_OUTPUT_GEOMETRY",
            "output_shape": list(cb_preds.shape),
            "expected_shape": [5, 99],
            "monotonic_quantiles": bool(np.all(np.diff(cb_preds, axis=1) >= 0)),
        }
    except ImportError as e:
        report["components"]["CatBoostMultiQuantile"] = {"status": "BLOCKED_IMPORT_ERROR", "error": str(e)}
    except Exception as e:
        report["components"]["CatBoostMultiQuantile"] = {"status": "FAILED", "error": str(e)}

    # 4. NGBoost Distributions Probe
    try:
        from ngboost import NGBRegressor
        from ngboost.distns import Normal, Laplace

        ng_results = {}
        for dist_name, dist_cls in [("Normal", Normal), ("Laplace", Laplace)]:
            ng = NGBRegressor(Dist=dist_cls, n_estimators=5, random_state=42, verbose=False)
            ng.fit(X_toy, y_toy)
            preds_dist = ng.pred_dist(X_toy[:5])
            has_cdf = hasattr(preds_dist.dist, "cdf")
            has_sf = hasattr(preds_dist.dist, "sf")
            ng_results[dist_name] = {"fitted": True, "has_cdf": has_cdf, "has_sf": has_sf}

        # Check TFixedDf if available
        try:
            from ngboost.distns import TFixedDf
            ng_results["TFixedDf"] = {"available": True}
        except ImportError:
            ng_results["TFixedDf"] = {"available": False}

        tf_ok = ng_results.get("TFixedDf", {}).get("available", False)
        base_ok = all(ng_results[k]["has_cdf"] and ng_results[k]["has_sf"] for k in ("Normal", "Laplace"))

        report["components"]["NGBoost"] = {
            "status": "PASSED" if tf_ok and base_ok else "FAILED_REGISTERED_DISTRIBUTION_SUPPORT",
            "distributions": ng_results,
        }
    except ImportError as e:
        report["components"]["NGBoost"] = {"status": "BLOCKED_IMPORT_ERROR", "error": str(e)}
    except Exception as e:
        report["components"]["NGBoost"] = {"status": "FAILED", "error": str(e)}

    # 5. Checkpoint Verification
    if checkpoint_path is not None and expected_checkpoint_sha256:
        p = Path(checkpoint_path)
        if not p.exists():
            report["components"]["Checkpoint"] = {
                "status": "BLOCKED_CHECKPOINT_FILE_NOT_FOUND",
                "path": str(p),
                "expected_sha256": expected_checkpoint_sha256,
            }
        else:
            observed_sha = compute_file_sha256(p)
            match = observed_sha.lower() == expected_checkpoint_sha256.lower()
            report["components"]["Checkpoint"] = {
                "status": "PASSED" if match else "FAILED_CHECKPOINT_HASH",
                "path": str(p),
                "sha256": observed_sha,
                "expected_sha256": expected_checkpoint_sha256,
                "hash_matched": match,
            }
    else:
        report["components"]["Checkpoint"] = {
            "status": "BLOCKED_APPROVED_CHECKPOINT_OR_HASH_MISSING"
        }

    # 6. Hardware & Memory Runtime Probe (3 Levels)
    hw_report = {
        "status": "BLOCKED",
        "level1_hardware_detection": {"status": "BLOCKED"},
        "level2_synthetic_smoke": {"status": "BLOCKED"},
        "level3_registered_workload": {"status": "BLOCKED"},
    }
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
        vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if cuda_avail else 0.0

        report["environment"]["pytorch_version"] = getattr(torch, "__version__", None)
        report["environment"]["cuda_version"] = getattr(torch.version, "cuda", None)
        report["environment"]["gpu_name"] = device_name
        report["environment"]["total_vram_gb"] = vram_gb

        # Level 1: Hardware detection
        if cuda_avail:
            hw_report["level1_hardware_detection"] = {
                "status": "PASSED",
                "cuda_available": True,
                "device_name": device_name,
                "total_vram_gb": vram_gb,
            }
        else:
            hw_report["level1_hardware_detection"] = {
                "status": "BLOCKED_CUDA_UNAVAILABLE",
                "cuda_available": False,
                "device_name": "CPU",
                "total_vram_gb": 0.0,
            }

        # Level 2: Synthetic inference smoke test (Basic CUDA functionality only)
        if cuda_avail and run_inference_smoke:
            try:
                # Basic matrix multiply smoke test on GPU device
                t_dummy = torch.randn((20, 5), device="cuda:0")
                _ = torch.matmul(t_dummy, t_dummy.T)
                hw_report["level2_synthetic_smoke"] = {
                    "status": "PASSED",
                    "smoke_test_device": "cuda:0",
                    "cuda_functional": True,
                }
            except Exception as e:
                hw_report["level2_synthetic_smoke"] = {
                    "status": "FAILED_CUDA_SMOKE",
                    "error": str(e),
                    "cuda_functional": False,
                }
        else:
            hw_report["level2_synthetic_smoke"] = {
                "status": "BLOCKED_INFERENCE_UNTESTED",
                "reason": "CUDA unavailable or synthetic inference smoke test not executed",
            }

        # Level 3: Registered workload feasibility
        # Requires execution of actual registered TabICLv2 configuration on GPU
        ckpt_status = report["components"].get("Checkpoint", {}).get("status")
        if not run_workload_test:
            hw_report["level3_registered_workload"] = {
                "status": "BLOCKED_WORKLOAD_UNVERIFIED",
                "reason": "Registered workload verification was not requested via CLI (--run-workload-test).",
            }
        elif workload_config is None:
            hw_report["level3_registered_workload"] = {
                "status": "BLOCKED_WORKLOAD_UNVERIFIED",
                "reason": "Registered workload configuration missing; toy fit does not satisfy Level 3.",
            }
        elif hw_report["level1_hardware_detection"]["status"] != "PASSED":
            hw_report["level3_registered_workload"] = {
                "status": "BLOCKED_CUDA_UNAVAILABLE",
                "reason": "CUDA is unavailable on this host; Level 3 requires GPU execution.",
            }
        elif hw_report["level2_synthetic_smoke"]["status"] != "PASSED":
            hw_report["level3_registered_workload"] = {
                "status": "BLOCKED_INFERENCE_UNTESTED",
                "reason": "Level 2 CUDA smoke test must pass before registered workload execution.",
            }
        elif ckpt_status == "FAILED_CHECKPOINT_HASH":
            hw_report["level3_registered_workload"] = {
                "status": "FAILED_CHECKPOINT_HASH",
                "reason": "Checkpoint hash mismatch; cannot execute unverified model artifact.",
            }
        elif ckpt_status != "PASSED":
            hw_report["level3_registered_workload"] = {
                "status": "BLOCKED_WORKLOAD_UNVERIFIED",
                "reason": "Approved checkpoint is missing or unverified; real model execution cannot proceed.",
            }
        else:
            # Attempt execution of actual registered TabICLv2 configuration
            try:
                from tabicl import TabICLRegressor
            except ImportError as e:
                hw_report["level3_registered_workload"] = {
                    "status": "BLOCKED_WORKLOAD_UNVERIFIED",
                    "reason": f"TabICL package not installed: {e}",
                }
            else:
                max_allowed_mb = float(workload_config.get("max_memory_mb", 11000.0))
                try:
                    tabicl_mod = sys.modules.get("tabicl")
                    is_mock = bool(
                        getattr(TabICLRegressor, "_is_mock", False) is True
                        or (getattr(tabicl_mod, "_is_mock", False) is True)
                        or (tabicl_mod is not None and type(tabicl_mod).__name__ in ("MagicMock", "Mock"))
                        or type(TabICLRegressor).__name__ in ("MagicMock", "Mock")
                        or getattr(TabICLRegressor, "__name__", "").startswith("Mock")
                    )

                    context_size = int(workload_config.get("context_size", 1000))
                    n_features = int(workload_config.get("n_features", 20))
                    n_estimators = int(workload_config.get("n_estimators", 10))

                    rng_wl = np.random.default_rng(42)
                    X_train_wl = rng_wl.normal(size=(context_size, n_features))
                    y_train_wl = rng_wl.normal(size=context_size)
                    X_eval_wl = rng_wl.normal(size=(min(100, context_size), n_features))

                    torch.cuda.reset_peak_memory_stats(0)

                    # Initialize model and link checkpoint
                    init_kwargs = {"n_estimators": n_estimators, "random_state": 42}
                    import inspect
                    sig = inspect.signature(TabICLRegressor.__init__)
                    if "device" in sig.parameters:
                        init_kwargs["device"] = "cuda"
                    if "checkpoint_path" in sig.parameters and checkpoint_path is not None:
                        init_kwargs["checkpoint_path"] = str(checkpoint_path)
                    elif "model_path" in sig.parameters and checkpoint_path is not None:
                        init_kwargs["model_path"] = str(checkpoint_path)

                    model = TabICLRegressor(**init_kwargs)
                    if hasattr(model, "load_checkpoint") and checkpoint_path is not None:
                        model.load_checkpoint(str(checkpoint_path))

                    loaded_ckpt_info = {
                        "checkpoint_path": str(checkpoint_path),
                        "checkpoint_sha256": report["components"]["Checkpoint"].get("sha256"),
                    }

                    # Fit and predict with actual model
                    model.fit(X_train_wl, y_train_wl)
                    Q_wl = model.predict(X_eval_wl, output_type="raw_quantiles")

                    # Measure peak GPU memory from the actual model execution
                    model_peak_mb = round(torch.cuda.max_memory_allocated(0) / (1024**2), 2)
                    report["environment"]["peak_vram_allocated_mb"] = model_peak_mb

                    # Incompatible output geometry or non-monotonic quantiles
                    shape_ok = (isinstance(Q_wl, np.ndarray) and Q_wl.ndim == 2 and Q_wl.shape[0] == len(X_eval_wl) and Q_wl.shape[1] == 999)
                    monotonic_ok = bool(np.isfinite(Q_wl).all() and np.all(np.diff(Q_wl, axis=1) >= 0)) if shape_ok else False

                    if not shape_ok or not monotonic_ok:
                        hw_report["level3_registered_workload"] = {
                            "status": "FAILED_GEOMETRY_OR_ALPHA_GRID",
                            "output_shape": list(Q_wl.shape) if hasattr(Q_wl, "shape") else None,
                            "expected_shape": [len(X_eval_wl), 999],
                            "monotonic": monotonic_ok,
                            "peak_vram_allocated_mb": model_peak_mb,
                        }
                    elif model_peak_mb > max_allowed_mb:
                        hw_report["level3_registered_workload"] = {
                            "status": "FAILED_EXCEEDS_MEMORY_LIMIT",
                            "workload_config": workload_config,
                            "peak_vram_allocated_mb": model_peak_mb,
                            "memory_limit_mb": max_allowed_mb,
                            "loaded_checkpoint": loaded_ckpt_info,
                        }
                    else:
                        evidence_type = "MOCK_ORCHESTRATION_ONLY" if is_mock else "REAL_EXECUTION"
                        hw_report["level3_registered_workload"] = {
                            "status": "PASSED",
                            "workload_config": workload_config,
                            "peak_vram_allocated_mb": model_peak_mb,
                            "memory_limit_mb": max_allowed_mb,
                            "loaded_checkpoint": loaded_ckpt_info,
                            "output_shape": list(Q_wl.shape),
                            "monotonic_quantiles": True,
                            "evidence_type": evidence_type,
                            "mocked": is_mock,
                        }
                except (getattr(torch.cuda, "OutOfMemoryError", MemoryError), MemoryError) as e:
                    hw_report["level3_registered_workload"] = {
                        "status": "FAILED_CUDA_OOM",
                        "error": str(e),
                        "workload_config": workload_config,
                        "memory_limit_mb": max_allowed_mb,
                    }
                except Exception as e:
                    hw_report["level3_registered_workload"] = {
                        "status": "FAILED",
                        "error": str(e),
                        "workload_config": workload_config,
                    }

        # Evaluate aggregated HardwareRuntime status
        hw_levels = [
            hw_report["level1_hardware_detection"]["status"],
            hw_report["level2_synthetic_smoke"]["status"],
            hw_report["level3_registered_workload"]["status"],
        ]
        if any("FAIL" in s for s in hw_levels):
            hw_report["status"] = "FAILED_HARDWARE_RUNTIME"
        elif any("BLOCK" in s or "UNTESTED" in s or "UNVERIFIED" in s for s in hw_levels):
            hw_report["status"] = "BLOCKED_INFERENCE_MEMORY_UNVERIFIED"
        elif all(s == "PASSED" for s in hw_levels):
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

    # Evaluate Overall Status
    statuses = [v.get("status") for v in report["components"].values()]
    has_failure = any(isinstance(s, str) and ("FAIL" in s or s.startswith("ERR")) for s in statuses)
    has_blocked = any(
        isinstance(s, str) and ("BLOCK" in s or "UNVERIFIED" in s or "MISSING" in s or "UNTESTED" in s)
        for s in statuses
    )

    if has_failure:
        report["status"] = "PROBES_FAILED"
    elif diagnostic_only:
        report["status"] = "DIAGNOSTIC_ONLY_PREFLIGHT_BLOCKED"
    elif has_blocked:
        report["status"] = "BLOCKED_DEPENDENCIES_OR_CHECKPOINT_UNVERIFIED"
    elif all(s == "PASSED" for s in statuses):
        report["status"] = "ALL_PROBES_PASSED"
    else:
        report["status"] = "PROBES_FAILED"

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API Compatibility and Environment Preflight Probe")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to approved model checkpoint")
    parser.add_argument("--expected-sha256", type=str, default=None, help="Expected SHA-256 hash of approved checkpoint")
    parser.add_argument("--output", type=Path, default=Path("outputs/G0/api_preflight.json"), help="Destination path for output JSON report")
    parser.add_argument("--diagnostic-only", action="store_true", help="Diagnostic-only mode (cannot produce G0 PASS)")
    parser.add_argument("--workload-config", type=Path, default=None, help="Path to registered workload configuration file (JSON/YAML)")
    parser.add_argument("--run-workload-test", action="store_true", help="Execute Level 3 registered workload verification on GPU")
    parser.add_argument("--no-inference-smoke", dest="run_inference_smoke", action="store_false", default=True, help="Skip Level 2 CUDA inference smoke test")
    args = parser.parse_args()

    workload_cfg = None
    if args.workload_config and args.workload_config.exists():
        with open(args.workload_config, "r", encoding="utf-8") as f:
            workload_cfg = json.load(f)

    result = probe_api_compatibility(
        checkpoint_path=args.checkpoint,
        expected_checkpoint_sha256=args.expected_sha256,
        diagnostic_only=args.diagnostic_only,
        workload_config=workload_cfg,
        run_inference_smoke=args.run_inference_smoke,
        run_workload_test=args.run_workload_test,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))

    if result["status"] == "ALL_PROBES_PASSED":
        sys.exit(0)
    elif "FAIL" in result["status"]:
        sys.exit(1)
    else:
        sys.exit(2)
