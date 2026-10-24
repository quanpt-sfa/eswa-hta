"""
API Compatibility Preflight Probe Script
========================================
Governance: docs/HTA_IMPLEMENTATION_SPEC_v1.2.md (§5 G0, §15.3)
Status: CANDIDATE_PROBE (Executes synthetic probes when dependencies are installed)

Probes:
  1. TabICLRegressor: constructor, fit on toy data, raw_quantiles shape and alpha grid.
  2. TabICLClassifier: constructor, fit on binary targets, predict_proba class mapping.
  3. CatBoostRegressor: MultiQuantile loss function syntax and quantile count.
  4. NGBoost: Normal, Laplace, and TFixedDf distribution availability and methods.
  5. GPU / CUDA device allocation and peak memory usage.
  6. Checkpoint identity and SHA-256 calculation.
"""

import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict

import numpy as np


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 of model checkpoint."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def probe_api_compatibility(
    checkpoint_path: Path = None,
    expected_checkpoint_sha256: str = None
) -> Dict[str, Any]:
    """Runs automated preflight probe across all modeling engines."""
    report: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "RUNNING",
        "components": {}
    }

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
            "monotonic_quantiles": quantile_ok
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
            "sums_to_one": bool(np.allclose(np.sum(probs, axis=1), 1.0))
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
            "monotonic_quantiles": bool(np.all(np.diff(cb_preds, axis=1) >= 0))
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

        report["components"]["NGBoost"] = {
            "status": "PASSED" if ng_results.get("TFixedDf", {}).get("available", False)
                and all(ng_results[k]["has_cdf"] and ng_results[k]["has_sf"]
                        for k in ("Normal", "Laplace"))
                else "FAILED_REGISTERED_DISTRIBUTION_SUPPORT",
            "distributions": ng_results
        }
    except ImportError as e:
        report["components"]["NGBoost"] = {"status": "BLOCKED_IMPORT_ERROR", "error": str(e)}
    except Exception as e:
        report["components"]["NGBoost"] = {"status": "FAILED", "error": str(e)}

    # 5. Checkpoint Verification
    if checkpoint_path is not None and checkpoint_path.exists() and expected_checkpoint_sha256:
        observed_sha = compute_file_sha256(checkpoint_path)
        report["components"]["Checkpoint"] = {
            "path": str(checkpoint_path),
            "sha256": observed_sha,
            "status": "PASSED" if observed_sha.lower() == expected_checkpoint_sha256.lower()
                else "FAILED_CHECKPOINT_HASH"
        }
    else:
        report["components"]["Checkpoint"] = {
            "status": "BLOCKED_APPROVED_CHECKPOINT_OR_HASH_MISSING"
        }

    # 6. Hardware & Memory Runtime Probe
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
        vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if cuda_avail else 0.0
        # Per §2 D13 and preflight contract: RTX 3060 12GB feasibility remains unverified until inference execution passes
        report["components"]["HardwareRuntime"] = {
            "status": "BLOCKED_INFERENCE_MEMORY_UNVERIFIED",
            "cuda_available": cuda_avail,
            "device_name": device_name,
            "total_vram_gb": vram_gb,
            "feasibility_note": "GPU detected; memory feasibility of TabICLv2 configuration remains unverified until executable inference passes"
        }
    except ImportError as e:
        report["components"]["HardwareRuntime"] = {
            "status": "BLOCKED_IMPORT_ERROR",
            "error": str(e)
        }

    # Evaluate Overall Status
    statuses = [v.get("status") for v in report["components"].values()]
    if all(s == "PASSED" for s in statuses):
        report["status"] = "ALL_PROBES_PASSED"
    elif any(isinstance(s, str) and "BLOCKED" in s for s in statuses):
        report["status"] = "BLOCKED_DEPENDENCIES_OR_CHECKPOINT_UNVERIFIED"
    else:
        report["status"] = "PROBES_FAILED"

    return report


if __name__ == "__main__":
    out_dir = Path("outputs/G0")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "api_preflight.json"

    result = probe_api_compatibility()
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
