"""Validate a future UMGS expected-camera-z A0 exporter qualification run.

This validator reads two approved repeat exports and checks the pre-registered
E0.1 tolerances. It does not run rendering and does not compare against
OpenMVS/DA3 or any method ranking proxy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_ARRAYS = {
    "accumulated_opacity",
    "weighted_camera_z_sum",
    "expected_camera_z",
    "numeric_valid",
    "weighted_camera_z2_sum",
    "camera_z_variance",
}
DEFAULT_TOLERANCES = {
    "identity_rtol": 1e-5,
    "identity_atol": 1e-6,
    "repeatability_rtol": 1e-6,
    "repeatability_atol": 1e-6,
    "opacity_min": -1e-6,
    "opacity_max": 1.0 + 1e-5,
    "variance_min": -1e-6,
    "rgb_compatibility_max_abs_delta": 1e-6,
}


def find_one(path: Path, pattern: str) -> Path:
    matches = sorted(path.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one {pattern} under {path}, found {len(matches)}")
    return matches[0]


def load_repeat(path: Path) -> dict[str, Any]:
    npz_path = find_one(path, "*_expected_camera_z_packet.npz")
    manifest_path = path / "expected_camera_z_packet_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    arrays_npz = np.load(npz_path)
    arrays = {k: arrays_npz[k] for k in arrays_npz.files}
    missing = sorted(EXPECTED_ARRAYS - set(arrays))
    if missing:
        raise KeyError(f"{npz_path} missing arrays: {missing}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {"npz_path": npz_path, "manifest_path": manifest_path, "arrays": arrays, "manifest": manifest}


def validate_single(rep: dict[str, Any], expected_height: int, expected_width: int, tol: dict[str, float]) -> dict[str, Any]:
    arrays = rep["arrays"]
    shape = arrays["expected_camera_z"].shape
    result: dict[str, Any] = {
        "npz_path": str(rep["npz_path"]),
        "shape": list(shape),
        "shape_passed": bool(shape == (expected_height, expected_width)),
    }
    valid = arrays["numeric_valid"].astype(bool)
    result["numeric_valid_count"] = int(valid.sum())
    result["numeric_valid_nonempty"] = bool(valid.sum() > 0)
    A = arrays["accumulated_opacity"]
    M1 = arrays["weighted_camera_z_sum"]
    expected_z = arrays["expected_camera_z"]
    variance = arrays["camera_z_variance"]
    identity = np.zeros_like(expected_z, dtype=bool)
    if valid.any():
        identity[valid] = np.isclose(
            expected_z[valid],
            M1[valid] / A[valid],
            rtol=tol["identity_rtol"],
            atol=tol["identity_atol"],
            equal_nan=False,
        )
    result["identity_passed"] = bool(valid.any() and identity[valid].all())
    result["opacity_passed"] = bool(
        np.isfinite(A[valid]).all()
        and (A[valid] >= tol["opacity_min"]).all()
        and (A[valid] <= tol["opacity_max"]).all()
    ) if valid.any() else False
    result["variance_passed"] = bool(
        np.isfinite(variance[valid]).all()
        and (variance[valid] >= tol["variance_min"]).all()
    ) if valid.any() else False
    result["single_passed"] = bool(
        result["shape_passed"]
        and result["numeric_valid_nonempty"]
        and result["identity_passed"]
        and result["opacity_passed"]
        and result["variance_passed"]
    )
    return result


def validate_repeatability(r1: dict[str, Any], r2: dict[str, Any], tol: dict[str, float]) -> dict[str, Any]:
    a = r1["arrays"]
    b = r2["arrays"]
    out: dict[str, Any] = {}
    mask_equal = np.array_equal(a["numeric_valid"], b["numeric_valid"])
    out["numeric_valid_masks_equal"] = bool(mask_equal)
    for key in [
        "accumulated_opacity",
        "weighted_camera_z_sum",
        "expected_camera_z",
        "weighted_camera_z2_sum",
        "camera_z_variance",
    ]:
        out[f"{key}_allclose"] = bool(
            np.allclose(
                a[key],
                b[key],
                rtol=tol["repeatability_rtol"],
                atol=tol["repeatability_atol"],
                equal_nan=True,
            )
        )
    out["repeatability_passed"] = bool(mask_equal and all(v for k, v in out.items() if k.endswith("_allclose")))
    return out


def validate_rgb_compatibility(repeat_path: Path, tol: dict[str, float]) -> dict[str, Any]:
    enabled = sorted(repeat_path.glob("*_rgb_enabled.npy"))
    disabled = sorted(repeat_path.glob("*_rgb_disabled.npy"))
    if len(enabled) != 1 or len(disabled) != 1:
        return {"available": False, "passed": False, "reason": "missing_rgb_compatibility_arrays"}
    e = np.load(enabled[0])
    d = np.load(disabled[0])
    max_abs = float(np.max(np.abs(e - d)))
    return {
        "available": True,
        "rgb_enabled_path": str(enabled[0]),
        "rgb_disabled_path": str(disabled[0]),
        "max_abs_rgb_delta": max_abs,
        "threshold": tol["rgb_compatibility_max_abs_delta"],
        "passed": bool(max_abs <= tol["rgb_compatibility_max_abs_delta"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate two future A0 expected-camera-z repeat exports.")
    parser.add_argument("--repeat-1", required=True)
    parser.add_argument("--repeat-2", required=True)
    parser.add_argument("--expected-height", type=int, required=True)
    parser.add_argument("--expected-width", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tol = dict(DEFAULT_TOLERANCES)
    r1_path = Path(args.repeat_1)
    r2_path = Path(args.repeat_2)
    r1 = load_repeat(r1_path)
    r2 = load_repeat(r2_path)
    summary = {
        "scope": "A0 exporter/interface qualification only; no OpenMVS/DA3 comparison and no method ranking",
        "tolerances": tol,
        "repeat_1": validate_single(r1, args.expected_height, args.expected_width, tol),
        "repeat_2": validate_single(r2, args.expected_height, args.expected_width, tol),
        "repeatability": validate_repeatability(r1, r2, tol),
        "rgb_compatibility_repeat_1": validate_rgb_compatibility(r1_path, tol),
        "rgb_compatibility_repeat_2": validate_rgb_compatibility(r2_path, tol),
    }
    summary["passed"] = bool(
        summary["repeat_1"]["single_passed"]
        and summary["repeat_2"]["single_passed"]
        and summary["repeatability"]["repeatability_passed"]
        and summary["rgb_compatibility_repeat_1"]["passed"]
        and summary["rgb_compatibility_repeat_2"]["passed"]
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"passed": summary["passed"], "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
