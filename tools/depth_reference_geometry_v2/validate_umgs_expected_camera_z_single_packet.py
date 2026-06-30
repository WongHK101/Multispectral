"""Validate one UMGS expected-camera-z packet export.

This validator is for A1-Q single-export qualification. It reuses the A0
numeric tolerance semantics for one packet only, and does not check
repeatability, RGB compatibility, OpenMVS/DA3 alignment, or method ranking.
"""

from __future__ import annotations

import argparse
import csv
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
SCHEMA_VERSION = "umgs_expected_camera_z_packet_v1"
DEFAULT_TOLERANCES = {
    "identity_rtol": 1e-5,
    "identity_atol": 1e-6,
    "opacity_min": -1e-6,
    "opacity_max": 1.0 + 1e-5,
    "variance_min": -1e-6,
}


def sha256_file(path: str | Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_one(path: Path, pattern: str) -> Path:
    matches = sorted(path.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one {pattern} under {path}, found {len(matches)}")
    return matches[0]


def nested_get(data: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: str = "") -> None:
    checks.append({"check": name, "passed": bool(passed), "detail": detail})


def validate_packet_dir(args: argparse.Namespace) -> dict[str, Any]:
    packet_dir = Path(args.packet_dir)
    npz_path = find_one(packet_dir, "*_expected_camera_z_packet.npz")
    manifest_path = packet_dir / "expected_camera_z_packet_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    npz = np.load(npz_path)
    arrays = {k: npz[k] for k in npz.files}
    checks: list[dict[str, Any]] = []
    tol = dict(DEFAULT_TOLERANCES)

    missing = sorted(EXPECTED_ARRAYS - set(arrays))
    add_check(checks, "required_npz_arrays_present", not missing, f"missing={missing}")
    if missing:
        summary = {
            "scope": "single-packet exporter qualification only",
            "passed": False,
            "tolerances": tol,
            "npz_path": str(npz_path),
            "manifest_path": str(manifest_path),
            "checks": checks,
        }
        return summary

    expected_shape = (int(args.expected_height), int(args.expected_width))
    shapes = {k: tuple(arrays[k].shape) for k in EXPECTED_ARRAYS}
    add_check(checks, "all_arrays_same_shape", len(set(shapes.values())) == 1, str(shapes))
    add_check(checks, "expected_raster_shape", shapes["expected_camera_z"] == expected_shape, str(shapes["expected_camera_z"]))

    valid_raw = arrays["numeric_valid"]
    valid_dtype_ok = valid_raw.dtype == np.bool_ or np.issubdtype(valid_raw.dtype, np.integer)
    unique_values = np.unique(valid_raw)
    valid_values_ok = set(unique_values.tolist()).issubset({0, 1, False, True})
    valid = valid_raw.astype(bool)
    add_check(checks, "numeric_valid_dtype", valid_dtype_ok, str(valid_raw.dtype))
    add_check(checks, "numeric_valid_binary", valid_values_ok, str(unique_values.tolist()))
    add_check(checks, "numeric_valid_nonempty", bool(valid.sum() > 0), str(int(valid.sum())))

    A = arrays["accumulated_opacity"]
    M1 = arrays["weighted_camera_z_sum"]
    expected_z = arrays["expected_camera_z"]
    variance = arrays["camera_z_variance"]

    add_check(checks, "opacity_finite_all_pixels", bool(np.isfinite(A).all()), "")
    add_check(
        checks,
        "opacity_bounds_all_pixels",
        bool(np.isfinite(A).all() and (A >= tol["opacity_min"]).all() and (A <= tol["opacity_max"]).all()),
        f"min={float(np.nanmin(A))} max={float(np.nanmax(A))}",
    )

    if valid.any():
        identity = np.isclose(
            expected_z[valid],
            M1[valid] / A[valid],
            rtol=tol["identity_rtol"],
            atol=tol["identity_atol"],
            equal_nan=False,
        )
        max_identity_abs = float(np.max(np.abs(expected_z[valid] - (M1[valid] / A[valid]))))
        add_check(checks, "expected_z_identity_valid_pixels", bool(identity.all()), f"max_abs={max_identity_abs}")
        add_check(checks, "valid_expected_z_finite", bool(np.isfinite(expected_z[valid]).all()), "")
        add_check(checks, "valid_expected_z_positive", bool((expected_z[valid] > 0).all()), "")
        add_check(checks, "valid_variance_finite", bool(np.isfinite(variance[valid]).all()), "")
        add_check(
            checks,
            "valid_variance_above_tolerance",
            bool((variance[valid] >= tol["variance_min"]).all()),
            f"min={float(np.nanmin(variance[valid]))}",
        )
    else:
        add_check(checks, "expected_z_identity_valid_pixels", False, "no valid pixels")
        add_check(checks, "valid_expected_z_finite", False, "no valid pixels")
        add_check(checks, "valid_expected_z_positive", False, "no valid pixels")
        add_check(checks, "valid_variance_finite", False, "no valid pixels")
        add_check(checks, "valid_variance_above_tolerance", False, "no valid pixels")

    invalid = ~valid
    add_check(checks, "invalid_expected_z_nan", bool(np.isnan(expected_z[invalid]).all()), f"invalid_count={int(invalid.sum())}")
    add_check(checks, "invalid_variance_nan", bool(np.isnan(variance[invalid]).all()), f"invalid_count={int(invalid.sum())}")

    actual_npz_sha = sha256_file(npz_path)
    add_check(checks, "manifest_npz_sha256_matches_file", manifest.get("npz_sha256") == actual_npz_sha, actual_npz_sha)
    add_check(checks, "manifest_schema_version", nested_get(manifest, ["schema", "schema_version"]) == SCHEMA_VERSION, str(nested_get(manifest, ["schema", "schema_version"])))
    add_check(checks, "manifest_scene", manifest.get("scene") == args.expected_scene, str(manifest.get("scene")))
    add_check(checks, "manifest_target", manifest.get("target") == args.expected_target, str(manifest.get("target")))
    add_check(checks, "manifest_raster_height", int(manifest.get("raster_height", -1)) == int(args.expected_height), str(manifest.get("raster_height")))
    add_check(checks, "manifest_raster_width", int(manifest.get("raster_width", -1)) == int(args.expected_width), str(manifest.get("raster_width")))
    add_check(checks, "checkpoint_sha256", manifest.get("checkpoint_sha256") == args.checkpoint_sha256, str(manifest.get("checkpoint_sha256")))
    add_check(
        checks,
        "camera_manifest_sha256",
        nested_get(manifest, ["preflight", "manifest_info", "camera_manifest", "sha256"]) == args.camera_manifest_sha256,
        str(nested_get(manifest, ["preflight", "manifest_info", "camera_manifest", "sha256"])),
    )
    add_check(
        checks,
        "split_manifest_sha256",
        nested_get(manifest, ["preflight", "manifest_info", "split_manifest", "sha256"]) == args.split_manifest_sha256,
        str(nested_get(manifest, ["preflight", "manifest_info", "split_manifest", "sha256"])),
    )
    add_check(
        checks,
        "train_file_sha256",
        nested_get(manifest, ["preflight", "manifest_info", "split_manifest", "train_file_actual_sha256"]) == args.train_file_sha256
        and nested_get(manifest, ["preflight", "manifest_info", "split_manifest", "train_file_expected_sha256"]) == args.train_file_sha256,
        str(nested_get(manifest, ["preflight", "manifest_info", "split_manifest", "train_file_actual_sha256"])),
    )
    add_check(
        checks,
        "test_file_sha256",
        nested_get(manifest, ["preflight", "manifest_info", "split_manifest", "test_file_actual_sha256"]) == args.test_file_sha256
        and nested_get(manifest, ["preflight", "manifest_info", "split_manifest", "test_file_expected_sha256"]) == args.test_file_sha256,
        str(nested_get(manifest, ["preflight", "manifest_info", "split_manifest", "test_file_actual_sha256"])),
    )
    add_check(
        checks,
        "colmap_sparse_hash_manifest_sha256",
        nested_get(manifest, ["preflight", "colmap_sparse_hash_manifest", "sha256"]) == args.colmap_sparse_hash_manifest_sha256,
        str(nested_get(manifest, ["preflight", "colmap_sparse_hash_manifest", "sha256"])),
    )
    add_check(
        checks,
        "runtime_source_hash_manifest_sha256",
        nested_get(manifest, ["preflight", "runtime_source_hash_manifest", "sha256"]) == args.runtime_source_hash_manifest_sha256,
        str(nested_get(manifest, ["preflight", "runtime_source_hash_manifest", "sha256"])),
    )
    add_check(
        checks,
        "rasterizer_extension_sha256",
        nested_get(manifest, ["rasterizer_extension", "imported_extension_sha256"]) == args.rasterizer_extension_sha256,
        str(nested_get(manifest, ["rasterizer_extension", "imported_extension_sha256"])),
    )
    add_check(
        checks,
        "expected_camera_fingerprint_file_sha256",
        nested_get(manifest, ["target_camera_fingerprint_comparison", "expected_file_sha256"]) == args.expected_camera_fingerprint_sha256,
        str(nested_get(manifest, ["target_camera_fingerprint_comparison", "expected_file_sha256"])),
    )
    add_check(
        checks,
        "expected_camera_fingerprint_payload_sha256",
        nested_get(manifest, ["target_camera_fingerprint_comparison", "expected_payload_sha256"]) == args.expected_camera_fingerprint_payload_sha256
        and nested_get(manifest, ["target_camera_fingerprint_comparison", "actual_payload_sha256"]) == args.expected_camera_fingerprint_payload_sha256
        and nested_get(manifest, ["target_camera_fingerprint", "fingerprint_sha256"]) == args.expected_camera_fingerprint_payload_sha256,
        str(nested_get(manifest, ["target_camera_fingerprint", "fingerprint_sha256"])),
    )
    add_check(
        checks,
        "expected_camera_fingerprint_matched",
        nested_get(manifest, ["target_camera_fingerprint_comparison", "matched"]) is True,
        str(nested_get(manifest, ["target_camera_fingerprint_comparison", "matched"])),
    )

    metrics = {
        "raster_height": int(expected_shape[0]),
        "raster_width": int(expected_shape[1]),
        "total_pixels": int(valid.size),
        "numeric_valid_count": int(valid.sum()),
        "numeric_valid_ratio": float(valid.mean()) if valid.size else 0.0,
        "invalid_count": int(invalid.sum()),
        "accumulated_opacity_min": float(np.nanmin(A)),
        "accumulated_opacity_max": float(np.nanmax(A)),
        "expected_camera_z_min_valid": float(np.nanmin(expected_z[valid])) if valid.any() else None,
        "expected_camera_z_max_valid": float(np.nanmax(expected_z[valid])) if valid.any() else None,
        "camera_z_variance_min_valid": float(np.nanmin(variance[valid])) if valid.any() else None,
        "camera_z_variance_max_valid": float(np.nanmax(variance[valid])) if valid.any() else None,
    }
    passed = all(bool(row["passed"]) for row in checks)
    return {
        "scope": "single-packet exporter qualification only; no repeatability, RGB compatibility, OpenMVS/DA3 comparison, or method ranking",
        "passed": bool(passed),
        "tolerances": tol,
        "npz_path": str(npz_path),
        "npz_sha256": actual_npz_sha,
        "manifest_path": str(manifest_path),
        "metrics": metrics,
        "checks": checks,
    }


def write_csv_summary(summary: dict[str, Any], path: Path) -> None:
    rows = []
    for check in summary["checks"]:
        rows.append(
            {
                "npz_path": summary["npz_path"],
                "passed": str(bool(summary["passed"])).lower(),
                "check": check["check"],
                "check_passed": str(bool(check["passed"])).lower(),
                "detail": check.get("detail", ""),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["npz_path", "passed", "check", "check_passed", "detail"])
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate one UMGS expected-camera-z packet export.")
    parser.add_argument("--packet-dir", required=True)
    parser.add_argument("--expected-height", type=int, required=True)
    parser.add_argument("--expected-width", type=int, required=True)
    parser.add_argument("--expected-scene", required=True)
    parser.add_argument("--expected-target", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--camera-manifest-sha256", required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--train-file-sha256", required=True)
    parser.add_argument("--test-file-sha256", required=True)
    parser.add_argument("--colmap-sparse-hash-manifest-sha256", required=True)
    parser.add_argument("--runtime-source-hash-manifest-sha256", required=True)
    parser.add_argument("--rasterizer-extension-sha256", required=True)
    parser.add_argument("--expected-camera-fingerprint-sha256", required=True)
    parser.add_argument("--expected-camera-fingerprint-payload-sha256", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = validate_packet_dir(args)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv_summary(summary, Path(args.output_csv))
    print(json.dumps({"passed": summary["passed"], "output_json": str(output_json), "output_csv": args.output_csv}, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
