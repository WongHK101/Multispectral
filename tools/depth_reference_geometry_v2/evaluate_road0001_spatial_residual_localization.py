#!/usr/bin/env python3
"""Road-0001 spatial residual localization evaluator.

This V1.1 evaluator is execution-capable, but only for the separately approved
post-hoc exploratory Road-0001 localization stage. It has two strict modes:

* --preflight-only validates bindings, schemas, formal result fields, runtime
  source manifests, and output guards without constructing residual maps,
  high-gradient masks, grid/border statistics, or PNGs.
* --execute-localization performs the approved localization only after all
  formal-result consistency and high-gradient identity checks pass. It writes
  into a fresh staging directory, validates all outputs, and atomically renames
  staging into the final output root.

The outputs are exploratory proxy-semantics diagnostics only. They do not alter
the archived Layer-2 taxonomy and are not geometry accuracy or method ranking.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import colormaps
import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import openmvs_da3_overlap_corrected as corrected
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"failed_to_import_corrected_overlap_protocol: {exc}") from exc


SCHEMA = "road0001_spatial_residual_localization_v1_1"
LOCALIZATION_RUNTIME_MANIFEST_SCHEMA = "road0001_spatial_residual_localization_evaluator_v1_1_runtime_source_manifest"
LAYER2_RUNTIME_MANIFEST_SCHEMA = "road0001_layer2_runtime_source_manifest_v1_2"
PHASE = "post_hoc_exploratory_diagnostic"
EXPECTED_SCENE = "road_01_20260602_1648_40m"
EXPECTED_TARGET = "DJI_20260602165038_0001_D.JPG"
EXPECTED_HEIGHT = 869
EXPECTED_WIDTH = 1200
TOTAL_PIXELS = EXPECTED_HEIGHT * EXPECTED_WIDTH
SHUFFLE_SEED = int(corrected.SHUFFLE_SEED)
SCALAR_ATOL = 1e-12
SCALAR_RTOL = 1e-10
EXPECTED_PRIMARY_COMMON_COUNT = 1_042_800
EXPECTED_PRIMARY_COMMON_RATIO = 1.0
EXPECTED_GRADIENT_LOCAL_VALID_COUNT = 1_038_666
EXPECTED_HIGH_GRADIENT_THRESHOLD = 0.00962846592602884
EXPECTED_HIGH_GRADIENT_PIXELS = 259_667
EXPECTED_HIGH_GRADIENT_PACKBITS_SHA256 = "0be0892c92879b01be06a92ee2b7126eeb9cbb4d85c9e30f6ee04326f10c12c5"
EXPECTED_STATUS = {
    "input_status": "correspondence_pass",
    "support_status": "common_support_sufficient",
    "control_shared_support_status": "control_shared_support_sufficient",
    "metric_status": "metrics_valid",
    "interpretation_status": "proxy_alignment_weak_or_mixed",
}
EXPECTED_SCALARS = {
    "mean_absolute_camera_z_disagreement": 0.6275477965489904,
    "median_absolute_camera_z_disagreement": 0.37821006774902344,
    "rmse_camera_z_disagreement": 1.0914610557405007,
    "spearman": 0.8990264403494209,
    "high_gradient_cosine_median": 0.2814362007619959,
}

UMGS_DTYPES = {
    "accumulated_opacity": np.dtype("float32"),
    "weighted_camera_z_sum": np.dtype("float32"),
    "expected_camera_z": np.dtype("float32"),
    "numeric_valid": np.dtype("uint8"),
    "weighted_camera_z2_sum": np.dtype("float32"),
    "camera_z_variance": np.dtype("float32"),
}
OPENMVS_DTYPES = {
    "depth": np.dtype("float32"),
    "valid": np.dtype("uint8"),
    "triangle_id": np.dtype("int32"),
    "barycentric": np.dtype("float32"),
}
OUTPUT_NPZ_DTYPES = {
    "primary_common_mask": np.dtype("uint8"),
    "frozen_high_gradient_mask": np.dtype("uint8"),
    "signed_camera_z_disagreement": np.dtype("float32"),
    "absolute_camera_z_disagreement": np.dtype("float32"),
    "openmvs_denominated_relative_disagreement": np.dtype("float32"),
    "openmvs_gradient_magnitude": np.dtype("float32"),
    "umgs_gradient_magnitude": np.dtype("float32"),
    "per_pixel_gradient_cosine": np.dtype("float64"),
    "umgs_accumulated_opacity": np.dtype("float32"),
    "umgs_camera_z_variance": np.dtype("float32"),
    "grid_id_map": np.dtype("int16"),
    "border_band_id_map": np.dtype("int8"),
}

ROW_EDGES = np.floor(np.linspace(0, EXPECTED_HEIGHT, 9)).astype(np.int64)
COL_EDGES = np.floor(np.linspace(0, EXPECTED_WIDTH, 9)).astype(np.int64)
BORDER_BANDS = [
    {"id": 0, "name": "border_0_15_px", "min_distance": 0, "max_distance": 15},
    {"id": 1, "name": "border_16_31_px", "min_distance": 16, "max_distance": 31},
    {"id": 2, "name": "border_32_63_px", "min_distance": 32, "max_distance": 63},
    {"id": 3, "name": "border_ge_64_px", "min_distance": 64, "max_distance": None},
]
PNG_SPECS = [
    ("signed_camera_z_disagreement", "signed_camera_z_disagreement.png"),
    ("absolute_camera_z_disagreement", "absolute_camera_z_disagreement.png"),
    ("openmvs_denominated_relative_disagreement", "openmvs_denominated_relative_disagreement.png"),
    ("openmvs_gradient_magnitude", "openmvs_gradient_magnitude.png"),
    ("umgs_gradient_magnitude", "umgs_gradient_magnitude.png"),
    ("per_pixel_gradient_cosine", "per_pixel_gradient_cosine.png"),
    ("umgs_camera_z_variance", "umgs_camera_z_variance.png"),
    ("umgs_accumulated_opacity", "umgs_accumulated_opacity.png"),
]


class ProtocolError(RuntimeError):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json_payload(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def mask_packbits_sha256(mask: np.ndarray) -> str:
    return hashlib.sha256(np.packbits(np.asarray(mask, dtype=bool).reshape(-1)).tobytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check(condition: bool, code: str, message: str, details: dict[str, Any] | None = None) -> None:
    if not condition:
        raise ProtocolError(code, message, details)


def verify_file(path: Path, expected_sha: str, label: str, checks: list[dict[str, Any]]) -> str:
    check(path.exists(), "input_missing", f"{label} missing: {path}")
    actual = sha256_file(path)
    passed = actual == expected_sha
    checks.append({"check": f"{label}_sha256", "status": "pass" if passed else "fail", "path": str(path), "actual": actual, "expected": expected_sha})
    check(passed, "input_hash_mismatch", f"{label} SHA mismatch", {"actual": actual, "expected": expected_sha})
    return actual


def npz_schema(path: Path, expected: dict[str, np.dtype], expected_shape: tuple[int, int], label: str) -> dict[str, Any]:
    info: dict[str, Any] = {"path": str(path), "arrays": {}}
    with np.load(path, allow_pickle=False) as npz:
        keys = set(npz.files)
        missing = sorted(set(expected) - keys)
        check(not missing, "npz_schema_mismatch", f"{label} missing keys: {missing}")
        for key, dtype in expected.items():
            arr = npz[key]
            expected_arr_shape = expected_shape if key != "barycentric" else (*expected_shape, 3)
            check(arr.shape == expected_arr_shape, "npz_shape_mismatch", f"{label}.{key} shape mismatch", {"actual": list(arr.shape), "expected": list(expected_arr_shape)})
            check(arr.dtype == dtype, "npz_dtype_mismatch", f"{label}.{key} dtype mismatch", {"actual": str(arr.dtype), "expected": str(dtype)})
            info["arrays"][key] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    return info


def load_umgs(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as npz:
        return {key: np.asarray(npz[key]) for key in UMGS_DTYPES}


def load_openmvs(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as npz:
        return {key: np.asarray(npz[key]) for key in OPENMVS_DTYPES}


def require_scalar(name: str, actual: float, expected: float, checks: list[dict[str, Any]]) -> None:
    diff = abs(actual - expected)
    tol = SCALAR_ATOL + SCALAR_RTOL * abs(expected)
    passed = diff <= tol
    checks.append({"check": f"formal_scalar_{name}", "status": "pass" if passed else "fail", "actual": actual, "expected": expected, "abs_diff": diff, "tolerance": tol})
    check(passed, "formal_result_scalar_mismatch", f"{name}: expected {expected}, got {actual}")


def nested_get(payload: dict[str, Any], dotted: str) -> Any:
    cur: Any = payload
    for part in dotted.split("."):
        check(isinstance(cur, dict) and part in cur, "formal_result_missing_field", f"missing {dotted}")
        cur = cur[part]
    return cur


def formal_result_gate(result: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    for key, expected in EXPECTED_STATUS.items():
        actual = result.get(key)
        passed = actual == expected
        checks.append({"check": f"formal_status_{key}", "status": "pass" if passed else "fail", "actual": actual, "expected": expected})
        check(passed, "formal_result_status_mismatch", f"{key}: expected {expected}, got {actual}")
    exact = {
        "shuffle_seed": SHUFFLE_SEED,
        "support_counts.primary_common_count": EXPECTED_PRIMARY_COMMON_COUNT,
        "support_counts.primary_common_ratio": EXPECTED_PRIMARY_COMMON_RATIO,
        "true_core_metrics_on_shared.high_gradient_pixels": EXPECTED_HIGH_GRADIENT_PIXELS,
        "true_core_metrics_on_shared.gradient_local_valid_pixels": EXPECTED_GRADIENT_LOCAL_VALID_COUNT,
    }
    for dotted, expected in exact.items():
        actual = nested_get(result, dotted)
        passed = actual == expected
        checks.append({"check": f"formal_exact_{dotted}", "status": "pass" if passed else "fail", "actual": actual, "expected": expected})
        check(passed, "formal_result_exact_mismatch", f"{dotted}: expected {expected}, got {actual}")
    require_scalar("mae", float(nested_get(result, "true_descriptive_metrics.mean_absolute_camera_z_disagreement")), EXPECTED_SCALARS["mean_absolute_camera_z_disagreement"], checks)
    require_scalar("median_abs", float(nested_get(result, "true_descriptive_metrics.median_absolute_camera_z_disagreement")), EXPECTED_SCALARS["median_absolute_camera_z_disagreement"], checks)
    require_scalar("rmse", float(nested_get(result, "true_descriptive_metrics.rmse_camera_z_disagreement")), EXPECTED_SCALARS["rmse_camera_z_disagreement"], checks)
    require_scalar("spearman", float(nested_get(result, "true_core_metrics_on_shared.spearman")), EXPECTED_SCALARS["spearman"], checks)
    require_scalar("high_gradient_cosine_median", float(nested_get(result, "true_core_metrics_on_shared.high_gradient_cosine_median")), EXPECTED_SCALARS["high_gradient_cosine_median"], checks)
    require_scalar("high_gradient_threshold", float(nested_get(result, "true_core_metrics_on_shared.high_gradient_threshold")), EXPECTED_HIGH_GRADIENT_THRESHOLD, checks)


def verify_runtime_manifest(
    path: Path,
    expected_sha: str,
    checks: list[dict[str, Any]],
    *,
    required_roles: set[str],
    expected_schema: str,
    required_runtime_roles: set[str] | None = None,
) -> dict[str, Any]:
    verify_file(path, expected_sha, path.stem, checks)
    manifest = read_json(path)
    schema = manifest.get("schema")
    passed = schema == expected_schema
    checks.append({"check": f"{path.stem}_schema", "status": "pass" if passed else "fail", "actual": schema, "expected": expected_schema})
    check(passed, "runtime_manifest_schema_mismatch", f"{path} schema mismatch")
    repo_root_value = manifest.get("repo_root")
    check(isinstance(repo_root_value, str) and bool(repo_root_value), "runtime_manifest_missing_repo_root", f"{path} missing repo_root")
    repo_root = Path(repo_root_value)
    sources = manifest.get("sources")
    check(isinstance(sources, list) and sources, "runtime_manifest_missing_sources", f"{path} missing sources")
    roles: set[str] = set()
    for idx, entry in enumerate(sources):
        check(isinstance(entry, dict), "runtime_manifest_source_entry_invalid", f"{path} source entry {idx} invalid")
        role = entry.get("role")
        rel = entry.get("repo_relative_path")
        sha = entry.get("sha256")
        check(isinstance(role, str) and role, "runtime_manifest_source_entry_invalid", f"{path} source entry {idx} missing role")
        check(isinstance(rel, str) and rel, "runtime_manifest_source_entry_invalid", f"{path} source entry {idx} missing repo_relative_path")
        check(isinstance(sha, str) and len(sha) == 64, "runtime_manifest_source_entry_invalid", f"{path} source entry {idx} missing sha256")
        roles.add(role)
        source_path = repo_root / rel
        if required_runtime_roles is None or role in required_runtime_roles:
            verify_file(source_path, sha, f"{path.stem}_{role}", checks)
    for role in sorted(required_roles):
        passed = role in roles
        checks.append({"check": f"{path.stem}_has_{role}", "status": "pass" if passed else "fail"})
        check(passed, "runtime_manifest_missing_source", f"missing runtime source role {role}")
    return manifest


def verify_simple_json_sha(path: Path, expected_sha: str, label: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
    verify_file(path, expected_sha, label, checks)
    return read_json(path)


def build_primary_mask(umgs: dict[str, np.ndarray], openmvs: dict[str, np.ndarray]) -> np.ndarray:
    z_u = umgs["expected_camera_z"]
    z_o = openmvs["depth"]
    return (
        (umgs["numeric_valid"].astype(bool))
        & (openmvs["valid"].astype(bool))
        & np.isfinite(z_u)
        & np.isfinite(z_o)
        & (z_u > 0)
        & (z_o > 0)
    )


def descriptive_metrics(umgs_z: np.ndarray, openmvs_z: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    diff = (umgs_z - openmvs_z)[mask].astype(np.float64)
    absdiff = np.abs(diff)
    return {
        "mean_absolute_camera_z_disagreement": float(np.mean(absdiff)),
        "median_absolute_camera_z_disagreement": float(np.median(absdiff)),
        "rmse_camera_z_disagreement": float(np.sqrt(np.mean(diff * diff))),
    }


def gradient_components(depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gy, gx = np.gradient(depth.astype(np.float64))
    return gx, gy


def gradient_magnitude(depth: np.ndarray) -> np.ndarray:
    gx, gy = gradient_components(depth)
    return np.sqrt(gx * gx + gy * gy)


def per_pixel_gradient_cosine(openmvs_z: np.ndarray, umgs_z: np.ndarray, local_valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gy_o, gx_o = np.gradient(openmvs_z.astype(np.float64))
    gy_u, gx_u = np.gradient(umgs_z.astype(np.float64))
    dot = gx_o * gx_u + gy_o * gy_u
    norm = np.sqrt(gx_o * gx_o + gy_o * gy_o) * np.sqrt(gx_u * gx_u + gy_u * gy_u)
    cosine = dot / np.maximum(norm, 1e-12)
    valid = local_valid & np.isfinite(gx_o) & np.isfinite(gy_o) & np.isfinite(gx_u) & np.isfinite(gy_u) & np.isfinite(cosine)
    out = np.full(openmvs_z.shape, np.nan, dtype=np.float64)
    out[valid] = cosine[valid]
    return out, valid


def grid_id_map(height: int = EXPECTED_HEIGHT, width: int = EXPECTED_WIDTH) -> np.ndarray:
    grid = np.full((height, width), -1, dtype=np.int16)
    row_edges = np.floor(np.linspace(0, height, 9)).astype(np.int64)
    col_edges = np.floor(np.linspace(0, width, 9)).astype(np.int64)
    cid = 0
    for i in range(8):
        for j in range(8):
            grid[row_edges[i] : row_edges[i + 1], col_edges[j] : col_edges[j + 1]] = cid
            cid += 1
    return grid


def border_band_id_map(height: int = EXPECTED_HEIGHT, width: int = EXPECTED_WIDTH) -> np.ndarray:
    y, x = np.indices((height, width), dtype=np.int64)
    d = np.minimum.reduce([y, x, height - 1 - y, width - 1 - x])
    band = np.full((height, width), -1, dtype=np.int8)
    band[(0 <= d) & (d <= 15)] = 0
    band[(16 <= d) & (d <= 31)] = 1
    band[(32 <= d) & (d <= 63)] = 2
    band[d >= 64] = 3
    return band


def finite_percentile(values: np.ndarray, q: float) -> float | None:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return None
    return float(np.percentile(vals, q))


def nullable_median(values: np.ndarray) -> float | None:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return None
    return float(np.median(vals))


def stats_by_label(
    labels: np.ndarray,
    label_names: dict[int, str],
    primary: np.ndarray,
    high: np.ndarray,
    cosine_valid: np.ndarray,
    absdiff: np.ndarray,
    rel: np.ndarray,
    cosine: np.ndarray,
    opacity: np.ndarray,
    variance: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lid in sorted(label_names):
        region = labels == lid
        primary_region = region & primary
        high_region = primary_region & high
        cosine_region = high_region & cosine_valid
        rows.append(
            {
                "id": lid,
                "name": label_names[lid],
                "total_pixel_count": int(region.sum()),
                "primary_common_count": int(primary_region.sum()),
                "high_gradient_count": int(high_region.sum()),
                "cosine_valid_high_gradient_count": int(cosine_region.sum()),
                "median_absolute_disagreement": nullable_median(absdiff[primary_region]),
                "p90_absolute_disagreement": finite_percentile(absdiff[primary_region], 90),
                "median_relative_disagreement": nullable_median(rel[primary_region]),
                "median_high_gradient_cosine": nullable_median(cosine[cosine_region]),
                "opacity_median": nullable_median(opacity[primary_region]),
                "variance_median": nullable_median(variance[primary_region]),
            }
        )
    return rows


def display_ranges(arrays: dict[str, np.ndarray], masks: dict[str, np.ndarray]) -> dict[str, Any]:
    def p99(name: str, domain: str) -> float | None:
        vals = arrays[name][masks[domain] & np.isfinite(arrays[name])]
        if vals.size == 0:
            return None
        return float(np.percentile(vals, 99))

    signed_vals = arrays["signed_camera_z_disagreement"][masks["primary_common"] & np.isfinite(arrays["signed_camera_z_disagreement"])]
    signed_p99 = float(np.percentile(np.abs(signed_vals), 99)) if signed_vals.size else None
    return {
        "signed_camera_z_disagreement": {"colormap": "RdBu_r", "center": 0.0, "range": [-signed_p99, signed_p99] if signed_p99 is not None else [None, None], "domain": "primary_common"},
        "absolute_camera_z_disagreement": {"colormap": "viridis", "range": [0.0, p99("absolute_camera_z_disagreement", "primary_common")], "domain": "primary_common"},
        "openmvs_denominated_relative_disagreement": {"colormap": "viridis", "range": [0.0, p99("openmvs_denominated_relative_disagreement", "primary_common")], "domain": "primary_common"},
        "openmvs_gradient_magnitude": {"colormap": "viridis", "range": [0.0, p99("openmvs_gradient_magnitude", "gradient_valid")], "domain": "gradient_valid"},
        "umgs_gradient_magnitude": {"colormap": "viridis", "range": [0.0, p99("umgs_gradient_magnitude", "gradient_valid")], "domain": "gradient_valid"},
        "per_pixel_gradient_cosine": {"colormap": "RdBu_r", "center": 0.0, "range": [-1.0, 1.0], "domain": "gradient_valid_and_cosine_valid"},
        "umgs_camera_z_variance": {"colormap": "viridis", "range": [0.0, p99("umgs_camera_z_variance", "finite_variance")], "domain": "primary_common_and_finite_variance"},
        "umgs_accumulated_opacity": {"colormap": "viridis", "range": [0.0, 1.0], "domain": "primary_common_and_finite_opacity"},
        "invalid_rendering": "transparent",
        "percentile_method": "numpy.percentile default linear method",
        "display_clipping_changes_statistics": False,
    }


def visualization_versions() -> dict[str, str]:
    return {"matplotlib": matplotlib.__version__, "Pillow": Image.__version__}


def save_png(path: Path, values: np.ndarray, valid: np.ndarray, display: dict[str, Any]) -> None:
    cmap_name = display["colormap"]
    if cmap_name not in colormaps:
        raise ProtocolError("visualization_dependency_or_colormap_missing", f"missing matplotlib colormap {cmap_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    lo, hi = display["range"]
    rgba = np.zeros((*values.shape, 4), dtype=np.uint8)
    if lo is None or hi is None or hi == lo:
        norm = np.full(values.shape, 0.5, dtype=np.float64)
    else:
        norm = np.clip((values.astype(np.float64) - float(lo)) / (float(hi) - float(lo)), 0.0, 1.0)
    cmap = colormaps[cmap_name]
    colors = cmap(norm)
    valid_mask = valid & np.isfinite(values)
    rgba[..., :3] = (np.clip(colors[..., :3], 0.0, 1.0) * 255).astype(np.uint8)
    rgba[..., 3] = (valid_mask.astype(np.uint8) * 255)
    Image.fromarray(rgba, "RGBA").save(path)


def run_preflight(args: argparse.Namespace, write_outputs: bool = True) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    verify_file(args.umgs_npz, args.umgs_npz_sha256, "umgs_npz", checks)
    verify_file(args.umgs_manifest, args.umgs_manifest_sha256, "umgs_manifest", checks)
    verify_file(args.openmvs_npz, args.openmvs_npz_sha256, "openmvs_npz", checks)
    verify_file(args.openmvs_manifest, args.openmvs_manifest_sha256, "openmvs_manifest", checks)
    verify_file(args.formal_result_json, args.formal_result_json_sha256, "formal_result_json", checks)
    verify_file(args.formal_result_csv, args.formal_result_csv_sha256, "formal_result_csv", checks)
    verify_file(args.frame_scale_decision_json, args.frame_scale_decision_sha256, "frame_scale_decision", checks)
    verify_runtime_manifest(
        args.layer2_runtime_source_manifest,
        args.layer2_runtime_source_manifest_sha256,
        checks,
        required_roles={"evaluator", "negative_control_source"},
        expected_schema=LAYER2_RUNTIME_MANIFEST_SCHEMA,
        required_runtime_roles={"evaluator", "negative_control_source"},
    )
    verify_file(Path(__file__).resolve(), args.localization_evaluator_sha256, "localization_evaluator_source", checks)
    verify_file(args.corrected_source, args.corrected_source_sha256, "corrected_source", checks)
    verify_file(args.layer2_evaluator, args.layer2_evaluator_sha256, "layer2_evaluator_source", checks)
    verify_runtime_manifest(
        args.localization_runtime_source_manifest,
        args.localization_runtime_source_manifest_sha256,
        checks,
        required_roles={"localization_evaluator", "layer2_evaluator_v1_2", "corrected_gradient_source"},
        expected_schema=LOCALIZATION_RUNTIME_MANIFEST_SCHEMA,
        required_runtime_roles={"localization_evaluator", "layer2_evaluator_v1_2", "corrected_gradient_source"},
    )
    verify_file(args.camera_fingerprint, args.camera_fingerprint_sha256, "camera_fingerprint", checks)

    fingerprint = read_json(args.camera_fingerprint)
    payload = fingerprint.get("payload", {})
    payload_sha = fingerprint.get("fingerprint_sha256")
    computed_payload_sha = sha256_json_payload(payload)
    for actual, expected, name in [
        (payload_sha, args.camera_fingerprint_payload_sha256, "camera_fingerprint_declared_payload_sha"),
        (computed_payload_sha, args.camera_fingerprint_payload_sha256, "camera_fingerprint_computed_payload_sha"),
    ]:
        passed = actual == expected
        checks.append({"check": name, "status": "pass" if passed else "fail", "actual": actual, "expected": expected})
        check(passed, "camera_fingerprint_mismatch", name)
    for name, actual, expected in [
        ("camera_fingerprint_target", payload.get("image_name"), args.expected_target),
        ("camera_fingerprint_width", payload.get("image_width"), args.expected_width),
        ("camera_fingerprint_height", payload.get("image_height"), args.expected_height),
    ]:
        passed = actual == expected
        checks.append({"check": name, "status": "pass" if passed else "fail", "actual": actual, "expected": expected})
        check(passed, "camera_fingerprint_mismatch", f"{name}: expected {expected}, got {actual}")

    result = read_json(args.formal_result_json)
    formal_result_gate(result, checks)

    frame = read_json(args.frame_scale_decision_json)
    frame_status = frame.get("status")
    checks.append({"check": "frame_scale_status", "status": "pass" if frame_status == "correspondence_pass" else "fail", "actual": frame_status})
    check(frame_status == "correspondence_pass", "frame_or_scale_correspondence_unresolved", "frame/scale not correspondence_pass")

    umgs_schema = npz_schema(args.umgs_npz, UMGS_DTYPES, (args.expected_height, args.expected_width), "umgs")
    openmvs_schema = npz_schema(args.openmvs_npz, OPENMVS_DTYPES, (args.expected_height, args.expected_width), "openmvs")

    check(not args.output_root.exists(), "output_overwrite_guard", f"output_root already exists: {args.output_root}")

    payload_out = {
        "schema": SCHEMA,
        "mode": "preflight_only",
        "preflight_status": "pass",
        "phase": PHASE,
        "scene": args.expected_scene,
        "target": args.expected_target,
        "height": args.expected_height,
        "width": args.expected_width,
        "real_localization_computed": False,
        "spatial_statistics_computed": False,
        "png_generated": False,
        "formal_result_gate": "pass",
        "high_gradient_mask_generated": False,
        "binding_checks": checks,
        "umgs_npz_schema": umgs_schema,
        "openmvs_npz_schema": openmvs_schema,
        "environment": environment_report(),
    }
    if write_outputs:
        args.output_root.mkdir(parents=True, exist_ok=False)
        write_json(args.output_root / "road0001_spatial_residual_localization_preflight.json", payload_out)
        write_csv(args.output_root / "road0001_spatial_residual_localization_preflight.csv", [flatten_for_csv(payload_out)])
    return payload_out


def environment_report() -> dict[str, Any]:
    return {
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "Pillow": Image.__version__,
    }


def verify_formal_scalars_against_arrays(
    umgs: dict[str, np.ndarray],
    openmvs: dict[str, np.ndarray],
    primary: np.ndarray,
) -> tuple[dict[str, float], Any, dict[str, Any], np.ndarray, np.ndarray]:
    count = int(primary.sum())
    ratio = float(count / primary.size)
    check(count == EXPECTED_PRIMARY_COMMON_COUNT and ratio == EXPECTED_PRIMARY_COMMON_RATIO, "primary_common_mismatch", "primary common support does not match frozen formal result", {"count": count, "ratio": ratio})
    recomputed = descriptive_metrics(umgs["expected_camera_z"], openmvs["depth"], primary)
    for key in ["mean_absolute_camera_z_disagreement", "median_absolute_camera_z_disagreement", "rmse_camera_z_disagreement"]:
        expected = EXPECTED_SCALARS[key]
        actual = recomputed[key]
        check(abs(actual - expected) <= SCALAR_ATOL + SCALAR_RTOL * abs(expected), "formal_full_image_scalar_mismatch", f"{key} mismatch", {"actual": actual, "expected": expected})

    gradient_domain = corrected.reference_high_gradient_domain(openmvs["depth"], primary)
    core = corrected.metrics_on_mask(openmvs["depth"], umgs["expected_camera_z"], primary, gradient_domain=gradient_domain)
    for key in ["spearman", "high_gradient_cosine_median"]:
        expected = EXPECTED_SCALARS[key]
        actual = float(core[key])
        check(abs(actual - expected) <= SCALAR_ATOL + SCALAR_RTOL * abs(expected), "formal_corrected_core_scalar_mismatch", f"{key} mismatch", {"actual": actual, "expected": expected})

    high = gradient_domain.high_mask
    high_sha = mask_packbits_sha256(high)
    check(gradient_domain.local_valid_count == EXPECTED_GRADIENT_LOCAL_VALID_COUNT, "gradient_local_valid_count_mismatch", "gradient local valid count mismatch", {"actual": gradient_domain.local_valid_count, "expected": EXPECTED_GRADIENT_LOCAL_VALID_COUNT})
    check(abs(float(gradient_domain.threshold) - EXPECTED_HIGH_GRADIENT_THRESHOLD) <= SCALAR_ATOL + SCALAR_RTOL * abs(EXPECTED_HIGH_GRADIENT_THRESHOLD), "high_gradient_threshold_mismatch", "high-gradient threshold mismatch", {"actual": gradient_domain.threshold, "expected": EXPECTED_HIGH_GRADIENT_THRESHOLD})
    check(gradient_domain.high_count == EXPECTED_HIGH_GRADIENT_PIXELS, "high_gradient_count_mismatch", "high-gradient count mismatch", {"actual": gradient_domain.high_count, "expected": EXPECTED_HIGH_GRADIENT_PIXELS})
    check(high_sha == EXPECTED_HIGH_GRADIENT_PACKBITS_SHA256, "high_gradient_packbits_sha_mismatch", "high-gradient packbits SHA mismatch", {"actual": high_sha, "expected": EXPECTED_HIGH_GRADIENT_PACKBITS_SHA256})

    local_valid = corrected.complete_local_mask(primary)
    cosine, cosine_valid = per_pixel_gradient_cosine(openmvs["depth"], umgs["expected_camera_z"], local_valid)
    hg_values = cosine[high & cosine_valid]
    check(hg_values.size > 0, "high_gradient_cosine_empty", "no valid high-gradient cosine values")
    hg_median = float(np.median(hg_values.astype(np.float64)))
    expected_hg = EXPECTED_SCALARS["high_gradient_cosine_median"]
    check(abs(hg_median - expected_hg) <= SCALAR_ATOL + SCALAR_RTOL * abs(expected_hg), "per_pixel_cosine_median_mismatch", "per-pixel high-gradient cosine median mismatch", {"actual": hg_median, "expected": expected_hg})
    return recomputed, gradient_domain, core, cosine, cosine_valid


def write_localization_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **arrays)


def validate_output_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with np.load(path, allow_pickle=False) as npz:
        check(set(npz.files) == set(arrays), "output_npz_reload_validation_failed", "output NPZ keys mismatch")
        for key, arr in arrays.items():
            check(npz[key].shape == arr.shape, "output_npz_reload_validation_failed", f"{key} shape mismatch")
            check(npz[key].dtype == arr.dtype, "output_npz_reload_validation_failed", f"{key} dtype mismatch")
            if np.issubdtype(arr.dtype, np.floating):
                same = np.allclose(npz[key], arr, equal_nan=True, rtol=0.0, atol=0.0)
            else:
                same = np.array_equal(npz[key], arr)
            check(bool(same), "output_npz_reload_validation_failed", f"{key} value mismatch")


def validate_csv_rows(path: Path, expected_count: int, label: str) -> None:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    check(len(rows) == expected_count, "output_csv_reload_validation_failed", f"{label} row count mismatch", {"actual": len(rows), "expected": expected_count})


def write_all_pngs(png_dir: Path, arrays: dict[str, np.ndarray], masks: dict[str, np.ndarray], displays: dict[str, Any]) -> list[Path]:
    png_paths: list[Path] = []
    for key, filename in PNG_SPECS:
        display = displays[key]
        domain = display["domain"]
        if key == "umgs_camera_z_variance":
            valid = masks["finite_variance"]
        elif key == "umgs_accumulated_opacity":
            valid = masks["finite_opacity"]
        elif key in {"openmvs_gradient_magnitude", "umgs_gradient_magnitude"}:
            valid = masks["gradient_valid"]
        elif key == "per_pixel_gradient_cosine":
            valid = masks["gradient_valid_and_cosine_valid"]
        else:
            valid = masks["primary_common"]
        path = png_dir / filename
        save_png(path, arrays[key], valid, display)
        png_paths.append(path)
    return png_paths


def build_output_manifest(root: Path, relative_paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rel in relative_paths:
        path = root / rel
        check(path.exists(), "output_manifest_missing_file", f"missing output artifact {rel}")
        rows.append({"path": rel, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def validate_output_manifest(root: Path, manifest_path: Path) -> None:
    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        path = root / row["path"]
        check(path.exists(), "output_manifest_validation_failed", f"manifest path missing {row['path']}")
        check(int(row["size_bytes"]) == path.stat().st_size, "output_manifest_validation_failed", f"manifest size mismatch {row['path']}")
        check(row["sha256"] == sha256_file(path), "output_manifest_validation_failed", f"manifest SHA mismatch {row['path']}")


def rename_staging_to_failure(staging: Path, output_root: Path, code: str, message: str, generated_files: list[str], tb: str) -> Path | None:
    if not staging.exists():
        return None
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    failure_root = output_root.with_name(f"{output_root.name}_FAILED_{timestamp}")
    suffix = 0
    while failure_root.exists():
        suffix += 1
        failure_root = output_root.with_name(f"{output_root.name}_FAILED_{timestamp}_{suffix}")
    failure_payload = {
        "schema": "road0001_spatial_residual_localization_failure_v1_1",
        "execution_status": "failed",
        "exception_code": code,
        "message": message,
        "generated_files_before_failure": generated_files,
        "traceback": tb,
    }
    write_json(staging / "road0001_spatial_residual_localization_failure.json", failure_payload)
    staging.rename(failure_root)
    return failure_root


def execute_localization(args: argparse.Namespace) -> dict[str, Any]:
    preflight = run_preflight(args, write_outputs=False)
    staging = args.output_root.with_name(args.output_root.name + "_STAGING")
    check(not args.output_root.exists(), "output_overwrite_guard", f"output root exists: {args.output_root}")
    check(not staging.exists(), "staging_overwrite_guard", f"staging root exists: {staging}")
    staging.mkdir(parents=True, exist_ok=False)
    generated: list[str] = []
    try:
        umgs = load_umgs(args.umgs_npz)
        openmvs = load_openmvs(args.openmvs_npz)
        primary = build_primary_mask(umgs, openmvs)
        recomputed, gradient_domain, core, cosine, cosine_valid = verify_formal_scalars_against_arrays(umgs, openmvs, primary)

        high = gradient_domain.high_mask
        signed = np.full(primary.shape, np.nan, dtype=np.float32)
        signed[primary] = (umgs["expected_camera_z"][primary] - openmvs["depth"][primary]).astype(np.float32)
        absolute = np.abs(signed).astype(np.float32)
        relative = np.full(primary.shape, np.nan, dtype=np.float32)
        relative[primary] = (absolute[primary] / np.maximum(np.abs(openmvs["depth"][primary]), 1e-6)).astype(np.float32)
        openmvs_grad = gradient_magnitude(openmvs["depth"]).astype(np.float32)
        umgs_grad = gradient_magnitude(umgs["expected_camera_z"]).astype(np.float32)
        local_valid = corrected.complete_local_mask(primary)
        grid = grid_id_map(*primary.shape)
        border = border_band_id_map(*primary.shape)

        arrays = {
            "primary_common_mask": primary.astype(np.uint8),
            "frozen_high_gradient_mask": high.astype(np.uint8),
            "signed_camera_z_disagreement": signed,
            "absolute_camera_z_disagreement": absolute,
            "openmvs_denominated_relative_disagreement": relative,
            "openmvs_gradient_magnitude": openmvs_grad,
            "umgs_gradient_magnitude": umgs_grad,
            "per_pixel_gradient_cosine": cosine.astype(np.float64),
            "umgs_accumulated_opacity": umgs["accumulated_opacity"].astype(np.float32),
            "umgs_camera_z_variance": umgs["camera_z_variance"].astype(np.float32),
            "grid_id_map": grid.astype(np.int16),
            "border_band_id_map": border.astype(np.int8),
        }
        for key, dtype in OUTPUT_NPZ_DTYPES.items():
            check(arrays[key].dtype == dtype, "output_array_dtype_mismatch", f"{key} dtype mismatch")
            check(arrays[key].shape == primary.shape, "output_array_shape_mismatch", f"{key} shape mismatch")

        grid_names = {i: f"grid_r{i // 8}_c{i % 8}" for i in range(64)}
        border_names = {row["id"]: row["name"] for row in BORDER_BANDS}
        grid_rows = stats_by_label(grid, grid_names, primary, high, cosine_valid, absolute, relative, cosine, umgs["accumulated_opacity"], umgs["camera_z_variance"])
        border_rows = stats_by_label(border, border_names, primary, high, cosine_valid, absolute, relative, cosine, umgs["accumulated_opacity"], umgs["camera_z_variance"])
        masks = {
            "primary_common": primary,
            "gradient_valid": local_valid,
            "gradient_valid_and_cosine_valid": local_valid & cosine_valid,
            "finite_variance": primary & np.isfinite(umgs["camera_z_variance"]),
            "finite_opacity": primary & np.isfinite(umgs["accumulated_opacity"]),
        }
        displays = display_ranges(arrays, masks)

        npz_rel = "road0001_spatial_residual_localization_arrays.npz"
        grid_rel = "road0001_spatial_residual_grid_stats.csv"
        border_rel = "road0001_spatial_residual_border_stats.csv"
        metadata_rel = "road0001_spatial_residual_localization_metadata.json"
        summary_json_rel = "road0001_spatial_residual_localization_execution_summary.json"
        summary_csv_rel = "road0001_spatial_residual_localization_execution_summary.csv"
        manifest_rel = "ROAD0001_SPATIAL_LOCALIZATION_OUTPUT_MANIFEST.csv"

        write_localization_npz(staging / npz_rel, arrays)
        generated.append(npz_rel)
        validate_output_npz(staging / npz_rel, arrays)

        write_csv(staging / grid_rel, grid_rows)
        generated.append(grid_rel)
        write_csv(staging / border_rel, border_rows)
        generated.append(border_rel)
        validate_csv_rows(staging / grid_rel, 64, "grid_stats")
        validate_csv_rows(staging / border_rel, 4, "border_stats")

        png_paths = write_all_pngs(staging / "png_previews", arrays, masks, displays)
        for path in png_paths:
            rel = path.relative_to(staging).as_posix()
            generated.append(rel)
            check(path.exists() and path.stat().st_size > 0, "png_generation_failed", f"PNG missing or empty: {rel}")

        metadata_outputs = [npz_rel, grid_rel, border_rel] + [p.relative_to(staging).as_posix() for p in png_paths]
        metadata = {
            "schema": "road0001_spatial_residual_localization_metadata_v1_1",
            "phase": PHASE,
            "scene": args.expected_scene,
            "target": args.expected_target,
            "formal_result_status": EXPECTED_STATUS,
            "formal_recomputed_scalars": {**recomputed, "spearman": core["spearman"], "high_gradient_cosine_median": core["high_gradient_cosine_median"]},
            "primary_common_count": int(primary.sum()),
            "primary_common_ratio": float(primary.sum() / primary.size),
            "high_gradient": {
                "threshold": float(gradient_domain.threshold),
                "local_valid_count": int(gradient_domain.local_valid_count),
                "count": int(gradient_domain.high_count),
                "packbits_sha256": mask_packbits_sha256(high),
                "packbits_rule": "np.packbits(mask.reshape(-1)).tobytes(), default bitorder, row-major C order",
            },
            "grid_edges": {"rows": ROW_EDGES.astype(int).tolist(), "cols": COL_EDGES.astype(int).tolist()},
            "border_bands": BORDER_BANDS,
            "display_ranges": displays,
            "visualization_dependencies": visualization_versions(),
            "outputs": {rel: {"sha256": sha256_file(staging / rel), "size_bytes": (staging / rel).stat().st_size} for rel in metadata_outputs},
            "metadata_self_sha256_recorded": False,
            "exploratory_non_causal_boundary": True,
            "does_not_change_layer2_taxonomy": True,
            "official_layer2_interpretation_status": EXPECTED_STATUS["interpretation_status"],
        }
        write_json(staging / metadata_rel, metadata)
        generated.append(metadata_rel)
        check("road0001_spatial_residual_localization_metadata.json" not in metadata["outputs"], "metadata_self_hash_recorded", "metadata must not record its own SHA in outputs")

        summary = {
            "schema": "road0001_spatial_residual_localization_execution_summary_v1_1",
            "execution_status": "pass",
            "real_localization_computed": True,
            "spatial_statistics_computed": True,
            "high_gradient_mask_generated": True,
            "png_generated": True,
            "formal_result_consistency": "pass",
            "high_gradient_identity": "pass",
            "does_not_change_layer2_taxonomy": True,
            "phase": PHASE,
            "scene": args.expected_scene,
            "target": args.expected_target,
            "input_hashes": {
                "umgs_npz": args.umgs_npz_sha256,
                "openmvs_npz": args.openmvs_npz_sha256,
                "formal_result_json": args.formal_result_json_sha256,
                "frame_scale_decision": args.frame_scale_decision_sha256,
                "localization_runtime_source_manifest": args.localization_runtime_source_manifest_sha256,
                "layer2_runtime_source_manifest": args.layer2_runtime_source_manifest_sha256,
            },
            "output_root": str(args.output_root),
            "output_artifact_count": len(metadata_outputs) + 3,
            "grid_row_count": len(grid_rows),
            "border_row_count": len(border_rows),
            "official_layer2_interpretation_status": EXPECTED_STATUS["interpretation_status"],
        }
        write_json(staging / summary_json_rel, summary)
        generated.append(summary_json_rel)
        write_csv(staging / summary_csv_rel, [summary])
        generated.append(summary_csv_rel)

        manifest_rows = build_output_manifest(staging, metadata_outputs + [metadata_rel, summary_json_rel, summary_csv_rel])
        write_csv(staging / manifest_rel, manifest_rows, fieldnames=["path", "size_bytes", "sha256"])
        generated.append(manifest_rel)
        validate_output_manifest(staging, staging / manifest_rel)

        check(not args.output_root.exists(), "output_overwrite_guard", f"output root exists before final rename: {args.output_root}")
        staging.rename(args.output_root)
        return {"schema": SCHEMA, "mode": "execute_localization", "output_root": str(args.output_root), "summary": summary, "preflight": preflight}
    except Exception as exc:
        code = exc.code if isinstance(exc, ProtocolError) else exc.__class__.__name__
        failure_root = rename_staging_to_failure(staging, args.output_root, code, str(exc), generated, traceback.format_exc())
        if isinstance(exc, ProtocolError):
            exc.details = {**exc.details, "failure_root": str(failure_root) if failure_root else None}
            raise
        raise ProtocolError("localization_execution_failed", str(exc), {"failure_root": str(failure_root) if failure_root else None}) from exc


def flatten_for_csv(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": payload.get("schema"),
        "mode": payload.get("mode"),
        "preflight_status": payload.get("preflight_status"),
        "phase": payload.get("phase"),
        "scene": payload.get("scene"),
        "target": payload.get("target"),
        "height": payload.get("height"),
        "width": payload.get("width"),
        "real_localization_computed": payload.get("real_localization_computed"),
        "spatial_statistics_computed": payload.get("spatial_statistics_computed"),
        "png_generated": payload.get("png_generated"),
        "formal_result_gate": payload.get("formal_result_gate"),
        "high_gradient_mask_generated": payload.get("high_gradient_mask_generated"),
        "binding_check_count": len(payload.get("binding_checks", [])),
    }


def build_future_command_not_executed(args: argparse.Namespace, python_exe: str = "python") -> str:
    output = str(args.output_root).replace("\\", "/")
    script = str(Path(__file__).as_posix())
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "# NOT_EXECUTED: this command is a frozen future execution request only.",
            f'OUTPUT_ROOT="{output}"',
            'test ! -e "$OUTPUT_ROOT"',
            f'{python_exe} "{script}" \\',
            "  --execute-localization \\",
            f'  --expected-scene "{args.expected_scene}" \\',
            f'  --expected-target "{args.expected_target}" \\',
            f"  --expected-height {args.expected_height} --expected-width {args.expected_width} \\",
            f'  --umgs-npz "{args.umgs_npz}" --umgs-npz-sha256 {args.umgs_npz_sha256} \\',
            f'  --umgs-manifest "{args.umgs_manifest}" --umgs-manifest-sha256 {args.umgs_manifest_sha256} \\',
            f'  --openmvs-npz "{args.openmvs_npz}" --openmvs-npz-sha256 {args.openmvs_npz_sha256} \\',
            f'  --openmvs-manifest "{args.openmvs_manifest}" --openmvs-manifest-sha256 {args.openmvs_manifest_sha256} \\',
            f'  --formal-result-json "{args.formal_result_json}" --formal-result-json-sha256 {args.formal_result_json_sha256} \\',
            f'  --formal-result-csv "{args.formal_result_csv}" --formal-result-csv-sha256 {args.formal_result_csv_sha256} \\',
            f'  --frame-scale-decision-json "{args.frame_scale_decision_json}" --frame-scale-decision-sha256 {args.frame_scale_decision_sha256} \\',
            f'  --layer2-runtime-source-manifest "{args.layer2_runtime_source_manifest}" --layer2-runtime-source-manifest-sha256 {args.layer2_runtime_source_manifest_sha256} \\',
            f'  --layer2-evaluator "{args.layer2_evaluator}" --layer2-evaluator-sha256 {args.layer2_evaluator_sha256} \\',
            f'  --corrected-source "{args.corrected_source}" --corrected-source-sha256 {args.corrected_source_sha256} \\',
            f'  --camera-fingerprint "{args.camera_fingerprint}" --camera-fingerprint-sha256 {args.camera_fingerprint_sha256} \\',
            f'  --camera-fingerprint-payload-sha256 {args.camera_fingerprint_payload_sha256} \\',
            f'  --localization-evaluator-sha256 {args.localization_evaluator_sha256} \\',
            f'  --localization-runtime-source-manifest "{args.localization_runtime_source_manifest}" --localization-runtime-source-manifest-sha256 {args.localization_runtime_source_manifest_sha256} \\',
            f'  --output-root "$OUTPUT_ROOT"',
            "",
        ]
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--execute-localization", action="store_true")
    p.add_argument("--expected-scene", default=EXPECTED_SCENE)
    p.add_argument("--expected-target", default=EXPECTED_TARGET)
    p.add_argument("--expected-height", type=int, default=EXPECTED_HEIGHT)
    p.add_argument("--expected-width", type=int, default=EXPECTED_WIDTH)
    p.add_argument("--umgs-npz", type=Path, required=True)
    p.add_argument("--umgs-npz-sha256", required=True)
    p.add_argument("--umgs-manifest", type=Path, required=True)
    p.add_argument("--umgs-manifest-sha256", required=True)
    p.add_argument("--openmvs-npz", type=Path, required=True)
    p.add_argument("--openmvs-npz-sha256", required=True)
    p.add_argument("--openmvs-manifest", type=Path, required=True)
    p.add_argument("--openmvs-manifest-sha256", required=True)
    p.add_argument("--formal-result-json", type=Path, required=True)
    p.add_argument("--formal-result-json-sha256", required=True)
    p.add_argument("--formal-result-csv", type=Path, required=True)
    p.add_argument("--formal-result-csv-sha256", required=True)
    p.add_argument("--frame-scale-decision-json", type=Path, required=True)
    p.add_argument("--frame-scale-decision-sha256", required=True)
    p.add_argument("--layer2-runtime-source-manifest", type=Path, required=True)
    p.add_argument("--layer2-runtime-source-manifest-sha256", required=True)
    p.add_argument("--layer2-evaluator", type=Path, default=SCRIPT_DIR / "evaluate_umgs_openmvs_camera_z_proxy_alignment.py")
    p.add_argument("--layer2-evaluator-sha256", required=True)
    p.add_argument("--corrected-source", type=Path, default=SCRIPT_DIR / "openmvs_da3_overlap_corrected.py")
    p.add_argument("--corrected-source-sha256", required=True)
    p.add_argument("--camera-fingerprint", type=Path, required=True)
    p.add_argument("--camera-fingerprint-sha256", required=True)
    p.add_argument("--camera-fingerprint-payload-sha256", required=True)
    p.add_argument("--localization-evaluator-sha256", required=True)
    p.add_argument("--localization-runtime-source-manifest", type=Path, required=True)
    p.add_argument("--localization-runtime-source-manifest-sha256", required=True)
    p.add_argument("--expected-primary-common-count", type=int, default=EXPECTED_PRIMARY_COMMON_COUNT)
    p.add_argument("--expected-primary-common-ratio", type=float, default=EXPECTED_PRIMARY_COMMON_RATIO)
    p.add_argument("--expected-gradient-local-valid-count", type=int, default=EXPECTED_GRADIENT_LOCAL_VALID_COUNT)
    p.add_argument("--expected-high-gradient-threshold", type=float, default=EXPECTED_HIGH_GRADIENT_THRESHOLD)
    p.add_argument("--expected-high-gradient-pixels", type=int, default=EXPECTED_HIGH_GRADIENT_PIXELS)
    p.add_argument("--expected-high-gradient-packbits-sha256", default=EXPECTED_HIGH_GRADIENT_PACKBITS_SHA256)
    p.add_argument("--output-root", type=Path, required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.expected_scene != EXPECTED_SCENE or args.expected_target != EXPECTED_TARGET:
        raise SystemExit("unexpected_scene_or_target")
    if args.expected_height != EXPECTED_HEIGHT or args.expected_width != EXPECTED_WIDTH:
        raise SystemExit("unexpected_raster")
    try:
        if args.preflight_only:
            run_preflight(args, write_outputs=True)
        else:
            execute_localization(args)
    except ProtocolError as exc:
        sys.stderr.write(f"{exc.code}: {exc}\n")
        if exc.details:
            sys.stderr.write(json.dumps(exc.details, indent=2, ensure_ascii=True) + "\n")
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
