#!/usr/bin/env python3
"""UMGS-OpenMVS camera-z proxy-alignment evaluator.

V1.2 hardens the Road-0001 Layer-2 evaluator protocol. It validates all
declared bindings before reading packet arrays for metrics, validates array
schema/dtype before conversion, reuses the corrected OpenMVS-DA3 shuffle and
core metric implementation, and stops without metrics when support gates fail.
It also provides a preflight-only mode that validates production bindings and
array schemas without constructing masks, controls, or metrics.

The output is proxy disagreement/agreement only. It is not geometry ground
truth, physical accuracy, or surface-depth validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import openmvs_da3_overlap_corrected as corrected
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"failed_to_import_corrected_overlap_protocol: {exc}") from exc


SCHEMA = "umgs_openmvs_camera_z_proxy_alignment_v1_2"
RUNTIME_MANIFEST_SCHEMA = "road0001_layer2_runtime_source_manifest_v1_2"
FRAME_SCALE_DECISION_SCHEMA = "road0001_frame_scale_correspondence_decision_v1_2"
CONTROL_PROTOCOL_STATUS = "existing_corrected_negative_control_reused"
CONTROL_SOURCE = "tools/depth_reference_geometry_v2/openmvs_da3_overlap_corrected.py"
EVALUATOR_SOURCE = "tools/depth_reference_geometry_v2/evaluate_umgs_openmvs_camera_z_proxy_alignment.py"

SHUFFLE_SEED = int(corrected.SHUFFLE_SEED)
METRIC_COMPARE_EPS = float(corrected.METRIC_COMPARE_EPS)
MIN_TRUE_PIXELS = int(corrected.MIN_TRUE_PIXELS)
MIN_TRUE_COVERAGE = float(corrected.MIN_TRUE_COVERAGE)
MIN_SHARED_PIXELS = int(corrected.MIN_SHARED_PIXELS)
MIN_SHARED_COVERAGE = float(corrected.MIN_SHARED_COVERAGE)

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


class ProtocolError(RuntimeError):
    """Fail-fast protocol error before metric computation."""

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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_nested(payload: dict[str, Any], *keys: str) -> Any:
    obj: Any = payload
    for key in keys:
        if not isinstance(obj, dict) or key not in obj:
            return None
        obj = obj[key]
    return obj


def check_equal(name: str, actual: Any, expected: Any, checks: list[dict[str, Any]]) -> None:
    passed = actual == expected
    checks.append({"check": name, "status": "pass" if passed else "fail", "actual": actual, "expected": expected})
    if not passed:
        raise ProtocolError("input_hash_mismatch", f"{name}: expected {expected}, got {actual}")


def check_true(name: str, condition: bool, checks: list[dict[str, Any]], **details: Any) -> None:
    checks.append({"check": name, "status": "pass" if condition else "fail", **details})
    if not condition:
        raise ProtocolError("input_hash_mismatch", f"{name} failed", details)


def verify_file_sha(label: str, path: Path, expected_sha: str, checks: list[dict[str, Any]]) -> None:
    if not path.exists():
        checks.append({"check": f"{label}_exists", "status": "fail", "path": str(path)})
        raise ProtocolError("input_hash_mismatch", f"{label} missing: {path}")
    checks.append({"check": f"{label}_exists", "status": "pass", "path": str(path)})
    actual = sha256_file(path)
    check_equal(f"{label}_sha256", actual, expected_sha, checks)


def canonical_payload_hash(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_fingerprint(path: Path, expected_file_sha: str, expected_payload_sha: str, label: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
    verify_file_sha(label, path, expected_file_sha, checks)
    fp = json_load(path)
    payload = fp.get("payload")
    if not isinstance(payload, dict):
        raise ProtocolError("camera_mismatch", f"{label} missing payload")
    declared = fp.get("fingerprint_sha256")
    computed = canonical_payload_hash(payload)
    check_equal(f"{label}_declared_payload_sha256", declared, expected_payload_sha, checks)
    check_equal(f"{label}_computed_payload_sha256", computed, expected_payload_sha, checks)
    return fp


def compare_fingerprints(umgs_fp: dict[str, Any], openmvs_fp: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    up = umgs_fp["payload"]
    op = openmvs_fp["payload"]
    for key in ["image_name", "image_width", "image_height", "FoVx", "FoVy", "R", "T", "world_view_transform", "projection_matrix", "full_proj_transform"]:
        check_equal(f"fingerprint_payload_{key}", op.get(key), up.get(key), checks)
    # COLMAP id may be stored as colmap_id in the canonical fingerprint.
    check_equal("fingerprint_payload_colmap_id", op.get("colmap_id"), up.get("colmap_id"), checks)


def verify_layer2_runtime_manifest(path: Path, expected_sha: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
    verify_file_sha("layer2_runtime_source_manifest", path, expected_sha, checks)
    manifest = json_load(path)
    check_equal("layer2_runtime_manifest_schema", manifest.get("schema"), RUNTIME_MANIFEST_SCHEMA, checks)
    repo_root_value = manifest.get("repo_root")
    check_true("layer2_runtime_manifest_has_repo_root", isinstance(repo_root_value, str) and bool(repo_root_value), checks, repo_root=repo_root_value)
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ProtocolError("input_hash_mismatch", "layer2 runtime manifest has no sources")
    repo_root = Path(repo_root_value)
    roles = set()
    for entry in sources:
        role = entry.get("role")
        rel = entry.get("repo_relative_path")
        sha = entry.get("sha256")
        check_true("runtime_source_entry_has_role", isinstance(role, str) and bool(role), checks, entry=entry)
        check_true("runtime_source_entry_has_repo_relative_path", isinstance(rel, str) and bool(rel), checks, entry=entry)
        check_true("runtime_source_entry_has_sha256", isinstance(sha, str) and len(sha) == 64, checks, entry=entry)
        roles.add(role)
        p = repo_root / rel
        verify_file_sha(f"runtime_source_{role}", p, sha, checks)
    check_true("runtime_source_manifest_has_evaluator", "evaluator" in roles, checks, roles=sorted(roles))
    check_true("runtime_source_manifest_has_negative_control_source", "negative_control_source" in roles, checks, roles=sorted(roles))
    return manifest


def sparse_hashes_from_umgs_manifest(manifest: dict[str, Any]) -> dict[str, str]:
    sparse = get_nested(manifest, "preflight", "colmap_sparse_hash_manifest")
    if not isinstance(sparse, dict):
        raise ProtocolError("input_hash_mismatch", "umgs sparse manifest missing")
    return {row.get("file"): row.get("sha256") for row in sparse.get("files", []) if isinstance(row, dict)}


def verify_frame_scale_decision(path: Path, expected_sha: str, args: argparse.Namespace, umgs_manifest: dict[str, Any], checks: list[dict[str, Any]]) -> dict[str, Any]:
    verify_file_sha("frame_scale_decision", path, expected_sha, checks)
    decision = json_load(path)
    check_equal("frame_scale_decision_schema", decision.get("schema"), FRAME_SCALE_DECISION_SCHEMA, checks)
    status = decision.get("status")
    if status != "correspondence_pass":
        checks.append({"check": "frame_scale_decision_status", "status": "fail", "actual": status})
        raise ProtocolError("frame_or_scale_correspondence_unresolved", "frame/scale decision is not correspondence_pass")
    checks.append({"check": "frame_scale_decision_status", "status": "pass", "actual": status})
    check_equal("frame_scale_decision_scene", decision.get("scene"), args.expected_scene, checks)
    check_equal("frame_scale_decision_target", decision.get("target"), args.expected_target, checks)
    required = [
        "umgs_checkpoint_sparse_root",
        "openmvs_materialization_source_sparse_root",
        "openmvs_source_image_only_materialized_sparse_root",
        "source_sparse_hashes",
        "source_image_only_materialized_sparse_hashes",
        "transform_audit",
        "mesh_provenance",
        "camera_qualification",
    ]
    for key in required:
        check_true(f"frame_scale_decision_has_{key}", key in decision, checks)

    check_equal(
        "frame_scale_decision_source_roots_match",
        decision.get("umgs_checkpoint_sparse_root"),
        decision.get("openmvs_materialization_source_sparse_root"),
        checks,
    )
    source_hashes = decision.get("source_sparse_hashes")
    materialized_hashes = decision.get("source_image_only_materialized_sparse_hashes")
    check_true("frame_scale_source_hashes_dict", isinstance(source_hashes, dict), checks)
    check_true("frame_scale_materialized_hashes_dict", isinstance(materialized_hashes, dict), checks)
    umgs_sparse = sparse_hashes_from_umgs_manifest(umgs_manifest)
    for key in ["cameras.bin", "images.bin", "points3D.bin"]:
        check_equal(f"frame_scale_source_sparse_hash_{key}", source_hashes.get(key), umgs_sparse.get(key), checks)
    check_equal("frame_scale_materialized_cameras_matches_source", materialized_hashes.get("cameras.bin"), source_hashes.get("cameras.bin"), checks)

    mesh = decision.get("mesh_provenance")
    camera = decision.get("camera_qualification")
    check_true("frame_scale_mesh_provenance_dict", isinstance(mesh, dict), checks)
    check_true("frame_scale_camera_qualification_dict", isinstance(camera, dict), checks)
    check_equal("frame_scale_mesh_ply_sha256", mesh.get("mesh_ply_sha256"), args.openmvs_mesh_ply_sha256, checks)
    check_equal("frame_scale_mesh_mvs_sha256", mesh.get("mesh_mvs_sha256"), args.openmvs_mesh_mvs_sha256, checks)
    check_equal("frame_scale_camera_file_sha256", camera.get("canonical_camera_file_sha256"), args.umgs_fingerprint_sha256, checks)
    check_equal("frame_scale_camera_payload_sha256", camera.get("canonical_camera_payload_sha256"), args.umgs_fingerprint_payload_sha256, checks)
    check_equal("frame_scale_camera_width", camera.get("width"), int(args.expected_width), checks)
    check_equal("frame_scale_camera_height", camera.get("height"), int(args.expected_height), checks)
    return decision


def verify_umgs_manifest(manifest: dict[str, Any], args: argparse.Namespace, checks: list[dict[str, Any]]) -> None:
    check_equal("umgs_schema_version", get_nested(manifest, "schema", "schema_version"), "umgs_expected_camera_z_packet_v1", checks)
    check_equal("umgs_scene", manifest.get("scene"), args.expected_scene, checks)
    check_equal("umgs_target", manifest.get("target"), args.expected_target, checks)
    check_equal("umgs_raster_height", manifest.get("raster_height"), int(args.expected_height), checks)
    check_equal("umgs_raster_width", manifest.get("raster_width"), int(args.expected_width), checks)
    check_equal("umgs_checkpoint_sha256", manifest.get("checkpoint_sha256"), args.umgs_checkpoint_sha256, checks)
    check_equal("umgs_npz_sha256", manifest.get("npz_sha256"), args.umgs_npz_sha256, checks)
    check_equal("umgs_fingerprint_file_sha256", get_nested(manifest, "target_camera_fingerprint_comparison", "expected_file_sha256"), args.umgs_fingerprint_sha256, checks)
    check_equal("umgs_fingerprint_payload_sha256", get_nested(manifest, "target_camera_fingerprint", "fingerprint_sha256"), args.umgs_fingerprint_payload_sha256, checks)
    check_equal("umgs_runtime_manifest_sha256", get_nested(manifest, "preflight", "runtime_source_hash_manifest", "sha256"), args.umgs_runtime_manifest_sha256, checks)
    sparse = get_nested(manifest, "preflight", "colmap_sparse_hash_manifest")
    check_true("umgs_sparse_manifest_present", isinstance(sparse, dict), checks)
    required_files = {r.get("file"): r.get("sha256") for r in sparse.get("files", [])}
    for key in ["cameras.bin", "images.bin", "points3D.bin"]:
        check_true(f"umgs_sparse_required_{key}", key in required_files and bool(required_files[key]), checks, sha256=required_files.get(key))
    split = get_nested(manifest, "preflight", "manifest_info", "split_manifest")
    check_true("umgs_split_manifest_present", isinstance(split, dict), checks)
    check_equal("umgs_train_hash_bound", split.get("train_file_actual_sha256"), split.get("train_file_expected_sha256"), checks)
    check_equal("umgs_test_hash_bound", split.get("test_file_actual_sha256"), split.get("test_file_expected_sha256"), checks)


def verify_openmvs_manifest(manifest: dict[str, Any], args: argparse.Namespace, checks: list[dict[str, Any]]) -> None:
    check_equal("openmvs_schema", manifest.get("schema"), "openmvs_canonical_camera_triangle_render_v1", checks)
    check_equal("openmvs_scene", manifest.get("scene"), args.expected_scene, checks)
    check_equal("openmvs_target", manifest.get("target"), args.expected_target, checks)
    check_equal("openmvs_output_height", get_nested(manifest, "output", "height"), int(args.expected_height), checks)
    check_equal("openmvs_output_width", get_nested(manifest, "output", "width"), int(args.expected_width), checks)
    check_equal("openmvs_npz_sha256", get_nested(manifest, "output", "npz_sha256"), args.openmvs_npz_sha256, checks)
    check_equal("openmvs_fingerprint_file_sha256", get_nested(manifest, "fingerprint", "file_sha256"), args.openmvs_fingerprint_sha256, checks)
    check_equal("openmvs_fingerprint_payload_sha256", get_nested(manifest, "fingerprint", "payload_sha256"), args.openmvs_fingerprint_payload_sha256, checks)
    check_equal("openmvs_mesh_ply_sha256", get_nested(manifest, "mesh", "ply_sha256"), args.openmvs_mesh_ply_sha256, checks)
    check_equal("openmvs_mesh_mvs_sha256", get_nested(manifest, "mesh", "mvs_sha256"), args.openmvs_mesh_mvs_sha256, checks)
    check_equal("openmvs_runtime_manifest_sha256", get_nested(manifest, "runtime_source_manifest", "sha256"), args.openmvs_runtime_manifest_sha256, checks)
    source_files = get_nested(manifest, "runtime_source_manifest", "source_files")
    check_true("openmvs_runtime_source_files_present", isinstance(source_files, list), checks)
    source_by_role = {row.get("role"): row.get("sha256") for row in source_files}
    check_equal("openmvs_adapter_sha256", source_by_role.get("adapter"), args.openmvs_adapter_sha256, checks)
    check_equal("openmvs_core_sha256", source_by_role.get("core_rasterizer_mesh_loader"), args.openmvs_core_sha256, checks)
    check_equal("openmvs_no_proxy_metric", manifest.get("no_proxy_metric"), True, checks)
    check_equal("openmvs_not_ground_truth", manifest.get("not_ground_truth"), True, checks)
    check_equal("openmvs_valid_definition", manifest.get("valid_definition"), "triangle_hit AND finite_camera_z AND camera_z_gt_0", checks)
    check_equal("openmvs_pixel_center_convention", manifest.get("pixel_center_convention"), "corner-origin_pixel-centers_at_index_plus_0.5", checks)
    check_equal("openmvs_principal_point_policy", manifest.get("principal_point_policy"), "centered_fov_projection_cx_width_over_2_cy_height_over_2", checks)


def validate_bindings(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    verify_file_sha("umgs_npz", args.umgs_npz, args.umgs_npz_sha256, checks)
    verify_file_sha("umgs_manifest", args.umgs_manifest, args.umgs_manifest_sha256, checks)
    verify_file_sha("openmvs_npz", args.openmvs_npz, args.openmvs_npz_sha256, checks)
    verify_file_sha("openmvs_manifest", args.openmvs_manifest, args.openmvs_manifest_sha256, checks)
    verify_layer2_runtime_manifest(args.layer2_runtime_source_manifest, args.layer2_runtime_source_manifest_sha256, checks)

    umgs_manifest = json_load(args.umgs_manifest)
    openmvs_manifest = json_load(args.openmvs_manifest)
    verify_umgs_manifest(umgs_manifest, args, checks)
    verify_openmvs_manifest(openmvs_manifest, args, checks)

    umgs_fp = load_fingerprint(args.umgs_fingerprint, args.umgs_fingerprint_sha256, args.umgs_fingerprint_payload_sha256, "umgs_fingerprint", checks)
    openmvs_fp = load_fingerprint(args.openmvs_fingerprint, args.openmvs_fingerprint_sha256, args.openmvs_fingerprint_payload_sha256, "openmvs_fingerprint", checks)
    compare_fingerprints(umgs_fp, openmvs_fp, checks)
    check_equal("manifest_fingerprint_payload_consistency", args.umgs_fingerprint_payload_sha256, args.openmvs_fingerprint_payload_sha256, checks)
    verify_frame_scale_decision(args.frame_scale_decision_json, args.frame_scale_decision_sha256, args, umgs_manifest, checks)
    return umgs_manifest, openmvs_manifest, umgs_fp, checks


def validate_npz_schema(path: Path, expected: dict[str, np.dtype], expected_shape: tuple[int, int], label: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(set(expected).difference(data.files))
        if missing:
            raise ProtocolError("schema_or_dtype_mismatch", f"{label} missing keys {missing}")
        for key, dtype in expected.items():
            arr = data[key]
            if arr.dtype != dtype:
                raise ProtocolError("schema_or_dtype_mismatch", f"{label}.{key} dtype {arr.dtype} != {dtype}")
            if key == "barycentric":
                if arr.shape != (expected_shape[0], expected_shape[1], 3):
                    raise ProtocolError("schema_or_dtype_mismatch", f"{label}.{key} shape {arr.shape} != {expected_shape + (3,)}")
            else:
                if arr.shape != expected_shape:
                    raise ProtocolError("schema_or_dtype_mismatch", f"{label}.{key} shape {arr.shape} != {expected_shape}")
            out[key] = np.array(arr, copy=True)
    return out


def finite_positive(arr: np.ndarray) -> np.ndarray:
    return np.isfinite(arr) & (arr > 0)


def bool_uint8_mask(arr: np.ndarray, label: str) -> np.ndarray:
    if arr.dtype != np.dtype("uint8"):
        raise ProtocolError("schema_or_dtype_mismatch", f"{label} dtype {arr.dtype} != uint8")
    vals = np.unique(arr)
    if not set(vals.tolist()).issubset({0, 1}):
        raise ProtocolError("schema_or_dtype_mismatch", f"{label} contains non-binary uint8 values {vals.tolist()}")
    return arr.astype(bool)


def array_schema_summary(umgs: dict[str, np.ndarray], openmvs: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "umgs_shape": list(umgs["expected_camera_z"].shape),
        "openmvs_shape": list(openmvs["depth"].shape),
        "umgs_expected_camera_z_dtype": str(umgs["expected_camera_z"].dtype),
        "umgs_numeric_valid_dtype": str(umgs["numeric_valid"].dtype),
        "openmvs_depth_dtype": str(openmvs["depth"].dtype),
        "openmvs_valid_dtype": str(openmvs["valid"].dtype),
        "openmvs_triangle_id_dtype": str(openmvs["triangle_id"].dtype),
        "openmvs_barycentric_dtype": str(openmvs["barycentric"].dtype),
        "openmvs_barycentric_shape": list(openmvs["barycentric"].shape),
    }


def environment_report() -> dict[str, str]:
    return {
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def build_primary_mask(umgs_z: np.ndarray, umgs_valid: np.ndarray, openmvs_z: np.ndarray, openmvs_valid: np.ndarray) -> np.ndarray:
    return (
        umgs_valid
        & openmvs_valid
        & np.isfinite(umgs_z)
        & np.isfinite(openmvs_z)
        & (umgs_z > 0)
        & (openmvs_z > 0)
    )


def support_counts(mask: np.ndarray) -> dict[str, Any]:
    pixels = int(mask.sum())
    coverage = float(pixels / mask.size) if mask.size else 0.0
    return {"pixels": pixels, "coverage": coverage}


def descriptive_native_metrics(umgs_z: np.ndarray, openmvs_z: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    m = mask & finite_positive(umgs_z) & finite_positive(openmvs_z)
    counts = support_counts(m)
    if counts["pixels"] == 0:
        return {
            **counts,
            "mean_signed_camera_z_disagreement": None,
            "median_signed_camera_z_disagreement": None,
            "mean_absolute_camera_z_disagreement": None,
            "median_absolute_camera_z_disagreement": None,
            "rmse_camera_z_disagreement": None,
            "absolute_camera_z_disagreement_p90": None,
            "absolute_camera_z_disagreement_p95": None,
        }
    uz = umgs_z[m].astype(np.float64)
    oz = openmvs_z[m].astype(np.float64)
    diff = uz - oz
    absdiff = np.abs(diff)
    return {
        **counts,
        "mean_signed_camera_z_disagreement": float(np.mean(diff)),
        "median_signed_camera_z_disagreement": float(np.median(diff)),
        "mean_absolute_camera_z_disagreement": float(np.mean(absdiff)),
        "median_absolute_camera_z_disagreement": float(np.median(absdiff)),
        "rmse_camera_z_disagreement": float(np.sqrt(np.mean(diff * diff))),
        "absolute_camera_z_disagreement_p90": float(np.percentile(absdiff, 90)),
        "absolute_camera_z_disagreement_p95": float(np.percentile(absdiff, 95)),
    }


def deterministic_shuffle(depth: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shuffled_depth, shuffled_valid = corrected.branch_native(depth, valid, "shuffle")
    return shuffled_depth, np.asarray(shuffled_valid, dtype=bool)


def corrected_core_metrics(openmvs_z: np.ndarray, candidate_z: np.ndarray, shared_mask: np.ndarray, gradient_domain: Any) -> dict[str, Any]:
    raw = corrected.metrics_on_mask(openmvs_z, candidate_z, shared_mask, gradient_domain=gradient_domain)
    return {
        "openmvs_denominated_relative_camera_z_disagreement_median": raw.get("absrel_median"),
        "openmvs_denominated_relative_camera_z_disagreement_p90": raw.get("absrel_p90"),
        "spearman": raw.get("spearman"),
        "pearson": raw.get("pearson"),
        "high_gradient_cosine_median": raw.get("high_gradient_cosine_median"),
        "high_gradient_threshold": raw.get("high_gradient_threshold"),
        "high_gradient_pixels": raw.get("high_gradient_pixels"),
        "gradient_local_valid_pixels": raw.get("gradient_local_valid_pixels"),
        "gradient_erosion_rule": raw.get("gradient_erosion_rule"),
        "corrected_source_fields": {
            "relative": "absrel_median",
            "spearman": "spearman",
            "high_gradient": "high_gradient_cosine_median",
        },
    }


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def compare_core(true_metrics: dict[str, Any], control_metrics: dict[str, Any]) -> dict[str, str]:
    comparisons: dict[str, str] = {}
    t_rel = safe_float(true_metrics.get("openmvs_denominated_relative_camera_z_disagreement_median"))
    c_rel = safe_float(control_metrics.get("openmvs_denominated_relative_camera_z_disagreement_median"))
    if t_rel is None or c_rel is None:
        comparisons["openmvs_denominated_relative_camera_z_disagreement_median"] = "invalid"
    elif t_rel < c_rel - METRIC_COMPARE_EPS:
        comparisons["openmvs_denominated_relative_camera_z_disagreement_median"] = "improve"
    elif t_rel > c_rel + METRIC_COMPARE_EPS:
        comparisons["openmvs_denominated_relative_camera_z_disagreement_median"] = "degrade"
    else:
        comparisons["openmvs_denominated_relative_camera_z_disagreement_median"] = "tie"

    for key in ["spearman", "high_gradient_cosine_median"]:
        t = safe_float(true_metrics.get(key))
        c = safe_float(control_metrics.get(key))
        if t is None or c is None:
            comparisons[key] = "invalid"
        elif t > c + METRIC_COMPARE_EPS:
            comparisons[key] = "improve"
        elif t < c - METRIC_COMPARE_EPS:
            comparisons[key] = "degrade"
        else:
            comparisons[key] = "tie"
    return comparisons


def classify_interpretation(true_core: dict[str, Any], comparisons: dict[str, str]) -> tuple[str, str]:
    valid = {k: v for k, v in comparisons.items() if v != "invalid"}
    if len(valid) < 2:
        return "not_evaluated", "fewer_than_two_valid_core_comparisons"
    improves = sum(1 for v in valid.values() if v == "improve")
    degrades = sum(1 for v in valid.values() if v == "degrade")
    if (
        len(valid) == 3
        and improves == 3
        and safe_float(true_core.get("spearman")) is not None
        and safe_float(true_core.get("high_gradient_cosine_median")) is not None
        and float(true_core["spearman"]) >= 0.70
        and float(true_core["high_gradient_cosine_median"]) >= 0.70
    ):
        return "proxy_alignment_strong", ""
    if len(valid) >= 2 and improves >= 1 and degrades >= 1:
        return "proxy_alignment_contradictory", ""
    return "proxy_alignment_weak_or_mixed", ""


def inconclusive_payload(input_status: str, support_status: str, reason: str, binding_checks: list[dict[str, Any]] | None = None, support: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "input_status": input_status,
        "support_status": support_status,
        "control_shared_support_status": "not_evaluated",
        "metric_status": "metric_inconclusive",
        "interpretation_status": "not_evaluated",
        "inconclusive_reason": reason,
        "binding_checks": binding_checks or [],
        "support_counts": support or {},
        "control_protocol_status": CONTROL_PROTOCOL_STATUS,
        "not_ground_truth": True,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    binding_checks: list[dict[str, Any]] = []
    try:
        _, _, _, binding_checks = validate_bindings(args)
        shape = (int(args.expected_height), int(args.expected_width))
        umgs = validate_npz_schema(args.umgs_npz, UMGS_DTYPES, shape, "umgs")
        openmvs = validate_npz_schema(args.openmvs_npz, OPENMVS_DTYPES, shape, "openmvs")
    except ProtocolError as exc:
        status = exc.code if exc.code in {"frame_or_scale_correspondence_unresolved", "camera_mismatch", "input_hash_mismatch"} else "input_hash_mismatch"
        return inconclusive_payload(status, "not_evaluated", exc.code, binding_checks, exc.details)

    if getattr(args, "preflight_only", False):
        return {
            "schema": SCHEMA,
            "preflight_only": True,
            "preflight_status": "preflight_pass",
            "real_metrics_computed": False,
            "input_status": "correspondence_pass",
            "support_status": "not_evaluated",
            "control_shared_support_status": "not_evaluated",
            "metric_status": "not_evaluated",
            "interpretation_status": "not_evaluated",
            "inconclusive_reason": "preflight_only_no_metric_execution",
            "control_protocol_status": args.negative_control_status,
            "binding_checks": binding_checks,
            "array_summary": array_schema_summary(umgs, openmvs),
            "environment": environment_report(),
            "real_common_mask_constructed": False,
            "support_count_computed": False,
            "shuffle_control_generated": False,
            "taxonomy_computed": False,
            "not_ground_truth": True,
        }

    try:
        umgs_z = umgs["expected_camera_z"]
        umgs_valid = bool_uint8_mask(umgs["numeric_valid"], "umgs.numeric_valid")
        openmvs_z = openmvs["depth"]
        openmvs_valid = bool_uint8_mask(openmvs["valid"], "openmvs.valid")
    except ProtocolError as exc:
        return inconclusive_payload("input_hash_mismatch", "not_evaluated", exc.code, binding_checks, exc.details)
    true_mask = build_primary_mask(umgs_z, umgs_valid, openmvs_z, openmvs_valid)
    primary = support_counts(true_mask)
    primary_support = {
        "total_pixels": int(true_mask.size),
        "umgs_valid_count": int((umgs_valid & finite_positive(umgs_z)).sum()),
        "umgs_valid_ratio": float((umgs_valid & finite_positive(umgs_z)).mean()),
        "openmvs_valid_count": int((openmvs_valid & finite_positive(openmvs_z)).sum()),
        "openmvs_valid_ratio": float((openmvs_valid & finite_positive(openmvs_z)).mean()),
        "primary_common_count": primary["pixels"],
        "primary_common_ratio": primary["coverage"],
        "min_true_pixels": MIN_TRUE_PIXELS,
        "min_true_coverage": MIN_TRUE_COVERAGE,
    }
    if primary["pixels"] < MIN_TRUE_PIXELS or primary["coverage"] < MIN_TRUE_COVERAGE:
        return inconclusive_payload("correspondence_pass", "insufficient_common_support", "insufficient_primary_common_support", binding_checks, primary_support)

    true_descriptive = descriptive_native_metrics(umgs_z, openmvs_z, true_mask)
    control_z, control_valid = deterministic_shuffle(umgs_z, umgs_valid)
    control_mask = build_primary_mask(control_z, control_valid, openmvs_z, openmvs_valid)
    shared_mask = true_mask & control_mask
    shared = support_counts(shared_mask)
    control_shared_status = "control_shared_support_sufficient" if shared["pixels"] >= MIN_SHARED_PIXELS and shared["coverage"] >= MIN_SHARED_COVERAGE else "insufficient_shared_control_support"
    support_block = {
        **primary_support,
        "shared_control_count": shared["pixels"],
        "shared_control_ratio": shared["coverage"],
        "min_shared_pixels": MIN_SHARED_PIXELS,
        "min_shared_coverage": MIN_SHARED_COVERAGE,
    }
    if control_shared_status != "control_shared_support_sufficient":
        return {
            "schema": SCHEMA,
            "input_status": "correspondence_pass",
            "support_status": "common_support_sufficient",
            "control_shared_support_status": control_shared_status,
            "metric_status": "metrics_valid",
            "interpretation_status": "not_evaluated",
            "inconclusive_reason": "negative_control_inconclusive_due_to_shared_support",
            "binding_checks": binding_checks,
            "support_counts": support_block,
            "true_descriptive_metrics": true_descriptive,
            "control_protocol_status": args.negative_control_status,
            "not_ground_truth": True,
        }

    if args.negative_control_status != CONTROL_PROTOCOL_STATUS:
        return {
            "schema": SCHEMA,
            "input_status": "correspondence_pass",
            "support_status": "common_support_sufficient",
            "control_shared_support_status": control_shared_status,
            "metric_status": "metrics_valid",
            "interpretation_status": "not_evaluated",
            "inconclusive_reason": "negative_control_not_approved",
            "binding_checks": binding_checks,
            "support_counts": support_block,
            "true_descriptive_metrics": true_descriptive,
            "control_protocol_status": args.negative_control_status,
            "not_ground_truth": True,
        }

    shared_gd = corrected.reference_high_gradient_domain(openmvs_z, shared_mask)
    true_core = corrected_core_metrics(openmvs_z, umgs_z, shared_mask, shared_gd)
    control_core = corrected_core_metrics(openmvs_z, control_z, shared_mask, shared_gd)
    comparisons = compare_core(true_core, control_core)
    valid_core = sum(1 for v in comparisons.values() if v != "invalid")
    metric_status = "metrics_valid" if valid_core >= 2 else "metric_inconclusive"
    if metric_status != "metrics_valid":
        interpretation, reason = "not_evaluated", "fewer_than_two_valid_core_comparisons"
    else:
        interpretation, reason = classify_interpretation(true_core, comparisons)
    return {
        "schema": SCHEMA,
        "input_status": "correspondence_pass",
        "support_status": "common_support_sufficient",
        "control_shared_support_status": control_shared_status,
        "metric_status": metric_status,
        "interpretation_status": interpretation,
        "inconclusive_reason": reason,
        "control_protocol_status": args.negative_control_status,
        "control_source": CONTROL_SOURCE,
        "shuffle_seed": SHUFFLE_SEED,
        "metric_compare_eps": METRIC_COMPARE_EPS,
        "min_true_pixels": MIN_TRUE_PIXELS,
        "min_true_coverage": MIN_TRUE_COVERAGE,
        "min_shared_pixels": MIN_SHARED_PIXELS,
        "min_shared_coverage": MIN_SHARED_COVERAGE,
        "primary_common_mask_definition": "UMGS numeric_valid AND OpenMVS valid AND finite UMGS expected_camera_z AND finite OpenMVS depth AND UMGS expected_camera_z > 0 AND OpenMVS depth > 0",
        "normalization_decision": "none_option_A_native_unit_plus_openmvs_denominated_relative_sensitivity_only",
        "no_opacity_or_variance_threshold": True,
        "not_ground_truth": True,
        "binding_checks": binding_checks,
        "array_summary": array_schema_summary(umgs, openmvs),
        "environment": environment_report(),
        "real_metrics_computed": True,
        "support_counts": support_block,
        "true_descriptive_metrics": true_descriptive,
        "true_core_metrics_on_shared": true_core,
        "shuffle_control_core_metrics_on_shared": control_core,
        "core_comparisons": comparisons,
    }


def flatten_for_csv(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    def rec(prefix: str, value: Any) -> None:
        key = prefix[:-1] if prefix.endswith("_") else prefix
        if isinstance(value, dict):
            for k, v in value.items():
                rec(f"{prefix}{k}_", v)
        elif isinstance(value, list):
            out[key] = json.dumps(value, ensure_ascii=True)
        else:
            out[key] = value

    rec("", payload)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--umgs-npz", type=Path, required=True)
    p.add_argument("--umgs-npz-sha256", required=True)
    p.add_argument("--umgs-manifest", type=Path, required=True)
    p.add_argument("--umgs-manifest-sha256", required=True)
    p.add_argument("--umgs-checkpoint-sha256", required=True)
    p.add_argument("--umgs-fingerprint", type=Path, required=True)
    p.add_argument("--umgs-fingerprint-sha256", required=True)
    p.add_argument("--umgs-fingerprint-payload-sha256", required=True)
    p.add_argument("--umgs-runtime-manifest-sha256", required=True)
    p.add_argument("--openmvs-npz", type=Path, required=True)
    p.add_argument("--openmvs-npz-sha256", required=True)
    p.add_argument("--openmvs-manifest", type=Path, required=True)
    p.add_argument("--openmvs-manifest-sha256", required=True)
    p.add_argument("--openmvs-mesh-ply-sha256", required=True)
    p.add_argument("--openmvs-mesh-mvs-sha256", required=True)
    p.add_argument("--openmvs-fingerprint", type=Path, required=True)
    p.add_argument("--openmvs-fingerprint-sha256", required=True)
    p.add_argument("--openmvs-fingerprint-payload-sha256", required=True)
    p.add_argument("--openmvs-runtime-manifest-sha256", required=True)
    p.add_argument("--openmvs-adapter-sha256", required=True)
    p.add_argument("--openmvs-core-sha256", required=True)
    p.add_argument("--frame-scale-decision-json", type=Path, required=True)
    p.add_argument("--frame-scale-decision-sha256", required=True)
    p.add_argument("--layer2-runtime-source-manifest", type=Path, required=True)
    p.add_argument("--layer2-runtime-source-manifest-sha256", required=True)
    p.add_argument("--expected-scene", required=True)
    p.add_argument("--expected-target", required=True)
    p.add_argument("--expected-height", type=int, required=True)
    p.add_argument("--expected-width", type=int, required=True)
    p.add_argument("--negative-control-status", default=CONTROL_PROTOCOL_STATUS)
    p.add_argument("--preflight-only", action="store_true", help="Validate bindings and array schemas only; do not build masks, controls, metrics, or taxonomy.")
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    return p


def guard_output_paths(output_json: Path, output_csv: Path) -> None:
    existing = [str(p) for p in [output_json, output_csv] if p.exists()]
    if existing:
        raise ProtocolError("output_path_exists", "refusing to overwrite existing output path", {"existing": existing})
    if output_json.resolve() == output_csv.resolve():
        raise ProtocolError("output_path_exists", "output-json and output-csv must be different files", {"output": str(output_json)})


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        guard_output_paths(args.output_json, args.output_csv)
    except ProtocolError as exc:
        sys.stderr.write(json.dumps({"schema": SCHEMA, "input_status": "input_hash_mismatch", "inconclusive_reason": exc.code, **exc.details}, indent=2) + "\n")
        return 9
    payload = evaluate(args)
    write_json(args.output_json, payload)
    write_csv(args.output_csv, flatten_for_csv(payload))
    if payload.get("preflight_only"):
        return 0 if payload["input_status"] == "correspondence_pass" else 2
    if payload["input_status"] != "correspondence_pass":
        return 2
    if payload["support_status"] == "insufficient_common_support":
        return 3
    if payload.get("control_shared_support_status") == "insufficient_shared_control_support":
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
