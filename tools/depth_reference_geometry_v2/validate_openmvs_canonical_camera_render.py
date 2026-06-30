#!/usr/bin/env python3
"""Validate a canonical OpenMVS triangle render packet.

This validator checks only camera/raster/support semantics for a canonical
OpenMVS triangle-rendered proxy. It does not compute OpenMVS/UMGS/DA3
alignment metrics and does not make geometry-accuracy claims.
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
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.depth_reference_geometry_v2.render_openmvs_canonical_camera import (  # noqa: E402
    FINGERPRINT_ATOL,
    PIXEL_CENTER_CONVENTION,
    PRINCIPAL_POINT_POLICY,
    load_umgs_canonical_camera,
)

SCHEMA_VERSION = "openmvs_canonical_camera_render_validator_v1"
BARYCENTRIC_SUM_ATOL = 1e-4
BARYCENTRIC_COMPONENT_MIN = -1e-5


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
        f.write("\n")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def nested_get(obj: Any, keys: Sequence[str], default: Any = None) -> Any:
    cur = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def check(condition: bool, name: str, details: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    checks.append({"check": name, "status": "pass" if condition else "fail", **details})


def check_close_array(
    actual: Any,
    expected: Any,
    name: str,
    checks: list[dict[str, Any]],
    *,
    atol: float,
) -> None:
    act = np.asarray(actual, dtype=np.float64)
    exp = np.asarray(expected, dtype=np.float64)
    ok = act.shape == exp.shape and bool(np.allclose(act, exp, rtol=0.0, atol=atol, equal_nan=True))
    delta = float(np.nanmax(np.abs(act - exp))) if act.shape == exp.shape and act.size else math.inf
    check(ok, name, {"actual_shape": list(act.shape), "expected_shape": list(exp.shape), "max_abs_delta": delta, "atol": float(atol)}, checks)


def check_close_scalar(actual: Any, expected: Any, name: str, checks: list[dict[str, Any]], *, atol: float) -> None:
    try:
        act = float(actual)
        exp = float(expected)
        ok = bool(np.isclose(act, exp, rtol=0.0, atol=atol, equal_nan=True))
        delta = abs(act - exp)
    except Exception:
        act = actual
        exp = expected
        ok = False
        delta = math.inf
    check(ok, name, {"actual": act, "expected": exp, "abs_delta": delta, "atol": float(atol)}, checks)


def arrays_equal(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind == "f":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def validate_render_packet(args: argparse.Namespace) -> dict[str, Any]:
    render_dir = Path(args.render_dir)
    npz_path = Path(args.npz) if args.npz else render_dir / "openmvs_canonical_triangle_render.npz"
    manifest_path = Path(args.manifest) if args.manifest else render_dir / "openmvs_canonical_triangle_render_manifest.json"
    checks: list[dict[str, Any]] = []

    check(npz_path.exists(), "npz_exists", {"path": str(npz_path)}, checks)
    check(manifest_path.exists(), "manifest_exists", {"path": str(manifest_path)}, checks)
    if not npz_path.exists() or not manifest_path.exists():
        return summary_payload(args, checks, npz_path, manifest_path, None, None)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data = np.load(npz_path)
    required = ["depth", "valid", "triangle_id", "barycentric"]
    for key in required:
        check(key in data.files, f"npz_key_{key}", {"available_keys": sorted(data.files)}, checks)
    if any(key not in data.files for key in required):
        return summary_payload(args, checks, npz_path, manifest_path, manifest, None)

    depth = np.asarray(data["depth"])
    valid = np.asarray(data["valid"])
    triangle_id = np.asarray(data["triangle_id"])
    bary = np.asarray(data["barycentric"])
    expected_hw = (int(args.expected_height), int(args.expected_width))
    check(depth.shape == expected_hw, "depth_shape", {"actual": list(depth.shape), "expected": list(expected_hw)}, checks)
    check(valid.shape == expected_hw, "valid_shape", {"actual": list(valid.shape), "expected": list(expected_hw)}, checks)
    check(triangle_id.shape == expected_hw, "triangle_id_shape", {"actual": list(triangle_id.shape), "expected": list(expected_hw)}, checks)
    check(bary.shape == expected_hw + (3,), "barycentric_shape", {"actual": list(bary.shape), "expected": [*expected_hw, 3]}, checks)
    check(depth.dtype == np.dtype("float32"), "depth_dtype_float32", {"dtype": str(depth.dtype)}, checks)
    check(valid.dtype == np.dtype("uint8"), "valid_dtype_uint8", {"dtype": str(valid.dtype)}, checks)
    check(triangle_id.dtype == np.dtype("int32"), "triangle_id_dtype_int32", {"dtype": str(triangle_id.dtype)}, checks)
    check(bary.dtype == np.dtype("float32"), "barycentric_dtype_float32", {"dtype": str(bary.dtype)}, checks)
    check(valid.dtype.kind in "biu", "valid_dtype_integer_bool", {"dtype": str(valid.dtype)}, checks)
    valid_bool = valid.astype(bool)
    check(np.array_equal(valid, valid_bool.astype(valid.dtype)), "valid_binary", {"true_count": int(valid_bool.sum())}, checks)
    check(valid_bool.any(), "valid_nonempty", {"valid_pixel_count": int(valid_bool.sum())}, checks)

    recomputed_valid = np.isfinite(depth) & (depth > 0.0) & (triangle_id >= 0)
    check(
        np.array_equal(valid_bool, recomputed_valid),
        "valid_definition_triangle_hit_finite_positive_z",
        {"mismatch_count": int(np.count_nonzero(valid_bool != recomputed_valid))},
        checks,
    )
    check(bool(np.all(np.isfinite(depth[valid_bool]))), "valid_depth_finite", {}, checks)
    check(bool(np.all(depth[valid_bool] > 0.0)), "valid_depth_positive", {}, checks)
    invalid = ~valid_bool
    check(bool(np.all(np.isnan(depth[invalid]))), "invalid_depth_nan", {"invalid_count": int(invalid.sum())}, checks)
    check(bool(np.all(triangle_id[invalid] == -1)), "invalid_triangle_id_minus_one", {}, checks)
    check(bool(np.all(np.isnan(bary[invalid]))), "invalid_barycentric_nan", {}, checks)

    face_count = int(nested_get(manifest, ["mesh", "geometry_summary", "face_count"], -1))
    if face_count >= 0 and valid_bool.any():
        valid_ids = triangle_id[valid_bool]
        check(int(valid_ids.min()) >= 0, "valid_triangle_id_min_nonnegative", {"min": int(valid_ids.min())}, checks)
        check(int(valid_ids.max()) < face_count, "valid_triangle_id_less_than_face_count", {"max": int(valid_ids.max()), "face_count": face_count}, checks)
    else:
        check(False, "face_count_available_for_triangle_id_check", {"face_count": face_count}, checks)

    valid_bary = bary[valid_bool]
    if valid_bary.size:
        bary_sum = valid_bary.sum(axis=1)
        check(bool(np.all(np.isfinite(valid_bary))), "valid_barycentric_finite", {}, checks)
        check(
            bool(np.allclose(bary_sum, 1.0, rtol=0.0, atol=BARYCENTRIC_SUM_ATOL)),
            "valid_barycentric_sum_one",
            {"max_abs_delta": float(np.max(np.abs(bary_sum - 1.0))), "atol": BARYCENTRIC_SUM_ATOL},
            checks,
        )
        check(
            bool(np.nanmin(valid_bary) >= BARYCENTRIC_COMPONENT_MIN),
            "valid_barycentric_components_not_below_tolerance",
            {"min_component": float(np.nanmin(valid_bary)), "tolerance": BARYCENTRIC_COMPONENT_MIN},
            checks,
        )

    check(str(manifest.get("schema", "")) == "openmvs_canonical_camera_triangle_render_v1", "manifest_schema", {"actual": manifest.get("schema")}, checks)
    check(manifest.get("no_proxy_metric") is True, "manifest_no_proxy_metric_true", {"actual": manifest.get("no_proxy_metric")}, checks)
    check(manifest.get("not_ground_truth") is True, "manifest_not_ground_truth_true", {"actual": manifest.get("not_ground_truth")}, checks)
    check(str(manifest.get("valid_definition", "")) == "triangle_hit AND finite_camera_z AND camera_z_gt_0", "manifest_valid_definition", {"actual": manifest.get("valid_definition")}, checks)
    check(str(manifest.get("pixel_center_convention", "")) == PIXEL_CENTER_CONVENTION, "manifest_pixel_center_convention", {"actual": manifest.get("pixel_center_convention"), "expected": PIXEL_CENTER_CONVENTION}, checks)
    check(str(manifest.get("principal_point_policy", "")) == PRINCIPAL_POINT_POLICY, "manifest_principal_point_policy", {"actual": manifest.get("principal_point_policy"), "expected": PRINCIPAL_POINT_POLICY}, checks)
    check(str(manifest.get("barycentric_semantics", "")) == "screen_space_affine_weights_at_pixel_center", "manifest_barycentric_semantics", {"actual": manifest.get("barycentric_semantics")}, checks)
    check(manifest.get("barycentric_not_perspective_correct_surface_weights") is True, "manifest_barycentric_not_perspective_correct", {"actual": manifest.get("barycentric_not_perspective_correct_surface_weights")}, checks)
    check(str(manifest.get("triangle_clipping_policy", "")) == "reject triangle if any vertex camera-z <= 1e-8; no near-plane clipping", "manifest_triangle_clipping_policy", {"actual": manifest.get("triangle_clipping_policy")}, checks)

    check(str(manifest.get("scene", "")) == args.expected_scene, "manifest_scene", {"actual": manifest.get("scene"), "expected": args.expected_scene}, checks)
    check(
        Path(str(manifest.get("target", ""))).name == Path(args.expected_target).name,
        "manifest_target",
        {"actual": manifest.get("target"), "expected": args.expected_target},
        checks,
    )
    for key, expected in [
        ("width", int(args.expected_width)),
        ("height", int(args.expected_height)),
    ]:
        actual = nested_get(manifest, ["camera", key])
        check(int(actual) == expected if actual is not None else False, f"manifest_camera_{key}", {"actual": actual, "expected": expected}, checks)
    camera = None
    if args.fingerprint:
        try:
            camera = load_umgs_canonical_camera(
                args.fingerprint,
                expected_file_sha256=args.expected_fingerprint_sha256,
                expected_payload_sha256=args.expected_fingerprint_payload_sha256,
                expected_target=args.expected_target,
                expected_width=args.expected_width,
                expected_height=args.expected_height,
                atol=float(args.camera_atol),
            )
            check(True, "fingerprint_reloaded_and_self_hash_verified", {"path": args.fingerprint}, checks)
        except Exception as exc:
            check(False, "fingerprint_reloaded_and_self_hash_verified", {"path": args.fingerprint, "error": str(exc)}, checks)
    if camera is not None:
        cam = manifest.get("camera", {})
        check(int(cam.get("image_id", -999)) == int(camera.view.image_id), "manifest_camera_image_id", {"actual": cam.get("image_id"), "expected": camera.view.image_id}, checks)
        check(Path(str(cam.get("image_name", ""))).name == Path(camera.view.image_name).name, "manifest_camera_image_name", {"actual": cam.get("image_name"), "expected": camera.view.image_name}, checks)
        for key, expected in [
            ("fx", camera.fx),
            ("fy", camera.fy),
            ("cx", camera.cx),
            ("cy", camera.cy),
            ("FoVx", camera.fovx),
            ("FoVy", camera.fovy),
            ("znear", camera.znear),
            ("zfar", camera.zfar),
        ]:
            check_close_scalar(cam.get(key), expected, f"manifest_camera_{key}", checks, atol=float(args.camera_atol))
        fp_payload = camera.payload
        has_r = "R" in cam
        has_t = "T" in cam
        check(has_r, "manifest_camera_R_present", {"present": has_r}, checks)
        check(has_t, "manifest_camera_T_present", {"present": has_t}, checks)
        if has_r:
            check_close_array(cam["R"], fp_payload["R"], "manifest_camera_R", checks, atol=float(args.camera_atol))
        else:
            check(False, "manifest_camera_R", {"reason": "missing_field"}, checks)
        if has_t:
            check_close_array(cam["T"], fp_payload["T"], "manifest_camera_T", checks, atol=float(args.camera_atol))
        else:
            check(False, "manifest_camera_T", {"reason": "missing_field"}, checks)
        check_close_array(cam.get("world_view_transform"), camera.world_view_row, "manifest_camera_world_view_transform", checks, atol=float(args.camera_atol))
        check_close_array(cam.get("camera_to_world"), camera.camera_to_world, "manifest_camera_camera_to_world", checks, atol=float(args.camera_atol))
        check_close_array(cam.get("projection_matrix"), camera.projection_row, "manifest_camera_projection_matrix", checks, atol=float(args.camera_atol))
        check_close_array(cam.get("full_proj_transform"), camera.full_projection_row, "manifest_camera_full_proj_transform", checks, atol=float(args.camera_atol))
    for field, expected in [
        ("fingerprint.file_sha256", args.expected_fingerprint_sha256),
        ("fingerprint.payload_sha256", args.expected_fingerprint_payload_sha256),
        ("mesh.ply_sha256", args.expected_mesh_ply_sha256),
        ("mesh.mvs_sha256", args.expected_mesh_mvs_sha256),
    ]:
        keys = field.split(".")
        actual = nested_get(manifest, keys, "")
        check(str(actual).lower() == str(expected).lower(), f"manifest_{field.replace('.', '_')}", {"actual": actual, "expected": expected}, checks)
    if args.expected_runtime_source_manifest_sha256:
        actual = nested_get(manifest, ["runtime_source_manifest", "sha256"], "")
        check(str(actual).lower() == str(args.expected_runtime_source_manifest_sha256).lower(), "manifest_runtime_source_manifest_sha256", {"actual": actual, "expected": args.expected_runtime_source_manifest_sha256}, checks)
    if args.expected_core_sha256:
        actual = ""
        for row in nested_get(manifest, ["runtime_source_manifest", "source_files"], []):
            if row.get("path") == "tools/depth_reference_geometry_v2/openmvs_campaign_core.py":
                actual = str(row.get("sha256", ""))
        check(str(actual).lower() == str(args.expected_core_sha256).lower(), "manifest_runtime_core_sha256", {"actual": actual, "expected": args.expected_core_sha256}, checks)

    if args.expected_adapter_sha256:
        actual = ""
        for row in nested_get(manifest, ["runtime_source_manifest", "source_files"], []):
            if row.get("path") == "tools/depth_reference_geometry_v2/render_openmvs_canonical_camera.py":
                actual = str(row.get("sha256", ""))
        check(actual.lower() == args.expected_adapter_sha256.lower(), "adapter_source_sha256", {"actual": actual, "expected": args.expected_adapter_sha256}, checks)
    loaded_meta_v = int(nested_get(manifest, ["mesh", "loaded_meta", "vertex_count"], -1))
    loaded_meta_f = int(nested_get(manifest, ["mesh", "loaded_meta", "face_count"], -1))
    geom_v = int(nested_get(manifest, ["mesh", "geometry_summary", "vertex_count"], -2))
    geom_f = int(nested_get(manifest, ["mesh", "geometry_summary", "face_count"], -2))
    if args.expected_vertex_count >= 0:
        check(loaded_meta_v == args.expected_vertex_count, "mesh_loaded_vertex_count", {"actual": loaded_meta_v, "expected": args.expected_vertex_count}, checks)
        check(geom_v == args.expected_vertex_count, "mesh_geometry_vertex_count", {"actual": geom_v, "expected": args.expected_vertex_count}, checks)
    if args.expected_face_count >= 0:
        check(loaded_meta_f == args.expected_face_count, "mesh_loaded_face_count", {"actual": loaded_meta_f, "expected": args.expected_face_count}, checks)
        check(geom_f == args.expected_face_count, "mesh_geometry_face_count", {"actual": geom_f, "expected": args.expected_face_count}, checks)
    check(loaded_meta_v == geom_v, "mesh_loaded_vs_geometry_vertex_count_consistent", {"loaded_meta": loaded_meta_v, "geometry_summary": geom_v}, checks)
    check(loaded_meta_f == geom_f, "mesh_loaded_vs_geometry_face_count_consistent", {"loaded_meta": loaded_meta_f, "geometry_summary": geom_f}, checks)

    manifest_h = nested_get(manifest, ["output", "height"])
    manifest_w = nested_get(manifest, ["output", "width"])
    manifest_valid_count = nested_get(manifest, ["output", "valid_pixel_count"])
    manifest_valid_ratio = nested_get(manifest, ["output", "valid_ratio"])
    check(int(manifest_h) == int(depth.shape[0]) if manifest_h is not None else False, "manifest_output_height_matches_array", {"actual": manifest_h, "array": int(depth.shape[0])}, checks)
    check(int(manifest_w) == int(depth.shape[1]) if manifest_w is not None else False, "manifest_output_width_matches_array", {"actual": manifest_w, "array": int(depth.shape[1])}, checks)
    check(int(manifest_valid_count) == int(valid_bool.sum()) if manifest_valid_count is not None else False, "manifest_output_valid_count_matches_array", {"actual": manifest_valid_count, "array": int(valid_bool.sum())}, checks)
    if manifest_valid_ratio is not None:
        check_close_scalar(manifest_valid_ratio, float(valid_bool.mean()), "manifest_output_valid_ratio_matches_array", checks, atol=1e-12)

    npz_sha = sha256_file(npz_path)
    check(str(nested_get(manifest, ["output", "npz_sha256"], "")).lower() == npz_sha.lower(), "manifest_npz_sha256", {"actual": npz_sha}, checks)

    if args.repeat_npz:
        repeat = np.load(args.repeat_npz)
        for key, arr in [("depth", depth), ("valid", valid), ("triangle_id", triangle_id), ("barycentric", bary)]:
            check(key in repeat.files, f"repeat_key_{key}", {"repeat_npz": args.repeat_npz}, checks)
            if key in repeat.files:
                check(arrays_equal(arr, np.asarray(repeat[key])), f"repeat_exact_array_equal_{key}", {}, checks)

    return summary_payload(args, checks, npz_path, manifest_path, manifest, {"depth": depth, "valid": valid_bool, "triangle_id": triangle_id, "barycentric": bary})


def summary_payload(
    args: argparse.Namespace,
    checks: list[dict[str, Any]],
    npz_path: Path,
    manifest_path: Path,
    manifest: dict[str, Any] | None,
    arrays: dict[str, np.ndarray] | None,
) -> dict[str, Any]:
    failed = [row for row in checks if row["status"] != "pass"]
    output = {
        "schema": SCHEMA_VERSION,
        "status": "pass" if not failed else "fail",
        "failed_checks": failed,
        "check_count": len(checks),
        "npz_path": str(npz_path),
        "npz_sha256": sha256_file(npz_path) if npz_path.exists() else "",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.exists() else "",
        "expected_scene": args.expected_scene,
        "expected_target": args.expected_target,
        "expected_height": int(args.expected_height),
        "expected_width": int(args.expected_width),
        "valid_semantics": "triangle_hit AND finite_camera_z AND camera_z_gt_0",
        "not_proxy_metric": True,
        "not_ground_truth": True,
        "checks": checks,
        "environment": {
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    if arrays is not None:
        valid = arrays["valid"]
        depth = arrays["depth"]
        output["array_summary"] = {
            "height": int(depth.shape[0]),
            "width": int(depth.shape[1]),
            "valid_pixel_count": int(valid.sum()),
            "valid_ratio": float(valid.mean()),
            "depth_min": float(np.nanmin(depth)) if np.isfinite(depth).any() else math.nan,
            "depth_max": float(np.nanmax(depth)) if np.isfinite(depth).any() else math.nan,
        }
    if manifest:
        output["render_manifest_schema"] = manifest.get("schema", "")
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-dir", required=True)
    parser.add_argument("--npz", default="")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--expected-scene", required=True)
    parser.add_argument("--expected-target", required=True)
    parser.add_argument("--expected-height", required=True, type=int)
    parser.add_argument("--expected-width", required=True, type=int)
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--camera-atol", type=float, default=FINGERPRINT_ATOL)
    parser.add_argument("--expected-fingerprint-sha256", required=True)
    parser.add_argument("--expected-fingerprint-payload-sha256", required=True)
    parser.add_argument("--expected-mesh-ply-sha256", required=True)
    parser.add_argument("--expected-mesh-mvs-sha256", required=True)
    parser.add_argument("--expected-adapter-sha256", default="")
    parser.add_argument("--expected-runtime-source-manifest-sha256", default="")
    parser.add_argument("--expected-core-sha256", default="")
    parser.add_argument("--expected-vertex-count", type=int, default=-1)
    parser.add_argument("--expected-face-count", type=int, default=-1)
    parser.add_argument("--repeat-npz", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = validate_render_packet(args)
    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv)
    write_json(out_json, summary)
    rows = summary["checks"]
    write_csv(out_csv, rows)
    print(json.dumps({"status": summary["status"], "failed_check_count": len(summary["failed_checks"])}, indent=2))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
