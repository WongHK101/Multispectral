#!/usr/bin/env python3
"""Synthetic tests for canonical OpenMVS camera adapter and validator."""

from __future__ import annotations

import argparse
import ast
import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import numpy as np

from render_openmvs_canonical_camera import (
    BARYCENTRIC_ATOL,
    FINGERPRINT_ATOL,
    PIXEL_CENTER_CONVENTION,
    PRINCIPAL_POINT_POLICY,
    PROJECTION_ATOL,
    canonical_json_sha256,
    camera_space_from_fingerprint_row,
    load_umgs_canonical_camera,
    project_points_with_camera_view,
    project_points_with_fingerprint_full_proj,
    projection_matrix_from_fov_row,
    sha256_file,
    verify_runtime_source_manifest,
    write_json,
)
from tools.depth_reference_geometry_v2.openmvs_campaign_core import rasterize_mesh_camera_z
from validate_openmvs_canonical_camera_render import validate_render_packet


def make_fingerprint(
    path: Path,
    *,
    width: int = 100,
    height: int = 80,
    fovx: float = math.radians(60.0),
    fovy: float = math.radians(45.0),
    world_view_row: np.ndarray | None = None,
    image_name: str = "synthetic_target.JPG",
) -> dict[str, Any]:
    if world_view_row is None:
        world_view_row = np.eye(4, dtype=np.float64)
    projection = projection_matrix_from_fov_row(fovx, fovy, 0.01, 100.0)
    full = world_view_row @ projection
    payload = {
        "image_name": image_name,
        "colmap_id": 1,
        "uid": 0,
        "image_width": width,
        "image_height": height,
        "FoVx": fovx,
        "FoVy": fovy,
        "znear": 0.01,
        "zfar": 100.0,
        "R": world_view_row[:3, :3].tolist(),
        "T": world_view_row[3, :3].tolist(),
        "world_view_transform": world_view_row.tolist(),
        "projection_matrix": projection.tolist(),
        "full_proj_transform": full.tolist(),
    }
    data = {"fingerprint_sha256": canonical_json_sha256(payload), "payload": payload}
    write_json(path, data)
    return data


def camera_to_world_points(points_camera: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    c2w = np.asarray(camera_to_world, dtype=np.float64)
    pts = np.asarray(points_camera, dtype=np.float64)
    return pts @ c2w[:3, :3].T + c2w[:3, 3]


def assert_close(a: np.ndarray, b: np.ndarray, *, atol: float, label: str) -> None:
    if not np.allclose(a, b, rtol=0.0, atol=atol, equal_nan=True):
        raise AssertionError(f"{label} mismatch: max_abs={np.nanmax(np.abs(a-b))}, a={a}, b={b}")


def test_centered_synthetic_camera() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "fingerprint.json"
        make_fingerprint(fp, width=100, height=80)
        record = load_umgs_canonical_camera(fp, expected_width=100, expected_height=80)
        assert record.cx == 50.0
        assert record.cy == 40.0
        center = np.array([[0.0, 0.0, 5.0]], dtype=np.float64)
        xy = project_points_with_camera_view(center, record.view)
        assert_close(xy, np.array([[50.0, 40.0]]), atol=1e-10, label="center projection")
        return {
            "cx": record.cx,
            "cy": record.cy,
            "pixel_center_convention": PIXEL_CENTER_CONVENTION,
            "center_projection": xy[0].tolist(),
        }


def test_off_axis_projection_matches_full_projection() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "fingerprint.json"
        make_fingerprint(fp, width=120, height=90)
        record = load_umgs_canonical_camera(fp, expected_width=120, expected_height=90)
        points = np.array(
            [
                [0.4, 0.0, 4.0],
                [-0.4, 0.0, 4.0],
                [0.0, 0.3, 4.0],
                [0.0, -0.3, 4.0],
                [0.2, -0.15, 6.0],
            ],
            dtype=np.float64,
        )
        xy_view = project_points_with_camera_view(points, record.view)
        xy_full = project_points_with_fingerprint_full_proj(points, record.payload)
        assert_close(xy_view, xy_full, atol=5e-5, label="off-axis projection")
        return {"max_abs_delta": float(np.max(np.abs(xy_view - xy_full))), "point_count": int(points.shape[0])}


def test_orientation_and_sign() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "fingerprint.json"
        make_fingerprint(fp, width=100, height=80)
        record = load_umgs_canonical_camera(fp, expected_width=100, expected_height=80)
        points = np.array(
            [
                [0.0, 0.0, 5.0],
                [0.0, 0.0, -5.0],
                [0.4, 0.0, 5.0],
                [-0.4, 0.0, 5.0],
                [0.0, 0.4, 5.0],
                [0.0, -0.4, 5.0],
            ],
            dtype=np.float64,
        )
        cam = camera_space_from_fingerprint_row(points, record.world_view_row)
        xy = project_points_with_camera_view(points, record.view)
        assert cam[0, 2] > 0
        assert cam[1, 2] < 0
        assert xy[2, 0] > record.cx
        assert xy[3, 0] < record.cx
        assert xy[4, 1] > record.cy
        assert xy[5, 1] < record.cy
        return {
            "front_z": float(cam[0, 2]),
            "behind_z": float(cam[1, 2]),
            "positive_x_pixel": float(xy[2, 0]),
            "negative_x_pixel": float(xy[3, 0]),
            "positive_y_pixel": float(xy[4, 1]),
            "negative_y_pixel": float(xy[5, 1]),
        }


def test_synthetic_triangle_raster() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "fingerprint.json"
        make_fingerprint(fp, width=80, height=60)
        record = load_umgs_canonical_camera(fp, expected_width=80, expected_height=60)
        vertices = np.array(
            [
                [-0.8, -0.6, 5.0],
                [0.8, -0.6, 5.0],
                [0.0, 0.8, 5.0],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        result = rasterize_mesh_camera_z(vertices, faces, record.view)
        valid = result.valid & (result.depth > 0)
        assert valid.any()
        assert np.allclose(result.depth[valid], 5.0, atol=1e-12)
        assert set(np.unique(result.triangle_id[valid]).tolist()) == {0}
        bary = result.barycentric[valid]
        assert np.allclose(bary.sum(axis=1), 1.0, atol=BARYCENTRIC_ATOL)
        assert np.nanmin(bary) >= -BARYCENTRIC_ATOL
        invalid = ~valid
        assert np.all(result.triangle_id[invalid] == -1)
        assert np.all(np.isnan(result.barycentric[invalid]))
        return {"valid_pixel_count": int(valid.sum()), "median_depth": float(np.nanmedian(result.depth[valid]))}


def test_principal_point_policy_changes_raster_hit_pattern() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "fingerprint.json"
        # Even width and odd height match Road parity: 1200 x 869.
        make_fingerprint(fp, width=120, height=87)
        record = load_umgs_canonical_camera(fp, expected_width=120, expected_height=87)

        z = 10.0
        pixel_triangle = np.array(
            [
                [record.cx + 0.0, record.cy - 0.5],
                [record.cx + 1.0, record.cy - 0.5],
                [record.cx + 0.5, record.cy + 0.5],
            ],
            dtype=np.float64,
        )
        vertices = np.column_stack(
            [
                (pixel_triangle[:, 0] - record.cx) * z / record.fx,
                (pixel_triangle[:, 1] - record.cy) * z / record.fy,
                np.full(3, z, dtype=np.float64),
            ]
        )
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        correct = rasterize_mesh_camera_z(vertices, faces, record.view)
        wrong_view = record.view.__class__(
            image_id=record.view.image_id,
            image_name=record.view.image_name,
            width=record.view.width,
            height=record.view.height,
            fx=record.view.fx,
            fy=record.view.fy,
            cx=(record.view.width - 1.0) / 2.0,
            cy=(record.view.height - 1.0) / 2.0,
            camera_to_world=record.view.camera_to_world,
        )
        wrong = rasterize_mesh_camera_z(vertices, faces, wrong_view)
        correct_hits = set(map(tuple, np.argwhere(correct.valid).tolist()))
        wrong_hits = set(map(tuple, np.argwhere(wrong.valid).tolist()))
        assert correct_hits
        assert wrong_hits
        assert correct_hits != wrong_hits
        expected_correct_hits = {(43, 60)}
        expected_wrong_hits = {(42, 59), (42, 60)}
        assert correct_hits == expected_correct_hits, correct_hits
        assert wrong_hits == expected_wrong_hits, wrong_hits
        return {
            "correct_hit_count": len(correct_hits),
            "wrong_hit_count": len(wrong_hits),
            "correct_hits": sorted([list(x) for x in correct_hits]),
            "wrong_hits": sorted([list(x) for x in wrong_hits]),
            "expected_correct_hits_yx": sorted([list(x) for x in expected_correct_hits]),
            "expected_wrong_hits_yx": sorted([list(x) for x in expected_wrong_hits]),
        }


def test_old_path_rejection() -> dict[str, Any]:
    source = Path(__file__).with_name("render_openmvs_canonical_camera.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                calls.append(func.id)
            elif isinstance(func, ast.Attribute):
                calls.append(func.attr)
    assert "view_from_colmap" not in calls
    return {"view_from_colmap_call_count": calls.count("view_from_colmap")}


def write_runtime_manifest(root: Path, rows: list[dict[str, str]]) -> Path:
    path = root / "runtime_source_manifest.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=["path", "sha256", "role"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def make_validator_case(root: Path) -> tuple[SimpleNamespace, dict[str, Any], dict[str, Path]]:
    render_dir = root / "render"
    render_dir.mkdir()
    fp = root / "fingerprint.json"
    fp_data = make_fingerprint(fp, width=4, height=4, image_name="target.JPG")
    record = load_umgs_canonical_camera(fp, expected_width=4, expected_height=4)
    depth = np.array(
        [
            [np.nan, 5.0, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan],
        ],
        dtype=np.float32,
    )
    valid = np.array(
        [
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    triangle_id = np.full((4, 4), -1, dtype=np.int32)
    triangle_id[0, 1] = 0
    bary = np.full((4, 4, 3), np.nan, dtype=np.float32)
    bary[0, 1] = np.array([0.2, 0.3, 0.5], dtype=np.float32)
    npz = render_dir / "openmvs_canonical_triangle_render.npz"
    np.savez_compressed(npz, depth=depth, valid=valid, triangle_id=triangle_id, barycentric=bary)
    adapter_sha = sha256_file(Path(__file__).with_name("render_openmvs_canonical_camera.py"))
    core_sha = sha256_file(Path(__file__).with_name("openmvs_campaign_core.py"))
    runtime_rows = [
        {"path": "tools/depth_reference_geometry_v2/render_openmvs_canonical_camera.py", "sha256": adapter_sha, "role": "adapter"},
        {"path": "tools/depth_reference_geometry_v2/validate_openmvs_canonical_camera_render.py", "sha256": sha256_file(Path(__file__).with_name("validate_openmvs_canonical_camera_render.py")), "role": "validator"},
        {"path": "tools/depth_reference_geometry_v2/openmvs_campaign_core.py", "sha256": core_sha, "role": "core"},
    ]
    runtime_manifest = write_runtime_manifest(root, runtime_rows)
    mesh_meta = {"vertex_count": 7, "face_count": 1, "sha256": "m" * 64}
    mesh_geom = {"vertex_count": 7, "face_count": 1, "finite_vertex_count": 7}
    manifest = {
        "schema": "openmvs_canonical_camera_triangle_render_v1",
        "scene": "scene",
        "target": "target.JPG",
        "no_proxy_metric": True,
        "not_ground_truth": True,
        "valid_definition": "triangle_hit AND finite_camera_z AND camera_z_gt_0",
        "invalid_convention": {"depth": "NaN", "triangle_id": -1, "barycentric": "NaN"},
        "barycentric_semantics": "screen_space_affine_weights_at_pixel_center",
        "barycentric_not_perspective_correct_surface_weights": True,
        "triangle_clipping_policy": "reject triangle if any vertex camera-z <= 1e-8; no near-plane clipping",
        "pixel_center_convention": PIXEL_CENTER_CONVENTION,
        "principal_point_policy": PRINCIPAL_POINT_POLICY,
        "camera": {
            "image_id": record.view.image_id,
            "image_name": record.view.image_name,
            "width": record.width,
            "height": record.height,
            "fx": record.fx,
            "fy": record.fy,
            "cx": record.cx,
            "cy": record.cy,
            "FoVx": record.fovx,
            "FoVy": record.fovy,
            "znear": record.znear,
            "zfar": record.zfar,
            "R": record.payload["R"],
            "T": record.payload["T"],
            "world_view_transform": record.world_view_row.tolist(),
            "camera_to_world": record.camera_to_world.tolist(),
            "projection_matrix": record.projection_row.tolist(),
            "full_proj_transform": record.full_projection_row.tolist(),
        },
        "fingerprint": {"file_sha256": sha256_file(fp), "payload_sha256": fp_data["fingerprint_sha256"]},
        "mesh": {
            "ply_sha256": "m" * 64,
            "mvs_sha256": "v" * 64,
            "loaded_meta": mesh_meta,
            "geometry_summary": mesh_geom,
        },
        "output": {
            "npz": str(npz),
            "npz_sha256": sha256_file(npz),
            "height": 4,
            "width": 4,
            "valid_pixel_count": 1,
            "valid_ratio": 1.0 / 16.0,
        },
        "runtime_source_manifest": {"path": str(runtime_manifest), "sha256": sha256_file(runtime_manifest), "source_files": runtime_rows},
    }
    manifest_path = render_dir / "openmvs_canonical_triangle_render_manifest.json"
    write_json(manifest_path, manifest)
    args = SimpleNamespace(
        render_dir=str(render_dir),
        npz="",
        manifest="",
        expected_scene="scene",
        expected_target="target.JPG",
        expected_height=4,
        expected_width=4,
        fingerprint=str(fp),
        camera_atol=FINGERPRINT_ATOL,
        expected_fingerprint_sha256=sha256_file(fp),
        expected_fingerprint_payload_sha256=fp_data["fingerprint_sha256"],
        expected_mesh_ply_sha256="m" * 64,
        expected_mesh_mvs_sha256="v" * 64,
        expected_adapter_sha256=adapter_sha,
        expected_runtime_source_manifest_sha256=sha256_file(runtime_manifest),
        expected_core_sha256=core_sha,
        expected_vertex_count=7,
        expected_face_count=1,
        repeat_npz="",
    )
    return args, manifest, {"npz": npz, "manifest": manifest_path, "fingerprint": fp, "runtime_manifest": runtime_manifest}


def run_validator_case_with_mutation(mutate: Callable[[SimpleNamespace, dict[str, Any], dict[str, Path], Path], None]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        args, manifest, paths = make_validator_case(root)
        mutate(args, manifest, paths, root)
        if paths["manifest"].exists():
            write_json(paths["manifest"], manifest)
        return validate_render_packet(args)


def test_validator_semantics_and_negative_cases() -> dict[str, Any]:
    results: dict[str, str] = {}
    with tempfile.TemporaryDirectory() as td:
        args, manifest, paths = make_validator_case(Path(td))
        summary = validate_render_packet(args)
        assert summary["status"] == "pass", summary["failed_checks"]
        results["valid_packet_passes"] = summary["status"]

    negative_cases: list[tuple[str, Callable[[SimpleNamespace, dict[str, Any], dict[str, Path], Path], None], str]] = [
        (
            "wrong_runtime_source_manifest_sha_fails",
            lambda args, manifest, paths, root: setattr(args, "expected_runtime_source_manifest_sha256", "0" * 64),
            "manifest_runtime_source_manifest_sha256",
        ),
        (
            "wrong_core_sha_fails",
            lambda args, manifest, paths, root: setattr(args, "expected_core_sha256", "0" * 64),
            "manifest_runtime_core_sha256",
        ),
        (
            "wrong_fx_fails",
            lambda args, manifest, paths, root: manifest["camera"].__setitem__("fx", manifest["camera"]["fx"] + 1.0),
            "manifest_camera_fx",
        ),
        (
            "missing_R_fails",
            lambda args, manifest, paths, root: manifest["camera"].pop("R"),
            "manifest_camera_R_present",
        ),
        (
            "missing_T_fails",
            lambda args, manifest, paths, root: manifest["camera"].pop("T"),
            "manifest_camera_T_present",
        ),
        (
            "wrong_R_fails",
            lambda args, manifest, paths, root: manifest["camera"]["R"][0].__setitem__(0, manifest["camera"]["R"][0][0] + 1.0),
            "manifest_camera_R",
        ),
        (
            "wrong_T_fails",
            lambda args, manifest, paths, root: manifest["camera"]["T"].__setitem__(0, manifest["camera"]["T"][0] + 1.0),
            "manifest_camera_T",
        ),
        (
            "wrong_world_view_fails",
            lambda args, manifest, paths, root: manifest["camera"]["world_view_transform"][0].__setitem__(0, manifest["camera"]["world_view_transform"][0][0] + 1.0),
            "manifest_camera_world_view_transform",
        ),
        (
            "wrong_camera_to_world_fails",
            lambda args, manifest, paths, root: manifest["camera"]["camera_to_world"][0].__setitem__(0, manifest["camera"]["camera_to_world"][0][0] + 1.0),
            "manifest_camera_camera_to_world",
        ),
        (
            "wrong_manifest_schema_fails",
            lambda args, manifest, paths, root: manifest.__setitem__("schema", "wrong_schema"),
            "manifest_schema",
        ),
        (
            "wrong_array_dtype_fails",
            lambda args, manifest, paths, root: np.savez_compressed(paths["npz"], depth=np.zeros((4, 4), dtype=np.float64), valid=np.zeros((4, 4), dtype=np.uint8), triangle_id=np.full((4, 4), -1, dtype=np.int32), barycentric=np.full((4, 4, 3), np.nan, dtype=np.float32)),
            "depth_dtype_float32",
        ),
        (
            "wrong_vertex_count_fails",
            lambda args, manifest, paths, root: manifest["mesh"]["loaded_meta"].__setitem__("vertex_count", 8),
            "mesh_loaded_vertex_count",
        ),
        (
            "repeat_array_mismatch_fails",
            lambda args, manifest, paths, root: (np.savez_compressed(root / "repeat_bad.npz", depth=np.full((4, 4), np.nan, dtype=np.float32), valid=np.zeros((4, 4), dtype=np.uint8), triangle_id=np.full((4, 4), -1, dtype=np.int32), barycentric=np.full((4, 4, 3), np.nan, dtype=np.float32)), setattr(args, "repeat_npz", str(root / "repeat_bad.npz"))),
            "repeat_exact_array_equal_depth",
        ),
    ]
    for name, mutate, expected_check in negative_cases:
        summary = run_validator_case_with_mutation(mutate)
        assert summary["status"] == "fail", name
        assert any(row["check"] == expected_check for row in summary["failed_checks"]), (name, summary["failed_checks"])
        results[name] = "fail_detected"

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        source = root / "dummy.py"
        source.write_text("x = 1\n", encoding="utf-8")
        rel = source.relative_to(root)
        manifest = write_runtime_manifest(root, [{"path": str(rel).replace("\\", "/"), "sha256": sha256_file(source), "role": "dummy"}])
        try:
            verify_runtime_source_manifest(manifest, "0" * 64, repo_root=root)
            raise AssertionError("wrong manifest SHA unexpectedly passed")
        except ValueError:
            results["wrong_runtime_manifest_file_sha_fails"] = "fail_detected"
    return results


def test_actual_road_fingerprint(path: Path) -> dict[str, Any]:
    record = load_umgs_canonical_camera(path, expected_target="DJI_20260602165038_0001_D.JPG", expected_width=1200, expected_height=869)
    cam_points = np.array(
        [
            [0.0, 0.0, 10.0],
            [0.7, 0.0, 10.0],
            [-0.7, 0.0, 10.0],
            [0.0, 0.7, 10.0],
            [0.0, -0.7, 10.0],
            [0.4, -0.3, 20.0],
        ],
        dtype=np.float64,
    )
    world_points = camera_to_world_points(cam_points, record.camera_to_world)
    cam_from_payload = camera_space_from_fingerprint_row(world_points, record.world_view_row)
    xy_view = project_points_with_camera_view(world_points, record.view)
    xy_full = project_points_with_fingerprint_full_proj(world_points, record.payload)
    assert_close(cam_from_payload, cam_points, atol=5e-5, label="actual Road camera-space")
    assert_close(xy_view, xy_full, atol=PROJECTION_ATOL, label="actual Road projection")
    return {
        "fingerprint_file": str(path),
        "fingerprint_sha256": sha256_file(path),
        "payload_sha256": record.fingerprint_payload_sha256,
        "width": record.width,
        "height": record.height,
        "fx": record.fx,
        "fy": record.fy,
        "cx": record.cx,
        "cy": record.cy,
        "camera_space_max_abs_delta": float(np.max(np.abs(cam_from_payload - cam_points))),
        "projection_max_abs_delta": float(np.max(np.abs(xy_view - xy_full))),
    }


def run_test(name: str, func: Callable[[], dict[str, Any]], results: list[dict[str, Any]]) -> None:
    payload = func()
    row = {"test": name, "status": "pass", **payload}
    results.append(row)
    print(f"PASS {name}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--road-fingerprint", default="")
    parser.add_argument("--summary-json", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results: list[dict[str, Any]] = []
    run_test("centered_synthetic_camera", test_centered_synthetic_camera, results)
    run_test("off_axis_projection_matches_full_projection", test_off_axis_projection_matches_full_projection, results)
    run_test("orientation_and_sign", test_orientation_and_sign, results)
    run_test("synthetic_triangle_raster", test_synthetic_triangle_raster, results)
    run_test("principal_point_policy_changes_raster_hit_pattern", test_principal_point_policy_changes_raster_hit_pattern, results)
    run_test("old_path_rejection", test_old_path_rejection, results)
    run_test("validator_semantics_and_negative_cases", test_validator_semantics_and_negative_cases, results)
    if args.road_fingerprint:
        road_path = Path(args.road_fingerprint)
        run_test("actual_road_fingerprint_projection", lambda: test_actual_road_fingerprint(road_path), results)
    else:
        results.append({"test": "actual_road_fingerprint_projection", "status": "skipped", "reason": "no --road-fingerprint provided"})
        print("SKIP actual_road_fingerprint_projection")
    if args.summary_json:
        write_json(Path(args.summary_json), {"status": "pass", "tests": results})
    print("ALL_OPENMVS_CANONICAL_CAMERA_ADAPTER_TESTS_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
