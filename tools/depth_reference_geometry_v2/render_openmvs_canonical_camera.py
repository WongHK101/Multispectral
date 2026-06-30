#!/usr/bin/env python3
"""Render OpenMVS mesh depth under a frozen UMGS canonical camera.

This is a preparation-stage adapter for Road-0001 canonical OpenMVS rerender.
It intentionally avoids the previous COLMAP max-width camera path:

* no ``view_from_colmap(..., max_width=1200)`` call;
* no crop, resize, or resample of an existing 870-row depth artifact;
* no modification or re-export of the OpenMVS mesh.

The adapter treats the UMGS expected-camera fingerprint as the authoritative
runtime camera record. The fingerprint stores row-vector world-view/projection
matrices from the renderer path. For compatibility with
``openmvs_campaign_core.rasterize_mesh_camera_z``, this module converts the
row-vector world-view matrix into a standard column-vector matrix before
inverting it to ``CameraView.camera_to_world``. Projection uses the existing
triangle rasterizer convention: sample pixel centers at ``px + 0.5`` and
``py + 0.5`` while continuous projected coordinates use ``cx = width / 2`` and
``cy = height / 2`` for the centered FoV camera.

The output is a diagnostic proxy raster, not geometry ground truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import numba

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.depth_reference_geometry_v2.openmvs_campaign_core import (  # noqa: E402
    CameraView,
    load_ply_mesh,
    rasterize_mesh_camera_z,
)

SCHEMA_VERSION = "openmvs_canonical_camera_triangle_render_v1"
FINGERPRINT_ATOL = 5e-6
PROJECTION_ATOL = 5e-5
BARYCENTRIC_ATOL = 1e-5
PIXEL_CENTER_CONVENTION = "corner-origin_pixel-centers_at_index_plus_0.5"
PRINCIPAL_POINT_POLICY = "centered_fov_projection_cx_width_over_2_cy_height_over_2"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


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


def verify_sha256(path: str | Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual.lower() != str(expected).lower():
        raise ValueError(f"{label} SHA256 mismatch: expected {expected}, got {actual} for {path}")
    return actual


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def verify_runtime_source_manifest(
    manifest_path: str | Path,
    expected_sha256: str,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Verify source files before mesh loading or rasterization."""

    path = Path(manifest_path)
    manifest_sha = verify_sha256(path, expected_sha256, "runtime source manifest")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel = str(row.get("path", "")).strip()
            expected = str(row.get("sha256", "")).strip()
            if not rel or not expected:
                continue
            source_path = (repo_root / rel).resolve()
            try:
                source_path.relative_to(repo_root.resolve())
            except ValueError as exc:
                raise ValueError(f"runtime source path escapes repo root: {source_path}") from exc
            actual = verify_sha256(source_path, expected, f"runtime source {rel}")
            rows.append(
                {
                    "path": rel,
                    "sha256": actual,
                    "bytes": int(source_path.stat().st_size),
                    "role": row.get("role", ""),
                    "status": "verified",
                }
            )
    required = {
        "tools/depth_reference_geometry_v2/render_openmvs_canonical_camera.py",
        "tools/depth_reference_geometry_v2/validate_openmvs_canonical_camera_render.py",
        "tools/depth_reference_geometry_v2/openmvs_campaign_core.py",
    }
    present = {row["path"] for row in rows}
    missing = sorted(required - present)
    if missing:
        raise ValueError(f"runtime source manifest missing required sources: {missing}")
    return {
        "path": str(path),
        "sha256": manifest_sha,
        "source_files": rows,
    }


@dataclass(frozen=True)
class CanonicalCameraRecord:
    fingerprint_path: Path
    fingerprint_file_sha256: str
    fingerprint_payload_sha256: str
    payload: dict[str, Any]
    width: int
    height: int
    fovx: float
    fovy: float
    znear: float
    zfar: float
    fx: float
    fy: float
    cx: float
    cy: float
    world_view_row: np.ndarray
    projection_row: np.ndarray
    full_projection_row: np.ndarray
    camera_to_world: np.ndarray
    view: CameraView
    comparison: dict[str, Any]


def projection_matrix_from_fov_row(fovx: float, fovy: float, znear: float, zfar: float) -> np.ndarray:
    """Return the renderer-style row-vector projection matrix."""

    tan_half_x = math.tan(float(fovx) / 2.0)
    tan_half_y = math.tan(float(fovy) / 2.0)
    z_sign = 1.0
    mat = np.zeros((4, 4), dtype=np.float64)
    mat[0, 0] = 1.0 / tan_half_x
    mat[1, 1] = z_sign / tan_half_y
    mat[2, 2] = z_sign * float(zfar) / (float(zfar) - float(znear))
    mat[2, 3] = z_sign
    mat[3, 2] = -(float(zfar) * float(znear)) / (float(zfar) - float(znear))
    return mat


def world_view_row_from_rt(payload: dict[str, Any]) -> np.ndarray:
    r = np.asarray(payload["R"], dtype=np.float64)
    t = np.asarray(payload["T"], dtype=np.float64)
    if r.shape != (3, 3):
        raise ValueError(f"fingerprint R must be 3x3, got {r.shape}")
    if t.shape != (3,):
        raise ValueError(f"fingerprint T must be length 3, got {t.shape}")
    w = np.eye(4, dtype=np.float64)
    w[:3, :3] = r
    w[3, :3] = t
    return w


def standard_w2c_from_world_view_row(world_view_row: np.ndarray) -> np.ndarray:
    """Convert renderer row-vector world-view matrix to standard W2C matrix.

    The existing triangle rasterizer computes camera coordinates as
    ``p_world @ w2c_standard[:3,:3].T + w2c_standard[:3,3]``. The renderer
    fingerprint applies row-vector homogeneous transforms as
    ``[p_world, 1] @ world_view_row``. Therefore ``w2c_standard`` is the
    transpose of the fingerprint row-vector matrix.
    """

    w = np.asarray(world_view_row, dtype=np.float64)
    if w.shape != (4, 4):
        raise ValueError(f"world_view_row must be 4x4, got {w.shape}")
    return w.T.copy()


def camera_to_world_from_world_view_row(world_view_row: np.ndarray) -> np.ndarray:
    return np.linalg.inv(standard_w2c_from_world_view_row(world_view_row))


def camera_space_from_fingerprint_row(points_world: np.ndarray, world_view_row: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_world, dtype=np.float64)
    hom = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    cam = hom @ np.asarray(world_view_row, dtype=np.float64)
    return cam[:, :3]


def project_points_with_camera_view(points_world: np.ndarray, view: CameraView) -> np.ndarray:
    pts = np.asarray(points_world, dtype=np.float64)
    w2c = np.linalg.inv(np.asarray(view.camera_to_world, dtype=np.float64))
    cam = pts @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2]
    xy = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
    ok = np.isfinite(z) & (z > 0.0)
    xy[ok, 0] = float(view.fx) * (cam[ok, 0] / z[ok]) + float(view.cx)
    xy[ok, 1] = float(view.fy) * (cam[ok, 1] / z[ok]) + float(view.cy)
    return xy


def project_points_with_fingerprint_full_proj(points_world: np.ndarray, payload: dict[str, Any]) -> np.ndarray:
    pts = np.asarray(points_world, dtype=np.float64)
    full = np.asarray(payload["full_proj_transform"], dtype=np.float64)
    width = int(payload["image_width"])
    height = int(payload["image_height"])
    hom = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    clip = hom @ full
    w = clip[:, 3]
    xy = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
    ok = np.isfinite(w) & (np.abs(w) > 1e-12)
    ndc = np.full((pts.shape[0], 3), np.nan, dtype=np.float64)
    ndc[ok] = clip[ok, :3] / w[ok, None]
    xy[ok, 0] = (ndc[ok, 0] + 1.0) * float(width) / 2.0
    xy[ok, 1] = (ndc[ok, 1] + 1.0) * float(height) / 2.0
    return xy


def load_umgs_canonical_camera(
    fingerprint_path: str | Path,
    *,
    expected_file_sha256: str | None = None,
    expected_payload_sha256: str | None = None,
    expected_target: str | None = None,
    expected_width: int | None = None,
    expected_height: int | None = None,
    atol: float = FINGERPRINT_ATOL,
) -> CanonicalCameraRecord:
    path = Path(fingerprint_path)
    file_sha = sha256_file(path)
    if expected_file_sha256:
        verify_sha256(path, expected_file_sha256, "UMGS canonical camera fingerprint")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("payload"), dict):
        raise ValueError(f"fingerprint must contain a payload object: {path}")
    payload = data["payload"]
    payload_sha = canonical_json_sha256(payload)
    recorded_payload_sha = str(data.get("fingerprint_sha256", ""))
    if recorded_payload_sha != payload_sha:
        raise ValueError(f"fingerprint payload self-hash mismatch: recorded {recorded_payload_sha}, computed {payload_sha}")
    if expected_payload_sha256 and payload_sha.lower() != str(expected_payload_sha256).lower():
        raise ValueError(f"fingerprint payload SHA mismatch: expected {expected_payload_sha256}, got {payload_sha}")
    if expected_target and Path(str(payload.get("image_name", ""))).name != Path(expected_target).name:
        raise ValueError(f"target mismatch: expected {expected_target}, got {payload.get('image_name')}")

    width = int(payload["image_width"])
    height = int(payload["image_height"])
    if expected_width is not None and width != int(expected_width):
        raise ValueError(f"width mismatch: expected {expected_width}, got {width}")
    if expected_height is not None and height != int(expected_height):
        raise ValueError(f"height mismatch: expected {expected_height}, got {height}")

    fovx = float(payload["FoVx"])
    fovy = float(payload["FoVy"])
    znear = float(payload.get("znear", 0.01))
    zfar = float(payload.get("zfar", 100.0))
    fx = 0.5 * float(width) / math.tan(0.5 * fovx)
    fy = 0.5 * float(height) / math.tan(0.5 * fovy)
    cx = float(width) / 2.0
    cy = float(height) / 2.0

    world_view_row = np.asarray(payload["world_view_transform"], dtype=np.float64)
    projection_row = np.asarray(payload["projection_matrix"], dtype=np.float64)
    full_projection_row = np.asarray(payload["full_proj_transform"], dtype=np.float64)
    rt_world_view = world_view_row_from_rt(payload)
    computed_projection = projection_matrix_from_fov_row(fovx, fovy, znear, zfar)
    computed_full = world_view_row @ projection_row

    checks = {
        "rt_world_view_matches_payload": bool(np.allclose(rt_world_view, world_view_row, rtol=0.0, atol=atol)),
        "projection_from_fov_matches_payload": bool(np.allclose(computed_projection, projection_row, rtol=0.0, atol=atol)),
        "world_view_times_projection_matches_full": bool(np.allclose(computed_full, full_projection_row, rtol=0.0, atol=atol)),
        "world_view_max_abs_delta_from_rt": float(np.max(np.abs(rt_world_view - world_view_row))),
        "projection_max_abs_delta_from_fov": float(np.max(np.abs(computed_projection - projection_row))),
        "full_projection_max_abs_delta": float(np.max(np.abs(computed_full - full_projection_row))),
        "atol": float(atol),
    }
    failed = [k for k, v in checks.items() if k.endswith("_matches_payload") and not v]
    if failed:
        raise ValueError(f"canonical fingerprint matrix checks failed: {failed}; {checks}")

    camera_to_world = camera_to_world_from_world_view_row(world_view_row)
    view = CameraView(
        image_id=int(payload.get("colmap_id", -1)),
        image_name=str(payload.get("image_name", "")),
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        camera_to_world=camera_to_world,
    )
    return CanonicalCameraRecord(
        fingerprint_path=path,
        fingerprint_file_sha256=file_sha,
        fingerprint_payload_sha256=payload_sha,
        payload=payload,
        width=width,
        height=height,
        fovx=fovx,
        fovy=fovy,
        znear=znear,
        zfar=zfar,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        world_view_row=world_view_row,
        projection_row=projection_row,
        full_projection_row=full_projection_row,
        camera_to_world=camera_to_world,
        view=view,
        comparison=checks,
    )


def mesh_geometry_summary(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(vertices).all(axis=1) if len(vertices) else np.zeros(0, dtype=bool)
    bbox_min = np.nanmin(vertices[finite], axis=0).tolist() if finite.any() else [math.nan, math.nan, math.nan]
    bbox_max = np.nanmax(vertices[finite], axis=0).tolist() if finite.any() else [math.nan, math.nan, math.nan]
    return {
        "vertex_count": int(vertices.shape[0]),
        "face_count": int(faces.shape[0]),
        "finite_vertex_count": int(finite.sum()),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
    }


def source_sha_manifest(repo_root: Path) -> dict[str, Any]:
    rels = [
        "tools/depth_reference_geometry_v2/render_openmvs_canonical_camera.py",
        "tools/depth_reference_geometry_v2/validate_openmvs_canonical_camera_render.py",
        "tools/depth_reference_geometry_v2/test_openmvs_canonical_camera_adapter.py",
        "tools/depth_reference_geometry_v2/openmvs_campaign_core.py",
    ]
    rows = []
    for rel in rels:
        path = repo_root / rel
        rows.append(
            {
                "path": rel,
                "exists": path.exists(),
                "sha256": sha256_file(path) if path.exists() else "",
                "bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    return {"repo_root": str(repo_root), "files": rows}


def assert_output_dir_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to use non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def render_canonical_mesh(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    assert_output_dir_empty(output)
    runtime_manifest = verify_runtime_source_manifest(
        args.runtime_source_manifest,
        args.runtime_source_manifest_sha256,
    )
    camera = load_umgs_canonical_camera(
        args.fingerprint,
        expected_file_sha256=args.fingerprint_sha256,
        expected_payload_sha256=args.fingerprint_payload_sha256,
        expected_target=args.expected_target,
        expected_width=args.expected_width,
        expected_height=args.expected_height,
        atol=float(args.camera_atol),
    )
    mesh_ply = Path(args.mesh_ply)
    mesh_ply_sha = verify_sha256(mesh_ply, args.mesh_ply_sha256, "OpenMVS mesh PLY")
    mesh_mvs_sha = verify_sha256(args.mesh_mvs, args.mesh_mvs_sha256, "OpenMVS mesh MVS") if args.mesh_mvs else ""
    vertices, faces, mesh_meta = load_ply_mesh(mesh_ply)
    result = rasterize_mesh_camera_z(vertices, faces, camera.view)
    valid = result.valid & np.isfinite(result.depth) & (result.depth > 0.0)
    depth = result.depth.astype(np.float32)
    triangle_id = result.triangle_id.astype(np.int32)
    barycentric = result.barycentric.astype(np.float32)
    depth[~valid] = np.nan
    triangle_id[~valid] = -1
    barycentric[~valid] = np.nan

    npz_path = output / "openmvs_canonical_triangle_render.npz"
    np.savez_compressed(
        npz_path,
        depth=depth,
        valid=valid.astype(np.uint8),
        triangle_id=triangle_id,
        barycentric=barycentric,
    )
    manifest = {
        "schema": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_label": args.run_label,
        "scene": args.expected_scene,
        "target": args.expected_target,
        "no_proxy_metric": True,
        "not_ground_truth": True,
        "valid_definition": "triangle_hit AND finite_camera_z AND camera_z_gt_0",
        "invalid_convention": {
            "depth": "NaN",
            "triangle_id": -1,
            "barycentric": "NaN",
        },
        "barycentric_semantics": "screen_space_affine_weights_at_pixel_center",
        "barycentric_not_perspective_correct_surface_weights": True,
        "triangle_clipping_policy": "reject triangle if any vertex camera-z <= 1e-8; no near-plane clipping",
        "pixel_center_convention": PIXEL_CENTER_CONVENTION,
        "principal_point_policy": PRINCIPAL_POINT_POLICY,
        "camera": {
            "image_id": camera.view.image_id,
            "image_name": camera.view.image_name,
            "width": camera.width,
            "height": camera.height,
            "fx": camera.fx,
            "fy": camera.fy,
            "cx": camera.cx,
            "cy": camera.cy,
            "FoVx": camera.fovx,
            "FoVy": camera.fovy,
            "znear": camera.znear,
            "zfar": camera.zfar,
            "R": camera.payload["R"],
            "T": camera.payload["T"],
            "world_view_transform": camera.world_view_row.tolist(),
            "camera_to_world": camera.camera_to_world.tolist(),
            "projection_matrix": camera.projection_row.tolist(),
            "full_proj_transform": camera.full_projection_row.tolist(),
            "fingerprint_matrix_checks": camera.comparison,
        },
        "fingerprint": {
            "path": str(camera.fingerprint_path),
            "file_sha256": camera.fingerprint_file_sha256,
            "payload_sha256": camera.fingerprint_payload_sha256,
        },
        "mesh": {
            "ply_path": str(mesh_ply),
            "ply_sha256": mesh_ply_sha,
            "mvs_path": str(args.mesh_mvs) if args.mesh_mvs else "",
            "mvs_sha256": mesh_mvs_sha,
            "loaded_meta": mesh_meta,
            "geometry_summary": mesh_geometry_summary(vertices, faces),
        },
        "output": {
            "npz": str(npz_path),
            "npz_sha256": sha256_file(npz_path),
            "height": int(depth.shape[0]),
            "width": int(depth.shape[1]),
            "valid_pixel_count": int(valid.sum()),
            "valid_ratio": float(valid.mean()),
            "unique_triangle_count": int(len(np.unique(triangle_id[valid]))) if valid.any() else 0,
        },
        "runtime_source_manifest": runtime_manifest,
        "source_hash_manifest": source_sha_manifest(REPO_ROOT),
        "environment": {
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "numba": numba.__version__,
            "plyfile": package_version("plyfile"),
        },
        "exact_command": " ".join(sys.argv),
    }
    manifest_path = output / "openmvs_canonical_triangle_render_manifest.json"
    write_json(manifest_path, manifest)
    write_csv(output / "openmvs_canonical_triangle_render_summary.csv", [manifest["output"]])
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-ply", required=True)
    parser.add_argument("--mesh-ply-sha256", required=True)
    parser.add_argument("--mesh-mvs", required=True)
    parser.add_argument("--mesh-mvs-sha256", required=True)
    parser.add_argument("--runtime-source-manifest", required=True)
    parser.add_argument("--runtime-source-manifest-sha256", required=True)
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--fingerprint-sha256", required=True)
    parser.add_argument("--fingerprint-payload-sha256", required=True)
    parser.add_argument("--expected-scene", required=True)
    parser.add_argument("--expected-target", required=True)
    parser.add_argument("--expected-height", required=True, type=int)
    parser.add_argument("--expected-width", required=True, type=int)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--camera-atol", type=float, default=FINGERPRINT_ATOL)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = render_canonical_mesh(args)
    print(json.dumps({"status": "pass", "output": manifest["output"]}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
