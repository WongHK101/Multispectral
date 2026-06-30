#!/usr/bin/env python3
"""Core utilities for the OpenMVS conventional-geometry campaign.

The campaign deliberately separates two ideas:

* the benchmark COLMAP poses/sparse model, which may inherit all registered
  images from the benchmark preprocessing provenance; and
* source-image-only OpenMVS densification/meshing, where official held-out RGB
  images are excluded from OpenMVS inputs and only used later by this module's
  triangle rasterizer for evaluation.

Nothing in this module calls UMGS training/rendering code or mutates
checkpoints/support.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from plyfile import PlyData
from numba import njit

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.read_write_model import (  # noqa: E402
    Image,
    Point3D,
    qvec2rotmat,
    read_model,
    write_model,
)


@dataclass(frozen=True)
class CameraView:
    image_id: int
    image_name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    camera_to_world: np.ndarray


@dataclass
class TriangleRender:
    depth: np.ndarray
    triangle_id: np.ndarray
    barycentric: np.ndarray

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.depth) & (self.triangle_id >= 0)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json_payload(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
        f.write("\n")


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_name_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]


def name_keys(name: str) -> set[str]:
    p = Path(str(name).replace("\\", "/"))
    return {str(name), p.name, p.stem}


def list_hash(path: Path) -> str:
    names = sorted(read_name_list(path))
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def camera_intrinsics(camera: Any) -> tuple[float, float, float, float]:
    model = str(camera.model)
    params = np.asarray(camera.params, dtype=np.float64)
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params[:3]
        return float(f), float(f), float(cx), float(cy)
    if model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}:
        fx, fy, cx, cy = params[:4]
        return float(fx), float(fy), float(cx), float(cy)
    if model in {"SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "FOV"}:
        f, cx, cy = params[:3]
        return float(f), float(f), float(cx), float(cy)
    raise ValueError(f"Unsupported COLMAP camera model: {model}")


def camera_to_world_from_image(image: Any) -> np.ndarray:
    rot_w2c = qvec2rotmat(np.asarray(image.qvec, dtype=np.float64))
    t_w2c = np.asarray(image.tvec, dtype=np.float64)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rot_w2c.T
    c2w[:3, 3] = -rot_w2c.T @ t_w2c
    return c2w


def view_from_colmap(image: Any, camera: Any, *, max_width: int = 0) -> CameraView:
    width = int(camera.width)
    height = int(camera.height)
    fx, fy, cx, cy = camera_intrinsics(camera)
    if max_width and width > int(max_width):
        scale = float(max_width) / float(width)
        width = int(round(width * scale))
        height = int(round(height * scale))
        fx *= scale
        fy *= scale
        cx *= scale
        cy *= scale
    return CameraView(
        image_id=int(image.id),
        image_name=str(image.name),
        width=width,
        height=height,
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
        camera_to_world=camera_to_world_from_image(image),
    )


def load_ply_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    ply = PlyData.read(str(path))
    vertices = ply["vertex"]
    xyz = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype(np.float64, copy=False)
    if "face" not in ply:
        faces = np.empty((0, 3), dtype=np.int64)
    else:
        face_el = ply["face"]
        if "vertex_indices" in face_el.data.dtype.names:
            raw = face_el.data["vertex_indices"]
        elif "vertex_index" in face_el.data.dtype.names:
            raw = face_el.data["vertex_index"]
        else:
            raise ValueError(f"{path} face element has no vertex_indices field")
        faces = np.asarray([np.asarray(f, dtype=np.int64) for f in raw], dtype=np.int64)
        if faces.size and (faces.ndim != 2 or faces.shape[1] != 3):
            raise ValueError(f"{path} must contain triangular faces, got {faces.shape}")
    meta = {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "vertex_count": int(xyz.shape[0]),
        "face_count": int(faces.shape[0]),
        "sha256": sha256_file(path),
    }
    return xyz, faces, meta


def transform_world_to_camera(points_world: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    w2c = np.linalg.inv(np.asarray(camera_to_world, dtype=np.float64))
    return points_world @ w2c[:3, :3].T + w2c[:3, 3]


def rasterize_mesh_camera_z(vertices_world: np.ndarray, faces: np.ndarray, view: CameraView) -> TriangleRender:
    """Rasterize all mesh triangles using perspective-correct camera-z.

    Vertices behind the camera are rejected at triangle level; the routine does
    not clip partially visible triangles, which is acceptable for this
    diagnostic gate and explicitly tracked in coverage statistics.
    """

    vertices_cam = transform_world_to_camera(vertices_world, view.camera_to_world)
    depth, triangle_id, barycentric = _rasterize_mesh_camera_z_numba(
        vertices_cam.astype(np.float64, copy=False),
        np.asarray(faces, dtype=np.int64),
        float(view.fx),
        float(view.fy),
        float(view.cx),
        float(view.cy),
        int(view.width),
        int(view.height),
    )
    return TriangleRender(depth=depth, triangle_id=triangle_id, barycentric=barycentric)


@njit(cache=True)
def _rasterize_mesh_camera_z_numba(
    vertices_cam: np.ndarray,
    faces: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = np.empty((height, width), dtype=np.float64)
    depth[:, :] = np.nan
    triangle_id = np.empty((height, width), dtype=np.int32)
    triangle_id[:, :] = -1
    barycentric = np.empty((height, width, 3), dtype=np.float32)
    barycentric[:, :, :] = np.nan
    eps = 1e-12
    for fid in range(faces.shape[0]):
        i0 = faces[fid, 0]
        i1 = faces[fid, 1]
        i2 = faces[fid, 2]
        x0c = vertices_cam[i0, 0]
        y0c = vertices_cam[i0, 1]
        z0 = vertices_cam[i0, 2]
        x1c = vertices_cam[i1, 0]
        y1c = vertices_cam[i1, 1]
        z1 = vertices_cam[i1, 2]
        x2c = vertices_cam[i2, 0]
        y2c = vertices_cam[i2, 1]
        z2 = vertices_cam[i2, 2]
        if z0 <= 1e-8 or z1 <= 1e-8 or z2 <= 1e-8:
            continue
        x0 = fx * (x0c / z0) + cx
        y0 = fy * (y0c / z0) + cy
        x1 = fx * (x1c / z1) + cx
        y1 = fy * (y1c / z1) + cy
        x2 = fx * (x2c / z2) + cx
        y2 = fy * (y2c / z2) + cy
        if not (
            np.isfinite(x0)
            and np.isfinite(y0)
            and np.isfinite(x1)
            and np.isfinite(y1)
            and np.isfinite(x2)
            and np.isfinite(y2)
        ):
            continue
        min_x = max(0, int(math.floor(min(x0, x1, x2))))
        max_x = min(width - 1, int(math.ceil(max(x0, x1, x2))))
        min_y = max(0, int(math.floor(min(y0, y1, y2))))
        max_y = min(height - 1, int(math.ceil(max(y0, y1, y2))))
        if min_x > max_x or min_y > max_y:
            continue
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < eps:
            continue
        for py in range(min_y, max_y + 1):
            sy = py + 0.5
            for px in range(min_x, max_x + 1):
                sx = px + 0.5
                w0 = ((y1 - y2) * (sx - x2) + (x2 - x1) * (sy - y2)) / denom
                w1 = ((y2 - y0) * (sx - x2) + (x0 - x2) * (sy - y2)) / denom
                w2 = 1.0 - w0 - w1
                if w0 < -eps or w1 < -eps or w2 < -eps:
                    continue
                inv_z = w0 / z0 + w1 / z1 + w2 / z2
                if inv_z <= eps:
                    continue
                cam_z = 1.0 / inv_z
                current = depth[py, px]
                if np.isnan(current) or cam_z < current:
                    depth[py, px] = cam_z
                    triangle_id[py, px] = fid
                    barycentric[py, px, 0] = w0
                    barycentric[py, px, 1] = w1
                    barycentric[py, px, 2] = w2
    return depth, triangle_id, barycentric


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def mesh_topology_stats(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    finite_vertices = np.isfinite(vertices).all(axis=1) if len(vertices) else np.zeros(0, dtype=bool)
    used_vertices = set(int(v) for v in faces.ravel()) if faces.size else set()
    bbox_min = np.nanmin(vertices[finite_vertices], axis=0) if finite_vertices.any() else np.full(3, np.nan)
    bbox_max = np.nanmax(vertices[finite_vertices], axis=0) if finite_vertices.any() else np.full(3, np.nan)
    bbox_extent = bbox_max - bbox_min
    bbox_diag = float(np.linalg.norm(bbox_extent)) if np.isfinite(bbox_extent).all() else float("nan")

    if faces.size:
        tri = vertices[faces]
        edge_vec = tri[:, [1, 2, 0], :] - tri[:, [0, 1, 2], :]
        edge_lengths = np.linalg.norm(edge_vec, axis=2)
        areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    else:
        edge_lengths = np.empty((0, 3), dtype=np.float64)
        areas = np.empty(0, dtype=np.float64)
    area_eps = max(1e-18, (bbox_diag if np.isfinite(bbox_diag) else 1.0) ** 2 * 1e-14)
    degenerate = areas <= area_eps

    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fid, (a, b, c) in enumerate(faces.tolist()):
        for u, v in ((a, b), (b, c), (c, a)):
            if u > v:
                u, v = v, u
            edge_to_faces[(int(u), int(v))].append(fid)
    boundary_edges = [e for e, fs in edge_to_faces.items() if len(fs) == 1]
    nonmanifold_edges = [e for e, fs in edge_to_faces.items() if len(fs) > 2]

    uf = UnionFind(len(faces))
    for fs in edge_to_faces.values():
        if len(fs) >= 2:
            base = fs[0]
            for other in fs[1:]:
                uf.union(base, other)
    comp_faces: Counter[int] = Counter()
    comp_vertices: dict[int, set[int]] = defaultdict(set)
    for fid, face in enumerate(faces.tolist()):
        root = uf.find(fid) if len(faces) else 0
        comp_faces[root] += 1
        comp_vertices[root].update(int(v) for v in face)
    face_components = sorted(comp_faces.values(), reverse=True)
    vertex_components = sorted((len(vs) for vs in comp_vertices.values()), reverse=True)

    rounded = np.round(vertices, decimals=9) if len(vertices) else vertices
    duplicate_vertices = int(len(vertices) - np.unique(rounded, axis=0).shape[0]) if len(vertices) else 0
    sorted_faces = np.sort(faces, axis=1) if faces.size else faces.reshape(0, 3)
    duplicate_faces = int(len(faces) - np.unique(sorted_faces, axis=0).shape[0]) if len(faces) else 0

    edge_flat = edge_lengths.reshape(-1) if edge_lengths.size else np.empty(0, dtype=np.float64)
    quantiles = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    return {
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "finite_vertex_count": int(finite_vertices.sum()),
        "finite_vertex_ratio": float(finite_vertices.mean()) if len(vertices) else 0.0,
        "used_vertex_count": int(len(used_vertices)),
        "bbox_min_xyz": bbox_min.tolist(),
        "bbox_max_xyz": bbox_max.tolist(),
        "bbox_extent_xyz": bbox_extent.tolist(),
        "bbox_diag": bbox_diag,
        "connected_component_count": int(len(face_components)),
        "largest_component_face_count": int(face_components[0]) if face_components else 0,
        "largest_component_face_ratio": float(face_components[0] / len(faces)) if len(faces) and face_components else 0.0,
        "largest_component_vertex_count": int(vertex_components[0]) if vertex_components else 0,
        "largest_component_vertex_ratio": float(vertex_components[0] / len(vertices)) if len(vertices) and vertex_components else 0.0,
        "small_component_face_count_p50": float(np.percentile(face_components, 50)) if face_components else 0.0,
        "small_component_face_count_p95": float(np.percentile(face_components, 95)) if face_components else 0.0,
        "boundary_edge_count": int(len(boundary_edges)),
        "edge_count": int(len(edge_to_faces)),
        "boundary_edge_ratio": float(len(boundary_edges) / len(edge_to_faces)) if edge_to_faces else 0.0,
        "non_manifold_edge_count": int(len(nonmanifold_edges)),
        "non_manifold_edge_ratio": float(len(nonmanifold_edges) / len(edge_to_faces)) if edge_to_faces else 0.0,
        "degenerate_face_count": int(degenerate.sum()),
        "degenerate_face_ratio": float(degenerate.mean()) if len(degenerate) else 0.0,
        "duplicate_vertex_count_rounded_1e9": duplicate_vertices,
        "duplicate_face_count": duplicate_faces,
        "face_area_quantiles": {f"q{q:02d}": float(np.percentile(areas, q)) if len(areas) else 0.0 for q in quantiles},
        "edge_length_quantiles": {f"q{q:02d}": float(np.percentile(edge_flat, q)) if len(edge_flat) else 0.0 for q in quantiles},
        "long_edge_p99_over_bbox_diag": float(np.percentile(edge_flat, 99) / bbox_diag) if len(edge_flat) and bbox_diag > 0 else float("nan"),
        "max_edge_over_bbox_diag": float(np.max(edge_flat) / bbox_diag) if len(edge_flat) and bbox_diag > 0 else float("nan"),
    }


def approximate_point_to_mesh_vertex_distance(points: np.ndarray, vertices: np.ndarray, *, sample_limit: int = 200_000) -> dict[str, Any]:
    from scipy.spatial import cKDTree

    if len(points) == 0 or len(vertices) == 0:
        return {"count": 0, "median": None, "p90": None, "p95": None}
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) > sample_limit:
        rng = np.random.default_rng(7)
        pts = pts[rng.choice(len(pts), size=sample_limit, replace=False)]
    tree = cKDTree(np.asarray(vertices, dtype=np.float64))
    dist, _ = tree.query(pts, k=1, workers=-1)
    return {
        "count": int(len(dist)),
        "median": float(np.median(dist)),
        "p90": float(np.percentile(dist, 90)),
        "p95": float(np.percentile(dist, 95)),
    }


def read_sparse_xyz(points3d: dict[int, Any]) -> np.ndarray:
    return np.asarray([p.xyz for p in points3d.values()], dtype=np.float64)


def heldout_and_train_names(sparse_dir: Path, images: dict[int, Any]) -> tuple[list[str], list[str], dict[str, Any]]:
    test_file = sparse_dir / "test.txt"
    train_file = sparse_dir / "train.txt"
    test_names = read_name_list(test_file)
    if not test_names:
        all_names = sorted(str(img.name) for img in images.values())
        test_names = [name for idx, name in enumerate(all_names) if idx % 8 == 0]
    test_lookup = set().union(*(name_keys(n) for n in test_names)) if test_names else set()
    train_names = read_name_list(train_file)
    if not train_names:
        train_names = [str(img.name) for img in images.values() if not (name_keys(str(img.name)) & test_lookup)]
    meta = {
        "test_file": str(test_file) if test_file.exists() else "",
        "train_file": str(train_file) if train_file.exists() else "",
        "test_hash": list_hash(test_file) if test_file.exists() else hashlib.sha256("\n".join(sorted(test_names)).encode()).hexdigest(),
        "train_hash": list_hash(train_file) if train_file.exists() else hashlib.sha256("\n".join(sorted(train_names)).encode()).hexdigest(),
        "test_rule_if_missing": "zero-based sorted registered RGB index % 8 == 0" if not test_file.exists() else "",
    }
    return train_names, test_names, meta


def materialize_source_image_only_model(
    *,
    sparse_dir: Path,
    output_sparse_dir: Path,
    image_dir: Path,
    min_training_observations: int = 1,
) -> dict[str, Any]:
    cameras, images, points3d = read_model(str(sparse_dir), ext="")
    if cameras is None or images is None or points3d is None:
        raise FileNotFoundError(f"Could not read COLMAP sparse model: {sparse_dir}")
    train_names, heldout_names, split_meta = heldout_and_train_names(sparse_dir, images)
    train_lookup = set().union(*(name_keys(n) for n in train_names)) if train_names else set()
    heldout_lookup = set().union(*(name_keys(n) for n in heldout_names)) if heldout_names else set()
    train_images = {iid: img for iid, img in images.items() if name_keys(str(img.name)) & train_lookup}
    heldout_images = {iid: img for iid, img in images.items() if name_keys(str(img.name)) & heldout_lookup and iid not in train_images}
    if not train_images:
        raise RuntimeError(f"No training images selected from {sparse_dir}")
    if set(train_images) & set(heldout_images):
        raise RuntimeError("Split overlap detected between training and held-out image IDs")

    retained_points: dict[int, Any] = {}
    pruned_points = 0
    pruned_observations = 0
    for pid, point in points3d.items():
        keep_mask = np.asarray([int(iid) in train_images for iid in point.image_ids], dtype=bool)
        if int(keep_mask.sum()) < int(min_training_observations):
            pruned_points += 1
            pruned_observations += int((~keep_mask).sum())
            continue
        if not keep_mask.all():
            pruned_observations += int((~keep_mask).sum())
        retained_points[pid] = Point3D(
            id=point.id,
            xyz=point.xyz,
            rgb=point.rgb,
            error=point.error,
            image_ids=np.asarray(point.image_ids[keep_mask], dtype=np.int32),
            point2D_idxs=np.asarray(point.point2D_idxs[keep_mask], dtype=np.int32),
        )
    retained_point_ids = set(retained_points)
    filtered_images: dict[int, Any] = {}
    cleared_training_observations = 0
    for iid, image in train_images.items():
        point_ids = np.asarray(image.point3D_ids, dtype=np.int64).copy()
        invalid = np.asarray([int(pid) not in retained_point_ids for pid in point_ids], dtype=bool)
        invalid &= point_ids != -1
        cleared_training_observations += int(invalid.sum())
        point_ids[invalid] = -1
        filtered_images[iid] = Image(
            id=image.id,
            qvec=image.qvec,
            tvec=image.tvec,
            camera_id=image.camera_id,
            name=image.name,
            xys=image.xys,
            point3D_ids=point_ids,
        )
    used_camera_ids = {int(img.camera_id) for img in filtered_images.values()}
    filtered_cameras = {cid: cam for cid, cam in cameras.items() if int(cid) in used_camera_ids}
    output_sparse_dir.mkdir(parents=True, exist_ok=True)
    write_model(filtered_cameras, filtered_images, retained_points, str(output_sparse_dir), ext=".txt")
    write_model(filtered_cameras, filtered_images, retained_points, str(output_sparse_dir), ext=".bin")

    eval_manifest = {
        "schema": "source_image_only_evaluation_heldout_cameras_v1",
        "heldout_camera_use": "post_reconstruction_triangle_rasterization_only",
        "heldout_rgb_excluded_from_openmvs_densification_and_meshing": True,
        "heldout_images": [
            {
                "image_id": int(img.id),
                "image_name": str(img.name),
                "camera_id": int(img.camera_id),
                "rgb_path": str((image_dir / Path(str(img.name)).name).resolve()),
            }
            for img in sorted(heldout_images.values(), key=lambda im: str(im.name))
        ],
    }
    write_json(output_sparse_dir.parent / "heldout_evaluation_manifest.json", eval_manifest)
    summary = {
        "schema": "source_image_only_colmap_materialization_v1",
        "source_sparse_dir": str(sparse_dir),
        "output_sparse_dir": str(output_sparse_dir),
        "image_dir": str(image_dir),
        "split": split_meta,
        "registered_image_count_full": int(len(images)),
        "training_image_count": int(len(filtered_images)),
        "excluded_heldout_image_count": int(len(heldout_images)),
        "retained_camera_count": int(len(filtered_cameras)),
        "full_point_count": int(len(points3d)),
        "retained_point_count": int(len(retained_points)),
        "pruned_point_count": int(pruned_points),
        "pruned_heldout_or_removed_observation_count": int(pruned_observations),
        "cleared_training_observations_referencing_pruned_points": int(cleared_training_observations),
        "min_training_observations": int(min_training_observations),
        "benchmark_colmap_pose_provenance_disclosure": (
            "Camera poses and sparse geometry inherit the benchmark COLMAP provenance and may have used all registered images; "
            "OpenMVS densification/meshing inputs exclude held-out/test RGB."
        ),
    }
    write_json(output_sparse_dir.parent / "source_image_only_materialization_summary.json", summary)
    return summary


def leakage_audit(
    *,
    root: Path,
    reconstruction_sparse_dir: Path,
    heldout_names: Sequence[str],
    min_training_observations: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    heldout_basenames = sorted({Path(n).name for n in heldout_names})
    rows: list[dict[str, Any]] = []
    scan_ext = {".mvs", ".json", ".txt", ".log", ".ini", ".yaml", ".yml", ".csv", ".sh"}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in scan_ext:
            continue
        # This manifest is the explicitly allowed boundary: held-out camera
        # records are retained here for post-reconstruction triangle rendering,
        # but OpenMVS command inputs and scene files must not reference them.
        if path.name == "heldout_evaluation_manifest.json":
            continue
        try:
            data = path.read_bytes()
        except Exception:
            continue
        for name in heldout_basenames:
            if name.encode("utf-8", errors="ignore") in data:
                rows.append({"check": "heldout_filename_in_generated_file", "path": str(path), "heldout_name": name, "pass": False})
    cameras, images, points3d = read_model(str(reconstruction_sparse_dir), ext="")
    heldout_lookup = set().union(*(name_keys(n) for n in heldout_basenames)) if heldout_basenames else set()
    for image in images.values():
        if name_keys(str(image.name)) & heldout_lookup:
            rows.append({"check": "heldout_image_record_in_reconstruction_model", "path": str(reconstruction_sparse_dir), "heldout_name": str(image.name), "pass": False})
    bad_points = 0
    for point in points3d.values():
        if len(point.image_ids) < min_training_observations:
            bad_points += 1
    if bad_points:
        rows.append({"check": "retained_point_below_min_training_observations", "path": str(reconstruction_sparse_dir), "heldout_name": "", "pass": False, "bad_point_count": bad_points})
    if not rows:
        rows.append({"check": "all_leakage_checks", "path": str(root), "heldout_name": "", "pass": True})
    summary = {
        "schema": "source_image_only_leakage_audit_v1",
        "root": str(root),
        "reconstruction_sparse_dir": str(reconstruction_sparse_dir),
        "heldout_name_count": int(len(heldout_basenames)),
        "leakage_issue_count": int(sum(1 for r in rows if not bool(r.get("pass")))),
        "pass": bool(all(bool(r.get("pass")) for r in rows)),
    }
    return rows, summary


def render_views_summary(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    cameras: dict[int, Any],
    images: dict[int, Any],
    target_names: Sequence[str],
    out_dir: Path,
    max_width: int = 1200,
) -> list[dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    target_lookup = set().union(*(name_keys(n) for n in target_names)) if target_names else set()
    rows: list[dict[str, Any]] = []
    for image in sorted(images.values(), key=lambda im: str(im.name)):
        if target_lookup and not (name_keys(str(image.name)) & target_lookup):
            continue
        view = view_from_colmap(image, cameras[int(image.camera_id)], max_width=max_width)
        result = rasterize_mesh_camera_z(vertices, faces, view)
        valid = result.valid
        finite_depth = result.depth[valid]
        row = {
            "image_name": str(image.name),
            "render_width": int(view.width),
            "render_height": int(view.height),
            "valid_pixel_count": int(valid.sum()),
            "coverage": float(valid.mean()),
            "depth_min": float(np.min(finite_depth)) if len(finite_depth) else None,
            "depth_p02": float(np.percentile(finite_depth, 2)) if len(finite_depth) else None,
            "depth_median": float(np.median(finite_depth)) if len(finite_depth) else None,
            "depth_p90": float(np.percentile(finite_depth, 90)) if len(finite_depth) else None,
            "depth_p98": float(np.percentile(finite_depth, 98)) if len(finite_depth) else None,
            "depth_max": float(np.max(finite_depth)) if len(finite_depth) else None,
            "depth_p98_over_p02": (
                float(np.percentile(finite_depth, 98) / max(np.percentile(finite_depth, 2), 1e-12))
                if len(finite_depth)
                else None
            ),
            "unique_triangle_count": int(len(np.unique(result.triangle_id[valid]))) if valid.any() else 0,
        }
        rows.append(row)
        preview_path = out_dir / f"triangle_depth_{Path(str(image.name)).stem}.png"
        fig, ax = plt.subplots(figsize=(5.0, 3.7), dpi=160)
        ax.axis("off")
        if valid.any():
            lo, hi = np.percentile(finite_depth, [2, 98])
            im = ax.imshow(result.depth, cmap="viridis", vmin=lo, vmax=hi)
            fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01)
        else:
            ax.imshow(np.zeros_like(result.depth), cmap="gray")
        ax.set_title(f"{Path(str(image.name)).stem} coverage={row['coverage']:.3f}", fontsize=8)
        fig.tight_layout(pad=0.1)
        fig.savefig(preview_path, bbox_inches="tight")
        plt.close(fig)
        row["preview_path"] = str(preview_path)
        np.savez_compressed(
            out_dir / f"triangle_render_{Path(str(image.name)).stem}.npz",
            depth=result.depth.astype(np.float32),
            valid=valid.astype(np.uint8),
            triangle_id=result.triangle_id.astype(np.int32),
            barycentric=result.barycentric.astype(np.float32),
        )
    return rows


def make_numeric_qualification_gates(all_image_audit: dict[str, Any]) -> dict[str, Any]:
    topology = all_image_audit.get("topology", {})
    render = all_image_audit.get("heldout_render_summary", {})
    sparse_ratio = float(all_image_audit.get("mesh_sparse_bbox_diag_ratio", 1.0) or 1.0)
    median_cov = float(render.get("coverage_median", 0.25) or 0.25)
    min_cov = float(render.get("coverage_min", 0.05) or 0.05)
    gates = {
        "schema": "openmvs_diagnostic_qualification_gates_v1",
        "immutable_after_sha256_recorded": True,
        "evidence_allowed_before_freeze": [
            "synthetic_rasterizer_topology_tests",
            "camera_depth_physical_semantics",
            "topology_engineering_constraints",
            "maize02_all_image_openmvs_formal_audit",
        ],
        "forbidden_after_freeze": [
            "maize_source_image_only_result",
            "papaya_legacy_agreement",
            "road_or_wogan_success_rate",
            "da3_result",
            "umgs_or_baseline_ranking",
        ],
        "finite_geometry": {
            "minimum_vertex_count": 1000,
            "minimum_face_count": 1000,
            "minimum_finite_vertex_ratio": 0.999,
        },
        "bbox_scale": {
            "mesh_to_sparse_bbox_diag_ratio_min": max(0.20, sparse_ratio / 4.0),
            "mesh_to_sparse_bbox_diag_ratio_max": min(5.00, sparse_ratio * 4.0),
            "normalization": "diag(mesh_bbox) / diag(benchmark_colmap_sparse_bbox)",
        },
        "topology": {
            "minimum_largest_component_face_ratio_pass": 0.70,
            "minimum_largest_component_face_ratio_caution": 0.50,
            "maximum_degeneration_face_ratio_pass": 0.005,
            "maximum_degeneration_face_ratio_caution": 0.020,
            "maximum_non_manifold_edge_ratio_pass": 0.020,
            "maximum_non_manifold_edge_ratio_caution": 0.050,
            "boundary_edge_ratio_pass_max": 0.60,
            "boundary_edge_ratio_caution_max": 0.80,
            "long_edge_p99_over_bbox_diag_pass_max": 0.10,
            "max_edge_over_bbox_diag_caution_max": 0.35,
        },
        "heldout_triangle_render": {
            "minimum_per_target_coverage_pass": max(0.05, min(0.20, min_cov * 0.50)),
            "minimum_median_coverage_pass": max(0.10, min(0.30, median_cov * 0.60)),
            "minimum_usable_heldout_target_fraction": 0.50,
            "usable_target_coverage_threshold": max(0.05, min(0.20, min_cov * 0.50)),
        },
        "depth_range": {
            "positive_depth_required": True,
            "depth_p98_over_p02_max": 200.0,
            "depth_range_rule": "computed per rendered held-out target on finite triangle-rendered camera-z",
        },
        "support_distance": {
            "sparse_to_mesh_vertex_distance_median_over_sparse_diag_caution_max": 0.05,
            "dense_to_mesh_vertex_distance_median_over_mesh_diag_caution_max": 0.03,
            "distance_backend": "nearest mesh vertex distance; diagnostic support proxy, not exact signed point-to-triangle distance",
        },
        "visual_mapping": {
            "pass": "no catastrophic fold, bridge, closure, inversion, large floating sheet, or systematic target-view collapse",
            "caution": "localized holes, boundary tears, high boundary ratio, or uncertain human inspection without catastrophic structure",
            "fail": "catastrophic bridge/closure/fold/inversion, empty useful surface, severe unsupported sheets, or camera/mesh convention mismatch",
            "automatic_use": "visual_caution_only",
        },
        "diagnostic_only_fields": [
            "dense_to_mesh distance if no dense cloud is available",
            "human visual assessment",
            "DA3 overlap",
            "legacy Papaya agreement",
        ],
    }
    gates["sha256"] = sha256_json_payload({k: v for k, v in gates.items() if k != "sha256"})
    return gates


def evaluate_audit_against_gates(audit: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    """Apply frozen engineering gates to a mesh audit.

    The output is a diagnostic eligibility decision, not an accuracy claim.
    """

    failed: list[str] = []
    caution: list[str] = []
    topo = audit.get("topology", {})
    render = audit.get("heldout_render_summary", {})
    mesh = audit.get("mesh_meta", {})
    ratio = audit.get("mesh_sparse_bbox_diag_ratio", None)

    finite = gates["finite_geometry"]
    if int(mesh.get("vertex_count", 0)) < int(finite["minimum_vertex_count"]):
        failed.append("minimum_vertex_count")
    if int(mesh.get("face_count", 0)) < int(finite["minimum_face_count"]):
        failed.append("minimum_face_count")
    if float(topo.get("finite_vertex_ratio", 0.0)) < float(finite["minimum_finite_vertex_ratio"]):
        failed.append("minimum_finite_vertex_ratio")

    bbox = gates["bbox_scale"]
    if ratio is None or not np.isfinite(float(ratio)):
        failed.append("mesh_sparse_bbox_diag_ratio_missing")
    elif not (float(bbox["mesh_to_sparse_bbox_diag_ratio_min"]) <= float(ratio) <= float(bbox["mesh_to_sparse_bbox_diag_ratio_max"])):
        failed.append("mesh_sparse_bbox_diag_ratio_out_of_range")

    tg = gates["topology"]
    lcf = float(topo.get("largest_component_face_ratio", 0.0))
    if lcf < float(tg["minimum_largest_component_face_ratio_caution"]):
        failed.append("largest_component_face_ratio")
    elif lcf < float(tg["minimum_largest_component_face_ratio_pass"]):
        caution.append("largest_component_face_ratio_caution")
    deg = float(topo.get("degenerate_face_ratio", 1.0))
    if deg > float(tg["maximum_degeneration_face_ratio_caution"]):
        failed.append("degenerate_face_ratio")
    elif deg > float(tg["maximum_degeneration_face_ratio_pass"]):
        caution.append("degenerate_face_ratio_caution")
    nm = float(topo.get("non_manifold_edge_ratio", 1.0))
    if nm > float(tg["maximum_non_manifold_edge_ratio_caution"]):
        failed.append("non_manifold_edge_ratio")
    elif nm > float(tg["maximum_non_manifold_edge_ratio_pass"]):
        caution.append("non_manifold_edge_ratio_caution")
    bd = float(topo.get("boundary_edge_ratio", 1.0))
    if bd > float(tg["boundary_edge_ratio_caution_max"]):
        failed.append("boundary_edge_ratio")
    elif bd > float(tg["boundary_edge_ratio_pass_max"]):
        caution.append("boundary_edge_ratio_caution")
    p99 = float(topo.get("long_edge_p99_over_bbox_diag", np.inf))
    if np.isfinite(p99) and p99 > float(tg["long_edge_p99_over_bbox_diag_pass_max"]):
        caution.append("long_edge_p99_over_bbox_diag_caution")
    mx = float(topo.get("max_edge_over_bbox_diag", np.inf))
    if np.isfinite(mx) and mx > float(tg["max_edge_over_bbox_diag_caution_max"]):
        failed.append("max_edge_over_bbox_diag")

    rg = gates["heldout_triangle_render"]
    if float(render.get("coverage_min", 0.0)) < float(rg["minimum_per_target_coverage_pass"]):
        failed.append("minimum_per_target_triangle_render_coverage")
    if float(render.get("coverage_median", 0.0)) < float(rg["minimum_median_coverage_pass"]):
        failed.append("minimum_median_triangle_render_coverage")
    usable_fraction = float(render.get("usable_target_fraction_coverage_ge_0p05", 0.0))
    if usable_fraction < float(rg["minimum_usable_heldout_target_fraction"]):
        failed.append("minimum_usable_heldout_target_fraction")

    status = "L3_source_image_only_diagnostic_eligible"
    if failed:
        status = "fail_scene_geometry"
    elif caution:
        status = "L3_source_image_only_caution"
    return {
        "schema": "openmvs_frozen_gate_qualification_v1",
        "scene": audit.get("scene", ""),
        "track": audit.get("track", ""),
        "gates_sha256_payload": gates.get("sha256", ""),
        "status": status,
        "auditable": status in {"L3_source_image_only_diagnostic_eligible", "L3_source_image_only_caution"},
        "failed_gates": failed,
        "caution_gates": caution,
        "claim_boundary": "Engineering diagnostic eligibility only; not ground truth or absolute geometry accuracy.",
    }
