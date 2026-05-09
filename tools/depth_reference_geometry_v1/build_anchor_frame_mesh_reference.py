from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import Delaunay, cKDTree

from depth_reference_common import (
    compute_quantile_bbox,
    load_json,
    load_ply_mesh,
    load_ply_points_xyz,
    render_mesh_depth_for_view,
    render_support_count_for_view,
    save_json,
    transform_native_points_to_strict_reference_units,
)


def _write_mesh(path: Path, vertices_xyz: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertex_data = np.empty(
        vertices_xyz.shape[0],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],
    )
    vertex_data["x"] = vertices_xyz[:, 0]
    vertex_data["y"] = vertices_xyz[:, 1]
    vertex_data["z"] = vertices_xyz[:, 2]
    face_data = np.empty(faces.shape[0], dtype=[("vertex_indices", "i4", (3,))])
    face_data["vertex_indices"] = faces.astype(np.int32, copy=False)
    PlyData([PlyElement.describe(vertex_data, "vertex"), PlyElement.describe(face_data, "face")], text=False).write(
        str(path)
    )


def _write_vertex_only_ply_like(src: Path, dst: Path, xyz: np.ndarray) -> None:
    ply = PlyData.read(str(src))
    src_vertices = ply["vertex"].data
    if xyz.shape[0] == len(src_vertices):
        vertices = src_vertices.copy()
        vertices["x"] = xyz[:, 0].astype(vertices["x"].dtype)
        vertices["y"] = xyz[:, 1].astype(vertices["y"].dtype)
        vertices["z"] = xyz[:, 2].astype(vertices["z"].dtype)
    else:
        vertices = np.empty(xyz.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
        vertices["x"] = xyz[:, 0]
        vertices["y"] = xyz[:, 1]
        vertices["z"] = xyz[:, 2]
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(str(dst))


def _copy_mesh_with_transformed_xyz(src: Path, dst: Path, xyz: np.ndarray) -> None:
    ply = PlyData.read(str(src))
    vertices = ply["vertex"].data.copy()
    vertices["x"] = xyz[:, 0].astype(vertices["x"].dtype)
    vertices["y"] = xyz[:, 1].astype(vertices["y"].dtype)
    vertices["z"] = xyz[:, 2].astype(vertices["z"].dtype)
    PlyData([PlyElement.describe(vertices, "vertex"), ply["face"]], text=False).write(str(dst))


def _pca_delaunay_mesh(
    points: np.ndarray,
    *,
    max_points: int,
    crop_low: float,
    crop_high: float,
    padding_ratio: float,
    edge_nn_quantile: float,
    edge_nn_scale: float,
) -> Dict[str, Any]:
    points = np.asarray(points, dtype=np.float64)
    lo = np.quantile(points, crop_low, axis=0)
    hi = np.quantile(points, crop_high, axis=0)
    pad = padding_ratio * float(np.linalg.norm(hi - lo))
    keep = np.all((points >= lo - pad) & (points <= hi + pad), axis=1)
    points = points[keep]
    if points.shape[0] < 4:
        raise RuntimeError(f"Too few points after crop: {points.shape[0]}")
    if points.shape[0] > max_points:
        idx = np.linspace(0, points.shape[0] - 1, max_points).round().astype(np.int64)
        points = points[idx]

    mean = points.mean(axis=0)
    centered = points - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    uv = centered @ vh[:2].T
    tri = Delaunay(uv)
    faces = tri.simplices.astype(np.int32, copy=False)

    nn = cKDTree(uv).query(uv, k=2)[0][:, 1]
    edge_threshold = float(np.quantile(nn, edge_nn_quantile) * edge_nn_scale)
    face_uv = uv[faces]
    edge_lengths = np.stack(
        [
            np.linalg.norm(face_uv[:, 0] - face_uv[:, 1], axis=1),
            np.linalg.norm(face_uv[:, 1] - face_uv[:, 2], axis=1),
            np.linalg.norm(face_uv[:, 2] - face_uv[:, 0], axis=1),
        ],
        axis=1,
    )
    faces = faces[np.max(edge_lengths, axis=1) <= edge_threshold]
    if faces.shape[0] <= 0:
        raise RuntimeError("PCA-Delaunay produced no faces after edge filtering")
    return {
        "points": points,
        "faces": faces,
        "audit": {
            "input_points_after_crop": int(np.count_nonzero(keep)),
            "used_points": int(points.shape[0]),
            "faces": int(faces.shape[0]),
            "crop_quantile_low": float(crop_low),
            "crop_quantile_high": float(crop_high),
            "padding_ratio": float(padding_ratio),
            "edge_nn_quantile": float(edge_nn_quantile),
            "edge_nn_scale": float(edge_nn_scale),
            "edge_threshold": edge_threshold,
            "pca_axes": vh[:3].tolist(),
        },
    }


def _load_cameras_json_views(
    cameras_json: Path,
    *,
    image_names: List[str],
    width_scale: float,
    height_scale: float,
) -> List[Dict[str, Any]]:
    cameras = load_json(cameras_json)
    by_stem = {Path(str(item["img_name"])).stem.lower(): item for item in cameras}
    views: List[Dict[str, Any]] = []
    for idx, image_name in enumerate(image_names):
        key = Path(image_name).stem.lower()
        if key not in by_stem:
            raise KeyError(f"{image_name} not found in {cameras_json}")
        cam = by_stem[key]
        width = int(round(int(cam["width"]) * width_scale))
        height = int(round(int(cam["height"]) * height_scale))
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = np.asarray(cam["rotation"], dtype=np.float64)
        c2w[:3, 3] = np.asarray(cam["position"], dtype=np.float64)
        views.append(
            {
                "view_id": f"{idx:05d}",
                "image_name": image_name,
                "width": width,
                "height": height,
                "fx": float(cam["fx"]) * width_scale,
                "fy": float(cam["fy"]) * height_scale,
                "cx": float(width) / 2.0,
                "cy": float(height) / 2.0,
                "camera_to_world": c2w.tolist(),
            }
        )
    return views


def _render_reference(
    *,
    scene_name: str,
    out_dir: Path,
    mesh_path: Path,
    support_points_path: Path,
    views: List[Dict[str, Any]],
    variant: str,
    build_audit: Dict[str, Any],
    support_radius_px: int,
    support_depth_tolerance: float,
) -> None:
    vertices, faces = load_ply_mesh(mesh_path)
    support_points = load_ply_points_xyz(support_points_path)
    roi = compute_quantile_bbox(
        support_points,
        lower_quantile=0.01,
        upper_quantile=0.99,
        padding_ratio_of_robust_diagonal=0.02,
    )
    save_json(
        out_dir / "reference_roi.json",
        {
            "scene_name": scene_name,
            "bbox_min": roi["bbox_min"].tolist(),
            "bbox_max": roi["bbox_max"].tolist(),
            "scene_diagonal": float(roi["scene_diagonal"]),
            "source_points_path": str(support_points_path),
        },
    )

    views_dir = out_dir / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    manifest_views: List[Dict[str, Any]] = []
    valid_ratios: List[float] = []
    for view in views:
        depth = render_mesh_depth_for_view(vertices, faces, view)
        support = render_support_count_for_view(
            support_points,
            view,
            depth_tolerance_m=support_depth_tolerance,
            support_radius_px=support_radius_px,
        )
        valid = np.isfinite(depth) & (depth > 1.0e-6) & (support >= 1)
        rel = Path("views") / f"{view['view_id']}.npz"
        np.savez_compressed(
            out_dir / rel,
            depth=np.asarray(depth, dtype=np.float64),
            valid_mask=np.asarray(valid, dtype=np.uint8),
            support_count=np.asarray(support, dtype=np.int32),
        )
        item = dict(view)
        item["npz_file"] = str(rel).replace("\\", "/")
        manifest_views.append(item)
        valid_ratios.append(float(valid.mean()))

    manifest = {
        "protocol_name": "reference-depth-based-geometric-evaluation-v1",
        "scene_name": scene_name,
        "reference_variant": variant,
        "reference_mesh_path": str(mesh_path),
        "reference_fused_ply_for_meshing": str(support_points_path),
        "reference_mesh_backend": build_audit.get("reference_mesh_backend", variant),
        "reference_depth_backend": "mesh",
        "depth_semantics": "metric_camera_z_reference_mesh",
        "distance_unit": build_audit.get("distance_unit", "scene_units"),
        "scale_mode": build_audit.get("scale_mode", "anchor_or_probe_frame"),
        "thresholds_m": [0.005, 0.01, 0.025, 0.05, 0.10, 0.20],
        "valid_ratio_mean": float(np.mean(valid_ratios)) if valid_ratios else 0.0,
        "valid_ratio_min": float(np.min(valid_ratios)) if valid_ratios else 0.0,
        "valid_ratio_max": float(np.max(valid_ratios)) if valid_ratios else 0.0,
        "support_rule": {
            "type": "mesh_depth_plus_support_points",
            "support_radius_px": int(support_radius_px),
            "support_depth_tolerance": float(support_depth_tolerance),
        },
        "views": manifest_views,
    }
    save_json(out_dir / "reference_depth_manifest.json", manifest)
    save_json(out_dir / "reference_build_manifest.json", {**build_audit, "reference_depth_manifest": str(out_dir / "reference_depth_manifest.json")})
    print(f"ANCHOR_FRAME_REFERENCE {out_dir / 'reference_depth_manifest.json'}")


def _image_names_from_reference(reference_manifest: Path) -> List[str]:
    manifest = load_json(reference_manifest)
    return [str(view["image_name"]) for view in manifest["views"]]


def _cmd_transform_existing(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    old_reference_dir = Path(args.old_reference_dir).resolve()
    alignment = load_json(Path(args.alignment_json).resolve())
    probe_manifest = load_json(Path(args.probe_manifest).resolve())

    mesh_in = Path(args.mesh_ply).resolve() if args.mesh_ply else old_reference_dir / "reference_mesh_poisson_self_m3m_m4_d0020_n20_roi_d10_trim6_pw4.ply"
    fused_in = Path(args.support_ply).resolve() if args.support_ply else old_reference_dir / "reference_fused_geometric_roi_crop.ply"

    vertices, faces = load_ply_mesh(mesh_in)
    transformed_vertices = transform_native_points_to_strict_reference_units(vertices, alignment)
    mesh_out = out_dir / "reference_mesh_probe_frame.ply"
    _write_mesh(mesh_out, transformed_vertices, faces)

    support_points = load_ply_points_xyz(fused_in)
    transformed_support = transform_native_points_to_strict_reference_units(support_points, alignment)
    support_out = out_dir / "reference_support_probe_frame.ply"
    _write_vertex_only_ply_like(fused_in, support_out, transformed_support)

    _render_reference(
        scene_name=str(args.scene_name),
        out_dir=out_dir,
        mesh_path=mesh_out,
        support_points_path=support_out,
        views=list(probe_manifest["views"]),
        variant="dense_mesh_transformed_to_probe_frame_v1",
        build_audit={
            "scene_name": str(args.scene_name),
            "source_reference_dir": str(old_reference_dir),
            "source_mesh": str(mesh_in),
            "source_support": str(fused_in),
            "alignment": alignment,
            "reference_mesh_backend": "transformed_existing_mesh",
            "distance_unit": "meters",
            "scale_mode": "probe_frame_metric_verified",
        },
        support_radius_px=int(args.support_radius_px),
        support_depth_tolerance=float(args.support_depth_tolerance),
    )


def _cmd_pca_from_points(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    points_path = Path(args.points_ply).resolve()
    points = load_ply_points_xyz(points_path)
    result = _pca_delaunay_mesh(
        points,
        max_points=int(args.max_points),
        crop_low=float(args.crop_low),
        crop_high=float(args.crop_high),
        padding_ratio=float(args.padding_ratio),
        edge_nn_quantile=float(args.edge_nn_quantile),
        edge_nn_scale=float(args.edge_nn_scale),
    )
    mesh_path = out_dir / "reference_mesh_pca_delaunay.ply"
    support_path = out_dir / "reference_support_points.ply"
    _write_mesh(mesh_path, result["points"], result["faces"])
    _write_vertex_only_ply_like(points_path, support_path, result["points"])

    image_names = _image_names_from_reference(Path(args.image_names_from_reference).resolve())
    views = _load_cameras_json_views(
        Path(args.cameras_json).resolve(),
        image_names=image_names,
        width_scale=float(args.width_scale),
        height_scale=float(args.height_scale),
    )
    _render_reference(
        scene_name=str(args.scene_name),
        out_dir=out_dir,
        mesh_path=mesh_path,
        support_points_path=support_path,
        views=views,
        variant="pca_delaunay_mesh_in_anchor_frame_v1",
        build_audit={
            "scene_name": str(args.scene_name),
            "points_ply": str(points_path),
            "cameras_json": str(Path(args.cameras_json).resolve()),
            "image_names_from_reference": str(Path(args.image_names_from_reference).resolve()),
            "mesh_audit": result["audit"],
            "reference_mesh_backend": "pca_delaunay_from_points",
            "distance_unit": "scene_units",
            "scale_mode": "anchor_frame",
        },
        support_radius_px=int(args.support_radius_px),
        support_depth_tolerance=float(args.support_depth_tolerance),
    )


def _argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build mesh-reference depth manifests in the method/probe frame.")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("transform_existing")
    t.add_argument("--scene_name", required=True)
    t.add_argument("--old_reference_dir", required=True)
    t.add_argument("--probe_manifest", required=True)
    t.add_argument("--alignment_json", required=True)
    t.add_argument("--out_dir", required=True)
    t.add_argument("--mesh_ply", default="")
    t.add_argument("--support_ply", default="")
    t.add_argument("--support_radius_px", type=int, default=2)
    t.add_argument("--support_depth_tolerance", type=float, default=0.10)
    t.set_defaults(func=_cmd_transform_existing)

    p = sub.add_parser("pca_from_points")
    p.add_argument("--scene_name", required=True)
    p.add_argument("--points_ply", required=True)
    p.add_argument("--cameras_json", required=True)
    p.add_argument("--image_names_from_reference", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--width_scale", type=float, default=0.5)
    p.add_argument("--height_scale", type=float, default=0.5)
    p.add_argument("--max_points", type=int, default=80000)
    p.add_argument("--crop_low", type=float, default=0.01)
    p.add_argument("--crop_high", type=float, default=0.99)
    p.add_argument("--padding_ratio", type=float, default=0.05)
    p.add_argument("--edge_nn_quantile", type=float, default=0.75)
    p.add_argument("--edge_nn_scale", type=float, default=8.0)
    p.add_argument("--support_radius_px", type=int, default=2)
    p.add_argument("--support_depth_tolerance", type=float, default=0.10)
    p.set_defaults(func=_cmd_pca_from_points)
    return parser


def main() -> None:
    args = _argparser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
