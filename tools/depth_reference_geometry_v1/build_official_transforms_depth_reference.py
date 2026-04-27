from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from depth_reference_common import (
    compute_inside_bbox_mask,
    compute_quantile_bbox,
    compute_scaled_resolution,
    load_json,
    load_ply_points_xyz,
    parse_thresholds_m,
    render_point_splat_depth_for_view,
    save_json,
)


def _argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a lightweight point-splat reference-depth smoke artifact from an "
            "official aligned transforms_train/test scene. This is for chain validation, "
            "not a COLMAP dense MVS formal reference."
        )
    )
    parser.add_argument("--source_path", required=True, help="Official prepared scene root containing transforms_train/test.json and points3d.ply")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--scene_name", default="")
    parser.add_argument("--resolution_arg", type=int, default=4)
    parser.add_argument("--thresholds_m", default="0.005,0.01,0.025,0.05,0.10,0.20")
    parser.add_argument("--distance_unit", default="scene_units")
    parser.add_argument("--scale_mode", default="scene_normalized")
    parser.add_argument("--bbox_lower_quantile", type=float, default=0.01)
    parser.add_argument("--bbox_upper_quantile", type=float, default=0.99)
    parser.add_argument("--bbox_padding_ratio", type=float, default=0.02)
    parser.add_argument("--splat_radius_px", type=int, default=1)
    parser.add_argument("--support_min_count", type=int, default=1)
    parser.add_argument("--support_depth_tolerance_m", type=float, default=0.10)
    parser.add_argument("--max_views", type=int, default=None)
    return parser


def _resolve_frame_path(source_path: Path, frame_file_path: str) -> str:
    path = Path(str(frame_file_path))
    if path.is_absolute():
        return str(path)
    return str((source_path / path).resolve())


def _view_from_frame(source_path: Path, frame: Dict[str, Any], idx: int, resolution_arg: int) -> Dict[str, Any]:
    file_path = _resolve_frame_path(source_path, str(frame["file_path"]))
    orig_w = int(frame.get("w", 0))
    orig_h = int(frame.get("h", 0))
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError(f"Frame is missing valid w/h: {frame}")
    width, height = compute_scaled_resolution(orig_w, orig_h, resolution_arg=resolution_arg)
    sx = float(width) / float(orig_w)
    sy = float(height) / float(orig_h)

    fx = float(frame.get("fl_x", 0.0)) * sx
    fy = float(frame.get("fl_y", 0.0)) * sy
    if fx <= 0.0 or fy <= 0.0:
        camera_angle_x = float(frame.get("camera_angle_x", 0.0))
        if camera_angle_x <= 0.0:
            raise ValueError(f"Frame is missing focal length and camera_angle_x: {frame}")
        fx = 0.5 * float(width) / np.tan(0.5 * camera_angle_x)
        fy = fx
    cx = float(frame.get("cx", orig_w / 2.0)) * sx
    cy = float(frame.get("cy", orig_h / 2.0)) * sy

    c2w = np.asarray(frame["transform_matrix"], dtype=np.float64).copy()
    if c2w.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform_matrix, got {c2w.shape}")
    # Match scene.dataset_readers.readCamerasFromTransforms: official transforms
    # use NeRF/OpenGL camera axes, while depth projection expects COLMAP-style
    # camera coordinates with +Z forward.
    c2w[:3, 1:3] *= -1.0

    return {
        "view_id": f"{idx:05d}",
        "image_name": Path(file_path).stem,
        "file_path": file_path,
        "width": int(width),
        "height": int(height),
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "camera_to_world": c2w.tolist(),
    }


def _load_views(source_path: Path, transforms_name: str, resolution_arg: int, max_views: int | None) -> List[Dict[str, Any]]:
    payload = load_json(source_path / transforms_name)
    frames = list(payload.get("frames", []))
    if max_views is not None:
        frames = frames[: int(max_views)]
    return [_view_from_frame(source_path, frame, idx, resolution_arg) for idx, frame in enumerate(frames)]


def main() -> None:
    args = _argparser().parse_args()
    source_path = Path(args.source_path).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_json = source_path / "transforms_train.json"
    test_json = source_path / "transforms_test.json"
    points_path = source_path / "points3d.ply"
    if not train_json.exists() or not test_json.exists() or not points_path.exists():
        raise FileNotFoundError(
            f"Expected transforms_train.json, transforms_test.json, and points3d.ply under {source_path}"
        )

    scene_name = str(args.scene_name).strip() or source_path.parent.name or source_path.name
    thresholds_m = parse_thresholds_m(args.thresholds_m)
    points = load_ply_points_xyz(points_path)
    roi = compute_quantile_bbox(
        points,
        lower_quantile=float(args.bbox_lower_quantile),
        upper_quantile=float(args.bbox_upper_quantile),
        padding_ratio_of_robust_diagonal=float(args.bbox_padding_ratio),
    )
    bbox_min = np.asarray(roi["bbox_min"], dtype=np.float64)
    bbox_max = np.asarray(roi["bbox_max"], dtype=np.float64)

    train_views = _load_views(source_path, "transforms_train.json", int(args.resolution_arg), None)
    probe_views = _load_views(source_path, "transforms_test.json", int(args.resolution_arg), args.max_views)

    save_json(
        out_dir / "reference_roi.json",
        {
            "protocol_name": "reference-depth-based-geometric-evaluation-v1",
            "scene_name": scene_name,
            "roi_rule": {
                "type": "official_transforms_pointcloud_quantile_aabb",
                "lower_quantile": float(args.bbox_lower_quantile),
                "upper_quantile": float(args.bbox_upper_quantile),
                "padding_ratio_of_robust_diagonal": float(args.bbox_padding_ratio),
            },
            "bbox_min": bbox_min.tolist(),
            "bbox_max": bbox_max.tolist(),
            "scene_diagonal": float(roi["scene_diagonal"]),
            "source_points_path": str(points_path),
        },
    )

    views_dir = out_dir / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    manifest_views: List[Dict[str, Any]] = []
    for view in probe_views:
        depth, support_count = render_point_splat_depth_for_view(
            points,
            view,
            depth_tolerance_m=float(args.support_depth_tolerance_m),
            splat_radius_px=int(args.splat_radius_px),
        )
        valid_mask = np.isfinite(depth) & (depth > 0.0)
        if np.any(valid_mask):
            inside_roi = compute_inside_bbox_mask(depth, view, bbox_min, bbox_max)
            valid_mask = valid_mask & inside_roi
        valid_mask = valid_mask & (support_count >= int(args.support_min_count))
        rel = Path("views") / f"{view['view_id']}.npz"
        np.savez_compressed(
            out_dir / rel,
            depth=np.asarray(depth, dtype=np.float64),
            valid_mask=np.asarray(valid_mask, dtype=np.uint8),
            support_count=np.asarray(support_count, dtype=np.int32),
        )
        manifest_views.append(
            {
                "view_id": str(view["view_id"]),
                "image_name": str(view["image_name"]),
                "width": int(view["width"]),
                "height": int(view["height"]),
                "fx": float(view["fx"]),
                "fy": float(view["fy"]),
                "cx": float(view["cx"]),
                "cy": float(view["cy"]),
                "camera_to_world": view["camera_to_world"],
                "npz_file": str(rel).replace("\\", "/"),
            }
        )

    save_json(
        out_dir / "probe_camera_manifest.json",
        {
            "camera_manifest_type": "official_transforms_probe_camera_manifest_v1",
            "scene_name": scene_name,
            "source_path": str(source_path),
            "resolution_arg": int(args.resolution_arg),
            "train_transforms": str(train_json),
            "test_transforms": str(test_json),
            "train_view_count": len(train_views),
            "probe_view_count": len(probe_views),
            "views": manifest_views,
        },
    )
    save_json(
        out_dir / "reference_depth_manifest.json",
        {
            "protocol_name": "reference-depth-based-geometric-evaluation-v1",
            "reference_variant": "official_transforms_point_splat_smoke_v1",
            "scene_name": scene_name,
            "reference_workspace_root": str(source_path),
            "reference_fused_ply": str(points_path),
            "reference_mesh_path": "",
            "reference_mesh_backend": "point_splat_smoke",
            "roi_path": str(out_dir / "reference_roi.json"),
            "camera_manifest_path": str(out_dir / "probe_camera_manifest.json"),
            "depth_semantics": "point_splat_camera_z_reference",
            "distance_unit": str(args.distance_unit),
            "scale_mode": str(args.scale_mode),
            "thresholds_m": thresholds_m,
            "support_rule": {
                "type": "official_transforms_point_splat",
                "min_support_count": int(args.support_min_count),
                "splat_radius_px": int(args.splat_radius_px),
                "support_depth_tolerance_m": float(args.support_depth_tolerance_m),
            },
            "views": manifest_views,
            "notes": (
                "Smoke-only official aligned reference built from the prepared points3d.ply and transforms_test.json. "
                "It validates probe/depth/evaluator plumbing and should not be described as dense MVS ground truth."
            ),
        },
    )
    save_json(
        out_dir / "reference_build_manifest.json",
        {
            "scene_name": scene_name,
            "source_path": str(source_path),
            "points_path": str(points_path),
            "train_view_count": len(train_views),
            "probe_view_count": len(probe_views),
            "resolution_arg": int(args.resolution_arg),
            "thresholds_m": thresholds_m,
            "distance_unit": str(args.distance_unit),
            "scale_mode": str(args.scale_mode),
            "reference_depth_manifest": str(out_dir / "reference_depth_manifest.json"),
            "roi_path": str(out_dir / "reference_roi.json"),
            "camera_manifest_path": str(out_dir / "probe_camera_manifest.json"),
        },
    )
    print(f"OFFICIAL_TRANSFORMS_REFERENCE {out_dir / 'reference_depth_manifest.json'}")


if __name__ == "__main__":
    main()
