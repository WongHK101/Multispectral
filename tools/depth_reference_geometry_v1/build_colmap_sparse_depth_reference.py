from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from depth_reference_common import (
    build_probe_view_manifest,
    compute_inside_bbox_mask,
    compute_quantile_bbox,
    load_json,
    load_ply_points_xyz,
    parse_thresholds_m,
    render_point_splat_depth_for_view,
    save_json,
)


def _argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a lightweight point-splat reference depth artifact from an "
            "existing COLMAP sparse point cloud. This is a diagnostic reference, "
            "not dense MVS ground truth."
        )
    )
    parser.add_argument("--strict_protocol_manifest", required=True)
    parser.add_argument("--out_dir", required=True)
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
    return parser


def _resolve_points_path(artifacts: Dict[str, Any]) -> Path:
    candidates: List[Path] = []
    for key in ("roi_source_points_path", "sparse_points_path", "reference_points_path"):
        value = str(artifacts.get(key, "")).strip()
        if value:
            candidates.append(Path(value))
    for key in ("registered_model_dir", "shared_frame_model_dir"):
        value = str(artifacts.get(key, "")).strip()
        if value:
            root = Path(value)
            candidates.extend([root / "points3D.ply", root / "points3d.ply"])
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(
        "Could not find a sparse point cloud. Tried manifest artifacts "
        "roi_source_points_path/sparse_points_path/reference_points_path and "
        "registered_model_dir/shared_frame_model_dir points3D.ply."
    )


def main() -> None:
    args = _argparser().parse_args()
    strict_manifest_path = Path(args.strict_protocol_manifest).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    strict = load_json(strict_manifest_path)
    scene_name = str(strict["scene_name"])
    artifacts = strict["artifacts"]
    lists = strict["lists"]

    strict_probe_root = Path(artifacts["strict_thermal_root"]).resolve()
    train_union_list = Path(lists["train_union"]).resolve()
    probe_list = Path(lists["probe_test"]).resolve()
    points_path = _resolve_points_path(artifacts)
    points = load_ply_points_xyz(points_path)
    if points.shape[0] <= 0:
        raise RuntimeError(f"Sparse point cloud is empty: {points_path}")

    roi = compute_quantile_bbox(
        points,
        lower_quantile=float(args.bbox_lower_quantile),
        upper_quantile=float(args.bbox_upper_quantile),
        padding_ratio_of_robust_diagonal=float(args.bbox_padding_ratio),
    )
    bbox_min = np.asarray(roi["bbox_min"], dtype=np.float64)
    bbox_max = np.asarray(roi["bbox_max"], dtype=np.float64)
    roi_path = out_dir / "reference_roi.json"
    save_json(
        roi_path,
        {
            "protocol_name": "reference-depth-based-geometric-evaluation-v1",
            "scene_name": scene_name,
            "roi_rule": {
                "type": "colmap_sparse_quantile_aabb",
                "lower_quantile": float(args.bbox_lower_quantile),
                "upper_quantile": float(args.bbox_upper_quantile),
                "padding_ratio_of_robust_diagonal": float(args.bbox_padding_ratio),
            },
            "bbox_min": bbox_min.tolist(),
            "bbox_max": bbox_max.tolist(),
            "scene_diagonal": float(roi["scene_diagonal"]),
            "source_points_path": str(points_path),
            "source_point_count": int(points.shape[0]),
        },
    )

    camera_manifest = build_probe_view_manifest(
        source_path=strict_probe_root,
        images_dir_name="images",
        resolution_arg=int(args.resolution_arg),
        train_list=train_union_list,
        test_list=probe_list,
        scene_name=scene_name,
    )
    camera_manifest_path = out_dir / "probe_camera_manifest.json"
    save_json(camera_manifest_path, camera_manifest)

    views_dir = out_dir / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    manifest_views: List[Dict[str, Any]] = []
    for view in camera_manifest["views"]:
        depth, support_count = render_point_splat_depth_for_view(
            points,
            view,
            depth_tolerance_m=float(args.support_depth_tolerance_m),
            splat_radius_px=int(args.splat_radius_px),
        )
        finite = np.isfinite(depth) & (depth > 0.0)
        inside_roi = (
            compute_inside_bbox_mask(depth, view, bbox_min=bbox_min, bbox_max=bbox_max)
            if np.any(finite)
            else np.zeros_like(finite, dtype=bool)
        )
        valid_mask = finite & inside_roi & (support_count >= int(args.support_min_count))

        view_rel = Path("views") / f"{view['view_id']}.npz"
        np.savez_compressed(
            out_dir / view_rel,
            depth=np.asarray(depth, dtype=np.float64),
            support_count=np.asarray(support_count, dtype=np.int32),
            valid_mask=np.asarray(valid_mask, dtype=np.uint8),
            inside_roi=np.asarray(inside_roi, dtype=np.uint8),
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
                "npz_file": str(view_rel).replace("\\", "/"),
            }
        )

    thresholds_m = parse_thresholds_m(args.thresholds_m)
    ref_manifest_path = out_dir / "reference_depth_manifest.json"
    save_json(
        ref_manifest_path,
        {
            "protocol_name": "reference-depth-based-geometric-evaluation-v1",
            "reference_variant": "colmap_sparse_point_splat_v1",
            "scene_name": scene_name,
            "strict_protocol_manifest": str(strict_manifest_path),
            "camera_manifest_path": str(camera_manifest_path),
            "reference_workspace_root": str(strict_probe_root),
            "reference_fused_ply": str(points_path),
            "reference_mesh_path": None,
            "reference_mesh_backend": "colmap_sparse_point_splat",
            "reference_depth_backend": "point_splat",
            "roi_path": str(roi_path),
            "depth_semantics": "metric_camera_z_reference_sparse_point_splat",
            "distance_unit": str(args.distance_unit),
            "scale_mode": str(args.scale_mode),
            "thresholds_m": thresholds_m,
            "support_rule": {
                "type": "colmap_sparse_point_splat",
                "min_support_count": int(args.support_min_count),
                "splat_radius_px": int(args.splat_radius_px),
                "support_depth_tolerance_m": float(args.support_depth_tolerance_m),
            },
            "views": manifest_views,
            "notes": (
                "Diagnostic reference built from the training-side COLMAP sparse point cloud. "
                "It should not be described as dense MVS ground truth."
            ),
        },
    )
    save_json(
        out_dir / "reference_build_manifest.json",
        {
            "scene_name": scene_name,
            "strict_protocol_manifest": str(strict_manifest_path),
            "source_points_path": str(points_path),
            "source_point_count": int(points.shape[0]),
            "strict_probe_root": str(strict_probe_root),
            "train_union_list": str(train_union_list),
            "probe_list": str(probe_list),
            "reference_depth_manifest": str(ref_manifest_path),
            "roi_path": str(roi_path),
            "camera_manifest_path": str(camera_manifest_path),
            "thresholds_m": thresholds_m,
            "distance_unit": str(args.distance_unit),
            "scale_mode": str(args.scale_mode),
        },
    )
    print(f"COLMAP_SPARSE_REFERENCE {ref_manifest_path}")


if __name__ == "__main__":
    main()
