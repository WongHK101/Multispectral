from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
        f.write("\n")


def _raw_depth_to_metric_camera_z(raw_depth: np.ndarray, depth_semantics: str) -> np.ndarray:
    raw_depth = np.asarray(raw_depth, dtype=np.float64)
    if depth_semantics == "metric_camera_z_from_renderer":
        return raw_depth
    if depth_semantics == "inverse_camera_z_from_renderer":
        metric = np.full(raw_depth.shape, np.nan, dtype=np.float64)
        positive = np.isfinite(raw_depth) & (raw_depth > 0.0)
        metric[positive] = 1.0 / raw_depth[positive]
        return metric
    raise ValueError(f"Unsupported depth semantics: {depth_semantics!r}")


def _parse_thresholds(text: str) -> List[float]:
    values = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values:
        raise ValueError("At least one threshold is required")
    if any(x <= 0.0 for x in values):
        raise ValueError(f"Thresholds must be positive, got {values!r}")
    return values


def _argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a proxy reference-depth manifest from an exported Gaussian bundle. "
            "This is for integration smoke tests only, not paper-facing geometry results."
        )
    )
    parser.add_argument("--bundle_manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--scene_name", default="")
    parser.add_argument("--reference_label", default="gaussian_proxy_reference_smoke")
    parser.add_argument("--thresholds", default="0.01,0.05,0.10")
    parser.add_argument("--opacity_threshold", type=float, default=0.5)
    parser.add_argument("--depth_min", type=float, default=1e-6)
    parser.add_argument("--max_views", type=int, default=None)
    return parser


def main() -> None:
    args = _argparser().parse_args()
    bundle_manifest_path = Path(args.bundle_manifest).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = _load_json(bundle_manifest_path)
    depth_semantics = str(bundle.get("depth_semantics", "inverse_camera_z_from_renderer"))
    scene_name = str(args.scene_name or bundle.get("scene_name", ""))
    views = list(bundle.get("views", []))
    if args.max_views is not None:
        views = views[: int(args.max_views)]

    manifest_views: List[Dict[str, Any]] = []
    for view in views:
        src_npz = bundle_manifest_path.parent / str(view["npz_file"])
        arrays = np.load(src_npz)
        metric_depth = _raw_depth_to_metric_camera_z(arrays["depth"], depth_semantics)
        opacity = np.asarray(arrays["opacity"], dtype=np.float64)
        valid_mask = (
            np.isfinite(metric_depth)
            & np.isfinite(opacity)
            & (metric_depth > float(args.depth_min))
            & (opacity >= float(args.opacity_threshold))
        )
        rel = Path("views") / f"{str(view['view_id'])}.npz"
        dst_npz = out_dir / rel
        dst_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dst_npz,
            depth=np.asarray(metric_depth, dtype=np.float64),
            valid_mask=np.asarray(valid_mask, dtype=np.uint8),
            proxy_opacity=np.asarray(opacity, dtype=np.float64),
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

    manifest_path = out_dir / "reference_depth_manifest.json"
    _save_json(
        manifest_path,
        {
            "protocol_name": "reference-depth-based-geometric-evaluation-smoke-proxy",
            "paper_facing": False,
            "scene_name": scene_name,
            "reference_label": str(args.reference_label),
            "source_bundle_manifest": str(bundle_manifest_path),
            "depth_semantics": "metric_camera_z_proxy_from_gaussian_bundle",
            "distance_unit": "scene_world_unit",
            "thresholds_m": _parse_thresholds(args.thresholds),
            "validity_rule": {
                "mode": "opacity_threshold",
                "opacity_threshold": float(args.opacity_threshold),
                "depth_min": float(args.depth_min),
            },
            "views": manifest_views,
        },
    )
    print(f"SMOKE_REFERENCE_DEPTH_MANIFEST {manifest_path}")


if __name__ == "__main__":
    main()
