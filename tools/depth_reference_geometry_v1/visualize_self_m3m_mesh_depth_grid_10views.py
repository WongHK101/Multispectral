from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.depth_reference_geometry_v1.depth_reference_common import (
    compute_inside_bbox_mask,
    estimate_strict_to_native_world_transform,
    image_stem_key,
    load_json,
    load_native_camera_entries,
    load_ply_mesh,
    render_mesh_depth_for_view,
    save_json,
)


METHOD_DISPLAY = {
    "From-scratch": "From",
    "Geometry-unfrozen": "Unfrozen",
    "MMS retained-self": "MMS",
    "MMS_retained-self": "MMS",
    "UMGS-J": "UMGS-J",
    "UMGS-I": "UMGS-I",
}


def _frame_number(image_name: str) -> int | None:
    match = re.search(r"_(\d{4})_", str(image_name))
    return int(match.group(1)) if match else None


def _even_indices(n: int, k: int) -> List[int]:
    if n <= 0:
        return []
    if n <= k:
        return list(range(n))
    return [int(round(x)) for x in np.linspace(0, n - 1, k)]


def _camera_to_world_from_native_entry(
    native_entry: Dict[str, Any],
    *,
    strict_to_native_alignment: Dict[str, Any],
    out_width: int,
    out_height: int,
) -> Dict[str, Any]:
    native_center = np.asarray(native_entry["position"], dtype=np.float64)
    native_rot = np.asarray(native_entry["rotation"], dtype=np.float64)
    strict_to_native = np.asarray(strict_to_native_alignment["strict_to_native_transform"], dtype=np.float64)
    linear = strict_to_native[:3, :3]
    trans = strict_to_native[:3, 3]
    scale = float(strict_to_native_alignment.get("similarity_scale_strict_to_native", 1.0))
    if scale <= 0.0:
        raise ValueError(f"Invalid similarity scale: {scale}")
    strict_to_native_rot = linear / scale
    strict_center = np.linalg.inv(linear) @ (native_center - trans)
    strict_rot = strict_to_native_rot.T @ native_rot
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = strict_rot
    c2w[:3, 3] = strict_center

    native_w = int(native_entry["width"])
    native_h = int(native_entry["height"])
    scale_x = float(out_width) / float(native_w)
    scale_y = float(out_height) / float(native_h)
    return {
        "width": int(out_width),
        "height": int(out_height),
        "fx": float(native_entry["fx"]) * scale_x,
        "fy": float(native_entry["fy"]) * scale_y,
        "cx": float(out_width) / 2.0,
        "cy": float(out_height) / 2.0,
        "camera_to_world": c2w.tolist(),
    }


def _select_native_views(
    *,
    native_cameras_by_stem: Dict[str, Dict[str, Any]],
    selection_mode: str,
    target_count: int,
    reference_candidate_image_dir: Path | None,
) -> List[Dict[str, Any]]:
    entries = list(native_cameras_by_stem.values())
    entries = [item for item in entries if _frame_number(str(item.get("img_name", ""))) is not None]
    if selection_mode == "original_holdout_every8":
        candidates = [
            item
            for item in entries
            if (_frame_number(str(item["img_name"])) - 1) % 8 == 0
        ]
    elif selection_mode == "reference_span_even":
        if reference_candidate_image_dir is None:
            raise ValueError("--reference_candidate_image_dir is required for reference_span_even")
        available = {path.name for path in reference_candidate_image_dir.glob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"}}
        candidates = [item for item in entries if str(item["img_name"]) in available]
    else:
        raise ValueError(f"Unsupported selection_mode: {selection_mode}")
    candidates = sorted(candidates, key=lambda item: _frame_number(str(item["img_name"])))
    if len(candidates) < target_count:
        raise ValueError(
            f"Selection mode {selection_mode!r} only found {len(candidates)} candidates, "
            f"but {target_count} are required."
        )
    return [candidates[i] for i in _even_indices(len(candidates), target_count)]


def prepare_reference(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=False if args.fail_if_exists else True)
    base_reference_manifest = load_json(Path(args.base_reference_manifest).resolve())
    base_probe_manifest = load_json(Path(args.base_probe_manifest).resolve())
    native_cameras_by_stem = load_native_camera_entries(Path(args.native_cameras_json).resolve())
    strict_to_native_alignment = estimate_strict_to_native_world_transform(
        strict_views=base_probe_manifest["views"],
        native_cameras_by_stem=native_cameras_by_stem,
        alignment_mode="similarity",
    )
    out_width = int(args.output_width) if args.output_width else int(base_probe_manifest["views"][0]["width"])
    out_height = int(args.output_height) if args.output_height else int(base_probe_manifest["views"][0]["height"])
    selected_native = _select_native_views(
        native_cameras_by_stem=native_cameras_by_stem,
        selection_mode=str(args.selection_mode),
        target_count=int(args.target_count),
        reference_candidate_image_dir=Path(args.reference_candidate_image_dir).resolve()
        if args.reference_candidate_image_dir
        else None,
    )

    views: List[Dict[str, Any]] = []
    selected_records: List[Dict[str, Any]] = []
    for idx, native_entry in enumerate(selected_native):
        image_name = str(native_entry["img_name"])
        camera_record = _camera_to_world_from_native_entry(
            native_entry,
            strict_to_native_alignment=strict_to_native_alignment,
            out_width=out_width,
            out_height=out_height,
        )
        frame_no = _frame_number(image_name)
        is_original_holdout = bool(frame_no is not None and (frame_no - 1) % 8 == 0)
        view = {
            "view_id": f"{idx:05d}",
            "image_name": image_name,
            **camera_record,
        }
        views.append(view)
        selected_records.append(
            {
                "view_id": view["view_id"],
                "image_name": image_name,
                "frame_number": frame_no,
                "selected_index": idx,
                "selection_mode": str(args.selection_mode),
                "is_original_every8_holdout": is_original_holdout,
                "camera_source": str(Path(args.native_cameras_json).resolve()),
                "reference_coordinate_alignment": "similarity_from_base_probe_manifest",
            }
        )

    mesh_path = Path(base_reference_manifest["reference_mesh_path"]).resolve()
    vertices, faces = load_ply_mesh(mesh_path)
    roi_path = Path(base_reference_manifest.get("roi_path", "")).resolve() if base_reference_manifest.get("roi_path") else None
    bbox_min = bbox_max = None
    if roi_path is not None and roi_path.exists():
        roi = load_json(roi_path)
        bbox_min = np.asarray(roi.get("bbox_min", roi.get("roi_min")), dtype=np.float64)
        bbox_max = np.asarray(roi.get("bbox_max", roi.get("roi_max")), dtype=np.float64)
        if bbox_min.shape != (3,) or bbox_max.shape != (3,):
            bbox_min = bbox_max = None

    views_dir = out_dir / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    reference_views: List[Dict[str, Any]] = []
    validity_rows: List[List[Any]] = []
    for idx, view in enumerate(views):
        depth = render_mesh_depth_for_view(vertices, faces, view)
        finite = np.isfinite(depth) & (depth > 0.0)
        inside_roi = np.ones(depth.shape, dtype=bool)
        if bbox_min is not None and bbox_max is not None:
            inside_roi = compute_inside_bbox_mask(depth, view, bbox_min, bbox_max)
        valid_mask = finite & inside_roi
        npz_rel = Path("views") / f"{idx:05d}.npz"
        np.savez_compressed(
            out_dir / npz_rel,
            depth=np.asarray(depth, dtype=np.float64),
            valid_mask=np.asarray(valid_mask, dtype=np.uint8),
            inside_roi=np.asarray(inside_roi, dtype=np.uint8),
        )
        ref_view = dict(view)
        ref_view["npz_file"] = str(npz_rel).replace("\\", "/")
        reference_views.append(ref_view)
        validity_rows.append(
            [
                view["image_name"],
                int(np.count_nonzero(valid_mask)),
                int(valid_mask.size),
                float(np.count_nonzero(valid_mask)) / float(valid_mask.size),
                float(np.nanmin(depth[finite])) if np.any(finite) else "nan",
                float(np.nanmax(depth[finite])) if np.any(finite) else "nan",
            ]
        )

    probe_manifest = {
        "camera_manifest_type": "heldout_probe_camera_manifest_v1",
        "camera_manifest_source": "native_cameras_json_similarity_to_verified_mesh_reference",
        "scene_name": str(args.scene_name),
        "source_path": str(args.rgb_image_dir) if args.rgb_image_dir else "",
        "images_dir_name": "images",
        "resolution_arg": None,
        "selection_mode": str(args.selection_mode),
        "selection_count": int(args.target_count),
        "base_probe_manifest": str(Path(args.base_probe_manifest).resolve()),
        "base_reference_manifest": str(Path(args.base_reference_manifest).resolve()),
        "strict_to_native_alignment": strict_to_native_alignment,
        "views": views,
    }
    reference_manifest = {
        **{k: v for k, v in base_reference_manifest.items() if k != "views"},
        "scene_name": str(args.scene_name),
        "camera_manifest_path": str((out_dir / "probe_camera_manifest.json").resolve()),
        "reference_mesh_path": str(mesh_path),
        "reference_depth_backend": "mesh",
        "depth_semantics": "metric_camera_z_reference_mesh",
        "thresholds_m": [0.005, 0.01, 0.025, 0.05, 0.10, 0.20],
        "views": reference_views,
    }
    save_json(out_dir / "probe_camera_manifest.json", probe_manifest)
    save_json(out_dir / "reference_depth_manifest.json", reference_manifest)
    save_json(
        out_dir / "selected_views.json",
        {
            "scene_name": str(args.scene_name),
            "selection_mode": str(args.selection_mode),
            "target_count": int(args.target_count),
            "all_selected_are_original_every8_holdout": all(r["is_original_every8_holdout"] for r in selected_records),
            "selected_views": selected_records,
        },
    )
    with (out_dir / "reference_validity_stats.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "valid_pixels", "total_pixels", "valid_ratio", "depth_min", "depth_max"])
        writer.writerows(validity_rows)
    print(f"REFERENCE_MANIFEST {out_dir / 'reference_depth_manifest.json'}")


def _load_depth_npz(root: Path, view: Dict[str, Any]) -> Dict[str, np.ndarray]:
    arr = np.load(root / view["npz_file"])
    return {key: np.asarray(arr[key]) for key in arr.files}


def _metric_depth(raw_depth: np.ndarray, semantics: str) -> np.ndarray:
    raw_depth = np.asarray(raw_depth, dtype=np.float64)
    if semantics in {"metric_camera_z_from_renderer", "metric_camera_z_from_point_splat_centers"}:
        return raw_depth
    if semantics == "inverse_camera_z_from_renderer":
        out = np.full(raw_depth.shape, np.nan, dtype=np.float64)
        positive = np.isfinite(raw_depth) & (raw_depth > 0.0)
        out[positive] = 1.0 / raw_depth[positive]
        return out
    raise ValueError(f"Unsupported depth semantics: {semantics!r}")


def _read_rgb_image(image_name: str, rgb_image_dir: Path, size: Tuple[int, int]) -> np.ndarray:
    path = rgb_image_dir / image_name
    if not path.exists():
        raise FileNotFoundError(f"RGB/source image not found for visualization: {path}")
    image = Image.open(path).convert("RGB").resize(size, Image.BILINEAR)
    return np.asarray(image)


def _depth_to_rgb(depth: np.ndarray, valid: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    cmap = plt.get_cmap("viridis")
    norm = np.full(depth.shape, np.nan, dtype=np.float64)
    good = valid & np.isfinite(depth) & (depth > 0.0)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    norm[good] = np.clip((depth[good] - vmin) / (vmax - vmin), 0.0, 1.0)
    rgba = cmap(np.nan_to_num(norm, nan=0.0))
    rgb = (rgba[:, :, :3] * 255.0).astype(np.uint8)
    rgb[~good] = 0
    return rgb


def _error_to_rgb(error: np.ndarray, valid: np.ndarray, vmin: float = -0.20, vmax: float = 0.20) -> np.ndarray:
    cmap = plt.get_cmap("coolwarm")
    good = valid & np.isfinite(error)
    norm = np.zeros(error.shape, dtype=np.float64)
    norm[good] = np.clip((error[good] - vmin) / (vmax - vmin), 0.0, 1.0)
    rgba = cmap(norm)
    rgb = (rgba[:, :, :3] * 255.0).astype(np.uint8)
    rgb[~good] = 0
    return rgb


def visualize(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_manifest_path = Path(args.reference_manifest).resolve()
    ref_manifest = load_json(ref_manifest_path)
    ref_root = ref_manifest_path.parent
    method_specs: List[Tuple[str, Path]] = []
    for spec in args.method_bundle:
        if "=" not in spec:
            raise ValueError(f"--method_bundle must be NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        method_specs.append((name, Path(path).resolve()))
    if len(method_specs) != 5:
        raise ValueError(f"Expected exactly 5 method bundles, got {len(method_specs)}")

    methods: List[Dict[str, Any]] = []
    for name, bundle_root in method_specs:
        split_manifest = load_json(bundle_root / "split_manifest.json")
        adapter_manifest = load_json(bundle_root / "adapter_manifest.json")
        methods.append(
            {
                "name": name,
                "display": METHOD_DISPLAY.get(name, name),
                "root": bundle_root,
                "split_manifest": split_manifest,
                "adapter_manifest": adapter_manifest,
                "views_by_name": {str(v["image_name"]): v for v in split_manifest["views"]},
            }
        )

    rgb_image_dir = Path(args.rgb_image_dir).resolve()
    rows = []
    stats_rows: List[Dict[str, Any]] = []
    for ref_view in ref_manifest["views"]:
        image_name = str(ref_view["image_name"])
        ref_npz = _load_depth_npz(ref_root, ref_view)
        ref_depth = np.asarray(ref_npz["depth"], dtype=np.float64)
        ref_valid = np.asarray(ref_npz.get("valid_mask", np.isfinite(ref_depth) & (ref_depth > 0.0))).astype(bool)
        h, w = ref_depth.shape
        rgb = _read_rgb_image(image_name, rgb_image_dir, (w, h))
        method_depths: List[np.ndarray] = []
        method_valids: List[np.ndarray] = []
        method_errors: List[np.ndarray] = []
        for method in methods:
            model_view = method["views_by_name"].get(image_name)
            if model_view is None:
                raise ValueError(f"{method['name']} bundle is missing {image_name}")
            model_npz = _load_depth_npz(method["root"], model_view)
            model_depth = _metric_depth(np.asarray(model_npz["depth"], dtype=np.float64), str(method["adapter_manifest"]["depth_semantics"]))
            opacity = np.asarray(model_npz["opacity"], dtype=np.float64)
            rule = method["adapter_manifest"].get("validity_rule", {})
            model_valid = (
                np.isfinite(model_depth)
                & (model_depth > float(rule.get("depth_min", 1e-6)))
                & np.isfinite(opacity)
                & (opacity >= float(rule.get("opacity_threshold", 0.5)))
            )
            valid = ref_valid & model_valid & np.isfinite(ref_depth) & (ref_depth > float(args.relative_depth_min))
            error = np.full(ref_depth.shape, np.nan, dtype=np.float64)
            error[valid] = (model_depth[valid] - ref_depth[valid]) / ref_depth[valid]
            method_depths.append(model_depth)
            method_valids.append(valid)
            method_errors.append(error)
            absrel = float(np.nanmean(np.abs(error[valid]))) if np.any(valid) else math.nan
            stats_rows.append(
                {
                    "image_name": image_name,
                    "method": method["name"],
                    "valid_pixels": int(np.count_nonzero(valid)),
                    "ref_valid_pixels": int(np.count_nonzero(ref_valid)),
                    "valid_ratio_on_ref": float(np.count_nonzero(valid)) / float(max(1, np.count_nonzero(ref_valid))),
                    "AbsRel": absrel,
                    "Agree@1%": float(np.count_nonzero(valid & (np.abs(error) <= 0.01))) / float(max(1, np.count_nonzero(ref_valid))),
                    "Agree@5%": float(np.count_nonzero(valid & (np.abs(error) <= 0.05))) / float(max(1, np.count_nonzero(ref_valid))),
                    "Agree@10%": float(np.count_nonzero(valid & (np.abs(error) <= 0.10))) / float(max(1, np.count_nonzero(ref_valid))),
                }
            )
        depth_values = [ref_depth[ref_valid & np.isfinite(ref_depth)]]
        for depth, valid in zip(method_depths, method_valids):
            depth_values.append(depth[valid & np.isfinite(depth)])
        concat = np.concatenate([x.reshape(-1) for x in depth_values if x.size > 0])
        if concat.size == 0:
            depth_vmin, depth_vmax = 0.0, 1.0
        else:
            depth_vmin, depth_vmax = np.percentile(concat, [2.0, 98.0])
        panels = [rgb, _depth_to_rgb(ref_depth, ref_valid, float(depth_vmin), float(depth_vmax))]
        for depth, valid, error in zip(method_depths, method_valids, method_errors):
            panels.append(_depth_to_rgb(depth, valid, float(depth_vmin), float(depth_vmax)))
            panels.append(_error_to_rgb(error, valid))
        rows.append(
            {
                "image_name": image_name,
                "panels": panels,
                "depth_vmin": float(depth_vmin),
                "depth_vmax": float(depth_vmax),
            }
        )

    fig_w = 24.0
    fig_h = max(11.1, 1.28 * len(rows) + 0.9)
    fig, axes = plt.subplots(len(rows), 12, figsize=(fig_w, fig_h), dpi=200)
    col_titles = ["RGB", "Mesh"]
    for method in methods:
        col_titles.extend([method["display"], "Rel. err"])
    for r_idx, row in enumerate(rows):
        for c_idx, panel in enumerate(row["panels"]):
            ax = axes[r_idx, c_idx] if len(rows) > 1 else axes[c_idx]
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            if r_idx == 0:
                ax.set_title(col_titles[c_idx], fontsize=8)
            if c_idx == 0:
                ax.set_ylabel(Path(row["image_name"]).stem.split("_")[-2], fontsize=7)
            if c_idx == 1:
                ax.text(
                    0.02,
                    0.96,
                    f"{row['depth_vmin']:.2f}-{row['depth_vmax']:.2f} m",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=5.5,
                    color="white",
                    bbox={"facecolor": "black", "alpha": 0.55, "pad": 1.2, "edgecolor": "none"},
                )
    fig.subplots_adjust(wspace=0.02, hspace=0.04, left=0.025, right=0.995, top=0.94, bottom=0.135)
    depth_cax = fig.add_axes([0.18, 0.065, 0.28, 0.018])
    depth_sm = plt.cm.ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap="viridis")
    depth_cb = fig.colorbar(depth_sm, cax=depth_cax, orientation="horizontal")
    depth_cb.set_ticks([0.0, 1.0])
    depth_cb.set_ticklabels(["near", "far"])
    depth_cb.ax.tick_params(labelsize=7)
    depth_cb.set_label(
        "Depth color: near to far. Per-row 2-98% range is printed on Mesh.",
        fontsize=7,
    )

    err_cax = fig.add_axes([0.57, 0.065, 0.28, 0.018])
    err_sm = plt.cm.ScalarMappable(norm=Normalize(vmin=-0.20, vmax=0.20), cmap="coolwarm")
    err_cb = fig.colorbar(err_sm, cax=err_cax, orientation="horizontal")
    err_cb.set_ticks([-0.20, -0.10, 0.0, 0.10, 0.20])
    err_cb.set_ticklabels(["-20%", "-10%", "0", "+10%", "+20%"])
    err_cb.ax.tick_params(labelsize=7)
    err_cb.set_label(
        "Relative error: blue = closer/front, red = farther/behind.",
        fontsize=7,
    )

    fig.text(
        0.5,
        0.026,
        "Error = (D_method - D_mesh) / D_mesh; maps are clipped to +/-20%. Black pixels are invalid or outside mesh/method support.",
        ha="center",
        va="center",
        fontsize=7,
    )
    png_path = out_dir / "depth_visual_grid_10views_core5_with_legend.png"
    pdf_path = out_dir / "depth_visual_grid_10views_core5_with_legend.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)

    stats_path = out_dir / "depth_visual_stats.csv"
    with stats_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["image_name", "method", "valid_pixels", "ref_valid_pixels", "valid_ratio_on_ref", "AbsRel", "Agree@1%", "Agree@5%", "Agree@10%"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats_rows)
    save_json(
        out_dir / "depth_visual_manifest.json",
        {
            "reference_manifest": str(ref_manifest_path),
            "rgb_image_dir": str(rgb_image_dir),
            "method_bundles": [{"method": name, "bundle_root": str(path)} for name, path in method_specs],
            "error_definition": "(D_method - D_mesh) / D_mesh",
            "valid_mask": "mesh_valid & method_valid & D_mesh > relative_depth_min",
            "depth_colormap": "viridis per-view 2-98 percentile over mesh and all methods",
            "error_colormap": "coolwarm fixed [-0.20, +0.20]",
            "legend": {
                "depth": "Viridis, row-normalized to per-view 2-98 percentile; numeric depth range printed on Mesh panel.",
                "relative_error": "Coolwarm fixed to [-0.20,+0.20]; blue means method depth is closer than mesh, red means farther, black means invalid.",
            },
            "outputs": {
                "png": str(png_path),
                "pdf": str(pdf_path),
                "stats_csv": str(stats_path),
            },
        },
    )
    print(f"VISUAL_GRID {png_path}")


def _argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and visualize self_m3m 10-view mesh-reference depth comparisons")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare_reference")
    p.add_argument("--base_reference_manifest", required=True)
    p.add_argument("--base_probe_manifest", required=True)
    p.add_argument("--native_cameras_json", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--scene_name", default="self_m3m_mesh_depth_visual_10views")
    p.add_argument("--selection_mode", default="original_holdout_every8", choices=("original_holdout_every8", "reference_span_even"))
    p.add_argument("--target_count", type=int, default=10)
    p.add_argument("--reference_candidate_image_dir", default="")
    p.add_argument("--rgb_image_dir", default="")
    p.add_argument("--output_width", type=int, default=0)
    p.add_argument("--output_height", type=int, default=0)
    p.add_argument("--fail_if_exists", action="store_true")
    p.set_defaults(func=prepare_reference)

    v = sub.add_parser("visualize")
    v.add_argument("--reference_manifest", required=True)
    v.add_argument("--rgb_image_dir", required=True)
    v.add_argument("--out_dir", required=True)
    v.add_argument("--method_bundle", action="append", required=True)
    v.add_argument("--relative_depth_min", type=float, default=1e-6)
    v.set_defaults(func=visualize)
    return parser


def main() -> None:
    args = _argparser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
