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
    "From-scratch": "Band-only",
    "Geometry-unfrozen": "Unfrozen",
    "MMS retained-self": "MS-Splatting",
    "MMS_retained-self": "MS-Splatting",
    "UMGS-J": "UMGS-J",
    "UMGS-I": "UMGS-I",
}

DEFAULT_BANDS = ("RGB", "G", "R", "RE", "NIR")


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


def _placeholder_panel(height: int, width: int) -> np.ndarray:
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:, :, :] = 10
    return panel


def _compute_reference_depth_ranges(
    ref_manifest_path: Path,
    ref_manifest: Dict[str, Any],
    *,
    q_low: float = 2.0,
    q_high: float = 98.0,
) -> Dict[str, Dict[str, Any]]:
    ref_root = ref_manifest_path.parent
    ranges: Dict[str, Dict[str, Any]] = {}
    for ref_view in ref_manifest["views"]:
        image_name = str(ref_view["image_name"])
        ref_npz = _load_depth_npz(ref_root, ref_view)
        ref_depth = np.asarray(ref_npz["depth"], dtype=np.float64)
        ref_valid = np.asarray(ref_npz.get("valid_mask", np.isfinite(ref_depth) & (ref_depth > 0.0))).astype(bool)
        values = ref_depth[ref_valid & np.isfinite(ref_depth)]
        if values.size == 0:
            depth_vmin, depth_vmax = 0.0, 1.0
        else:
            depth_vmin, depth_vmax = np.percentile(values, [q_low, q_high])
            if float(depth_vmax) <= float(depth_vmin):
                depth_vmax = float(depth_vmin) + 1e-6
        ranges[image_name] = {
            "depth_vmin": float(depth_vmin),
            "depth_vmax": float(depth_vmax),
            "source": "reference_mesh_depth",
            "percentile_low": float(q_low),
            "percentile_high": float(q_high),
            "valid_pixels": int(values.size),
        }
    return ranges


def _render_single_band_grid(
    *,
    out_dir: Path,
    ref_manifest_path: Path,
    ref_manifest: Dict[str, Any],
    methods: Sequence[Dict[str, Any]],
    rgb_image_dir: Path,
    relative_depth_min: float,
    grid_prefix: str,
    stats_filename: str,
    manifest_filename: str,
    wspace: float,
    hspace: float,
    band: str,
    mms_rgb_substitute_band: str,
    depth_range_source: str,
    shared_depth_ranges: Dict[str, Dict[str, Any]] | None,
    show_mesh_range_text: bool,
    title_fontsize: float,
    label_fontsize: float,
    note_fontsize: float,
    fig_width: float,
    row_height: float,
    bottom_margin: float,
    panel_gap_px: int,
) -> Dict[str, Any]:
    ref_root = ref_manifest_path.parent
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
        method_depth_annotations: List[str | None] = []
        method_error_annotations: List[str | None] = []

        for method in methods:
            if not method["available"]:
                model_depth = np.full(ref_depth.shape, np.nan, dtype=np.float64)
                valid = np.zeros(ref_depth.shape, dtype=bool)
                error = np.full(ref_depth.shape, np.nan, dtype=np.float64)
                method_depths.append(model_depth)
                method_valids.append(valid)
                method_errors.append(error)
                method_depth_annotations.append("NO MODEL")
                method_error_annotations.append("NO DATA")
                stats_rows.append(
                    {
                        "band": band,
                        "image_name": image_name,
                        "method": method["name"],
                        "method_source": method.get("source_tag", ""),
                        "available": False,
                        "missing_reason": method.get("missing_reason", "missing_model_bundle"),
                        "valid_pixels": 0,
                        "ref_valid_pixels": int(np.count_nonzero(ref_valid)),
                        "valid_ratio_on_ref": 0.0,
                        "AbsRel": math.nan,
                        "Agree@1%": 0.0,
                        "Agree@5%": 0.0,
                        "Agree@10%": 0.0,
                    }
                )
                continue

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
            valid = ref_valid & model_valid & np.isfinite(ref_depth) & (ref_depth > float(relative_depth_min))
            error = np.full(ref_depth.shape, np.nan, dtype=np.float64)
            error[valid] = (model_depth[valid] - ref_depth[valid]) / ref_depth[valid]
            method_depths.append(model_depth)
            method_valids.append(valid)
            method_errors.append(error)
            method_depth_annotations.append(None)
            method_error_annotations.append(None)
            absrel = float(np.nanmean(np.abs(error[valid]))) if np.any(valid) else math.nan
            stats_rows.append(
                {
                    "band": band,
                    "image_name": image_name,
                    "method": method["name"],
                    "method_source": method.get("source_tag", ""),
                    "available": True,
                    "missing_reason": "",
                    "valid_pixels": int(np.count_nonzero(valid)),
                    "ref_valid_pixels": int(np.count_nonzero(ref_valid)),
                    "valid_ratio_on_ref": float(np.count_nonzero(valid)) / float(max(1, np.count_nonzero(ref_valid))),
                    "AbsRel": absrel,
                    "Agree@1%": float(np.count_nonzero(valid & (np.abs(error) <= 0.01))) / float(max(1, np.count_nonzero(ref_valid))),
                    "Agree@5%": float(np.count_nonzero(valid & (np.abs(error) <= 0.05))) / float(max(1, np.count_nonzero(ref_valid))),
                    "Agree@10%": float(np.count_nonzero(valid & (np.abs(error) <= 0.10))) / float(max(1, np.count_nonzero(ref_valid))),
                }
            )

        if shared_depth_ranges is not None and image_name in shared_depth_ranges:
            depth_range_record = shared_depth_ranges[image_name]
            depth_vmin = float(depth_range_record["depth_vmin"])
            depth_vmax = float(depth_range_record["depth_vmax"])
            depth_range_note = str(depth_range_record.get("source", depth_range_source))
        else:
            depth_values = [ref_depth[ref_valid & np.isfinite(ref_depth)]]
            if depth_range_source == "mesh_and_methods":
                for depth, valid in zip(method_depths, method_valids):
                    depth_values.append(depth[valid & np.isfinite(depth)])
            usable_depth_arrays = [x.reshape(-1) for x in depth_values if x.size > 0]
            if not usable_depth_arrays:
                depth_vmin, depth_vmax = 0.0, 1.0
            else:
                concat = np.concatenate(usable_depth_arrays)
                depth_vmin, depth_vmax = np.percentile(concat, [2.0, 98.0])
            depth_range_note = depth_range_source

        panels = [rgb, _depth_to_rgb(ref_depth, ref_valid, float(depth_vmin), float(depth_vmax))]
        panel_annotations: List[str | None] = [None, None]
        for depth, valid, error, depth_note, err_note in zip(
            method_depths,
            method_valids,
            method_errors,
            method_depth_annotations,
            method_error_annotations,
        ):
            if depth_note is not None:
                panels.append(_placeholder_panel(h, w))
                panel_annotations.append(depth_note)
            else:
                panels.append(_depth_to_rgb(depth, valid, float(depth_vmin), float(depth_vmax)))
                panel_annotations.append(None)
            if err_note is not None:
                panels.append(_placeholder_panel(h, w))
                panel_annotations.append(err_note)
            else:
                panels.append(_error_to_rgb(error, valid))
                panel_annotations.append(None)

        rows.append(
            {
                "image_name": image_name,
                "panels": panels,
                "panel_annotations": panel_annotations,
                "depth_vmin": float(depth_vmin),
                "depth_vmax": float(depth_vmax),
                "depth_range_source": depth_range_note,
            }
        )

    fig_w = float(fig_width)
    dpi = 200
    n_rows = len(rows)
    n_cols = 12
    first_panel = rows[0]["panels"][0]
    panel_src_h, panel_src_w = first_panel.shape[:2]
    exact_gap_mode = int(panel_gap_px) > 0
    if exact_gap_mode:
        gap_px = int(panel_gap_px)
        fig_w_px = int(round(fig_w * dpi))
        left_px = 76
        right_px = 24
        top_px = 78
        bottom_px = 270
        panel_w_px = (fig_w_px - left_px - right_px - (n_cols - 1) * gap_px) / float(n_cols)
        panel_h_px = panel_w_px * (float(panel_src_h) / float(panel_src_w))
        block_h_px = n_rows * panel_h_px + (n_rows - 1) * gap_px
        fig_h_px = int(math.ceil(top_px + block_h_px + bottom_px))
        fig_h = fig_h_px / float(dpi)
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        axes = []
        for r_idx in range(n_rows):
            row_axes = []
            y_px = bottom_px + (n_rows - 1 - r_idx) * (panel_h_px + gap_px)
            for c_idx in range(n_cols):
                x_px = left_px + c_idx * (panel_w_px + gap_px)
                row_axes.append(
                    fig.add_axes(
                        [
                            x_px / fig_w_px,
                            y_px / fig_h_px,
                            panel_w_px / fig_w_px,
                            panel_h_px / fig_h_px,
                        ]
                    )
                )
            axes.append(row_axes)
        legend_bar_y = (bottom_px - 90) / fig_h_px
        legend_note_y = 38 / fig_h_px
        depth_cax_rect = [0.18, legend_bar_y, 0.28, 0.018]
        err_cax_rect = [0.57, legend_bar_y, 0.28, 0.018]
    else:
        gap_px = 0
        fig_h = max(8.0, float(row_height) * n_rows + 0.9)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), dpi=dpi)
        legend_bar_y = max(0.055, float(bottom_margin) - 0.035)
        legend_note_y = max(0.02, legend_bar_y - 0.11)
        depth_cax_rect = [0.18, legend_bar_y, 0.28, 0.018]
        err_cax_rect = [0.57, legend_bar_y, 0.28, 0.018]
    col_titles = ["RGB", "Mesh"]
    for method in methods:
        col_titles.extend([method["display"], "Rel. err"])
    legend_fontsize = float(title_fontsize)
    for r_idx, row in enumerate(rows):
        for c_idx, (panel, note) in enumerate(zip(row["panels"], row["panel_annotations"])):
            ax = axes[r_idx][c_idx] if exact_gap_mode else (axes[r_idx, c_idx] if len(rows) > 1 else axes[c_idx])
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if note:
                ax.text(
                    0.5,
                    0.5,
                    note,
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=float(note_fontsize),
                    color="white",
                    bbox={"facecolor": "black", "alpha": 0.6, "pad": 1.0, "edgecolor": "none"},
                )
            if r_idx == 0:
                ax.set_title(col_titles[c_idx], fontsize=float(title_fontsize), fontweight="normal")
            if c_idx == 0:
                ax.set_ylabel(Path(row["image_name"]).stem.split("_")[-2], fontsize=float(label_fontsize))
            if c_idx == 1 and show_mesh_range_text:
                ax.text(
                    0.02,
                    0.96,
                    f"{row['depth_vmin']:.2f}-{row['depth_vmax']:.2f} m",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=float(note_fontsize),
                    color="white",
                    bbox={"facecolor": "black", "alpha": 0.55, "pad": 1.2, "edgecolor": "none"},
                )
    if not exact_gap_mode:
        fig.subplots_adjust(wspace=float(wspace), hspace=float(hspace), left=0.025, right=0.995, top=0.94, bottom=float(bottom_margin))
    depth_cax = fig.add_axes(depth_cax_rect)
    depth_sm = plt.cm.ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap="viridis")
    depth_cb = fig.colorbar(depth_sm, cax=depth_cax, orientation="horizontal")
    depth_cb.set_ticks([0.0, 1.0])
    depth_cb.set_ticklabels(["near", "far"])
    depth_cb.ax.tick_params(labelsize=legend_fontsize)
    depth_cb.set_label(
        "Depth color: near to far. Per-view 2-98% mesh range.",
        fontsize=legend_fontsize,
    )

    err_cax = fig.add_axes(err_cax_rect)
    err_sm = plt.cm.ScalarMappable(norm=Normalize(vmin=-0.20, vmax=0.20), cmap="coolwarm")
    err_cb = fig.colorbar(err_sm, cax=err_cax, orientation="horizontal")
    err_cb.set_ticks([-0.20, -0.10, 0.0, 0.10, 0.20])
    err_cb.set_ticklabels(["-20%", "-10%", "0", "+10%", "+20%"])
    err_cb.ax.tick_params(labelsize=legend_fontsize)
    err_cb.set_label(
        "Relative error: blue = closer/front, red = farther/behind.",
        fontsize=legend_fontsize,
    )

    fig.text(
        0.5,
        legend_note_y,
        "Error = (D_method - D_mesh) / D_mesh; maps are clipped to +/-20%. Black pixels are invalid or outside mesh/method support.",
        ha="center",
        va="center",
        fontsize=legend_fontsize,
    )

    png_path = out_dir / f"{grid_prefix}.png"
    pdf_path = out_dir / f"{grid_prefix}.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)

    stats_path = out_dir / stats_filename
    with stats_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "band",
            "image_name",
            "method",
            "method_source",
            "available",
            "missing_reason",
            "valid_pixels",
            "ref_valid_pixels",
            "valid_ratio_on_ref",
            "AbsRel",
            "Agree@1%",
            "Agree@5%",
            "Agree@10%",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats_rows)

    method_bundle_records = []
    for method in methods:
        method_bundle_records.append(
            {
                "method": method["name"],
                "display": method["display"],
                "available": method["available"],
                "bundle_root": str(method["root"]) if method["root"] is not None else "",
                "source_tag": method.get("source_tag", ""),
                "missing_reason": method.get("missing_reason", ""),
            }
        )
    manifest_payload = {
        "band": band,
        "reference_manifest": str(ref_manifest_path),
        "rgb_image_dir": str(rgb_image_dir),
        "method_bundles": method_bundle_records,
        "mms_rgb_substitute_band": mms_rgb_substitute_band if band == "RGB" else "",
        "error_definition": "(D_method - D_mesh) / D_mesh",
        "valid_mask": "mesh_valid & method_valid & D_mesh > relative_depth_min",
        "depth_colormap": (
            "viridis per-view 2-98 percentile over reference mesh depth"
            if depth_range_source == "reference"
            else "viridis per-view 2-98 percentile over mesh and all methods"
        ),
        "depth_range_source": depth_range_source,
        "shared_depth_ranges": shared_depth_ranges or {},
        "error_colormap": "coolwarm fixed [-0.20, +0.20]",
        "figure_layout": {
            "columns": 12,
            "wspace": float(wspace),
            "hspace": float(hspace),
            "show_mesh_range_text": bool(show_mesh_range_text),
            "title_fontsize": float(title_fontsize),
            "title_fontweight": "normal",
            "label_fontsize": float(label_fontsize),
            "note_fontsize": float(note_fontsize),
            "legend_fontsize": float(title_fontsize),
            "fig_width": float(fig_width),
            "row_height": float(row_height),
            "bottom_margin": float(bottom_margin),
            "panel_gap_px": int(panel_gap_px),
            "exact_gap_mode": bool(exact_gap_mode),
            "legend_bar_y": float(legend_bar_y),
            "legend_note_y": float(legend_note_y),
        },
        "legend": {
            "depth": (
                "Viridis, row-normalized to per-view reference-mesh 2-98 percentile."
                if depth_range_source == "reference"
                else "Viridis, row-normalized to per-view mesh+method 2-98 percentile."
            ),
            "relative_error": "Coolwarm fixed to [-0.20,+0.20]; blue means method depth is closer than mesh, red means farther, black means invalid.",
        },
        "outputs": {
            "png": str(png_path),
            "pdf": str(pdf_path),
            "stats_csv": str(stats_path),
        },
    }
    manifest_path = out_dir / manifest_filename
    save_json(manifest_path, manifest_payload)
    print(f"VISUAL_GRID[{band}] {png_path}")
    return {
        "band": band,
        "manifest_path": str(manifest_path),
        "png": str(png_path),
        "pdf": str(pdf_path),
        "stats_csv": str(stats_path),
        "method_bundles": method_bundle_records,
    }


def _parse_method_specs(method_specs: Sequence[str]) -> List[Tuple[str, Path]]:
    parsed: List[Tuple[str, Path]] = []
    for spec in method_specs:
        if "=" not in spec:
            raise ValueError(f"Expected NAME=PATH format, got {spec!r}")
        name, path = spec.split("=", 1)
        parsed.append((name.strip(), Path(path).resolve()))
    return parsed


def _resolve_bundle_root_for_band(
    *,
    method_name: str,
    method_root: Path,
    band: str,
    mms_rgb_substitute_band: str,
    shared_rgb_anchor_root: Path | None,
) -> Tuple[Path | None, str, str]:
    # Supports both direct Model_X bundle roots and parent roots containing Model_* subdirs.
    direct_split = method_root / "split_manifest.json"
    direct_adapter = method_root / "adapter_manifest.json"
    if direct_split.exists() and direct_adapter.exists():
        return method_root, "explicit_bundle_root", ""

    source_tag = f"Model_{band}"
    if band == "RGB" and method_name in {"MMS retained-self", "MMS_retained-self"}:
        substitute_band = str(mms_rgb_substitute_band).upper()
        source_tag = f"MMS_{substitute_band}_as_RGB_substitute"
        bundle_root = method_root / f"Model_{substitute_band}"
    elif band == "RGB" and method_name in {"From-scratch", "Geometry-unfrozen", "UMGS-J"}:
        if shared_rgb_anchor_root is not None:
            bundle_root = shared_rgb_anchor_root / "Model_RGB"
            source_tag = "UMGS-I_RGB_anchor_substitute"
        else:
            bundle_root = method_root / f"Model_{band}"
    else:
        bundle_root = method_root / f"Model_{band}"

    if not bundle_root.exists():
        return None, source_tag, f"missing_model_dir:{bundle_root}"
    if not (bundle_root / "split_manifest.json").exists():
        return None, source_tag, f"missing_split_manifest:{bundle_root / 'split_manifest.json'}"
    if not (bundle_root / "adapter_manifest.json").exists():
        return None, source_tag, f"missing_adapter_manifest:{bundle_root / 'adapter_manifest.json'}"
    return bundle_root, source_tag, ""


def _build_methods_for_band(
    *,
    method_specs: Sequence[Tuple[str, Path]],
    band: str,
    mms_rgb_substitute_band: str,
    shared_rgb_anchor_root: Path | None,
) -> List[Dict[str, Any]]:
    methods: List[Dict[str, Any]] = []
    for name, root in method_specs:
        bundle_root, source_tag, missing_reason = _resolve_bundle_root_for_band(
            method_name=name,
            method_root=root,
            band=band,
            mms_rgb_substitute_band=mms_rgb_substitute_band,
            shared_rgb_anchor_root=shared_rgb_anchor_root,
        )
        if bundle_root is None:
            methods.append(
                {
                    "name": name,
                    "display": METHOD_DISPLAY.get(name, name),
                    "root": None,
                    "source_tag": source_tag,
                    "available": False,
                    "missing_reason": missing_reason,
                }
            )
            continue
        split_manifest = load_json(bundle_root / "split_manifest.json")
        adapter_manifest = load_json(bundle_root / "adapter_manifest.json")
        methods.append(
            {
                "name": name,
                "display": METHOD_DISPLAY.get(name, name),
                "root": bundle_root,
                "source_tag": source_tag,
                "split_manifest": split_manifest,
                "adapter_manifest": adapter_manifest,
                "views_by_name": {str(v["image_name"]): v for v in split_manifest["views"]},
                "available": True,
                "missing_reason": "",
            }
        )
    return methods


def visualize(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_manifest_path = Path(args.reference_manifest).resolve()
    ref_manifest = load_json(ref_manifest_path)
    method_specs = _parse_method_specs(args.method_bundle)
    methods = _build_methods_for_band(
        method_specs=method_specs,
        band=str(args.band).upper(),
        mms_rgb_substitute_band=str(args.mms_rgb_substitute_band).upper(),
        shared_rgb_anchor_root=Path(args.shared_rgb_anchor_root).resolve()
        if str(args.shared_rgb_anchor_root).strip()
        else None,
    )
    if len(methods) != 5:
        raise ValueError(f"Expected exactly 5 methods, got {len(methods)}")
    _render_single_band_grid(
        out_dir=out_dir,
        ref_manifest_path=ref_manifest_path,
        ref_manifest=ref_manifest,
        methods=methods,
        rgb_image_dir=Path(args.rgb_image_dir).resolve(),
        relative_depth_min=float(args.relative_depth_min),
        grid_prefix=str(args.grid_prefix),
        stats_filename=str(args.stats_filename),
        manifest_filename=str(args.manifest_filename),
        wspace=float(args.wspace),
        hspace=float(args.hspace),
        band=str(args.band).upper(),
        mms_rgb_substitute_band=str(args.mms_rgb_substitute_band).upper(),
        depth_range_source=str(args.depth_range_source),
        shared_depth_ranges=_compute_reference_depth_ranges(ref_manifest_path, ref_manifest)
        if str(args.depth_range_source) == "reference"
        else None,
        show_mesh_range_text=not bool(args.hide_mesh_range_text),
        title_fontsize=float(args.title_fontsize),
        label_fontsize=float(args.label_fontsize),
        note_fontsize=float(args.note_fontsize),
        fig_width=float(args.fig_width),
        row_height=float(args.row_height),
        bottom_margin=float(args.bottom_margin),
        panel_gap_px=int(args.panel_gap_px),
    )


def visualize_multiband(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_manifest_path = Path(args.reference_manifest).resolve()
    ref_manifest = load_json(ref_manifest_path)
    bands = [item.strip().upper() for item in str(args.bands).split(",") if item.strip()]
    if not bands:
        raise ValueError("No bands provided for --bands")
    method_specs = _parse_method_specs(args.method_root)
    if len(method_specs) != 5:
        raise ValueError(f"Expected exactly 5 method roots via --method_root, got {len(method_specs)}")

    all_manifest_records: Dict[str, Any] = {
        "reference_manifest": str(ref_manifest_path),
        "rgb_image_dir": str(Path(args.rgb_image_dir).resolve()),
        "bands": bands,
        "relative_depth_min": float(args.relative_depth_min),
        "figure_layout": {
            "columns": 12,
            "wspace": float(args.wspace),
            "hspace": float(args.hspace),
            "hide_mesh_range_text": bool(args.hide_mesh_range_text),
            "title_fontsize": float(args.title_fontsize),
            "label_fontsize": float(args.label_fontsize),
            "note_fontsize": float(args.note_fontsize),
            "fig_width": float(args.fig_width),
            "row_height": float(args.row_height),
            "bottom_margin": float(args.bottom_margin),
            "panel_gap_px": int(args.panel_gap_px),
        },
        "mms_rgb_substitute_band": str(args.mms_rgb_substitute_band).upper(),
        "depth_range_source": str(args.depth_range_source),
        "per_band_outputs": {},
    }
    shared_depth_ranges = (
        _compute_reference_depth_ranges(ref_manifest_path, ref_manifest)
        if str(args.depth_range_source) == "reference"
        else None
    )
    if shared_depth_ranges is not None:
        all_manifest_records["shared_depth_ranges"] = shared_depth_ranges

    for band in bands:
        methods = _build_methods_for_band(
            method_specs=method_specs,
            band=band,
            mms_rgb_substitute_band=str(args.mms_rgb_substitute_band).upper(),
            shared_rgb_anchor_root=Path(args.shared_rgb_anchor_root).resolve()
            if str(args.shared_rgb_anchor_root).strip()
            else None,
        )
        result = _render_single_band_grid(
            out_dir=out_dir,
            ref_manifest_path=ref_manifest_path,
            ref_manifest=ref_manifest,
            methods=methods,
            rgb_image_dir=Path(args.rgb_image_dir).resolve(),
            relative_depth_min=float(args.relative_depth_min),
            grid_prefix=f"depth_visual_grid_10views_core5_{band}",
            stats_filename=f"depth_visual_stats_{band}.csv",
            manifest_filename=f"depth_visual_manifest_{band}.json",
            wspace=float(args.wspace),
            hspace=float(args.hspace),
            band=band,
            mms_rgb_substitute_band=str(args.mms_rgb_substitute_band).upper(),
            depth_range_source=str(args.depth_range_source),
            shared_depth_ranges=shared_depth_ranges,
            show_mesh_range_text=not bool(args.hide_mesh_range_text),
            title_fontsize=float(args.title_fontsize),
            label_fontsize=float(args.label_fontsize),
            note_fontsize=float(args.note_fontsize),
            fig_width=float(args.fig_width),
            row_height=float(args.row_height),
            bottom_margin=float(args.bottom_margin),
            panel_gap_px=int(args.panel_gap_px),
        )
        all_manifest_records["per_band_outputs"][band] = result

    save_json(out_dir / "depth_visual_manifest_all_bands.json", all_manifest_records)
    print(f"VISUAL_MULTI_BAND_MANIFEST {out_dir / 'depth_visual_manifest_all_bands.json'}")


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
    v.add_argument("--band", default="G")
    v.add_argument("--mms_rgb_substitute_band", default="D")
    v.add_argument("--shared_rgb_anchor_root", default="")
    v.add_argument("--depth_range_source", default="reference", choices=("reference", "mesh_and_methods"))
    v.add_argument("--wspace", type=float, default=0.006)
    v.add_argument("--hspace", type=float, default=0.025)
    v.add_argument("--hide_mesh_range_text", action="store_true")
    v.add_argument("--title_fontsize", type=float, default=8.0)
    v.add_argument("--label_fontsize", type=float, default=7.0)
    v.add_argument("--note_fontsize", type=float, default=7.0)
    v.add_argument("--fig_width", type=float, default=24.0)
    v.add_argument("--row_height", type=float, default=1.28)
    v.add_argument("--bottom_margin", type=float, default=0.135)
    v.add_argument("--panel_gap_px", type=int, default=0, help="If >0, place all image panels with this exact pixel gap.")
    v.add_argument("--grid_prefix", default="depth_visual_grid_10views_core5_with_legend")
    v.add_argument("--stats_filename", default="depth_visual_stats.csv")
    v.add_argument("--manifest_filename", default="depth_visual_manifest.json")
    v.set_defaults(func=visualize)

    vb = sub.add_parser("visualize_multiband")
    vb.add_argument("--reference_manifest", required=True)
    vb.add_argument("--rgb_image_dir", required=True)
    vb.add_argument("--out_dir", required=True)
    vb.add_argument("--method_root", action="append", required=True, help="NAME=ROOT where ROOT contains Model_<BAND> subdirs or direct bundle manifests")
    vb.add_argument("--bands", default="RGB,G,R,RE,NIR")
    vb.add_argument("--mms_rgb_substitute_band", default="D")
    vb.add_argument("--shared_rgb_anchor_root", default="")
    vb.add_argument("--depth_range_source", default="reference", choices=("reference", "mesh_and_methods"))
    vb.add_argument("--relative_depth_min", type=float, default=1e-6)
    vb.add_argument("--wspace", type=float, default=0.006)
    vb.add_argument("--hspace", type=float, default=0.025)
    vb.add_argument("--hide_mesh_range_text", action="store_true")
    vb.add_argument("--title_fontsize", type=float, default=8.0)
    vb.add_argument("--label_fontsize", type=float, default=7.0)
    vb.add_argument("--note_fontsize", type=float, default=7.0)
    vb.add_argument("--fig_width", type=float, default=24.0)
    vb.add_argument("--row_height", type=float, default=1.28)
    vb.add_argument("--bottom_margin", type=float, default=0.135)
    vb.add_argument("--panel_gap_px", type=int, default=0, help="If >0, place all image panels with this exact pixel gap.")
    vb.set_defaults(func=visualize_multiband)
    return parser


def main() -> None:
    args = _argparser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
