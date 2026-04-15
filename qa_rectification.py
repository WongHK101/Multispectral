from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image

from utils.rectification_utils import (
    determine_pass_from_summary,
    export_rectification_debug_panel,
    load_rgb_plane_image,
    prepare_alignment_images,
    score_edge_overlap_f1,
    score_gradient_ncc,
    write_rectification_diagnostics_json,
)
from utils.spectral_image_utils import load_image_preserve_dtype, normalize_scalar_band_image


DEFAULT_BANDS = ("G", "R", "RE", "NIR")


def _parse_bands(bands) -> List[str]:
    if bands is None:
        return list(DEFAULT_BANDS)
    if isinstance(bands, str):
        items = [item.strip() for item in bands.split(",")]
    else:
        items = [str(item).strip() for item in bands]
    parsed: List[str] = []
    for item in items:
        if not item:
            continue
        if item not in parsed:
            parsed.append(item)
    if not parsed:
        raise ValueError("No valid bands provided to qa_rectification.")
    return parsed


def _load_manifest(scene_root: Path) -> Dict[str, object]:
    manifest_path = scene_root / "spectral_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing spectral_manifest.json: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _image_map(manifest: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    return {
        str(item.get("image_name", "")).strip(): item
        for item in manifest.get("images", [])
        if str(item.get("image_name", "")).strip()
    }


def _save_uint8_image(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _load_scalar_image(path: Path, dynamic_range: str, radiometric_mode: str) -> np.ndarray:
    loaded = load_image_preserve_dtype(path)
    raw = np.asarray(loaded.array)
    if raw.ndim == 2 or (raw.ndim == 3 and raw.shape[2] == 1):
        return normalize_scalar_band_image(loaded, loaded.metadata, mode=radiometric_mode, dynamic_range=dynamic_range)
    if raw.ndim == 3 and raw.shape[2] >= 3:
        arr = raw.astype(np.float32, copy=False)
        if np.issubdtype(raw.dtype, np.integer):
            if str(dynamic_range).lower() == "uint8":
                denom = 255.0
            elif str(dynamic_range).lower() == "uint16":
                denom = 65535.0
            else:
                denom = float(np.iinfo(raw.dtype).max)
            arr = arr / max(denom, 1.0)
        gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        gray = np.nan_to_num(gray, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(gray, 0.0, 1.0).astype(np.float32)
    raise ValueError(f"Unsupported image shape for QA scalar projection: {raw.shape}")


def _summary_without_frames(summary: dict) -> dict:
    return {key: value for key, value in summary.items() if key != "per_frame"}


def run_rectification_qa(prepared_root: Path,
                         rectified_root: Path,
                         out_root: Path,
                         bands: List[str] | str | None = None,
                         frame_count: int = 6,
                         radiometric_mode: str = "exposure_normalized",
                         input_dynamic_range: str = "uint16",
                         edge_dilate_radius: int = 1,
                         min_improved_ratio: float = 0.6,
                         max_severe_outliers: int = 0,
                         qa_scale: float = 1.0) -> Dict[str, object]:
    prepared_root = prepared_root.resolve()
    rectified_root = rectified_root.resolve()
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    config_path = rectified_root / "rectification_homographies.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing rectification config for QA: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    rgb_scene_root = prepared_root / "RGB"
    rgb_items = _image_map(_load_manifest(rgb_scene_root))
    band_list = _parse_bands(bands)
    summary = {
        "prepared_root": str(prepared_root),
        "rectified_root": str(rectified_root),
        "out_root": str(out_root),
        "bands_order": band_list,
        "bands": {},
    }

    for band in band_list:
        raw_scene_root = prepared_root / f"{band}_raw"
        rect_scene_root = rectified_root / f"{band}_rectified"
        raw_map = _image_map(_load_manifest(raw_scene_root))
        rect_map = _image_map(_load_manifest(rect_scene_root))
        band_cfg = config["bands"][band]
        selected_names = list(band_cfg.get("selected_image_names", []))
        if not selected_names:
            common_names = sorted(set(rgb_items.keys()) & set(raw_map.keys()) & set(rect_map.keys()))
            selected_names = common_names[:frame_count]
        else:
            selected_names = selected_names[:frame_count]

        band_dir = out_root / band
        band_dir.mkdir(parents=True, exist_ok=True)

        frame_records: List[dict] = []
        baseline_edge = []
        baseline_grad = []
        rect_edge = []
        rect_grad = []

        for image_name in selected_names:
            rgb_image = load_rgb_plane_image(rgb_scene_root / "images" / image_name)
            raw_band = _load_scalar_image(Path(str(raw_map[image_name].get("source_path"))), dynamic_range=input_dynamic_range, radiometric_mode=radiometric_mode)
            rect_band = _load_scalar_image(rect_scene_root / "images" / image_name, dynamic_range=input_dynamic_range, radiometric_mode=radiometric_mode)
            mask_path = Path(str(rect_map[image_name].get("validity_mask_path", "")))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Missing QA mask: {mask_path}")
            mask = (mask.astype(np.float32) / 255.0)

            if float(qa_scale) < 0.999:
                target_h, target_w = rgb_image.shape
                scaled_w = max(int(round(target_w * float(qa_scale))), 16)
                scaled_h = max(int(round(target_h * float(qa_scale))), 16)
                rgb_image = cv2.resize(rgb_image, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
                raw_band = cv2.resize(raw_band, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
                rect_band = cv2.resize(rect_band, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)

            target_h, target_w = rgb_image.shape
            naive_resized = cv2.resize(raw_band, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            rgb_prepared = prepare_alignment_images(rgb_image, raw_band)
            naive_prepared = prepare_alignment_images(rgb_image, naive_resized)
            rect_prepared = prepare_alignment_images(rgb_image, rect_band)

            baseline_edge_f1 = score_edge_overlap_f1(
                rgb_prepared["rgb_edges"],
                naive_prepared["band_edges"],
                mask,
                dilate_radius=edge_dilate_radius,
            )
            baseline_grad_ncc = score_gradient_ncc(
                rgb_prepared["rgb_grad"],
                naive_prepared["band_grad"],
                mask,
            )
            rectified_edge_f1 = score_edge_overlap_f1(
                rgb_prepared["rgb_edges"],
                rect_prepared["band_edges"],
                mask,
                dilate_radius=edge_dilate_radius,
            )
            rectified_grad_ncc = score_gradient_ncc(
                rgb_prepared["rgb_grad"],
                rect_prepared["band_grad"],
                mask,
            )

            delta_edge = float(rectified_edge_f1 - baseline_edge_f1)
            delta_grad = float(rectified_grad_ncc - baseline_grad_ncc)
            validity_ratio = float(np.mean(mask))
            panel_path = band_dir / f"{Path(image_name).stem}_panel.png"
            export_rectification_debug_panel(
                rgb_img=rgb_image,
                raw_band_img=raw_band,
                naive_resized=naive_resized,
                rectified_img=rect_band,
                mask=mask,
                out_path=panel_path,
            )
            _save_uint8_image(
                band_dir / f"{Path(image_name).stem}_edge_overlay_naive.png",
                np.stack([rgb_prepared["rgb_edges"] * 255, naive_prepared["band_edges"] * 255, np.zeros_like(rgb_prepared["rgb_edges"])], axis=-1).astype(np.uint8),
            )
            _save_uint8_image(
                band_dir / f"{Path(image_name).stem}_edge_overlay_rectified.png",
                np.stack([rgb_prepared["rgb_edges"] * 255, rect_prepared["band_edges"] * 255, np.zeros_like(rgb_prepared["rgb_edges"])], axis=-1).astype(np.uint8),
            )

            severe = delta_edge < -0.01 and delta_grad < -0.01
            frame_records.append(
                {
                    "frame_id": str(raw_map[image_name].get("frame_id", image_name)),
                    "image_name": image_name,
                    "baseline_edge_f1": float(baseline_edge_f1),
                    "rectified_edge_f1": float(rectified_edge_f1),
                    "baseline_grad_ncc": float(baseline_grad_ncc),
                    "rectified_grad_ncc": float(rectified_grad_ncc),
                    "delta_edge_f1": delta_edge,
                    "delta_grad_ncc": delta_grad,
                    "validity_ratio": validity_ratio,
                    "debug_panel_path": str(panel_path),
                    "severe_misalignment": bool(severe),
                }
            )
            baseline_edge.append(float(baseline_edge_f1))
            baseline_grad.append(float(baseline_grad_ncc))
            rect_edge.append(float(rectified_edge_f1))
            rect_grad.append(float(rectified_grad_ncc))

        band_summary = {
            "baseline_mean_edge_f1": float(np.mean(baseline_edge)) if baseline_edge else float("nan"),
            "baseline_mean_grad_ncc": float(np.mean(baseline_grad)) if baseline_grad else float("nan"),
            "rectified_mean_edge_f1": float(np.mean(rect_edge)) if rect_edge else float("nan"),
            "rectified_mean_grad_ncc": float(np.mean(rect_grad)) if rect_grad else float("nan"),
            "delta_edge_f1": (float(np.mean(rect_edge)) - float(np.mean(baseline_edge))) if baseline_edge else float("nan"),
            "delta_grad_ncc": (float(np.mean(rect_grad)) - float(np.mean(baseline_grad))) if baseline_grad else float("nan"),
            "num_frames": len(frame_records),
            "num_frames_improved_edge": sum(1 for item in frame_records if item["delta_edge_f1"] > 0),
            "num_frames_improved_grad": sum(1 for item in frame_records if item["delta_grad_ncc"] > 0),
            "num_frames_improved_either": sum(1 for item in frame_records if item["delta_edge_f1"] > 0 or item["delta_grad_ncc"] > 0),
            "severe_outlier_count": sum(1 for item in frame_records if item["severe_misalignment"]),
            "per_frame": frame_records,
        }
        band_summary["improved_ratio"] = float(band_summary["num_frames_improved_either"]) / float(max(band_summary["num_frames"], 1))
        band_summary["pass"] = determine_pass_from_summary(
            band_summary,
            min_improved_ratio=min_improved_ratio,
            max_severe_outliers=max_severe_outliers,
        )
        summary["bands"][band] = band_summary

    write_rectification_diagnostics_json(summary, out_root / "rectification_qa_summary.json")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate dual-metric QA artifacts for rectified band datasets.")
    ap.add_argument("--prepared_root", required=True, help="Prepared raw root with RGB and *_raw scenes.")
    ap.add_argument("--rectified_root", required=True, help="Root containing *_rectified scenes.")
    ap.add_argument("--out_root", required=True, help="Output directory for QA overlays and summary JSON.")
    ap.add_argument("--frame_count", type=int, default=6)
    ap.add_argument("--bands", default="G,R,RE,NIR", help="Comma-separated modality list to QA (e.g., G,R,RE,NIR or T).")
    ap.add_argument("--input_dynamic_range", default="uint16", choices=["uint8", "uint16", "float"])
    ap.add_argument("--radiometric_mode", default="exposure_normalized", choices=["raw_dn", "exposure_normalized", "reflectance_ready_stub"])
    ap.add_argument("--rectification_edge_dilate_radius", type=int, default=1)
    ap.add_argument("--rectification_min_improved_ratio", type=float, default=0.6)
    ap.add_argument("--rectification_max_severe_outliers", type=int, default=0)
    ap.add_argument("--rectification_qa_scale", type=float, default=1.0)
    args = ap.parse_args()

    summary = run_rectification_qa(
        prepared_root=Path(args.prepared_root),
        rectified_root=Path(args.rectified_root),
        out_root=Path(args.out_root),
        bands=args.bands,
        frame_count=args.frame_count,
        radiometric_mode=args.radiometric_mode,
        input_dynamic_range=args.input_dynamic_range,
        edge_dilate_radius=args.rectification_edge_dilate_radius,
        min_improved_ratio=args.rectification_min_improved_ratio,
        max_severe_outliers=args.rectification_max_severe_outliers,
        qa_scale=args.rectification_qa_scale,
    )
    print(json.dumps(
        {
            "bands": {
                band: {
                    "pass": summary["bands"][band]["pass"],
                    "delta_edge_f1": summary["bands"][band]["delta_edge_f1"],
                    "delta_grad_ncc": summary["bands"][band]["delta_grad_ncc"],
                }
                for band in summary.get("bands", {}).keys()
            }
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
