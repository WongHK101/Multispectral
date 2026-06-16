from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps
import tifffile
from skimage import measure, morphology


DJI_RGB_RE = re.compile(r"DJI_(\d{14})_(\d{4})_D[.]JPG$", re.IGNORECASE)
DJI_FLIR_RE = re.compile(r"DJI_(\d{14})_(\d{4})_F[.]JPG$", re.IGNORECASE)
DJI_MS_RE = re.compile(r"DJI_(\d{14})_(\d{4})_MS_(G|R|RE|NIR)[.]TIF$", re.IGNORECASE)


@dataclass
class ImageRecord:
    path: Path
    timestamp: datetime
    seq: str
    band: str
    panel_group: str
    calib_root: str


def parse_image_name(path: Path) -> tuple[datetime, str, str] | None:
    name = path.name
    for regex, band_fn in (
        (DJI_RGB_RE, lambda m: "RGB"),
        (DJI_FLIR_RE, lambda m: "F"),
        (DJI_MS_RE, lambda m: m.group(3).upper()),
    ):
        match = regex.search(name)
        if match:
            return datetime.strptime(match.group(1), "%Y%m%d%H%M%S"), match.group(2), band_fn(match)
    return None


def direct_image_records(folder: Path, group: str, root: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for path in sorted(folder.iterdir(), key=lambda p: p.name):
        if not path.is_file():
            continue
        parsed = parse_image_name(path)
        if not parsed:
            continue
        timestamp, seq, band = parsed
        records.append(ImageRecord(path, timestamp, seq, band, group, str(root)))
    return records


def discover_panel_groups(roots: Iterable[Path]) -> list[tuple[str, Path, Path, list[ImageRecord]]]:
    groups = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for folder in [root, *[p for p in root.rglob("*") if p.is_dir()]]:
            if any(part.startswith("_not_reflectance") or part.startswith("_archive") for part in folder.parts):
                continue
            if folder in seen:
                continue
            seen.add(folder)
            group = str(folder.relative_to(root)) if folder != root else folder.name
            records = direct_image_records(folder, group, root)
            if records:
                groups.append((group, root, folder, records))
    return sorted(groups, key=lambda item: min(r.timestamp for r in item[3]))


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def image_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        arr = tifffile.imread(path)
        if arr.ndim > 2:
            arr = arr[..., 0]
        return np.asarray(arr)
    with Image.open(path) as img:
        return np.asarray(ImageOps.exif_transpose(img.convert("RGB")))


def normalize_for_detection(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr_f = arr.astype(np.float32)
        gray = 0.299 * arr_f[..., 0] + 0.587 * arr_f[..., 1] + 0.114 * arr_f[..., 2]
    else:
        gray = arr.astype(np.float32)
    lo, hi = np.percentile(gray, [1, 99])
    if hi <= lo:
        return np.zeros_like(gray, dtype=np.float32)
    return np.clip((gray - lo) / (hi - lo), 0.0, 1.0)


def _empty_roi(path: Path, reason: str) -> dict[str, object]:
    arr = image_array(path)
    gray_full = normalize_for_detection(arr)
    full_h, full_w = gray_full.shape[:2]
    x0, y0, x1, y1 = int(full_w * 0.35), int(full_h * 0.35), int(full_w * 0.65), int(full_h * 0.65)
    roi = arr[y0:y1, x0:x1]
    return {
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "panel_x0": x0,
        "panel_y0": y0,
        "panel_x1": x1,
        "panel_y1": y1,
        "width": full_w,
        "height": full_h,
        "method": "no_automatic_roi",
        "roi_area_fraction": (x1 - x0) * (y1 - y0) / float(full_w * full_h),
        "panel_area_fraction": (x1 - x0) * (y1 - y0) / float(full_w * full_h),
        "roi_to_panel_area_ratio": 1.0,
        "roi_pixel_count": 0,
        "excluded_edge_or_wrinkle_fraction": 0.0,
        "saturation_fraction": 0.0,
        "gradient_p95": "",
        "roi_cv": "",
        "qc_status": "reject",
        "confidence": "low",
        "requires_manual_roi": True,
        "reject_reason": reason,
        "median": "",
        "mean": "",
        "std": "",
        "_mask": np.zeros((full_h, full_w), dtype=bool),
        "_panel_mask": np.zeros((full_h, full_w), dtype=bool),
        "_excluded_mask": np.zeros((full_h, full_w), dtype=bool),
    }


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels = measure.label(mask)
    props = measure.regionprops(labels)
    if not props:
        return np.zeros_like(mask, dtype=bool)
    prop = max(props, key=lambda item: item.area)
    return labels == prop.label


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _sanitize_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")[:120]


def propose_roi(path: Path) -> dict[str, object]:
    arr = image_array(path)
    gray_full = normalize_for_detection(arr)
    full_h, full_w = gray_full.shape[:2]
    stride = max(1, int(math.ceil(max(full_h, full_w) / 1200.0)))
    gray = gray_full[::stride, ::stride]
    h, w = gray.shape[:2]

    # The reflectance sheet is not a rigid planar square in these captures.
    # We therefore detect a bright panel candidate per band, then keep only
    # an eroded low-gradient interior mask for response estimation. The mask
    # intentionally excludes panel borders, wood supports, wrinkles, and
    # near-saturated highlights.
    candidates = []
    for pct in (99, 98, 97, 96, 95, 94, 92, 90):
        threshold = max(0.58, float(np.percentile(gray, pct)))
        mask = gray >= threshold
        radius = max(2, int(min(h, w) * 0.008))
        mask = morphology.binary_closing(mask, morphology.disk(radius))
        mask = morphology.remove_small_objects(mask, min_size=max(12, int(w * h * 0.0002)))
        labels = measure.label(mask)
        for prop in measure.regionprops(labels, intensity_image=gray):
            y0, x0, y1, x1 = prop.bbox
            area_fraction = prop.area / float(w * h)
            if area_fraction < 0.00015 or area_fraction > 0.12:
                continue
            bw = max(1, x1 - x0)
            bh = max(1, y1 - y0)
            aspect = bw / float(bh)
            if aspect > 2.8 or aspect < 0.35:
                continue
            if prop.extent < 0.12:
                continue
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            center_dist = math.sqrt(((cx - w / 2.0) / max(w, 1)) ** 2 + ((cy - h / 2.0) / max(h, 1)) ** 2)
            square_score = 1.0 - min(abs(math.log(aspect)), 1.3) / 1.3
            size_score = 1.0 - min(abs(math.log(max(area_fraction, 1e-6) / 0.006)), 2.0) / 2.0
            brightness = float(prop.mean_intensity)
            score = prop.area * (0.45 + prop.extent) * (0.35 + square_score) * (0.35 + max(size_score, 0.0)) * (0.6 + brightness) * (1.0 - min(center_dist, 0.65))
            candidates.append((score, x0, y0, x1, y1, area_fraction, prop.extent, center_dist, pct))

    if not candidates:
        return _empty_roi(path, "no_compact_high_reflectance_component")

    _, x0, y0, x1, y1, component_area_fraction, extent, center_dist, pct = max(candidates, key=lambda item: item[0])
    bw = x1 - x0
    bh = y1 - y0
    pad_x = max(3, int(0.22 * bw))
    pad_y = max(3, int(0.22 * bh))
    x0, x1 = max(0, x0 - pad_x), min(w, x1 + pad_x)
    y0, y1 = max(0, y0 - pad_y), min(h, y1 + pad_y)
    px0, px1 = int(x0 * stride), min(full_w, int(math.ceil(x1 * stride)))
    py0, py1 = int(y0 * stride), min(full_h, int(math.ceil(y1 * stride)))
    if px1 <= px0 or py1 <= py0:
        return _empty_roi(path, "invalid_candidate_bbox")

    local = gray_full[py0:py1, px0:px1]
    local_hi = float(np.percentile(local, 99.5))
    local_mid = float(np.percentile(local, 75))
    local_thr = max(0.50, min(local_hi * 0.82, local_mid))
    panel_local = local >= local_thr
    panel_local = morphology.binary_closing(panel_local, morphology.disk(max(2, int(min(local.shape) * 0.015))))
    panel_local = morphology.remove_small_objects(panel_local, min_size=max(16, int(local.size * 0.015)))
    panel_local = _largest_component(panel_local)
    if not panel_local.any():
        return _empty_roi(path, "no_refined_panel_component")

    panel_bbox_local = _mask_bbox(panel_local)
    if panel_bbox_local is None:
        return _empty_roi(path, "empty_panel_bbox")
    lx0, ly0, lx1, ly1 = panel_bbox_local
    min_panel_dim = max(1, min(lx1 - lx0, ly1 - ly0))
    erode_radius = max(2, int(min_panel_dim * 0.08))
    interior = morphology.binary_erosion(panel_local, morphology.disk(erode_radius))
    if interior.sum() < max(20, panel_local.sum() * 0.08):
        erode_radius = max(1, int(min_panel_dim * 0.04))
        interior = morphology.binary_erosion(panel_local, morphology.disk(erode_radius))
    if interior.sum() < max(20, panel_local.sum() * 0.04):
        interior = panel_local.copy()

    gy, gx = np.gradient(local.astype(np.float32))
    grad = np.hypot(gx, gy)
    if interior.any():
        grad_cut = float(np.percentile(grad[interior], 92))
        bright_cut = float(np.percentile(local[interior], 3))
    else:
        grad_cut = float(np.percentile(grad, 90))
        bright_cut = float(np.percentile(local, 60))
    low_gradient = grad <= max(grad_cut, 0.015)
    not_low_brightness = local >= max(0.35, bright_cut)
    not_saturated = local <= 0.999
    clean = interior & low_gradient & not_low_brightness & not_saturated
    clean = morphology.remove_small_objects(clean, min_size=max(10, int(panel_local.sum() * 0.01)))
    clean = _largest_component(clean)
    if clean.sum() < max(25, panel_local.sum() * 0.06):
        clean = interior & not_low_brightness & not_saturated
        clean = _largest_component(clean)
    if clean.sum() < max(25, panel_local.sum() * 0.04):
        clean = interior.copy()

    panel_mask = np.zeros((full_h, full_w), dtype=bool)
    roi_mask = np.zeros((full_h, full_w), dtype=bool)
    excluded_mask = np.zeros((full_h, full_w), dtype=bool)
    panel_mask[py0:py1, px0:px1] = panel_local
    roi_mask[py0:py1, px0:px1] = clean
    excluded_mask[py0:py1, px0:px1] = panel_local & ~clean
    bbox = _mask_bbox(roi_mask)
    if bbox is None:
        return _empty_roi(path, "empty_clean_roi")
    rx0, ry0, rx1, ry1 = bbox

    roi_values = arr[roi_mask]
    roi_gray_values = gray_full[roi_mask]
    panel_pixels = int(panel_mask.sum())
    roi_pixels = int(roi_mask.sum())
    excluded_fraction = float(excluded_mask.sum() / max(panel_pixels, 1))
    saturation_fraction = float((gray_full[roi_mask] >= 0.995).sum() / max(roi_pixels, 1))
    roi_mean = float(np.mean(roi_values))
    roi_std = float(np.std(roi_values))
    roi_cv = float(roi_std / roi_mean) if roi_mean > 0 else float("inf")
    gradient_p95 = float(np.percentile(grad[clean], 95)) if clean.any() else float("inf")
    roi_to_panel = float(roi_pixels / max(panel_pixels, 1))

    qc_status = "accept"
    reject_reason = ""
    if roi_pixels < 50:
        qc_status = "reject"
        reject_reason = "roi_too_small"
    elif roi_to_panel < 0.08:
        qc_status = "requires_manual_roi"
        reject_reason = "small_clean_interior"
    elif roi_cv > 0.20 or excluded_fraction > 0.78:
        qc_status = "usable_with_caution"
        reject_reason = "wrinkle_or_nonuniform_panel"
    elif center_dist > 0.55:
        qc_status = "usable_with_caution"
        reject_reason = "off_center_panel"
    confidence = {"accept": "high", "usable_with_caution": "medium", "requires_manual_roi": "low", "reject": "low"}[qc_status]
    return {
        "x0": rx0,
        "y0": ry0,
        "x1": rx1,
        "y1": ry1,
        "panel_x0": px0 + lx0,
        "panel_y0": py0 + ly0,
        "panel_x1": px0 + lx1,
        "panel_y1": py0 + ly1,
        "width": full_w,
        "height": full_h,
        "method": f"per_band_eroded_low_gradient_mask_p{pct}",
        "roi_area_fraction": roi_pixels / float(full_w * full_h),
        "panel_area_fraction": panel_pixels / float(full_w * full_h),
        "roi_to_panel_area_ratio": roi_to_panel,
        "roi_pixel_count": roi_pixels,
        "excluded_edge_or_wrinkle_fraction": excluded_fraction,
        "saturation_fraction": saturation_fraction,
        "gradient_p95": gradient_p95,
        "roi_cv": roi_cv,
        "qc_status": qc_status,
        "confidence": confidence,
        "requires_manual_roi": qc_status in {"requires_manual_roi", "reject"},
        "reject_reason": reject_reason,
        "median": float(np.median(roi_values)),
        "mean": roi_mean,
        "std": roi_std,
        "_mask": roi_mask,
        "_panel_mask": panel_mask,
        "_excluded_mask": excluded_mask,
    }


def _strip_private_roi(row: dict[str, object]) -> dict[str, object]:
    return {k: v for k, v in row.items() if not str(k).startswith("_")}


def _overlay_mask(base: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: int) -> Image.Image:
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    mask_img = Image.fromarray(np.uint8(mask) * alpha, mode="L").resize(base.size, Image.Resampling.NEAREST)
    solid = Image.new("RGBA", base.size, (*color, alpha))
    overlay.paste(solid, (0, 0), mask_img)
    return Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")


def image_thumbnail_with_roi(path: Path, roi: dict[str, object], size: tuple[int, int] = (260, 190)) -> Image.Image:
    arr = image_array(path)
    if arr.ndim == 2:
        gray = normalize_for_detection(arr)
        img = Image.fromarray(np.uint8(gray * 255), mode="L").convert("RGB")
    else:
        img = Image.fromarray(np.asarray(arr, dtype=np.uint8) if arr.dtype == np.uint8 else np.uint8(normalize_for_detection(arr) * 255)).convert("RGB")
    img = ImageOps.contain(img, size)
    draw = ImageDraw.Draw(img)
    sx = img.width / float(roi["width"])
    sy = img.height / float(roi["height"])
    panel_box = [int(roi["panel_x0"] * sx), int(roi["panel_y0"] * sy), int(roi["panel_x1"] * sx), int(roi["panel_y1"] * sy)]
    roi_box = [int(roi["x0"] * sx), int(roi["y0"] * sy), int(roi["x1"] * sx), int(roi["y1"] * sy)]
    status = str(roi.get("qc_status", "requires_manual_roi"))
    color = {"accept": "lime", "usable_with_caution": "yellow", "requires_manual_roi": "orange", "reject": "red"}.get(status, "red")
    draw.rectangle(panel_box, outline="cyan", width=2)
    draw.rectangle(roi_box, outline=color, width=3)
    return img


def make_contact_sheet(panel_rois: list[dict[str, object]], out_path: Path) -> None:
    font = ImageFont.load_default()
    thumbs: list[tuple[Image.Image, str]] = []
    for row in panel_rois:
        if row["band"] not in {"G", "R", "RE", "NIR"}:
            continue
        if int(row["is_representative"]) != 1:
            continue
        img = image_thumbnail_with_roi(Path(str(row["file_path"])), row)
        label = f"{row['panel_group']} | {row['band']} | {row.get('qc_status', row['confidence'])}"
        thumbs.append((img, label))
    if not thumbs:
        return
    cols = 2
    cell_w, cell_h = 300, 235
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, (img, label) in enumerate(thumbs):
        x = (idx % cols) * cell_w
        y = (idx // cols) * cell_h
        sheet.paste(img, (x + 20, y + 10))
        draw.text((x + 8, y + 205), label[:52], fill="black", font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def make_roi_diagnostic(path: Path, roi: dict[str, object], out_path: Path) -> None:
    arr = image_array(path)
    if arr.ndim == 2:
        gray = normalize_for_detection(arr)
        base = Image.fromarray(np.uint8(gray * 255), mode="L").convert("RGB")
    else:
        base = Image.fromarray(np.uint8(normalize_for_detection(arr) * 255), mode="L").convert("RGB")
    view = ImageOps.contain(base, (420, 300))
    sx = view.width / float(roi["width"])
    sy = view.height / float(roi["height"])
    mask = Image.fromarray(np.uint8(roi["_mask"]) * 255, mode="L").resize(view.size, Image.Resampling.NEAREST)
    panel_mask = Image.fromarray(np.uint8(roi["_panel_mask"]) * 255, mode="L").resize(view.size, Image.Resampling.NEAREST)
    excluded_mask = Image.fromarray(np.uint8(roi["_excluded_mask"]) * 255, mode="L").resize(view.size, Image.Resampling.NEAREST)
    panel_bool = np.asarray(panel_mask) > 0
    roi_bool = np.asarray(mask) > 0
    excl_bool = np.asarray(excluded_mask) > 0

    left = view.copy()
    draw = ImageDraw.Draw(left)
    panel_box = [int(roi["panel_x0"] * sx), int(roi["panel_y0"] * sy), int(roi["panel_x1"] * sx), int(roi["panel_y1"] * sy)]
    roi_box = [int(roi["x0"] * sx), int(roi["y0"] * sy), int(roi["x1"] * sx), int(roi["y1"] * sy)]
    draw.rectangle(panel_box, outline="cyan", width=2)
    draw.rectangle(roi_box, outline="lime", width=3)

    middle = _overlay_mask(view, roi_bool, (0, 255, 80), 105)
    right = _overlay_mask(view, panel_bool, (0, 180, 255), 70)
    right = _overlay_mask(right, excl_bool, (255, 80, 0), 125)

    font = ImageFont.load_default()
    pad = 16
    title_h = 28
    cell_w = view.width
    cell_h = view.height + title_h
    sheet = Image.new("RGB", (cell_w * 3 + pad * 4, cell_h + 78), "white")
    d = ImageDraw.Draw(sheet)
    labels = ["candidate panel + clean ROI", "clean ROI mask", "excluded edge/wrinkle"]
    for idx, (img, label) in enumerate(zip([left, middle, right], labels)):
        x = pad + idx * (cell_w + pad)
        d.text((x, 8), label, fill="black", font=font)
        sheet.paste(img, (x, title_h))
    def _fmt(value: object, digits: int = 3) -> str:
        try:
            return f"{float(value):.{digits}f}"
        except Exception:  # noqa: BLE001
            return str(value)

    status = (
        f"{path.name} | qc={roi['qc_status']} | pixels={roi['roi_pixel_count']} | "
        f"cv={_fmt(roi['roi_cv'])} | excluded={_fmt(roi['excluded_edge_or_wrinkle_fraction'], 2)}"
    )
    d.text((pad, cell_h + 42), status[:190], fill="black", font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def make_diagnostic_contact_sheet(diag_paths: list[Path], out_path: Path) -> None:
    thumbs: list[Image.Image] = []
    for path in diag_paths:
        with Image.open(path) as img:
            thumbs.append(ImageOps.contain(img.convert("RGB"), (560, 190)))
    if not thumbs:
        return
    cols = 1
    cell_w, cell_h = 590, 205
    rows = len(thumbs)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    for idx, img in enumerate(thumbs):
        y = idx * cell_h
        sheet.paste(img, (15, y + 8))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def exif_summary(path: Path) -> dict[str, object]:
    result = {
        "make": "",
        "model": "",
        "datetime_original": "",
        "exposure_time": "",
        "f_number": "",
        "iso": "",
    }
    try:
        if path.suffix.lower() not in {".jpg", ".jpeg"}:
            return result
        with Image.open(path) as img:
            exif = img.getexif()
            for key, value in exif.items():
                name = Image.ExifTags.TAGS.get(key, str(key)) if hasattr(Image, "ExifTags") else str(key)
                if name == "Make":
                    result["make"] = str(value)
                elif name == "Model":
                    result["model"] = str(value)
                elif name == "DateTimeOriginal":
                    result["datetime_original"] = str(value)
                elif name == "ExposureTime":
                    result["exposure_time"] = str(value)
                elif name == "FNumber":
                    result["f_number"] = str(value)
                elif name in {"ISOSpeedRatings", "PhotographicSensitivity"}:
                    result["iso"] = str(value)
    except Exception as exc:  # noqa: BLE001
        result["metadata_error"] = str(exc)
    return result


def tier_for_gap(row: dict[str, str]) -> str:
    status = row.get("panel_status_le40min", "")
    scene = row.get("scene", "")
    before_gap = row.get("best_before_gap_min", "")
    after_gap = row.get("best_after_gap_min", "")
    gaps = [float(g) for g in [before_gap, after_gap] if str(g).strip()]
    min_gap = min(gaps) if gaps else 9999.0
    if scene in {"maize_02_20260526_1658", "cassava_01_20260526_1603", "chunya_01_20260526_1021", "eucalyptus_01_20260526_1053_pruned"}:
        return "Tier A strong"
    if min_gap <= 40 and status != "no close panel":
        return "Tier B usable_with_caution"
    if row.get("same_session_bracket_le120min") == "yes":
        return "Tier C metadata_only"
    return "No close panel"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib-root", action="append", required=True)
    parser.add_argument("--scene-panel-summary", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    roots = [Path(p) for p in args.calib_root]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_groups = discover_panel_groups(roots)
    inventory_rows: list[dict[str, object]] = []
    roi_rows: list[dict[str, object]] = []
    stats_rows: list[dict[str, object]] = []
    factor_rows: list[dict[str, object]] = []
    diagnostic_paths: list[Path] = []
    diagnostic_paths_by_group: dict[str, list[Path]] = {}
    diagnostics_dir = out_dir / "roi_diagnostics_v2"

    for group, root, folder, records in panel_groups:
        by_seq: dict[str, set[str]] = {}
        for record in records:
            by_seq.setdefault(record.seq, set()).add(record.band)
        complete_sequences = [seq for seq, bands in by_seq.items() if {"RGB", "G", "R", "RE", "NIR"}.issubset(bands)]
        start = min(r.timestamp for r in records)
        end = max(r.timestamp for r in records)
        bands_all = sorted({r.band for r in records})
        inventory_rows.append(
            {
                "panel_group": group,
                "panel_dir_name": folder.name,
                "calib_root": str(root),
                "panel_path": str(folder),
                "start": start.isoformat(sep=" "),
                "end": end.isoformat(sep=" "),
                "image_count": len(records),
                "rgb_count": sum(1 for r in records if r.band == "RGB"),
                "bands_present": "|".join(bands_all),
                "complete_rgb_g_r_re_nir_sequence_count": len(complete_sequences),
                "complete_sequences": "|".join(complete_sequences),
                "has_mrk": int(any(p.suffix.lower() == ".mrk" for p in folder.iterdir() if p.is_file())),
                "has_ppk": int(any(p.suffix.lower() in {".nav", ".obs", ".bin"} for p in folder.iterdir() if p.is_file())),
            }
        )
        representative_seq = complete_sequences[0] if complete_sequences else sorted(by_seq.keys())[0]
        for record in records:
            is_repr = int(record.seq == representative_seq and record.band in {"G", "R", "RE", "NIR"})
            metadata = exif_summary(record.path) if record.band in {"RGB", "F"} else {}
            if is_repr:
                roi = propose_roi(record.path)
                diag_name = f"{_sanitize_name(group)}__seq{record.seq}__{record.band}__{_sanitize_name(record.path.stem)}.png"
                diag_path = diagnostics_dir / diag_name
                make_roi_diagnostic(record.path, roi, diag_path)
                diagnostic_paths.append(diag_path)
                diagnostic_paths_by_group.setdefault(group, []).append(diag_path)
                roi_row = {
                    "panel_group": group,
                    "panel_dir_name": folder.name,
                    "calib_root": str(root),
                    "file_path": str(record.path),
                    "file_name": record.path.name,
                    "timestamp": record.timestamp.isoformat(sep=" "),
                    "seq": record.seq,
                    "band": record.band,
                    "is_representative": is_repr,
                    "diagnostic_png": str(diag_path),
                    **_strip_private_roi(roi),
                }
                roi_rows.append(roi_row)
                stats_rows.append(
                    {
                        "panel_group": group,
                        "seq": record.seq,
                        "band": record.band,
                        "file_name": record.path.name,
                        "roi_qc_status": roi["qc_status"],
                        "roi_confidence": roi["confidence"],
                        "requires_manual_roi": roi["requires_manual_roi"],
                        "reject_reason": roi["reject_reason"],
                        "roi_pixel_count": roi["roi_pixel_count"],
                        "roi_to_panel_area_ratio": roi["roi_to_panel_area_ratio"],
                        "excluded_edge_or_wrinkle_fraction": roi["excluded_edge_or_wrinkle_fraction"],
                        "saturation_fraction": roi["saturation_fraction"],
                        "gradient_p95": roi["gradient_p95"],
                        "roi_cv": roi["roi_cv"],
                        "median": roi["median"],
                        "mean": roi["mean"],
                        "std": roi["std"],
                    }
                )
                if record.band in {"G", "R", "RE", "NIR"}:
                    median_raw = roi["median"]
                    try:
                        median = float(median_raw)
                    except Exception:  # noqa: BLE001
                        median = 0.0
                    scale_value: float | str = ""
                    if median > 0 and roi["qc_status"] != "reject":
                        scale_value = 0.8 / median
                    factor_rows.append(
                        {
                            "panel_group": group,
                            "seq": record.seq,
                            "band": record.band,
                            "median": median_raw,
                            "preliminary_scale_0p8_over_median": scale_value,
                            "roi_qc_status": roi["qc_status"],
                            "roi_confidence": roi["confidence"],
                            "requires_manual_roi": roi["requires_manual_roi"],
                            "use_for_formal_eval": "no",
                            "notes": "preliminary single-reference normalization factor; not used in manuscript conclusions",
                        }
                    )
            else:
                # Keep one lightweight metadata row per non-representative file in inventory only.
                pass
            if metadata:
                inventory_rows[-1].setdefault("sample_make", metadata.get("make", ""))
                inventory_rows[-1].setdefault("sample_model", metadata.get("model", ""))
                inventory_rows[-1].setdefault("sample_exposure_time", metadata.get("exposure_time", ""))
                inventory_rows[-1].setdefault("sample_f_number", metadata.get("f_number", ""))
                inventory_rows[-1].setdefault("sample_iso", metadata.get("iso", ""))

    scene_rows = read_csv_dicts(Path(args.scene_panel_summary))
    tier_rows: list[dict[str, object]] = []
    for row in scene_rows:
        tier_rows.append(
            {
                "scene": row.get("scene", ""),
                "scene_start": row.get("scene_start", ""),
                "scene_end": row.get("scene_end", ""),
                "panel_status_le40min": row.get("panel_status_le40min", ""),
                "tier": tier_for_gap(row),
                "best_before_panel_le40min": row.get("best_before_panel_le40min", ""),
                "best_after_panel_le40min": row.get("best_after_panel_le40min", ""),
                "same_session_previous_panel": row.get("same_session_previous_panel", ""),
                "same_session_next_panel": row.get("same_session_next_panel", ""),
                "same_session_bracket_le120min": row.get("same_session_bracket_le120min", ""),
            }
        )

    write_csv(
        out_dir / "panel_inventory.csv",
        inventory_rows,
        [
            "panel_group",
            "panel_dir_name",
            "calib_root",
            "panel_path",
            "start",
            "end",
            "image_count",
            "rgb_count",
            "bands_present",
            "complete_rgb_g_r_re_nir_sequence_count",
            "complete_sequences",
            "has_mrk",
            "has_ppk",
            "sample_make",
            "sample_model",
            "sample_exposure_time",
            "sample_f_number",
            "sample_iso",
        ],
    )
    write_csv(
        out_dir / "panel_time_gap_tiers.csv",
        tier_rows,
        [
            "scene",
            "scene_start",
            "scene_end",
            "panel_status_le40min",
            "tier",
            "best_before_panel_le40min",
            "best_after_panel_le40min",
            "same_session_previous_panel",
            "same_session_next_panel",
            "same_session_bracket_le120min",
        ],
    )
    write_csv(
        out_dir / "panel_roi_manifest.csv",
        roi_rows,
        [
            "panel_group",
            "panel_dir_name",
            "calib_root",
            "file_path",
            "file_name",
            "timestamp",
            "seq",
            "band",
            "is_representative",
            "diagnostic_png",
            "x0",
            "y0",
            "x1",
            "y1",
            "panel_x0",
            "panel_y0",
            "panel_x1",
            "panel_y1",
            "width",
            "height",
            "method",
            "roi_area_fraction",
            "panel_area_fraction",
            "roi_to_panel_area_ratio",
            "roi_pixel_count",
            "excluded_edge_or_wrinkle_fraction",
            "saturation_fraction",
            "gradient_p95",
            "roi_cv",
            "qc_status",
            "confidence",
            "requires_manual_roi",
            "reject_reason",
            "median",
            "mean",
            "std",
        ],
    )
    write_csv(
        out_dir / "panel_response_stats.csv",
        stats_rows,
        [
            "panel_group",
            "seq",
            "band",
            "file_name",
            "roi_qc_status",
            "roi_confidence",
            "requires_manual_roi",
            "reject_reason",
            "roi_pixel_count",
            "roi_to_panel_area_ratio",
            "excluded_edge_or_wrinkle_fraction",
            "saturation_fraction",
            "gradient_p95",
            "roi_cv",
            "median",
            "mean",
            "std",
        ],
    )
    write_csv(
        out_dir / "panel_normalization_factors.csv",
        factor_rows,
        [
            "panel_group",
            "seq",
            "band",
            "median",
            "preliminary_scale_0p8_over_median",
            "roi_qc_status",
            "roi_confidence",
            "requires_manual_roi",
            "use_for_formal_eval",
            "notes",
        ],
    )
    make_contact_sheet(roi_rows, out_dir / "panel_roi_contact_sheet.png")
    make_diagnostic_contact_sheet(diagnostic_paths, out_dir / "panel_roi_diagnostic_contact_sheet_v2.png")
    group_sheet_dir = out_dir / "panel_roi_diagnostic_group_sheets_v2"
    for group, paths in diagnostic_paths_by_group.items():
        make_diagnostic_contact_sheet(paths, group_sheet_dir / f"{_sanitize_name(group)}.png")

    low_count = sum(1 for r in roi_rows if r.get("requires_manual_roi"))
    high_count = sum(1 for r in roi_rows if r.get("confidence") == "high")
    qc_counts: dict[str, int] = {}
    for row in roi_rows:
        qc_counts[str(row.get("qc_status", "unknown"))] = qc_counts.get(str(row.get("qc_status", "unknown")), 0) + 1
    md = [
        "# Reflectance-panel feasibility audit\n\n",
        "This audit is metadata and ROI feasibility only. It does not apply reflectance calibration, does not change UMGS checkpoints, and does not enter manuscript main results.\n\n",
        f"- Panel groups inventoried: `{len(inventory_rows)}`\n",
        f"- Representative ROI files checked: `{len(roi_rows)}`\n",
        "- ROI method: per-band bright-panel detection followed by inward erosion and low-gradient interior masking. The mask intentionally excludes panel borders, wood supports, wrinkles, and near-saturated highlights.\n",
        f"- High-confidence automatic ROIs: `{high_count}`\n",
        f"- ROIs requiring manual review: `{low_count}`\n\n",
        "## ROI QC counts\n",
        *[f"- `{key}`: `{value}`\n" for key, value in sorted(qc_counts.items())],
        "\n",
        "## Output files\n",
        "- `panel_inventory.csv`\n",
        "- `panel_time_gap_tiers.csv`\n",
        "- `panel_roi_manifest.csv`\n",
        "- `panel_response_stats.csv`\n",
        "- `panel_normalization_factors.csv`\n",
        "- `panel_roi_contact_sheet.png`\n\n",
        "- `panel_roi_diagnostic_contact_sheet_v2.png`\n",
        "- `panel_roi_diagnostic_group_sheets_v2/*.png`\n",
        "- `roi_diagnostics_v2/*.png`\n\n",
        "## Use boundary\n",
        "The scale factors are preliminary single-reference normalization factors using `0.8 / panel_median_b`. They are not certified reflectance calibration and should not be used for formal paper conclusions before manual ROI review.\n",
    ]
    (out_dir / "panel_audit_summary.md").write_text("".join(md), encoding="utf-8")
    (out_dir / "panel_audit_summary.json").write_text(
        json.dumps(
            {
                "panel_group_count": len(inventory_rows),
                "representative_roi_count": len(roi_rows),
                "high_confidence_roi_count": high_count,
                "requires_manual_roi_count": low_count,
                "qc_counts": qc_counts,
                "outputs": [
                    "panel_inventory.csv",
                    "panel_time_gap_tiers.csv",
                    "panel_roi_manifest.csv",
                    "panel_response_stats.csv",
                    "panel_normalization_factors.csv",
                    "panel_roi_contact_sheet.png",
                    "panel_roi_diagnostic_contact_sheet_v2.png",
                    "panel_roi_diagnostic_group_sheets_v2",
                    "roi_diagnostics_v2",
                    "panel_audit_summary.md",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"panel groups: {len(inventory_rows)}")
    print(f"roi rows: {len(roi_rows)} high={high_count} manual={low_count}")
    print(out_dir)


if __name__ == "__main__":
    main()
