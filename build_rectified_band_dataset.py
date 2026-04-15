from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image

from utils.rectification_utils import (
    build_validity_mask_from_warp,
    save_scalar_tiff_with_sidecar,
    warp_band_to_rgb_plane,
)
from utils.spectral_image_utils import load_image_preserve_dtype


DEFAULT_BANDS = ("G", "R", "RE", "NIR")


def _parse_bands(bands) -> Tuple[str, ...]:
    if bands is None:
        return tuple(DEFAULT_BANDS)
    if isinstance(bands, str):
        items = [item.strip() for item in bands.split(",")]
    else:
        items = [str(item).strip() for item in bands]
    parsed = []
    for item in items:
        if not item:
            continue
        if item not in parsed:
            parsed.append(item)
    if not parsed:
        raise ValueError("No valid bands provided to build_rectified_band_dataset.")
    return tuple(parsed)


def _load_manifest(scene_root: Path) -> Dict[str, object]:
    manifest_path = scene_root / "spectral_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing spectral_manifest.json: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _copy_rgb_sparse_to_scene(rgb_scene_root: Path, rectified_scene_root: Path) -> None:
    src = rgb_scene_root / "sparse" / "0"
    if not src.exists():
        raise FileNotFoundError(f"Missing RGB sparse/0: {src}")
    dst = rectified_scene_root / "sparse" / "0"
    if dst.parent.exists():
        shutil.rmtree(dst.parent)
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, dst / item.name)


def _rgb_target_size(rgb_scene_root: Path, image_name: str) -> Tuple[int, int]:
    rgb_image_path = rgb_scene_root / "images" / image_name
    loaded = load_image_preserve_dtype(rgb_image_path)
    return loaded.width, loaded.height


def _clear_scene_root(scene_root: Path) -> None:
    if scene_root.exists():
        shutil.rmtree(scene_root)
    (scene_root / "images").mkdir(parents=True, exist_ok=True)
    (scene_root / "validity_masks").mkdir(parents=True, exist_ok=True)


def _warp_raw_array(raw_array: np.ndarray, homography: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    source = np.asarray(raw_array)
    source_dtype = source.dtype
    if source.ndim == 3 and source.shape[2] == 1:
        source = source[..., 0]
    if source.ndim not in (2, 3):
        raise ValueError(f"Expected 2D/3D raw array, got shape={source.shape}")

    if np.issubdtype(source_dtype, np.integer):
        working = source.astype(np.float32)
        warped = warp_band_to_rgb_plane(working, homography=homography, target_size=target_size, interpolation=cv2.INTER_LINEAR)
        info = np.iinfo(source_dtype)
        warped = np.clip(np.rint(warped), info.min, info.max).astype(source_dtype)
    else:
        working = source.astype(np.float32)
        warped = warp_band_to_rgb_plane(working, homography=homography, target_size=target_size, interpolation=cv2.INTER_LINEAR)
        warped = warped.astype(source_dtype)
    return np.ascontiguousarray(warped)


def _save_image_with_sidecar(path: Path, image_array: np.ndarray, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(image_array)
    if arr.ndim == 2:
        Image.fromarray(arr).save(path)
    elif arr.ndim == 3:
        Image.fromarray(arr).save(path)
    else:
        raise ValueError(f"Unsupported warped image shape for save: {arr.shape}")
    sidecar_path = path.with_suffix(path.suffix + ".meta.json")
    sidecar_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def build_rectified_band_dataset(prepared_root: Path,
                                 rectified_root: Path,
                                 homography_json: Path,
                                 bands: Tuple[str, ...] | str | None = None) -> Dict[str, object]:
    prepared_root = prepared_root.resolve()
    rectified_root = rectified_root.resolve()
    homography_json = homography_json.resolve()

    config = json.loads(homography_json.read_text(encoding="utf-8"))
    rgb_scene_root = prepared_root / "RGB"
    rgb_manifest = _load_manifest(rgb_scene_root)
    rgb_items = list(rgb_manifest.get("images", []))
    if not rgb_items:
        raise RuntimeError(f"No RGB images found in scene manifest: {rgb_scene_root}")

    band_list = _parse_bands(bands)

    summary = {
        "prepared_root": str(prepared_root),
        "rectified_root": str(rectified_root),
        "homography_json": str(homography_json),
        "bands_order": list(band_list),
        "bands": {},
    }

    for band in band_list:
        raw_scene_root = prepared_root / f"{band}_raw"
        raw_manifest = _load_manifest(raw_scene_root)
        image_map = {
            str(item.get("image_name", "")).strip(): item
            for item in raw_manifest.get("images", [])
            if str(item.get("image_name", "")).strip()
        }
        if band not in config.get("bands", {}):
            raise KeyError(f"Homography config missing band {band}: {homography_json}")
        band_config = config["bands"][band]
        homography = np.asarray(band_config.get("T_opt", band_config.get("H")), dtype=np.float64)

        rect_scene_root = rectified_root / f"{band}_rectified"
        _clear_scene_root(rect_scene_root)
        _copy_rgb_sparse_to_scene(rgb_scene_root, rect_scene_root)

        scene_manifest = {
            "scene_name": f"{band}_rectified",
            "scene_root": str(rect_scene_root),
            "scene_kind": "rectified_band",
            "rectification_status": "rectified",
            "trainable_with_rgb_sparse": True,
            "modality_kind": "band",
            "target_band": band,
            "carrier_mode": "replicated_scalar_rgb",
            "rectification_method": config.get("rectification_method", "fixed_homography_ecc"),
            "global_homography_ref": str(homography_json),
            "transform_mode": str(config.get("transform_mode", "legacy_fixed_h")),
            "images": [],
        }

        for rgb_item in rgb_items:
            image_name = str(rgb_item.get("image_name", "")).strip()
            if not image_name:
                continue
            raw_item = image_map.get(image_name)
            if raw_item is None:
                raise KeyError(f"Raw scene {raw_scene_root.name} missing frame {image_name}")

            source_path = Path(str(raw_item.get("source_path") or (raw_scene_root / "images" / image_name)))
            loaded = load_image_preserve_dtype(source_path)
            target_size = _rgb_target_size(rgb_scene_root, image_name)
            warped = _warp_raw_array(loaded.array, homography=homography, target_size=target_size)
            mask = build_validity_mask_from_warp(loaded.array.shape[:2], homography=homography, target_size=target_size)
            is_scalar = (loaded.channel_count == 1) or (np.asarray(warped).ndim == 2) or (np.asarray(warped).ndim == 3 and np.asarray(warped).shape[2] == 1)

            image_out_path = rect_scene_root / "images" / image_name
            mask_out_path = rect_scene_root / "validity_masks" / f"{Path(image_name).stem}.png"
            sidecar = dict(loaded.metadata or {})
            sidecar.update(
                {
                    "source_path": str(source_path),
                    "rectification_status": "rectified",
                    "scene_kind": "rectified_band",
                    "band_name": band,
                    "rectification_method": scene_manifest["rectification_method"],
                    "transform_mode": scene_manifest["transform_mode"],
                }
            )
            if is_scalar:
                save_scalar_tiff_with_sidecar(image_out_path, warped, sidecar)
            else:
                _save_image_with_sidecar(image_out_path, warped, sidecar)
            cv2.imwrite(str(mask_out_path), (mask * 255.0).astype(np.uint8))

            scene_manifest["images"].append(
                {
                    "image_name": image_name,
                    "source_path": str(source_path),
                    "rectified_path": str(image_out_path),
                    "validity_mask_path": str(mask_out_path),
                    "frame_id": raw_item.get("frame_id", ""),
                    "paired_group_id": raw_item.get("paired_group_id", ""),
                    "modality_type": "scalar_band" if is_scalar else "rgb",
                    "band_name": band,
                    "carrier_mode": "replicated_scalar_rgb" if is_scalar else "native_rgb",
                    "metadata": sidecar,
                    "scene_kind": "rectified_band",
                    "rectification_status": "rectified",
                    "transform_mode": scene_manifest["transform_mode"],
                }
            )

        (rect_scene_root / "spectral_manifest.json").write_text(
            json.dumps(scene_manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        summary["bands"][band] = {
            "scene_root": str(rect_scene_root),
            "image_count": len(scene_manifest["images"]),
        }

    (rectified_root / "rectified_dataset_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Warp raw band scenes into the RGB training plane and build rectified band datasets.")
    ap.add_argument("--prepared_root", required=True, help="Prepared raw root containing RGB and *_raw scenes.")
    ap.add_argument("--rectified_root", required=True, help="Root where *_rectified scenes will be written.")
    ap.add_argument("--homography_json", required=True, help="Fixed homography JSON produced by estimate_band_homographies.py.")
    ap.add_argument("--bands", default="G,R,RE,NIR", help="Comma-separated modality list to build (e.g., G,R,RE,NIR or T).")
    args = ap.parse_args()

    summary = build_rectified_band_dataset(
        prepared_root=Path(args.prepared_root),
        rectified_root=Path(args.rectified_root),
        homography_json=Path(args.homography_json),
        bands=args.bands,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
