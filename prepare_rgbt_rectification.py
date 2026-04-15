from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

from utils.spectral_image_utils import load_image_preserve_dtype


def _copy_sparse(rgb_scene_root: Path, rgb_out_root: Path) -> None:
    src = rgb_scene_root / "sparse" / "0"
    if not src.exists():
        raise FileNotFoundError(f"Missing RGB sparse model: {src}")
    dst = rgb_out_root / "sparse" / "0"
    if dst.parent.exists():
        shutil.rmtree(dst.parent)
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, dst / item.name)


def _list_images(scene_root: Path) -> List[Path]:
    image_root = scene_root / "images"
    if not image_root.exists():
        raise FileNotFoundError(f"Missing images directory: {image_root}")
    files = [p for p in image_root.iterdir() if p.is_file()]
    files = [p for p in files if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")]
    return sorted(files, key=lambda p: p.name)


def _build_item(image_name: str, image_path: Path, frame_id: str, paired_group_id: str, modality_type: str, band_name: str) -> Dict[str, object]:
    loaded = load_image_preserve_dtype(image_path)
    metadata = dict(loaded.metadata or {})
    metadata.update(
        {
            "source_path": str(image_path),
            "width": int(loaded.width),
            "height": int(loaded.height),
            "dtype": str(loaded.dtype_name),
            "channel_count": int(loaded.channel_count),
        }
    )
    return {
        "image_name": image_name,
        "source_path": str(image_path),
        "frame_id": frame_id,
        "paired_group_id": paired_group_id,
        "modality_type": modality_type,
        "band_name": band_name,
        "carrier_mode": "native_rgb" if modality_type == "rgb" else "replicated_scalar_rgb",
        "metadata": metadata,
    }


def prepare_rgbt_rectification(
    rgb_scene_root: Path,
    thermal_scene_root: Path,
    prepared_root: Path,
    thermal_band_name: str = "T",
) -> Dict[str, object]:
    rgb_scene_root = rgb_scene_root.resolve()
    thermal_scene_root = thermal_scene_root.resolve()
    prepared_root = prepared_root.resolve()
    thermal_band_name = str(thermal_band_name).strip() or "T"

    rgb_files = _list_images(rgb_scene_root)
    thermal_files = _list_images(thermal_scene_root)
    rgb_map = {p.name: p for p in rgb_files}
    thermal_map = {p.name: p for p in thermal_files}
    common = sorted(set(rgb_map.keys()) & set(thermal_map.keys()))
    if not common:
        raise RuntimeError(
            f"No paired image names between RGB and thermal scenes:\n"
            f"  RGB: {rgb_scene_root}\n"
            f"  TH:  {thermal_scene_root}"
        )

    rgb_out = prepared_root / "RGB"
    thermal_out = prepared_root / f"{thermal_band_name}_raw"
    if prepared_root.exists():
        shutil.rmtree(prepared_root)
    (rgb_out / "images").mkdir(parents=True, exist_ok=True)
    (thermal_out / "images").mkdir(parents=True, exist_ok=True)

    _copy_sparse(rgb_scene_root, rgb_out)

    rgb_items = []
    thermal_items = []
    for idx, image_name in enumerate(common):
        frame_id = Path(image_name).stem
        paired_group_id = frame_id

        rgb_src = rgb_map[image_name]
        thermal_src = thermal_map[image_name]
        rgb_dst = rgb_out / "images" / image_name
        thermal_dst = thermal_out / "images" / image_name
        shutil.copy2(rgb_src, rgb_dst)
        shutil.copy2(thermal_src, thermal_dst)

        rgb_items.append(
            _build_item(
                image_name=image_name,
                image_path=rgb_dst,
                frame_id=frame_id,
                paired_group_id=paired_group_id,
                modality_type="rgb",
                band_name="",
            )
        )
        thermal_items.append(
            _build_item(
                image_name=image_name,
                image_path=thermal_dst,
                frame_id=frame_id,
                paired_group_id=paired_group_id,
                modality_type="rgb",
                band_name=thermal_band_name,
            )
        )

    rgb_manifest = {
        "scene_name": "RGB",
        "scene_root": str(rgb_out),
        "scene_kind": "rgb_anchor_source",
        "rectification_status": "rgb_native",
        "trainable_with_rgb_sparse": True,
        "modality_kind": "rgb",
        "carrier_mode": "native_rgb",
        "images": rgb_items,
    }
    thermal_manifest = {
        "scene_name": f"{thermal_band_name}_raw",
        "scene_root": str(thermal_out),
        "scene_kind": "raw_band_unrectified",
        "rectification_status": "raw",
        "trainable_with_rgb_sparse": False,
        "modality_kind": "thermal",
        "target_band": thermal_band_name,
        "carrier_mode": "native_rgb",
        "images": thermal_items,
    }

    (rgb_out / "spectral_manifest.json").write_text(json.dumps(rgb_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (thermal_out / "spectral_manifest.json").write_text(json.dumps(thermal_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "prepared_root": str(prepared_root),
        "rgb_scene_root": str(rgb_scene_root),
        "thermal_scene_root": str(thermal_scene_root),
        "thermal_band_name": thermal_band_name,
        "paired_frames": len(common),
    }
    (prepared_root / "prepared_manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare RGB+thermal paired scenes for MINIMA-based rectification.")
    ap.add_argument("--rgb_scene_root", required=True, help="Scene root containing RGB images + sparse/0.")
    ap.add_argument("--thermal_scene_root", required=True, help="Scene root containing thermal images (same image names).")
    ap.add_argument("--prepared_root", required=True, help="Output root with RGB and <band>_raw manifests for rectification.")
    ap.add_argument("--thermal_band_name", default="T", help="Band label for thermal modality (default: T).")
    args = ap.parse_args()

    summary = prepare_rgbt_rectification(
        rgb_scene_root=Path(args.rgb_scene_root),
        thermal_scene_root=Path(args.thermal_scene_root),
        prepared_root=Path(args.prepared_root),
        thermal_band_name=args.thermal_band_name,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
