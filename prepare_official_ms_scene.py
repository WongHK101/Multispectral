from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from plyfile import PlyData, PlyElement


CHANNEL_TO_SCENE = {
    "D": ("RGB", "rgb", "", "native_rgb"),
    "MS_G": ("G_aligned", "band", "G", "replicated_scalar_rgb"),
    "MS_R": ("R_aligned", "band", "R", "replicated_scalar_rgb"),
    "MS_RE": ("RE_aligned", "band", "RE", "replicated_scalar_rgb"),
    "MS_NIR": ("NIR_aligned", "band", "NIR", "replicated_scalar_rgb"),
}


def _load_loose_json(path: Path) -> Dict[str, object]:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r",(\s*[\]}])", r"\1", text)
    return json.loads(text)


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        raise ValueError(f"Unsupported link mode: {mode}")


def _write_repo_compatible_ply(src: Path, dst: Path) -> None:
    """Normalize official sparse_pc.ply to the x/y/z + normals + uint8 RGB schema."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    ply = PlyData.read(str(src))
    vertices = ply["vertex"].data
    names = vertices.dtype.names or ()
    required_xyz = {"x", "y", "z"}
    if not required_xyz.issubset(names):
        raise RuntimeError(f"Official PLY is missing xyz fields: {src}")

    count = len(vertices)
    out_dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    out = np.empty(count, dtype=out_dtype)
    for key in ("x", "y", "z"):
        out[key] = vertices[key].astype(np.float32, copy=False)
    for key in ("nx", "ny", "nz"):
        out[key] = vertices[key].astype(np.float32, copy=False) if key in names else 0.0
    for key in ("red", "green", "blue"):
        if key in names:
            out[key] = np.clip(vertices[key], 0, 255).astype(np.uint8, copy=False)
        else:
            out[key] = np.uint8(127)
    PlyData([PlyElement.describe(out, "vertex")]).write(str(dst))


def _frame_path_for_image_root(frame_path: str, image_root: str) -> str:
    parts = Path(frame_path.replace("\\", "/")).parts
    if len(parts) >= 3 and parts[0] == "images":
        return str(Path(image_root, *parts[1:])).replace("\\", "/")
    return frame_path.replace("images/", f"{image_root}/", 1)


def _make_transform_payload(frames: List[Dict[str, object]]) -> Dict[str, object]:
    if not frames:
        raise RuntimeError("Cannot build transforms payload from empty frame list.")
    first = frames[0]
    if "camera_angle_x" in first:
        camera_angle_x = float(first["camera_angle_x"])
    else:
        w = float(first.get("w", 0) or 0)
        fl_x = float(first.get("fl_x", 0) or 0)
        if w <= 0 or fl_x <= 0:
            raise RuntimeError("Official MS frame is missing w/fl_x needed for camera_angle_x.")
        camera_angle_x = 2.0 * math.atan(w / (2.0 * fl_x))
    return {
        "camera_angle_x": camera_angle_x,
        "frames": frames,
    }


def _split_paths_by_channel(split_items: Iterable[object]) -> Dict[str, set]:
    out = {channel: set() for channel in CHANNEL_TO_SCENE}
    for item in split_items:
        if isinstance(item, dict):
            value = item.get("official_path") or item.get("images2_path") or item.get("path") or item.get("file_path")
        else:
            value = item
        rel = str(value).replace("\\", "/")
        for channel in CHANNEL_TO_SCENE:
            token = f"/{channel}/"
            if token in f"/{rel}":
                out[channel].add(Path(rel).name)
                break
    return out


def _group_id_from_image_name(image_name: str) -> str:
    return Path(image_name).stem.split("_")[-1]


def _split_group_ids_by_channel(split_items: Iterable[object]) -> Dict[str, set]:
    by_name = _split_paths_by_channel(split_items)
    return {
        channel: {_group_id_from_image_name(name) for name in names}
        for channel, names in by_name.items()
    }


def prepare_official_ms_scene(source_root: Path, out_root: Path, image_root: str, split_json: Path, link_mode: str = "hardlink") -> Dict[str, object]:
    source_root = source_root.resolve()
    out_root = out_root.resolve()
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    transforms = _load_loose_json(source_root / "transforms.json")
    split = _load_loose_json(split_json)
    train_group_ids_by_channel = _split_group_ids_by_channel(split.get("train", []))
    test_group_ids_by_channel = _split_group_ids_by_channel(split.get("test", split.get("eval", [])))
    common_train_group_ids = set.intersection(*train_group_ids_by_channel.values())
    common_test_group_ids = set.intersection(*test_group_ids_by_channel.values())
    if not common_train_group_ids or not common_test_group_ids:
        raise RuntimeError(
            "Official split did not contain complete RGB/G/R/RE/NIR group intersections: "
            f"train={len(common_train_group_ids)}, test={len(common_test_group_ids)}"
        )

    source_ply = source_root / "sparse_pc.ply"
    if not source_ply.exists():
        raise FileNotFoundError(f"Missing official sparse point cloud: {source_ply}")

    available_group_ids_by_channel = {channel: set() for channel in CHANNEL_TO_SCENE}
    for frame in transforms.get("frames", []):
        channel = str(frame.get("mm_channel", ""))
        if channel not in CHANNEL_TO_SCENE:
            continue
        rel = _frame_path_for_image_root(str(frame["file_path"]), image_root)
        image_path = source_root / rel
        if image_path.exists():
            available_group_ids_by_channel[channel].add(_group_id_from_image_name(image_path.name))
    common_available_group_ids = set.intersection(*available_group_ids_by_channel.values())
    common_train_group_ids &= common_available_group_ids
    common_test_group_ids &= common_available_group_ids
    if not common_train_group_ids or not common_test_group_ids:
        raise RuntimeError(
            "Official split had no complete RGB/G/R/RE/NIR group intersections after checking available images: "
            f"train={len(common_train_group_ids)}, test={len(common_test_group_ids)}"
        )

    summary = {
        "source_root": str(source_root),
        "out_root": str(out_root),
        "image_root": image_root,
        "split_json": str(split_json.resolve()),
        "split_group_policy": "complete_rgb_g_r_re_nir_group_intersection",
        "common_train_group_count": len(common_train_group_ids),
        "common_test_group_count": len(common_test_group_ids),
        "common_available_group_count": len(common_available_group_ids),
        "scenes": {},
    }

    for channel, (scene_name, modality_kind, band_name, carrier_mode) in CHANNEL_TO_SCENE.items():
        scene_root = out_root / scene_name
        scene_root.mkdir(parents=True, exist_ok=True)
        _write_repo_compatible_ply(source_ply, scene_root / "points3d.ply")

        channel_frames = []
        for frame in transforms.get("frames", []):
            if str(frame.get("mm_channel", "")) != channel:
                continue
            updated = dict(frame)
            rel = _frame_path_for_image_root(str(frame["file_path"]), image_root)
            image_path = source_root / rel
            if not image_path.exists():
                raise FileNotFoundError(f"Missing official MS image: {image_path}")
            updated["file_path"] = str(image_path)
            channel_frames.append(updated)

        frames_by_group_id: Dict[str, Dict[str, object]] = {}
        for frame in channel_frames:
            group_id = _group_id_from_image_name(Path(str(frame["file_path"])).name)
            frames_by_group_id.setdefault(group_id, frame)
        train_frames = [
            frames_by_group_id[group_id]
            for group_id in sorted(common_train_group_ids)
            if group_id in frames_by_group_id
        ]
        test_frames = [
            frames_by_group_id[group_id]
            for group_id in sorted(common_test_group_ids)
            if group_id in frames_by_group_id
        ]
        if not train_frames or not test_frames:
            raise RuntimeError(
                f"Official split did not yield train/test frames for {channel}: "
                f"train={len(train_frames)}, test={len(test_frames)}"
            )
        if len(train_frames) != len(common_train_group_ids) or len(test_frames) != len(common_test_group_ids):
            raise RuntimeError(
                f"Official split/transform mismatch for {channel}: "
                f"train={len(train_frames)}/{len(common_train_group_ids)}, "
                f"test={len(test_frames)}/{len(common_test_group_ids)}"
            )

        (scene_root / "transforms_train.json").write_text(
            json.dumps(_make_transform_payload(train_frames), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (scene_root / "transforms_test.json").write_text(
            json.dumps(_make_transform_payload(test_frames), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        manifest_images = []
        for frame in train_frames + test_frames:
            image_path = Path(str(frame["file_path"]))
            manifest_images.append(
                {
                    "image_name": image_path.name,
                    "source_path": str(image_path),
                    "frame_id": image_path.stem.split("_")[-1],
                    "paired_group_id": image_path.stem.split("_")[-1],
                    "modality_type": "rgb" if modality_kind == "rgb" else "scalar_band",
                    "band_name": band_name,
                    "carrier_mode": carrier_mode,
                    "metadata": {
                        "source_dataset": "ms-splatting-dataset",
                        "source_channel": channel,
                        "image_root": image_root,
                    },
                }
            )
        manifest = {
            "scene_name": scene_name,
            "scene_root": str(scene_root),
            "scene_kind": "official_ms_aligned",
            "rectification_status": "aligned_by_dataset",
            "trainable_with_rgb_sparse": True,
            "modality_kind": modality_kind,
            "target_band": band_name,
            "carrier_mode": carrier_mode,
            "images": manifest_images,
        }
        (scene_root / "spectral_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        summary["scenes"][scene_name] = {
            "scene_root": str(scene_root),
            "channel": channel,
            "train_count": len(train_frames),
            "test_count": len(test_frames),
        }

    (out_root / "official_ms_prepared_manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare official MS-Splatting scenes into repo-native aligned band scenes.")
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--image_root", default="images_2", choices=["images", "images_2", "images_4"])
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--link_mode", default="hardlink", choices=["copy", "hardlink", "symlink"])
    args = parser.parse_args()

    summary = prepare_official_ms_scene(
        source_root=Path(args.source_root),
        out_root=Path(args.out_root),
        image_root=args.image_root,
        split_json=Path(args.split_json),
        link_mode=args.link_mode,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
