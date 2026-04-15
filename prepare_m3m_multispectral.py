import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional

from utils.spectral_image_utils import load_image_preserve_dtype


RGB_RE = re.compile(r"^(?P<stem>.+?)_(?P<frame>\d{4})_D\.(?P<ext>jpe?g)$", re.IGNORECASE)
BAND_RE = re.compile(r"^(?P<stem>.+?)_(?P<frame>\d{4})_MS_(?P<band>G|R|RE|NIR)\.(?P<ext>tiff?|TIF)$", re.IGNORECASE)
REQUIRED_MODALITIES = ["RGB", "G", "R", "RE", "NIR"]
RAW_SCENE_ORDER = ["RGB", "G_raw", "R_raw", "RE_raw", "NIR_raw"]
BAND_TO_RAW_SCENE = {
    "G": "G_raw",
    "R": "R_raw",
    "RE": "RE_raw",
    "NIR": "NIR_raw",
}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _iter_candidate_files(raw_root: Path) -> Iterable[Path]:
    for path in sorted(raw_root.rglob("*")):
        if path.is_file():
            yield path


def _summarize_record(record: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    mappings = {
        "Make": "make",
        "Model": "model",
        "CameraModelName": "camera_model",
        "Camera Model Name": "camera_model",
        "BlackLevel": "black_level",
        "Black Level": "black_level",
        "ExposureTime": "exposure_time",
        "Exposure Time": "exposure_time",
        "ISO": "iso",
        "Gain": "gain",
        "Irradiance": "irradiance",
        "SolarIrradiance": "irradiance",
        "Solar Irradiance": "irradiance",
        "GpsStatus": "gps_status",
        "GPS Status": "gps_status",
    }
    for src_key, dst_key in mappings.items():
        if src_key in record and dst_key not in out:
            out[dst_key] = record[src_key]
    return out


def _run_exiftool(paths: List[Path], exiftool_executable: str) -> Dict[str, Dict[str, object]]:
    if not paths:
        return {}
    exe_path = shutil.which(exiftool_executable) or exiftool_executable
    records: Dict[str, Dict[str, object]] = {}
    chunk_size = 100
    for start in range(0, len(paths), chunk_size):
        chunk = paths[start:start + chunk_size]
        cmd = [exe_path, "-j", "-n"] + [str(p) for p in chunk]
        try:
            proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
            payload = json.loads(proc.stdout)
        except Exception:
            continue
        for record in payload:
            source_file = record.get("SourceFile")
            if not source_file:
                continue
            records[str(Path(source_file).resolve())] = _summarize_record(record)
    return records


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "symlink":
        os.symlink(src, dst)
        return
    if mode == "hardlink":
        os.link(src, dst)
        return
    raise ValueError(f"Unsupported link mode: {mode}")


def _resolve_sparse_dir(path: Path) -> Path:
    if (path / "cameras.bin").exists() or (path / "cameras.txt").exists():
        return path
    if (path / "sparse" / "0").exists():
        return path / "sparse" / "0"
    raise FileNotFoundError(f"Could not resolve sparse/0 from: {path}")


def _copy_sparse_to_scene(scene_dir: Path, sparse_source: Path) -> None:
    sparse_dir = _resolve_sparse_dir(sparse_source)
    dst = scene_dir / "sparse" / "0"
    if dst.exists():
        shutil.rmtree(dst.parent)
    _ensure_dir(dst)
    for item in sparse_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, dst / item.name)


def _scan_groups(raw_root: Path) -> Dict[str, Dict[str, Path]]:
    groups: Dict[str, Dict[str, Path]] = {}
    for path in _iter_candidate_files(raw_root):
        rgb_match = RGB_RE.match(path.name)
        if rgb_match:
            frame_id = rgb_match.group("frame")
            groups.setdefault(frame_id, {})["RGB"] = path
            continue
        band_match = BAND_RE.match(path.name)
        if band_match:
            frame_id = band_match.group("frame")
            band = band_match.group("band").upper()
            groups.setdefault(frame_id, {})[band] = path
    return groups


def prepare_m3m_dataset(raw_root: Path, out_root: Path, link_mode: str = "hardlink",
                        sparse_source: Optional[Path] = None, exiftool_executable: str = "exiftool") -> Dict[str, object]:
    groups = _scan_groups(raw_root)
    if not groups:
        raise FileNotFoundError(f"No DJI M3M frame groups found under: {raw_root}")

    complete_groups = {frame_id: items for frame_id, items in groups.items() if all(k in items for k in REQUIRED_MODALITIES)}
    missing_groups = {
        frame_id: sorted(set(REQUIRED_MODALITIES) - set(items.keys()))
        for frame_id, items in groups.items()
        if frame_id not in complete_groups
    }

    all_paths = []
    for items in complete_groups.values():
        all_paths.extend(items.values())
    metadata_map = _run_exiftool(all_paths, exiftool_executable)

    if out_root.exists():
        shutil.rmtree(out_root)
    _ensure_dir(out_root)

    scene_roots = {name: out_root / name for name in RAW_SCENE_ORDER}
    _ensure_dir(scene_roots["RGB"] / "input")
    if sparse_source is not None:
        _ensure_dir(scene_roots["RGB"] / "images")
    for scene_name in ("G_raw", "R_raw", "RE_raw", "NIR_raw"):
        _ensure_dir(scene_roots[scene_name] / "images")

    manifests: Dict[str, Dict[str, object]] = {}
    prepared_groups = []
    for scene_name in RAW_SCENE_ORDER:
        is_rgb = scene_name == "RGB"
        target_band = "" if is_rgb else scene_name.replace("_raw", "")
        manifests[scene_name] = {
            "scene_name": scene_name,
            "scene_root": str(scene_roots[scene_name]),
            "scene_kind": "rgb_anchor_source" if is_rgb else "raw_band_unrectified",
            "rectification_status": "rgb_native" if is_rgb else "raw",
            "trainable_with_rgb_sparse": True if is_rgb else False,
            "modality_kind": "rgb" if is_rgb else "band",
            "target_band": target_band,
            "carrier_mode": "native_rgb" if is_rgb else "replicated_scalar_rgb",
            "images": [],
        }

    for frame_id in sorted(complete_groups.keys()):
        items = complete_groups[frame_id]
        rgb_path = items["RGB"]
        group_payload = {
            "frame_id": frame_id,
            "paired_group_id": frame_id,
            "rgb_name": rgb_path.name,
            "sources": {band: str(path) for band, path in items.items()},
        }
        prepared_groups.append(group_payload)

        for modality_name in REQUIRED_MODALITIES:
            src = items[modality_name]
            if modality_name == "RGB":
                scene_name = "RGB"
                dst_dir = scene_roots["RGB"] / "input"
                dst_name = rgb_path.name
                modality_type = "rgb"
                band_name = None
            else:
                scene_name = BAND_TO_RAW_SCENE[modality_name]
                dst_dir = scene_roots[scene_name] / "images"
                dst_name = rgb_path.name
                modality_type = "scalar_band"
                band_name = modality_name

            dst = dst_dir / dst_name
            _link_or_copy(src, dst, link_mode)

            if modality_name == "RGB" and sparse_source is not None:
                rgb_images_dst = scene_roots["RGB"] / "images" / dst_name
                _link_or_copy(src, rgb_images_dst, link_mode)

            summary = metadata_map.get(str(src.resolve()), {})
            if not summary:
                try:
                    summary = load_image_preserve_dtype(src).metadata
                except Exception:
                    summary = {}

            manifests[scene_name]["images"].append({
                "image_name": dst_name,
                "source_path": str(src),
                "frame_id": frame_id,
                "paired_group_id": frame_id,
                "modality_type": modality_type,
                "band_name": band_name,
                "carrier_mode": "native_rgb" if modality_name == "RGB" else "replicated_scalar_rgb",
                "metadata": summary,
                "scene_kind": manifests[scene_name]["scene_kind"],
                "rectification_status": manifests[scene_name]["rectification_status"],
            })

    if sparse_source is not None:
        _copy_sparse_to_scene(scene_roots["RGB"], sparse_source)

    for scene_name, manifest in manifests.items():
        (scene_roots[scene_name] / "spectral_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    summary = {
        "raw_root": str(raw_root),
        "out_root": str(out_root),
        "scene_roots": {name: str(root) for name, root in scene_roots.items()},
        "paired_group_count": len(complete_groups),
        "missing_groups": missing_groups,
        "groups": prepared_groups,
        "sparse_source": str(sparse_source) if sparse_source is not None else "",
    }
    (out_root / "prepared_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare DJI M3M RGB + multispectral scenes for SpectralIndexGS.")
    ap.add_argument("--raw_root", required=True, help="Raw M3M dataset root.")
    ap.add_argument("--out_root", required=True, help="Prepared output root.")
    ap.add_argument("--link_mode", default="hardlink", choices=["copy", "hardlink", "symlink"])
    ap.add_argument("--sparse_source", default="", help="Optional sparse/0 source to copy into every prepared scene.")
    ap.add_argument("--exiftool_executable", default="exiftool", help="Optional exiftool path for metadata extraction.")
    args = ap.parse_args()

    summary = prepare_m3m_dataset(
        raw_root=Path(args.raw_root).resolve(),
        out_root=Path(args.out_root).resolve(),
        link_mode=args.link_mode,
        sparse_source=Path(args.sparse_source).resolve() if args.sparse_source else None,
        exiftool_executable=args.exiftool_executable,
    )
    print(json.dumps({
        "paired_group_count": summary["paired_group_count"],
        "missing_group_count": len(summary["missing_groups"]),
        "out_root": summary["out_root"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
