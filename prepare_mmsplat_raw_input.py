from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image


def log_info(msg: str) -> None:
    print(f"INFO: {msg}", flush=True)


def log_warn(msg: str) -> None:
    print(f"WARNING: {msg}", flush=True)


RGB_FRAME_RE = re.compile(r"^(?P<stem>.+?)_(?P<frame>\d{4})_D\.[^.]+$", re.IGNORECASE)
MS_FRAME_RE = re.compile(r"^(?P<stem>.+?)_(?P<frame>\d{4})_MS_(?P<band>G|R|RE|NIR)\.[^.]+$", re.IGNORECASE)


def _capture_key(path: Path, channel: str) -> str | None:
    name = path.name
    if channel == "D":
        m = RGB_FRAME_RE.match(name)
        if not m:
            return None
        return f"{m.group('stem')}_{m.group('frame')}"
    m = MS_FRAME_RE.match(name)
    if not m:
        return None
    return f"{m.group('stem')}_{m.group('frame')}"


def _frame_id(path: Path, channel: str) -> str | None:
    """Return the camera frame counter without assuming equal sensor timestamps."""
    name = path.name
    if channel == "D":
        m = RGB_FRAME_RE.match(name)
    else:
        m = MS_FRAME_RE.match(name)
    return m.group("frame") if m else None


def _flat_channel_files(input_root: Path, channel: str) -> List[Path]:
    files = [p for p in input_root.iterdir() if p.is_file()]
    if channel == "D":
        return sorted([p for p in files if RGB_FRAME_RE.match(p.name)])
    m = re.fullmatch(r"MS_(G|R|RE|NIR)", channel, flags=re.IGNORECASE)
    if not m:
        return []
    band = m.group(1).upper()
    out = []
    for p in files:
        pm = MS_FRAME_RE.match(p.name)
        if pm and pm.group("band").upper() == band:
            out.append(p)
    return sorted(out)


def _link_or_copy(src: Path, dst: Path, mode: str) -> str:
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
    return mode


def _normalize_uint16_png(img: Image.Image) -> Image.Image:
    arr = np.array(img)
    if arr.ndim != 2:
        raise ValueError(f"Expected single-channel TIFF, got shape={arr.shape}")
    if arr.dtype == np.uint16:
        out = arr
    elif np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        scale = 65535.0 / max(1, info.max)
        out = np.clip(np.round(arr.astype(np.float64) * scale), 0, 65535).astype(np.uint16)
    elif np.issubdtype(arr.dtype, np.floating):
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            out = np.zeros_like(arr, dtype=np.uint16)
        else:
            lo = float(finite.min())
            hi = float(finite.max())
            if hi <= lo:
                out = np.zeros_like(arr, dtype=np.uint16)
            else:
                norm = (arr - lo) / (hi - lo)
                out = np.clip(np.round(norm * 65535.0), 0, 65535).astype(np.uint16)
    else:
        raise ValueError(f"Unsupported TIFF dtype: {arr.dtype}")
    return Image.fromarray(out, mode="I;16")


def _convert_tiff_to_png(
    src: Path,
    dst: Path,
    compress_level: int = 6,
    backend: str = "pil",
) -> Dict[str, object]:
    if backend == "opencv":
        import cv2

        arr = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise RuntimeError(f"OpenCV could not read {src}")
        if arr.ndim != 2 or arr.dtype != np.uint16:
            raise ValueError(f"Expected uint16 single-channel TIFF, got shape={arr.shape} dtype={arr.dtype}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(
            str(dst),
            arr,
            [cv2.IMWRITE_PNG_COMPRESSION, int(compress_level)],
        ):
            raise RuntimeError(f"OpenCV could not write {dst}")
        return {
            "src": str(src),
            "dst": str(dst),
            "src_mode": str(arr.dtype),
            "src_size": [int(arr.shape[1]), int(arr.shape[0])],
            "dst_mode": "uint16",
            "png_compress_level": int(compress_level),
            "image_backend": "opencv",
        }
    if backend != "pil":
        raise ValueError(f"Unsupported image backend: {backend}")
    with Image.open(src) as img:
        png = _normalize_uint16_png(img)
        dst.parent.mkdir(parents=True, exist_ok=True)
        png.save(dst, format="PNG", compress_level=int(compress_level))
        return {
            "src": str(src),
            "dst": str(dst),
            "src_mode": str(img.mode),
            "src_size": [int(img.size[0]), int(img.size[1])],
            "dst_mode": str(png.mode),
            "png_compress_level": int(compress_level),
            "image_backend": "pil",
        }


def _copy_gps_metadata(src: Path, dst: Path, exiftool_cmd: str) -> Dict[str, object]:
    cmd = [
        exiftool_cmd,
        "-overwrite_original",
        "-TagsFromFile",
        str(src),
        "-GPS:all",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"exiftool GPS copy failed src={src} dst={dst} "
            f"code={proc.returncode} stderr={proc.stderr.strip()}"
        )
    return {
        "src": str(src),
        "dst": str(dst),
        "action": "copy_gps_all",
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _prepare_file_job(job: tuple[str, str, str, str, bool, int, str]) -> tuple[str, str, str | None, dict, bool]:
    src_text, dst_dir_text, channel, link_mode, force_copy_d, png_compress_level, image_backend = job
    src = Path(src_text)
    dst_dir = Path(dst_dir_text)
    cap_key = _capture_key(src, channel)
    ext = src.suffix.lower()
    if ext in {".tif", ".tiff"}:
        dst = dst_dir / f"{src.stem}.png"
        rec = _convert_tiff_to_png(
            src,
            dst,
            compress_level=png_compress_level,
            backend=image_backend,
        )
        rec["action"] = "convert_tiff_to_png"
        converted = True
    else:
        dst = dst_dir / src.name
        materialize_mode = "copy" if channel == "D" and force_copy_d else link_mode
        action = _link_or_copy(src, dst, materialize_mode)
        rec = {
            "src": str(src),
            "dst": str(dst),
            "action": action,
            "requested_link_mode": link_mode,
        }
        converted = False
    return str(src), str(dst), cap_key, rec, converted


def prepare_input(
    input_root: Path,
    output_root: Path,
    channels: List[str],
    link_mode: str,
    overwrite: bool,
    gps_copy_from_band: str | None = None,
    exiftool_cmd: str = "exiftool",
    max_workers: int = 1,
    png_compress_level: int = 6,
    image_backend: str = "pil",
) -> Dict[str, object]:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output root already exists: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, object] = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "channels": channels,
        "link_mode": link_mode,
        "max_workers": int(max_workers),
        "png_compress_level": int(png_compress_level),
        "image_backend": str(image_backend),
        "conversion_policy": {
            "tiff_to_png": True,
            "png_mode": "uint16_preserve_range",
            "non_tiff_files": "linked_or_copied_without_modification",
            "gps_copy_from_band": gps_copy_from_band,
        },
        "per_channel": {},
        "records": [],
        "gps_copy_records": [],
    }

    total_converted = 0
    total_linked = 0
    source_index: Dict[str, Dict[str, Path]] = {}
    source_frame_index: Dict[str, Dict[str, Path]] = {}
    ambiguous_source_frames: Dict[str, set[str]] = {}
    output_index: Dict[str, Dict[str, Path]] = {}
    for channel in channels:
        src_dir = input_root / channel
        input_layout = "channel_directory"
        if src_dir.is_dir():
            files = sorted([p for p in src_dir.iterdir() if p.is_file()])
            input_dir_for_audit = src_dir
        else:
            files = _flat_channel_files(input_root, channel)
            input_dir_for_audit = input_root
            input_layout = "flat_raw_root"
            if not files:
                raise FileNotFoundError(
                    f"Missing channel directory and no flat-layout files found for "
                    f"channel={channel}: {src_dir}"
                )
        dst_dir = output_root / channel
        dst_dir.mkdir(parents=True, exist_ok=True)
        source_index[channel] = {}
        source_frame_index[channel] = {}
        ambiguous_source_frames[channel] = set()
        output_index[channel] = {}

        ch_info = {
            "input_dir": str(input_dir_for_audit),
            "input_layout": input_layout,
            "output_dir": str(dst_dir),
            "num_input_files": len(files),
            "num_converted_tiff": 0,
            "num_linked_or_copied": 0,
            "output_files": [],
        }
        if not files:
            log_warn(f"No files found in channel directory: {src_dir}")

        workers = max(1, int(max_workers))
        jobs = [
            (
                str(src),
                str(dst_dir),
                channel,
                link_mode,
                gps_copy_from_band is not None,
                int(png_compress_level),
                str(image_backend),
            )
            for src in files
        ]
        if workers == 1:
            prepared = map(_prepare_file_job, jobs)
        else:
            executor = ProcessPoolExecutor(max_workers=workers)
            prepared = executor.map(_prepare_file_job, jobs)

        try:
            for src_text, dst_text, cap_key, rec, converted in prepared:
                src = Path(src_text)
                dst = Path(dst_text)
                if cap_key is not None:
                    source_index[channel][cap_key] = src
                    output_index[channel][cap_key] = dst
                frame_id = _frame_id(src, channel)
                if frame_id is not None:
                    previous = source_frame_index[channel].get(frame_id)
                    if previous is not None and previous != src:
                        source_frame_index[channel].pop(frame_id, None)
                        ambiguous_source_frames[channel].add(frame_id)
                    elif frame_id not in ambiguous_source_frames[channel]:
                        source_frame_index[channel][frame_id] = src
                summary["records"].append(rec)
                ch_info["output_files"].append(dst.name)
                if converted:
                    ch_info["num_converted_tiff"] += 1
                    total_converted += 1
                else:
                    ch_info["num_linked_or_copied"] += 1
                    total_linked += 1
        finally:
            if workers != 1:
                executor.shutdown(wait=True)

        summary["per_channel"][channel] = ch_info
        log_info(
            f"Prepared channel={channel} files={len(files)} "
            f"converted_tiff={ch_info['num_converted_tiff']} "
            f"linked_or_copied={ch_info['num_linked_or_copied']}"
        )

    if gps_copy_from_band is not None:
        if "D" not in output_index:
            raise ValueError("gps_copy_from_band requires channel D to be present in channels.")
        if gps_copy_from_band not in source_index:
            raise ValueError(
                f"gps_copy_from_band={gps_copy_from_band!r} is not present in channels={channels!r}"
            )
        migrated = 0
        frame_fallback = 0
        missing_source = 0
        missing_target = 0
        for cap_key, dst_d in sorted(output_index["D"].items()):
            src_band = source_index[gps_copy_from_band].get(cap_key)
            pairing_mode = "exact_capture_key"
            if src_band is None:
                frame_id = _frame_id(dst_d, "D")
                if frame_id is not None and frame_id not in ambiguous_source_frames[gps_copy_from_band]:
                    src_band = source_frame_index[gps_copy_from_band].get(frame_id)
                    if src_band is not None:
                        pairing_mode = "unique_frame_fallback"
                        frame_fallback += 1
            if src_band is None:
                missing_source += 1
                log_warn(f"Missing GPS source for capture={cap_key} band={gps_copy_from_band}")
                continue
            if not dst_d.exists():
                missing_target += 1
                log_warn(f"Missing D output for capture={cap_key}: {dst_d}")
                continue
            rec = _copy_gps_metadata(src_band, dst_d, exiftool_cmd=exiftool_cmd)
            rec["capture_key"] = cap_key
            rec["gps_source_band"] = gps_copy_from_band
            rec["pairing_mode"] = pairing_mode
            summary["gps_copy_records"].append(rec)
            migrated += 1
        summary["gps_copy_summary"] = {
            "enabled": True,
            "gps_source_band": gps_copy_from_band,
            "num_migrated": migrated,
            "num_unique_frame_fallback": frame_fallback,
            "num_missing_source": missing_source,
            "num_missing_target": missing_target,
            "exiftool_cmd": exiftool_cmd,
        }
        log_info(
            f"Copied GPS metadata from {gps_copy_from_band} to D: "
            f"migrated={migrated} frame_fallback={frame_fallback} "
            f"missing_source={missing_source} missing_target={missing_target}"
        )
    else:
        summary["gps_copy_summary"] = {
            "enabled": False,
        }

    summary["totals"] = {
        "converted_tiff": total_converted,
        "linked_or_copied": total_linked,
    }
    audit_path = output_root / "mmsplat_raw_input_audit.json"
    audit_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log_info(f"Wrote audit: {audit_path}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Prepare a MMSplat-compatible raw multispectral input tree by "
            "converting TIFF band files to 16-bit PNG while preserving the "
            "channel-directory layout."
        )
    )
    ap.add_argument("--input_root", required=True)
    ap.add_argument("--output_root", required=True)
    ap.add_argument(
        "--channels",
        default="D,MS_G,MS_R,MS_RE,MS_NIR",
        help="Comma-separated channel subdirectories to process.",
    )
    ap.add_argument(
        "--link_mode",
        choices=["hardlink", "copy", "symlink"],
        default="hardlink",
        help="How to materialize non-TIFF files in the output tree.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate output_root if it already exists.",
    )
    ap.add_argument(
        "--gps_copy_from_band",
        default=None,
        help=(
            "Optional source band whose GPS:all metadata will be copied to the corresponding "
            "D image outputs. Recommended value: MS_G"
        ),
    )
    ap.add_argument(
        "--exiftool_cmd",
        default="exiftool",
        help="Executable used for GPS metadata migration when --gps_copy_from_band is set.",
    )
    ap.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Number of independent image preparations to run concurrently (default: 1).",
    )
    ap.add_argument(
        "--png_compress_level",
        type=int,
        choices=range(10),
        default=6,
        help="Lossless PNG compression level, 0-9 (default: 6).",
    )
    ap.add_argument(
        "--image_backend",
        choices=["pil", "opencv"],
        default="pil",
        help="Lossless TIFF/PNG I/O backend (default: pil).",
    )
    args = ap.parse_args()

    channels = [c.strip() for c in str(args.channels).split(",") if c.strip()]
    prepare_input(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        channels=channels,
        link_mode=str(args.link_mode),
        overwrite=bool(args.overwrite),
        gps_copy_from_band=(
            str(args.gps_copy_from_band).strip()
            if args.gps_copy_from_band is not None
            else None
        ),
        exiftool_cmd=str(args.exiftool_cmd),
        max_workers=int(args.max_workers),
        png_compress_level=int(args.png_compress_level),
        image_backend=str(args.image_backend),
    )


if __name__ == "__main__":
    main()
