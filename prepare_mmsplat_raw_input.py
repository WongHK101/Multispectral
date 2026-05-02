from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
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


def _convert_tiff_to_png(src: Path, dst: Path) -> Dict[str, object]:
    with Image.open(src) as img:
        png = _normalize_uint16_png(img)
        dst.parent.mkdir(parents=True, exist_ok=True)
        png.save(dst, format="PNG")
        return {
            "src": str(src),
            "dst": str(dst),
            "src_mode": str(img.mode),
            "src_size": [int(img.size[0]), int(img.size[1])],
            "dst_mode": str(png.mode),
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


def prepare_input(
    input_root: Path,
    output_root: Path,
    channels: List[str],
    link_mode: str,
    overwrite: bool,
    gps_copy_from_band: str | None = None,
    exiftool_cmd: str = "exiftool",
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

        for src in files:
            cap_key = _capture_key(src, channel)
            if cap_key is not None:
                source_index[channel][cap_key] = src
            ext = src.suffix.lower()
            if ext in {".tif", ".tiff"}:
                dst = dst_dir / f"{src.stem}.png"
                rec = _convert_tiff_to_png(src, dst)
                rec["action"] = "convert_tiff_to_png"
                summary["records"].append(rec)
                ch_info["num_converted_tiff"] += 1
                total_converted += 1
                ch_info["output_files"].append(dst.name)
            else:
                dst = dst_dir / src.name
                materialize_mode = link_mode
                if channel == "D" and gps_copy_from_band is not None:
                    # D outputs are edited by exiftool when GPS is migrated, so they
                    # must be independent files rather than hardlinks to raw data.
                    materialize_mode = "copy"
                action = _link_or_copy(src, dst, materialize_mode)
                summary["records"].append({
                    "src": str(src),
                    "dst": str(dst),
                    "action": action,
                    "requested_link_mode": link_mode,
                })
                ch_info["num_linked_or_copied"] += 1
                total_linked += 1
                ch_info["output_files"].append(dst.name)
            if cap_key is not None:
                output_index[channel][cap_key] = dst

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
        missing_source = 0
        missing_target = 0
        for cap_key, dst_d in sorted(output_index["D"].items()):
            src_band = source_index[gps_copy_from_band].get(cap_key)
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
            summary["gps_copy_records"].append(rec)
            migrated += 1
        summary["gps_copy_summary"] = {
            "enabled": True,
            "gps_source_band": gps_copy_from_band,
            "num_migrated": migrated,
            "num_missing_source": missing_source,
            "num_missing_target": missing_target,
            "exiftool_cmd": exiftool_cmd,
        }
        log_info(
            f"Copied GPS metadata from {gps_copy_from_band} to D: "
            f"migrated={migrated} missing_source={missing_source} missing_target={missing_target}"
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
    args = ap.parse_args()

    channels = [c.strip() for c in str(args.channels).split(",") if c.strip()]
    prepare_input(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        channels=channels,
        link_mode=str(args.link_mode),
        overwrite=bool(args.overwrite),
        gps_copy_from_band=(str(args.gps_copy_from_band).strip() or None),
        exiftool_cmd=str(args.exiftool_cmd),
    )


if __name__ == "__main__":
    main()
