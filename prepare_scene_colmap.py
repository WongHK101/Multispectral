#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_scene_colmap.py

Prepare COLMAP- and 3DGS-compatible scene assets for the current
SpectralIndexGS pipeline.
This script converts standardized raw or aligned scene folders into the camera,
database, sparse-model, and layout artifacts expected by the downstream
reconstruction pipeline, and exports optional pose-prior records for
GPS-enabled runs.
"""

import argparse
import json
import os
import math
import re
import sqlite3
import struct
import subprocess
import sys
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np


def log_info(msg: str) -> None:
    print(f"INFO: {msg}", flush=True)


def log_warn(msg: str) -> None:
    print(f"WARNING: {msg}", flush=True)


def log_err(msg: str) -> None:
    print(f"ERROR: {msg}", flush=True)


def _resolve_executable(exe: str) -> str:
    exe_expanded = os.path.expandvars(exe)
    if os.path.isabs(exe_expanded) and os.path.exists(exe_expanded):
        return exe_expanded

    candidates = [exe_expanded]
    if os.name == "nt":
        base = exe_expanded
        if not base.lower().endswith((".exe", ".cmd", ".bat")):
            candidates = [base, base + ".exe", base + ".cmd", base + ".bat"]

    for v in candidates:
        p = shutil.which(v)
        if p:
            return p
    return exe_expanded


def _should_use_shell(resolved_exe: str) -> bool:
    if os.name != "nt":
        return False
    low = resolved_exe.lower()
    return low.endswith(".bat") or low.endswith(".cmd")


def run_cmd(cmd_list, cwd=None, extra_env: Optional[Dict[str, str]] = None):
    if not cmd_list:
        raise ValueError("Empty command list")

    resolved0 = _resolve_executable(cmd_list[0])
    use_shell = _should_use_shell(resolved0)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    if use_shell:
        cmd_str = " ".join([f'"{x}"' if (" " in str(x)) else str(x) for x in [resolved0] + cmd_list[1:]])
        log_info("Running (shell): " + cmd_str)
        subprocess.run(cmd_str, cwd=cwd, check=True, shell=True, env=env)
    else:
        cmd = [resolved0] + cmd_list[1:]
        log_info("Running: " + " ".join([str(x) for x in cmd]))
        subprocess.run(cmd, cwd=cwd, check=True, env=env)


def split_args(s: str):
    return s.strip().split() if s else []


def default_colmap_executable() -> str:
    for env_name in ("SIGS_COLMAP_EXECUTABLE", "COLMAP_EXECUTABLE"):
        value = str(os.environ.get(env_name, "")).strip()
        if value:
            return value
    home = Path.home()
    candidates = [
        home / "opt" / "colmap-cuda" / "bin" / "colmap",
        home / "opt" / "colmap-cuda-3.7" / "bin" / "colmap",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return "colmap"


def normalize_colmap_gpu_mode(mode: str) -> str:
    value = str(mode).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return "1"
    if value in {"0", "false", "no", "off"}:
        return "0"
    return "auto"


def _run_cmd_capture_output(cmd_list, cwd=None, extra_env: Optional[Dict[str, str]] = None) -> str:
    if not cmd_list:
        raise ValueError("Empty command list")

    resolved0 = _resolve_executable(cmd_list[0])
    use_shell = _should_use_shell(resolved0)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    if use_shell:
        cmd_str = " ".join([f'"{x}"' if (" " in str(x)) else str(x) for x in [resolved0] + cmd_list[1:]])
        log_info("Running (shell): " + cmd_str)
        result = subprocess.run(
            cmd_str,
            cwd=cwd,
            check=False,
            shell=True,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
    else:
        cmd = [resolved0] + cmd_list[1:]
        log_info("Running: " + " ".join([str(x) for x in cmd]))
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=False,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )

    output = result.stdout or ""
    if output:
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd_list, output=output)
    return output


def _looks_like_colmap_gpu_failure(output: str) -> bool:
    text = str(output).lower()
    markers = (
        "failed to extract features",
        "not enough gpu memory",
        "out of memory",
        "siftgpu not fully supported",
        "could not connect to display",
        "could not load the qt platform plugin",
        "check failed: context_.create()",
    )
    return any(marker in text for marker in markers)


def run_colmap_gpu_cmd(
    cmd_prefix: List[str],
    gpu_flag_name: str,
    gpu_mode: str,
    cwd=None,
    retry_cleanup_paths: Optional[List[Path]] = None,
) -> None:
    def gpu_attempt_env() -> Optional[Dict[str, str]]:
        if os.name == "nt":
            return None
        if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            return None
        if os.environ.get("QT_QPA_PLATFORM"):
            return None
        return {"QT_QPA_PLATFORM": "offscreen"}

    normalized_mode = normalize_colmap_gpu_mode(gpu_mode)
    if normalized_mode != "auto":
        output = _run_cmd_capture_output(
            cmd_prefix + [gpu_flag_name, normalized_mode],
            cwd=cwd,
            extra_env=gpu_attempt_env() if normalized_mode == "1" else None,
        )
        if normalized_mode == "1" and _looks_like_colmap_gpu_failure(output):
            raise RuntimeError(f"{gpu_flag_name}=1 completed with GPU-failure markers in output.")
        return

    try:
        log_info(f"{gpu_flag_name}=auto -> trying GPU first")
        output = _run_cmd_capture_output(cmd_prefix + [gpu_flag_name, "1"], cwd=cwd, extra_env=gpu_attempt_env())
        if _looks_like_colmap_gpu_failure(output):
            raise RuntimeError(f"{gpu_flag_name}=1 completed with GPU-failure markers in output.")
        return
    except (subprocess.CalledProcessError, RuntimeError):
        log_warn(
            f"{gpu_flag_name}=1 failed; retrying with CPU. "
            "This often means COLMAP GPU SIFT is unavailable in the current runtime."
        )
        for path in retry_cleanup_paths or []:
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()

    run_cmd(cmd_prefix + [gpu_flag_name, "0"], cwd=cwd)



def replace_alignment_max_error(arg_list, value):
    """Replace/insert --alignment_max_error in a tokenized arg list."""
    out = []
    i = 0
    found = False
    while i < len(arg_list):
        a = arg_list[i]
        if a.startswith("--alignment_max_error="):
            out.append(f"--alignment_max_error={value}")
            found = True
            i += 1
            continue
        if a == "--alignment_max_error":
            out.append(a)
            # replace next token if present, else append
            if i + 1 < len(arg_list) and not arg_list[i + 1].startswith("--"):
                out.append(str(value))
                i += 2
            else:
                out.append(str(value))
                i += 1
            found = True
            continue
        out.append(a)
        i += 1
    if not found:
        out += ["--alignment_max_error", str(value)]
    return out


def is_number(x):
    try:
        float(x)
        return True
    except Exception:
        return False

def _read_c_string(f) -> str:
    chars = []
    while True:
        c = f.read(1)
        if not c or c == b"\x00":
            break
        chars.append(c)
    return b"".join(chars).decode("utf-8", errors="replace")


def read_num_registered_images(model_dir: Path) -> int:
    images_bin = model_dir / "images.bin"
    images_txt = model_dir / "images.txt"
    if images_bin.exists():
        with images_bin.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
        return int(n)
    if images_txt.exists():
        n = 0
        with images_txt.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 10 and parts[0].isdigit():
                    n += 1
        return n
    return -1


def read_num_points3d(model_dir: Path) -> int:
    points_bin = model_dir / "points3D.bin"
    points_txt = model_dir / "points3D.txt"
    if points_bin.exists():
        with points_bin.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
        return int(n)
    if points_txt.exists():
        n = 0
        with points_txt.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                n += 1
        return n
    return -1


def read_image_names(model_dir: Path):
    names = []
    images_txt = model_dir / "images.txt"
    images_bin = model_dir / "images.bin"

    if images_txt.exists():
        with images_txt.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 10 and parts[0].isdigit():
                    names.append(parts[9])
        return names

    if images_bin.exists():
        with images_bin.open("rb") as f:
            num_images = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_images):
                f.read(4)
                f.read(8 * 7)
                f.read(4)
                name = _read_c_string(f)
                names.append(name)
                num_points2d = struct.unpack("<Q", f.read(8))[0]
                f.read(num_points2d * (8 + 8 + 8))
        return names

    return names


def read_name_list(path: Path) -> List[str]:
    if not path or not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _name_keys(name: str) -> set:
    normalized = str(name).replace("\\", "/").strip()
    return {normalized, Path(normalized).name}


def _registered_name_hits(registered_names: List[str], required_names: List[str]) -> Tuple[List[str], List[str]]:
    registered_keys = set()
    for name in registered_names:
        registered_keys.update(_name_keys(name))
    hits = []
    missing = []
    for name in required_names:
        if _name_keys(name) & registered_keys:
            hits.append(name)
        else:
            missing.append(name)
    return hits, missing


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def select_best_sparse_model(sparse_root: Path) -> Path:
    if not sparse_root.exists():
        raise FileNotFoundError(f"sparse root not found: {sparse_root}")

    candidates = []
    for p in sparse_root.iterdir():
        if not p.is_dir():
            continue
        if not re.fullmatch(r"\d+", p.name):
            continue
        reg = read_num_registered_images(p)
        pts = read_num_points3d(p)
        if reg < 0 and pts < 0:
            continue
        candidates.append((reg, pts, p))

    if not candidates:
        raise RuntimeError(f"No valid sparse models under: {sparse_root}")

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best = candidates[0][2]

    log_info(f"Sparse models found under {sparse_root}:")
    for reg, pts, p in candidates[:30]:
        log_info(f"  model={p.name} | registered={reg} | points3D={pts}")
    log_info(f"Selected best sparse model: {best}")
    return best


def exiftool_extract_gps(input_dir: Path, exiftool_exe: str):
    if not input_dir.exists():
        raise FileNotFoundError(f"input dir not found: {input_dir}")

    exiftool_path = _resolve_executable(exiftool_exe)

    cmd = [
        exiftool_path,
        "-q", "-q",
        "-json",
        "-n",
        # Ask exiftool for all groups so PNG EXIF/XMP GPS tags are not dropped.
        "-G",
        "-GPSLatitude",
        "-GPSLongitude",
        "-GPSAltitude",
        "-EXIF:GPSLatitude",
        "-EXIF:GPSLongitude",
        "-EXIF:GPSAltitude",
        "-XMP:GPSLatitude",
        "-XMP:GPSLongitude",
        "-XMP:GPSAltitude",
        "-Composite:GPSLatitude",
        "-Composite:GPSLongitude",
        "-Composite:GPSAltitude",
        "-XMP-drone-dji:AbsoluteAltitude",
        "-r",
        "-ext", "jpg",
        "-ext", "jpeg",
        "-ext", "JPG",
        "-ext", "JPEG",
        "-ext", "png",
        "-ext", "PNG",
        str(input_dir),
    ]

    log_info("Extracting GPS from images via exiftool (may take a bit)...")
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    text = p.stdout.decode("utf-8", errors="replace")

    records = []
    if text.strip():
        try:
            records = json.loads(text)
        except json.JSONDecodeError:
            start = text.find('[')
            end = text.rfind(']')
            if start >= 0 and end > start:
                try:
                    records = json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    log_warn("Exiftool output is not valid JSON after bracket recovery; treat as no GPS.")
                    records = []
            else:
                log_warn("Exiftool output has no JSON payload; treat as no GPS.")
                records = []
    else:
        log_warn("Exiftool returned empty output; treat as no GPS.")

    gps = {}
    def _pick_tag(rec: Dict, names: List[str]):
        # 1) exact key
        for n in names:
            if n in rec and rec[n] is not None:
                return rec[n]
        # 2) key suffix match (handles keys like "EXIF:GPSLatitude", "XMP:GPSLatitude")
        rec_items = list(rec.items())
        for n in names:
            n_low = n.lower()
            for k, v in rec_items:
                if v is None:
                    continue
                ks = str(k).lower()
                if ks == n_low or ks.endswith(":" + n_low):
                    return v
        return None

    for r in records:
        src = r.get("SourceFile", "")
        base = os.path.basename(src)
        lat = _pick_tag(r, ["GPSLatitude"])
        lon = _pick_tag(r, ["GPSLongitude"])
        alt = _pick_tag(r, ["GPSAltitude"])
        if alt is None:
            alt = _pick_tag(r, ["AbsoluteAltitude", "RelativeAltitude"])
        if lat is None or lon is None or alt is None:
            continue
        if not (is_number(lat) and is_number(lon) and is_number(alt)):
            continue
        gps[base] = (float(lat), float(lon), float(alt))

    log_info(f"EXIF GPS entries found: {len(gps)} (keyed by basename)")
    return gps


def ensure_pose_priors_table(con: sqlite3.Connection):
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='pose_priors'")
    if cur.fetchone():
        return
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS pose_priors (
            image_id INTEGER PRIMARY KEY NOT NULL,
            position BLOB,
            coordinate_system INTEGER NOT NULL,
            position_covariance BLOB,
            FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
        )
        """
    )
    con.commit()


def get_pose_priors_schema(con: sqlite3.Connection):
    cur = con.cursor()
    cur.execute("PRAGMA table_info(pose_priors)")
    rows = cur.fetchall()
    return [r[1] for r in rows]


def populate_pose_priors_from_exif(
    database_path: Path,
    input_dir: Path,
    exiftool_exe: str,
    wgs84_code: int,
    prior_position_std_m: Optional[float] = None,
    swap_latlon: bool = False,
):
    """
    Populate COLMAP pose_priors from EXIF GPS.

    Notes:
    - Newer COLMAP versions store pose priors in a dedicated `pose_priors` table.
    - For GPS, COLMAP expects WGS84 coordinate system code (in your build it's 0).
    - `position_covariance` should be NULL or finite values. Writing NaNs can make
      downstream alignment fail.

    prior_position_std_m:
      If provided, we write a diagonal covariance:
        [std_lat_deg^2, std_lon_deg^2, std_alt_m^2]
      where std_lat_deg/std_lon_deg are meters converted to degrees at that latitude.
      If None, position_covariance is left NULL.
    """
    gps = exiftool_extract_gps(input_dir, exiftool_exe=exiftool_exe)

    con = sqlite3.connect(str(database_path))
    try:
        ensure_pose_priors_table(con)
        cols = get_pose_priors_schema(con)
        expected = ["image_id", "position", "coordinate_system", "position_covariance"]
        if cols != expected:
            raise RuntimeError(f"pose_priors columns unexpected: {cols}")

        cur = con.cursor()
        cur.execute("SELECT image_id, name FROM images")
        rows = cur.fetchall()

        # Map by basename (works if basenames are unique). Also keep a duplicate check.
        name2id = {}
        dup_bases = set()
        for image_id, name in rows:
            base = os.path.basename(name)
            if base in name2id:
                dup_bases.add(base)
            name2id[base] = image_id
        if dup_bases:
            log_warn(
                f"Duplicate basenames detected in DB images table (showing up to 5): {list(sorted(dup_bases))[:5]}. "
                "Basename-based EXIF matching may be ambiguous."
            )

        inserted = 0
        matched = 0

        # Precompute some sanity stats.
        lat_list, lon_list, alt_list = [], [], []

        for base, image_id in name2id.items():
            if base not in gps:
                continue
            matched += 1
            lat, lon, alt = gps[base]

            if swap_latlon:
                lat, lon = lon, lat  # experimental switch if needed

            # position is stored as 3 float64 values (lat_deg, lon_deg, alt_m)
            pos = np.asarray([lat, lon, alt], dtype=np.float64)

            # Optional covariance (NULL by default).
            cov_blob = None
            if prior_position_std_m is not None and prior_position_std_m > 0:
                # Convert meters -> degrees at this latitude (rough WGS84 approximation).
                meters_per_deg_lat = 111320.0
                meters_per_deg_lon = 111320.0 * max(1e-6, math.cos(math.radians(lat)))
                std_lat_deg = prior_position_std_m / meters_per_deg_lat
                std_lon_deg = prior_position_std_m / meters_per_deg_lon
                cov = np.diag([std_lat_deg**2, std_lon_deg**2, float(prior_position_std_m) ** 2]).astype(np.float64)
                cov_blob = cov.tobytes()

            cur.execute(
                "INSERT OR REPLACE INTO pose_priors(image_id, position, coordinate_system, position_covariance) "
                "VALUES (?, ?, ?, ?)",
                (int(image_id), pos.tobytes(), int(wgs84_code), cov_blob),
            )
            inserted += 1

            lat_list.append(pos[0])
            lon_list.append(pos[1])
            alt_list.append(pos[2])

        con.commit()
        log_info(f"Pose priors populated: inserted={inserted}, matched_images_in_db={matched}, db_images_total={len(rows)}")

        cur.execute("SELECT COUNT(*) FROM pose_priors")
        cnt = cur.fetchone()[0]
        log_info(f"pose_priors rows in DB: {cnt}")

        cur.execute("SELECT MIN(coordinate_system), MAX(coordinate_system) FROM pose_priors")
        mn, mx = cur.fetchone()
        log_info(f"pose_priors.coordinate_system range: {mn}..{mx}")

        if lat_list and lon_list and alt_list:
            log_info(
                "pose_priors position ranges (from inserted rows): "
                f"lat[{min(lat_list):.8f},{max(lat_list):.8f}] "
                f"lon[{min(lon_list):.8f},{max(lon_list):.8f}] "
                f"alt[{min(alt_list):.3f},{max(alt_list):.3f}]"
            )
            # Basic plausibility checks (WGS84 degrees)
            if not (-90 <= min(lat_list) <= 90 and -90 <= max(lat_list) <= 90):
                log_warn("Latitude range looks suspicious (outside [-90, 90]).")
            if not (-180 <= min(lon_list) <= 180 and -180 <= max(lon_list) <= 180):
                log_warn("Longitude range looks suspicious (outside [-180, 180]).")

    finally:
        con.close()

    return gps


def sanity_check_overlap(model_dir: Path, gps_by_basename: dict):
    model_names = read_image_names(model_dir)
    if not model_names:
        log_warn("Could not read image names from selected model; skip overlap check.")
        return {"model_image_count": 0, "gps_overlap_count": 0}
    bases = [os.path.basename(n) for n in model_names]
    overlap = sum(1 for b in bases if b in gps_by_basename)
    log_info(f"Selected model images: {len(bases)}; images-with-GPS(overlap by basename): {overlap}")
    if overlap < 3:
        log_warn("Overlap < 3; model_aligner likely fails (min_common_images default is 3).")
    return {"model_image_count": len(bases), "gps_overlap_count": overlap}


def _lla_to_local_enu(lat: float, lon: float, alt: float, lat0: float, lon0: float, alt0: float) -> np.ndarray:
    """Approximate WGS84 lat/lon/alt as a local ENU frame in meters."""
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * max(1e-9, math.cos(math.radians(lat0)))
    return np.asarray(
        [
            (float(lon) - float(lon0)) * meters_per_deg_lon,
            (float(lat) - float(lat0)) * meters_per_deg_lat,
            float(alt) - float(alt0),
        ],
        dtype=np.float64,
    )


def _estimate_similarity_umeyama(source_xyz: np.ndarray, target_xyz: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Estimate target ~= scale * R * source + t."""
    source_xyz = np.asarray(source_xyz, dtype=np.float64)
    target_xyz = np.asarray(target_xyz, dtype=np.float64)
    if source_xyz.shape != target_xyz.shape or source_xyz.ndim != 2 or source_xyz.shape[1] != 3:
        raise ValueError(f"Invalid similarity inputs: source={source_xyz.shape}, target={target_xyz.shape}")
    if source_xyz.shape[0] < 3:
        raise ValueError("At least 3 matched camera centers are required for similarity alignment.")

    src_mean = source_xyz.mean(axis=0)
    dst_mean = target_xyz.mean(axis=0)
    src_centered = source_xyz - src_mean
    dst_centered = target_xyz - dst_mean
    src_var = float(np.mean(np.sum(src_centered * src_centered, axis=1)))
    if src_var <= 0:
        raise ValueError("Degenerate source camera centers; cannot estimate similarity.")

    cov = (dst_centered.T @ src_centered) / source_xyz.shape[0]
    u, singular_values, vt = np.linalg.svd(cov)
    d = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        d[-1, -1] = -1
    rot = u @ d @ vt
    scale = float(np.sum(singular_values * np.diag(d)) / src_var)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid estimated alignment scale: {scale}")
    trans = dst_mean - scale * rot @ src_mean
    return scale, rot, trans


def write_gps_ref_images(model_dir: Path, gps_by_basename: dict, output_path: Path) -> int:
    rows = []
    for image_name in read_image_names(model_dir):
        base = os.path.basename(image_name)
        if base not in gps_by_basename:
            continue
        lat, lon, alt = gps_by_basename[base]
        rows.append(f"{image_name} {float(lat):.12f} {float(lon):.12f} {float(alt):.6f}\n")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(rows), encoding="utf-8")
    log_info(f"Wrote GPS reference image file for georegistration: {output_path} rows={len(rows)}")
    return len(rows)


def custom_sim3_georegister_model(
    input_model: Path,
    output_model: Path,
    gps_by_basename: dict,
) -> dict:
    """Write a WGS84-derived local ENU-aligned COLMAP model using matched EXIF/RTK priors."""
    from utils.read_write_model import (  # local import keeps normal COLMAP conversion startup light
        Image,
        Point3D,
        qvec2rotmat,
        read_model,
        rotmat2qvec,
        write_model,
    )

    cameras, images, points3d = read_model(str(input_model), ext="")
    if cameras is None:
        raise RuntimeError(f"Could not read COLMAP model: {input_model}")

    matched = []
    for image_id, image in images.items():
        base = os.path.basename(image.name)
        if base not in gps_by_basename:
            continue
        r_cw = qvec2rotmat(image.qvec)
        center = -r_cw.T @ image.tvec
        lat, lon, alt = gps_by_basename[base]
        matched.append((image_id, center, (float(lat), float(lon), float(alt))))
    if len(matched) < 3:
        raise RuntimeError(f"Insufficient GPS/RTK priors for custom Sim3 alignment: {len(matched)}")

    llas = np.asarray([m[2] for m in matched], dtype=np.float64)
    lat0, lon0, alt0 = llas.mean(axis=0)
    source = np.asarray([m[1] for m in matched], dtype=np.float64)
    target = np.asarray([_lla_to_local_enu(*m[2], lat0, lon0, alt0) for m in matched], dtype=np.float64)
    scale, rot, trans = _estimate_similarity_umeyama(source, target)

    transformed_centers = scale * (rot @ source.T).T + trans
    residual = np.linalg.norm(transformed_centers - target, axis=1)

    new_images = {}
    for image_id, image in images.items():
        r_cw = qvec2rotmat(image.qvec)
        r_new = r_cw @ rot.T
        t_new = scale * image.tvec - r_new @ trans
        q_new = rotmat2qvec(r_new)
        new_images[image_id] = Image(
            id=image.id,
            qvec=q_new.astype(np.float64),
            tvec=t_new.astype(np.float64),
            camera_id=image.camera_id,
            name=image.name,
            xys=image.xys,
            point3D_ids=image.point3D_ids,
        )

    new_points3d = {}
    for point_id, point in points3d.items():
        xyz_new = scale * (rot @ point.xyz) + trans
        new_points3d[point_id] = Point3D(
            id=point.id,
            xyz=xyz_new.astype(np.float64),
            rgb=point.rgb,
            error=point.error,
            image_ids=point.image_ids,
            point2D_idxs=point.point2D_idxs,
        )

    if output_model.exists():
        shutil.rmtree(output_model)
    output_model.mkdir(parents=True, exist_ok=True)
    write_model(cameras, new_images, new_points3d, str(output_model), ext=".bin")
    write_model(cameras, new_images, new_points3d, str(output_model), ext=".txt")

    aligned_centers = np.asarray([-(qvec2rotmat(im.qvec).T @ im.tvec) for im in new_images.values()], dtype=np.float64)
    points_xyz = np.asarray([p.xyz for p in new_points3d.values()], dtype=np.float64) if new_points3d else np.empty((0, 3))
    summary = {
        "backend": "custom_sim3",
        "coordinate_frame": "local_enu_from_wgs84_pose_priors",
        "enu_origin_lat_lon_alt": [float(lat0), float(lon0), float(alt0)],
        "matched_registered_images": int(len(matched)),
        "estimated_scale": float(scale),
        "estimated_rotation": rot.tolist(),
        "estimated_translation": trans.tolist(),
        "alignment_error_m": {
            "mean": float(residual.mean()),
            "median": float(np.median(residual)),
            "min": float(residual.min()),
            "max": float(residual.max()),
            "p90": float(np.percentile(residual, 90)),
            "p95": float(np.percentile(residual, 95)),
        },
        "target_enu_extent_xyz": (target.max(axis=0) - target.min(axis=0)).tolist(),
        "aligned_camera_extent_xyz": (aligned_centers.max(axis=0) - aligned_centers.min(axis=0)).tolist(),
        "aligned_camera_min_xyz": aligned_centers.min(axis=0).tolist(),
        "aligned_camera_max_xyz": aligned_centers.max(axis=0).tolist(),
        "aligned_points3d_count": int(len(new_points3d)),
    }
    if len(points_xyz):
        summary.update({
            "aligned_points3d_min_xyz": points_xyz.min(axis=0).tolist(),
            "aligned_points3d_max_xyz": points_xyz.max(axis=0).tolist(),
            "aligned_points3d_extent_xyz": (points_xyz.max(axis=0) - points_xyz.min(axis=0)).tolist(),
        })
    (output_model / "georegistration_alignment_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary

def ensure_3dgs_sparse_layout(root: Path) -> Path:
    """Normalize COLMAP output layout for 3DGS.

    3DGS loaders typically expect:
      <root>/images
      <root>/sparse/0/{cameras,images,points3D}.{bin|txt}

    COLMAP may instead write the model directly under <root>/sparse (no /0)
    or under a different numeric subfolder. This function makes sure that
    <root>/sparse/0 exists and contains the model files.
    """

    sparse = root / "sparse"
    if not sparse.exists():
        raise FileNotFoundError(f"COLMAP undistorted sparse folder not found: {sparse}")

    model0 = sparse / "0"

    def has_any_model_files(p: Path) -> bool:
        for fn in (
            "cameras.bin",
            "images.bin",
            "points3D.bin",
            "cameras.txt",
            "images.txt",
            "points3D.txt",
            "rigs.bin",
            "frames.bin",
        ):
            if (p / fn).exists():
                return True
        return False

    # Case 1: already in sparse/0
    if model0.is_dir() and has_any_model_files(model0):
        return model0

    # Case 2: model files are directly under sparse/
    if has_any_model_files(sparse):
        model0.mkdir(parents=True, exist_ok=True)
        for child in list(sparse.iterdir()):
            if child.name == "0":
                continue
            if child.is_file():
                dst = model0 / child.name
                if dst.exists():
                    dst.unlink()
                shutil.move(str(child), str(dst))
        return model0

    # Case 3: numeric subfolders (but no /0)
    subdirs = [p for p in sparse.iterdir() if p.is_dir() and re.fullmatch(r"\d+", p.name)]
    if subdirs:
        candidates = [p for p in subdirs if has_any_model_files(p)] or subdirs

        def score(p: Path) -> int:
            s = 0
            pts_bin = p / "points3D.bin"
            imgs_bin = p / "images.bin"
            pts_txt = p / "points3D.txt"
            if pts_bin.exists():
                s += int(pts_bin.stat().st_size)
            if imgs_bin.exists():
                s += int(imgs_bin.stat().st_size) // 10
            if pts_txt.exists():
                s += int(pts_txt.stat().st_size) // 50
            return s

        best = max(candidates, key=score)
        if best.name != "0":
            if model0.exists():
                shutil.rmtree(model0)
            shutil.copytree(best, model0)
        return model0

    raise RuntimeError(f"Could not locate any COLMAP model under: {sparse}")


def export_model_as_txt(colmap_exe: str, model_dir: Path) -> None:
    """Export cameras/images/points to TXT for 3DGS fallback readers."""
    if (model_dir / "cameras.txt").exists() and (model_dir / "images.txt").exists() and (model_dir / "points3D.txt").exists():
        return

    log_info(f"Exporting COLMAP model as TXT for 3DGS compatibility: {model_dir}")
    try:
        run_cmd([
            colmap_exe,
            "model_converter",
            "--input_path",
            str(model_dir),
            "--output_path",
            str(model_dir),
            "--output_type",
            "TXT",
        ])
    except Exception as e:
        log_warn(f"model_converter failed (will continue): {e}")

def ensure_camera_models_supported_for_3dgs(colmap_exe: str, model_dir: Path) -> None:
    """3DGS loaders often only accept PINHOLE / SIMPLE_PINHOLE camera models.
    Some COLMAP outputs (or previous conversions) may keep e.g. SIMPLE_RADIAL/OPENCV models.
    This function:
      1) Ensures cameras.txt exists (via model_converter -> TXT).
      2) Rewrites cameras.txt to PINHOLE or SIMPLE_PINHOLE when possible by dropping distortion params.
      3) Regenerates cameras.bin/images.bin/points3D.bin from the (possibly) edited TXT model.
    """
    export_model_as_txt(colmap_exe, model_dir)
    cam_txt = model_dir / "cameras.txt"
    if not cam_txt.exists():
        log_warn(f"cameras.txt not found under {model_dir}; cannot enforce camera model.")
        return

    txt = cam_txt.read_text(encoding="utf-8", errors="replace").splitlines()
    changed = False
    new_lines: List[str] = []
    conversions = []

    def fmt_params(ps: List[float]) -> List[str]:
        # use repr to preserve enough precision without trailing '+'
        return [repr(float(p)) for p in ps]

    for line in txt:
        s = line.strip()
        if not s or s.startswith("#"):
            new_lines.append(line)
            continue

        parts = s.split()
        if len(parts) < 5:
            new_lines.append(line)
            continue

        cam_id, model, w, h = parts[0], parts[1], parts[2], parts[3]
        # params may include leading '+'; float() handles it
        try:
            params = [float(x) for x in parts[4:]]
        except Exception:
            new_lines.append(line)
            continue

        if model in ("PINHOLE", "SIMPLE_PINHOLE"):
            new_lines.append(line)
            continue

        model_new = None
        params_new: List[float] = []

        # Most models start with f cx cy (SIMPLE_*) or fx fy cx cy (OPENCV-like)
        if model in ("SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL", "RADIAL_FISHEYE", "FOV"):
            if len(params) >= 3:
                f, cx, cy = params[0], params[1], params[2]
                model_new = "SIMPLE_PINHOLE"
                params_new = [f, cx, cy]
        elif model in ("OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "THIN_PRISM_FISHEYE"):
            if len(params) >= 4:
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
                model_new = "PINHOLE"
                params_new = [fx, fy, cx, cy]
        else:
            # Fallback heuristic:
            # - if >=4 params, treat as fx fy cx cy -> PINHOLE
            # - else if >=3 params, treat as f cx cy -> SIMPLE_PINHOLE
            if len(params) >= 4:
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
                model_new = "PINHOLE"
                params_new = [fx, fy, cx, cy]
            elif len(params) >= 3:
                f, cx, cy = params[0], params[1], params[2]
                model_new = "SIMPLE_PINHOLE"
                params_new = [f, cx, cy]

        if model_new is None:
            # keep original line if we can't safely convert
            new_lines.append(line)
            continue

        changed = True
        conversions.append(f"{cam_id}:{model}->{model_new}")
        new_line = " ".join([cam_id, model_new, w, h] + fmt_params(params_new))
        new_lines.append(new_line)

    if changed:
        cam_txt.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        log_info(f"Adjusted camera models for 3DGS: {', '.join(conversions[:8])}{' ...' if len(conversions)>8 else ''}")

        # Regenerate BIN from TXT so 3DGS (which prefers .bin) reads the supported model.
        try:
            run_cmd([
                colmap_exe,
                "model_converter",
                "--input_path",
                str(model_dir),
                "--output_path",
                str(model_dir),
                "--output_type",
                "BIN",
            ])
        except Exception as e:
            log_warn(f"Failed to regenerate BIN model after camera-model fix (will keep TXT): {e}")
    else:
        log_info("Camera models already supported (PINHOLE/SIMPLE_PINHOLE); no conversion needed.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_path", required=True, help="Dataset root (contains input/)")
    parser.add_argument(
        "--colmap_executable",
        default=default_colmap_executable(),
        help=(
            "Path/name of colmap executable. "
            "Defaults to SIGS_COLMAP_EXECUTABLE/COLMAP_EXECUTABLE, then ~/opt/colmap-cuda/bin/colmap if present."
        ),
    )
    parser.add_argument("--exiftool_executable", default="exiftool", help="Path/name of exiftool executable")
    parser.add_argument(
        "--wgs84_code", type=int, default=0,
        help="Integer code for PosePrior coordinate_system=WGS84. Your COLMAP error showed WGS84==0."
    )

    parser.add_argument(
        "--prior_position_std_m", type=float, default=None,
        help="If set, write pose_priors.position_covariance as a diagonal covariance based on this std-dev in meters. \nDefault: NULL covariance (recommended)."
    )
    parser.add_argument(
        "--swap_latlon", action="store_true",
        help="Swap lat/lon when writing pose priors (debug option if alignment fails due to ordering)."
    )


    parser.add_argument("--camera", default="SIMPLE_RADIAL")
    parser.add_argument("--matching", default="spatial", choices=["spatial", "exhaustive", "sequential", "vocab_tree"])
    parser.add_argument("--matcher_args", default="")
    parser.add_argument("--sift_num_threads", type=int, default=-1)
    parser.add_argument("--sift_max_image_size", type=int, default=3200)
    parser.add_argument("--sift_max_num_features", type=int, default=8192)
    parser.add_argument("--sift_matching_max_num_matches", type=int, default=32768)
    parser.add_argument(
        "--sift_use_gpu",
        default="auto",
        help="COLMAP SiftExtraction.use_gpu. Use auto to try GPU first and fall back to CPU if needed.",
    )
    parser.add_argument(
        "--sift_matching_use_gpu",
        default="auto",
        help="COLMAP SiftMatching.use_gpu. Use auto to try GPU first and fall back to CPU if needed.",
    )
    parser.add_argument("--mapper_multiple_models", type=int, default=1)
    parser.add_argument("--min_model_size", type=int, default=10)
    parser.add_argument("--init_min_num_inliers", type=int, default=100)
    parser.add_argument("--abs_pose_min_num_inliers", type=int, default=30)
    parser.add_argument(
        "--image_list_path",
        default="",
        help=(
            "Optional COLMAP image list used by feature_extractor and image_undistorter. "
            "For strict protocol runs this should contain train+test images only."
        ),
    )
    parser.add_argument(
        "--mapper_image_list_path",
        default="",
        help="Optional COLMAP image list passed only to mapper, typically the frozen train split.",
    )
    parser.add_argument(
        "--register_images_after_mapper",
        action="store_true",
        help=(
            "After train-only mapping, localize remaining database images into the frozen mapper model "
            "with image_registrator before undistortion."
        ),
    )
    parser.add_argument(
        "--registration_required_image_list_path",
        default="",
        help="Optional list of images that must be registered by image_registrator, typically held-out test images.",
    )
    parser.add_argument(
        "--registration_audit_path",
        default="",
        help="Optional JSON path for train-only SfM / test-localization audit payload.",
    )
    parser.add_argument(
        "--fail_on_missing_registration",
        action="store_true",
        help="Fail if any image in --registration_required_image_list_path is not localized.",
    )
    parser.add_argument(
        "--strict_no_point_growth_after_registration",
        action="store_true",
        help="Fail if image_registrator increases the 3D point count, guarding against test-driven triangulation.",
    )
    parser.add_argument(
        "--image_registrator_args",
        default="",
        help="Extra arguments appended to COLMAP image_registrator.",
    )

    parser.add_argument(
        "--georegistration_mode",
        choices=["auto", "force", "off"],
        default="auto",
        help=(
            "WGS84 pose-prior alignment policy. auto (default) runs the selected "
            "georegistration backend when enough EXIF/RTK GPS priors overlap the selected "
            "sparse model, and falls back to the unaligned COLMAP model if priors are "
            "missing or alignment fails. force requires alignment to succeed. off preserves "
            "the historical local-COLMAP mode."
        ),
    )
    parser.add_argument(
        "--min_georegistration_overlap",
        type=int,
        default=3,
        help="Minimum registered model images with EXIF/RTK GPS priors required before auto georegistration is attempted.",
    )
    parser.add_argument(
        "--georegistration_backend",
        choices=["custom_sim3", "colmap_model_aligner"],
        default="custom_sim3",
        help=(
            "Georegistration backend. custom_sim3 writes a local ENU COLMAP sparse model by fitting "
            "registered camera centers to WGS84 EXIF/RTK priors. colmap_model_aligner uses COLMAP's "
            "model_aligner command."
        ),
    )
    parser.add_argument(
        "--use_model_aligner",
        action="store_true",
        help="Legacy alias for --georegistration_mode force.",
    )
    parser.add_argument(
        "--model_aligner_args",
        default="--ref_is_gps 1 --alignment_type enu --alignment_max_error 30.0",
        help="Arguments appended to COLMAP model_aligner when WGS84/ENU georegistration is attempted.",
    )

    parser.add_argument("--resize", action="store_true")

    args = parser.parse_args()

    root = Path(args.source_path)
    input_dir = root / "input"
    feature_image_list = Path(args.image_list_path).resolve() if args.image_list_path else None
    mapper_image_list = Path(args.mapper_image_list_path).resolve() if args.mapper_image_list_path else None
    registration_required_list = Path(args.registration_required_image_list_path).resolve() if args.registration_required_image_list_path else None
    registration_audit_path = Path(args.registration_audit_path).resolve() if args.registration_audit_path else None
    distorted_dir = root / "distorted"
    sparse_root = distorted_dir / "sparse"
    db_path = distorted_dir / "database.db"
    sparse_aligned = distorted_dir / "sparse_aligned"
    sparse_registered = distorted_dir / "sparse_registered"

    distorted_dir.mkdir(parents=True, exist_ok=True)
    sparse_root.mkdir(parents=True, exist_ok=True)

    colmap_exe = args.colmap_executable
    sift_use_gpu = normalize_colmap_gpu_mode(args.sift_use_gpu)
    sift_matching_use_gpu = normalize_colmap_gpu_mode(args.sift_matching_use_gpu)

    feature_cmd = [
        colmap_exe, "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(input_dir),
        "--ImageReader.camera_model", str(args.camera),
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.num_threads", str(args.sift_num_threads),
        "--SiftExtraction.max_image_size", str(args.sift_max_image_size),
        "--SiftExtraction.max_num_features", str(args.sift_max_num_features),
    ]
    if feature_image_list:
        feature_cmd += ["--image_list_path", str(feature_image_list)]
    run_colmap_gpu_cmd(feature_cmd, "--SiftExtraction.use_gpu", sift_use_gpu, retry_cleanup_paths=[db_path])

    matcher_common_args = [
        "--SiftMatching.max_num_matches", str(args.sift_matching_max_num_matches),
    ] + split_args(args.matcher_args)

    if args.matching == "spatial":
        run_colmap_gpu_cmd(
            [colmap_exe, "spatial_matcher", "--database_path", str(db_path)] + matcher_common_args,
            "--SiftMatching.use_gpu",
            sift_matching_use_gpu,
        )
    elif args.matching == "exhaustive":
        run_colmap_gpu_cmd(
            [colmap_exe, "exhaustive_matcher", "--database_path", str(db_path)] + matcher_common_args,
            "--SiftMatching.use_gpu",
            sift_matching_use_gpu,
        )
    elif args.matching == "sequential":
        run_colmap_gpu_cmd(
            [colmap_exe, "sequential_matcher", "--database_path", str(db_path)] + matcher_common_args,
            "--SiftMatching.use_gpu",
            sift_matching_use_gpu,
        )
    else:
        run_colmap_gpu_cmd(
            [colmap_exe, "vocab_tree_matcher", "--database_path", str(db_path)] + matcher_common_args,
            "--SiftMatching.use_gpu",
            sift_matching_use_gpu,
        )

    mapper_cmd = [
        colmap_exe, "mapper",
        "--database_path", str(db_path),
        "--image_path", str(input_dir),
        "--output_path", str(sparse_root),
        "--Mapper.multiple_models", str(args.mapper_multiple_models),
        "--Mapper.min_model_size", str(args.min_model_size),
        "--Mapper.init_min_num_inliers", str(args.init_min_num_inliers),
        "--Mapper.abs_pose_min_num_inliers", str(args.abs_pose_min_num_inliers),
    ]
    if mapper_image_list:
        mapper_cmd += ["--image_list_path", str(mapper_image_list)]
    run_cmd(mapper_cmd)

    best_model = select_best_sparse_model(sparse_root)

    gps_map = populate_pose_priors_from_exif(db_path, input_dir, exiftool_exe=args.exiftool_executable, wgs84_code=args.wgs84_code, prior_position_std_m=args.prior_position_std_m, swap_latlon=args.swap_latlon)
    georeg_overlap = sanity_check_overlap(best_model, gps_map)

    aligned_model_for_undistort = best_model

    georeg_mode = "force" if args.use_model_aligner else str(args.georegistration_mode)
    georeg_status = {
        "mode": georeg_mode,
        "wgs84_code": int(args.wgs84_code),
        "pose_priors_from_exif_count": len(gps_map),
        "model_image_count": int(georeg_overlap.get("model_image_count", 0)),
        "gps_overlap_count": int(georeg_overlap.get("gps_overlap_count", 0)),
        "min_overlap": int(args.min_georegistration_overlap),
        "attempted": False,
        "succeeded": False,
        "used_for_undistort": False,
        "model_aligner_args": str(args.model_aligner_args),
        "backend": str(args.georegistration_backend),
        "aligned_model_path": "",
        "fallback_reason": "",
    }

    enough_gps = georeg_status["gps_overlap_count"] >= int(args.min_georegistration_overlap)
    if georeg_mode == "off":
        georeg_status["fallback_reason"] = "georegistration_mode=off"
        log_info("WGS84/RTK model alignment disabled; using local COLMAP sparse model.")
    elif not enough_gps:
        reason = (
            f"insufficient EXIF/RTK GPS overlap for georegistration "
            f"({georeg_status['gps_overlap_count']}/{args.min_georegistration_overlap})"
        )
        georeg_status["fallback_reason"] = reason
        if georeg_mode == "force":
            raise RuntimeError(reason)
        log_warn(f"{reason}; falling back to local COLMAP sparse model.")
    else:
        if sparse_aligned.exists():
            shutil.rmtree(sparse_aligned)
        sparse_aligned.mkdir(parents=True, exist_ok=True)
        georeg_status["attempted"] = True
        try:
            if str(args.georegistration_backend) == "custom_sim3":
                summary = custom_sim3_georegister_model(best_model, sparse_aligned, gps_map)
                georeg_status["succeeded"] = True
                georeg_status["custom_sim3_summary"] = summary
                log_info(
                    "Custom WGS84/ENU Sim3 alignment succeeded: "
                    f"mean_error={summary['alignment_error_m']['mean']:.3f}m "
                    f"median_error={summary['alignment_error_m']['median']:.3f}m "
                    f"scale={summary['estimated_scale']:.6f}"
                )
            else:
                ref_images_path = distorted_dir / "gps_ref_images.txt"
                write_gps_ref_images(best_model, gps_map, ref_images_path)
                base_cmd = [
                    colmap_exe, "model_aligner",
                    "--input_path", str(best_model),
                    "--output_path", str(sparse_aligned),
                    "--ref_images_path", str(ref_images_path),
                ]
                user_tokens = split_args(args.model_aligner_args)
                log_info("Running WGS84/RTK COLMAP model_aligner:")
                try:
                    run_cmd(base_cmd + user_tokens)
                    georeg_status["succeeded"] = True
                except subprocess.CalledProcessError as e:
                    log_warn(f"model_aligner failed (exit={getattr(e, 'returncode', None)}). Will retry with relaxed alignment_max_error.")
                    retry_vals = [100, 300, 1000]
                    success = False
                    last_err = e
                    for v in retry_vals:
                        tokens2 = replace_alignment_max_error(user_tokens, v)
                        log_info(f"Retrying model_aligner with --alignment_max_error={v}")
                        try:
                            run_cmd(base_cmd + tokens2)
                            success = True
                            break
                        except subprocess.CalledProcessError as e2:
                            last_err = e2
                            continue
                    georeg_status["succeeded"] = success
                    if not success:
                        raise last_err
                export_model_as_txt(colmap_exe, sparse_aligned)
        except Exception as e:
            reason = f"WGS84/RTK georegistration failed: {e}"
            georeg_status["fallback_reason"] = reason
            georeg_status["succeeded"] = False
            if georeg_mode == "force":
                raise
            log_warn(f"{reason}; falling back to local COLMAP sparse model.")

        if georeg_status["succeeded"]:
            aligned_model_for_undistort = sparse_aligned
            georeg_status["used_for_undistort"] = True
            georeg_status["aligned_model_path"] = str(sparse_aligned)
            log_info(f"WGS84/RTK model alignment succeeded; undistorting from: {sparse_aligned}")

    raw_sfm_audit = {
        "protocol": "all_images_sfm",
        "source_path": str(root),
        "input_dir": str(input_dir),
        "feature_image_list_path": str(feature_image_list) if feature_image_list else "",
        "feature_image_list_count": len(read_name_list(feature_image_list)) if feature_image_list else None,
        "mapper_image_list_path": str(mapper_image_list) if mapper_image_list else "",
        "sfm_images_used_for_mapper_count": len(read_name_list(mapper_image_list)) if mapper_image_list else None,
        "mapper_model_path": str(best_model),
        "mapper_registered_image_count": read_num_registered_images(best_model),
        "mapper_points3d_count": read_num_points3d(best_model),
        "test_localization_requested": bool(args.register_images_after_mapper),
        "test_localization_count": None,
        "test_localization_missing_count": None,
        "post_registration_ba_applied": False,
        "triangulation_extended_with_test": False,
        "georegistration": georeg_status,
    }

    if args.register_images_after_mapper:
        required_names = read_name_list(registration_required_list) if registration_required_list else []
        before_points = read_num_points3d(aligned_model_for_undistort)
        before_names = read_image_names(aligned_model_for_undistort)
        if sparse_registered.exists():
            shutil.rmtree(sparse_registered)
        sparse_registered.mkdir(parents=True, exist_ok=True)
        cmd = [
            colmap_exe, "image_registrator",
            "--database_path", str(db_path),
            "--input_path", str(aligned_model_for_undistort),
            "--output_path", str(sparse_registered),
            "--Mapper.fix_existing_frames", "1",
        ] + split_args(args.image_registrator_args)
        run_cmd(cmd)

        after_names = read_image_names(sparse_registered)
        after_points = read_num_points3d(sparse_registered)
        localized_required, missing_required = _registered_name_hits(after_names, required_names)
        point_growth = after_points > before_points if before_points >= 0 and after_points >= 0 else False
        raw_sfm_audit.update({
            "protocol": "train_only_sfm_test_localization",
            "registration_input_model_path": str(aligned_model_for_undistort),
            "registration_output_model_path": str(sparse_registered),
            "registration_required_image_list_path": str(registration_required_list) if registration_required_list else "",
            "registration_required_image_count": len(required_names),
            "train_model_registered_image_count": len(before_names),
            "registered_model_image_count": len(after_names),
            "test_localization_count": len(localized_required),
            "test_localization_missing_count": len(missing_required),
            "test_localization_missing_preview": missing_required[:16],
            "points3d_before_registration": before_points,
            "points3d_after_registration": after_points,
            "triangulation_extended_with_test": bool(point_growth),
            "post_registration_ba_applied": False,
            "post_registration_ba_note": (
                "No explicit post-registration bundle adjustment is launched by this pipeline. "
                "COLMAP image_registrator may internally refine localized test poses; "
                "existing train frames are fixed and point growth is guarded."
            ),
            "registration_refinement_policy": (
                "COLMAP image_registrator internal refinement may run; "
                "existing train frames are fixed and point growth is guarded."
            ),
            "fix_existing_frames": True,
        })
        if args.fail_on_missing_registration and missing_required:
            if registration_audit_path:
                _write_json(registration_audit_path, raw_sfm_audit)
            raise RuntimeError(
                f"image_registrator failed to localize {len(missing_required)}/{len(required_names)} required images. "
                f"First missing: {missing_required[:8]}"
            )
        if args.strict_no_point_growth_after_registration and point_growth:
            if registration_audit_path:
                _write_json(registration_audit_path, raw_sfm_audit)
            raise RuntimeError(
                f"image_registrator increased 3D point count from {before_points} to {after_points}; "
                "strict no-test-triangulation guard failed."
            )
        aligned_model_for_undistort = sparse_registered

    if registration_audit_path:
        _write_json(registration_audit_path, raw_sfm_audit)

    undistort_cmd = [
        colmap_exe, "image_undistorter",
        "--image_path", str(input_dir),
        "--input_path", str(aligned_model_for_undistort),
        "--output_path", str(root),
        "--output_type", "COLMAP",
    ]
    if feature_image_list:
        undistort_cmd += ["--image_list_path", str(feature_image_list)]
    run_cmd(undistort_cmd)

    # Make the output compatible with 3DGS (expects <root>/sparse/0).
    try:
        model0 = ensure_3dgs_sparse_layout(root)
        export_model_as_txt(colmap_exe, model0)
        ensure_camera_models_supported_for_3dgs(colmap_exe, model0)
        export_model_as_txt(colmap_exe, model0)  # keep TXT after possible BIN regen
        if registration_audit_path and registration_audit_path.exists():
            shutil.copy2(registration_audit_path, model0 / "raw_sfm_protocol_audit.json")
        log_info(f"3DGS layout: images={root/'images'} | sparse_model={model0}")
    except Exception as e:
        log_warn(f"Failed to normalize/export sparse model for 3DGS: {e}")

    if args.resize:
        try:
            from PIL import Image
        except Exception as e:
            log_warn(f"PIL not available, skip resize. ({e})")
            return

        images_dir = root / "images"
        if not images_dir.exists():
            log_warn(f"images dir not found after undistort: {images_dir}; skip resize.")
            return

        scales = [2, 4, 8]
        for s in scales:
            (root / f"images_{s}").mkdir(parents=True, exist_ok=True)

        img_files = []
        for ext in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"):
            img_files.extend(images_dir.glob(ext))

        log_info(f"Resizing {len(img_files)} undistorted images...")
        for fp in img_files:
            try:
                im = Image.open(fp)
                w, h = im.size
                for s in scales:
                    ow, oh = max(1, w // s), max(1, h // s)
                    im_s = im.resize((ow, oh), resample=Image.BILINEAR)
                    out_path = root / f"images_{s}" / fp.name
                    im_s.save(out_path)
            except Exception as e:
                log_warn(f"Failed to resize {fp}: {e}")

    log_info("Done.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        log_err(f"Command failed: {e}")
        sys.exit(e.returncode)
    except Exception as e:
        log_err(str(e))
        raise
