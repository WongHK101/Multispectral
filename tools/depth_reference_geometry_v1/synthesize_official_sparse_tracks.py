from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from plyfile import PlyData

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene.colmap_loader import rotmat2qvec


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def _frame_image_name(frame: Dict[str, Any]) -> str:
    return Path(str(frame["file_path"])).name


def _frame_world_to_camera(frame: Dict[str, Any]) -> np.ndarray:
    c2w = np.asarray(frame["transform_matrix"], dtype=np.float64).copy()
    if c2w.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform_matrix, got {c2w.shape}")
    c2w[:3, 1:3] *= -1.0
    return np.linalg.inv(c2w)


def _frame_to_colmap_pose(frame: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    w2c = _frame_world_to_camera(frame)
    return rotmat2qvec(w2c[:3, :3]), w2c[:3, 3]


def _actual_image_size(frame: Dict[str, Any]) -> tuple[int, int] | None:
    image_path = Path(str(frame.get("file_path", "")))
    if not image_path.exists():
        return None
    from PIL import Image

    with Image.open(image_path) as image:
        return int(image.width), int(image.height)


def _camera_params_from_frame(
    frame: Dict[str, Any],
    *,
    camera_model: str,
) -> tuple[int, int, str, list[float], Dict[str, Any]]:
    frame_width = int(frame.get("w", 0))
    frame_height = int(frame.get("h", 0))
    width = frame_width
    height = frame_height
    if width <= 0 or height <= 0:
        raise ValueError(f"Frame is missing valid w/h: {frame}")
    fx = float(frame.get("fl_x", 0.0))
    fy = float(frame.get("fl_y", 0.0))
    if fx <= 0.0 or fy <= 0.0:
        angle_x = float(frame.get("camera_angle_x", 0.0))
        if angle_x <= 0.0:
            raise ValueError(f"Frame is missing focal length and camera_angle_x: {frame}")
        fx = 0.5 * float(width) / np.tan(0.5 * angle_x)
        fy = fx
    cx = float(frame.get("cx", width / 2.0))
    cy = float(frame.get("cy", height / 2.0))
    actual_size = _actual_image_size(frame)
    audit = {
        "frame_width": int(frame_width),
        "frame_height": int(frame_height),
        "actual_width": int(width),
        "actual_height": int(height),
        "intrinsics_scaled_to_actual_image": False,
    }
    if actual_size is not None:
        actual_width, actual_height = actual_size
        if actual_width != frame_width or actual_height != frame_height:
            sx = float(actual_width) / float(frame_width)
            sy = float(actual_height) / float(frame_height)
            fx *= sx
            fy *= sy
            cx *= sx
            cy *= sy
            width = int(actual_width)
            height = int(actual_height)
            audit.update(
                {
                    "actual_width": int(actual_width),
                    "actual_height": int(actual_height),
                    "scale_x": float(sx),
                    "scale_y": float(sy),
                    "intrinsics_scaled_to_actual_image": True,
                }
            )
    model = str(camera_model).upper()
    if model == "PINHOLE":
        params = [fx, fy, cx, cy]
    elif model == "OPENCV":
        params = [
            fx,
            fy,
            cx,
            cy,
            float(frame.get("k1", 0.0)),
            float(frame.get("k2", 0.0)),
            float(frame.get("p1", 0.0)),
            float(frame.get("p2", 0.0)),
        ]
    else:
        raise ValueError(f"Unsupported camera model: {camera_model}")
    audit["camera_model"] = model
    return width, height, model, params, audit


def _project_camera_point(cam: np.ndarray, params: list[float], camera_model: str) -> tuple[float, float] | None:
    z = float(cam[2])
    if not np.isfinite(z) or z <= 1.0e-6:
        return None
    fx, fy, cx, cy = [float(x) for x in params[:4]]
    xn = float(cam[0]) / z
    yn = float(cam[1]) / z
    if str(camera_model).upper() == "OPENCV":
        k1, k2, p1, p2 = [float(x) for x in params[4:8]]
        r2 = xn * xn + yn * yn
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        x_dist = xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
        y_dist = yn * radial + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn
        xn, yn = x_dist, y_dist
    return fx * xn + cx, fy * yn + cy


def _load_ply_vertices(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    xyz = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T.astype(np.float64, copy=False)
    names = {prop.name for prop in vertex.properties}
    if {"red", "green", "blue"}.issubset(names):
        rgb = np.vstack([vertex["red"], vertex["green"], vertex["blue"]]).T.astype(np.uint8, copy=False)
    else:
        rgb = np.full((xyz.shape[0], 3), 255, dtype=np.uint8)
    return xyz, rgb


def _link_or_copy_dir(src: Path, dst: Path, *, symlink: bool) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if symlink:
        os.symlink(str(src), str(dst), target_is_directory=True)
    else:
        shutil.copytree(src, dst)


def _copy_workspace_shell(base_workspace: Path, out_workspace: Path, *, symlink_images: bool) -> None:
    if out_workspace.exists():
        shutil.rmtree(out_workspace)
    out_workspace.mkdir(parents=True, exist_ok=True)
    for name in ("images", "input", "distorted"):
        src = base_workspace / name
        if src.exists():
            _link_or_copy_dir(src, out_workspace / name, symlink=symlink_images)
    stereo_src = base_workspace / "stereo"
    stereo_dst = out_workspace / "stereo"
    stereo_dst.mkdir(parents=True, exist_ok=True)
    for cfg_name in ("patch-match.cfg", "fusion.cfg"):
        cfg_src = stereo_src / cfg_name
        if cfg_src.exists():
            shutil.copy2(cfg_src, stereo_dst / cfg_name)
    (stereo_dst / "depth_maps").mkdir(parents=True, exist_ok=True)
    (stereo_dst / "normal_maps").mkdir(parents=True, exist_ok=True)
    (stereo_dst / "consistency_graphs").mkdir(parents=True, exist_ok=True)
    (out_workspace / "sparse" / "0").mkdir(parents=True, exist_ok=True)


def _synthesize_tracks(
    *,
    points_xyz: np.ndarray,
    points_rgb: np.ndarray,
    frames: List[Dict[str, Any]],
    max_points: int,
    min_track_length: int,
    max_track_length: int,
    stride_offset: int,
    border_px: float,
    camera_model: str,
) -> Dict[str, Any]:
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    width, height, model, params, camera_audit = _camera_params_from_frame(frames[0], camera_model=camera_model)
    image_obs: list[list[tuple[int, float, float]]] = [[] for _ in frames]
    kept_points: list[tuple[int, np.ndarray, np.ndarray, list[tuple[int, float, float]]]] = []
    candidate_indices = np.arange(points_xyz.shape[0], dtype=np.int64)
    if stride_offset:
        candidate_indices = np.roll(candidate_indices, int(stride_offset) % max(1, candidate_indices.size))

    w2c_list = [_frame_world_to_camera(frame) for frame in frames]
    for point_idx in candidate_indices:
        xyz = points_xyz[int(point_idx)]
        observations: list[tuple[int, float, float]] = []
        for image_zero, w2c in enumerate(w2c_list):
            cam = w2c[:3, :3] @ xyz + w2c[:3, 3]
            projected = _project_camera_point(cam, params, model)
            if projected is None:
                continue
            x, y = projected
            if (
                np.isfinite(x)
                and np.isfinite(y)
                and border_px <= x < (width - border_px)
                and border_px <= y < (height - border_px)
            ):
                observations.append((image_zero + 1, x, y))
                if len(observations) >= max_track_length:
                    break
        if len(observations) < min_track_length:
            continue
        point3d_id = len(kept_points) + 1
        for image_id, x, y in observations:
            image_obs[image_id - 1].append((point3d_id, x, y))
        kept_points.append((point3d_id, xyz, points_rgb[int(point_idx)], observations))
        if len(kept_points) >= max_points:
            break

    point_tracks: dict[int, list[tuple[int, int]]] = {pid: [] for pid, *_ in kept_points}
    for image_zero, obs in enumerate(image_obs):
        for point2d_idx, (point3d_id, _x, _y) in enumerate(obs):
            point_tracks[int(point3d_id)].append((image_zero + 1, point2d_idx))

    return {
        "width": width,
        "height": height,
        "params": params,
        "camera_model": model,
        "kept_points": kept_points,
        "image_obs": image_obs,
        "point_tracks": point_tracks,
        "camera_audit": camera_audit,
    }


def _write_colmap_model(model_dir: Path, frames: List[Dict[str, Any]], track_data: Dict[str, Any]) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    width = int(track_data["width"])
    height = int(track_data["height"])
    params = [float(x) for x in track_data["params"]]
    camera_model = str(track_data.get("camera_model", "PINHOLE")).upper()
    image_obs = track_data["image_obs"]
    kept_points = track_data["kept_points"]
    point_tracks = track_data["point_tracks"]
    mean_obs = float(np.mean([len(obs) for obs in image_obs])) if image_obs else 0.0
    mean_track = float(np.mean([len(point_tracks[pid]) for pid, *_ in kept_points])) if kept_points else 0.0

    with (model_dir / "cameras.txt").open("w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(
            f"1 {camera_model} "
            f"{width} {height} "
            + " ".join(f"{x:.17g}" for x in params)
            + "\n"
        )

    with (model_dir / "images.txt").open("w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(frames)}, mean observations per image: {mean_obs:.6f}\n")
        for image_id, frame in enumerate(frames, start=1):
            qvec, tvec = _frame_to_colmap_pose(frame)
            values = [
                str(image_id),
                *(f"{float(x):.17g}" for x in qvec),
                *(f"{float(x):.17g}" for x in tvec),
                "1",
                _frame_image_name(frame),
            ]
            f.write(" ".join(values) + "\n")
            obs_parts: list[str] = []
            for point3d_id, x, y in image_obs[image_id - 1]:
                obs_parts.extend([f"{float(x):.6f}", f"{float(y):.6f}", str(int(point3d_id))])
            f.write(" ".join(obs_parts) + "\n")

    with (model_dir / "points3D.txt").open("w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(kept_points)}, mean track length: {mean_track:.6f}\n")
        for point3d_id, xyz, rgb, _observations in kept_points:
            track = point_tracks[int(point3d_id)]
            track_parts: list[str] = []
            for image_id, point2d_idx in track:
                track_parts.extend([str(int(image_id)), str(int(point2d_idx))])
            f.write(
                f"{int(point3d_id)} "
                + " ".join(f"{float(x):.17g}" for x in xyz)
                + f" {int(rgb[0])} {int(rgb[1])} {int(rgb[2])} 0 "
                + " ".join(track_parts)
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create synthetic COLMAP tracks for official transforms dense-MVS workspaces."
    )
    parser.add_argument("--source_path", required=True, help="Prepared official scene root with transforms_train.json")
    parser.add_argument("--base_workspace", required=True, help="Existing manual dense workspace to copy/symlink")
    parser.add_argument("--out_workspace", required=True)
    parser.add_argument("--strict_protocol_manifest", required=True)
    parser.add_argument("--out_manifest", required=True)
    parser.add_argument("--audit_path", required=True)
    parser.add_argument("--max_points", type=int, default=80000)
    parser.add_argument("--min_track_length", type=int, default=3)
    parser.add_argument("--max_track_length", type=int, default=24)
    parser.add_argument("--stride_offset", type=int, default=0)
    parser.add_argument("--border_px", type=float, default=4.0)
    parser.add_argument("--camera_model", default="PINHOLE", choices=("PINHOLE", "OPENCV"))
    parser.add_argument("--symlink_images", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    source_path = Path(args.source_path).resolve()
    base_workspace = Path(args.base_workspace).resolve()
    out_workspace = Path(args.out_workspace).resolve()
    transforms = _load_json(source_path / "transforms_train.json")
    frames = list(transforms["frames"])
    points_path = source_path / "points3d.ply"
    points_xyz, points_rgb = _load_ply_vertices(points_path)

    _copy_workspace_shell(base_workspace, out_workspace, symlink_images=bool(args.symlink_images))
    track_data = _synthesize_tracks(
        points_xyz=points_xyz,
        points_rgb=points_rgb,
        frames=frames,
        max_points=int(args.max_points),
        min_track_length=int(args.min_track_length),
        max_track_length=int(args.max_track_length),
        stride_offset=int(args.stride_offset),
        border_px=float(args.border_px),
        camera_model=str(args.camera_model),
    )
    _write_colmap_model(out_workspace / "sparse" / "0", frames, track_data)

    strict = _load_json(Path(args.strict_protocol_manifest).resolve())
    strict["artifacts"]["train_union_source_root"] = str(out_workspace)
    strict["synthetic_sparse_tracks"] = {
        "enabled": True,
        "source_points_path": str(points_path),
        "base_workspace": str(base_workspace),
        "out_workspace": str(out_workspace),
        "max_points": int(args.max_points),
        "min_track_length": int(args.min_track_length),
        "max_track_length": int(args.max_track_length),
        "border_px": float(args.border_px),
        "camera_model": str(args.camera_model),
    }
    _save_json(Path(args.out_manifest).resolve(), strict)

    image_obs = track_data["image_obs"]
    kept_points = track_data["kept_points"]
    point_tracks = track_data["point_tracks"]
    audit = {
        "source_path": str(source_path),
        "points_path": str(points_path),
        "input_point_count": int(points_xyz.shape[0]),
        "kept_point_count": int(len(kept_points)),
        "train_image_count": int(len(frames)),
        "min_observations_per_image": int(min(len(obs) for obs in image_obs)) if image_obs else 0,
        "median_observations_per_image": float(np.median([len(obs) for obs in image_obs])) if image_obs else 0.0,
        "max_observations_per_image": int(max(len(obs) for obs in image_obs)) if image_obs else 0,
        "mean_track_length": float(np.mean([len(point_tracks[pid]) for pid, *_ in kept_points])) if kept_points else 0.0,
        "camera_audit": track_data.get("camera_audit", {}),
        "out_workspace": str(out_workspace),
        "out_manifest": str(Path(args.out_manifest).resolve()),
    }
    _save_json(Path(args.audit_path).resolve(), audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
