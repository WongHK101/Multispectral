from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from plyfile import PlyData

from depth_reference_common import save_json
from scene.colmap_loader import rotmat2qvec


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _frame_image_name(frame: Dict[str, Any]) -> str:
    return Path(str(frame["file_path"])).name


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        raise ValueError(f"Unsupported link mode: {mode}")


def _frame_to_colmap_pose(frame: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    w2c = _frame_world_to_camera(frame)
    qvec = rotmat2qvec(w2c[:3, :3])
    tvec = w2c[:3, 3]
    return qvec, tvec


def _frame_world_to_camera(frame: Dict[str, Any]) -> np.ndarray:
    c2w = np.asarray(frame["transform_matrix"], dtype=np.float64).copy()
    if c2w.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform_matrix, got {c2w.shape}")
    # Official transforms use NeRF/OpenGL camera axes. Match the repo's
    # readCamerasFromTransforms conversion to COLMAP-style +Z-forward cameras.
    c2w[:3, 1:3] *= -1.0
    return np.linalg.inv(c2w)


def _load_ply_points_xyz(path: Path) -> np.ndarray:
    ply = PlyData.read(str(path))
    vertices = ply["vertex"]
    return np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype(np.float64, copy=False)


def _compute_patch_match_depth_range(source_path: Path, train_frames: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    points_path = source_path / "points3d.ply"
    if not points_path.exists():
        return None
    points = _load_ply_points_xyz(points_path)
    if points.size == 0:
        return None
    low_values: List[float] = []
    high_values: List[float] = []
    for frame in train_frames:
        w2c = _frame_world_to_camera(frame)
        points_cam = points @ w2c[:3, :3].T + w2c[:3, 3]
        positive_z = points_cam[:, 2]
        positive_z = positive_z[np.isfinite(positive_z) & (positive_z > 1.0e-6)]
        if positive_z.size < 10:
            continue
        low_values.append(float(np.quantile(positive_z, 0.01)))
        high_values.append(float(np.quantile(positive_z, 0.99)))
    if not low_values or not high_values:
        return None
    depth_min = max(1.0e-6, min(low_values) * 0.8)
    depth_max = max(high_values) * 1.2
    if not np.isfinite(depth_min) or not np.isfinite(depth_max) or depth_max <= depth_min:
        return None
    return {
        "depth_min": float(depth_min),
        "depth_max": float(depth_max),
        "source": str(points_path),
        "method": "train_camera_positive_z_1_99_quantiles_with_0_8_1_2_padding",
        "num_points": int(points.shape[0]),
        "num_train_views_used": int(len(low_values)),
    }


def _camera_params_from_frame(frame: Dict[str, Any]) -> tuple[int, int, List[float]]:
    width = int(frame.get("w", 0))
    height = int(frame.get("h", 0))
    if width <= 0 or height <= 0:
        raise ValueError(f"Frame is missing valid w/h: {frame}")
    fx = float(frame.get("fl_x", 0.0))
    fy = float(frame.get("fl_y", 0.0))
    if fx <= 0.0 or fy <= 0.0:
        camera_angle_x = float(frame.get("camera_angle_x", 0.0))
        if camera_angle_x <= 0.0:
            raise ValueError(f"Frame is missing focal length and camera_angle_x: {frame}")
        fx = 0.5 * float(width) / np.tan(0.5 * camera_angle_x)
        fy = fx
    cx = float(frame.get("cx", width / 2.0))
    cy = float(frame.get("cy", height / 2.0))
    return width, height, [fx, fy, cx, cy]


def _write_colmap_text_model(model_dir: Path, frames: List[Dict[str, Any]]) -> None:
    if not frames:
        raise ValueError("Cannot write an empty COLMAP model")
    model_dir.mkdir(parents=True, exist_ok=True)
    width, height, params = _camera_params_from_frame(frames[0])
    with (model_dir / "cameras.txt").open("w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(
            "1 PINHOLE "
            f"{int(width)} {int(height)} "
            + " ".join(f"{float(x):.17g}" for x in params)
            + "\n"
        )
    with (model_dir / "images.txt").open("w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(frames)}, mean observations per image: 0\n")
        for idx, frame in enumerate(frames, start=1):
            qvec, tvec = _frame_to_colmap_pose(frame)
            image_name = _frame_image_name(frame)
            values = [
                str(idx),
                *(f"{float(x):.17g}" for x in qvec),
                *(f"{float(x):.17g}" for x in tvec),
                "1",
                image_name,
            ]
            f.write(" ".join(values) + "\n\n")
    with (model_dir / "points3D.txt").open("w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 0, mean track length: 0\n")


def _write_split_lists(out_dir: Path, train_frames: List[Dict[str, Any]], test_frames: List[Dict[str, Any]]) -> Dict[str, str]:
    lists_dir = out_dir / "lists"
    lists_dir.mkdir(parents=True, exist_ok=True)
    train_list = lists_dir / "train_union.txt"
    test_list = lists_dir / "probe_test.txt"
    train_list.write_text("\n".join(_frame_image_name(f) for f in train_frames) + "\n", encoding="utf-8")
    test_list.write_text("\n".join(_frame_image_name(f) for f in test_frames) + "\n", encoding="utf-8")
    return {"train_union": str(train_list), "probe_test": str(test_list)}


def _write_pair_configs(stereo_dir: Path, image_names: List[str], window: int) -> None:
    stereo_dir.mkdir(parents=True, exist_ok=True)
    patch_lines: List[str] = []
    for idx, name in enumerate(image_names):
        left = max(0, idx - int(window))
        right = min(len(image_names), idx + int(window) + 1)
        sources = [image_names[j] for j in range(left, right) if j != idx]
        if not sources:
            sources = [n for n in image_names if n != name]
        patch_lines.append(name)
        patch_lines.append(", ".join(sources[: max(1, int(window) * 2)]))
    (stereo_dir / "patch-match.cfg").write_text("\n".join(patch_lines) + "\n", encoding="utf-8")
    (stereo_dir / "fusion.cfg").write_text("\n".join(image_names) + "\n", encoding="utf-8")


def _write_pose_neighbor_pair_configs(stereo_dir: Path, frames: List[Dict[str, Any]], num_sources: int) -> None:
    stereo_dir.mkdir(parents=True, exist_ok=True)
    names = [_frame_image_name(frame) for frame in frames]
    centers = []
    for frame in frames:
        c2w = np.asarray(frame["transform_matrix"], dtype=np.float64)
        if c2w.shape != (4, 4):
            raise ValueError(f"Expected 4x4 transform_matrix, got {c2w.shape}")
        centers.append(c2w[:3, 3].astype(np.float64, copy=False))
    centers_np = np.vstack(centers)
    max_sources = max(1, min(len(names) - 1, int(num_sources)))
    patch_lines: List[str] = []
    for idx, name in enumerate(names):
        distances = np.linalg.norm(centers_np - centers_np[idx], axis=1)
        order = np.argsort(distances)
        sources = [names[int(j)] for j in order if int(j) != idx][:max_sources]
        patch_lines.append(name)
        patch_lines.append(", ".join(sources))
    (stereo_dir / "patch-match.cfg").write_text("\n".join(patch_lines) + "\n", encoding="utf-8")
    (stereo_dir / "fusion.cfg").write_text("\n".join(names) + "\n", encoding="utf-8")


def _run(cmd: List[str], cwd: Path | None = None) -> None:
    print("Running: " + " ".join(str(x) for x in cmd), flush=True)
    completed = subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(str(x) for x in cmd)}")


def _prepare_manual_pinhole_dense_workspace(
    *,
    dense_workspace: Path,
    images_train: Path,
    sparse_train: Path,
    train_frames: List[Dict[str, Any]],
    link_mode: str,
    patch_match_source_window: int,
) -> None:
    images_dst = dense_workspace / "images"
    sparse_dst = dense_workspace / "sparse" / "0"
    stereo_dst = dense_workspace / "stereo"
    images_dst.mkdir(parents=True, exist_ok=True)
    sparse_dst.mkdir(parents=True, exist_ok=True)
    for frame in train_frames:
        image_name = _frame_image_name(frame)
        src = images_train / image_name
        if not src.exists():
            raise FileNotFoundError(f"Training image missing from prepared raw image dir: {src}")
        _link_or_copy(src, images_dst / image_name, link_mode)
    for src_file in sorted(sparse_train.iterdir()):
        if src_file.is_file():
            shutil.copy2(src_file, sparse_dst / src_file.name)
    _write_pose_neighbor_pair_configs(stereo_dst, train_frames, max(1, int(patch_match_source_window) * 2))
    (stereo_dst / "depth_maps").mkdir(parents=True, exist_ok=True)
    (stereo_dst / "normal_maps").mkdir(parents=True, exist_ok=True)
    (stereo_dst / "consistency_graphs").mkdir(parents=True, exist_ok=True)


def _argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare an official transforms scene for dense reference-depth construction via COLMAP."
    )
    parser.add_argument("--source_path", required=True, help="Prepared official scene root with transforms_train/test.json")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--scene_name", default="")
    parser.add_argument("--colmap_cmd", default="colmap")
    parser.add_argument("--link_mode", default="hardlink", choices=("copy", "hardlink", "symlink"))
    parser.add_argument("--max_train_views", type=int, default=0, help="Smoke-only cap. 0 keeps all training views.")
    parser.add_argument("--max_test_views", type=int, default=0, help="Smoke-only cap. 0 keeps all test views.")
    parser.add_argument("--image_undistorter_max_image_size", type=int, default=0)
    parser.add_argument("--patch_match_source_window", type=int, default=10)
    parser.add_argument("--run_image_undistorter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--manual_pinhole_workspace",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Construct the COLMAP dense workspace directly from PINHOLE official transforms, "
            "without running image_undistorter. This is intended for official transformed scenes "
            "that are already distortion-free and can avoid COLMAP image_undistorter crashes on "
            "zero-point text reconstructions."
        ),
    )
    return parser


def main() -> None:
    args = _argparser().parse_args()
    source_path = Path(args.source_path).resolve()
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_frames = list(_load_json(source_path / "transforms_train.json").get("frames", []))
    test_frames = list(_load_json(source_path / "transforms_test.json").get("frames", []))
    if int(args.max_train_views) > 0:
        train_frames = train_frames[: int(args.max_train_views)]
    if int(args.max_test_views) > 0:
        test_frames = test_frames[: int(args.max_test_views)]
    if not train_frames or not test_frames:
        raise RuntimeError(f"Expected non-empty train/test frames under {source_path}")

    scene_name = str(args.scene_name).strip() or source_path.parent.name or source_path.name

    raw_root = out_dir / "raw_colmap"
    images_train = raw_root / "images_train"
    images_all = raw_root / "images_all"
    sparse_train = raw_root / "sparse_train" / "0"
    sparse_all = raw_root / "sparse_all" / "0"
    for frame in train_frames:
        src = Path(str(frame["file_path"])).resolve()
        _link_or_copy(src, images_train / src.name, str(args.link_mode))
        _link_or_copy(src, images_all / src.name, str(args.link_mode))
    for frame in test_frames:
        src = Path(str(frame["file_path"])).resolve()
        _link_or_copy(src, images_all / src.name, str(args.link_mode))

    _write_colmap_text_model(sparse_train, train_frames)
    _write_colmap_text_model(sparse_all, train_frames + test_frames)
    lists = _write_split_lists(out_dir, train_frames, test_frames)
    patch_match_depth_range = _compute_patch_match_depth_range(source_path, train_frames)

    dense_workspace = out_dir / "reference_source_workspace_training_only"
    dense_workspace_mode = "image_undistorter"
    if bool(args.manual_pinhole_workspace):
        dense_workspace_mode = "manual_pinhole"
        _prepare_manual_pinhole_dense_workspace(
            dense_workspace=dense_workspace,
            images_train=images_train,
            sparse_train=sparse_train,
            train_frames=train_frames,
            link_mode=str(args.link_mode),
            patch_match_source_window=int(args.patch_match_source_window),
        )
    elif bool(args.run_image_undistorter):
        cmd = [
            str(args.colmap_cmd),
            "image_undistorter",
            "--image_path",
            str(images_train),
            "--input_path",
            str(sparse_train),
            "--output_path",
            str(dense_workspace),
            "--output_type",
            "COLMAP",
        ]
        if int(args.image_undistorter_max_image_size) > 0:
            cmd += ["--max_image_size", str(int(args.image_undistorter_max_image_size))]
        _run(cmd)
    else:
        raise RuntimeError(
            "--no-run_image_undistorter requires --manual_pinhole_workspace for dense reference preparation"
        )

    # Normalize COLMAP's undistorted output into the sparse/0 layout expected by
    # build_depth_reference.py.
    sparse_flat = dense_workspace / "sparse"
    sparse_zero = dense_workspace / "sparse" / "0"
    if not sparse_zero.exists():
        sparse_zero.mkdir(parents=True, exist_ok=True)
        for item in sparse_flat.iterdir():
            if item.is_file():
                shutil.copy2(item, sparse_zero / item.name)
    _write_pose_neighbor_pair_configs(dense_workspace / "stereo", train_frames, max(1, int(args.patch_match_source_window) * 2))

    probe_scene = out_dir / "probe_colmap_scene"
    (probe_scene / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    for src_file in sparse_all.iterdir():
        if src_file.is_file():
            shutil.copy2(src_file, probe_scene / "sparse" / "0" / src_file.name)
    if (probe_scene / "images").exists() or (probe_scene / "images").is_symlink():
        if (probe_scene / "images").is_symlink():
            (probe_scene / "images").unlink()
        else:
            shutil.rmtree(probe_scene / "images")
    if os.name != "nt":
        os.symlink(str(images_all), str(probe_scene / "images"), target_is_directory=True)
    else:
        shutil.copytree(images_all, probe_scene / "images")

    strict_manifest = {
        "protocol_name": "official-transforms-dense-reference-preparation-v1",
        "scene_name": scene_name,
        "source_path": str(source_path),
        "artifacts": {
            "train_union_source_root": str(dense_workspace),
            "strict_thermal_root": str(probe_scene),
        },
        "lists": lists,
        "counts": {
            "train_views": len(train_frames),
            "test_views": len(test_frames),
        },
        "dense_workspace_mode": dense_workspace_mode,
        "patch_match_depth_range": patch_match_depth_range,
        "notes": (
            "Official transforms scene converted to COLMAP text sparse models. "
            "Dense reference uses train views only; probe scene contains train+test cameras for held-out rendering."
        ),
    }
    strict_manifest_path = out_dir / "strict_protocol_manifest.json"
    save_json(strict_manifest_path, strict_manifest)
    save_json(
        out_dir / "prepare_manifest.json",
        {
            "scene_name": scene_name,
            "source_path": str(source_path),
            "out_dir": str(out_dir),
            "dense_workspace": str(dense_workspace),
            "probe_scene": str(probe_scene),
            "strict_protocol_manifest": str(strict_manifest_path),
            "train_image_count": len(train_frames),
            "test_image_count": len(test_frames),
            "dense_workspace_mode": dense_workspace_mode,
            "manual_pinhole_workspace": bool(args.manual_pinhole_workspace),
            "patch_match_depth_range": patch_match_depth_range,
            "image_undistorter_max_image_size": int(args.image_undistorter_max_image_size),
            "patch_match_source_window": int(args.patch_match_source_window),
        },
    )
    print(f"STRICT_PROTOCOL_MANIFEST {strict_manifest_path}")
    print(f"DENSE_WORKSPACE {dense_workspace}")
    print(f"PROBE_SCENE {probe_scene}")


if __name__ == "__main__":
    main()
