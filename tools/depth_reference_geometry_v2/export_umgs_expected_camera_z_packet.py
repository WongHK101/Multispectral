"""Opt-in UMGS expected-camera-z packet exporter.

This module is an interface/exporter patch only. It is intentionally disabled
by default and is not used by training or regular RGB rendering. The exported
camera-z values are opacity-composited Gaussian-mean camera-z in the native
reconstruction coordinate system, not physical metre-unit surface depth and
not geometry ground truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


SCHEMA_VERSION = "umgs_expected_camera_z_packet_v1"
PACKET_CHANNELS = {
    "accumulated_opacity": 0,
    "weighted_camera_z_sum": 1,
    "expected_camera_z": 2,
    "numeric_valid": 3,
    "weighted_camera_z2_sum": 4,
    "camera_z_variance": 5,
}
REQUIRED_RUNTIME_SOURCE_PATHS = [
    "gaussian_renderer/__init__.py",
    "submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py",
    "submodules/diff-gaussian-rasterization/rasterize_points.cu",
    "submodules/diff-gaussian-rasterization/rasterize_points.h",
    "submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu",
    "submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.h",
    "submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer.h",
    "submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu",
    "tools/depth_reference_geometry_v2/export_umgs_expected_camera_z_packet.py",
]
DEFAULT_A0_TOLERANCES = {
    "identity_rtol": 1e-5,
    "identity_atol": 1e-6,
    "repeatability_rtol": 1e-6,
    "repeatability_atol": 1e-6,
    "opacity_min": -1e-6,
    "opacity_max": 1.0 + 1e-5,
    "variance_min": -1e-6,
    "rgb_compatibility_max_abs_delta": 1e-6,
}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_file_sha256(path: str | Path, expected_sha256: str, label: str) -> str:
    actual = sha256_file(path)
    if actual.lower() != str(expected_sha256).lower():
        raise ValueError(f"{label} SHA256 mismatch: expected {expected_sha256}, got {actual} for {path}")
    return actual


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical_json_sha256(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def git_text(repo_root: Path, args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def git_commit(repo_root: Path) -> str:
    return git_text(repo_root, ["rev-parse", "HEAD"])


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def name_keys(name: str | Path) -> set[str]:
    p = Path(str(name))
    return {str(name), p.name, p.stem}


def target_matches(candidate: str, target: str) -> bool:
    return bool(name_keys(candidate) & name_keys(target))


def read_name_file(path: str | Path, *, label: str) -> list[str]:
    names: list[str] = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{label} split file does not exist: {p}")
    if not p.is_file():
        raise FileNotFoundError(f"{label} split path is not a regular file: {p}")
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        names.append(Path(s).name)
    return names


def verify_heldout_manifest(args: argparse.Namespace) -> dict[str, Any]:
    camera_manifest = Path(args.camera_manifest)
    split_manifest = Path(args.split_manifest)
    verify_file_sha256(camera_manifest, args.camera_manifest_sha256, "camera manifest")
    verify_file_sha256(split_manifest, args.split_manifest_sha256, "split/materialization manifest")

    camera_data = read_json(camera_manifest)
    split_data = read_json(split_manifest)
    heldout = list(camera_data.get("heldout_images", []))
    matches = [row for row in heldout if target_matches(str(row.get("image_name", "")), args.target)]
    if len(matches) != 1:
        raise ValueError(
            f"target {args.target!r} must match exactly one held-out camera in {camera_manifest}, found {len(matches)}"
        )
    target_row = matches[0]

    split = split_data.get("split", {}) if isinstance(split_data, dict) else {}
    train_file = split.get("train_file", "")
    test_file = split.get("test_file", "")
    if not train_file or not test_file:
        raise ValueError("split manifest must define non-empty train_file and test_file")
    train_sha256 = verify_file_sha256(train_file, args.train_file_sha256, "train split file")
    test_sha256 = verify_file_sha256(test_file, args.test_file_sha256, "test split file")
    test_names = read_name_file(test_file, label="test")
    train_names = read_name_file(train_file, label="train")
    if not any(target_matches(n, args.target) for n in test_names):
        raise ValueError(f"target {args.target!r} is not listed in official test split {split.get('test_file')}")
    if any(target_matches(n, args.target) for n in train_names):
        raise ValueError(f"target {args.target!r} unexpectedly appears in training split {split.get('train_file')}")

    return {
        "camera_manifest": {
            "path": str(camera_manifest),
            "sha256": sha256_file(camera_manifest),
            "schema": camera_data.get("schema", ""),
            "heldout_camera_use": camera_data.get("heldout_camera_use", ""),
            "heldout_count": len(heldout),
            "target_row": target_row,
        },
        "split_manifest": {
            "path": str(split_manifest),
            "sha256": sha256_file(split_manifest),
            "schema": split_data.get("schema", "") if isinstance(split_data, dict) else "",
            "split": split,
            "authoritative_a0_split_statement": (
                "The current train/test files, bound by explicit SHA256 values, are the authoritative A0 execution split. "
                "The historical materialization summary contains stale embedded split hashes and is retained only as provenance."
            ),
            "train_file": str(train_file),
            "train_file_expected_sha256": str(args.train_file_sha256),
            "train_file_actual_sha256": train_sha256,
            "train_file_line_count": len(train_names),
            "test_file": str(test_file),
            "test_file_expected_sha256": str(args.test_file_sha256),
            "test_file_actual_sha256": test_sha256,
            "test_file_line_count": len(test_names),
            "target_in_test_file": bool(test_names and any(target_matches(n, args.target) for n in test_names)),
            "target_in_train_file": bool(train_names and any(target_matches(n, args.target) for n in train_names)),
        },
        "manifest_camera_parameter_comparison": "not_available_in_current_heldout_manifest",
    }


def verify_colmap_sparse_hash_manifest(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.colmap_sparse_hash_manifest)
    manifest_sha256 = verify_file_sha256(
        manifest_path,
        args.colmap_sparse_hash_manifest_sha256,
        "COLMAP sparse hash manifest",
    )
    approved_sparse_dir = (Path(args.source_path) / "sparse" / "0").resolve()
    if not approved_sparse_dir.is_dir():
        raise FileNotFoundError(f"approved COLMAP sparse dir does not exist: {approved_sparse_dir}")
    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_name = str(row.get("file", "")).strip()
            row_path = str(row.get("path", "")).strip()
            expected_sha256 = str(row.get("sha256", "")).strip()
            if not file_name or not expected_sha256:
                continue
            candidate = Path(row_path) if row_path else approved_sparse_dir / file_name
            if not candidate.is_absolute():
                candidate = approved_sparse_dir / candidate
            actual_path = candidate.resolve()
            try:
                actual_path.relative_to(approved_sparse_dir)
            except ValueError as exc:
                raise ValueError(
                    f"COLMAP sparse manifest path escapes approved sparse dir: {actual_path} not under {approved_sparse_dir}"
                ) from exc
            if actual_path.name != file_name:
                raise ValueError(f"COLMAP sparse manifest file/path mismatch: file={file_name}, path={actual_path}")
            actual_sha256 = verify_file_sha256(actual_path, expected_sha256, f"COLMAP sparse file {file_name}")
            rows.append(
                {
                    "file": file_name,
                    "path": str(actual_path),
                    "sha256": actual_sha256,
                    "bytes": actual_path.stat().st_size,
                }
            )
    present = {row["file"] for row in rows}
    required = {"cameras.bin", "images.bin", "points3D.bin", "train.txt", "test.txt"}
    missing = sorted(required - present)
    if missing:
        raise ValueError(f"COLMAP sparse hash manifest missing required files: {missing}")
    return {
        "path": str(manifest_path),
        "sha256": manifest_sha256,
        "approved_sparse_dir": str(approved_sparse_dir),
        "required_files": sorted(required),
        "files": rows,
    }


def verify_runtime_source_hash_manifest(path: str | Path, expected_sha256: str, repo_root: Path) -> list[dict[str, Any]]:
    manifest_path = Path(path)
    verify_file_sha256(manifest_path, expected_sha256, "runtime source hash manifest")
    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel = str(row.get("path", "")).replace("\\", "/").strip()
            expected = str(row.get("sha256", "")).strip()
            if not rel or not expected:
                continue
            source_path = Path(rel) if Path(rel).is_absolute() else repo_root / rel
            actual = verify_file_sha256(source_path, expected, f"runtime source {rel}")
            rows.append({"path": rel, "sha256": actual})
    present = {r["path"] for r in rows}
    missing = [p for p in REQUIRED_RUNTIME_SOURCE_PATHS if p not in present]
    if missing:
        raise ValueError(f"runtime source hash manifest missing required paths: {missing}")
    return rows


def verify_model_path_and_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    model_path = Path(args.model_path)
    checkpoint = Path(args.checkpoint)
    if not model_path.exists():
        raise FileNotFoundError(f"model path does not exist: {model_path}")
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    try:
        checkpoint.relative_to(model_path)
        checkpoint_under_model_path = True
    except ValueError:
        checkpoint_under_model_path = False
    if not checkpoint_under_model_path:
        raise ValueError(f"checkpoint must be under --model-path: checkpoint={checkpoint}, model_path={model_path}")
    checkpoint_sha = verify_file_sha256(checkpoint, args.checkpoint_sha256, "checkpoint")
    config_candidates = [model_path / "cfg_args", model_path / "config.json", model_path / "run_manifest.json"]
    configs = []
    for p in config_candidates:
        if p.exists() and p.is_file():
            configs.append({"path": str(p), "sha256": sha256_file(p)})
    return {
        "model_path": str(model_path),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_under_model_path": checkpoint_under_model_path,
        "model_config_files": configs,
    }


def preflight_before_cuda(args: argparse.Namespace, repo_root: Path) -> dict[str, Any]:
    manifest_info = verify_heldout_manifest(args)
    sparse_hash_manifest = verify_colmap_sparse_hash_manifest(args)
    source_rows = verify_runtime_source_hash_manifest(
        args.runtime_source_hash_manifest,
        args.runtime_source_hash_manifest_sha256,
        repo_root,
    )
    model_info = verify_model_path_and_checkpoint(args)
    return {
        "preflight_checked_before_cuda_or_checkpoint_load": True,
        "manifest_info": manifest_info,
        "colmap_sparse_hash_manifest": sparse_hash_manifest,
        "runtime_source_hash_manifest": {
            "path": str(args.runtime_source_hash_manifest),
            "sha256": sha256_file(args.runtime_source_hash_manifest),
            "working_tree_overlay": True,
            "source_files": source_rows,
        },
        "model_info": model_info,
    }


def packet_schema() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "description": (
            "Opacity-composited expected camera-z in native reconstruction "
            "coordinates; not physical metre-unit surface depth and not geometry ground truth."
        ),
        "compositing_weight": "w_i = T_i * alpha_i, matching the RGB renderer front-to-back alpha compositing",
        "z_source": "Gaussian mean transformed by the renderer world-to-camera transform; positive camera-z is required",
        "numeric_valid_formula": (
            "finite(A) and finite(M1) and finite(expected_camera_z) and "
            "A > opacity_epsilon and expected_camera_z > 0 and camera_z_variance >= 0 after tolerance clamp"
        ),
        "channels": PACKET_CHANNELS,
        "primary_arrays": [
            "accumulated_opacity",
            "weighted_camera_z_sum",
            "expected_camera_z",
            "numeric_valid",
        ],
        "sensitivity_arrays": [
            "weighted_camera_z2_sum",
            "camera_z_variance",
        ],
    }


def packet_tensor_to_named_arrays(
    packet: torch.Tensor | np.ndarray,
    opacity_epsilon: float,
    variance_clamp_tolerance: float = 1e-6,
) -> dict[str, np.ndarray]:
    """Convert the 6-plane rasterizer packet into named arrays for NPZ output."""
    if isinstance(packet, torch.Tensor):
        arr = packet.detach().cpu().numpy()
    else:
        arr = np.asarray(packet)
    if arr.ndim != 3 or arr.shape[0] != len(PACKET_CHANNELS):
        raise ValueError(f"expected packet shape (6,H,W), got {arr.shape}")

    out = {name: arr[idx].astype(np.float32, copy=True) for name, idx in PACKET_CHANNELS.items()}
    variance = out["camera_z_variance"]
    small_negative = (variance < 0.0) & (variance >= -float(variance_clamp_tolerance))
    variance[small_negative] = 0.0
    numeric = (
        np.isfinite(out["accumulated_opacity"])
        & np.isfinite(out["weighted_camera_z_sum"])
        & np.isfinite(out["expected_camera_z"])
        & np.isfinite(variance)
        & (out["accumulated_opacity"] > float(opacity_epsilon))
        & (out["expected_camera_z"] > 0.0)
        & (variance >= 0.0)
        & (out["numeric_valid"] > 0.5)
    )
    out["expected_camera_z"][~numeric] = np.nan
    out["camera_z_variance"][~numeric] = np.nan
    out["numeric_valid"] = numeric.astype(np.uint8)
    return out


def assert_output_dir_is_safe(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)


def tensor_to_list(t: torch.Tensor | np.ndarray | Any) -> Any:
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy().tolist()
    if isinstance(t, np.ndarray):
        return t.tolist()
    return json_safe(t)


def _compare_fingerprint_values(path: str, expected: Any, actual: Any, mismatches: list[dict[str, Any]], atol: float) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            mismatches.append({"path": path, "reason": "type_mismatch", "expected_type": "dict", "actual_type": type(actual).__name__})
            return
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            mismatches.append(
                {
                    "path": path,
                    "reason": "key_mismatch",
                    "expected_only": sorted(expected_keys - actual_keys),
                    "actual_only": sorted(actual_keys - expected_keys),
                }
            )
            return
        for key in sorted(expected_keys):
            _compare_fingerprint_values(f"{path}.{key}" if path else str(key), expected[key], actual[key], mismatches, atol)
        return
    if isinstance(expected, list):
        if not isinstance(actual, list):
            mismatches.append({"path": path, "reason": "type_mismatch", "expected_type": "list", "actual_type": type(actual).__name__})
            return
        exp_arr = np.asarray(expected)
        act_arr = np.asarray(actual)
        if exp_arr.shape != act_arr.shape:
            mismatches.append(
                {
                    "path": path,
                    "reason": "shape_mismatch",
                    "expected_shape": list(exp_arr.shape),
                    "actual_shape": list(act_arr.shape),
                }
            )
            return
        if exp_arr.dtype.kind in "fiu" and act_arr.dtype.kind in "fiu":
            if not np.allclose(exp_arr.astype(float), act_arr.astype(float), rtol=0.0, atol=atol, equal_nan=True):
                diff = np.abs(exp_arr.astype(float) - act_arr.astype(float))
                mismatches.append(
                    {
                        "path": path,
                        "reason": "float_array_mismatch",
                        "max_abs_delta": float(np.nanmax(diff)) if diff.size else 0.0,
                        "atol": float(atol),
                    }
                )
            return
        if expected != actual:
            mismatches.append({"path": path, "reason": "list_value_mismatch"})
        return
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            exp_f = float(expected)
            act_f = float(actual)
        except Exception:
            mismatches.append({"path": path, "reason": "type_mismatch"})
            return
        if not np.isclose(exp_f, act_f, rtol=0.0, atol=atol, equal_nan=True):
            mismatches.append(
                {
                    "path": path,
                    "reason": "float_scalar_mismatch",
                    "expected": exp_f,
                    "actual": act_f,
                    "abs_delta": abs(exp_f - act_f),
                    "atol": float(atol),
                }
            )
        return
    if expected != actual:
        mismatches.append({"path": path, "reason": "exact_value_mismatch", "expected": expected, "actual": actual})


def camera_fingerprint(camera: Any) -> dict[str, Any]:
    payload = {
        "image_name": str(getattr(camera, "image_name", "")),
        "colmap_id": int(getattr(camera, "colmap_id", -1)),
        "uid": int(getattr(camera, "uid", -1)),
        "image_width": int(getattr(camera, "image_width", -1)),
        "image_height": int(getattr(camera, "image_height", -1)),
        "FoVx": float(getattr(camera, "FoVx", math.nan)),
        "FoVy": float(getattr(camera, "FoVy", math.nan)),
        "znear": float(getattr(camera, "znear", math.nan)),
        "zfar": float(getattr(camera, "zfar", math.nan)),
        "R": tensor_to_list(getattr(camera, "R", [])),
        "T": tensor_to_list(getattr(camera, "T", [])),
        "world_view_transform": tensor_to_list(getattr(camera, "world_view_transform", [])),
        "projection_matrix": tensor_to_list(getattr(camera, "projection_matrix", [])),
        "full_proj_transform": tensor_to_list(getattr(camera, "full_proj_transform", [])),
    }
    return {
        "fingerprint_sha256": canonical_json_sha256(payload),
        "payload": payload,
    }


def verify_expected_camera_fingerprint(actual: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    expected_path = Path(args.expected_camera_fingerprint)
    expected_file_sha256 = verify_file_sha256(
        expected_path,
        args.expected_camera_fingerprint_sha256,
        "expected camera fingerprint JSON",
    )
    expected = read_json(expected_path)
    if not isinstance(expected, dict) or not isinstance(expected.get("payload"), dict):
        raise ValueError(f"expected camera fingerprint must contain a payload object: {expected_path}")
    expected_payload_hash = canonical_json_sha256(expected["payload"])
    expected_recorded_hash = str(expected.get("fingerprint_sha256", ""))
    if expected_recorded_hash and expected_recorded_hash != expected_payload_hash:
        raise ValueError(
            f"expected camera fingerprint self-hash mismatch: recorded {expected_recorded_hash}, payload {expected_payload_hash}"
        )
    mismatches: list[dict[str, Any]] = []
    _compare_fingerprint_values(
        "",
        expected["payload"],
        actual["payload"],
        mismatches,
        float(args.camera_fingerprint_atol),
    )
    comparison = {
        "expected_path": str(expected_path),
        "expected_file_sha256": expected_file_sha256,
        "expected_payload_sha256": expected_payload_hash,
        "actual_payload_sha256": actual["fingerprint_sha256"],
        "rtol": 0.0,
        "atol": float(args.camera_fingerprint_atol),
        "matched": not mismatches,
        "mismatches": mismatches,
        "expected": expected,
        "actual": actual,
    }
    if mismatches:
        raise ValueError(f"target camera fingerprint mismatch: {mismatches[:5]}")
    return comparison


def load_target_camera(args: argparse.Namespace):
    from scene.dataset_readers import sceneLoadTypeCallbacks
    from utils.camera_utils import cameraList_from_camInfos

    source_path = Path(args.source_path)
    if not (source_path / "sparse").exists():
        raise FileNotFoundError(f"COLMAP source_path must contain sparse/: {source_path}")

    scene_info = sceneLoadTypeCallbacks["Colmap"](
        str(source_path),
        args.images,
        "",
        True,
        bool(args.train_test_exp),
    )
    all_infos = list(scene_info.train_cameras) + list(scene_info.test_cameras)
    matches = [cam for cam in all_infos if target_matches(cam.image_name, args.target) or target_matches(cam.image_path, args.target)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one loaded target camera named {args.target!r}, found {len(matches)}")

    loader_args = SimpleNamespace(
        resolution=int(args.resolution),
        data_device="cuda",
        train_test_exp=bool(args.train_test_exp),
        white_background=bool(args.white_background),
        source_path=str(source_path),
    )
    cameras = cameraList_from_camInfos(matches, 1.0, loader_args, scene_info.is_nerf_synthetic, True)
    return cameras[0]


def source_provenance(repo_root: Path) -> dict[str, Any]:
    submodule = repo_root / "submodules" / "diff-gaussian-rasterization"
    return {
        "parent_repo_commit": git_commit(repo_root),
        "parent_repo_status_short": git_text(repo_root, ["status", "--short"]),
        "rasterizer_submodule_commit": git_commit(submodule),
        "rasterizer_submodule_status_short": git_text(submodule, ["status", "--short"]),
        "working_tree_overlay": True,
    }


def verify_imported_rasterizer_extension(args: argparse.Namespace) -> dict[str, Any]:
    from diff_gaussian_rasterization import _C

    actual_path = Path(_C.__file__).resolve()
    expected_path = Path(args.rasterizer_extension_path).resolve()
    if actual_path != expected_path:
        raise ValueError(f"imported rasterizer extension path mismatch: expected {expected_path}, got {actual_path}")
    actual_sha256 = verify_file_sha256(actual_path, args.rasterizer_extension_sha256, "rasterizer extension binary")
    return {
        "imported_extension_path": str(actual_path),
        "imported_extension_sha256": actual_sha256,
    }


def export_expected_camera_z_packet(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    preflight = preflight_before_cuda(args, repo_root)

    # Import CUDA-dependent modules only after all manifest/hash preflight checks.
    from gaussian_renderer import render
    from utils.joint_multispectral_utils import load_rgb_checkpoint_geometry_only

    rasterizer_extension = verify_imported_rasterizer_extension(args)

    output = Path(args.output)
    assert_output_dir_is_safe(output)

    gaussians, _, ckpt_meta = load_rgb_checkpoint_geometry_only(
        Path(args.checkpoint),
        sh_degree=int(args.sh_degree),
        device="cuda",
    )
    target_camera = load_target_camera(args)
    target_fingerprint = camera_fingerprint(target_camera)
    target_fingerprint_comparison = verify_expected_camera_fingerprint(target_fingerprint, args)
    background = torch.tensor(
        [1, 1, 1] if args.white_background else [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )
    pipeline = SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=bool(args.debug),
        antialiasing=bool(args.antialiasing),
    )

    with torch.no_grad():
        disabled_rgb = None
        if bool(args.write_rgb_compatibility_arrays):
            disabled_pkg = render(
                target_camera,
                gaussians,
                pipeline,
                background,
                use_trained_exp=bool(args.train_test_exp),
                separate_sh=False,
                return_expected_camera_z_packet=False,
                opacity_epsilon=float(args.opacity_epsilon),
                variance_clamp_tolerance=float(args.variance_clamp_tolerance),
            )
            disabled_rgb = disabled_pkg["render"].detach().cpu().numpy().astype(np.float32)
        render_pkg = render(
            target_camera,
            gaussians,
            pipeline,
            background,
            use_trained_exp=bool(args.train_test_exp),
            separate_sh=False,
            return_expected_camera_z_packet=True,
            opacity_epsilon=float(args.opacity_epsilon),
            variance_clamp_tolerance=float(args.variance_clamp_tolerance),
        )
    arrays = packet_tensor_to_named_arrays(
        render_pkg["expected_camera_z_packet"],
        opacity_epsilon=float(args.opacity_epsilon),
        variance_clamp_tolerance=float(args.variance_clamp_tolerance),
    )

    npz_path = output / f"{Path(args.target).stem}_{args.run_label}_expected_camera_z_packet.npz"
    np.savez_compressed(npz_path, **arrays)
    npz_sha256 = sha256_file(npz_path)
    rgb_compatibility: dict[str, Any] = {"enabled": bool(args.write_rgb_compatibility_arrays)}
    if disabled_rgb is not None:
        enabled_rgb = render_pkg["render"].detach().cpu().numpy().astype(np.float32)
        rgb_disabled_path = output / f"{Path(args.target).stem}_{args.run_label}_rgb_disabled.npy"
        rgb_enabled_path = output / f"{Path(args.target).stem}_{args.run_label}_rgb_enabled.npy"
        np.save(rgb_disabled_path, disabled_rgb)
        np.save(rgb_enabled_path, enabled_rgb)
        max_abs_delta = float(np.max(np.abs(enabled_rgb - disabled_rgb)))
        rgb_compatibility = {
            "enabled": True,
            "rgb_disabled_path": str(rgb_disabled_path),
            "rgb_disabled_sha256": sha256_file(rgb_disabled_path),
            "rgb_enabled_path": str(rgb_enabled_path),
            "rgb_enabled_sha256": sha256_file(rgb_enabled_path),
            "max_abs_rgb_delta": max_abs_delta,
            "pass_threshold": DEFAULT_A0_TOLERANCES["rgb_compatibility_max_abs_delta"],
            "passed": bool(max_abs_delta <= DEFAULT_A0_TOLERANCES["rgb_compatibility_max_abs_delta"]),
        }

    metadata = {
        "schema": packet_schema(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scene": args.scene,
        "target": args.target,
        "run_label": args.run_label,
        "raster_height": int(arrays["expected_camera_z"].shape[0]),
        "raster_width": int(arrays["expected_camera_z"].shape[1]),
        "checkpoint_path": str(Path(args.checkpoint)),
        "checkpoint_sha256": preflight["model_info"]["checkpoint_sha256"],
        "checkpoint_meta": json_safe(ckpt_meta),
        "model_path": str(Path(args.model_path)),
        "source_path": str(Path(args.source_path)),
        "images_subdirectory": str(args.images),
        "resolution": int(args.resolution),
        "target_camera_fingerprint": target_fingerprint,
        "target_camera_fingerprint_comparison": target_fingerprint_comparison,
        "preflight": preflight,
        "source_provenance": source_provenance(repo_root),
        "rasterizer_extension": rasterizer_extension,
        "world_to_camera_convention": "renderer world_view_transform; camera-z follows existing rasterizer positive-z convention",
        "z_sign_convention": "expected_camera_z must be positive for numeric_valid",
        "background": "white" if args.white_background else "black",
        "sh_degree": int(args.sh_degree),
        "renderer_flags": {
            "convert_SHs_python": False,
            "compute_cov3D_python": False,
            "antialiasing": bool(args.antialiasing),
            "train_test_exp": bool(args.train_test_exp),
        },
        "opacity_epsilon": float(args.opacity_epsilon),
        "variance_clamp_tolerance": float(args.variance_clamp_tolerance),
        "a0_tolerances": DEFAULT_A0_TOLERANCES,
        "rgb_compatibility": rgb_compatibility,
        "exact_command": " ".join(os.sys.argv),
        "python_executable": os.sys.executable,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "npz_path": str(npz_path),
        "npz_sha256": npz_sha256,
    }
    (output / "expected_camera_z_packet_manifest.json").write_text(
        json.dumps(json_safe(metadata), indent=2),
        encoding="utf-8",
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a UMGS expected-camera-z packet for one explicit target.")
    parser.add_argument("--checkpoint", required=True, help="Exact checkpoint path; no latest-checkpoint discovery.")
    parser.add_argument("--checkpoint-sha256", required=True, help="Expected SHA256 for --checkpoint.")
    parser.add_argument("--source-path", required=True, help="Prepared COLMAP scene root containing sparse/ and images/.")
    parser.add_argument("--model-path", required=True, help="Exact model/run path; checkpoint must be under this path.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--run-label", required=True, help="Explicit run label, e.g. a0_repeat_1.")
    parser.add_argument("--output", required=True, help="New isolated output directory; must be empty if it exists.")
    parser.add_argument("--camera-manifest", required=True)
    parser.add_argument("--camera-manifest-sha256", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--runtime-source-hash-manifest", required=True)
    parser.add_argument("--runtime-source-hash-manifest-sha256", required=True)
    parser.add_argument("--rasterizer-extension-path", required=True)
    parser.add_argument("--rasterizer-extension-sha256", required=True)
    parser.add_argument("--train-file-sha256", required=True)
    parser.add_argument("--test-file-sha256", required=True)
    parser.add_argument("--colmap-sparse-hash-manifest", required=True)
    parser.add_argument("--colmap-sparse-hash-manifest-sha256", required=True)
    parser.add_argument("--expected-camera-fingerprint", required=True)
    parser.add_argument("--expected-camera-fingerprint-sha256", required=True)
    parser.add_argument("--camera-fingerprint-atol", type=float, default=1e-8)
    parser.add_argument("--images", default="images")
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--opacity-epsilon", type=float, default=1e-6)
    parser.add_argument("--variance-clamp-tolerance", type=float, default=1e-6)
    parser.add_argument("--white-background", action="store_true")
    parser.add_argument("--train-test-exp", action="store_true")
    parser.add_argument("--antialiasing", action="store_true")
    parser.add_argument("--write-rgb-compatibility-arrays", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    export_expected_camera_z_packet(args)


if __name__ == "__main__":
    main()
