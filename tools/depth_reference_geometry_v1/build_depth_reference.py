from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from depth_reference_common import (
    build_probe_view_manifest,
    compute_inside_bbox_mask,
    compute_quantile_bbox,
    load_json,
    load_ply_mesh,
    load_ply_points_xyz,
    parse_thresholds_m,
    relative_or_abs,
    render_mesh_depth_for_view,
    render_support_count_for_view,
    run_colmap,
    save_json,
)


def _resolve_executable(exe: str) -> str:
    exe_expanded = os.path.expandvars(exe)
    if os.path.isabs(exe_expanded) and os.path.exists(exe_expanded):
        return exe_expanded

    candidates = [exe_expanded]
    if os.name == "nt":
        base = exe_expanded
        if not base.lower().endswith((".exe", ".cmd", ".bat")):
            candidates = [base, base + ".exe", base + ".cmd", base + ".bat"]

    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return exe_expanded


def _should_use_shell(resolved_exe: str) -> bool:
    if os.name != "nt":
        return False
    low = resolved_exe.lower()
    return low.endswith(".bat") or low.endswith(".cmd")


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


def _gpu_attempt_env() -> Dict[str, str] | None:
    if os.name == "nt":
        return None
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return None
    if os.environ.get("QT_QPA_PLATFORM"):
        return None
    return {"QT_QPA_PLATFORM": "offscreen"}


def _run_cmd_capture_output(cmd_list: List[str], cwd: Path | None = None, extra_env: Dict[str, str] | None = None) -> str:
    if not cmd_list:
        raise ValueError("Empty command list")
    resolved0 = _resolve_executable(cmd_list[0])
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    if _should_use_shell(resolved0):
        cmd_str = " ".join([f'"{x}"' if (" " in str(x)) else str(x) for x in [resolved0] + cmd_list[1:]])
        print("Running (shell): " + cmd_str, flush=True)
        completed = subprocess.run(
            cmd_str,
            cwd=str(cwd) if cwd is not None else None,
            check=False,
            shell=True,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
    else:
        cmd = [resolved0] + [str(x) for x in cmd_list[1:]]
        print("Running: " + " ".join(cmd), flush=True)
        completed = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            check=False,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
    output = completed.stdout or ""
    if output:
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, cmd_list, output=output)
    return output


def _looks_like_gpu_failure(output: str) -> bool:
    text = str(output).lower()
    markers = (
        "cuda",
        "gpu",
        "out of memory",
        "not enough gpu memory",
        "check failed: context_.create()",
        "no cuda-capable device",
        "all cuda-capable devices are busy",
    )
    return any(marker in text for marker in markers)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build training-only reference depth artifacts for held-out geometry evaluation")
    parser.add_argument("--strict_protocol_manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--colmap_cmd",
        default=default_colmap_executable(),
        help=(
            "COLMAP executable. Defaults to SIGS_COLMAP_EXECUTABLE/COLMAP_EXECUTABLE, "
            "then ~/opt/colmap-cuda/bin/colmap, then PATH colmap."
        ),
    )
    parser.add_argument("--resolution_arg", type=int, default=4)
    parser.add_argument("--thresholds_m", default="0.10,0.25,0.50,1.00,2.00,5.00,10.00,20.00,30.00")
    parser.add_argument(
        "--distance_unit",
        default="meters",
        choices=("meters", "scene_units"),
        help="Unit carried by rendered depths and thresholds. Use scene_units for non-metric COLMAP reconstructions.",
    )
    parser.add_argument(
        "--scale_mode",
        default="metric_verified",
        choices=("metric_verified", "scene_normalized"),
        help="Protocol scale interpretation recorded in the output manifest.",
    )
    parser.add_argument("--bbox_lower_quantile", type=float, default=0.01)
    parser.add_argument("--bbox_upper_quantile", type=float, default=0.99)
    parser.add_argument("--bbox_padding_ratio", type=float, default=0.02)
    parser.add_argument("--support_min_count", type=int, default=1)
    parser.add_argument("--support_radius_px", type=int, default=1)
    parser.add_argument("--support_depth_tolerance_m", type=float, default=0.10)
    parser.add_argument("--patch_match_max_image_size", type=int, default=2000)
    parser.add_argument(
        "--patch_match_gpu_index",
        default="0",
        help=(
            "COLMAP PatchMatchStereo.gpu_index used for the first attempt. "
            "Default 0 so a CUDA_VISIBLE_DEVICES-scoped launcher uses its selected GPU only."
        ),
    )
    parser.add_argument(
        "--patch_match_allow_cpu_fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "On PatchMatch GPU failure/OOM, retry once with CUDA_VISIBLE_DEVICES cleared "
            "using the same COLMAP executable. Some COLMAP builds do not support CPU dense stereo; "
            "in that case the retry fails explicitly."
        ),
    )
    parser.add_argument("--force_patch_match", action="store_true")
    parser.add_argument("--force_fusion", action="store_true")
    parser.add_argument("--force_mesh", action="store_true")
    parser.add_argument("--force_views", action="store_true")
    return parser


def _is_reparse_point(path: Path) -> bool:
    try:
        attrs = os.lstat(path).st_file_attributes
    except (AttributeError, FileNotFoundError):
        return False
    return bool(attrs & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _remove_tree_or_link(path: Path) -> None:
    if not path.exists():
        return
    if path.is_symlink():
        path.unlink()
        return
    if _is_reparse_point(path):
        completed = subprocess.run(["cmd", "/c", "rmdir", str(path)], check=False, capture_output=True, text=True)
        if completed.returncode != 0 and path.exists():
            raise RuntimeError(f"Failed to remove junction {path}: {completed.stdout}\n{completed.stderr}")
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _ensure_dir_junction(link_path: Path, target_path: Path) -> None:
    if link_path.exists():
        return
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        os.symlink(str(target_path), str(link_path), target_is_directory=True)
        return
    cmd = ["cmd", "/c", "mklink", "/J", str(link_path), str(target_path)]
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0 and (not link_path.exists()):
        raise RuntimeError(
            f"Failed to create junction {link_path} -> {target_path}: "
            f"{completed.stdout}\n{completed.stderr}"
        )


def _ensure_file_link_or_copy(link_path: Path, target_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        return
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        os.symlink(str(target_path), str(link_path))
        return
    shutil.copy2(target_path, link_path)


def _prepare_flat_colmap_workspace(source_workspace_root: Path, prepared_root: Path) -> Path:
    images_src = source_workspace_root / "images"
    input_src = source_workspace_root / "input"
    distorted_src = source_workspace_root / "distorted"
    stereo_src = source_workspace_root / "stereo"
    sparse_src = source_workspace_root / "sparse" / "0"
    if not images_src.exists() or not stereo_src.exists() or not sparse_src.exists():
        raise FileNotFoundError(
            "Expected source workspace to contain images/, stereo/, and sparse/0; "
            f"got images={images_src.exists()} stereo={stereo_src.exists()} sparse0={sparse_src.exists()}"
        )
    prepared_root.mkdir(parents=True, exist_ok=True)
    _ensure_dir_junction(prepared_root / "images", images_src)
    if input_src.exists():
        _ensure_dir_junction(prepared_root / "input", input_src)
    if distorted_src.exists():
        _ensure_dir_junction(prepared_root / "distorted", distorted_src)
    stereo_dst = prepared_root / "stereo"
    if stereo_dst.exists() and _is_reparse_point(stereo_dst):
        _remove_tree_or_link(stereo_dst)
    stereo_dst.mkdir(parents=True, exist_ok=True)
    for cfg_name in ("patch-match.cfg", "fusion.cfg"):
        cfg_src = stereo_src / cfg_name
        if cfg_src.exists():
            shutil.copy2(cfg_src, stereo_dst / cfg_name)
    (stereo_dst / "depth_maps").mkdir(parents=True, exist_ok=True)
    (stereo_dst / "normal_maps").mkdir(parents=True, exist_ok=True)
    (stereo_dst / "consistency_graphs").mkdir(parents=True, exist_ok=True)
    sparse_dst = prepared_root / "sparse"
    sparse_dst.mkdir(parents=True, exist_ok=True)
    for src_file in sorted(sparse_src.iterdir()):
        if not src_file.is_file():
            continue
        dst_file = sparse_dst / src_file.name
        if not dst_file.exists():
            shutil.copy2(src_file, dst_file)
    return prepared_root


def _has_dense_outputs(prepared_workspace: Path) -> bool:
    depth_maps = prepared_workspace / "stereo" / "depth_maps"
    if not depth_maps.exists():
        return False
    return any(depth_maps.rglob("*.bin"))


def _run_patch_match_stereo(args: argparse.Namespace, prepared_workspace: Path) -> Dict[str, Any]:
    base_cmd = [
        str(args.colmap_cmd),
        "patch_match_stereo",
        "--workspace_path",
        str(prepared_workspace),
        "--workspace_format",
        "COLMAP",
        "--PatchMatchStereo.max_image_size",
        str(int(args.patch_match_max_image_size)),
        "--PatchMatchStereo.geom_consistency",
        "true",
    ]
    gpu_index = str(args.patch_match_gpu_index).strip()
    gpu_cmd = base_cmd + ["--PatchMatchStereo.gpu_index", gpu_index]
    audit: Dict[str, Any] = {
        "colmap_cmd": str(args.colmap_cmd),
        "patch_match_gpu_index": gpu_index,
        "patch_match_gpu_first": True,
        "patch_match_cpu_fallback_enabled": bool(args.patch_match_allow_cpu_fallback),
        "patch_match_attempts": [],
    }
    try:
        _run_cmd_capture_output(gpu_cmd, cwd=prepared_workspace, extra_env=_gpu_attempt_env())
        audit["patch_match_attempts"].append({"mode": "gpu", "status": "success", "gpu_index": gpu_index})
        audit["patch_match_selected_mode"] = "gpu"
        return audit
    except (subprocess.CalledProcessError, RuntimeError) as exc:
        output = getattr(exc, "output", "") or str(exc)
        audit["patch_match_attempts"].append(
            {
                "mode": "gpu",
                "status": "failed",
                "gpu_index": gpu_index,
                "looks_like_gpu_failure": _looks_like_gpu_failure(output),
                "error": str(exc),
            }
        )
        if not bool(args.patch_match_allow_cpu_fallback):
            raise

    print(
        "WARNING: PatchMatch GPU attempt failed; retrying with CUDA_VISIBLE_DEVICES cleared "
        "using the same COLMAP executable. If this COLMAP build requires CUDA for dense stereo, "
        "the CPU retry will fail explicitly.",
        flush=True,
    )
    cpu_env = {"CUDA_VISIBLE_DEVICES": ""}
    try:
        _run_cmd_capture_output(base_cmd, cwd=prepared_workspace, extra_env=cpu_env)
        audit["patch_match_attempts"].append({"mode": "cpu_fallback", "status": "success"})
        audit["patch_match_selected_mode"] = "cpu_fallback"
        return audit
    except (subprocess.CalledProcessError, RuntimeError) as exc:
        audit["patch_match_attempts"].append(
            {
                "mode": "cpu_fallback",
                "status": "failed",
                "error": str(exc),
            }
        )
        audit["patch_match_selected_mode"] = "failed"
        audit_path = prepared_workspace.parent / "patch_match_failure_audit.json"
        save_json(audit_path, audit)
        raise RuntimeError(
            "PatchMatch failed on GPU and CPU fallback also failed. "
            f"Audit saved to {audit_path}. Use a CUDA-enabled COLMAP and a free GPU, "
            "or reduce --patch_match_max_image_size."
        ) from exc


def main() -> None:
    args = _build_argparser().parse_args()
    strict_manifest_path = Path(args.strict_protocol_manifest).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    strict = load_json(strict_manifest_path)
    scene_name = str(strict["scene_name"])
    artifacts = strict["artifacts"]
    lists = strict["lists"]
    workspace_root = Path(artifacts["train_union_source_root"]).resolve()
    prepared_workspace = _prepare_flat_colmap_workspace(workspace_root, out_dir / "_colmap_workspace_flat")
    strict_thermal_root = Path(artifacts["strict_thermal_root"]).resolve()
    train_union_list = Path(lists["train_union"]).resolve()
    probe_list = Path(lists["probe_test"]).resolve()

    stereo_root = workspace_root / "stereo"
    if not stereo_root.exists():
        raise FileNotFoundError(f"Expected COLMAP stereo workspace at {stereo_root}")

    fused_ply = out_dir / "reference_fused_geometric.ply"
    delaunay_mesh = out_dir / "reference_mesh_delaunay.ply"
    poisson_mesh = out_dir / "reference_mesh_poisson.ply"
    mesh_path = delaunay_mesh
    mesh_backend = "delaunay_mesher"
    reference_runtime_audit: Dict[str, Any] = {
        "colmap_cmd": str(args.colmap_cmd),
        "colmap_cmd_resolved": _resolve_executable(str(args.colmap_cmd)),
        "patch_match_max_image_size": int(args.patch_match_max_image_size),
        "patch_match_gpu_index": str(args.patch_match_gpu_index),
        "patch_match_allow_cpu_fallback": bool(args.patch_match_allow_cpu_fallback),
    }

    if args.force_patch_match or (not _has_dense_outputs(prepared_workspace)):
        reference_runtime_audit.update(_run_patch_match_stereo(args, prepared_workspace))
    else:
        reference_runtime_audit["patch_match_selected_mode"] = "skipped_existing_dense_outputs"
        reference_runtime_audit["patch_match_attempts"] = []

    if args.force_fusion or (not fused_ply.exists()):
        run_colmap(
            args.colmap_cmd,
            [
                "stereo_fusion",
                "--workspace_path",
                str(prepared_workspace),
                "--workspace_format",
                "COLMAP",
                "--input_type",
                "geometric",
                "--output_path",
                str(fused_ply),
            ],
            cwd=prepared_workspace,
        )
    if (not fused_ply.exists()) or fused_ply.stat().st_size <= 0:
        raise RuntimeError(f"COLMAP stereo_fusion did not produce a valid fused point cloud at {fused_ply}")
    _ensure_file_link_or_copy(prepared_workspace / "fused.ply", fused_ply)
    fused_vis = Path(str(fused_ply) + ".vis")
    if fused_vis.exists():
        _ensure_file_link_or_copy(prepared_workspace / "fused.ply.vis", fused_vis)

    if args.force_mesh or (not delaunay_mesh.exists() and not poisson_mesh.exists()):
        try:
            run_colmap(
                args.colmap_cmd,
                [
                    "delaunay_mesher",
                    "--input_path",
                    str(prepared_workspace),
                    "--input_type",
                    "dense",
                    "--output_path",
                    str(delaunay_mesh),
                ],
                cwd=prepared_workspace,
            )
            mesh_path = delaunay_mesh
            mesh_backend = "delaunay_mesher"
        except Exception:
            run_colmap(
                args.colmap_cmd,
                [
                    "poisson_mesher",
                    "--input_path",
                    str(fused_ply),
                    "--output_path",
                    str(poisson_mesh),
                ],
                cwd=workspace_root,
            )
            mesh_path = poisson_mesh
            mesh_backend = "poisson_mesher"
    elif delaunay_mesh.exists():
        mesh_path = delaunay_mesh
        mesh_backend = "delaunay_mesher"
    elif poisson_mesh.exists():
        mesh_path = poisson_mesh
        mesh_backend = "poisson_mesher"
    else:
        raise FileNotFoundError("No reference mesh was produced")

    fused_points = load_ply_points_xyz(fused_ply)
    roi = compute_quantile_bbox(
        fused_points,
        lower_quantile=float(args.bbox_lower_quantile),
        upper_quantile=float(args.bbox_upper_quantile),
        padding_ratio_of_robust_diagonal=float(args.bbox_padding_ratio),
    )
    roi_path = out_dir / "reference_roi.json"
    save_json(
        roi_path,
        {
            "protocol_name": "reference-depth-based-geometric-evaluation-v1",
            "scene_name": scene_name,
            "roi_rule": {
                "type": "training_reference_dense_quantile_aabb",
                "lower_quantile": float(args.bbox_lower_quantile),
                "upper_quantile": float(args.bbox_upper_quantile),
                "padding_ratio_of_robust_diagonal": float(args.bbox_padding_ratio),
            },
            "bbox_min": roi["bbox_min"].tolist(),
            "bbox_max": roi["bbox_max"].tolist(),
            "scene_diagonal": float(roi["scene_diagonal"]),
            "source_points_path": str(fused_ply),
        },
    )

    camera_manifest_path = out_dir / "probe_camera_manifest.json"
    if args.force_views or (not camera_manifest_path.exists()):
        camera_manifest = build_probe_view_manifest(
            source_path=strict_thermal_root,
            images_dir_name="images",
            resolution_arg=int(args.resolution_arg),
            train_list=train_union_list,
            test_list=probe_list,
            scene_name=scene_name,
        )
        save_json(camera_manifest_path, camera_manifest)
    else:
        camera_manifest = load_json(camera_manifest_path)

    vertices_world, faces = load_ply_mesh(mesh_path)
    bbox_min = np.asarray(roi["bbox_min"], dtype=np.float64)
    bbox_max = np.asarray(roi["bbox_max"], dtype=np.float64)
    thresholds_m = parse_thresholds_m(args.thresholds_m)

    views_dir = out_dir / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    manifest_views: List[Dict[str, Any]] = []
    for view in camera_manifest["views"]:
        depth = render_mesh_depth_for_view(vertices_world, faces, view)
        support_count = render_support_count_for_view(
            fused_points,
            view,
            depth_tolerance_m=float(args.support_depth_tolerance_m),
            support_radius_px=int(args.support_radius_px),
        )
        finite = np.isfinite(depth) & (depth > 0.0)
        inside_roi = compute_inside_bbox_mask(depth, view, bbox_min=bbox_min, bbox_max=bbox_max) if np.any(finite) else np.zeros_like(finite, dtype=bool)
        valid_mask = finite & inside_roi & (support_count >= int(args.support_min_count))

        view_rel = Path("views") / f"{view['view_id']}.npz"
        view_path = out_dir / view_rel
        np.savez_compressed(
            view_path,
            depth=np.asarray(depth, dtype=np.float64),
            support_count=np.asarray(support_count, dtype=np.int32),
            valid_mask=np.asarray(valid_mask, dtype=np.uint8),
            inside_roi=np.asarray(inside_roi, dtype=np.uint8),
        )
        manifest_views.append(
            {
                "view_id": str(view["view_id"]),
                "image_name": str(view["image_name"]),
                "width": int(view["width"]),
                "height": int(view["height"]),
                "fx": float(view["fx"]),
                "fy": float(view["fy"]),
                "cx": float(view["cx"]),
                "cy": float(view["cy"]),
                "camera_to_world": view["camera_to_world"],
                "npz_file": str(view_rel).replace("\\", "/"),
            }
        )

    ref_manifest_path = out_dir / "reference_depth_manifest.json"
    save_json(
        ref_manifest_path,
        {
            "protocol_name": "reference-depth-based-geometric-evaluation-v1",
            "scene_name": scene_name,
            "strict_protocol_manifest": str(strict_manifest_path),
            "camera_manifest_path": str(camera_manifest_path),
            "reference_workspace_root": str(workspace_root),
            "reference_fused_ply": str(fused_ply),
            "reference_mesh_path": str(mesh_path),
            "reference_mesh_backend": mesh_backend,
            "roi_path": str(roi_path),
            "depth_semantics": "metric_camera_z_reference_mesh",
            "distance_unit": str(args.distance_unit),
            "scale_mode": str(args.scale_mode),
            "thresholds_m": thresholds_m,
            "support_rule": {
                "type": "training_dense_projected_support_count",
                "min_support_count": int(args.support_min_count),
                "support_radius_px": int(args.support_radius_px),
                "support_depth_tolerance_m": float(args.support_depth_tolerance_m),
            },
            "views": manifest_views,
        },
    )

    build_manifest_path = out_dir / "reference_build_manifest.json"
    save_json(
        build_manifest_path,
        {
            "scene_name": scene_name,
            "strict_protocol_manifest": str(strict_manifest_path),
            "source_workspace_root": str(workspace_root),
            "prepared_workspace_root": str(prepared_workspace),
            "strict_thermal_root": str(strict_thermal_root),
            "train_union_list": str(train_union_list),
            "probe_list": str(probe_list),
            "reference_fused_ply": str(fused_ply),
            "reference_mesh_path": str(mesh_path),
            "reference_mesh_backend": mesh_backend,
            "reference_depth_manifest": str(ref_manifest_path),
            "roi_path": str(roi_path),
            "camera_manifest_path": str(camera_manifest_path),
            "thresholds_m": thresholds_m,
            "distance_unit": str(args.distance_unit),
            "scale_mode": str(args.scale_mode),
            "runtime_audit": reference_runtime_audit,
            "support_rule": {
                "min_support_count": int(args.support_min_count),
                "support_radius_px": int(args.support_radius_px),
                "support_depth_tolerance_m": float(args.support_depth_tolerance_m),
            },
        },
    )
    print(f"REFERENCE_DEPTH_MANIFEST {ref_manifest_path}")
    print(f"REFERENCE_BUILD_MANIFEST {build_manifest_path}")


if __name__ == "__main__":
    main()
