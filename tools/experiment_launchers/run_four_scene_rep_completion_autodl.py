from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from plyfile import PlyData, PlyElement

PYTHON = Path("/root/autodl-tmp/envs/spectralindexgs_bw/bin/python")
REPO = Path("/root/autodl-tmp/Multispectral")
ROOT = Path("/root/autodl-tmp/runs/paper_autodl_full_20260429/four_scene_rep_completion_20260511")
LOGS = ROOT / "logs"
STATUS = ROOT / "status"
ABL = ROOT / "ablations"
DEPTH = ROOT / "depth_eval"
SUMMARY = ROOT / "summary"
for directory in [LOGS, STATUS, ABL, DEPTH, SUMMARY]:
    directory.mkdir(parents=True, exist_ok=True)

E3_RAW = Path("/root/autodl-tmp/runs/paper_autodl_full_20260429/e3_raw7_gpu_colmap_20260429_180200")
E3_OFF = Path("/root/autodl-tmp/runs/paper_autodl_full_20260429/e3_official7_20260430_142031")
E4 = Path("/root/autodl-tmp/runs/paper_autodl_full_20260429/e4b_zero_depth14_20260501_053151")
REF_ROOT = Path("/root/autodl-tmp/umgs_runs/depth_reference/meshref_10scene_repair_20260510")
MMS_OFF = Path("/root/autodl-tmp/runs/paper_autodl_full_20260429/mms_official7_repaired_20260502_100007/mms_official7")

BANDS = ["G", "R", "RE", "NIR"]
THRESHOLDS = "0.005,0.01,0.025,0.05,0.10,0.20"
SH_C0 = 0.28209479177387814
CHANNEL_TO_FEATURE_INDEX = {
    "D": 0,
    "D_R": 0,
    "D_G": 1,
    "D_B": 2,
    "G": 3,
    "MS_G": 3,
    "NIR": 4,
    "MS_NIR": 4,
    "R": 5,
    "MS_R": 5,
    "RE": 6,
    "MS_RE": 6,
}


@dataclass
class SceneSpec:
    name: str
    kind: str
    e3_root: Path
    ref_manifest: Path
    rgb_image_dir: Path


SCENES: Dict[str, SceneSpec] = {
    "raw001": SceneSpec(
        "raw001",
        "raw",
        E3_RAW / "raw001",
        REF_ROOT / "raw001/reference/reference_depth_manifest.json",
        E3_RAW / "raw001/prepared/RGB/images",
    ),
    "raw003": SceneSpec(
        "raw003",
        "raw",
        E3_RAW / "raw003",
        REF_ROOT / "raw003/reference/reference_depth_manifest.json",
        E3_RAW / "raw003/prepared/RGB/images",
    ),
    "ms_golf": SceneSpec(
        "ms_golf",
        "official",
        E3_OFF / "ms_golf",
        REF_ROOT / "ms_golf/reference/reference_depth_manifest.json",
        E3_OFF / "ms_golf/prepared/RGB/images",
    ),
}


def stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def event(message: str) -> None:
    line = f"[{stamp()}] {message}"
    print(line, flush=True)
    with (STATUS / "events.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run(cmd: List[str], log_path: Path, cwd: Path = REPO, env: Optional[dict] = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    event("RUN " + " ".join(map(str, cmd)) + f" > {log_path}")
    merged = os.environ.copy()
    merged.setdefault("CUDA_VISIBLE_DEVICES", "0")
    merged.setdefault("QT_QPA_PLATFORM", "offscreen")
    if env:
        merged.update(env)
    with log_path.open("w", encoding="utf-8", errors="ignore") as log:
        log.write("$ " + " ".join(map(str, cmd)) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), env=merged, stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}: {cmd}; log={log_path}")


def band_source(scene: SceneSpec, band: str) -> Path:
    if scene.kind == "raw":
        return scene.e3_root / "rectified" / f"{band}_rectified"
    return scene.e3_root / "prepared" / f"{band}_aligned"


def scene_train_args(scene: SceneSpec) -> List[str]:
    if scene.kind == "raw":
        return [
            "--resolution",
            "8",
            "--input_dynamic_range",
            "uint16",
            "--radiometric_mode",
            "exposure_normalized",
            "--rectification_config",
            str(scene.e3_root / "rectified/rectification_homographies.json"),
            "--rectified_root",
            str(scene.e3_root / "rectified"),
            "--require_rectified_band_scene",
            "true",
            "--use_validity_mask",
            "true",
            "--rectification_method",
            "minima_assisted_global_homography",
        ]
    return [
        "--resolution",
        "1",
        "--input_dynamic_range",
        "uint8",
        "--radiometric_mode",
        "raw_dn",
        "--require_rectified_band_scene",
        "false",
        "--use_validity_mask",
        "false",
        "--rectification_method",
        "none",
    ]


def model_done(model_dir: Path, iteration: int = 60000) -> bool:
    return (model_dir / f"point_cloud/iteration_{iteration}/point_cloud.ply").exists()


def resolve_model_iteration(model_dir: Path, preferred: int = 60000) -> Optional[int]:
    preferred_ply = model_dir / f"point_cloud/iteration_{preferred}/point_cloud.ply"
    if preferred_ply.exists():
        return preferred
    point_root = model_dir / "point_cloud"
    if not point_root.exists():
        return None
    iterations = []
    for child in point_root.glob("iteration_*"):
        if not child.is_dir():
            continue
        try:
            iteration = int(child.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if (child / "point_cloud.ply").exists():
            iterations.append(iteration)
    if not iterations:
        return None
    return max(iterations)


def capture_id_from_name(name: str) -> Optional[str]:
    match = re.search(r"(\d+)(?!.*\d)", Path(str(name)).stem)
    return match.group(1) if match else None


def native_cameras_for_depth(scene: SceneSpec, method_label: str, band_label: str, model_dir: Path) -> Path:
    native_cams = model_dir / "cameras.json"
    if scene.kind != "official" or band_label == "RGB":
        return native_cams
    if not native_cams.exists():
        return native_cams

    ref = json.loads(scene.ref_manifest.read_text(encoding="utf-8"))
    native = json.loads(native_cams.read_text(encoding="utf-8"))
    source_cams = native_cams
    if (
        method_label == "MMS_retained-self"
        and (not native or "position" not in native[0] or "rotation" not in native[0])
    ):
        source_band = "G" if band_label == "D" else band_label
        shim_cams = MMS_OFF / scene.name / "e3_compare_shim" / f"Model_{source_band}" / "cameras.json"
        if shim_cams.exists():
            native = json.loads(shim_cams.read_text(encoding="utf-8"))
            source_cams = shim_cams
            event(
                f"Using MMS official pose shim for depth adapter {scene.name} {band_label}: "
                f"{shim_cams}"
            )

    by_capture: Dict[str, dict] = {}
    for entry in native:
        capture_id = capture_id_from_name(str(entry.get("img_name", "")))
        if capture_id and capture_id not in by_capture:
            by_capture[capture_id] = entry

    aliased: List[dict] = []
    missing: List[str] = []
    for view in ref.get("views", []):
        ref_name = str(view.get("image_name", ""))
        capture_id = capture_id_from_name(ref_name)
        source = by_capture.get(capture_id or "")
        if source is None:
            missing.append(ref_name)
            continue
        item = dict(source)
        item["img_name"] = Path(ref_name).stem
        aliased.append(item)

    if len(aliased) < 3:
        event(
            f"OFFICIAL camera alias skipped {scene.name} {method_label} {band_label}: "
            f"only {len(aliased)} matched captures, missing={missing[:5]}"
        )
        return native_cams

    alias_dir = DEPTH / scene.name / "camera_aliases" / method_label / f"Model_{band_label}"
    alias_dir.mkdir(parents=True, exist_ok=True)
    alias_path = alias_dir / "cameras.json"
    audit_path = alias_dir / "camera_alias_audit.json"
    alias_path.write_text(json.dumps(aliased, indent=2) + "\n", encoding="utf-8")
    audit = {
        "scene": scene.name,
        "method": method_label,
        "band": band_label,
        "source_cameras_json": str(source_cams),
        "model_cameras_json": str(native_cams),
        "alias_cameras_json": str(alias_path),
        "reference_manifest": str(scene.ref_manifest),
        "matched_count": len(aliased),
        "missing_reference_views": missing,
        "purpose": "official aligned depth adapter camera-name alias by capture id",
        "created_at": stamp(),
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return alias_path


def train_ablation(scene: SceneSpec, method: str, band: str) -> Path:
    out = ABL / f"{method}_{scene.name}" / "out" / f"Model_{band}"
    if model_done(out):
        event(f"SKIP train {method} {scene.name} {band}: existing {out}")
        return out
    source = band_source(scene, band)
    if not source.exists():
        raise FileNotFoundError(source)
    cmd = [
        str(PYTHON),
        "train.py",
        "-s",
        str(source),
        "-m",
        str(out),
        "--eval",
        "--disable_viewer",
        "--quiet",
        "--iterations",
        "60000",
        "--save_iterations",
        "60000",
        "--checkpoint_iterations",
        "60000",
        "--test_iterations",
        "60000",
        "--modality_kind",
        "band",
        "--target_band",
        band,
        "--single_band_mode",
        "true",
        "--single_band_replicate_to_rgb",
        "true",
        "--tied_scalar_carrier",
        "true",
    ]
    cmd += scene_train_args(scene)
    if method == "fromscratch":
        cmd += ["--stage2_mode", "none"]
    elif method == "geom_unfrozen":
        ckpt = scene.e3_root / "out/Model_RGB/chkpnt30000.pth"
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        cmd += [
            "--start_checkpoint",
            str(ckpt),
            "--restore_geometry_only",
            "true",
            "--reset_appearance_features",
            "true",
            "--freeze_geometry",
            "false",
            "--freeze_opacity",
            "false",
            "--stage2_mode",
            "band_transfer",
        ]
    else:
        raise ValueError(method)
    run(cmd, LOGS / f"{method}_{scene.name}_train_{band}.log")
    if not model_done(out):
        raise RuntimeError(f"missing trained point cloud after training: {out}")
    return out


def render_model(scene: SceneSpec, method: str, band: str, model: Path, iteration: int = 60000) -> None:
    render_dir = model / f"test/ours_{iteration}/renders"
    if render_dir.exists() and any(render_dir.glob("*.png")):
        event(f"SKIP render {method} {scene.name} {band}")
        return
    run(
        [str(PYTHON), "render.py", "-m", str(model), "--iteration", str(iteration), "--skip_train", "--quiet"],
        LOGS / f"{method}_{scene.name}_render_{band}.log",
    )


def eval_quality(scene: SceneSpec, method: str, method_root: Path) -> None:
    marker = SUMMARY / f"{method}_{scene.name}_metrics_done.txt"
    if marker.exists():
        event(f"SKIP metrics {method} {scene.name}")
        return
    model_paths = [str(method_root / "out" / f"Model_{band}") for band in BANDS]
    run([str(PYTHON), "metrics.py", "-m", *model_paths, "--mask_mode", "gt_nonzero"], LOGS / f"{method}_{scene.name}_metrics.log")
    run(
        [
            str(PYTHON),
            "evaluate_spectral_indices.py",
            "--g_model_dir",
            model_paths[0],
            "--r_model_dir",
            model_paths[1],
            "--re_model_dir",
            model_paths[2],
            "--nir_model_dir",
            model_paths[3],
            "--iteration",
            "60000",
            "--indices",
            "NDVI,GNDVI,NDRE",
            "--mask_mode",
            "gt_nonzero_intersection",
            "--out_json",
            str(SUMMARY / f"{method}_{scene.name}_index_metrics.json"),
        ],
        LOGS / f"{method}_{scene.name}_index_eval.log",
    )
    marker.write_text(stamp() + "\n", encoding="utf-8")


def ply_xyz(path: Path) -> np.ndarray:
    ply = PlyData.read(str(path))
    vertex = ply["vertex"].data
    return np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)


def shared_support_audit(scene: SceneSpec, method: str, method_root: Path) -> None:
    audit_path = SUMMARY / f"{method}_{scene.name}_shared_support_audit.json"
    if audit_path.exists():
        return
    rows = {}
    ref_xyz = None
    ref_band = None
    for band in BANDS:
        ply = method_root / "out" / f"Model_{band}" / "point_cloud/iteration_60000/point_cloud.ply"
        if not ply.exists():
            rows[band] = {"exists": False}
            continue
        xyz = ply_xyz(ply)
        rows[band] = {"exists": True, "num_gaussians": int(xyz.shape[0])}
        if ref_xyz is None:
            ref_xyz = xyz
            ref_band = band
        else:
            same_shape = xyz.shape == ref_xyz.shape
            rows[band][f"same_shape_as_{ref_band}"] = bool(same_shape)
            if same_shape:
                rows[band][f"max_abs_xyz_delta_vs_{ref_band}"] = float(np.max(np.abs(xyz - ref_xyz)) if xyz.size else 0.0)
    audit = {"scene": scene.name, "method": method, "bands": rows, "created_at": stamp()}
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


def export_depth(scene: SceneSpec, method_label: str, band_label: str, model_dir: Path, iteration: int = 60000) -> Optional[Path]:
    if not scene.ref_manifest.exists():
        raise FileNotFoundError(scene.ref_manifest)
    if not model_dir.exists():
        event(f"MISSING model for depth {scene.name} {method_label} {band_label}: {model_dir}")
        return None
    resolved_iteration = resolve_model_iteration(model_dir, iteration)
    if resolved_iteration is None:
        event(f"MISSING point_cloud for depth {scene.name} {method_label} {band_label}: {model_dir}")
        return None
    if resolved_iteration != iteration:
        event(
            f"DEPTH iteration fallback {scene.name} {method_label} {band_label}: "
            f"requested {iteration}, using {resolved_iteration}"
        )
    bundle = DEPTH / scene.name / "bundles" / method_label / f"Model_{band_label}"
    manifest = bundle / "split_manifest.json"
    if not manifest.exists():
        native_cams = native_cameras_for_depth(scene, method_label, band_label, model_dir)
        if not native_cams.exists():
            event(f"MISSING cameras for depth {scene.name} {method_label} {band_label}: {native_cams}")
            return None
        run(
            [
                str(PYTHON),
                "tools/depth_reference_geometry_v1/export_gaussian_probe_bundle.py",
                "-m",
                str(model_dir),
                "--iteration",
                str(resolved_iteration),
                "--out_dir",
                str(bundle),
                "--split_label",
                f"{method_label}_{scene.name}_Model_{band_label}",
                "--depth_backend",
                "gaussian_point_splat",
                "--camera_frame_mode",
                "probe_manifest_native_align",
                "--world_alignment_mode",
                "similarity",
                "--probe_camera_manifest",
                str(scene.ref_manifest),
                "--native_cameras_json",
                str(native_cams),
                "--scene_name_override",
                scene.name,
                "--quiet",
            ],
            LOGS / f"depth_export_{scene.name}_{method_label}_{band_label}.log",
        )
    adapter_manifest = bundle / "adapter_manifest.json"
    if not adapter_manifest.exists():
        split_data = json.loads(manifest.read_text(encoding="utf-8"))
        point_cloud = model_dir / f"point_cloud/iteration_{resolved_iteration}/point_cloud.ply"
        adapter = {
            "protocol_name": "reference-depth-based-geometric-evaluation-v1",
            "method_name": f"{method_label}_{scene.name}_Model_{band_label}",
            "depth_semantics": split_data.get("depth_semantics", "metric_camera_z_from_point_splat_centers"),
            "validity_rule": {
                "mode": "opacity_threshold",
                "opacity_threshold": 0.5,
                "depth_min": 1e-6,
            },
            "notes": (
                "Generated by four-scene completion launcher for compatibility with "
                "evaluate_depth_reference.py. Depth bundle is exported from learned Gaussian "
                "geometry with gaussian_point_splat; no rendered PNG inverse-depth is used."
            ),
            "point_source": {
                "point_source": "3dgs_point_cloud_ply",
                "point_cloud_path": str(point_cloud),
                "opacity_source": "3dgs_ply_opacity_logit_sigmoid_or_support_count",
            },
            "split_manifest": str(manifest),
            "created_at": stamp(),
        }
        adapter_manifest.write_text(json.dumps(adapter, indent=2) + "\n", encoding="utf-8")
    out_eval = DEPTH / scene.name / "eval" / method_label / f"Model_{band_label}"
    if not (out_eval / "metrics_summary.json").exists():
        run(
            [
                str(PYTHON),
                "tools/depth_reference_geometry_v1/evaluate_depth_reference.py",
                "--reference_manifest",
                str(scene.ref_manifest),
                "--model_manifest",
                str(manifest),
                "--adapter_manifest",
                str(adapter_manifest),
                "--out_dir",
                str(out_eval),
                "--enable_agreement_metrics",
                "--error_mode",
                "relative_depth",
                "--thresholds",
                THRESHOLDS,
            ],
            LOGS / f"depth_eval_{scene.name}_{method_label}_{band_label}.log",
        )
    return bundle


def export_method_depths(scene: SceneSpec) -> None:
    roots = {
        "UMGS-I": scene.e3_root / "out",
        "UMGS-J": E4 / scene.name / "out",
        "Scratch-MS": ABL / f"fromscratch_{scene.name}" / "out",
        "Unfrozen-support": ABL / f"geom_unfrozen_{scene.name}" / "out",
    }
    for method, outroot in roots.items():
        bands = ["RGB"] + BANDS if method == "UMGS-I" else BANDS
        for band in bands:
            export_depth(scene, method, band, outroot / f"Model_{band}")


def sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float32)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def rgb_to_sh(rgb: np.ndarray) -> np.ndarray:
    return (rgb - 0.5) / SH_C0


def write_3dgs_ply(path: Path, means: np.ndarray, features_dc: np.ndarray, opacities: np.ndarray, scales: np.ndarray, quats: np.ndarray, channel_key: str) -> None:
    n = int(means.shape[0])
    idx = CHANNEL_TO_FEATURE_INDEX[channel_key]
    if channel_key == "D":
        rgb = sigmoid(features_dc[:, :3]).astype(np.float32)
    else:
        gray = sigmoid(features_dc[:, idx : idx + 1]).astype(np.float32)
        rgb = np.repeat(gray, 3, axis=1)
    f_dc = rgb_to_sh(rgb).astype(np.float32)
    f_rest = np.zeros((n, 45), dtype=np.float32)
    normals = np.zeros((n, 3), dtype=np.float32)
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4"), ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4")]
    dtype += [(f"f_rest_{i}", "f4") for i in range(45)]
    dtype += [("opacity", "f4"), ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"), ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4")]
    arr = np.empty(n, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = means[:, 0], means[:, 1], means[:, 2]
    arr["nx"], arr["ny"], arr["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    for i in range(45):
        arr[f"f_rest_{i}"] = f_rest[:, i]
    arr["opacity"] = opacities.reshape(-1)
    arr["scale_0"], arr["scale_1"], arr["scale_2"] = scales[:, 0], scales[:, 1], scales[:, 2]
    arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


def write_mms_failure(scene: SceneSpec, band: str, reason: str, root: str = "") -> None:
    fail_root = DEPTH / scene.name / "eval" / "MMS_retained-self" / (f"Model_{band}" if band != "all" else "scene")
    fail_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "MMS_retained-self",
        "scene": scene.name,
        "band": band,
        "status": "unavailable",
        "reason": reason,
        "root": root,
        "created_at": stamp(),
        "depth_semantics": "unavailable",
    }
    (fail_root / "mms_depth_adapter_failure.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    event(f"MMS failure recorded {scene.name} {band}: {reason}")


def convert_mms_official(scene: SceneSpec) -> bool:
    src_root = MMS_OFF / scene.name
    ckpt = src_root / f"train/{scene.name}_official7/mmsplat/autodl/nerfstudio_models/step-000029999.ckpt"
    adapter_repo = src_root / "adapter_repo"
    if not ckpt.exists():
        write_mms_failure(scene, "all", "missing_mms_official_checkpoint", str(ckpt))
        return False
    if not adapter_repo.exists():
        write_mms_failure(scene, "all", "missing_mms_adapter_repo", str(adapter_repo))
        return False
    converted_root = DEPTH / scene.name / "mms_converted_repo_models"
    done_marker = converted_root / "mms_conversion_audit.json"
    if not done_marker.exists():
        event(f"Converting MMS checkpoint for {scene.name}: {ckpt}")
        state = torch.load(str(ckpt), map_location="cpu", weights_only=False)["pipeline"]
        means = state["_model.gauss_params.means"].detach().cpu().numpy().astype(np.float32)
        features_dc = state["_model.gauss_params.features_dc"].detach().cpu().numpy().astype(np.float32)
        opacities = state["_model.gauss_params.opacities"].detach().cpu().numpy().astype(np.float32)
        scales = state["_model.gauss_params.scales"].detach().cpu().numpy().astype(np.float32)
        quats = state["_model.gauss_params.quats"].detach().cpu().numpy().astype(np.float32)
        for band in ["D"] + BANDS:
            model_dir = converted_root / f"Model_{band}"
            write_3dgs_ply(model_dir / "point_cloud/iteration_30000/point_cloud.ply", means, features_dc, opacities, scales, quats, band)
            src_model = adapter_repo / (f"Model_{band}" if band != "D" else "Model_G")
            if (src_model / "cameras.json").exists():
                shutil.copy2(src_model / "cameras.json", model_dir / "cameras.json")
            if (src_model / "cfg_args").exists():
                shutil.copy2(src_model / "cfg_args", model_dir / "cfg_args")
            else:
                (model_dir / "cfg_args").write_text(f"Namespace(source_path='{src_root}', use_validity_mask=False)\n", encoding="utf-8")
        audit = {
            "status": "converted",
            "scene": scene.name,
            "source_checkpoint": str(ckpt),
            "adapter_repo": str(adapter_repo),
            "num_points": int(means.shape[0]),
            "depth_semantics": "converted_mms_learned_gaussian_depth",
            "note": "MMS native checkpoint converted non-invasively to repo-compatible 3DGS PLY per spectral channel; cameras copied from official adapter_repo for depth-adapter smoke.",
        }
        done_marker.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    ok = True
    for band in BANDS:
        ok = export_depth(scene, "MMS_retained-self", band, converted_root / f"Model_{band}", iteration=30000) is not None and ok
    export_depth(scene, "MMS_retained-self", "D", converted_root / "Model_D", iteration=30000)
    return ok


def mms_smoke_or_failure(scene: SceneSpec) -> None:
    if scene.kind == "official":
        try:
            convert_mms_official(scene)
        except Exception as exc:
            write_mms_failure(scene, "all", "mms_official_conversion_or_depth_export_failed", repr(exc))
            event(f"MMS official adapter failed for {scene.name}: {exc!r}")
    else:
        raw_candidates = list(Path("/root/autodl-tmp/runs/paper_autodl_full_20260429").glob(f"**/*{scene.name}*/**/nerfstudio_models/step-*.ckpt"))
        raw_candidates = [path for path in raw_candidates if "mms" in str(path).lower()]
        if not raw_candidates:
            write_mms_failure(scene, "all", "no_scene_specific_mms_learned_gaussian_checkpoint_found_under_current_protocol")
            return
        write_mms_failure(scene, "all", "raw_mms_checkpoint_found_but_no_audited_raw_scene_depth_adapter_implemented", str(raw_candidates[0]))


def visualize_scene(scene: SceneSpec) -> None:
    out_dir = DEPTH / scene.name / "allband_visuals"
    if (out_dir / "depth_visual_manifest_all_bands.json").exists():
        return
    cmd = [
        str(PYTHON),
        "tools/depth_reference_geometry_v1/visualize_self_m3m_mesh_depth_grid_10views.py",
        "visualize_multiband",
        "--reference_manifest",
        str(scene.ref_manifest),
        "--rgb_image_dir",
        str(scene.rgb_image_dir),
        "--out_dir",
        str(out_dir),
        "--bands",
        "RGB,G,R,RE,NIR",
        "--shared_rgb_anchor_root",
        str(DEPTH / scene.name / "bundles" / "UMGS-I" / "Model_RGB"),
        "--mms_rgb_substitute_band",
        "D",
        "--wspace",
        "0.006",
        "--hspace",
        "0.025",
    ]
    for arg in [
        f"Scratch-MS={DEPTH / scene.name / 'bundles' / 'Scratch-MS'}",
        f"Unfrozen-support={DEPTH / scene.name / 'bundles' / 'Unfrozen-support'}",
        f"MMS retained-self={DEPTH / scene.name / 'bundles' / 'MMS_retained-self'}",
        f"UMGS-J={DEPTH / scene.name / 'bundles' / 'UMGS-J'}",
        f"UMGS-I={DEPTH / scene.name / 'bundles' / 'UMGS-I'}",
    ]:
        cmd += ["--method_root", arg]
    run(cmd, LOGS / f"visualize_{scene.name}_allbands.log")


def aggregate_depth() -> None:
    rows = []
    for scene in SCENES.values():
        eval_root = DEPTH / scene.name / "eval"
        if not eval_root.exists():
            continue
        for method_dir in eval_root.glob("*"):
            if not method_dir.is_dir():
                continue
            for model_dir in method_dir.glob("Model_*"):
                summary = model_dir / "metrics_summary.json"
                if not summary.exists():
                    continue
                data = json.loads(summary.read_text())
                row = {"scene": scene.name, "method": method_dir.name, "band": model_dir.name.replace("Model_", "")}
                secondary = data.get("secondary_metrics", {})
                row["AbsRelMean"] = (
                    secondary.get("AbsRelativeDepthError_Mean")
                    or secondary.get("AbsRelDepthError_Mean")
                    or secondary.get("AbsDepthError_Mean")
                )
                counts = data.get("counts", {})
                row["Valid"] = counts.get("model_valid_on_reference_pixels")
                row["ReferenceValid"] = counts.get("reference_valid_pixels")
                for metric in data.get("threshold_metrics", []):
                    threshold = metric.get("threshold")
                    row[f"Agree@{threshold}"] = metric.get("Agree") or metric.get("agreement_rate") or metric.get("Agreement")
                rows.append(row)
    if rows:
        keys = sorted({key for row in rows for key in row.keys()})
        out = SUMMARY / "four_scene_depth_metrics_aggregate.csv"
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        event(f"WROTE {out} rows={len(rows)}")


def main() -> None:
    event("four_scene_rep_completion started")
    manifest = {
        "run_root": str(ROOT),
        "created_at": stamp(),
        "scenes": {
            key: {
                "kind": value.kind,
                "e3_root": str(value.e3_root),
                "ref_manifest": str(value.ref_manifest),
                "rgb_image_dir": str(value.rgb_image_dir),
            }
            for key, value in SCENES.items()
        },
        "thresholds": THRESHOLDS,
        "methods": ["UMGS-I", "UMGS-J", "Scratch-MS", "Unfrozen-support", "MMS_retained-self"],
    }
    (ROOT / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # Smoke phase.
    for method in ["fromscratch", "geom_unfrozen"]:
        model = train_ablation(SCENES["raw001"], method, "G")
        render_model(SCENES["raw001"], method, "G", model)
    export_method_depths(SCENES["raw001"])
    mms_smoke_or_failure(SCENES["ms_golf"])
    mms_smoke_or_failure(SCENES["raw001"])

    # Formal completion.
    for scene in SCENES.values():
        for method in ["fromscratch", "geom_unfrozen"]:
            for band in BANDS:
                model = train_ablation(scene, method, band)
                render_model(scene, method, band, model)
            eval_quality(scene, method, ABL / f"{method}_{scene.name}")
            shared_support_audit(scene, method, ABL / f"{method}_{scene.name}")
        export_method_depths(scene)
        mms_smoke_or_failure(scene)
        try:
            visualize_scene(scene)
        except Exception as exc:
            event(f"VISUALIZE failed non-fatal for {scene.name}: {exc!r}")
    aggregate_depth()
    (STATUS / "finished.txt").write_text(stamp() + "\n", encoding="utf-8")
    event("four_scene_rep_completion finished")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        event(f"FATAL {exc!r}")
        (STATUS / "failed.txt").write_text(stamp() + "\n" + repr(exc) + "\n", encoding="utf-8")
        raise
