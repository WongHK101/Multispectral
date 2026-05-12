#!/usr/bin/env python3
"""Complete MMSplat retained raw-scene depth evaluation for raw001/raw003.

This is an external-baseline retained-subset comparison, not an internal
ablation. It preserves the MMS-default raw failure/success records and writes
new retained artifacts under an independent run root.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from plyfile import PlyData, PlyElement


REPO = Path("/root/autodl-tmp/Multispectral")
PY = Path("/root/autodl-tmp/envs/spectralindexgs_bw/bin/python")
MMS_ENV = Path("/root/autodl-tmp/envs/mmsplat_bw")
MMS_SRC = Path("/root/autodl-tmp/src/MS-Splatting")
COLMAP = Path("/root/autodl-tmp/opt/colmap-cuda-3.9.1/bin/colmap")
RUN_BASE = Path("/root/autodl-tmp/runs/paper_autodl_full_20260429")
RAW_ATTEMPT = RUN_BASE / "mms_raw_representative_attempt_20260512"
RUN = RUN_BASE / "mms_raw_retained_depth_20260512"
LOGS = RUN / "logs"
STATUS = RUN / "status"
DEPTH = RUN / "depth"
SUMMARY = RUN / "summary"
THRESHOLDS = "0.005,0.01,0.025,0.05,0.10,0.20"
SH_C0 = 0.28209479177387814
BANDS = ["G", "R", "RE", "NIR"]
CHANNEL_TO_FEATURE_INDEX = {"D": 0, "G": 3, "R": 4, "RE": 5, "NIR": 6}


@dataclass(frozen=True)
class Scene:
    name: str
    raw_root: Path
    e3_root: Path
    ref_manifest: Path
    source_attempt_root: Path


SCENES = [
    Scene(
        "raw001",
        Path("/root/autodl-tmp/datasets/Multispectral/20240528/DJI_202405281154_001_reynoldsTR01crossrtk"),
        RUN_BASE / "e3_raw7_gpu_colmap_20260429_180200/raw001",
        Path("/root/autodl-tmp/umgs_runs/depth_reference/meshref_10scene_repair_20260510/raw001/reference/reference_depth_manifest.json"),
        RAW_ATTEMPT / "raw001",
    ),
    Scene(
        "raw003",
        Path("/root/autodl-tmp/datasets/Multispectral/20240528/DJI_202405281220_003_reynoldsAb01crossrtk"),
        RUN_BASE / "e3_raw7_gpu_colmap_20260429_180200/raw003",
        Path("/root/autodl-tmp/umgs_runs/depth_reference/meshref_10scene_repair_20260510/raw003/reference/reference_depth_manifest.json"),
        RAW_ATTEMPT / "raw003",
    ),
]


def stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def event(message: str) -> None:
    STATUS.mkdir(parents=True, exist_ok=True)
    line = f"[{stamp()}] {message}"
    print(line, flush=True)
    with (STATUS / "events.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def run(cmd: list[object], log: Path, *, cwd: Optional[Path] = None, env: Optional[dict] = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    event("RUN " + " ".join(map(str, cmd)) + f" > {log}")
    with log.open("w", encoding="utf-8", errors="replace") as handle:
        handle.write("$ " + " ".join(map(str, cmd)) + "\n")
        handle.flush()
        proc = subprocess.run(
            [str(x) for x in cmd],
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}: {cmd}; log={log}")


def mms_env() -> dict:
    env = os.environ.copy()
    env["PATH"] = f"{MMS_ENV / 'bin'}:{env.get('PATH', '')}"
    env["PYTHONPATH"] = f"{MMS_SRC}:{env.get('PYTHONPATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    return env


def checkpoint_path(scene_root: Path, scene: Scene) -> Optional[Path]:
    ckpt_dir = scene_root / "train" / f"{scene.name}_mms_retained" / "mmsplat" / "autodl" / "nerfstudio_models"
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob("step-*.ckpt"))
    return ckpts[-1] if ckpts else None


def source_checkpoint(scene: Scene) -> Optional[Path]:
    # Reuse raw003's completed default-full run when available.
    default_dir = scene.source_attempt_root / "train" / f"{scene.name}_mms_raw" / "mmsplat" / "autodl" / "nerfstudio_models"
    ckpts = sorted(default_dir.glob("step-*.ckpt")) if default_dir.exists() else []
    return ckpts[-1] if ckpts else None


def ensure_preprocessed(scene: Scene) -> Path:
    if scene.name == "raw003":
        p = scene.source_attempt_root / "preprocessed"
        if (p / "transforms.json").exists():
            return p
    out = RUN / scene.name / "preprocessed_full_retained"
    if (out / "transforms.json").exists():
        return out
    raw_png = scene.source_attempt_root / "inputs" / "raw_png"
    if not (raw_png / "mmsplat_raw_input_audit.json").exists():
        raise FileNotFoundError(f"Missing prepared MMS raw input: {raw_png}")
    run(
        [
            MMS_ENV / "bin/python",
            "mmsplat/scripts/process_data.py",
            "images",
            "--data",
            raw_png,
            "--output-dir",
            out,
            "--num-downscales",
            "1",
            "--primary-channel",
            "D",
            "--colmap-cmd",
            COLMAP,
            "--method",
            "full",
        ],
        LOGS / scene.name / "process_data_full_retained.log",
        cwd=MMS_SRC,
        env=mms_env(),
    )
    if not (out / "transforms.json").exists():
        raise RuntimeError(f"Retained full process_data did not write transforms.json: {out}")
    return out


def ensure_split(scene: Scene, preprocessed: Path) -> Path:
    root = RUN / scene.name
    split_json = root / "inputs" / "train_split.llffhold8.json"
    split_audit = root / "inputs" / "train_split.llffhold8.audit.json"
    sanitized = root / "inputs" / "train_split.llffhold8.sanitized.json"
    sanitized_audit = root / "inputs" / "train_split.llffhold8.sanitized.audit.json"
    if sanitized.exists():
        return sanitized
    run(
        [
            PY,
            "build_mmsplat_raw_json_split.py",
            "--raw_root",
            scene.raw_root,
            "--eval_hold",
            "8",
            "--out_json",
            split_json,
            "--audit_json",
            split_audit,
            "--group_mode",
            "frame_only",
        ],
        LOGS / scene.name / "build_split.log",
        cwd=REPO,
    )
    run(
        [
            PY,
            "sanitize_mmsplat_json_list.py",
            "--src_json",
            split_json,
            "--transforms_json",
            preprocessed / "transforms.json",
            "--dst_json",
            sanitized,
            "--audit_json",
            sanitized_audit,
        ],
        LOGS / scene.name / "sanitize_split.log",
        cwd=REPO,
    )
    return sanitized


def ensure_train(scene: Scene, preprocessed: Path, split_json: Path) -> Path:
    root = RUN / scene.name
    done = root / "status" / "train_done.txt"
    ckpt = checkpoint_path(root, scene)
    if done.exists() and ckpt:
        return ckpt
    src_ckpt = source_checkpoint(scene)
    if src_ckpt and scene.name == "raw003":
        mirror = root / "train" / f"{scene.name}_mms_retained" / "mmsplat" / "autodl"
        if not mirror.exists():
            event(f"Reusing raw003 MMS checkpoint from {src_ckpt.parent.parent}")
            mirror.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src_ckpt.parent.parent, mirror, dirs_exist_ok=True)
        done.parent.mkdir(parents=True, exist_ok=True)
        done.write_text(stamp() + "\n", encoding="utf-8")
        ckpt = checkpoint_path(root, scene)
        if ckpt:
            return ckpt
    run(
        [
            MMS_ENV / "bin/ns-train",
            "mmsplat",
            "--output-dir",
            root / "train",
            "--experiment-name",
            f"{scene.name}_mms_retained",
            "--timestamp",
            "autodl",
            "--max-num-iterations",
            "30000",
            "--steps-per-save",
            "30000",
            "--vis",
            "tensorboard",
            "--data",
            preprocessed,
            "mmsplat-dataparser",
            "--eval-mode",
            "json-list",
            "--json-list-path",
            split_json,
            "--downscale-factor",
            "1",
        ],
        LOGS / scene.name / "train.log",
        cwd=MMS_SRC,
        env=mms_env(),
    )
    ckpt = checkpoint_path(root, scene)
    if not ckpt:
        raise RuntimeError(f"MMS retained checkpoint missing after train: {root}")
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(stamp() + "\n", encoding="utf-8")
    return ckpt


def ensure_eval(scene: Scene, preprocessed: Path, split_json: Path) -> None:
    root = RUN / scene.name
    config = root / "train" / f"{scene.name}_mms_retained" / "mmsplat" / "autodl" / "config.yml"
    out = root / "eval" / "eval.json"
    if out.exists():
        return
    if not config.exists():
        raise FileNotFoundError(config)
    run(
        [
            MMS_ENV / "bin/python",
            "mmsplat/scripts/eval.py",
            "--load-config",
            config,
            "--output-path",
            out,
            "--render-output-path",
            root / "eval" / "renders",
            "--load-step",
            "29999",
        ],
        LOGS / scene.name / "eval.log",
        cwd=MMS_SRC,
        env=mms_env(),
    )


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


def raw_d_names(scene: Scene) -> list[str]:
    names = sorted([p.name for p in scene.raw_root.glob("*_D.JPG")])
    if not names:
        names = sorted([p.name for p in scene.raw_root.glob("*_D.jpg")])
    if not names:
        raise RuntimeError(f"No raw D images under {scene.raw_root}")
    return names


def d_index_from_path(file_path: str) -> Optional[int]:
    m = re.search(r"D_(\d+)\.", str(file_path), flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def write_native_d_cameras(scene: Scene, preprocessed: Path, out_path: Path) -> Path:
    transforms = json.loads((preprocessed / "transforms.json").read_text(encoding="utf-8"))
    names = raw_d_names(scene)
    rows = []
    for frame in transforms.get("frames", []):
        if str(frame.get("mm_channel", "")) != "D":
            continue
        idx = d_index_from_path(str(frame.get("file_path", "")))
        if idx is None or idx < 1 or idx > len(names):
            continue
        c2w = np.asarray(frame["transform_matrix"], dtype=np.float64)
        rows.append(
            {
                "img_name": names[idx - 1],
                "position": c2w[:3, 3].tolist(),
                "rotation": c2w[:3, :3].tolist(),
                "width": int(frame.get("w", 0)),
                "height": int(frame.get("h", 0)),
                "fx": float(frame.get("fl_x", 0.0)),
                "fy": float(frame.get("fl_y", frame.get("fl_x", 0.0))),
                "cx": float(frame.get("cx", 0.0)),
                "cy": float(frame.get("cy", 0.0)),
                "mms_file_path": str(frame.get("file_path", "")),
            }
        )
    if len(rows) < 3:
        raise RuntimeError(f"Could not build enough native D cameras from {preprocessed}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return out_path


def ensure_converted_models(scene: Scene, preprocessed: Path, ckpt: Path) -> Path:
    out_root = RUN / scene.name / "converted_repo_models"
    audit_path = out_root / "mms_conversion_audit.json"
    if audit_path.exists():
        return out_root
    event(f"Converting MMS retained checkpoint for {scene.name}: {ckpt}")
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)["pipeline"]
    means = state["_model.gauss_params.means"].detach().cpu().numpy().astype(np.float32)
    features_dc = state["_model.gauss_params.features_dc"].detach().cpu().numpy().astype(np.float32)
    opacities = state["_model.gauss_params.opacities"].detach().cpu().numpy().astype(np.float32)
    scales = state["_model.gauss_params.scales"].detach().cpu().numpy().astype(np.float32)
    quats = state["_model.gauss_params.quats"].detach().cpu().numpy().astype(np.float32)
    native_cams = write_native_d_cameras(scene, preprocessed, out_root / "native_d_cameras_for_depth.json")
    cfg_source = scene.e3_root / "out/Model_G/cfg_args"
    cfg_rgb = scene.e3_root / "out/Model_RGB/cfg_args"
    for band in ["D"] + BANDS:
        model_dir = out_root / f"Model_{band}"
        write_3dgs_ply(model_dir / "point_cloud/iteration_30000/point_cloud.ply", means, features_dc, opacities, scales, quats, band)
        shutil.copy2(native_cams, model_dir / "cameras.json")
        cfg = cfg_rgb if band == "D" and cfg_rgb.exists() else cfg_source
        if cfg.exists():
            shutil.copy2(cfg, model_dir / "cfg_args")
        else:
            (model_dir / "cfg_args").write_text(f"Namespace(source_path='{scene.e3_root}', use_validity_mask=False)\n", encoding="utf-8")
    write_json(
        audit_path,
        {
            "status": "converted",
            "scene": scene.name,
            "source_checkpoint": str(ckpt),
            "preprocessed_transforms": str(preprocessed / "transforms.json"),
            "native_cameras_json": str(native_cams),
            "num_points": int(means.shape[0]),
            "depth_semantics": "converted_mms_retained_learned_gaussian_depth",
            "camera_semantics": "MMS native D-camera poses aliased back to original raw D image names for reference-depth alignment.",
            "created_at": stamp(),
        },
    )
    return out_root


def resolve_model_iteration(model_dir: Path, preferred: int) -> Optional[int]:
    if (model_dir / f"point_cloud/iteration_{preferred}/point_cloud.ply").exists():
        return preferred
    candidates = []
    for child in (model_dir / "point_cloud").glob("iteration_*"):
        try:
            value = int(child.name.split("_", 1)[1])
        except Exception:
            continue
        if (child / "point_cloud.ply").exists():
            candidates.append(value)
    return max(candidates) if candidates else None


def export_depth(scene: Scene, band: str, model_dir: Path) -> None:
    iteration = resolve_model_iteration(model_dir, 30000)
    if iteration is None:
        raise RuntimeError(f"No point cloud for {model_dir}")
    bundle = DEPTH / scene.name / "bundles" / "MMS-retained" / f"Model_{band}"
    manifest = bundle / "split_manifest.json"
    if not manifest.exists():
        run(
            [
                PY,
                "tools/depth_reference_geometry_v1/export_gaussian_probe_bundle.py",
                "-m",
                model_dir,
                "--iteration",
                iteration,
                "--out_dir",
                bundle,
                "--split_label",
                f"MMS-retained_{scene.name}_Model_{band}",
                "--depth_backend",
                "gaussian_point_splat",
                "--camera_frame_mode",
                "probe_manifest_native_align",
                "--world_alignment_mode",
                "similarity",
                "--probe_camera_manifest",
                scene.ref_manifest,
                "--native_cameras_json",
                model_dir / "cameras.json",
                "--scene_name_override",
                scene.name,
                "--quiet",
            ],
            LOGS / scene.name / f"depth_export_mms_retained_{band}.log",
            cwd=REPO,
        )
    adapter = bundle / "adapter_manifest.json"
    split_semantics = "metric_camera_z_from_point_splat_centers"
    if manifest.exists():
        try:
            split_semantics = json.loads(manifest.read_text(encoding="utf-8")).get("depth_semantics", split_semantics)
        except Exception:
            split_semantics = "metric_camera_z_from_point_splat_centers"
    rewrite_adapter = True
    if adapter.exists():
        try:
            existing = json.loads(adapter.read_text(encoding="utf-8"))
            rewrite_adapter = existing.get("depth_semantics") not in {
                "metric_camera_z_from_point_splat_centers",
                "metric_camera_z",
                "inverse_camera_z_from_renderer",
            }
        except Exception:
            rewrite_adapter = True
    if rewrite_adapter:
        write_json(
            adapter,
            {
                "protocol_name": "reference-depth-based-geometric-evaluation-v1",
                "method_name": f"MMS-retained_{scene.name}_Model_{band}",
                "depth_semantics": split_semantics,
                "validity_rule": {"mode": "opacity_threshold", "opacity_threshold": 0.5, "depth_min": 1e-6},
                "notes": "MMS retained raw checkpoint converted to repo-compatible 3DGS PLY; depth exported from learned Gaussian centers, not from rendered PNGs or COLMAP sparse points.",
                "point_source": {
                    "point_source": "converted_mms_native_checkpoint_gaussian_ply",
                    "point_cloud_path": str(model_dir / f"point_cloud/iteration_{iteration}/point_cloud.ply"),
                },
                "depth_semantics_note": "Semantic label follows the exported point-splat depth bundle; MMS conversion semantics are recorded in the conversion audit.",
                "split_manifest": str(manifest),
                "created_at": stamp(),
            },
        )
    out_eval = DEPTH / scene.name / "eval" / "MMS-retained" / f"Model_{band}"
    if not (out_eval / "metrics_summary.json").exists():
        run(
            [
                PY,
                "tools/depth_reference_geometry_v1/evaluate_depth_reference.py",
                "--reference_manifest",
                scene.ref_manifest,
                "--model_manifest",
                manifest,
                "--adapter_manifest",
                adapter,
                "--out_dir",
                out_eval,
                "--enable_agreement_metrics",
                "--error_mode",
                "relative_depth",
                "--thresholds",
                THRESHOLDS,
            ],
            LOGS / scene.name / f"depth_eval_mms_retained_{band}.log",
            cwd=REPO,
        )


def aggregate_depth() -> None:
    rows: list[dict] = []
    for scene in SCENES:
        root = DEPTH / scene.name / "eval" / "MMS-retained"
        for model_dir in sorted(root.glob("Model_*")):
            summary = model_dir / "metrics_summary.json"
            if not summary.exists():
                continue
            data = json.loads(summary.read_text(encoding="utf-8"))
            row = {"scene": scene.name, "method": "MMS-retained", "band": model_dir.name.replace("Model_", "")}
            secondary = data.get("secondary_metrics", {})
            row["AbsRelMean"] = (
                secondary.get("AbsRelativeDepthError_Mean")
                or secondary.get("AbsRelDepthError_Mean")
                or secondary.get("AbsDepthError_Mean")
            )
            row["ModelValidRate"] = secondary.get("ModelValidOnReferenceRate")
            counts = data.get("counts", {})
            row["Valid"] = counts.get("model_valid_on_reference_pixels")
            row["ReferenceValid"] = counts.get("reference_valid_pixels")
            for metric in data.get("threshold_metrics", []):
                threshold = metric.get("threshold")
                row[f"Agree@{threshold}"] = (
                    metric.get("Agree")
                    or metric.get("agreement_rate")
                    or metric.get("Agreement")
                    or metric.get("DepthAgreementRate")
                )
                row[f"Front@{threshold}"] = metric.get("FrontIntrusionRate")
                row[f"Deep@{threshold}"] = metric.get("TooDeepRate")
            rows.append(row)
    if not rows:
        return
    SUMMARY.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys()})
    out = SUMMARY / "mms_raw_retained_depth_metrics.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    event(f"WROTE {out} rows={len(rows)}")


def process_scene(scene: Scene) -> None:
    event(f"START {scene.name}")
    preprocessed = ensure_preprocessed(scene)
    split = ensure_split(scene, preprocessed)
    ckpt = ensure_train(scene, preprocessed, split)
    ensure_eval(scene, preprocessed, split)
    converted = ensure_converted_models(scene, preprocessed, ckpt)
    for band in ["D"] + BANDS:
        export_depth(scene, band, converted / f"Model_{band}")
    write_json(
        RUN / scene.name / "mms_retained_success.json",
        {
            "scene": scene.name,
            "status": "depth_complete",
            "preprocessed": str(preprocessed),
            "split_json": str(split),
            "checkpoint": str(ckpt),
            "converted_models": str(converted),
            "depth_root": str(DEPTH / scene.name),
            "created_at": stamp(),
        },
    )
    event(f"DONE {scene.name}")


def main() -> None:
    STATUS.mkdir(parents=True, exist_ok=True)
    write_json(
        RUN / "run_manifest.json",
        {
            "run_root": str(RUN),
            "raw_attempt_root": str(RAW_ATTEMPT),
            "scenes": [
                {
                    "name": scene.name,
                    "raw_root": str(scene.raw_root),
                    "e3_root": str(scene.e3_root),
                    "ref_manifest": str(scene.ref_manifest),
                    "source_attempt_root": str(scene.source_attempt_root),
                }
                for scene in SCENES
            ],
            "thresholds": THRESHOLDS,
            "started_at": stamp(),
        },
    )
    for scene in SCENES:
        try:
            process_scene(scene)
        except Exception as exc:
            write_json(
                RUN / scene.name / "mms_retained_failure.json",
                {
                    "scene": scene.name,
                    "status": "failed",
                    "reason": "mms_retained_depth_failed",
                    "detail": repr(exc),
                    "created_at": stamp(),
                },
            )
            event(f"FAILED {scene.name}: {exc!r}")
    aggregate_depth()
    (STATUS / "finished.txt").write_text(stamp() + "\n", encoding="utf-8")
    event("mms_raw_retained_depth finished")


if __name__ == "__main__":
    main()
