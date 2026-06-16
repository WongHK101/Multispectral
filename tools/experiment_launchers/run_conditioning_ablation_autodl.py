#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from plyfile import PlyData

REPO = Path("/root/autodl-tmp/Multispectral")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from build_spectral_products import build_products


PY = Path("/root/autodl-tmp/envs/spectralindexgs_bw/bin/python")
RUN = Path("/root/autodl-tmp/runs/nightly_20260517_p1/conditioning_ablation")

RGB_ITER = 30000
BANDS = ("G", "R", "RE", "NIR")


@dataclass(frozen=True)
class SceneConfig:
    name: str
    published_scene: str
    e3_root: Path
    train_count: int | None = None
    test_count: int | None = None
    split_source: str | None = None
    registered_names_sha256: str | None = None
    rectified_root_override: Path | None = None
    rgb_checkpoint_override: Path | None = None

    @property
    def rectified_root(self) -> Path:
        if self.rectified_root_override is not None:
            return self.rectified_root_override
        return self.e3_root / "rectified"

    @property
    def rgb_checkpoint(self) -> Path:
        if self.rgb_checkpoint_override is not None:
            return self.rgb_checkpoint_override
        return self.e3_root / f"out/Model_RGB/chkpnt{RGB_ITER}.pth"


SCENES = {
    "raw_self": SceneConfig(
        "raw_self",
        "UMGS-Field",
        Path("/root/autodl-tmp/runs/paper_autodl_full_20260429/e3_raw_self_clean_cost_20260501_230446/raw_self"),
    ),
    "raw003": SceneConfig(
        "raw003",
        "Vineyard-03",
        Path("/root/autodl-tmp/runs/paper_autodl_full_20260429/e3_raw7_gpu_colmap_20260429_180200/raw003"),
    ),
    "raw001": SceneConfig(
        "raw001",
        "Vineyard-01",
        Path("/root/autodl-tmp/runs/paper_autodl_full_20260429/e3_raw7_gpu_colmap_20260429_180200/raw001"),
    ),
    "maize_02_20260526_1658": SceneConfig(
        "maize_02_20260526_1658",
        "Maize-02",
        Path("/root/autodl-tmp/runs/uav_multispec3d_active17_registered100_umgs_i_20260602_120858/maize_02_20260526_1658"),
        train_count=75,
        test_count=11,
        split_source="run_registered_split",
        registered_names_sha256="34d170075b778174778683bea22d0f7a212f6852a9afa9bb44b17e6184a226d6",
    ),
    "cassava_01_20260526_1603": SceneConfig(
        "cassava_01_20260526_1603",
        "Cassava",
        Path("/root/autodl-tmp/runs/uav_multispec3d_release15_clean8_umgs_i_20260529_030125/cassava_01_20260526_1603"),
        train_count=105,
        test_count=15,
        split_source="run_registered_split",
        registered_names_sha256="1389fd17aadccc1f5f7cf413ec9d0cfe1c107611cc4d4814d9a04f67440fa5fc",
    ),
    "chunya_01_20260526_1021": SceneConfig(
        "chunya_01_20260526_1021",
        "Chunya",
        Path("/root/autodl-tmp/runs/uav_multispec3d_release15_clean8_umgs_i_20260529_030125/chunya_01_20260526_1021"),
        train_count=59,
        test_count=9,
        split_source="run_registered_split",
        registered_names_sha256="8ead38d2f047575c1e233d5612f36c8c45072e9aa0677f3f7b5b7bef3747f1d5",
    ),
}
EXPECTED_PRODUCTS = (
    "false_color_nir_r_g",
    "false_color_re_nir_r",
    "false_color_nir_re_g",
    "ndvi_gray",
    "ndvi_pseudocolor",
    "ndre_gray",
    "ndre_pseudocolor",
    "gndvi_gray",
    "gndvi_pseudocolor",
    "savi_gray",
    "savi_pseudocolor",
    "osavi_gray",
    "osavi_pseudocolor",
)
SUPPORT_FIELDS = (
    "x",
    "y",
    "z",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
)


def stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def scene_from_json(path: Path) -> SceneConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = ("name", "published_scene", "e3_root")
    missing = [key for key in required if not payload.get(key)]
    if missing:
        raise ValueError(f"scene config missing required keys {missing}: {path}")
    return SceneConfig(
        name=str(payload["name"]),
        published_scene=str(payload["published_scene"]),
        e3_root=Path(payload["e3_root"]),
        train_count=int(payload["train_count"]) if payload.get("train_count") is not None else None,
        test_count=int(payload["test_count"]) if payload.get("test_count") is not None else None,
        split_source=payload.get("split_source"),
        registered_names_sha256=payload.get("registered_names_sha256"),
        rectified_root_override=Path(payload["rectified_root"]) if payload.get("rectified_root") else None,
        rgb_checkpoint_override=Path(payload["rgb_checkpoint"]) if payload.get("rgb_checkpoint") else None,
    )


def resolve_scene_config(args: argparse.Namespace) -> SceneConfig:
    if args.scene_config_json:
        scene = scene_from_json(Path(args.scene_config_json))
    else:
        if args.scene not in SCENES:
            known = ", ".join(sorted(SCENES))
            raise ValueError(f"unknown scene {args.scene!r}; known scenes: {known}")
        scene = SCENES[args.scene]

    return SceneConfig(
        name=args.scene_id or scene.name,
        published_scene=args.published_scene or scene.published_scene,
        e3_root=Path(args.e3_root) if args.e3_root else scene.e3_root,
        train_count=args.train_count if args.train_count is not None else scene.train_count,
        test_count=args.test_count if args.test_count is not None else scene.test_count,
        split_source=args.split_source or scene.split_source,
        registered_names_sha256=args.registered_names_sha256 or scene.registered_names_sha256,
        rectified_root_override=Path(args.rectified_root) if args.rectified_root else scene.rectified_root_override,
        rgb_checkpoint_override=Path(args.rgb_checkpoint) if args.rgb_checkpoint else scene.rgb_checkpoint_override,
    )


def event(run_dir: Path, message: str) -> None:
    line = f"[{stamp()}] {message}"
    print(line, flush=True)
    status = run_dir / "status"
    status.mkdir(parents=True, exist_ok=True)
    with (status / "events.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run_cmd(run_dir: Path, cmd: Iterable[object], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd_text = " ".join(str(part) for part in cmd)
    event(run_dir, f"RUN {cmd_text} > {log}")
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    with log.open("w", encoding="utf-8", errors="replace") as handle:
        handle.write("$ " + cmd_text + "\n")
        handle.flush()
        proc = subprocess.run(
            [str(part) for part in cmd],
            cwd=str(REPO),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}: {cmd_text}; log={log}")


def point_cloud_path(model_dir: Path, iteration: int) -> Path:
    return model_dir / f"point_cloud/iteration_{iteration}/point_cloud.ply"


def train_band(scene: SceneConfig, run_dir: Path, out_root: Path, band: str, iteration: int) -> None:
    model_dir = out_root / "out" / f"Model_{band}"
    if point_cloud_path(model_dir, iteration).exists():
        event(run_dir, f"SKIP train {band}: existing {point_cloud_path(model_dir, iteration)}")
        return
    source = scene.rectified_root / f"{band}_rectified"
    if not source.exists():
        raise FileNotFoundError(source)
    if not scene.rgb_checkpoint.exists():
        raise FileNotFoundError(scene.rgb_checkpoint)
    cmd = [
        PY,
        "train.py",
        "-s",
        source,
        "--images",
        "images",
        "-m",
        model_dir,
        "--start_checkpoint",
        scene.rgb_checkpoint,
        "--restore_geometry_only",
        "true",
        "-r",
        "8",
        "--iterations",
        str(iteration),
        "--save_iterations",
        str(iteration),
        "--checkpoint_iterations",
        str(iteration),
        "--test_iterations",
        str(iteration),
        "--eval",
        "--disable_viewer",
        "--quiet",
        "--modality_kind",
        "band",
        "--target_band",
        band,
        "--single_band_mode",
        "true",
        "--single_band_replicate_to_rgb",
        "true",
        "--input_dynamic_range",
        "uint16",
        "--radiometric_mode",
        "exposure_normalized",
        "--stage2_mode",
        "band_transfer",
        "--reset_appearance_features",
        "true",
        "--freeze_geometry",
        "true",
        "--freeze_opacity",
        "true",
        "--tied_scalar_carrier",
        "true",
        "--feature_lr",
        "0.001",
        "--lambda_dssim",
        "0",
        "--require_rectified_band_scene",
        "true",
        "--use_validity_mask",
        "false",
        "--rectified_root",
        scene.rectified_root,
        "--rectification_config",
        scene.rectified_root / "rectification_homographies.json",
        "--rectification_method",
        "minima_assisted_global_homography",
        "--opacity_lr",
        "0",
        "--ss_enable",
        "false",
        "--ss_prune_before_thermal",
        "false",
        "--ss_prune_after_rgb",
        "false",
        "--clamp_scale_after_densify",
        "false",
        "--clamp_scale_after_rgb_final",
        "false",
        "--thermal_reset_features",
        "false",
        "--t_struct_grad_w",
        "0.0",
        "--sgf_disable",
        "false",
        "--baseline_modules_off",
        "false",
        "--baseline_restore_ssp",
        "false",
        "--baseline_restore_stt",
        "false",
    ]
    run_cmd(run_dir, cmd, run_dir / "logs" / f"train_{band}.log")
    if not point_cloud_path(model_dir, iteration).exists():
        raise RuntimeError(f"missing point cloud after train: {point_cloud_path(model_dir, iteration)}")


def render_band(run_dir: Path, out_root: Path, band: str, iteration: int) -> None:
    model_dir = out_root / "out" / f"Model_{band}"
    render_dir = model_dir / f"test/ours_{iteration}/renders"
    if render_dir.exists() and any(render_dir.glob("*.png")):
        event(run_dir, f"SKIP render {band}: existing {render_dir}")
        return
    run_cmd(
        run_dir,
        [PY, "render.py", "-m", model_dir, "--iteration", str(iteration), "--skip_train", "--quiet"],
        run_dir / "logs" / f"render_{band}.log",
    )
    if not render_dir.exists() or not any(render_dir.glob("*.png")):
        raise RuntimeError(f"missing renders after render: {render_dir}")


def run_metrics(run_dir: Path, out_root: Path, iteration: int) -> None:
    model_dirs = [out_root / "out" / f"Model_{band}" for band in BANDS]
    if all((model_dir / "results.json").exists() for model_dir in model_dirs):
        event(run_dir, "SKIP image metrics: existing results.json for all bands")
    else:
        run_cmd(
            run_dir,
            [PY, "metrics.py", "-m", *model_dirs, "--mask_mode", "gt_nonzero"],
            run_dir / "logs" / "metrics.log",
        )
    missing = [str(model_dir / "results.json") for model_dir in model_dirs if not (model_dir / "results.json").exists()]
    if missing:
        raise RuntimeError(f"metrics.py completed but results are missing: {missing}")

    index_json = run_dir / "summary" / "index_metrics.json"
    if not index_json.exists():
        run_cmd(
            run_dir,
            [
                PY,
                "evaluate_spectral_indices.py",
                "--g_model_dir",
                out_root / "out/Model_G",
                "--r_model_dir",
                out_root / "out/Model_R",
                "--re_model_dir",
                out_root / "out/Model_RE",
                "--nir_model_dir",
                out_root / "out/Model_NIR",
                "--iteration",
                str(iteration),
                "--indices",
                "NDVI,GNDVI,NDRE",
                "--mask_mode",
                "gt_nonzero_intersection",
                "--out_json",
                index_json,
            ],
            run_dir / "logs" / "index_eval.log",
        )
    if not index_json.exists():
        raise RuntimeError(f"missing index metrics: {index_json}")


def build_products_for_run(run_dir: Path, out_root: Path, iteration: int) -> None:
    products_root = out_root / "out" / "Products"
    expected_summaries = [products_root / name / "product_summary.json" for name in EXPECTED_PRODUCTS]
    if all(path.exists() for path in expected_summaries):
        event(run_dir, f"SKIP products: existing summaries under {products_root}")
        return
    build_products(
        model_dirs={band: out_root / "out" / f"Model_{band}" for band in BANDS},
        iterations={band: iteration for band in BANDS},
        out_root=products_root,
        savi_l=0.5,
        eps=1e-6,
        require_opacity_match=True,
        tol=1e-6,
    )
    missing = [str(path) for path in expected_summaries if not path.exists()]
    if missing:
        raise RuntimeError(f"missing product summaries after build_products: {missing}")


def support_matrix(ply_path: Path) -> np.ndarray:
    vertex = PlyData.read(str(ply_path))["vertex"].data
    return np.stack([np.asarray(vertex[field], dtype=np.float64) for field in SUPPORT_FIELDS], axis=1)


def audit_support(scene: SceneConfig, run_dir: Path, out_root: Path, iteration: int) -> None:
    rows = {}
    ref = None
    ref_band = None
    for band in BANDS:
        ply = point_cloud_path(out_root / "out" / f"Model_{band}", iteration)
        exists = ply.exists()
        row = {"exists": exists, "point_cloud": str(ply)}
        if exists:
            arr = support_matrix(ply)
            row["num_gaussians"] = int(arr.shape[0])
            if ref is None:
                ref = arr
                ref_band = band
                row["reference_band"] = True
            else:
                same_shape = arr.shape == ref.shape
                row[f"same_shape_as_{ref_band}"] = bool(same_shape)
                if same_shape:
                    delta = np.abs(arr - ref)
                    row[f"max_support_delta_vs_{ref_band}"] = float(delta.max()) if delta.size else 0.0
                    row[f"mean_support_delta_vs_{ref_band}"] = float(delta.mean()) if delta.size else 0.0
        rows[band] = row
    write_json(
        run_dir / "summary" / "support_audit.json",
        {
            "scene": scene.name,
            "published_scene": scene.published_scene,
            "variant": "no_valid_mask",
            "iteration": int(iteration),
            "support_fields": SUPPORT_FIELDS,
            "bands": rows,
            "created_at": stamp(),
        },
    )


def validate_training_audits(run_dir: Path, out_root: Path) -> None:
    audits = {}
    for band in BANDS:
        path = out_root / "out" / f"Model_{band}" / "training_protocol_audit.json"
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        audits[band] = {
            "path": str(path),
            "use_validity_mask": payload.get("use_validity_mask"),
            "masked_loss_active_expected": payload.get("masked_loss_active_expected"),
            "masked_loss_observed": payload.get("masked_loss_observed"),
            "masked_loss_observed_steps": payload.get("masked_loss_observed_steps"),
            "masked_loss_protocol_ok": payload.get("masked_loss_protocol_ok"),
        }
        if payload.get("use_validity_mask") is not False:
            raise RuntimeError(f"use_validity_mask did not resolve to false for {band}: {path}")
        if payload.get("masked_loss_observed") not in (False, 0, None):
            raise RuntimeError(f"masked loss unexpectedly observed for no-valid-mask {band}: {path}")
    write_json(run_dir / "summary" / "training_protocol_audits.json", audits)


def run_variant(scene: SceneConfig, mode: str, run_root: Path) -> None:
    iteration = 30020 if mode == "smoke" else 60000
    run_dir = run_root / scene.name / f"no_valid_mask_{mode}_{iteration}"
    out_root = run_dir / "models"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        run_dir / "run_manifest.json",
        {
            "scene": scene.name,
            "published_scene": scene.published_scene,
            "variant": "no_valid_mask",
            "mode": mode,
            "iteration": iteration,
            "rgb_iter": RGB_ITER,
            "repo": str(REPO),
            "e3_root": str(scene.e3_root),
            "rectified_root": str(scene.rectified_root),
            "rgb_checkpoint": str(scene.rgb_checkpoint),
            "output_root": str(out_root),
            "train_count": scene.train_count,
            "test_count": scene.test_count,
            "split_source": scene.split_source,
            "registered_names_sha256": scene.registered_names_sha256,
            "created_at": stamp(),
        },
    )
    event(run_dir, f"conditioning ablation started mode={mode} iteration={iteration}")
    for band in BANDS:
        train_band(scene, run_dir, out_root, band, iteration)
        render_band(run_dir, out_root, band, iteration)
    run_metrics(run_dir, out_root, iteration)
    build_products_for_run(run_dir, out_root, iteration)
    audit_support(scene, run_dir, out_root, iteration)
    validate_training_audits(run_dir, out_root)
    marker = run_dir / "status" / ("smoke_passed.txt" if mode == "smoke" else "finished.txt")
    marker.write_text(stamp() + "\n", encoding="utf-8")
    event(run_dir, f"conditioning ablation finished mode={mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="raw_self", help="Built-in scene id. Use --scene_config_json for external metadata.")
    parser.add_argument("--scene_config_json", help="JSON file with scene metadata. Overrides built-in scene lookup.")
    parser.add_argument("--scene_id", help="Override output/provenance scene id.")
    parser.add_argument("--published_scene", help="Override publication-facing scene name.")
    parser.add_argument("--e3_root", help="Override scene run root containing rectified/ and out/Model_RGB.")
    parser.add_argument("--rectified_root", help="Override rectified root.")
    parser.add_argument("--rgb_checkpoint", help="Override RGB checkpoint path.")
    parser.add_argument("--train_count", type=int, help="Train-view count for provenance.")
    parser.add_argument("--test_count", type=int, help="Test-view count for provenance.")
    parser.add_argument("--split_source", help="Train/test split provenance label.")
    parser.add_argument("--registered_names_sha256", help="Registered image-list hash for provenance.")
    parser.add_argument("--run_root", default=str(RUN), help="Output run root for no-valid-mask ablations.")
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    scene = resolve_scene_config(args)
    run_variant(scene, args.mode, Path(args.run_root))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Best effort failure marker. The run directory is mode-dependent, so
        # write a top-level marker if setup failed before run_dir creation.
        fail_dir = RUN / "status"
        fail_dir.mkdir(parents=True, exist_ok=True)
        (fail_dir / "last_failure.txt").write_text(stamp() + "\n" + repr(exc) + "\n", encoding="utf-8")
        raise
