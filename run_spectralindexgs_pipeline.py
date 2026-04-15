from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from build_rectified_band_dataset import build_rectified_band_dataset
from build_spectral_products import build_products
from estimate_band_homographies import estimate_band_homographies
from prepare_m3m_multispectral import prepare_m3m_dataset
from qa_rectification import run_rectification_qa
from utils.minima_bridge import check_backend_available


STEP_NAMES = [
    "01_prepare",
    "02_train_rgb",
    "03_build_rectified_bands",
    "04_train_band_g",
    "05_train_band_r",
    "06_train_band_re",
    "07_train_band_nir",
    "08_build_products",
    "09_optional_render",
]
BANDS = ("G", "R", "RE", "NIR")


def _run(cmd, cwd: Path) -> None:
    quoted = " ".join([f'"{c}"' if (" " in c or "\t" in c) else c for c in cmd])
    print(f"\n[RUN] {quoted}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _in_step_range(args, index: int) -> bool:
    return args.from_step <= index <= args.to_step


def _str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in ("0", "false", "no", "off")


def _load_scene_manifest_payload(scene_root: Path) -> dict:
    manifest_path = scene_root / "spectral_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _assert_rectified_scene(scene_root: Path) -> None:
    payload = _load_scene_manifest_payload(scene_root)
    rectification_status = str(payload.get("rectification_status", "") or "").strip().lower()
    scene_kind = str(payload.get("scene_kind", "") or "").strip().lower()
    if rectification_status != "rectified" or scene_kind != "rectified_band":
        raise RuntimeError(
            f"Rectified band scene required, but got scene_root={scene_root}, "
            f"rectification_status={rectification_status!r}, scene_kind={scene_kind!r}."
        )


def _load_rectification_qa_summary(rectified_root: Path) -> dict:
    summary_path = rectified_root / "rectification_qa" / "rectification_qa_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing rectification QA summary: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _assert_rectification_qa_pass(rectified_root: Path, fail_fast: bool = True) -> None:
    summary = _load_rectification_qa_summary(rectified_root)
    failed = []
    for band in BANDS:
        band_summary = summary.get("bands", {}).get(band, {})
        if not bool(band_summary.get("pass", False)):
            failed.append(
                {
                    "band": band,
                    "delta_edge_f1": band_summary.get("delta_edge_f1", None),
                    "delta_grad_ncc": band_summary.get("delta_grad_ncc", None),
                }
            )
    if not failed:
        return
    msg = f"Rectification QA failed for bands: {failed}"
    print(f"[ERROR] {msg}")
    if fail_fast:
        raise RuntimeError(msg)
    print("[WARN] Continuing despite rectification QA failure because fail-fast is disabled.")


def _sync_sparse_from_rgb(prepared_root: Path, rectified_root: Path) -> None:
    rgb_sparse = prepared_root / "RGB" / "sparse" / "0"
    if not rgb_sparse.exists():
        raise FileNotFoundError(f"RGB sparse/0 missing: {rgb_sparse}")
    for band in BANDS:
        scene_root = rectified_root / f"{band}_rectified"
        payload = _load_scene_manifest_payload(scene_root)
        if str(payload.get("rectification_status", "") or "").strip().lower() != "rectified":
            raise RuntimeError(f"_sync_sparse_from_rgb only supports rectified band scenes, got: {scene_root}")
        dst = scene_root / "sparse" / "0"
        if dst.parent.exists():
            shutil.rmtree(dst.parent)
        dst.mkdir(parents=True, exist_ok=True)
        for item in rgb_sparse.iterdir():
            if item.is_file():
                shutil.copy2(item, dst / item.name)


def _clean_rgb_convert_outputs(rgb_scene_root: Path) -> None:
    for rel in ("distorted", "images", "sparse"):
        target = rgb_scene_root / rel
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)


def _ensure_rgb_colmap(repo_root: Path, rgb_scene_root: Path, args) -> None:
    sparse0 = rgb_scene_root / "sparse" / "0"
    images_dir = rgb_scene_root / "images"
    if sparse0.exists() and images_dir.exists():
        return

    def _make_convert_cmd(matching_name: str):
        matcher_args = str(args.matcher_args) if matching_name == str(args.matching) else ""
        return [
            sys.executable, "convert_uavfgs.py",
            "-s", str(rgb_scene_root),
            "--colmap_executable", str(args.colmap_executable),
            "--exiftool_executable", str(args.exiftool_executable),
            "--camera", str(args.camera),
            "--matching", str(matching_name),
            "--matcher_args", matcher_args,
            "--prior_position_std_m", str(args.prior_position_std_m),
            "--wgs84_code", str(args.wgs84_code),
        ]

    try:
        _run(_make_convert_cmd(str(args.matching)), cwd=repo_root)
    except subprocess.CalledProcessError:
        if str(args.matching) != "spatial":
            raise
        print("[WARN] RGB COLMAP convert failed with spatial matching; retrying once with exhaustive matching.")
        _clean_rgb_convert_outputs(rgb_scene_root)
        _run(_make_convert_cmd("exhaustive"), cwd=repo_root)

    if not sparse0.exists():
        raise FileNotFoundError(f"RGB convert did not create sparse/0: {sparse0}")


def _train_rgb(repo_root: Path, prepared_root: Path, out_root: Path, args) -> None:
    rgb_scene_root = prepared_root / "RGB"
    _ensure_rgb_colmap(repo_root, rgb_scene_root, args)
    model_dir = out_root / "Model_RGB"
    cmd = [
        sys.executable, "train.py",
        "-s", str(rgb_scene_root),
        "--images", "images",
        "-m", str(model_dir),
        "-r", str(args.rgb_res),
        "--iterations", str(args.rgb_iter),
        "--checkpoint_iterations", str(args.rgb_iter),
        "--save_iterations", str(args.rgb_iter),
        "--test_iterations", str(args.rgb_iter),
        "--disable_viewer",
        "--eval",
        "--modality_kind", "rgb",
    ]
    _run(cmd, cwd=repo_root)


def _build_rectified_bands(prepared_root: Path, rectified_root: Path, args) -> None:
    if str(args.rectification_backend).strip().lower() == "minima":
        if not check_backend_available(args.minima_method, minima_root=args.minima_root):
            raise RuntimeError(
                f"MINIMA backend unavailable before rectification step: "
                f"method={args.minima_method}, minima_root={Path(args.minima_root).resolve()}"
            )
    homography_json = Path(args.rectification_config).resolve() if args.rectification_config else (rectified_root / "rectification_homographies.json")
    args.rectification_config = str(homography_json)
    estimate_band_homographies(
        prepared_root=prepared_root,
        out_json=homography_json,
        frame_count=int(args.rectification_estimation_frames),
        input_dynamic_range=str(args.input_dynamic_range),
        radiometric_mode=str(args.radiometric_mode),
        rectification_use_metadata_h0=bool(args.rectification_use_metadata_h0),
        rectification_global_mode=str(args.rectification_global_mode),
        rectification_optimizer_backend=str(args.rectification_optimizer_backend),
        rectification_search_restarts=int(args.rectification_search_restarts),
        rectification_search_steps=int(args.rectification_search_steps),
        rectification_edge_dilate_radius=int(args.rectification_edge_dilate_radius),
        rectification_residual_reg=float(args.rectification_residual_reg),
        rectification_alignment_scale=float(args.rectification_alignment_scale),
        rectification_min_structure_score=args.rectification_min_structure_score,
        rectification_debug_use_legacy_ecc=bool(args.rectification_debug_use_legacy_ecc),
        rectification_min_improved_ratio=float(args.rectification_min_improved_ratio),
        rectification_max_severe_outliers=int(args.rectification_max_severe_outliers),
        rectification_backend=str(args.rectification_backend),
        minima_method=str(args.minima_method),
        minima_root=str(args.minima_root),
        minima_device=str(args.minima_device),
        minima_ckpt=str(args.minima_ckpt),
        minima_roma_size=str(args.minima_roma_size),
        minima_match_threshold=float(args.minima_match_threshold),
        minima_fine_threshold=float(args.minima_fine_threshold),
        minima_match_conf_thresh=float(args.minima_match_conf_thresh),
        minima_min_matches=int(args.minima_min_matches),
        minima_min_inlier_ratio=float(args.minima_min_inlier_ratio),
        minima_max_reproj_error=float(args.minima_max_reproj_error),
        minima_min_coverage=float(args.minima_min_coverage),
        minima_coverage_grid=int(args.minima_coverage_grid),
        minima_ransac_method=str(args.minima_ransac_method),
        minima_ransac_thresh=float(args.minima_ransac_thresh),
        minima_ransac_confidence=float(args.minima_ransac_confidence),
        minima_ransac_max_iters=int(args.minima_ransac_max_iters),
        minima_min_good_frames=int(args.minima_min_good_frames),
        minima_initial_candidate_ratio=float(args.minima_initial_candidate_ratio),
        minima_candidate_ratio_step=float(args.minima_candidate_ratio_step),
        minima_max_candidate_ratio=float(args.minima_max_candidate_ratio),
        minima_use_all_if_needed=bool(args.minima_use_all_if_needed),
        rectification_enable_residual_refine=bool(args.rectification_enable_residual_refine),
    )
    build_rectified_band_dataset(
        prepared_root=prepared_root,
        rectified_root=rectified_root,
        homography_json=homography_json,
    )
    _sync_sparse_from_rgb(prepared_root, rectified_root)
    if int(args.rectification_qa_frames) > 0:
        run_rectification_qa(
            prepared_root=prepared_root,
            rectified_root=rectified_root,
            out_root=rectified_root / "rectification_qa",
            frame_count=int(args.rectification_qa_frames),
            input_dynamic_range=str(args.input_dynamic_range),
            radiometric_mode=str(args.radiometric_mode),
            edge_dilate_radius=int(args.rectification_edge_dilate_radius),
            min_improved_ratio=float(args.rectification_min_improved_ratio),
            max_severe_outliers=int(args.rectification_max_severe_outliers),
            qa_scale=float(args.rectification_qa_scale if args.rectification_qa_scale > 0 else args.rectification_alignment_scale),
        )
        _assert_rectification_qa_pass(rectified_root, fail_fast=bool(args.rectification_qa_fail_fast))


def _train_band(repo_root: Path, rectified_root: Path, out_root: Path, args, band: str) -> None:
    if int(args.band_iter) <= int(args.rgb_iter):
        raise ValueError(
            f"band_iter must be greater than rgb_iter for stage-2 restore. "
            f"Got rgb_iter={args.rgb_iter}, band_iter={args.band_iter}"
        )
    if bool(args.rectification_qa_fail_fast):
        _assert_rectification_qa_pass(rectified_root, fail_fast=True)
    scene_root = rectified_root / f"{band}_rectified"
    if args.require_rectified_band_scene:
        _assert_rectified_scene(scene_root)
    model_dir = out_root / f"Model_{band}"
    rgb_ckpt = out_root / "Model_RGB" / f"chkpnt{args.rgb_iter}.pth"
    if not rgb_ckpt.exists():
        raise FileNotFoundError(f"Missing RGB checkpoint: {rgb_ckpt}")
    cmd = [
        sys.executable, "train.py",
        "-s", str(scene_root),
        "--images", "images",
        "-m", str(model_dir),
        "--start_checkpoint", str(rgb_ckpt),
        "-r", str(args.band_res),
        "--iterations", str(args.band_iter),
        "--checkpoint_iterations", str(args.band_iter),
        "--save_iterations", str(args.band_iter),
        "--test_iterations", str(args.band_iter),
        "--disable_viewer",
        "--eval",
        "--modality_kind", "band",
        "--target_band", band,
        "--single_band_mode", "true",
        "--single_band_replicate_to_rgb", "true",
        "--input_dynamic_range", str(args.input_dynamic_range),
        "--radiometric_mode", str(args.radiometric_mode),
        "--stage2_mode", "band_transfer",
        "--reset_appearance_features", "true",
        "--freeze_geometry", "true",
        "--freeze_opacity", "true" if args.freeze_opacity else "false",
        "--tied_scalar_carrier", "true",
        "--feature_lr", str(args.band_feature_lr),
        "--lambda_dssim", "0",
        "--require_rectified_band_scene", "true" if args.require_rectified_band_scene else "false",
        "--use_validity_mask", "true" if args.use_validity_mask else "false",
        "--rectified_root", str(rectified_root),
        "--rectification_method", str(args.rectification_method),
    ]
    if args.rectification_config:
        cmd.extend(["--rectification_config", str(Path(args.rectification_config).resolve())])
    if args.freeze_opacity:
        cmd.extend(["--opacity_lr", "0"])
    elif args.band_opacity_lr is not None:
        cmd.extend(["--opacity_lr", str(args.band_opacity_lr)])
    _run(cmd, cwd=repo_root)


def _optional_render(repo_root: Path, rectified_root: Path, out_root: Path, args) -> None:
    if not args.auto_render:
        return
    render_targets = [
        (out_root / "Model_G", rectified_root / "G_rectified"),
        (out_root / "Model_R", rectified_root / "R_rectified"),
        (out_root / "Model_RE", rectified_root / "RE_rectified"),
        (out_root / "Model_NIR", rectified_root / "NIR_rectified"),
    ]
    products_root = out_root / "Products"
    if products_root.exists():
        for product_dir in sorted(products_root.iterdir()):
            if product_dir.is_dir():
                render_targets.append((product_dir, rectified_root / "G_rectified"))
    for model_dir, scene_root in render_targets:
        if not model_dir.exists():
            continue
        cmd = [
            sys.executable, "render.py",
            "-m", str(model_dir),
            "-s", str(scene_root),
            "-r", str(args.band_res),
            "--skip_train",
        ]
        _run(cmd, cwd=repo_root)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the SpectralIndexGS rectified-band pipeline without touching the legacy thermal pipeline.")
    ap.add_argument("--raw_root", required=True, help="Raw M3M dataset root.")
    ap.add_argument("--prepared_root", required=True, help="Prepared raw scene root.")
    ap.add_argument("--rectified_root", default="", help="Optional rectified scene root. Defaults to prepared_root.")
    ap.add_argument("--out_root", required=True, help="Output model root.")
    ap.add_argument("--rectification_config", default="", help="Optional fixed-homography JSON path.")
    ap.add_argument("--rectification_method", default="minima_assisted_global_homography")
    ap.add_argument("--rectification_backend", default="minima", choices=["minima"])
    ap.add_argument("--rectification_estimation_frames", type=int, default=12)
    ap.add_argument("--rectification_qa_frames", type=int, default=6)
    ap.add_argument("--rectification_use_metadata_h0", type=str, default="true")
    ap.add_argument("--rectification_global_mode", default="affine_residual_over_h0", choices=["affine_residual_over_h0", "projective_residual_over_h0"])
    ap.add_argument("--rectification_optimizer_backend", default="opencv_search", choices=["opencv_search", "scipy_minimize"])
    ap.add_argument("--rectification_search_restarts", type=int, default=5)
    ap.add_argument("--rectification_search_steps", type=int, default=50)
    ap.add_argument("--rectification_edge_dilate_radius", type=int, default=1)
    ap.add_argument("--rectification_residual_reg", type=float, default=1e-3)
    ap.add_argument("--rectification_alignment_scale", type=float, default=0.25)
    ap.add_argument("--rectification_min_structure_score", type=float, default=None)
    ap.add_argument("--rectification_qa_fail_fast", type=str, default="true")
    ap.add_argument("--rectification_debug_use_legacy_ecc", type=str, default="false")
    ap.add_argument("--rectification_min_improved_ratio", type=float, default=0.6)
    ap.add_argument("--rectification_max_severe_outliers", type=int, default=0)
    ap.add_argument("--rectification_qa_scale", type=float, default=-1.0)
    ap.add_argument("--rectification_enable_residual_refine", type=str, default="false")

    ap.add_argument("--minima_method", default="roma", choices=["roma", "xoftr"])
    ap.add_argument("--minima_root", default=r"G:\2DSOTA\MINIMA")
    ap.add_argument("--minima_device", default="cuda")
    ap.add_argument("--minima_ckpt", default="")
    ap.add_argument("--minima_roma_size", default="large", choices=["large", "tiny"])
    ap.add_argument("--minima_match_threshold", type=float, default=0.3)
    ap.add_argument("--minima_fine_threshold", type=float, default=0.1)
    ap.add_argument("--minima_match_conf_thresh", type=float, default=0.2)
    ap.add_argument("--minima_min_matches", type=int, default=80)
    ap.add_argument("--minima_min_inlier_ratio", type=float, default=0.30)
    ap.add_argument("--minima_max_reproj_error", type=float, default=4.0)
    ap.add_argument("--minima_min_coverage", type=float, default=0.25)
    ap.add_argument("--minima_coverage_grid", type=int, default=4)
    ap.add_argument("--minima_ransac_method", default="usac_magsac", choices=["usac_magsac", "ransac"])
    ap.add_argument("--minima_ransac_thresh", type=float, default=3.0)
    ap.add_argument("--minima_ransac_confidence", type=float, default=0.999)
    ap.add_argument("--minima_ransac_max_iters", type=int, default=10000)
    ap.add_argument("--minima_min_good_frames", type=int, default=0)
    ap.add_argument("--minima_initial_candidate_ratio", type=float, default=0.15)
    ap.add_argument("--minima_candidate_ratio_step", type=float, default=0.15)
    ap.add_argument("--minima_max_candidate_ratio", type=float, default=0.50)
    ap.add_argument("--minima_use_all_if_needed", type=str, default="true")
    ap.add_argument("--from_step", type=int, default=1)
    ap.add_argument("--to_step", type=int, default=len(STEP_NAMES))
    ap.add_argument("--link_mode", default="hardlink", choices=["copy", "hardlink", "symlink"])
    ap.add_argument("--rgb_iter", type=int, default=30000)
    ap.add_argument("--band_iter", type=int, default=40000)
    ap.add_argument("--rgb_res", type=int, default=4)
    ap.add_argument("--band_res", type=int, default=4)
    ap.add_argument("--input_dynamic_range", default="uint16", choices=["uint8", "uint16", "float"])
    ap.add_argument("--band_feature_lr", type=float, default=0.001)
    ap.add_argument("--band_opacity_lr", type=float, default=None)
    ap.add_argument("--freeze_opacity", type=str, default="true")
    ap.add_argument("--require_rectified_band_scene", type=str, default="true")
    ap.add_argument("--use_validity_mask", type=str, default="true")
    ap.add_argument("--radiometric_mode", default="exposure_normalized", choices=["raw_dn", "exposure_normalized", "reflectance_ready_stub"])
    ap.add_argument("--savi_l", type=float, default=0.5)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--colmap_executable", default="colmap")
    ap.add_argument("--exiftool_executable", default="exiftool")
    ap.add_argument("--camera", default="SIMPLE_RADIAL")
    ap.add_argument("--matching", default="spatial", choices=["spatial", "exhaustive", "sequential", "vocab_tree"])
    ap.add_argument("--matcher_args", default="--SpatialMatching.max_num_neighbors=80 --SpatialMatching.max_distance=500")
    ap.add_argument("--prior_position_std_m", type=float, default=1.0)
    ap.add_argument("--wgs84_code", type=int, default=0)
    ap.add_argument("--sparse_source", default="", help="Optional existing sparse/0 to seed the prepared RGB scene only.")
    ap.add_argument("--auto_render", action="store_true", default=False)
    args = ap.parse_args()

    args.freeze_opacity = _str2bool(args.freeze_opacity)
    args.require_rectified_band_scene = _str2bool(args.require_rectified_band_scene)
    args.use_validity_mask = _str2bool(args.use_validity_mask)
    args.rectification_use_metadata_h0 = _str2bool(args.rectification_use_metadata_h0)
    args.rectification_qa_fail_fast = _str2bool(args.rectification_qa_fail_fast)
    args.rectification_debug_use_legacy_ecc = _str2bool(args.rectification_debug_use_legacy_ecc)
    args.rectification_enable_residual_refine = _str2bool(args.rectification_enable_residual_refine)
    args.minima_use_all_if_needed = _str2bool(args.minima_use_all_if_needed)

    repo_root = Path(__file__).resolve().parent
    raw_root = Path(args.raw_root).resolve()
    prepared_root = Path(args.prepared_root).resolve()
    rectified_root = Path(args.rectified_root).resolve() if args.rectified_root else prepared_root
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    rectified_root.mkdir(parents=True, exist_ok=True)
    if not args.rectification_config:
        args.rectification_config = str((rectified_root / "rectification_homographies.json").resolve())

    if _in_step_range(args, 1):
        prepare_m3m_dataset(
            raw_root=raw_root,
            out_root=prepared_root,
            link_mode=args.link_mode,
            sparse_source=Path(args.sparse_source).resolve() if args.sparse_source else None,
            exiftool_executable=args.exiftool_executable,
        )
    if _in_step_range(args, 2):
        _train_rgb(repo_root, prepared_root, out_root, args)
    if _in_step_range(args, 3):
        _build_rectified_bands(prepared_root, rectified_root, args)
    if _in_step_range(args, 4):
        _train_band(repo_root, rectified_root, out_root, args, "G")
    if _in_step_range(args, 5):
        _train_band(repo_root, rectified_root, out_root, args, "R")
    if _in_step_range(args, 6):
        _train_band(repo_root, rectified_root, out_root, args, "RE")
    if _in_step_range(args, 7):
        _train_band(repo_root, rectified_root, out_root, args, "NIR")
    if _in_step_range(args, 8):
        build_products(
            model_dirs={
                "G": out_root / "Model_G",
                "R": out_root / "Model_R",
                "RE": out_root / "Model_RE",
                "NIR": out_root / "Model_NIR",
            },
            iterations={
                "G": args.band_iter,
                "R": args.band_iter,
                "RE": args.band_iter,
                "NIR": args.band_iter,
            },
            out_root=out_root / "Products",
            savi_l=args.savi_l,
            eps=args.eps,
            require_opacity_match=args.freeze_opacity,
            tol=1e-6,
        )
    if _in_step_range(args, 9):
        _optional_render(repo_root, rectified_root, out_root, args)


if __name__ == "__main__":
    main()
