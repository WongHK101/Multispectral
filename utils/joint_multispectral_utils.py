import hashlib
import json
import os
import shutil
import subprocess
from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from plyfile import PlyData
from torch import nn

from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON


EXPECTED_GAUSSIAN_CAPTURE_LEN = 12


@dataclass
class BandAppearanceBank:
    band: str
    features_dc: nn.Parameter
    features_rest: nn.Parameter


@dataclass
class JointMultispectralState:
    gaussians: GaussianModel
    bands: List[str]
    banks: Dict[str, BandAppearanceBank]
    optimizers: Dict[str, torch.optim.Optimizer]
    updates_per_band: Dict[str, int]
    geometry_ref: Dict[str, torch.Tensor]
    active_band: str = ""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_git_commit(repo_root: Optional[Path] = None) -> str:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[1]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def _clone_to_device(x: torch.Tensor, device: str, requires_grad: bool) -> nn.Parameter:
    t = x.detach().clone().to(device)
    return nn.Parameter(t.requires_grad_(requires_grad))


def _as_tensor_payload(x: torch.Tensor) -> torch.Tensor:
    return x.detach().clone().cpu()


def load_rgb_checkpoint_geometry_only(
    rgb_checkpoint: str | Path,
    sh_degree: int,
    device: str = "cuda",
    optimizer_type: str = "default",
) -> Tuple[GaussianModel, Dict[str, torch.Tensor], Dict[str, object]]:
    """Load only shared geometry and tensor shapes from an RGB Gaussian checkpoint.

    This deliberately does not call GaussianModel.restore(), because restore()
    also initializes and reloads the original optimizer state.
    """
    ckpt_path = Path(rgb_checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing RGB checkpoint: {ckpt_path}")

    payload = torch.load(str(ckpt_path), map_location=device)
    if not (isinstance(payload, (tuple, list)) and len(payload) == 2):
        raise ValueError(f"Expected checkpoint payload (model_params, iteration), got {type(payload)}")
    model_params, checkpoint_iter = payload
    if not isinstance(model_params, (tuple, list)):
        raise ValueError(f"Expected model_params tuple/list, got {type(model_params)}")
    if len(model_params) != EXPECTED_GAUSSIAN_CAPTURE_LEN:
        raise ValueError(
            f"Unsupported GaussianModel.capture schema: expected length "
            f"{EXPECTED_GAUSSIAN_CAPTURE_LEN}, got {len(model_params)}"
        )

    active_sh_degree = int(model_params[0])
    xyz = model_params[1]
    features_dc = model_params[2]
    features_rest = model_params[3]
    scaling = model_params[4]
    rotation = model_params[5]
    opacity = model_params[6]
    max_radii2d = model_params[7]
    spatial_lr_scale = float(model_params[11])

    if active_sh_degree > int(sh_degree):
        raise ValueError(f"active_sh_degree={active_sh_degree} exceeds sh_degree={sh_degree}")
    if features_dc.ndim != 3 or features_dc.shape[1] != 1 or features_dc.shape[2] != 3:
        raise ValueError(f"Unexpected features_dc shape: {tuple(features_dc.shape)}")
    expected_rest = (int(sh_degree) + 1) ** 2 - 1
    if features_rest.ndim != 3 or features_rest.shape[1] != expected_rest or features_rest.shape[2] != 3:
        raise ValueError(
            f"Unexpected features_rest shape: {tuple(features_rest.shape)}; "
            f"expected (*,{expected_rest},3) for sh_degree={sh_degree}"
        )

    gaussians = GaussianModel(sh_degree, optimizer_type=optimizer_type)
    gaussians.active_sh_degree = active_sh_degree
    gaussians._xyz = _clone_to_device(xyz, device, requires_grad=False)
    gaussians._scaling = _clone_to_device(scaling, device, requires_grad=False)
    gaussians._rotation = _clone_to_device(rotation, device, requires_grad=False)
    gaussians._opacity = _clone_to_device(opacity, device, requires_grad=False)
    gaussians._features_dc = _clone_to_device(torch.zeros_like(features_dc), device, requires_grad=True)
    gaussians._features_rest = _clone_to_device(torch.zeros_like(features_rest), device, requires_grad=True)
    gaussians.max_radii2D = max_radii2d.detach().clone().to(device)
    gaussians.spatial_lr_scale = spatial_lr_scale

    geometry_ref = {
        "xyz": gaussians._xyz.detach().clone(),
        "scaling": gaussians._scaling.detach().clone(),
        "rotation": gaussians._rotation.detach().clone(),
        "opacity": gaussians._opacity.detach().clone(),
    }
    rgb_meta = {
        "checkpoint_path": str(ckpt_path),
        "checkpoint_rgb_iter": int(checkpoint_iter),
        "capture_tuple_length": len(model_params),
        "active_sh_degree": active_sh_degree,
        "max_sh_degree": int(sh_degree),
        "num_gaussians": int(xyz.shape[0]),
        "features_dc_shape": list(features_dc.shape),
        "features_rest_shape": list(features_rest.shape),
        "xyz_shape": list(xyz.shape),
        "scaling_shape": list(scaling.shape),
        "rotation_shape": list(rotation.shape),
        "opacity_shape": list(opacity.shape),
        "spatial_lr_scale": spatial_lr_scale,
    }
    return gaussians, geometry_ref, rgb_meta


def init_band_banks(
    rgb_feature_shapes: Mapping[str, Sequence[int]],
    bands: Sequence[str],
    bank_init: str = "zero",
    rgb_features: Optional[Mapping[str, torch.Tensor]] = None,
    device: str = "cuda",
) -> Dict[str, BandAppearanceBank]:
    bank_init = str(bank_init).lower()
    if bank_init not in {"zero", "rgb_tied"}:
        raise ValueError(f"Unsupported bank_init={bank_init!r}")
    dc_shape = tuple(int(v) for v in rgb_feature_shapes["features_dc"])
    rest_shape = tuple(int(v) for v in rgb_feature_shapes["features_rest"])
    banks: Dict[str, BandAppearanceBank] = {}
    for band in bands:
        if bank_init == "zero":
            dc = torch.zeros(dc_shape, dtype=torch.float32, device=device)
            rest = torch.zeros(rest_shape, dtype=torch.float32, device=device)
        else:
            if rgb_features is None:
                raise ValueError("rgb_features is required for bank_init=rgb_tied")
            dc = rgb_features["features_dc"].detach().clone().to(device)
            rest = rgb_features["features_rest"].detach().clone().to(device)
        banks[str(band)] = BandAppearanceBank(
            band=str(band),
            features_dc=nn.Parameter(dc.requires_grad_(True)),
            features_rest=nn.Parameter(rest.requires_grad_(True)),
        )
    return banks


def make_band_optimizers(
    banks: Mapping[str, BandAppearanceBank],
    feature_lr: float,
    eps: float = 1e-15,
) -> Dict[str, torch.optim.Optimizer]:
    optimizers: Dict[str, torch.optim.Optimizer] = {}
    for band, bank in banks.items():
        optimizers[band] = torch.optim.Adam(
            [
                {"params": [bank.features_dc], "lr": float(feature_lr), "name": "f_dc"},
                {"params": [bank.features_rest], "lr": float(feature_lr) / 20.0, "name": "f_rest"},
            ],
            lr=0.0,
            eps=eps,
        )
    return optimizers


def set_active_bank(gaussians: GaussianModel, banks: Mapping[str, BandAppearanceBank], band: str) -> None:
    if band not in banks:
        raise KeyError(f"Unknown band {band!r}; available={sorted(banks)}")
    gaussians._features_dc = banks[band].features_dc
    gaussians._features_rest = banks[band].features_rest


def project_active_bank_to_tied_scalar(bank: BandAppearanceBank) -> None:
    with torch.no_grad():
        if bank.features_dc.ndim == 3 and bank.features_dc.shape[-1] == 3:
            dc_mean = bank.features_dc.mean(dim=2, keepdim=True)
            bank.features_dc.copy_(dc_mean.expand_as(bank.features_dc))
        if bank.features_rest.ndim == 3 and bank.features_rest.shape[-1] == 3:
            rest_mean = bank.features_rest.mean(dim=2, keepdim=True)
            bank.features_rest.copy_(rest_mean.expand_as(bank.features_rest))


def compute_geometry_drift(
    gaussians: GaussianModel,
    geometry_ref: Mapping[str, torch.Tensor],
    atol: float = 1e-8,
) -> Dict[str, object]:
    current = {
        "xyz": gaussians._xyz.detach(),
        "scaling": gaussians._scaling.detach(),
        "rotation": gaussians._rotation.detach(),
        "opacity": gaussians._opacity.detach(),
    }
    drift: Dict[str, object] = {}
    same_all = True
    for key, cur in current.items():
        ref = geometry_ref[key].to(cur.device)
        if tuple(ref.shape) != tuple(cur.shape):
            max_abs = float("inf")
            same = False
        elif cur.numel() == 0:
            max_abs = 0.0
            same = True
        else:
            max_abs = float(torch.max(torch.abs(cur - ref)).detach().cpu().item())
            same = bool(max_abs <= float(atol))
        drift[f"max_abs_{key}_delta"] = max_abs
        drift[f"same_{key}"] = same
        same_all = same_all and same
    drift["same_num_gaussians"] = bool(current["xyz"].shape[0] == geometry_ref["xyz"].shape[0])
    drift["topology_identity"] = bool(
        same_all
        and drift["same_num_gaussians"]
        and drift["same_xyz"]
        and drift["same_scaling"]
        and drift["same_rotation"]
        and drift["same_opacity"]
    )
    return drift


def assert_joint_invariants(audit: Mapping[str, object]) -> None:
    required_true = (
        "shared_geometry",
        "freeze_geometry",
        "freeze_opacity",
        "appearance_only_update",
        "same_num_gaussians",
        "same_xyz",
        "same_scaling",
        "same_rotation",
        "same_opacity",
        "topology_identity",
    )
    failed = [key for key in required_true if not bool(audit.get(key, False))]
    if failed:
        raise RuntimeError(f"Joint multispectral invariants failed: {failed}")


def _sha256_names(names: Sequence[str]) -> str:
    h = hashlib.sha256()
    for name in names:
        h.update(str(name).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def make_band_dataset_args(
    scene_root: str | Path,
    band: str,
    model_path: str | Path,
    resolution: int,
    input_dynamic_range: str,
    radiometric_mode: str,
    data_device: str = "cuda",
    use_validity_mask: bool = True,
) -> Namespace:
    return Namespace(
        sh_degree=3,
        source_path=str(Path(scene_root).resolve()),
        model_path=str(Path(model_path).resolve()),
        images="images",
        depths="",
        resolution=int(resolution),
        white_background=False,
        train_test_exp=False,
        data_device=data_device,
        eval=True,
        modality_kind="band",
        target_band=str(band),
        single_band_mode=True,
        single_band_replicate_to_rgb=True,
        input_dynamic_range=str(input_dynamic_range),
        radiometric_mode=str(radiometric_mode),
        rectification_config="",
        rectified_root=str(Path(scene_root).resolve().parent),
        require_rectified_band_scene=True,
        use_validity_mask=bool(use_validity_mask),
        rectification_method="minima_assisted_global_homography",
        reset_appearance_features=True,
        freeze_geometry=True,
        freeze_opacity=True,
        tied_scalar_carrier=True,
        stage2_mode="joint_multispectral_export_view",
        rectification_backend="minima",
    )


def load_band_cameras_without_scene(
    scene_root: str | Path,
    band: str,
    args_like: Namespace,
    shuffle: bool = False,
    load_test: bool = True,
) -> Tuple[List[object], List[object], object, Dict[str, object]]:
    if shuffle:
        raise ValueError("shuffle=True is intentionally unsupported for audit-friendly E4b camera loading")
    scene_root = Path(scene_root)
    if not (scene_root / "sparse").exists():
        raise FileNotFoundError(f"Missing sparse directory in rectified band scene: {scene_root}")
    scene_info = sceneLoadTypeCallbacks["Colmap"](
        str(scene_root),
        getattr(args_like, "images", "images"),
        getattr(args_like, "depths", ""),
        bool(getattr(args_like, "eval", True)),
        bool(getattr(args_like, "train_test_exp", False)),
    )
    train_cameras = cameraList_from_camInfos(
        scene_info.train_cameras,
        1.0,
        args_like,
        scene_info.is_nerf_synthetic,
        False,
    )
    test_cameras = []
    if load_test:
        test_cameras = cameraList_from_camInfos(
            scene_info.test_cameras,
            1.0,
            args_like,
            scene_info.is_nerf_synthetic,
            True,
        )
    train_names = [str(c.image_name) for c in train_cameras]
    test_names = [str(c.image_name) for c in test_cameras]
    audit = {
        "band": str(band),
        "scene_root": str(scene_root),
        "num_train": len(train_cameras),
        "num_test": len(test_cameras) if load_test else len(scene_info.test_cameras),
        "test_cameras_loaded": bool(load_test),
        "train_image_names_sample": train_names[:5],
        "test_image_names_sample": test_names[:5],
        "train_names_sha256": _sha256_names(train_names),
        "test_names_sha256": _sha256_names(test_names),
        "resolution": int(getattr(args_like, "resolution", -1)),
        "input_dynamic_range": str(getattr(args_like, "input_dynamic_range", "")),
        "radiometric_mode": str(getattr(args_like, "radiometric_mode", "")),
        "use_validity_mask": bool(getattr(args_like, "use_validity_mask", False)),
        "ply_path": str(scene_info.ply_path),
    }
    return train_cameras, test_cameras, scene_info, audit


def load_band_scene_info_without_scene(
    scene_root: str | Path,
    band: str,
    args_like: Namespace,
) -> Tuple[object, Dict[str, object]]:
    """Read lightweight COLMAP scene_info without constructing GPU Camera objects."""
    scene_root = Path(scene_root)
    if not (scene_root / "sparse").exists():
        raise FileNotFoundError(f"Missing sparse directory in rectified band scene: {scene_root}")
    scene_info = sceneLoadTypeCallbacks["Colmap"](
        str(scene_root),
        getattr(args_like, "images", "images"),
        getattr(args_like, "depths", ""),
        bool(getattr(args_like, "eval", True)),
        bool(getattr(args_like, "train_test_exp", False)),
    )
    train_names = [str(c.image_name) for c in scene_info.train_cameras]
    test_names = [str(c.image_name) for c in scene_info.test_cameras]
    audit = {
        "band": str(band),
        "scene_root": str(scene_root),
        "num_train": len(train_names),
        "num_test": len(test_names),
        "test_cameras_loaded": False,
        "train_image_names_sample": train_names[:5],
        "test_image_names_sample": test_names[:5],
        "train_names_sha256": _sha256_names(train_names),
        "test_names_sha256": _sha256_names(test_names),
        "resolution": int(getattr(args_like, "resolution", -1)),
        "input_dynamic_range": str(getattr(args_like, "input_dynamic_range", "")),
        "radiometric_mode": str(getattr(args_like, "radiometric_mode", "")),
        "use_validity_mask": bool(getattr(args_like, "use_validity_mask", False)),
        "ply_path": str(scene_info.ply_path),
    }
    return scene_info, audit


def _tensorize_bank_payload(banks: Mapping[str, BandAppearanceBank]) -> Dict[str, Dict[str, torch.Tensor]]:
    return {
        band: {
            "features_dc": _as_tensor_payload(bank.features_dc),
            "features_rest": _as_tensor_payload(bank.features_rest),
        }
        for band, bank in banks.items()
    }


def save_unified_checkpoint(
    path: str | Path,
    state: JointMultispectralState,
    config: Mapping[str, object],
    audit: Mapping[str, object],
) -> Tuple[Path, Path]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "method": "E4b_geometry_locked_joint_multispectral_banks",
        "checkpoint_created_at": utc_now_iso(),
        "code_version_or_git_commit": get_git_commit(),
        "source_rgb_checkpoint": str(config.get("source_rgb_checkpoint", "")),
        "rgb_checkpoint_iteration": int(config.get("rgb_checkpoint_iteration", -1)),
        "bands": list(state.bands),
        "active_sh_degree": int(state.gaussians.active_sh_degree),
        "max_sh_degree": int(state.gaussians.max_sh_degree),
        "shared_geometry": {
            "xyz": _as_tensor_payload(state.gaussians._xyz),
            "scaling": _as_tensor_payload(state.gaussians._scaling),
            "rotation": _as_tensor_payload(state.gaussians._rotation),
            "opacity": _as_tensor_payload(state.gaussians._opacity),
            "spatial_lr_scale": float(state.gaussians.spatial_lr_scale),
            "max_radii2D": _as_tensor_payload(state.gaussians.max_radii2D),
        },
        "banks": _tensorize_bank_payload(state.banks),
        "optimizer_states": {band: opt.state_dict() for band, opt in state.optimizers.items()},
        "updates_per_band": {band: int(v) for band, v in state.updates_per_band.items()},
        "joint_total_updates": int(sum(state.updates_per_band.values())),
        "per_band_updates_target": int(config.get("per_band_updates_target", -1)),
        "band_sampling": str(config.get("band_sampling", "")),
        "band_sampling_effective_order": list(config.get("band_sampling_effective_order", [])),
        "bank_init": str(config.get("bank_init", "")),
        "training_config": dict(config),
        "audit": dict(audit),
    }
    torch.save(payload, str(path))
    audit_path = path.parent / "joint_training_audit.json"
    audit_json = dict(audit)
    audit_json.update(
        {
            "schema_version": 1,
            "method": payload["method"],
            "checkpoint_path": str(path),
            "checkpoint_created_at": payload["checkpoint_created_at"],
            "code_version_or_git_commit": payload["code_version_or_git_commit"],
            "bands": payload["bands"],
            "updates_per_band": payload["updates_per_band"],
            "joint_total_updates": payload["joint_total_updates"],
            "per_band_updates_target": payload["per_band_updates_target"],
            "band_sampling": payload["band_sampling"],
            "band_sampling_effective_order": payload["band_sampling_effective_order"],
            "bank_init": payload["bank_init"],
        }
    )
    audit_path.write_text(json.dumps(_json_safe(audit_json), indent=2), encoding="utf-8")
    return path, audit_path


def load_unified_checkpoint(path: str | Path, device: str = "cuda") -> Dict[str, object]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing joint checkpoint: {path}")
    payload = torch.load(str(path), map_location=device)
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported joint checkpoint schema_version={payload.get('schema_version')}")
    if str(payload.get("method", "")) != "E4b_geometry_locked_joint_multispectral_banks":
        raise ValueError(f"Unsupported joint checkpoint method={payload.get('method')}")
    return payload


def _json_safe(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return list(obj.shape)
    return obj


def _make_export_gaussian(payload: Mapping[str, object], band: str, device: str) -> GaussianModel:
    gaussians = GaussianModel(int(payload["max_sh_degree"]))
    geom = payload["shared_geometry"]
    bank = payload["banks"][band]
    gaussians.active_sh_degree = int(payload["active_sh_degree"])
    gaussians._xyz = _clone_to_device(geom["xyz"], device, requires_grad=False)
    gaussians._scaling = _clone_to_device(geom["scaling"], device, requires_grad=False)
    gaussians._rotation = _clone_to_device(geom["rotation"], device, requires_grad=False)
    gaussians._opacity = _clone_to_device(geom["opacity"], device, requires_grad=False)
    gaussians._features_dc = _clone_to_device(bank["features_dc"], device, requires_grad=False)
    gaussians._features_rest = _clone_to_device(bank["features_rest"], device, requires_grad=False)
    gaussians.max_radii2D = geom["max_radii2D"].detach().clone().to(device)
    gaussians.spatial_lr_scale = float(geom.get("spatial_lr_scale", 0.0))
    return gaussians


def write_cfg_args(
    out_model_dir: str | Path,
    scene_root: str | Path,
    band: str,
    sh_degree: int,
    resolution: int,
    input_dynamic_range: str,
    radiometric_mode: str,
    rectified_root: str | Path,
) -> None:
    ns = Namespace(
        sh_degree=int(sh_degree),
        source_path=str(Path(scene_root).resolve()),
        model_path=str(Path(out_model_dir).resolve()),
        images="images",
        depths="",
        resolution=int(resolution),
        white_background=False,
        train_test_exp=False,
        data_device="cuda",
        eval=True,
        modality_kind="band",
        target_band=str(band),
        single_band_mode=True,
        single_band_replicate_to_rgb=True,
        input_dynamic_range=str(input_dynamic_range),
        radiometric_mode=str(radiometric_mode),
        rectified_root=str(Path(rectified_root).resolve()),
        require_rectified_band_scene=True,
        use_validity_mask=True,
        rectification_method="minima_assisted_global_homography",
        reset_appearance_features=True,
        freeze_geometry=True,
        freeze_opacity=True,
        tied_scalar_carrier=True,
        stage2_mode="joint_multispectral_export_view",
        rectification_backend="minima",
    )
    Path(out_model_dir, "cfg_args").write_text(str(ns), encoding="utf-8")


def write_cameras_json(out_model_dir: str | Path, scene_info) -> None:
    camlist = []
    if scene_info.test_cameras:
        camlist.extend(scene_info.test_cameras)
    if scene_info.train_cameras:
        camlist.extend(scene_info.train_cameras)
    json_cams = [camera_to_JSON(idx, cam) for idx, cam in enumerate(camlist)]
    Path(out_model_dir, "cameras.json").write_text(json.dumps(json_cams, indent=2), encoding="utf-8")


def export_band_model(
    joint_payload: Mapping[str, object],
    band: str,
    out_model_dir: str | Path,
    scene_root: str | Path,
    iteration: int,
    resolution: int,
    input_dynamic_range: str,
    radiometric_mode: str,
    device: str = "cuda",
) -> Dict[str, object]:
    if band not in joint_payload["banks"]:
        raise KeyError(f"Band {band!r} not present in joint checkpoint")
    out_model_dir = Path(out_model_dir)
    scene_root = Path(scene_root)
    point_cloud_dir = out_model_dir / "point_cloud" / f"iteration_{int(iteration)}"
    point_cloud_dir.mkdir(parents=True, exist_ok=True)

    gaussians = _make_export_gaussian(joint_payload, band, device=device)
    ply_path = point_cloud_dir / "point_cloud.ply"
    gaussians.save_ply(str(ply_path))

    args_like = make_band_dataset_args(
        scene_root=scene_root,
        band=band,
        model_path=out_model_dir,
        resolution=resolution,
        input_dynamic_range=input_dynamic_range,
        radiometric_mode=radiometric_mode,
    )
    scene_info, camera_audit = load_band_scene_info_without_scene(scene_root, band, args_like)
    write_cameras_json(out_model_dir, scene_info)
    write_cfg_args(
        out_model_dir=out_model_dir,
        scene_root=scene_root,
        band=band,
        sh_degree=int(joint_payload["max_sh_degree"]),
        resolution=resolution,
        input_dynamic_range=input_dynamic_range,
        radiometric_mode=radiometric_mode,
        rectified_root=scene_root.parent,
    )

    input_src = Path(scene_info.ply_path)
    if input_src.exists():
        shutil.copy2(input_src, out_model_dir / "input.ply")

    exposure_candidates = [scene_root / "exposure.json", scene_root.parent / "exposure.json"]
    for src in exposure_candidates:
        if src.exists():
            shutil.copy2(src, out_model_dir / "exposure.json")
            break

    return {
        "band": band,
        "out_model_dir": str(out_model_dir),
        "scene_root": str(scene_root),
        "iteration": int(iteration),
        "point_cloud_ply": str(ply_path),
        "cfg_args_written": (out_model_dir / "cfg_args").exists(),
        "cameras_json_written": (out_model_dir / "cameras.json").exists(),
        "input_ply_written": (out_model_dir / "input.ply").exists(),
        "render_compatible": True,
        "camera_audit": camera_audit,
    }


def _stack_fields(vertex, names: Sequence[str]) -> np.ndarray:
    if not names:
        return np.zeros((len(vertex), 0), dtype=np.float32)
    return np.stack([np.asarray(vertex[name], dtype=np.float32) for name in names], axis=1)


def _read_export_vertex(model_dir: str | Path, iteration: int):
    ply_path = Path(model_dir) / "point_cloud" / f"iteration_{int(iteration)}" / "point_cloud.ply"
    ply = PlyData.read(str(ply_path))
    return ply.elements[0].data


def compute_export_topology_audit(
    model_dirs: Mapping[str, str | Path],
    iteration: int,
    tol: float = 1e-6,
) -> Dict[str, object]:
    vertices = {band: _read_export_vertex(path, iteration) for band, path in model_dirs.items()}
    bands = list(vertices.keys())
    ref_band = bands[0]
    ref = vertices[ref_band]
    scale_names = sorted([n for n in ref.dtype.names if n.startswith("scale_")], key=lambda x: int(x.split("_")[-1]))
    rot_names = sorted([n for n in ref.dtype.names if n.startswith("rot_")], key=lambda x: int(x.split("_")[-1]))
    ref_xyz = _stack_fields(ref, ["x", "y", "z"])
    ref_scale = _stack_fields(ref, scale_names)
    ref_rot = _stack_fields(ref, rot_names)
    ref_opacity = np.asarray(ref["opacity"], dtype=np.float32)

    max_diffs = {}
    same_num = True
    same_xyz = True
    same_scaling = True
    same_rotation = True
    same_opacity = True
    for band in bands[1:]:
        cur = vertices[band]
        same_num = same_num and (len(cur) == len(ref))
        cur_scale_names = sorted([n for n in cur.dtype.names if n.startswith("scale_")], key=lambda x: int(x.split("_")[-1]))
        cur_rot_names = sorted([n for n in cur.dtype.names if n.startswith("rot_")], key=lambda x: int(x.split("_")[-1]))
        diffs = {
            "xyz": float(np.max(np.abs(ref_xyz - _stack_fields(cur, ["x", "y", "z"])))),
            "scaling": float(np.max(np.abs(ref_scale - _stack_fields(cur, cur_scale_names)))),
            "rotation": float(np.max(np.abs(ref_rot - _stack_fields(cur, cur_rot_names)))),
            "opacity": float(np.max(np.abs(ref_opacity - np.asarray(cur["opacity"], dtype=np.float32)))),
        }
        max_diffs[band] = diffs
        same_xyz = same_xyz and diffs["xyz"] <= tol
        same_scaling = same_scaling and diffs["scaling"] <= tol
        same_rotation = same_rotation and diffs["rotation"] <= tol
        same_opacity = same_opacity and diffs["opacity"] <= tol

    return {
        "same_num_gaussians_exported": bool(same_num),
        "same_xyz_exported": bool(same_xyz),
        "same_scaling_exported": bool(same_scaling),
        "same_rotation_exported": bool(same_rotation),
        "same_opacity_exported": bool(same_opacity),
        "topology_identity_exported": bool(same_num and same_xyz and same_scaling and same_rotation and same_opacity),
        "reference_band": ref_band,
        "max_diffs_by_band_vs_reference": max_diffs,
    }
