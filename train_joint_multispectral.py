import argparse
import json
import math
import random
from collections import defaultdict, deque
from pathlib import Path
from types import SimpleNamespace


def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {v}")


def _parse_bands(text: str):
    return [x.strip() for x in str(text).replace(",", " ").split() if x.strip()]


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Train an E4b geometry-locked joint multispectral appearance representation."
    )
    ap.add_argument("--rgb_checkpoint", required=True)
    ap.add_argument("--rectified_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--bands", default="G,R,RE,NIR")
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--band_res", type=int, default=8)
    ap.add_argument("--per_band_updates", type=int, default=30000)
    ap.add_argument("--feature_lr", type=float, default=0.001)
    ap.add_argument("--input_dynamic_range", default="uint16", choices=["uint8", "uint16", "float"])
    ap.add_argument("--radiometric_mode", default="exposure_normalized", choices=["raw_dn", "exposure_normalized", "reflectance_ready_stub"])
    ap.add_argument("--use_validity_mask", type=_str2bool, nargs="?", const=True, default=True)
    ap.add_argument("--bank_init", default="zero", choices=["zero", "rgb_tied"])
    ap.add_argument("--band_sampling", default="round_robin", choices=["round_robin"])
    ap.add_argument("--view_sampling", default="shuffled_cycle", choices=["shuffled_cycle", "cyclic"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--save_every", type=int, default=0)
    ap.add_argument("--export_after_train", type=_str2bool, nargs="?", const=True, default=False)
    ap.add_argument("--export_iteration", type=int, default=60000)
    ap.add_argument("--render_smoke", type=_str2bool, nargs="?", const=True, default=False)
    ap.add_argument("--max_nan_steps_per_band", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    return ap


def _masked_l1_loss(network_output, gt, mask, eps=1e-6):
    import torch

    diff = torch.abs(network_output - gt)
    valid_mask = mask
    if valid_mask.ndim == 2:
        valid_mask = valid_mask.unsqueeze(0)
    if valid_mask.ndim != 3:
        raise ValueError(f"Expected mask with shape [1,H,W] or [H,W], got {tuple(valid_mask.shape)}")
    if valid_mask.shape[0] == 1 and diff.shape[0] > 1:
        valid_mask = valid_mask.expand(diff.shape[0], -1, -1)
    valid_mask = valid_mask.to(diff.device, dtype=diff.dtype).clamp(0.0, 1.0)
    denom = valid_mask.sum().clamp_min(eps) * float(diff.shape[0])
    return (diff * valid_mask).sum() / denom


class CameraSampler:
    def __init__(self, cameras, seed: int, mode: str):
        self.cameras = list(cameras)
        if not self.cameras:
            raise ValueError("Cannot sample from an empty camera list")
        self.mode = mode
        self.rng = random.Random(int(seed))
        self.order = list(range(len(self.cameras)))
        self.cursor = 0
        if self.mode == "shuffled_cycle":
            self.rng.shuffle(self.order)

    def next(self):
        if self.cursor >= len(self.order):
            self.cursor = 0
            if self.mode == "shuffled_cycle":
                self.rng.shuffle(self.order)
        idx = self.order[self.cursor]
        self.cursor += 1
        return self.cameras[idx], idx


def _load_rgb_features_for_bank_init(rgb_checkpoint, device):
    from utils.joint_multispectral_utils import torch_load_trusted

    model_params, _ = torch_load_trusted(rgb_checkpoint, map_location=device)
    return {
        "features_dc": model_params[2].detach().clone().to(device),
        "features_rest": model_params[3].detach().clone().to(device),
    }


def _mean_or_nan(values):
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _make_loss_audit(loss_windows, final_losses, nan_steps):
    bands = list(loss_windows.keys())
    return {
        "loss_mean_last_k_by_band": {band: _mean_or_nan(list(vals)) for band, vals in loss_windows.items()},
        "loss_final_by_band": {band: float(final_losses.get(band, float("nan"))) for band in bands},
        "num_nan_steps_by_band": {band: int(nan_steps.get(band, 0)) for band in bands},
    }


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    import numpy as np
    import torch
    from tqdm import tqdm

    from gaussian_renderer import render
    from utils.loss_utils import l1_loss
    from utils.joint_multispectral_utils import (
        JointMultispectralState,
        assert_joint_invariants,
        compute_geometry_drift,
        export_band_model,
        get_git_commit,
        init_band_banks,
        load_band_cameras_without_scene,
        load_rgb_checkpoint_geometry_only,
        make_band_dataset_args,
        make_band_optimizers,
        project_active_bank_to_tied_scalar,
        resolve_band_scene_root,
        save_unified_checkpoint,
        set_active_bank,
    )

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if args.device.startswith("cuda") and torch.cuda.is_available():
        cuda_device = torch.device(args.device)
        torch.cuda.set_device(0 if cuda_device.index is None else cuda_device.index)

    bands = _parse_bands(args.bands)
    if not bands:
        raise ValueError("At least one band is required")
    out_dir = Path(args.out_dir).resolve()
    joint_dir = out_dir / "Model_MSJoint"
    joint_dir.mkdir(parents=True, exist_ok=True)
    rectified_root = Path(args.rectified_root).resolve()
    rgb_checkpoint = Path(args.rgb_checkpoint).resolve()

    gaussians, geometry_ref, rgb_meta = load_rgb_checkpoint_geometry_only(
        rgb_checkpoint=rgb_checkpoint,
        sh_degree=int(args.sh_degree),
        device=args.device,
    )
    rgb_feature_shapes = {
        "features_dc": rgb_meta["features_dc_shape"],
        "features_rest": rgb_meta["features_rest_shape"],
    }
    rgb_features = None
    if args.bank_init == "rgb_tied":
        rgb_features = _load_rgb_features_for_bank_init(rgb_checkpoint, args.device)
    banks = init_band_banks(
        rgb_feature_shapes=rgb_feature_shapes,
        bands=bands,
        bank_init=args.bank_init,
        rgb_features=rgb_features,
        device=args.device,
    )
    optimizers = make_band_optimizers(banks, feature_lr=float(args.feature_lr))
    updates_per_band = {band: 0 for band in bands}
    state = JointMultispectralState(
        gaussians=gaussians,
        bands=bands,
        banks=banks,
        optimizers=optimizers,
        updates_per_band=updates_per_band,
        geometry_ref=geometry_ref,
    )

    camera_sets = {}
    camera_audit = {}
    samplers = {}
    for band_idx, band in enumerate(bands):
        scene_root = resolve_band_scene_root(rectified_root, band)
        args_like = make_band_dataset_args(
            scene_root=scene_root,
            band=band,
            model_path=out_dir / f"Model_{band}",
            resolution=int(args.band_res),
            input_dynamic_range=args.input_dynamic_range,
            radiometric_mode=args.radiometric_mode,
            use_validity_mask=bool(args.use_validity_mask),
        )
        train_cameras, test_cameras, _scene_info, audit = load_band_cameras_without_scene(
            scene_root=scene_root,
            band=band,
            args_like=args_like,
            shuffle=False,
            load_test=False,
        )
        camera_sets[band] = {"train": train_cameras, "test": test_cameras}
        camera_audit[band] = audit
        samplers[band] = CameraSampler(train_cameras, seed=int(args.seed) + band_idx * 1009, mode=args.view_sampling)

    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=args.device)
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)

    total_target = int(args.per_band_updates) * len(bands)
    progress = tqdm(total=total_target, desc="E4b joint training", disable=bool(args.quiet))
    loss_windows = {band: deque(maxlen=100) for band in bands}
    final_losses = {}
    nan_steps = defaultdict(int)
    last_active_band = ""
    joint_step = 0

    while any(updates_per_band[band] < int(args.per_band_updates) for band in bands):
        for band in bands:
            if updates_per_band[band] >= int(args.per_band_updates):
                continue
            joint_step += 1
            set_active_bank(gaussians, banks, band)
            state.active_band = band
            last_active_band = band
            viewpoint_cam, _view_idx = samplers[band].next()

            render_pkg = render(viewpoint_cam, gaussians, pipe, background, use_trained_exp=False, separate_sh=False)
            image = render_pkg["render"]
            if viewpoint_cam.alpha_mask is not None:
                image = image * viewpoint_cam.alpha_mask.cuda()
            gt_image = viewpoint_cam.original_image.cuda()
            validity_mask = getattr(viewpoint_cam, "validity_mask", None)
            if bool(args.use_validity_mask) and validity_mask is not None:
                loss = _masked_l1_loss(image, gt_image, validity_mask.cuda())
            else:
                loss = l1_loss(image, gt_image)

            if not torch.isfinite(loss).all():
                nan_steps[band] += 1
                optimizers[band].zero_grad(set_to_none=True)
                if nan_steps[band] > int(args.max_nan_steps_per_band):
                    raise RuntimeError(f"Too many non-finite loss steps for band={band}: {nan_steps[band]}")
                continue

            loss.backward()
            optimizers[band].step()
            optimizers[band].zero_grad(set_to_none=True)
            project_active_bank_to_tied_scalar(banks[band])

            loss_value = float(loss.detach().cpu().item())
            loss_windows[band].append(loss_value)
            final_losses[band] = loss_value
            updates_per_band[band] += 1
            progress.update(1)
            if not args.quiet:
                progress.set_postfix({f"{b}_u": updates_per_band[b] for b in bands})

            if int(args.save_every) > 0 and joint_step % int(args.save_every) == 0:
                drift = compute_geometry_drift(gaussians, geometry_ref)
                audit = _make_audit(args, bands, rgb_meta, camera_audit, drift, last_active_band, loss_windows, final_losses, nan_steps)
                save_unified_checkpoint(joint_dir / f"joint_multispectral_checkpoint_step{joint_step}.pth", state, audit["training_config"], audit)

    progress.close()
    drift = compute_geometry_drift(gaussians, geometry_ref)
    audit = _make_audit(args, bands, rgb_meta, camera_audit, drift, last_active_band, loss_windows, final_losses, nan_steps)
    assert_joint_invariants(audit)
    ckpt_path, audit_path = save_unified_checkpoint(
        joint_dir / "joint_multispectral_checkpoint.pth",
        state,
        audit["training_config"],
        audit,
    )
    print(f"[E4b] Saved unified checkpoint: {ckpt_path}")
    print(f"[E4b] Saved audit: {audit_path}")

    if bool(args.export_after_train):
        from utils.joint_multispectral_utils import compute_export_topology_audit, load_unified_checkpoint

        del camera_sets
        del samplers
        del state
        del banks
        del optimizers
        del gaussians
        del geometry_ref
        torch.cuda.empty_cache()
        # Keep the authoritative multi-band checkpoint on CPU during export.
        # Loading the full joint checkpoint onto GPU duplicates all band banks
        # and can OOM large official scenes; export_band_model only moves the
        # active band view to GPU.
        payload = load_unified_checkpoint(ckpt_path, device="cpu")
        export_root = out_dir
        export_fragments = {}
        for band in bands:
            scene_root = resolve_band_scene_root(rectified_root, band)
            fragment = export_band_model(
                joint_payload=payload,
                band=band,
                out_model_dir=export_root / f"Model_{band}",
                scene_root=scene_root,
                iteration=int(args.export_iteration),
                resolution=int(args.band_res),
                input_dynamic_range=args.input_dynamic_range,
                radiometric_mode=args.radiometric_mode,
                device=args.device,
            )
            export_fragments[band] = fragment
            torch.cuda.empty_cache()
        topology_export = compute_export_topology_audit(
            {band: export_root / f"Model_{band}" for band in bands},
            iteration=int(args.export_iteration),
        )
        export_audit = {
            "method": "E4b_geometry_locked_joint_multispectral_banks",
            "joint_checkpoint": str(ckpt_path),
            "export_iteration": int(args.export_iteration),
            "export_iteration_note": "Aligned with E3 band_iter evaluation iteration; joint_total_updates remains per_band_updates * number_of_bands.",
            "exported_band_models": export_fragments,
            "render_smoke_requested": bool(args.render_smoke),
            "render_smoke_passed": None,
            **topology_export,
        }
        export_dir = out_dir / "Model_MSJoint_exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        _write_json(export_dir / "export_audit.json", export_audit)
        print(f"[E4b] Exported per-band views under: {export_root}")


def _make_audit(args, bands, rgb_meta, camera_audit, drift, last_active_band, loss_windows, final_losses, nan_steps):
    from utils.joint_multispectral_utils import get_git_commit

    training_config = {
        "source_rgb_checkpoint": str(Path(args.rgb_checkpoint).resolve()),
        "rgb_checkpoint_iteration": int(rgb_meta["checkpoint_rgb_iter"]),
        "rectified_root": str(Path(args.rectified_root).resolve()),
        "out_dir": str(Path(args.out_dir).resolve()),
        "bands": list(bands),
        "sh_degree": int(args.sh_degree),
        "band_res": int(args.band_res),
        "per_band_updates_target": int(args.per_band_updates),
        "joint_total_updates_target": int(args.per_band_updates) * len(bands),
        "feature_lr_dc": float(args.feature_lr),
        "feature_lr_rest": float(args.feature_lr) / 20.0,
        "input_dynamic_range": str(args.input_dynamic_range),
        "radiometric_mode": str(args.radiometric_mode),
        "use_validity_mask": bool(args.use_validity_mask),
        "bank_init": str(args.bank_init),
        "band_sampling": str(args.band_sampling),
        "band_sampling_effective_order": list(bands),
        "view_sampling": str(args.view_sampling),
        "background": [0.0, 0.0, 0.0],
        "lambda_dssim": 0.0,
        "use_trained_exp": False,
        "pipe": {
            "convert_SHs_python": False,
            "compute_cov3D_python": False,
            "debug": False,
            "antialiasing": False,
        },
        "seed": int(args.seed),
        "export_iteration": int(args.export_iteration),
        "export_iteration_note": "Aligned with E3 band_iter evaluation iteration; joint_total_updates remains per_band_updates * number_of_bands.",
        "code_version_or_git_commit": get_git_commit(),
    }
    loss_audit = _make_loss_audit(loss_windows, final_losses, nan_steps)
    audit = {
        "authoritative_artifact": True,
        "export_views_only": True,
        "shared_geometry": True,
        "freeze_geometry": True,
        "freeze_opacity": True,
        "appearance_only_update": True,
        "active_bank_switching": True,
        "active_band_last_step": str(last_active_band),
        "optimizers_per_band": True,
        "num_gaussians": int(rgb_meta["num_gaussians"]),
        "rgb_checkpoint_meta": dict(rgb_meta),
        "camera_audit": camera_audit,
        "training_config": training_config,
        **drift,
        **loss_audit,
    }
    return audit


if __name__ == "__main__":
    main()
