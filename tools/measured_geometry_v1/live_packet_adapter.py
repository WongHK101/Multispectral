"""Evaluation-only native render adapters. No launcher or checkpoint loader.

The caller must authenticate source/binary, model, actual camera, normalization,
and user GPU authorization before calling. Importing this module uses no torch
or CUDA. Helpers return model-unit raw accumulators, never surveyed metres.
"""
from __future__ import annotations

from .native_moments import graphdeco_live_outputs, gsplat_live_feature_outputs


def _numpy(tensor):
    return tensor.detach().cpu().numpy()


def graphdeco_moments(*, native_render, camera, gaussians, pipeline, background):
    """Use the existing opt-in packet and same-call raw inverse-depth sum."""
    if not gaussians.get_xyz.is_cuda or not background.is_cuda:
        raise ValueError("Live Graphdeco export requires the qualified CUDA runtime")
    import torch

    with torch.no_grad():
        output = native_render(camera, gaussians, pipeline, background,
                               separate_sh=False, use_trained_exp=False,
                               return_expected_camera_z_packet=True,
                               opacity_epsilon=1e-6, variance_clamp_tolerance=1e-6)
    raw = graphdeco_live_outputs(_numpy(output["expected_camera_z_packet"]), _numpy(output["depth"]))
    # Raw A/M1/M2/H are independent of the old opt-in derived valid/variance planes.
    return raw


def validate_ms_export_config(model):
    if model.training or model._get_downscale_factor() != 1:
        raise ValueError("Export requires eval mode at the fixed native camera grid")
    if model.config.opacity_correction_flag:
        raise ValueError("Channel-dependent opacity requires a separately reviewed adapter")
    if model.config.camera_optimizer_rgb.mode != "off" or model.config.camera_optimizer_ms.mode != "off":
        raise ValueError("Optimized cameras are outside the shared-pose comparison")
    if model.config.rasterize_mode not in {"classic", "antialiased"}:
        raise ValueError("Unknown native rasterize mode")


def gsplat_moments(*, native_rasterization, native_get_viewmat, model, camera):
    """Render [1,z,z*z,1/z] with the model's unchanged native raster settings.

    native_get_viewmat is the pinned method's OpenGL->OpenCV camera helper.
    K remains its corner-origin K; conversion to array indices is separate.
    """
    validate_ms_export_config(model)
    if not model.gauss_params["means"].is_cuda:
        raise ValueError("Live gsplat export requires the qualified CUDA runtime")
    import torch
    from nerfstudio.cameras.cameras import CameraType

    if camera.shape != (1,) or camera.width.numel() != 1 or camera.height.numel() != 1:
        raise ValueError("Expected exactly one full-frame packet camera")
    if torch.any(camera.camera_type != CameraType.PERSPECTIVE.value):
        raise ValueError("Expected a PINHOLE perspective packet camera")
    if camera.distortion_params is not None and torch.any(camera.distortion_params != 0):
        raise ValueError("Expected already prepared undistorted camera")
    with torch.no_grad():
        params = model.gauss_params
        device = params["means"].device
        viewmat = native_get_viewmat(camera.camera_to_worlds.to(device))
        k = camera.get_intrinsics_matrices().to(device)
        width, height = int(camera.width.item()), int(camera.height.item())
        z = params["means"] @ viewmat[0, 2, :3] + viewmat[0, 2, 3]
        if not torch.isfinite(z).all():
            raise ValueError("Nonfinite native camera-z")
        # Culled z values must not produce NaN/Inf features before culling.
        visible_z = (z >= .01) & (z <= 1e10)
        safe_z = torch.where(visible_z, z, torch.ones_like(z))
        features = torch.stack((torch.ones_like(z), safe_z, safe_z * safe_z, 1 / safe_z), dim=-1)
        features = torch.where(visible_z[:, None], features, torch.zeros_like(features))
        rendered, alpha, _ = native_rasterization(
            means=params["means"], quats=params["quats"], scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]).squeeze(-1), colors=features,
            viewmats=viewmat, Ks=k, width=width, height=height, packed=False,
            near_plane=.01, far_plane=1e10, render_mode="RGB", sh_degree=None,
            sparse_grad=False, absgrad=model.strategy.absgrad,
            rasterize_mode=model.config.rasterize_mode,
            backgrounds=torch.zeros((1, 4), dtype=features.dtype, device=device))
    return gsplat_live_feature_outputs(_numpy(rendered[0]), _numpy(alpha[0]))
