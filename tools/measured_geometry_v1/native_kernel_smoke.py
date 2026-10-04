"""Small synthetic native-kernel checks for a supervised GPU qualification.

Import is CPU-only. These functions are not launchers or GPU authorization;
the operator must first pass the campaign qualification gate and enforce its
deadline. No datasets, model checkpoints, GCPs or formal metrics are read.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from .live_packet_adapter import graphdeco_moments, gsplat_moments
from .native_moments import RAW_NAMES, packet_from_reference
from .raster_weight_reference import scalar_moments

CASES = (
    ("gentle_odd", 33, 25, [.2, .35, .15]),
    ("gentle_even", 34, 26, [.2, .35, .15]),
    ("exclusive_stop", 33, 25, [.99999, .99999, .5]),
    ("below_alpha_cutoff", 33, 25, [.001, .002, .003]),
)


def compare_center(raw, opacity, z, renderer, px, py):
    """Synthetic raw-only fp32 roundoff bound, not a formal scoring tolerance."""
    expected = scalar_moments(z, opacity, renderer=renderer)
    checks = []
    for name in RAW_NAMES:
        actual = float(raw[name][py, px])
        wanted = float(expected["raw"][name].item())
        # FMA and separate mul/add may differ. Inputs are <=3 positive layers
        # at a constructed zero-splat-offset pixel; no fitted pixel/threshold.
        bound = 16 * np.finfo(np.float32).eps * abs(wanted) + np.finfo(np.float32).tiny
        error = abs(actual - wanted)
        if not math.isfinite(actual) or error > bound:
            raise ValueError(f"Native center accumulator mismatch {name}: {actual} vs {wanted}, bound={bound}")
        checks.append({"tensor": name, "actual": actual, "expected": wanted,
                       "abs_error": error, "raw_synthetic_fp32_bound": float(bound)})
    return {"checks": checks, "native_reference": {k: v for k, v in expected.items() if k != "raw"}}


def _fixture(width, height, opacity_values):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("An authorized and qualified CUDA runtime is required")
    z = torch.tensor([2., 5., 9.], dtype=torch.float32, device="cuda")
    px, py = width // 2, height // 2
    fx, fy = 40., 43.
    ray = ((px + .5 - width / 2) / fx, (py + .5 - height / 2) / fy)
    means = torch.stack((z * ray[0], z * ray[1], z), dim=1)
    quats = torch.tensor([[1., 0., 0., 0.]] * 3, dtype=torch.float32, device="cuda")
    scales = torch.full((3, 3), .07, dtype=torch.float32, device="cuda")
    logits = torch.logit(torch.tensor(opacity_values, dtype=torch.float32, device="cuda")).reshape(3, 1)
    return {"z": z, "px": px, "py": py, "fx": fx, "fy": fy, "means": means,
            "quats": quats, "scales": scales, "logits": logits}


def _check_case(raw, f, renderer, reference, name, width, height):
    import torch

    actual_opacity = torch.sigmoid(f["logits"]).flatten().detach().cpu().numpy()
    checked = compare_center(raw, actual_opacity, f["z"].detach().cpu().numpy(), renderer, f["px"], f["py"])
    packet, numeric = packet_from_reference(raw, reference)
    assert numeric["passed"] is True
    return {"case": name, "width": width, "height": height, "center": [f["px"], f["py"]],
            "status": "PASS", "accumulators": checked,
            "frozen_packet_ref_passed": True,
            "valid_pixel_count": int(np.count_nonzero(packet["metric_depth_valid_mask"]))}


def gsplat_synthetic(*, reference):
    import torch
    from nerfstudio.cameras.cameras import Cameras, CameraType
    from mmsplat.mmsplat_model import rasterization, get_viewmat

    rows = []
    for name, width, height, opacity in CASES:
        f = _fixture(width, height, opacity)
        c2w = torch.tensor([[[1., 0., 0., 0.], [0., -1., 0., 0.], [0., 0., -1., 0.]]], device="cuda")
        camera = Cameras(camera_to_worlds=c2w, fx=f["fx"], fy=f["fy"],
                         cx=width/2, cy=height/2, width=width, height=height,
                         camera_type=CameraType.PERSPECTIVE)
        model = SimpleNamespace(training=False, _get_downscale_factor=lambda: 1,
            gauss_params={"means": f["means"], "quats": f["quats"],
                          "scales": f["scales"].log(), "opacities": f["logits"]},
            strategy=SimpleNamespace(absgrad=True),
            config=SimpleNamespace(opacity_correction_flag=False, rasterize_mode="classic",
                camera_optimizer_rgb=SimpleNamespace(mode="off"), camera_optimizer_ms=SimpleNamespace(mode="off")))
        raw = gsplat_moments(native_rasterization=rasterization, native_get_viewmat=get_viewmat,
                            model=model, camera=camera)
        rows.append(_check_case(raw, f, "gsplat_1_4_0_exclusive_stop_v1", reference, name, width, height))
    return _report("gsplat", rows)


def graphdeco_synthetic(*, native_render, native_projection, reference):
    import torch

    rows = []
    for name, width, height, opacity in CASES:
        f = _fixture(width, height, opacity)
        fovx, fovy = 2 * math.atan(width / (2 * f["fx"])), 2 * math.atan(height / (2 * f["fy"]))
        projection = native_projection(znear=.01, zfar=100., fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
        camera = SimpleNamespace(FoVx=fovx, FoVy=fovy, image_height=height, image_width=width,
            world_view_transform=torch.eye(4, device="cuda"), full_proj_transform=projection,
            camera_center=torch.zeros(3, device="cuda"))
        cloud = SimpleNamespace(get_xyz=f["means"], get_opacity=torch.sigmoid(f["logits"]),
            get_scaling=f["scales"], get_rotation=f["quats"],
            get_features=torch.zeros((3, 1, 3), device="cuda"), active_sh_degree=0, max_sh_degree=0)
        pipeline = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False,
                                   debug=False, antialiasing=False)
        raw = graphdeco_moments(native_render=native_render, camera=camera, gaussians=cloud,
                               pipeline=pipeline, background=torch.zeros(3, device="cuda"))
        rows.append(_check_case(raw, f, "umgs_graphdeco_exclusive_stop_v1", reference, name, width, height))
    return _report("graphdeco", rows)


def _report(renderer, rows):
    return {"status": "PASS_SYNTHETIC_NATIVE_KERNEL_ONLY", "renderer": renderer, "tests": rows,
            "training_started": False, "formal_metrics_generated": False,
            "checkpoint_loaded": False, "method_runtime_qualified": False,
            "remaining": ["actual camera/normalization binding", "model save/reload",
                          "real packet/reference validation", "full scoring chain"]}
