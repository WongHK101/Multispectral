"""Array wiring for NEW native renders, not a legacy packet conversion tool.

No imports of training/CUDA code and no file loading. The caller must prove
same-call camera, source/kernel and unit identities before invoking the frozen
reference packet builder. Synthetic tests cannot qualify a CUDA renderer.
"""
from __future__ import annotations

import numpy as np

from .contracts import FLOAT_TENSORS
from .camera_bridge import source_unit_accumulators

RAW_NAMES = FLOAT_TENSORS[:4]


def validate_raw_moments(values):
    if set(values) != set(RAW_NAMES):
        raise ValueError("Require explicit A, M1, M2, H from the live native render")
    arrays = {k: np.asarray(values[k]) for k in RAW_NAMES}
    shape = arrays[RAW_NAMES[0]].shape
    if (len(shape) != 2 or min(shape) < 1 or any(
            a.shape != shape or a.dtype != np.dtype("float32") or not np.isfinite(a).all()
            for a in arrays.values())):
        raise ValueError("Raw moments must be finite same-shape 2D float32 arrays")
    if any(np.any(a < 0) for a in arrays.values()) or np.any(arrays[RAW_NAMES[0]] > 1+1e-5):
        raise ValueError("Invalid raw alpha/moment range")
    zero = arrays[RAW_NAMES[0]] == 0
    if any(np.any(a[zero] != 0) for a in list(arrays.values())[1:]):
        raise ValueError("Nonzero moment with zero support")
    return {k: a.copy() for k, a in arrays.items()}


def graphdeco_live_outputs(packet_six_planes, raw_inverse_sum):
    """Pair six-plane opt-in output with SAME CALL's unnormalized invdepth H.

    UMGS forward.cu accumulates H=sum(weight/z) next to A/M1/M2. Inverting
    expected depth, or using a separately loaded historical depth, is not this
    contract. Legacy derived valid/variance planes are deliberately not reused.
    """
    planes = np.asarray(packet_six_planes)
    h = np.asarray(raw_inverse_sum)
    if planes.ndim != 3 or planes.shape[0] != 6 or planes.dtype != np.float32:
        raise ValueError("Expected live native (6,H,W) float32 output")
    if h.shape == (1, *planes.shape[1:]):
        h = h[0]
    return validate_raw_moments(dict(zip(RAW_NAMES, (planes[0], planes[1], planes[4], h))))


def gsplat_live_feature_outputs(feature_sum, render_alpha):
    """New zero-background feature render with colors=[1,z,z*z,1/z].

    All four channels use the same native rasterization weights. In addition
    compare A=sum(w) against the native 1-T alpha output. Z is positive camera-z
    in the actual model frame, not distance along a unit ray.
    """
    features, alpha = np.asarray(feature_sum), np.asarray(render_alpha)
    if features.ndim != 3 or features.shape[-1] != 4 or features.dtype != np.float32:
        raise ValueError("Expected live (H,W,4) float32 feature output")
    if alpha.shape == (*features.shape[:2], 1):
        alpha = alpha[..., 0]
    if (alpha.dtype != np.float32 or alpha.shape != features.shape[:2]
            or not np.isfinite(alpha).all()
            or not np.allclose(features[..., 0], alpha, atol=1e-5, rtol=1e-5)):
        raise ValueError("Native feature A vs render-alpha consistency failed")
    return validate_raw_moments({name: features[..., i] for i, name in enumerate(RAW_NAMES)})


def packet_from_reference(raw, reference):
    """Use the caller-authenticated frozen builder; does not imply GPU parity."""
    values = validate_raw_moments(raw)
    if reference.METRIC_PACKET_SCHEMA != "ms_gcp_metric_depth_packet_v2":
        raise ValueError("Unknown frozen reference packet schema")
    packet = reference.derive_metric_depth_packet(**values)
    report = reference.recompute_and_compare_packet(packet)
    if report["passed"] is not True:
        raise ValueError("Frozen reference packet consistency failed")
    return packet, report


def source_unit_wire(raw, normalization_scale):
    """Convert all four model-unit moments once, then quantize to wire float32.

    Use source-world camera poses with this output. Do not also inverse-scale
    the resulting 3D point or merely relabel the packet units.
    """
    values = validate_raw_moments(raw)
    source = source_unit_accumulators(values, normalization_scale)
    with np.errstate(over="ignore", under="ignore"):
        wire = {key: value.astype(np.float32) for key, value in source.items()}
    return validate_raw_moments(wire), {
        "conversion": "normalized_model_moments_to_source_model_moments_v1",
        "normalization_scale": float(normalization_scale),
        "input_dtype": "float32", "compute_dtype": "float64", "output_dtype": "float32",
        "unit_scale_applications": 1, "backprojection_pose_domain": "source_model",
        "already_in_survey_metres": False,
    }
