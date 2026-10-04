"""Diagnostic scalar references for pinned native compositing, not training.

Input alpha is already evaluated at a pixel and layers already depth-sorted.
These references do not emulate projection, sorting, CUDA exp/FMA, or certify
kernel parity. They deliberately differ from the generic frozen packet layer
helper at alpha caps and early termination. Packet derivation remains external.
"""
from __future__ import annotations

import numpy as np

from .native_moments import RAW_NAMES

GRAPHDECO = "umgs_graphdeco_exclusive_stop_v1"
GSPLAT = "gsplat_1_4_0_exclusive_stop_v1"


def is_terminated(next_t, renderer):
    if renderer == GRAPHDECO:
        return next_t < np.float32(1e-4)
    if renderer == GSPLAT:
        # The source uses an unsuffixed double literal in this comparison.
        return float(next_t) <= 1e-4
    raise ValueError("Unverified renderer weight convention")


def scalar_moments(camera_z, pixel_alpha, *, renderer):
    is_terminated(np.float32(1), renderer)
    z = np.asarray(camera_z, dtype=np.float32)
    a = np.asarray(pixel_alpha, dtype=np.float32)
    if (z.ndim != 1 or z.shape != a.shape or not np.isfinite(z).all()
            or not np.isfinite(a).all() or np.any(z <= 0) or np.any(a < 0)
            or np.any(a > 1) or np.any(np.diff(z) < 0)):
        raise ValueError("Require finite positive sorted z and alpha in [0,1]")
    cap = np.float32(.99 if renderer == GRAPHDECO else .999)
    cutoff = np.float32(1) / np.float32(255)
    t = np.float32(1)
    moments = np.zeros(4, dtype=np.float32)
    accepted, skipped, terminating = [], [], None
    for index, (depth, alpha) in enumerate(zip(z, a)):
        alpha = min(alpha, cap)
        if alpha < cutoff:
            skipped.append(index)
            continue
        next_t = np.float32(t * np.float32(1 - alpha))
        if is_terminated(next_t, renderer):
            terminating = index
            break
        weight = np.float32(t * alpha)
        features = np.array([1, depth, depth * depth, np.float32(1) / depth], dtype=np.float32)
        moments = np.asarray(moments + weight * features, dtype=np.float32)
        t = next_t
        accepted.append(index)
    if not np.isfinite(moments).all():
        raise ValueError("Synthetic moment overflow")
    return {"raw": {name: np.array([[v]], dtype=np.float32) for name, v in zip(RAW_NAMES, moments)},
            "accepted_layer_indices": accepted, "cutoff_skipped_indices": skipped,
            "excluded_terminating_layer": terminating, "remaining_transmittance": float(t),
            "renderer_weight_convention": renderer, "gpu_parity_proven": False}
