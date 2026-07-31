#!/usr/bin/env python3
"""Shared metric primitives for validation-gated camera-z proxy alignment.

The routines in this module are intentionally independent of any particular
scene or auxiliary depth model.  True and deterministic control branches are
compared on the same pixel support, and the reference high-gradient domain is
computed once and reused for both branches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.stats import pearsonr, spearmanr


SHUFFLE_SEED = 230626
METRIC_COMPARE_EPS = 1e-6
MIN_SHARED_PIXELS = 10_000
MIN_SHARED_COVERAGE = 0.05
MIN_TRUE_PIXELS = 10_000
MIN_TRUE_COVERAGE = 0.05
HIGHGRAD_QUANTILE = 75.0


def finite_positive(array: np.ndarray) -> np.ndarray:
    """Return the finite, strictly positive support of an array."""

    return np.isfinite(array) & (array > 0)


def branch_native(
    depth: np.ndarray,
    accepted: np.ndarray,
    branch: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a native-raster true, mirror, or deterministic shuffle branch.

    The same permutation is applied to depth and validity, preserving their
    correspondence and the number of valid pixels.
    """

    depth = np.asarray(depth)
    accepted = np.asarray(accepted, dtype=bool)
    if depth.shape != accepted.shape:
        raise ValueError(
            f"depth/accepted shape mismatch: {depth.shape} != {accepted.shape}"
        )
    if branch == "true":
        return depth.copy(), accepted.copy()
    if branch == "mirror":
        return np.fliplr(depth), np.fliplr(accepted)
    if branch == "shuffle":
        rng = np.random.default_rng(SHUFFLE_SEED)
        indices = rng.permutation(depth.size)
        return (
            depth.reshape(-1)[indices].reshape(depth.shape),
            accepted.reshape(-1)[indices].reshape(accepted.shape),
        )
    raise ValueError(f"unknown branch {branch!r}")


def safe_corr(kind: str, x: np.ndarray, y: np.ndarray) -> float | None:
    """Compute a finite Pearson or Spearman coefficient when support permits."""

    if len(x) < 10:
        return None
    try:
        if kind == "pearson":
            value = pearsonr(x, y).statistic
        elif kind == "spearman":
            value = spearmanr(x, y, nan_policy="omit").correlation
        else:
            raise ValueError(f"unsupported correlation kind {kind!r}")
    except Exception:
        return None
    return float(value) if np.isfinite(value) else None


def complete_local_mask(mask: np.ndarray) -> np.ndarray:
    """Require a complete finite 3x3 neighborhood for gradient comparisons."""

    base = np.asarray(mask, dtype=bool)
    if base.shape[0] < 3 or base.shape[1] < 3:
        return np.zeros_like(base, dtype=bool)
    return binary_erosion(
        base,
        structure=np.ones((3, 3), dtype=bool),
        iterations=1,
        border_value=0,
    )


@dataclass
class GradientDomain:
    high_mask: np.ndarray
    threshold: float | None
    local_valid_count: int
    high_count: int
    erosion_rule: str = (
        "3x3_binary_erosion_one_iteration_plus_complete_finite_gradient_neighborhood"
    )


def reference_high_gradient_domain(
    reference: np.ndarray,
    mask: np.ndarray,
) -> GradientDomain:
    """Freeze the reference-defined top-quartile gradient comparison domain."""

    local = complete_local_mask(mask & np.isfinite(reference))
    if int(local.sum()) < 100:
        return GradientDomain(np.zeros_like(mask, dtype=bool), None, int(local.sum()), 0)
    grad_y, grad_x = np.gradient(
        np.where(np.isfinite(reference), reference, np.nan).astype(np.float64)
    )
    finite = local & np.isfinite(grad_x) & np.isfinite(grad_y)
    magnitude = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    values = magnitude[finite]
    if len(values) < 100:
        return GradientDomain(np.zeros_like(mask, dtype=bool), None, int(len(values)), 0)
    threshold = float(np.percentile(values, HIGHGRAD_QUANTILE))
    high = finite & (magnitude >= threshold)
    return GradientDomain(high, threshold, int(finite.sum()), int(high.sum()))


def metrics_on_mask(
    reference: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
    *,
    gradient_domain: GradientDomain | None = None,
) -> dict[str, Any]:
    """Compute proxy-alignment metrics on one explicitly shared pixel mask."""

    selected = mask & finite_positive(reference) & finite_positive(candidate)
    pixels = int(selected.sum())
    coverage = float(pixels / selected.size) if selected.size else 0.0
    if pixels == 0:
        return {
            "pixels": 0,
            "coverage": coverage,
            "absrel_median": None,
            "absrel_p90": None,
            "pearson": None,
            "spearman": None,
            "high_gradient_cosine_median": None,
            "high_gradient_threshold": None,
            "high_gradient_pixels": 0,
            "gradient_local_valid_pixels": 0,
            "gradient_erosion_rule": (
                "3x3_binary_erosion_one_iteration_plus_"
                "complete_finite_gradient_neighborhood"
            ),
        }

    ref_values = reference[selected].astype(np.float64)
    candidate_values = candidate[selected].astype(np.float64)
    relative = np.abs(candidate_values - ref_values) / np.maximum(
        np.abs(ref_values), 1e-6
    )
    domain = gradient_domain or reference_high_gradient_domain(reference, selected)
    cosine_median: float | None = None
    if domain.high_count > 0:
        ref_grad_y, ref_grad_x = np.gradient(reference.astype(np.float64))
        cand_grad_y, cand_grad_x = np.gradient(candidate.astype(np.float64))
        high = (
            domain.high_mask
            & np.isfinite(ref_grad_x)
            & np.isfinite(ref_grad_y)
            & np.isfinite(cand_grad_x)
            & np.isfinite(cand_grad_y)
        )
        if int(high.sum()) > 0:
            dot = (
                ref_grad_x[high] * cand_grad_x[high]
                + ref_grad_y[high] * cand_grad_y[high]
            )
            norm = np.sqrt(ref_grad_x[high] ** 2 + ref_grad_y[high] ** 2) * np.sqrt(
                cand_grad_x[high] ** 2 + cand_grad_y[high] ** 2
            )
            cosine = dot / np.maximum(norm, 1e-12)
            cosine_median = float(np.median(cosine)) if len(cosine) else None

    return {
        "pixels": pixels,
        "coverage": coverage,
        "absrel_median": float(np.median(relative)),
        "absrel_p90": float(np.percentile(relative, 90)),
        "pearson": safe_corr("pearson", ref_values, candidate_values),
        "spearman": safe_corr("spearman", ref_values, candidate_values),
        "high_gradient_cosine_median": cosine_median,
        "high_gradient_threshold": domain.threshold,
        "high_gradient_pixels": int(domain.high_count),
        "gradient_local_valid_pixels": int(domain.local_valid_count),
        "gradient_erosion_rule": domain.erosion_rule,
    }
