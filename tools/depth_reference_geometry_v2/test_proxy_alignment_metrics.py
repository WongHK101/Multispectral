#!/usr/bin/env python3
"""Synthetic regression tests for shared proxy-alignment metric primitives."""

from __future__ import annotations

import numpy as np

import proxy_alignment_metrics as metrics


def test_shuffle_is_deterministic_and_keeps_depth_mask_correspondence() -> None:
    depth = np.arange(1, 13, dtype=np.float32).reshape(3, 4)
    valid = (depth % 3) != 0

    depth_a, valid_a = metrics.branch_native(depth, valid, "shuffle")
    depth_b, valid_b = metrics.branch_native(depth, valid, "shuffle")

    assert np.array_equal(depth_a, depth_b)
    assert np.array_equal(valid_a, valid_b)
    assert int(valid_a.sum()) == int(valid.sum())
    assert np.array_equal(valid_a, (depth_a % 3) != 0)


def test_true_and_control_use_identical_shared_support_and_gradient_domain() -> None:
    y, x = np.mgrid[0:32, 0:36]
    reference = (1.0 + 0.2 * x + 0.1 * y).astype(np.float32)
    candidate = (reference * 1.03 + 0.01).astype(np.float32)
    valid = np.ones_like(reference, dtype=bool)
    valid[:2, :] = False
    valid[:, :3] = False
    control, control_valid = metrics.branch_native(candidate, valid, "shuffle")
    shared = valid & control_valid
    domain = metrics.reference_high_gradient_domain(reference, shared)

    true_result = metrics.metrics_on_mask(
        reference, candidate, shared, gradient_domain=domain
    )
    control_result = metrics.metrics_on_mask(
        reference, control, shared, gradient_domain=domain
    )

    assert true_result["pixels"] == control_result["pixels"] == int(shared.sum())
    assert (
        true_result["high_gradient_pixels"]
        == control_result["high_gradient_pixels"]
        == domain.high_count
    )
    assert true_result["high_gradient_threshold"] == control_result[
        "high_gradient_threshold"
    ]


def test_metric_values_match_frozen_reference_formulas() -> None:
    y, x = np.mgrid[0:24, 0:28]
    reference = (2.0 + 0.07 * x + 0.11 * y).astype(np.float32)
    candidate = (reference * 1.1).astype(np.float32)
    mask = np.ones_like(reference, dtype=bool)

    result = metrics.metrics_on_mask(reference, candidate, mask)

    expected = np.abs(candidate.astype(np.float64) - reference) / np.maximum(
        np.abs(reference), 1e-6
    )
    assert result["absrel_median"] == float(np.median(expected))
    assert result["absrel_p90"] == float(np.percentile(expected, 90))
    assert result["pixels"] == reference.size
    assert np.isclose(result["pearson"], 1.0)
    assert np.isclose(result["spearman"], 1.0)


def test_empty_support_is_explicitly_inconclusive() -> None:
    reference = np.ones((8, 9), dtype=np.float32)
    result = metrics.metrics_on_mask(
        reference, reference, np.zeros_like(reference, dtype=bool)
    )

    assert result["pixels"] == 0
    assert result["coverage"] == 0.0
    assert result["absrel_median"] is None
    assert result["spearman"] is None
