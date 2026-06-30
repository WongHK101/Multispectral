#!/usr/bin/env python3
"""Self-tests for tri_proxy_comparison metric utilities."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("tri_proxy_comparison.py")
SPEC = importlib.util.spec_from_file_location("tri_proxy_comparison", SCRIPT)
assert SPEC and SPEC.loader
tri = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tri)


def test_masked_resize_blocks_invalid_bleed() -> None:
    arr = np.array([[1.0, 1.0, 1000.0, 1000.0], [1.0, 1.0, 1000.0, 1000.0]], dtype=np.float64)
    mask = np.array([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=bool)
    resized, valid, weight = tri.resize_masked_float(arr, mask, (1, 2), min_weight=0.5)
    assert valid[0, 0]
    assert not valid[0, 1]
    assert abs(resized[0, 0] - 1.0) < 1e-6


def test_mirror_transforms_mask_with_depth() -> None:
    arr = np.arange(6, dtype=np.float64).reshape(2, 3)
    mask = np.array([[1, 0, 0], [1, 1, 0]], dtype=bool)
    mirrored_arr, mirrored_mask = tri.mirrored(arr, mask)
    assert np.array_equal(mirrored_arr, np.fliplr(arr))
    assert np.array_equal(mirrored_mask, np.fliplr(mask))


def test_gradient_erodes_mask_boundary() -> None:
    arr = np.arange(64, dtype=np.float64).reshape(8, 8)
    mask = np.ones((8, 8), dtype=bool)
    mask[:, 0] = False
    eroded = tri.eroded_local_mask(mask)
    assert eroded.sum() < mask.sum()
    metrics = tri.gradient_metrics(arr, arr, mask)
    assert metrics["gradient_local_valid_count"] == int(eroded.sum())
    assert metrics["gradient_primary_cosine_highgrad_median"] > 0.99
    assert metrics["gradient_sensitivity_cosine_allvalid_median"] > 0.99
    assert metrics["gradient_highgrad_threshold"] >= 0.0


def test_nan_invalid_neighborhood_gradient() -> None:
    arr = np.arange(100, dtype=np.float64).reshape(10, 10)
    ref = arr.copy()
    arr[4, 4] = np.nan
    mask = np.ones((10, 10), dtype=bool)
    metrics = tri.gradient_metrics(arr, ref, mask)
    assert metrics["gradient_local_valid_count"] < int(tri.eroded_local_mask(mask).sum())
    assert np.isfinite(metrics["gradient_primary_cosine_highgrad_median"])


def test_high_gradient_subset_selects_edges() -> None:
    ref = np.zeros((12, 12), dtype=np.float64)
    ref[:, 6:] = 10.0
    cand = ref.copy()
    mask = np.ones_like(ref, dtype=bool)
    metrics = tri.gradient_metrics(cand, ref, mask, highgrad_quantile=70.0)
    assert metrics["gradient_highgrad_count"] > 0
    assert metrics["gradient_highgrad_count"] < metrics["gradient_local_valid_count"]
    assert metrics["gradient_primary_cosine_highgrad_median"] > 0.99


def test_spatial_block_bootstrap_deterministic() -> None:
    yy, xx = np.mgrid[0:32, 0:32]
    ref = xx.astype(np.float64) + 1.0
    cand = ref * 1.05
    mask = np.ones_like(ref, dtype=bool)
    a = tri.spatial_block_bootstrap_ci(cand, ref, mask, block=8, reps=30, seed=7)
    b = tri.spatial_block_bootstrap_ci(cand, ref, mask, block=8, reps=30, seed=7)
    assert a == b
    assert a["spatial_block_count"] == 16
    assert a["spatial_block_bootstrap_absrel_median_ci_low"] >= 0.0


def test_constant_map_bootstrap_has_nan_spearman() -> None:
    ref = np.ones((24, 24), dtype=np.float64)
    cand = np.ones((24, 24), dtype=np.float64) * 1.1
    mask = np.ones_like(ref, dtype=bool)
    stats = tri.spatial_block_bootstrap_ci(cand, ref, mask, block=8, reps=20, seed=1)
    assert stats["spatial_block_count"] == 9
    assert np.isnan(stats["spatial_block_bootstrap_spearman_ci_low"])


def test_insufficient_block_bootstrap_returns_nan() -> None:
    ref = np.ones((8, 8), dtype=np.float64)
    cand = ref.copy()
    mask = np.zeros_like(ref, dtype=bool)
    mask[:3, :3] = True
    stats = tri.spatial_block_bootstrap_ci(cand, ref, mask, block=8, reps=20)
    assert stats["spatial_block_count"] < 4
    assert np.isnan(stats["spatial_block_bootstrap_absrel_median_ci_low"])


def test_metric_row_has_no_redundant_or_fake_normal_fields() -> None:
    arr = np.arange(25, dtype=np.float64).reshape(5, 5) + 1.0
    mask = np.ones((5, 5), dtype=bool)
    row = tri.metric_row("self", "unit", arr, arr, mask)
    assert "standardized_corr" not in row
    assert "surface_normal_cosine_median" not in row
    assert "spatial_block_bootstrap_absrel_median_ci_low" in row
    assert row["absrel_median"] == 0.0
    assert "scale_aligned_absrel_median_sensitivity" in row


def test_perfect_scale_synthetic_map() -> None:
    ref = np.linspace(1.0, 5.0, 100).reshape(10, 10)
    cand = ref / 2.0
    mask = np.ones_like(ref, dtype=bool)
    row = tri.metric_row("scale", "unit", cand, ref, mask)
    assert row["absrel_median"] > 0.4
    assert abs(row["scale_aligned_factor_sensitivity"] - 2.0) < 1e-8
    assert row["scale_aligned_absrel_median_sensitivity"] < 1e-8


def test_spatial_shuffle_synthetic_degrades_rank() -> None:
    ref = np.arange(1, 101, dtype=np.float64).reshape(10, 10)
    cand = ref.copy()
    mask = np.ones_like(ref, dtype=bool)
    shuffled = tri.shuffled(cand, mask, seed=3)
    true_row = tri.metric_row("true", "unit", cand, ref, mask)
    shuffled_row = tri.metric_row("shuffle", "unit", shuffled, ref, mask)
    assert true_row["spearman"] > 0.99
    assert shuffled_row["spearman"] < 0.4


def test_known_gradient_synthetic() -> None:
    yy, xx = np.mgrid[0:20, 0:20]
    ref = xx.astype(np.float64)
    cand = ref * 2.0
    mask = np.ones_like(ref, dtype=bool)
    metrics = tri.gradient_metrics(cand, ref, mask)
    assert metrics["gradient_primary_cosine_highgrad_median"] > 0.99
    assert metrics["gradient_primary_magnitude_absrel_highgrad_median"] > 0.9


def main() -> None:
    tests = [
        test_masked_resize_blocks_invalid_bleed,
        test_mirror_transforms_mask_with_depth,
        test_gradient_erodes_mask_boundary,
        test_nan_invalid_neighborhood_gradient,
        test_high_gradient_subset_selects_edges,
        test_spatial_block_bootstrap_deterministic,
        test_constant_map_bootstrap_has_nan_spearman,
        test_insufficient_block_bootstrap_returns_nan,
        test_metric_row_has_no_redundant_or_fake_normal_fields,
        test_perfect_scale_synthetic_map,
        test_spatial_shuffle_synthetic_degrades_rank,
        test_known_gradient_synthetic,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("ALL_TRI_PROXY_METRIC_TESTS_PASSED")


if __name__ == "__main__":
    main()
