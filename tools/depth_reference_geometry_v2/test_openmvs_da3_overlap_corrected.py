#!/usr/bin/env python3
"""Self-tests for corrected OpenMVS--DA3 overlap evaluator."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("openmvs_da3_overlap_corrected.py")
SPEC = importlib.util.spec_from_file_location("openmvs_da3_overlap_corrected", SCRIPT)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def test_native_mirror_then_resize_syncs_depth_and_mask() -> None:
    depth = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    mask = np.array([[1, 0, 1], [0, 1, 1]], dtype=bool)
    out, valid, weight = mod.resize_da3_branch(depth, mask, depth.shape, "mirror")
    expected_depth = np.fliplr(depth)
    expected_mask = np.fliplr(mask)
    assert np.array_equal(valid, expected_mask)
    assert np.allclose(out[valid], expected_depth[expected_mask])
    assert int(weight[valid].sum()) == int(expected_mask.sum())


def test_shuffle_preserves_value_and_mask_counts() -> None:
    depth = np.arange(12, dtype=np.float32).reshape(3, 4) + 1
    mask = np.array([[1, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1]], dtype=bool)
    shuffled_depth, shuffled_mask = mod.branch_native(depth, mask, "shuffle")
    assert sorted(shuffled_depth.reshape(-1).tolist()) == sorted(depth.reshape(-1).tolist())
    assert int(shuffled_mask.sum()) == int(mask.sum())


def test_shared_control_mask_count_is_intersection() -> None:
    openmvs_domain = np.array([[1, 1, 1], [1, 0, 1]], dtype=bool)
    true_valid = np.array([[1, 0, 1], [1, 1, 0]], dtype=bool)
    control_valid = np.array([[0, 1, 1], [1, 0, 1]], dtype=bool)
    true_native = openmvs_domain & true_valid
    control_native = openmvs_domain & control_valid
    shared = true_native & control_native
    assert int(true_native.sum()) == 3
    assert int(control_native.sum()) == 4
    assert int(shared.sum()) == 2


def test_native_and_shared_metric_fields_are_separate() -> None:
    ref = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
    true = ref.copy()
    ctrl = np.fliplr(ref)
    true_mask = np.ones_like(ref, dtype=bool)
    ctrl_mask = np.ones_like(ref, dtype=bool)
    ctrl_mask[:, 0] = False
    shared = true_mask & ctrl_mask
    native = mod.metrics_on_mask(ref, true, true_mask)
    control_native = mod.metrics_on_mask(ref, ctrl, ctrl_mask)
    gd = mod.reference_high_gradient_domain(ref, shared)
    true_shared = mod.metrics_on_mask(ref, true, shared, gradient_domain=gd)
    ctrl_shared = mod.metrics_on_mask(ref, ctrl, shared, gradient_domain=gd)
    row = {
        **mod.prefix(native, "true_native_"),
        **mod.prefix(control_native, "control_native_"),
        **mod.prefix(true_shared, "true_on_shared_"),
        **mod.prefix(ctrl_shared, "control_on_shared_"),
    }
    assert row["true_native_pixels"] != row["control_native_pixels"]
    assert row["true_on_shared_pixels"] == row["control_on_shared_pixels"]
    assert "true_native_absrel_median" in row
    assert "true_on_shared_absrel_median" in row


def test_high_gradient_pixels_shared_for_true_and_control() -> None:
    yy, xx = np.mgrid[0:20, 0:20]
    ref = xx.astype(np.float32) + 1.0
    true = ref.copy()
    ctrl = np.fliplr(ref)
    mask = np.ones_like(ref, dtype=bool)
    gd = mod.reference_high_gradient_domain(ref, mask)
    a = mod.metrics_on_mask(ref, true, mask, gradient_domain=gd)
    b = mod.metrics_on_mask(ref, ctrl, mask, gradient_domain=gd)
    assert a["high_gradient_pixels"] == b["high_gradient_pixels"] == gd.high_count
    assert a["high_gradient_threshold"] == b["high_gradient_threshold"]


def test_insufficient_shared_support_is_not_contradiction() -> None:
    true_native = {
        "pixels": 20_000,
        "coverage": 0.10,
        "spearman": 0.9,
        "high_gradient_cosine_median": 0.9,
    }
    true_shared = {"absrel_median": 0.1, "spearman": 0.9, "high_gradient_cosine_median": 0.9}
    ctrl_shared = {"absrel_median": 0.2, "spearman": 0.1, "high_gradient_cosine_median": 0.1}
    status, caution = mod.control_status(true_native, true_shared, ctrl_shared, usable=False)
    assert status == "weak_or_mixed_proxy_agreement"
    assert caution == "negative_control_inconclusive_due_to_shared_support"


def test_missing_comparison_metrics_are_inconclusive_not_failure() -> None:
    true_native = {
        "pixels": 20_000,
        "coverage": 0.10,
        "spearman": 0.8,
        "high_gradient_cosine_median": 0.8,
    }
    true_shared = {"absrel_median": 0.1, "spearman": None, "high_gradient_cosine_median": None}
    ctrl_shared = {"absrel_median": 0.2, "spearman": None, "high_gradient_cosine_median": None}
    status, caution = mod.control_status(true_native, true_shared, ctrl_shared, usable=True)
    assert status == "weak_or_mixed_proxy_agreement"
    assert caution == "metric_inconclusive"


def main() -> None:
    tests = [
        test_native_mirror_then_resize_syncs_depth_and_mask,
        test_shuffle_preserves_value_and_mask_counts,
        test_shared_control_mask_count_is_intersection,
        test_native_and_shared_metric_fields_are_separate,
        test_high_gradient_pixels_shared_for_true_and_control,
        test_insufficient_shared_support_is_not_contradiction,
        test_missing_comparison_metrics_are_inconclusive_not_failure,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("ALL_OPENMVS_DA3_OVERLAP_CORRECTED_TESTS_PASSED")


if __name__ == "__main__":
    main()
