import unittest

import numpy as np

from .raster_weight_reference import GRAPHDECO, GSPLAT, is_terminated, scalar_moments


class NativeWeightReferenceTests(unittest.TestCase):
    def test_native_alpha_caps_differ(self):
        for renderer, cap in ((GRAPHDECO, .99), (GSPLAT, .999)):
            result = scalar_moments([2], [1], renderer=renderer)
            self.assertEqual(result["raw"]["accumulated_alpha"][0, 0], np.float32(cap))

    def test_terminating_layer_excluded_in_both_renderers(self):
        for renderer in (GRAPHDECO, GSPLAT):
            result = scalar_moments([2, 4, 6], [1, 1, 1], renderer=renderer)
            self.assertEqual(result["accepted_layer_indices"], [0])
            self.assertEqual(result["excluded_terminating_layer"], 1)
            self.assertEqual(result["raw"]["weighted_camera_z_sum"][0, 0],
                             result["raw"]["accumulated_alpha"][0, 0] * 2)

    def test_termination_literal_and_comparison(self):
        threshold = np.float32(1e-4)
        self.assertFalse(is_terminated(threshold, GRAPHDECO))
        self.assertTrue(is_terminated(threshold, GSPLAT))

    def test_cutoff_layer_does_not_change_transmittance(self):
        result = scalar_moments([2, 4], [.001, .5], renderer=GRAPHDECO)
        self.assertEqual(result["cutoff_skipped_indices"], [0])
        self.assertEqual(result["remaining_transmittance"], .5)
        self.assertEqual(result["raw"]["weighted_camera_z_sum"][0, 0], 2)

    def test_gentle_case_same_weights(self):
        a = scalar_moments([2, 4], [.5, .5], renderer=GRAPHDECO)
        b = scalar_moments([2, 4], [.5, .5], renderer=GSPLAT)
        for key in a["raw"]:
            np.testing.assert_array_equal(a["raw"][key], b["raw"][key])
        self.assertFalse(a["gpu_parity_proven"])

    def test_bad_or_unsorted_inputs_rejected(self):
        for z, a in (([0], [.5]), ([np.nan], [.5]), ([2], [1.1]), ([4, 2], [.5, .5])):
            with self.assertRaises(ValueError):
                scalar_moments(z, a, renderer=GRAPHDECO)
        with self.assertRaises(ValueError):
            scalar_moments([2], [.5], renderer="generic")


if __name__ == "__main__":
    unittest.main()
