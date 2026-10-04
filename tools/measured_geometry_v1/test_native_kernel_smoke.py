import unittest

import numpy as np

from .native_kernel_smoke import CASES, compare_center
from .raster_weight_reference import scalar_moments


class KernelFixtureTests(unittest.TestCase):
    def test_center_oracle_uses_frozen_native_rules(self):
        for renderer in ("umgs_graphdeco_exclusive_stop_v1", "gsplat_1_4_0_exclusive_stop_v1"):
            for _, _, _, alpha in CASES:
                ref = scalar_moments([2., 5., 9.], alpha, renderer=renderer)
                raw = {k: v.copy() for k, v in ref["raw"].items()}
                self.assertEqual(len(compare_center(raw, alpha, [2., 5., 9.], renderer, 0, 0)["checks"]), 4)

    def test_wrong_moment_and_nonfinite_fail(self):
        alpha, z, renderer = [.2, .3, .4], [2., 5., 9.], "gsplat_1_4_0_exclusive_stop_v1"
        ref = scalar_moments(z, alpha, renderer=renderer)
        for wrong in (9., float("nan"), float("inf")):
            raw = {k: v.copy() for k, v in ref["raw"].items()}
            raw["weighted_inverse_camera_z_sum"][0, 0] = wrong
            with self.assertRaises(ValueError):
                compare_center(raw, alpha, z, renderer, 0, 0)


if __name__ == "__main__":
    unittest.main()
