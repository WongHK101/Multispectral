import unittest

import numpy as np

from .native_moments import RAW_NAMES
from .proxy_checkpoint_export import source_moments_with_legacy_parity


class ProxyFreshPacketTests(unittest.TestCase):
    def setUp(self):
        self.planes = np.zeros((6, 2, 3), np.float32)
        self.planes[0], self.planes[1], self.planes[4] = .625, 3.5, 25
        self.h = np.full((1, 2, 3), .171875, np.float32)
        self.dp = {name: self.planes[i].copy() for name, i in (
            ("accumulated_opacity", 0), ("weighted_camera_z_sum", 1),
            ("expected_camera_z", 2), ("numeric_valid", 3),
            ("weighted_camera_z2_sum", 4), ("camera_z_variance", 5))}
        self.source = {k: v.copy() for k, v in self.dp.items()}
        self.source["weighted_camera_z_sum"] /= 2
        self.source["weighted_camera_z2_sum"] /= 4

    def test_real_h_and_preserved_legacy_raw_arrays(self):
        result = source_moments_with_legacy_parity(self.planes, self.h, self.dp, self.source, 2)
        for name, old in zip(RAW_NAMES[:3], ("accumulated_opacity", "weighted_camera_z_sum", "weighted_camera_z2_sum")):
            self.assertEqual(result[name].tobytes(), self.source[old].tobytes())
        self.assertEqual(result[RAW_NAMES[3]][0, 0], .34375)
        self.assertFalse(self.source["numeric_valid"].any())
        self.assertTrue((result[RAW_NAMES[0]] > 0).all())

    def test_wrong_accumulator_or_expanded_legacy_schema_rejected(self):
        self.dp["weighted_camera_z_sum"][0, 0] += 1
        with self.assertRaisesRegex(ValueError, "accumulator"):
            source_moments_with_legacy_parity(self.planes, self.h, self.dp, self.source, 2)
        self.dp["new_tensor"] = self.h
        with self.assertRaisesRegex(ValueError, "exactly six"):
            source_moments_with_legacy_parity(self.planes, self.h, self.dp, self.source, 2)


if __name__ == "__main__":
    unittest.main()
