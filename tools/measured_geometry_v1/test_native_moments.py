"""Synthetic wiring tests, not real tensor loading or CUDA qualification."""
import unittest
import numpy as np

from .native_moments import (RAW_NAMES, graphdeco_live_outputs, gsplat_live_feature_outputs,
                             validate_raw_moments, source_unit_wire)


class NativeMomentTests(unittest.TestCase):
    def setUp(self):
        # Two layers at z=2,8, weights=.25,.375. H differs from A^2/M1.
        self.raw = {name: np.full((2, 3), value, np.float32) for name, value in
                    zip(RAW_NAMES, (.625, 3.5, 25, .171875))}

    def test_graphdeco_uses_actual_h_not_inverse_mean(self):
        plane = np.full((6, 2, 3), np.nan, np.float32)
        plane[0], plane[1], plane[4] = [self.raw[k] for k in RAW_NAMES[:3]]
        result = graphdeco_live_outputs(plane, self.raw[RAW_NAMES[3]][None])
        self.assertEqual(float(result[RAW_NAMES[3]][0, 0]), .171875)
        self.assertNotEqual(float(result[RAW_NAMES[3]][0, 0]), .625**2/3.5)
        # Unrelated legacy variance/valid planes do not censor formal support.
        np.testing.assert_array_equal(result[RAW_NAMES[0]], self.raw[RAW_NAMES[0]])

    def test_gsplat_four_channel_wiring(self):
        features = np.stack(list(self.raw.values()), axis=-1)
        result = gsplat_live_feature_outputs(features, self.raw[RAW_NAMES[0]][..., None])
        for name in RAW_NAMES:
            np.testing.assert_array_equal(result[name], self.raw[name])

    def test_alpha_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "alpha consistency"):
            gsplat_live_feature_outputs(np.stack(list(self.raw.values()), -1), np.zeros((2, 3), np.float32))

    def test_missing_h_rejected(self):
        del self.raw[RAW_NAMES[-1]]
        with self.assertRaisesRegex(ValueError, "explicit"):
            validate_raw_moments(self.raw)

    def test_shape_dtype_nonfinite_rejected(self):
        for value in (np.zeros((3, 2), np.float32), np.zeros((2, 3), np.float64),
                      np.full((2, 3), np.nan, np.float32), np.full((2, 3), -1, np.float32)):
            raw = dict(self.raw); raw[RAW_NAMES[-1]] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_raw_moments(raw)

    def test_zero_support_and_no_mutation(self):
        raw = {name: np.zeros((1, 1), np.float32) for name in RAW_NAMES}
        result = validate_raw_moments(raw)
        result[RAW_NAMES[0]][:] = 1
        self.assertEqual(raw[RAW_NAMES[0]].item(), 0)
        raw[RAW_NAMES[1]][:] = 1
        with self.assertRaisesRegex(ValueError, "zero support"):
            validate_raw_moments(raw)

    def test_legacy_packet_with_no_same_call_h_is_not_accepted(self):
        with self.assertRaises(ValueError):
            graphdeco_live_outputs(np.zeros((6, 2, 3), np.float32), None)

    def test_normalized_moments_explicit_float32_wire(self):
        converted, record = source_unit_wire(self.raw, 2)
        for i, factor in enumerate((1, .5, .25, 2)):
            key = RAW_NAMES[i]
            self.assertEqual(converted[key].dtype, np.float32)
            np.testing.assert_array_equal(converted[key], self.raw[key] * factor)
        self.assertEqual(record["unit_scale_applications"], 1)
        self.assertEqual(record["backprojection_pose_domain"], "source_model")

    def test_wire_overflow_and_bad_scale_fail(self):
        for scale in (0, -1, np.nan, 1e-30):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                source_unit_wire(self.raw, scale)


if __name__ == "__main__":
    unittest.main()
