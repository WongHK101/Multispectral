import unittest
import numpy as np
from .score_native_appearance import quantize_gt, quantize_prediction, spectral_metrics


class NativeAppearanceTests(unittest.TestCase):
    def test_frozen_truncation_and_scalar_channels(self):
        raw = np.array([[0, 128, 32768, 65535]], dtype=np.uint16)
        gt = quantize_gt(raw)
        np.testing.assert_array_equal(gt[0, :, 0], [0, 0, 127, 255])
        np.testing.assert_array_equal(gt[..., 0], gt[..., 2])
        pred = np.array([[[0.5], [0.999], [1.1], [-.1]]], dtype=np.float32)
        np.testing.assert_array_equal(quantize_prediction(pred)[0, :, 0], [127, 254, 255, 0])
        rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        np.testing.assert_array_equal(quantize_gt(rgb), rgb)
        with self.assertRaises(ValueError):
            quantize_prediction(np.array([[[np.nan]]], dtype=np.float32))

    def test_mask_and_zero_spectrum_semantics(self):
        p = np.zeros((1, 2, 4), dtype=np.float32)
        g = p.copy()
        p[0, 1] = 1.
        result = spectral_metrics(p, g, np.array([[True, False]]))
        self.assertEqual(result["rmse_4band"], 0.)
        self.assertEqual(result["sam_deg"], 90.)
        with self.assertRaises(ValueError):
            spectral_metrics(p, g, np.array([[False, False]]))


if __name__ == "__main__":
    unittest.main()
