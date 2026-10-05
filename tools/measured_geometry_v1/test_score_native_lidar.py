import tempfile
from pathlib import Path
import unittest

import numpy as np

from .score_native_lidar import native_surface_samples, recompute_metrics


class NativeSurfaceTests(unittest.TestCase):
    def test_actual_camera_inverse_stride_support(self):
        mapping = np.eye(4)
        mapping[0, 0] = 1.00000003
        mapping[:3, 3] = [1., 2., 3.]
        camera = {"native_camera_array": {"width": 5, "height": 5, "fx": 2., "fy": 2., "cx": 2., "cy": 2.}}
        packet = {"alpha_normalized_expected_camera_z": np.full((5, 5), 4., dtype=np.float32),
                  "accumulated_alpha": np.ones((5, 5), dtype=np.float32),
                  "metric_depth_valid_mask": np.ones((5, 5), dtype=bool)}
        packet["accumulated_alpha"][4, 4] = .49
        points, counts = native_surface_samples(packet, camera, mapping)
        self.assertEqual(counts, {"sampled_pixels": 4, "supported_samples": 3})
        expected = np.array([[-4., -4., 4.], [4., -4., 4.], [-4., 4., 4.]])
        np.testing.assert_allclose(points @ mapping[:3, :3].T + mapping[:3, 3], expected, rtol=0, atol=1e-14)
        packet["metric_depth_valid_mask"] = np.ones((4, 5), dtype=bool)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            native_surface_samples(packet, camera, mapping)

    def test_distance_recomputation_detects_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"d.npz"
            a = np.array([0., .05, .2])
            c = np.array([0., .1])
            np.savez(p, reconstruction_to_reference_m=a, reference_to_reconstruction_m=c)
            metrics = {"reconstruction_points": 3, "reference_points": 2,
                "chamfer_l1_mean_m": (a.mean()+c.mean())/2,
                "symmetric_rmse_m": np.sqrt(((a*a).mean()+(c*c).mean())/2)}
            for prefix, values in (("accuracy", a), ("completeness", c)):
                metrics.update({prefix+"_mean_m": values.mean(), prefix+"_median_m": np.median(values),
                                prefix+"_p95_m": np.quantile(values, .95)})
            for t in [.05, .1, .2]:
                label = str(int(round(t*100)))+"cm"
                precision, recall = (a<=t+1e-9).mean(), (c<=t+1e-9).mean()
                metrics.update({"precision_"+label: precision, "recall_"+label: recall,
                                "fscore_"+label: 2*precision*recall/(precision+recall)})
            self.assertEqual(recompute_metrics(p, metrics, [.05, .1, .2], 1e-9)["status"], "PASS")
            metrics["recall_10cm"] = 0.
            with self.assertRaisesRegex(ValueError, "recomputation"):
                recompute_metrics(p, metrics, [.05, .1, .2], 1e-9)


if __name__ == "__main__":
    unittest.main()
