import unittest
import numpy as np

from .score_native_gcp import source_camera_mapping, native_source_point


class NativeBoundaryTests(unittest.TestCase):
    def test_exact_inverse_not_transpose_or_double_scale(self):
        v = np.eye(4)
        v[:3, :3] = [[1., 1e-7, 0], [0, 1, 2e-7], [0, 0, 1]]
        v[:3, 3] = [1, 2, 3]
        t = np.eye(4)
        t[:3, 3] = [2, -1, 4]
        scale = 2.
        source = np.array([.2, .7, 10.])
        model = scale * (t[:3, :3] @ source + t[:3, 3])
        camera = v[:3, :3] @ model + v[:3, 3]
        forward = t.copy()
        forward[:3] *= scale
        mapping = v @ forward
        mapping[:3] /= scale
        record = {"model_w2c_opencv": v.tolist(), "normalization": {"transform": t[:3].tolist(), "scale": scale},
                  "source_w2c_opencv_inverse_normalized": mapping.tolist()}
        calculated = source_camera_mapping(record)
        recovered = native_source_point(calculated, camera[:2]/camera[2], camera[2]/scale)
        np.testing.assert_allclose(recovered, source, rtol=0, atol=1e-12)
        record["source_w2c_opencv_inverse_normalized"][0][3] += .01
        with self.assertRaises(AssertionError):
            source_camera_mapping(record)


if __name__ == "__main__":
    unittest.main()
