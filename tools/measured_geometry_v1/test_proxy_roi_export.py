import copy
import unittest

import numpy as np

from .contracts import record_hash
from .proxy_roi import PIXELS
from .proxy_roi_export import model_pose, normalization_record, projection_audit, projection_matrix, PROJECTION_TOLERANCE


def camera():
    row = dict(image_name='view.JPG', image_id=1, camera_id=1, width=1200, height=869,
        fx=784.1066742283879, fy=784.1066742283879, cx=600., cy=434.6765641569459,
        source_c2w_opencv=np.eye(4).tolist(), pixel_sampling=PIXELS, construction='colmap_float64')
    row['record_sha256'] = record_hash(row)
    return row


class ReferenceCameraExport(unittest.TestCase):
    def test_exact_offcenter_camera(self):
        c = camera()
        self.assertLess(projection_audit(c, projection_matrix(c), 1e-9)['max_pixel_error'], 1e-9)

    def test_float32_projection_roundoff(self):
        c = camera()
        self.assertLess(projection_audit(c, projection_matrix(c).astype(np.float32), PROJECTION_TOLERANCE)['max_pixel_error'], PROJECTION_TOLERANCE)

    def test_centered_wrong_principal_point(self):
        c = camera(); p = projection_matrix(c); p[2, 1] = 0
        with self.assertRaises(ValueError):
            projection_audit(c, p, PROJECTION_TOLERANCE)

    def test_half_pixel_mismatch(self):
        c = camera(); p = projection_matrix(c); p[2, 0] += 1/c['width']
        with self.assertRaises(ValueError):
            projection_audit(c, p, PROJECTION_TOLERANCE)

    def test_transpose_mismatch(self):
        c = camera()
        with self.assertRaises(ValueError):
            projection_audit(c, projection_matrix(c).T, PROJECTION_TOLERANCE)

    def test_wrong_focal_length(self):
        c = camera(); p = projection_matrix(c); p[0, 0] *= 1.01
        with self.assertRaises(ValueError):
            projection_audit(c, p, PROJECTION_TOLERANCE)

    def test_camera_hash_tamper(self):
        c = camera(); c['cx'] += .1
        with self.assertRaises(ValueError):
            projection_matrix(c)

    def test_pixel_convention_tamper(self):
        c = camera(); c['pixel_sampling'] = 'integer_corner_without_conversion'
        c['record_sha256'] = record_hash({k:v for k,v in c.items() if k != 'record_sha256'})
        with self.assertRaises(ValueError):
            projection_matrix(c)

    def test_identity_source_pose(self):
        c = camera()
        p, error = model_pose(c, dict(transform=np.eye(4).tolist(), scale=1))
        np.testing.assert_array_equal(p, c['source_c2w_opencv'])
        self.assertEqual(error, 0)

    def test_saved_rotated_translated_scaled_dataparser(self):
        c = camera(); t = np.eye(4)
        t[:3, :3] = [[0,-1,0],[1,0,0],[0,0,1]]; t[:3, 3] = [3,4,5]
        p, error = model_pose(c, dict(transform=t[:3].tolist(), scale=2))
        np.testing.assert_array_equal(p[:3, 3], [6,8,10])
        np.testing.assert_array_equal(p[:3, :3], t[:3, :3])
        self.assertLess(error, 1e-7)

    def test_normalization_axis_flip_rejection(self):
        with self.assertRaises(ValueError):
            normalization_record(dict(transform=np.diag([-1,1,1,1]).tolist(), scale=1))

    def test_normalization_unknown_scale(self):
        with self.assertRaises(ValueError):
            normalization_record(dict(transform=np.eye(4).tolist(), scale=0))


if __name__ == '__main__':
    unittest.main(verbosity=2)
