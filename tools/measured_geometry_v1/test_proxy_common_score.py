import unittest
import numpy as np

from .proxy_common_score import common_masks, complete_mean, mask_digest


def packet():
    one = np.ones((3, 4), dtype=np.float32)
    return dict(accumulated_opacity=one.copy(), weighted_camera_z_sum=one.copy(),
                weighted_camera_z2_sum=one.copy(), expected_camera_z=one.copy(),
                numeric_valid=one.astype(np.uint8), camera_z_variance=np.zeros_like(one))


class CommonProxyTests(unittest.TestCase):
    def test_full_set_not_pairwise(self):
        ps = {name: packet() for name in ['a', 'b', 'c']}
        ps['b']['numeric_valid'][0, 0] = 0
        ps['c']['expected_camera_z'][1, 0] = np.nan
        ps['c']['accumulated_opacity'][2, 0] = .4
        masks = common_masks(np.ones((3, 4)), np.ones((3, 4)), ps, tuple(ps))
        self.assertEqual(masks['common_primary'].sum(), 10)
        self.assertEqual(masks['common_opacity_ge_0_5'].sum(), 9)
        self.assertFalse(masks['common_primary'][0, 0])
        self.assertFalse(masks['common_primary'][1, 0])

    def test_missing_or_extra_method_rejected(self):
        for methods in [('a', 'b'), ()]:
            with self.assertRaisesRegex(ValueError, 'method set'):
                common_masks(np.ones((3, 4)), np.ones((3, 4)), {'a': packet()}, methods)

    def test_shape_rejected(self):
        p = packet(); p['camera_z_variance'] = np.zeros((4, 3))
        with self.assertRaisesRegex(ValueError, 'shape'):
            common_masks(np.ones((3, 4)), np.ones((3, 4)), {'a': p}, ('a',))

    def test_variance_sensitivity_does_not_drop_primary(self):
        p = packet(); p['camera_z_variance'][0, 0] = -1
        m = common_masks(np.ones((3, 4)), np.ones((3, 4)), {'a': p}, ('a',))
        self.assertEqual(m['common_primary'].sum(), 12)
        self.assertEqual(m['common_variance_valid'].sum(), 11)

    def test_no_partial_mean(self):
        self.assertIsNone(complete_mean([1., None], 2)['value'])
        self.assertIsNone(complete_mean([1., np.nan], 2)['value'])
        self.assertEqual(complete_mean([1., 3.], 2)['value'], 2.)
        with self.assertRaises(ValueError): complete_mean([1.], 2)

    def test_mask_digest_binds_shape_and_content(self):
        a = np.ones((3, 4), dtype=bool)
        self.assertEqual(mask_digest(a), mask_digest(a.astype(np.uint8)))
        self.assertNotEqual(mask_digest(a), mask_digest(a.reshape(4, 3)))
        b = a.copy(); b[0, 0] = False
        self.assertNotEqual(mask_digest(a), mask_digest(b))


if __name__ == '__main__': unittest.main()
