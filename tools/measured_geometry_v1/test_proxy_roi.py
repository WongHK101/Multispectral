import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from .contracts import record_hash
from .proxy_common_score import CORE_METHODS, complete_mean
from .proxy_roi import (camera_record, qualify_reference, roi_mask, roi_common_masks,
    validate_camera, validate_reference, reference_camera, validate_triangle_ray_binding)
from .proxy_roi_score import validate_packet_index
from .test_proxy_common_score import packet


def cam():
    im = SimpleNamespace(id=1, name='view.JPG', camera_id=1,
        qvec=np.array([1., 0, 0, 0]), tvec=np.zeros(3))
    source = SimpleNamespace(model='PINHOLE', width=4, height=3, params=np.array([1., 1., 0., 0.]))
    return camera_record(im, source)


def transform():
    return dict(rotation=np.eye(3).tolist(), translation=[0., 0., 0.], scale=1.,
        method_specific_fit=False, lidar_used_in_fit=False)


def ref():
    return dict(depth=np.ones((3, 4), np.float32), valid=np.ones((3, 4), np.uint8),
        triangle_id=np.zeros((3, 4), np.int32),
        barycentric=np.tile(np.array([.2, .3, .5], np.float32), (3, 4, 1)))


class RoiTests(unittest.TestCase):
    def test_strict_boundary_and_half_pixel(self):
        mask, regions = roi_mask(ref()['depth'], ref()['valid'], cam(), transform(),
            [.5, .5, 3.5, 2.5], lambda x, y: (x, y))
        expected = np.zeros((3, 4), bool); expected[1, 1:3] = True
        np.testing.assert_array_equal(mask, expected)
        np.testing.assert_array_equal(regions < 16, expected)

    def test_actual_first_hit_not_a_surface_behind_occluder(self):
        depth = ref()['depth']; depth[1, 1] = 10.
        mask, _ = roi_mask(depth, ref()['valid'], cam(), transform(), [1., 1., 2., 2.], lambda x, y: (x, y))
        self.assertFalse(mask.any())

    def test_roi_transform_does_not_change_depth_units(self):
        depth = ref()['depth']; before = depth.copy()
        t = transform(); t['translation'] = [100., 200., 300.]; t['scale'] = 2.
        mask, _ = roi_mask(depth, ref()['valid'], cam(), t, [100., 200., 108., 206.], lambda x, y: (x, y))
        self.assertTrue(mask.all()); np.testing.assert_array_equal(depth, before)

    def test_nonfinite_conversion_rejected(self):
        with self.assertRaisesRegex(ValueError, 'CRS'):
            roi_mask(ref()['depth'], ref()['valid'], cam(), transform(), [0, 0, 4, 3], lambda x, y: (x * np.nan, y))

    def test_no_lidar_or_method_fit(self):
        for key in ['method_specific_fit', 'lidar_used_in_fit']:
            t = transform(); t[key] = True
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'fit prohibited'):
                roi_mask(ref()['depth'], ref()['valid'], cam(), t, [0, 0, 4, 3], lambda x, y: (x, y))

    def test_reference_camera_tamper(self):
        for key in ['fx', 'cx', 'width', 'source_c2w_opencv']:
            c = cam(); c[key] = 9
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'hash'):
                validate_camera(c)

    def test_unknown_convention_even_with_rehashed_camera(self):
        c = cam(); c['pixel_sampling'] = 'one_based'; c.pop('record_sha256'); c['record_sha256'] = record_hash(c)
        with self.assertRaisesRegex(ValueError, 'pixel sampling'):
            validate_camera(c)

    def test_reference_shape_not_resized(self):
        p = ref(); p['depth'] = np.ones((2, 2), np.float32)
        with self.assertRaisesRegex(ValueError, 'resizing prohibited'):
            validate_reference(p, cam())

    def test_road_cannot_fall_back_to_colmap(self):
        with self.assertRaisesRegex(ValueError, 'no COLMAP fallback'):
            reference_camera('road', {}, None, None, '0' * 64)

    def test_triangle_ray_binding_rejects_wrong_view_and_mesh(self):
        p = ref(); y, x = np.indices((3, 4))
        b1, b2 = (x + .5) / 8., (y + .5) / 8.
        p['barycentric'] = np.stack([1 - b1 - b2, b1, b2], axis=-1).astype(np.float32)
        vertices = np.array([[0., 0, 1], [8., 0, 1], [0., 8, 1]])
        faces = np.array([[0, 1, 2]])
        report = validate_triangle_ray_binding(p, cam(), vertices, faces)
        self.assertEqual(report['max_error_source_model_units'], 0.)
        self.assertEqual(len(report['samples']), 12)
        wrong = cam(); wrong['source_c2w_opencv'][0][3] = 1.
        wrong.pop('record_sha256'); wrong['record_sha256'] = record_hash(wrong)
        with self.assertRaisesRegex(ValueError, 'binding mismatch'):
            validate_triangle_ray_binding(p, wrong, vertices, faces)
        with self.assertRaisesRegex(ValueError, 'binding mismatch'):
            validate_triangle_ray_binding(p, cam(), vertices + [1., 0, 0], faces)

    def test_invalid_depth_or_barycentric(self):
        for key, value in [('depth', np.nan), ('depth', -1), ('barycentric', np.nan), ('valid', 2)]:
            p = ref(); p[key][0, 0] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_reference(p, cam())

    def test_bad_method_depth_is_not_dropped_by_world_roi(self):
        ps = {m: packet() for m in CORE_METHODS}
        ps['jo']['expected_camera_z'][1, 1] = 10000.
        roi = np.zeros((3, 4), bool); roi[1, 1] = True
        masks, count = roi_common_masks(ref()['depth'], ref()['valid'].astype(bool), ps, CORE_METHODS, roi)
        self.assertTrue(masks['common_primary'][1, 1])
        self.assertEqual(count['common_primary']['coverage'], 1.)

    def test_common_denominator_reports_failure(self):
        ps = {m: packet() for m in CORE_METHODS}; ps['jo']['numeric_valid'][1, 1] = 0
        masks, count = roi_common_masks(ref()['depth'], ref()['valid'].astype(bool), ps, CORE_METHODS, np.ones((3, 4), bool))
        self.assertEqual(count['common_primary']['reference_roi_pixels'], 12)
        self.assertEqual(count['common_primary']['missing_pixels'], 1)
        self.assertEqual(masks['common_primary'].sum(), 11)

    def test_zero_roi_is_null_coverage(self):
        _, count = roi_common_masks(ref()['depth'], ref()['valid'].astype(bool),
            {m: packet() for m in CORE_METHODS}, CORE_METHODS, np.zeros((3, 4), bool))
        self.assertIsNone(count['common_primary']['coverage'])

    def test_sensitivity_does_not_override_primary(self):
        ps = {m: packet() for m in CORE_METHODS}; ps['jo']['camera_z_variance'][0, 0] = -1
        masks, _ = roi_common_masks(ref()['depth'], ref()['valid'].astype(bool), ps, CORE_METHODS, np.ones((3, 4), bool))
        self.assertEqual(masks['common_primary'].sum(), 12)
        self.assertEqual(masks['common_variance_valid'].sum(), 11)

    def test_missing_target_does_not_create_primary_mean(self):
        self.assertIsNone(complete_mean([1., None], 2)['value'])

    def test_qualification_keeps_old_failure_and_checks_frozen_depth_range(self):
        original = dict(failed_gates=['mesh_sparse_bbox_diag_ratio_out_of_range'], caution_gates=[])
        module = SimpleNamespace(evaluate_audit_against_gates=lambda audit, gates: original)
        gates = dict(depth_range=dict(depth_p98_over_p02_max=200.), heldout_triangle_render=dict(
            minimum_per_target_coverage_pass=.2, minimum_median_coverage_pass=.3,
            usable_target_coverage_threshold=.2, minimum_usable_heldout_target_fraction=.5))
        row = dict(image_name='view', reference_roi_pixels=1, positive_depth_p98_over_p02=10.,
            full_mesh_pixels=12, camera=dict(width=4, height=3))
        with patch.dict('sys.modules', {'tools.depth_reference_geometry_v2.openmvs_campaign_core': module}):
            out = qualify_reference({}, gates, [row], original)
            self.assertTrue(out['scorable'])
            self.assertEqual(out['original_qualification']['failed_gates'], original['failed_gates'])
            row['positive_depth_p98_over_p02'] = 201.
            self.assertIn('reference_depth_range:view', qualify_reference({}, gates, [row], original)['failed_gates'])
            row.update(positive_depth_p98_over_p02=10., full_mesh_pixels=1)
            self.assertIn('actual_minimum_usable_heldout_target_fraction', qualify_reference({}, gates, [row], original)['failed_gates'])

    def test_five_method_index_cannot_shrink_or_resize(self):
        from .proxy_roi import PROTOCOL
        roi = dict(scene='five_k', target_names=['view'], targets=[dict(image_name='view', camera=cam())])
        index = dict(schema='umgs_core_roi_proxy_packet_index_v1', protocol=PROTOCOL,
            scene='five_k', target_names=['view'], methods=list(CORE_METHODS),
            depth_units='source_common_SfM_reconstruction_units_not_metres', resized_depth=False,
            method_specific_alignment=False, same_reference_camera_verified=True, all_five_methods_bound=True,
            targets=[dict(image_name='view', reference_camera_sha256=cam()['record_sha256'],
                packets={m: dict(path='shared', sha256='0'*64) for m in CORE_METHODS})])
        index['records_root_sha256'] = record_hash(index)
        self.assertIn('view', validate_packet_index(index, roi))
        for field, value in [('resized_depth', True), ('methods', list(CORE_METHODS[:-1]))]:
            altered = {**index, field: value}; altered.pop('records_root_sha256')
            altered['records_root_sha256'] = record_hash(altered)
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_packet_index(altered, roi)
        altered = {**index, 'records_root_sha256': '0'*64}
        with self.assertRaisesRegex(ValueError, 'root'):
            validate_packet_index(altered, roi)


if __name__ == '__main__':
    unittest.main(verbosity=2)
