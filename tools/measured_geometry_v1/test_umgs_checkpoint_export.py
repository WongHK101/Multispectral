import unittest
import numpy as np

from .umgs_checkpoint_export import archived_camera_parameters, require_locked_support, support_identity, validate_model_artifact


class SupportTest(unittest.TestCase):
    def setUp(self):
        self.values = {key: np.arange(4*width, dtype=np.float32).reshape(4, width)
                       for key, width in zip(('xyz','scaling','rotation','opacity'), (3,3,4,1))}

    def test_exact_support_pass(self):
        self.assertEqual(require_locked_support(self.values, {k:v.copy() for k,v in self.values.items()})['gaussian_count'], 4)

    def test_each_support_property_tamper_rejected(self):
        for key in self.values:
            with self.subTest(key=key):
                other={k:v.copy() for k,v in self.values.items()};other[key][0,0]+=.001
                with self.assertRaises(ValueError):require_locked_support(self.values,other)

    def test_order_count_dtype_and_nonfinite_rejected(self):
        variants=[{k:v[::-1] for k,v in self.values.items()},
                  {k:v[:-1] for k,v in self.values.items()},
                  {k:v.astype(np.float64) for k,v in self.values.items()}]
        bad={k:v.copy() for k,v in self.values.items()};bad['xyz'][0,0]=np.nan;variants.append(bad)
        for other in variants:
            with self.assertRaises(ValueError):require_locked_support(self.values,other)

    def test_missing_support_property_rejected(self):
        other=dict(self.values);other.pop('opacity')
        with self.assertRaises(ValueError):support_identity(other)

    def test_archived_ply_does_not_claim_checkpoint(self):
        row=dict(artifact_kind='archived_final_ply',ply_sha256='a'*64,archive_authority_manifest_sha256='b'*64)
        self.assertEqual(validate_model_artifact(row),'archived_final_ply')
        with self.assertRaises(ValueError): validate_model_artifact(dict(row,checkpoint='missing.pth'))
        with self.assertRaises(ValueError): validate_model_artifact(dict(row,archive_authority_manifest_sha256=''))
        self.assertEqual(validate_model_artifact(dict(checkpoint_sha256='a'*64,ply_sha256='b'*64)), 'checkpoint_and_final_ply')

    def test_archived_camera_pose_intrinsics_and_rounding(self):
        row=dict(rotation=np.eye(3),position=[1.,2.,3.],width=5654,height=4098,fx=3703.,fy=3703.)
        pose=np.eye(4);pose[:3,3]=row['position']
        frame=dict(transform_matrix=(pose@np.diag([1.,-1.,-1.,1.])).tolist(),
                   fl_x=3703.*707/5654,fl_y=3703.*512/4098,cx=707/2,cy=256.)
        image=dict(width=707,height=512)
        view=archived_camera_parameters(row,frame,image,row)
        np.testing.assert_array_equal(view[:3,3],[-1.,-2.,-3.])
        for change in [dict(position=[2.,2.,3.]),dict(fx=3704.),dict(width=5653)]:
            with self.assertRaises((ValueError,AssertionError)):
                archived_camera_parameters(dict(row,**change),frame,image,row)
        with self.assertRaises((ValueError,AssertionError)):
            archived_camera_parameters(row,dict(frame,cx=353.),image,row)


if __name__ == '__main__':unittest.main()
