import unittest
from types import SimpleNamespace as S

from .live_packet_adapter import graphdeco_moments, gsplat_moments, validate_ms_export_config


def fixture():
    return S(training=False, _get_downscale_factor=lambda: 1,
             gauss_params={"means": S(is_cuda=False)},
             config=S(opacity_correction_flag=False, camera_optimizer_rgb=S(mode="off"),
                      camera_optimizer_ms=S(mode="off"), rasterize_mode="classic"))


class LiveAdapterPreconditions(unittest.TestCase):
    def test_metadata_only_config_does_not_render(self):
        validate_ms_export_config(fixture())

    def test_training_and_grid_rejected(self):
        for field, value in (("training", True), ("_get_downscale_factor", lambda: 2)):
            m = fixture()
            setattr(m, field, value)
            with self.assertRaises(ValueError):
                validate_ms_export_config(m)

    def test_channel_opacity_pose_and_unknown_mode_rejected(self):
        for change in (lambda c: setattr(c, "opacity_correction_flag", True),
                       lambda c: setattr(c.camera_optimizer_rgb, "mode", "SO3xR3"),
                       lambda c: setattr(c.camera_optimizer_ms, "mode", "SO3xR3"),
                       lambda c: setattr(c, "rasterize_mode", "custom")):
            m = fixture()
            change(m.config)
            with self.assertRaises(ValueError):
                validate_ms_export_config(m)

    def test_no_gpu_does_not_call_native_renderer(self):
        def forbidden(*args, **kwargs):
            self.fail("No native renderer may execute in a CPU precondition test")
        with self.assertRaises(ValueError):
            gsplat_moments(native_rasterization=forbidden, native_get_viewmat=forbidden,
                           model=fixture(), camera=None)
        with self.assertRaises(ValueError):
            graphdeco_moments(native_render=forbidden, camera=None,
                             gaussians=S(get_xyz=S(is_cuda=False)), pipeline=None, background=None)


if __name__ == "__main__":
    unittest.main()
