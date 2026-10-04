"""Synthetic CPU checks. No experiment tensors, renderer imports or GPU queries."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .camera_bridge import (Pinhole, compare_poses, inverse_dataparser_points,
                            map_rays, projection_to_pinhole, source_unit_accumulators,
                            camera_to_array_intrinsics)
from .contracts import (FLOAT_TENSORS, PACKET_SCHEMA, PRIMARY_TENSOR, canonical_bytes,
                        read_json, read_npz_headers, safe_member, sha256,
                        validate_packet_headers, verify_source_snapshot)
from .method_recipe import channel_delays, inspect_resolved_recipe, verify_training_split


class CameraTests(unittest.TestCase):
    def test_gsplat_corner_sample_matches_index_camera(self):
        camera = Pinhole(707, 512, 461.25, 462.5, 353.5, 256)
        index = camera_to_array_intrinsics(camera, sampling_convention="camera_corner_origin_samples_at_half_v1")
        pixels = np.array([[0, 0], [17.2, 45.3], [706, 511]])
        np.testing.assert_allclose(index.unproject(pixels), camera.unproject(pixels + .5), atol=1e-15, rtol=0)

    def test_already_index_projection_is_not_offset_twice(self):
        camera = Pinhole(707, 512, 461.25, 462.5, 353, 255.5)
        self.assertIs(camera_to_array_intrinsics(camera, sampling_convention="array_index_integer_samples_v1"), camera)

    def test_offcenter_camera_principal_point_is_preserved(self):
        camera = Pinhole(100, 80, 60, 70, 47.2, 33.1)
        index = camera_to_array_intrinsics(camera, sampling_convention="camera_corner_origin_samples_at_half_v1")
        self.assertEqual((index.cx, index.cy), (46.7, 32.6))
        self.assertNotEqual(index.cx, (camera.width-1)/2)

    def test_center_token_alone_cannot_choose_projection(self):
        camera = Pinhole(100, 80, 60, 70, 50, 40)
        for token in ("zero_based_pixel_centers", "zero_indexed_pixel_centers", "R8"):
            with self.subTest(token=token), self.assertRaisesRegex(ValueError, "sampling convention"):
                camera_to_array_intrinsics(camera, sampling_convention=token)

    def test_graphdeco_center_is_not_corner_origin(self):
        p = np.zeros((4, 4)); p[0, 0] = 2; p[1, 1] = 3; p[2, 3] = 1
        a = projection_to_pinhole(p, 1200, 869, convention="graphdeco_ndc_index_centers_v1")
        b = projection_to_pinhole(p, 1200, 869, convention="corner_origin_centers_plus_half_v1")
        self.assertEqual((a.cx, a.cy), (599.5, 434.0))
        self.assertEqual((b.cx, b.cy), (600.0, 434.5))
        np.testing.assert_array_equal(b.project([[0, 0]]) - a.project([[0, 0]]), [[.5, .5]])

    def test_matrix_projection_matches_formula(self):
        p = np.zeros((4, 4)); p[0, 0] = 1.7; p[1, 1] = 1.9
        p[2, 0] = .07; p[2, 1] = -.04; p[2, 3] = 1
        xyz = np.array([[0, 0, 1], [-.4, .3, 2], [.3, -.8, 3], [.1, .2, .6]])
        for width, height in [(5654, 4098), (1414, 1024), (707, 512)]:
            c = projection_to_pinhole(p, width, height, convention="graphdeco_ndc_index_centers_v1")
            clip = np.column_stack([xyz, np.ones(len(xyz))]) @ p
            expected = ((clip[:, :2] / clip[:, 3:] + 1) * [width, height] - 1) * .5
            np.testing.assert_allclose(c.project(xyz[:, :2] / xyz[:, 2:]), expected, atol=1e-9, rtol=0)

    def test_unknown_convention_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown"):
            projection_to_pinhole(np.eye(4), 10, 8, convention="zero_indexed_alias_guess")

    def test_transposed_projection_rejected(self):
        p = np.zeros((4, 4)); p[0, 0] = p[1, 1] = 1; p[2, 3] = 1; p[3, 2] = -.01
        with self.assertRaises(ValueError):
            projection_to_pinhole(p.T, 10, 8, convention="graphdeco_ndc_index_centers_v1")

    def test_rounding_anisotropic_ray_equivalence(self):
        source = Pinhole(5654, 4098, 3704, 3704, 2827, 2049)
        pixels = np.array([[1000.4, 500.7], [3340.9, 1821.3]])
        for divisor in (1, 2, 4, 8):
            w, h = round(5654 / divisor), round(4098 / divisor)
            target = Pinhole(w, h, 3704*w/5654, 3704*h/4098, (w-1)/2, (h-1)/2)
            result = map_rays(source, target, pixels)
            self.assertTrue(result["in_bounds"].all())
            self.assertLessEqual(result["coordinate_error"].max(), 1e-12)
            self.assertLessEqual(result["angular_error_rad"].max(), 1e-7)

    def test_python_round_ties_golden(self):
        self.assertEqual((round(5654/4), round(4098/4)), (1414, 1024))
        self.assertEqual((round(10/4), round(14/4)), (2, 4))

    def test_oob_is_explicit(self):
        cam = Pinhole(10, 10, 4, 4, 5, 5)
        self.assertEqual(map_rays(cam, cam, [[-1, 1], [10, 3]])["in_bounds"].tolist(), [False, False])

    def test_nonfinite_intrinsics_rejected(self):
        for bad in (float("nan"), float("inf"), -1, 0):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                Pinhole(10, 10, bad, 4, 5, 5)

    def test_pose_identity(self):
        r = compare_poses(np.eye(4), np.eye(4), center_atol=1e-8, angle_atol=1e-8)
        self.assertEqual(r["center_difference"], 0)

    def test_pose_translation_rejected(self):
        b = np.eye(4); b[0, 3] = .01
        with self.assertRaisesRegex(ValueError, "Non-equivalent"):
            compare_poses(np.eye(4), b, center_atol=1e-8, angle_atol=1e-8)

    def test_pose_rotation_rejected(self):
        b = np.eye(4); b[:2, :2] = [[0, -1], [1, 0]]
        with self.assertRaisesRegex(ValueError, "Non-equivalent"):
            compare_poses(np.eye(4), b, center_atol=1e-8, angle_atol=1e-8)

    def test_axis_flip_rejected(self):
        b = np.eye(4); b[0, 0] = -1
        with self.assertRaisesRegex(ValueError, "reflection"):
            compare_poses(np.eye(4), b, center_atol=1e-8, angle_atol=1e-8)

    def test_dataparser_inverse_includes_translation_and_scale(self):
        m = np.eye(4); m[:2, :2] = [[0, -1], [1, 0]]; m[:3, 3] = [12, -3, 6]
        x = np.array([[1, 2, 3], [-3, 5, 7]], dtype=float)
        normalized = 2.5 * (x @ m[:3, :3].T + m[:3, 3])
        np.testing.assert_allclose(inverse_dataparser_points(normalized, m, 2.5), x, atol=1e-12, rtol=0)

    def test_scale_and_shear_rejected(self):
        for scale in (0, -1, float("nan")):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                inverse_dataparser_points([[1, 2, 3]], np.eye(4), scale)
        m = np.eye(4); m[0, 1] = .1
        with self.assertRaises(ValueError):
            inverse_dataparser_points([[1, 2, 3]], m, 1)

    def test_moment_units(self):
        raw = dict(zip(FLOAT_TENSORS[:4], (np.array([[x]]) for x in (.5, 2, 8, .125))))
        scaled = source_unit_accumulators(raw, 2)
        self.assertEqual([v.item() for v in scaled.values()], [.5, 1, 2, .25])
        self.assertEqual(raw["weighted_camera_z_sum"].item(), 2)

    def test_legacy_missing_h_never_synthesized(self):
        raw = {key: np.ones((1, 1)) for key in FLOAT_TENSORS[:3]}
        with self.assertRaises(ValueError):
            source_unit_accumulators(raw, 1)

    def test_numeric_overflow_rejected(self):
        raw = {key: np.ones((1, 1)) for key in FLOAT_TENSORS[:4]}
        for scale in (1e-300, 1e300, 0, -1, float("inf")):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                source_unit_accumulators(raw, scale)
        raw["weighted_camera_z_sum"][:] = 1e308
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            source_unit_accumulators(raw, .1)
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            Pinhole(10, 10, 1e308, 1, 0, 0).project([[10, 1]])
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            inverse_dataparser_points([[1e308, 1, 1]], np.eye(4), 1e-10)
        with self.assertRaisesRegex(ValueError, "equivalence"):
            cam = Pinhole(10, 10, 1, 1, 0, 0)
            map_rays(cam, cam, [[1e300, 1e300]])


class PacketTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "synthetic.npz"
        self.arrays = {key: np.zeros((3, 5), dtype="<f4") for key in FLOAT_TENSORS}
        self.arrays["metric_depth_valid_mask"] = np.zeros((3, 5), dtype=bool)
        self.contract = {"schema": PACKET_SCHEMA, "version": 2, "primary_tensor": PRIMARY_TENSOR,
            "semantics": "camera_z", "formula": "M1/A",
            "required_tensor_dtypes": {k: v.dtype.str for k, v in self.arrays.items()}}

    def validate(self):
        np.savez_compressed(self.path, **self.arrays)
        return validate_packet_headers(self.path, expected_sha=sha256(self.path),
                                       contract=self.contract, width=5, height=3)

    def test_headers_without_array_loading(self):
        with patch("numpy.load", side_effect=AssertionError("array body read")):
            result = self.validate()
        self.assertEqual(result["status"], "PASS_HEADERS_ONLY")
        self.assertFalse(result["tensor_values_loaded"])
        self.assertFalse(result["formal_ready"])

    def test_contract_fields_individually_rejected(self):
        for key in ("schema", "version", "primary_tensor", "semantics", "formula"):
            original = self.contract[key]
            with self.subTest(field=key):
                self.contract[key] = "wrong"
                with self.assertRaisesRegex(ValueError, key):
                    self.validate()
            self.contract[key] = original

    def test_missing_tensor_rejected(self):
        del self.arrays["weighted_inverse_camera_z_sum"]
        with self.assertRaisesRegex(ValueError, "Missing required tensor"):
            self.validate()

    def test_float64_dtype_rejected(self):
        self.arrays[PRIMARY_TENSOR] = np.zeros((3, 5), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "dtype/shape"):
            self.validate()

    def test_shape_rejected(self):
        self.arrays[PRIMARY_TENSOR] = np.zeros((5, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "dtype/shape"):
            self.validate()

    def test_declared_dtype_rejected(self):
        self.contract["required_tensor_dtypes"][PRIMARY_TENSOR] = "<f8"
        with self.assertRaisesRegex(ValueError, "declaration"):
            self.validate()

    def test_object_array_rejected_without_unpickle(self):
        self.arrays[PRIMARY_TENSOR] = np.array([[{"unsafe": 1}]], dtype=object)
        with self.assertRaisesRegex(ValueError, "numeric"):
            self.validate()

    def test_packet_sha_rejected(self):
        self.validate()
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            validate_packet_headers(self.path, expected_sha="0"*64, contract=self.contract, width=5, height=3)

    def test_duplicate_zip_members_rejected(self):
        self.validate()
        with zipfile.ZipFile(self.path) as z:
            raw = z.read(PRIMARY_TENSOR + ".npy")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(self.path, "a") as z:
                z.writestr(PRIMARY_TENSOR + ".npy", raw)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            read_npz_headers(self.path)

    def test_short_body_rejected(self):
        self.validate()
        with zipfile.ZipFile(self.path) as z:
            raw = z.read(PRIMARY_TENSOR + ".npy")
        with zipfile.ZipFile(self.path, "w") as z:
            z.writestr(PRIMARY_TENSOR + ".npy", raw[:-1])
        with self.assertRaisesRegex(ValueError, "body size"):
            read_npz_headers(self.path)


class IntegrityAndGuardTests(unittest.TestCase):
    def test_preflight_rejects_input_tree_output_and_overwrite(self):
        from .preflight import run

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = {"release_root": str(root / "release"),
                       "reference_root": str(root / "reference"),
                       "raw_roots": {"scene": str(root / "raw")}}
            for directory in [profile["release_root"], profile["reference_root"], *profile["raw_roots"].values()]:
                with self.subTest(directory=directory), self.assertRaisesRegex(ValueError, "input/reference"):
                    run(profile, Path(directory) / "bad.json")
            output = root / "existing.json"
            output.write_bytes(b"preserve")
            with self.assertRaises(FileExistsError):
                run(profile, output)
            self.assertEqual(output.read_bytes(), b"preserve")

    def test_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for value in ("../escape", "/absolute", "C:/escape", "a/../b", "a\\b", "./file"):
                with self.subTest(path=value), self.assertRaises(ValueError):
                    safe_member(tmp, value)

    def test_json_duplicates_and_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "record.json"
            for value in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
                p.write_text(value)
                with self.subTest(value=value), self.assertRaises(ValueError):
                    read_json(p)

    def test_external_snapshot_pin_and_unregistered_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); p = root / "source.py"; p.write_bytes(b"pass\n")
            manifest = root / "REFERENCE_MANIFEST.json"
            manifest.write_bytes(canonical_bytes({"schema": "gs_gcp_evaluation_reference_source_bundle_v1",
                "source_git_head": "synthetic", "files": [{"path": p.name, "bytes": 5, "sha256": sha256(p)}]}))
            digest = sha256(manifest)
            self.assertEqual(verify_source_snapshot(root, digest)["file_count"], 1)
            with self.assertRaises(ValueError):
                verify_source_snapshot(root, "0"*64)
            (root / "extra.py").write_bytes(b"pass\n")
            with self.assertRaisesRegex(ValueError, "Unregistered"):
                verify_source_snapshot(root, digest)

    def test_guard_blocks_gpu_network_and_external_children(self):
        script = '''
from tools.measured_geometry_v1.cpu_guard import install_cpu_guard, evidence
try: evidence()
except RuntimeError: pass
else: raise AssertionError('Uninstalled guard reported protection')
install_cpu_guard()
import importlib, socket, subprocess, sys
for operation in [lambda: importlib.import_module('torch'),
                  lambda: socket.getaddrinfo('example.org', 80),
                  lambda: subprocess.run([sys.executable, '-V'])]:
    try: operation()
    except RuntimeError: pass
    else: raise AssertionError('CPU boundary bypass')
assert evidence()['gpu_experiment_authorized'] is False
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
try: evidence()
except RuntimeError: pass
else: raise AssertionError('Environment tampering not detected')
'''
        result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True,
                                cwd=Path(__file__).resolve().parents[2])
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_has_no_execute_switch(self):
        result = subprocess.run([sys.executable, "-B", "-m", "tools.measured_geometry_v1.preflight",
                                 "--profile", "unused", "--output", "unused", "--execute"],
                                capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2])
        self.assertEqual(result.returncode, 2)
        self.assertIn("unrecognized arguments", result.stderr)


class RecipeTests(unittest.TestCase):
    def config(self):
        return {"max_num_iterations": 120000, "pipeline": {"model": {
            "camera_optimizer_rgb": {"mode": "off"}, "camera_optimizer_ms": {"mode": "off"},
            "pos_optim_delay_channels": ["D", "500", "MS_G", "MS_NIR", "32000"],
            "opacity_optim_delay_channels": ["MS_G", "MS_NIR", "32000"],
            "use_mlp_channels": [], "use_sh_channels": ["D", "MS_G", "MS_NIR"]}}}

    def inspect(self, value):
        return inspect_resolved_recipe(value, channels=["D", "MS_G", "MS_NIR"], rgb_channel="D")

    def test_jo_is_not_sig_or_neural_color(self):
        result = self.inspect(self.config())
        self.assertEqual(result["mechanism"], "joint_structure_optimization")
        self.assertFalse(result["paper_recipe_verified"])

    def test_isolation_is_not_full_rgb_support_lock(self):
        config = self.config()
        model = config["pipeline"]["model"]
        model["pos_optim_delay_channels"] = ["MS_G", "MS_NIR", "-1"]
        model["opacity_optim_delay_channels"] = ["MS_G", "MS_NIR", "-1"]
        result = self.inspect(config)
        self.assertEqual(result["mechanism"], "non_rgb_structure_gradient_isolation")
        self.assertTrue(result["rgb_structure_can_update"])
        self.assertFalse(result["full_support_lock_proven"])

    def test_neural_color_requires_explicit_mlp_channels(self):
        config = self.config(); config["pipeline"]["model"]["use_mlp_channels"] = ["MS_G", "MS_NIR"]
        self.assertEqual(self.inspect(config)["mechanism"], "neural_spectral_color_enabled")

    def test_camera_optimization_rejected(self):
        for field in ("camera_optimizer_rgb", "camera_optimizer_ms"):
            config = self.config(); config["pipeline"]["model"][field]["mode"] = "SO3xR3"
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.inspect(config)

    def test_malformed_delay_groups_rejected(self):
        for values in (["MS_G"], ["bad", "1"], ["MS_G", "1.5"], ["MS_G", "1", "MS_G", "2"]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                channel_delays(values, ["D", "MS_G"])

    def test_split_partition(self):
        result = verify_training_split(["A", "B"], ["C"], ["A", "B", "C"])
        self.assertTrue(result["disjoint"])

    def test_split_leakage_missing_and_duplicate_rejected(self):
        for train, test in ((["A", "B"], ["B", "C"]), (["A"], ["B"]), (["A", "A"], ["B", "C"])):
            with self.subTest(train=train, test=test), self.assertRaises(ValueError):
                verify_training_split(train, test, ["A", "B", "C"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
