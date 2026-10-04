import copy
import unittest
from types import SimpleNamespace

import numpy as np

from .common_geometry import common_control_geometry, heldout_surface_views, validate_source_sidecar
from .contracts import record_hash


def refresh(sidecar):
    for v in sidecar["views"]:
        v["mapping_sha256"] = record_hash({k: x for k, x in v.items() if k != "mapping_sha256"})
    by_name = {v["image_name"]: v for v in sidecar["views"]}
    for row in sidecar["observations"]:
        row["mapping_sha256"] = by_name[row["image_name"]]["mapping_sha256"]
    sidecar["records_root_sha256"] = record_hash({k: sidecar[k] for k in ("views", "observations")})


def fixture():
    positions = [(-1., 0., 5.), (1., 0., 5.), (0., 1., 5.)]
    points = {f"p{i}": np.array(x) for i, x in enumerate(
        [(-.2, -.2, 0.), (.2, -.2, 0.), (.2, .2, 0.), (-.2, .2, .1), (0., 0., 0.)])}
    roles = {k: "control" if k != "p4" else "checkpoint" for k in points}
    sidecar = {"schema": "umgs_old_camera_raw_click_sidecar_v1", "scene": "synthetic",
               "views": [], "observations": []}
    adapter = {"groups": []}
    r = np.diag([1., -1., -1.])
    for i, centre in enumerate(positions):
        p = np.eye(4)
        p[:3, :3], p[:3, 3] = r, -r @ centre
        view = {"image_name": f"image{i}", "image_id": i+1, "camera_id": 1,
            "source_w2c_opencv": p.tolist(), "source_c2w_opencv": np.linalg.inv(p).tolist(),
            "native_camera_array": dict(width=100, height=100, fx=30., fy=30., cx=50., cy=50.),
            "native_file": f"images/D/{i}.png", "native_sha256": str(i)*64}
        sidecar["views"].append(view)
        adapter["groups"].append({"image_name": view["image_name"], "image_id": i+1, "camera_id": 1,
            "split": "eval" if i == 0 else "train", "output_images": {"D": {
                "sha256": view["native_sha256"], "relative_path": view["native_file"], "width": 100, "height": 100}}})
        for name, xyz in points.items():
            cam = r @ xyz + p[:3, 3]
            ray = cam[:2] / cam[2]
            sidecar["observations"].append({"observation_id": f"{name}_{i}", "image_name": view["image_name"],
                "point_name": name, "old_camera_ray_xy": ray.tolist(), "native_array_xy": (ray*30+50).tolist(),
                "formal_eligible": True, "formal_role": roles[name], "annotation_quality": "good", "in_bounds": True})
    refresh(sidecar)
    return sidecar, adapter, points, roles


class CommonGeometryTests(unittest.TestCase):
    def setUp(self):
        self.sidecar, self.adapter, self.points, self.roles = fixture()
        self.colmap = SimpleNamespace(rotmat2qvec=lambda _: [0, 1, 0, 0],
                                      qvec2rotmat=lambda _: np.diag([1., -1., -1.]))

    def solve(self, targets=None):
        targets = targets if targets is not None else {n: self.points[n] for n in self.points if n != "p4"}

        def triangulate(obs, cams, images):
            rows = []
            for o in obs:
                image = images[o["image_name"]]
                p = np.column_stack((self.colmap.qvec2rotmat(image.qvec), image.tvec))
                self.assertEqual(cams[image.camera_id].params, [1., 1., 0., 0.])
                rows.extend([o["u_px"]*p[2]-p[0], o["v_px"]*p[2]-p[1]])
            h = np.linalg.svd(np.array(rows))[2][-1]
            return h[:3]/h[3]

        def fit(source, target, estimate_scale):
            self.assertTrue(estimate_scale)
            self.assertEqual(source.shape, (4, 3))
            np.testing.assert_allclose(source, target, atol=1e-14)
            return 1., np.eye(3), np.zeros(3)

        return common_control_geometry(self.sidecar, self.roles, targets, triangulator=triangulate,
                                       fitter=fit, colmap=self.colmap)

    def test_control_only_and_view_groups(self):
        result = self.solve()
        self.assertEqual(result["control_count"], 4)
        self.assertEqual(result["checkpoint_count"], 1)
        self.assertEqual(result["formal_observation_count"], 15)
        self.assertEqual(result["transform"]["excluded_checkpoints"], ["p4"])
        self.assertTrue(all(r["view_class"] == "nadir" for r in result["observations"]))
        self.assertFalse(result["method_metrics_generated"])
        self.assertEqual(result, self.solve())

    def test_checkpoint_or_missing_control_target_rejected(self):
        for targets in (self.points, {k: v for k, v in self.points.items() if k not in {"p0", "p4"}}):
            with self.assertRaises(ValueError):
                self.solve(targets)

    def test_mapping_pose_ray_tamper(self):
        original = copy.deepcopy(self.sidecar)
        self.sidecar["views"][0]["source_w2c_opencv"][0][3] += 2
        refresh(self.sidecar)
        with self.assertRaises(ValueError):
            validate_source_sidecar(self.sidecar)
        self.sidecar = original
        self.sidecar["observations"][0]["native_array_xy"][0] += 1
        refresh(self.sidecar)
        with self.assertRaises(ValueError):
            validate_source_sidecar(self.sidecar)

    def test_role_and_population_tamper(self):
        self.sidecar["observations"][0]["formal_role"] = "checkpoint"
        refresh(self.sidecar)
        with self.assertRaises(ValueError):
            self.solve()

    def test_surface_actual_split_and_identity(self):
        result = heldout_surface_views(self.sidecar, self.adapter)
        self.assertEqual(result["counts"], {"train": 2, "eval": 1})
        self.assertEqual([r["image_name"] for r in result["views"]], ["image0"])
        self.adapter["groups"][0]["output_images"]["D"]["sha256"] = "f"*64
        with self.assertRaises(ValueError):
            heldout_surface_views(self.sidecar, self.adapter)

    def test_duplicate_or_missing_surface_view(self):
        for groups in (self.adapter["groups"] + [self.adapter["groups"][0]], self.adapter["groups"][:-1]):
            with self.assertRaises(ValueError):
                heldout_surface_views(self.sidecar, {"groups": groups})


if __name__ == "__main__":
    unittest.main()
