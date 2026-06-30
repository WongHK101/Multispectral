#!/usr/bin/env python3
"""Synthetic tests for OpenMVS campaign geometry and split utilities."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from openmvs_campaign_core import (
    CameraView,
    leakage_audit,
    materialize_source_image_only_model,
    mesh_topology_stats,
    rasterize_mesh_camera_z,
)
from utils.read_write_model import Camera, Image, Point3D, write_model


def identity_view(width: int = 80, height: int = 60, f: float = 50.0) -> CameraView:
    return CameraView(
        image_id=1,
        image_name="target.jpg",
        width=width,
        height=height,
        fx=f,
        fy=f,
        cx=width / 2,
        cy=height / 2,
        camera_to_world=np.eye(4, dtype=np.float64),
    )


def test_planar_mesh_camera_z_and_barycentric() -> None:
    vertices = np.array(
        [
            [-1.0, -1.0, 5.0],
            [1.0, -1.0, 5.0],
            [1.0, 1.0, 5.0],
            [-1.0, 1.0, 5.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    result = rasterize_mesh_camera_z(vertices, faces, identity_view())
    assert result.valid.sum() > 100
    assert np.nanmedian(result.depth) == 5.0
    assert result.triangle_id[result.valid].min() >= 0
    bary = result.barycentric[result.valid]
    assert np.allclose(bary.sum(axis=1), 1.0, atol=1e-5)


def test_front_triangle_occludes_back_triangle() -> None:
    vertices = np.array(
        [
            [-1.0, -1.0, 5.0],
            [1.0, -1.0, 5.0],
            [0.0, 1.0, 5.0],
            [-1.0, -1.0, 8.0],
            [1.0, -1.0, 8.0],
            [0.0, 1.0, 8.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[3, 4, 5], [0, 1, 2]], dtype=np.int64)
    result = rasterize_mesh_camera_z(vertices, faces, identity_view())
    assert result.valid.any()
    assert np.nanmedian(result.depth[result.valid]) == 5.0
    assert 1 in set(result.triangle_id[result.valid].tolist())


def test_behind_camera_triangle_rejected() -> None:
    vertices = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [0.0, 1.0, -1.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    result = rasterize_mesh_camera_z(vertices, faces, identity_view())
    assert result.valid.sum() == 0
    assert np.all(result.triangle_id == -1)


def test_topology_boundary_nonmanifold_degenerate_components() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [10.0, 10.0, 0.0],
            [11.0, 10.0, 0.0],
            [10.0, 11.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [1, 3, 2],
            [1, 4, 3],
            [1, 5, 3],  # makes edge (1,3) non-manifold with three incident faces
            [6, 7, 8],  # second component
            [0, 0, 1],  # degenerate
        ],
        dtype=np.int64,
    )
    stats = mesh_topology_stats(vertices, faces)
    assert stats["connected_component_count"] >= 2
    assert stats["boundary_edge_count"] > 0
    assert stats["non_manifold_edge_count"] >= 1
    assert stats["degenerate_face_count"] >= 1
    assert stats["largest_component_face_ratio"] < 1.0


def test_source_image_only_materializer_prunes_heldout_tracks_and_leakage() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        sparse = root / "sparse" / "0"
        sparse.mkdir(parents=True)
        images_dir = root / "images"
        images_dir.mkdir()
        cameras = {
            1: Camera(id=1, model="PINHOLE", width=100, height=80, params=np.array([50.0, 50.0, 50.0, 40.0])),
        }
        images = {
            1: Image(
                id=1,
                qvec=np.array([1.0, 0.0, 0.0, 0.0]),
                tvec=np.array([0.0, 0.0, 0.0]),
                camera_id=1,
                name="train_A.jpg",
                xys=np.array([[10.0, 10.0], [20.0, 20.0]]),
                point3D_ids=np.array([1, 2]),
            ),
            2: Image(
                id=2,
                qvec=np.array([1.0, 0.0, 0.0, 0.0]),
                tvec=np.array([0.0, 0.0, 0.0]),
                camera_id=1,
                name="target_B.jpg",
                xys=np.array([[11.0, 11.0], [21.0, 21.0]]),
                point3D_ids=np.array([1, 3]),
            ),
        }
        points = {
            1: Point3D(
                id=1,
                xyz=np.array([0.0, 0.0, 5.0]),
                rgb=np.array([255, 255, 255]),
                error=0.1,
                image_ids=np.array([1, 2], dtype=np.int32),
                point2D_idxs=np.array([0, 0], dtype=np.int32),
            ),
            2: Point3D(
                id=2,
                xyz=np.array([1.0, 0.0, 5.0]),
                rgb=np.array([255, 0, 0]),
                error=0.2,
                image_ids=np.array([1], dtype=np.int32),
                point2D_idxs=np.array([1], dtype=np.int32),
            ),
            3: Point3D(
                id=3,
                xyz=np.array([0.0, 1.0, 5.0]),
                rgb=np.array([0, 255, 0]),
                error=0.3,
                image_ids=np.array([2], dtype=np.int32),
                point2D_idxs=np.array([1], dtype=np.int32),
            ),
        }
        write_model(cameras, images, points, str(sparse), ext=".txt")
        (sparse / "train.txt").write_text("train_A.jpg\n", encoding="utf-8")
        (sparse / "test.txt").write_text("target_B.jpg\n", encoding="utf-8")

        out_sparse = root / "source_only" / "sparse" / "0"
        summary = materialize_source_image_only_model(
            sparse_dir=sparse,
            output_sparse_dir=out_sparse,
            image_dir=images_dir,
            min_training_observations=1,
        )
        assert summary["training_image_count"] == 1
        assert summary["excluded_heldout_image_count"] == 1
        assert summary["retained_point_count"] == 2
        assert summary["pruned_point_count"] == 1
        rows, audit = leakage_audit(
            root=root / "source_only",
            reconstruction_sparse_dir=out_sparse,
            heldout_names=["target_B.jpg"],
            min_training_observations=1,
        )
        assert audit["pass"], rows

        # A manifest containing a held-out filename must be caught.
        bad = root / "source_only" / "openmvs_command.log"
        bad.write_text("DensifyPointCloud target_B.jpg\n", encoding="utf-8")
        rows, audit = leakage_audit(
            root=root / "source_only",
            reconstruction_sparse_dir=out_sparse,
            heldout_names=["target_B.jpg"],
            min_training_observations=1,
        )
        assert not audit["pass"]
        assert any(r["check"] == "heldout_filename_in_generated_file" for r in rows)


def main() -> None:
    tests = [
        test_planar_mesh_camera_z_and_barycentric,
        test_front_triangle_occludes_back_triangle,
        test_behind_camera_triangle_rejected,
        test_topology_boundary_nonmanifold_degenerate_components,
        test_source_image_only_materializer_prunes_heldout_tracks_and_leakage,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("ALL_OPENMVS_CAMPAIGN_CORE_TESTS_PASSED")


if __name__ == "__main__":
    main()

