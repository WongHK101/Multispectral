"""Static/unit checks for the opt-in expected-camera-z exporter patch.

These tests do not import the CUDA extension and do not render.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.depth_reference_geometry_v2.export_umgs_expected_camera_z_packet import (
    PACKET_CHANNELS,
    REQUIRED_RUNTIME_SOURCE_PATHS,
    build_parser,
    camera_fingerprint,
    canonical_json_sha256,
    packet_schema,
    packet_tensor_to_named_arrays,
    sha256_file,
    verify_colmap_sparse_hash_manifest,
    verify_expected_camera_fingerprint,
    verify_heldout_manifest,
    verify_runtime_source_hash_manifest,
)


def test_packet_schema_uses_named_expected_camera_z_arrays() -> None:
    schema = packet_schema()
    assert schema["schema_version"] == "umgs_expected_camera_z_packet_v1"
    assert schema["channels"]["accumulated_opacity"] == 0
    assert schema["channels"]["weighted_camera_z_sum"] == 1
    assert schema["channels"]["expected_camera_z"] == 2
    assert schema["channels"]["numeric_valid"] == 3
    assert "not physical metre-unit surface depth" in schema["description"]
    assert "expected_camera_z > 0" in schema["numeric_valid_formula"]


def test_packet_tensor_to_named_arrays_recomputes_numeric_valid_and_variance() -> None:
    packet = torch.zeros((len(PACKET_CHANNELS), 2, 4), dtype=torch.float32)
    packet[0] = torch.tensor([[0.1, 0.0, 0.2, 0.3], [0.3, 0.4, 0.5, 0.6]])
    packet[1] = torch.tensor([[0.2, 0.0, -1.0, 0.6], [0.6, 1.2, 2.0, 1.8]])
    packet[2] = torch.tensor([[2.0, float("nan"), -5.0, 2.0], [2.0, 3.0, 4.0, 3.0]])
    packet[3] = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    packet[4] = packet[1] * packet[2]
    packet[5] = torch.tensor([[0.0, 0.0, 0.0, 0.0], [-5e-7, -2e-6, 0.2, float("nan")]])

    arrays = packet_tensor_to_named_arrays(packet, opacity_epsilon=1e-6, variance_clamp_tolerance=1e-6)
    assert set(arrays) == set(PACKET_CHANNELS)
    np.testing.assert_allclose(arrays["expected_camera_z"][0, 0], 2.0)
    expected_valid = np.array([[1, 0, 0, 0], [1, 0, 1, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(arrays["numeric_valid"], expected_valid)
    assert arrays["camera_z_variance"][1, 0] == 0.0
    assert np.isnan(arrays["camera_z_variance"][1, 1])
    assert np.isnan(arrays["expected_camera_z"][0, 2])


def test_manifest_hash_and_heldout_membership_preflight() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        train = root / "train.txt"
        test = root / "test.txt"
        train.write_text("DJI_train_D.JPG\n", encoding="utf-8")
        test.write_text("DJI_20260526170850_0001_D.JPG\n", encoding="utf-8")
        camera_manifest = root / "heldout.json"
        split_manifest = root / "split.json"
        camera_manifest.write_text(
            json.dumps(
                {
                    "schema": "source_image_only_evaluation_heldout_cameras_v1",
                    "heldout_images": [
                        {
                            "image_id": 1,
                            "image_name": "DJI_20260526170850_0001_D.JPG",
                            "camera_id": 1,
                            "rgb_path": "/tmp/DJI_20260526170850_0001_D.JPG",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        split_manifest.write_text(
            json.dumps({"schema": "split", "split": {"train_file": str(train), "test_file": str(test)}}),
            encoding="utf-8",
        )
        args = argparse.Namespace(
            camera_manifest=str(camera_manifest),
            camera_manifest_sha256=sha256_file(camera_manifest),
            split_manifest=str(split_manifest),
            split_manifest_sha256=sha256_file(split_manifest),
            train_file_sha256=sha256_file(train),
            test_file_sha256=sha256_file(test),
            target="DJI_20260526170850_0001_D",
        )
        info = verify_heldout_manifest(args)
        assert info["camera_manifest"]["target_row"]["image_id"] == 1
        assert info["split_manifest"]["target_in_test_file"] is True
        assert info["split_manifest"]["target_in_train_file"] is False


def test_split_files_missing_fail_fast() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        missing_train = root / "missing_train.txt"
        missing_test = root / "missing_test.txt"
        camera_manifest = root / "heldout.json"
        split_manifest = root / "split.json"
        camera_manifest.write_text(
            json.dumps(
                {
                    "schema": "source_image_only_evaluation_heldout_cameras_v1",
                    "heldout_images": [
                        {"image_id": 1, "image_name": "DJI_20260526170850_0001_D.JPG", "camera_id": 1}
                    ],
                }
            ),
            encoding="utf-8",
        )
        split_manifest.write_text(
            json.dumps({"schema": "split", "split": {"train_file": str(missing_train), "test_file": str(missing_test)}}),
            encoding="utf-8",
        )
        args = argparse.Namespace(
            camera_manifest=str(camera_manifest),
            camera_manifest_sha256=sha256_file(camera_manifest),
            split_manifest=str(split_manifest),
            split_manifest_sha256=sha256_file(split_manifest),
            train_file_sha256="0" * 64,
            test_file_sha256="0" * 64,
            target="DJI_20260526170850_0001_D",
        )
        try:
            verify_heldout_manifest(args)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("missing split files must fail fast")


def test_train_sha_mismatch_fails_before_membership() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        train = root / "train.txt"
        test = root / "test.txt"
        train.write_text("DJI_train_D.JPG\n", encoding="utf-8")
        test.write_text("DJI_20260526170850_0001_D.JPG\n", encoding="utf-8")
        camera_manifest = root / "heldout.json"
        split_manifest = root / "split.json"
        camera_manifest.write_text(
            json.dumps({"heldout_images": [{"image_id": 1, "image_name": "DJI_20260526170850_0001_D.JPG"}]}),
            encoding="utf-8",
        )
        split_manifest.write_text(
            json.dumps({"split": {"train_file": str(train), "test_file": str(test)}}),
            encoding="utf-8",
        )
        args = argparse.Namespace(
            camera_manifest=str(camera_manifest),
            camera_manifest_sha256=sha256_file(camera_manifest),
            split_manifest=str(split_manifest),
            split_manifest_sha256=sha256_file(split_manifest),
            train_file_sha256="0" * 64,
            test_file_sha256=sha256_file(test),
            target="DJI_20260526170850_0001_D",
        )
        try:
            verify_heldout_manifest(args)
        except ValueError as exc:
            assert "train split file SHA256 mismatch" in str(exc)
        else:
            raise AssertionError("train SHA mismatch must fail")


def test_test_sha_mismatch_fails_before_membership() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        train = root / "train.txt"
        test = root / "test.txt"
        train.write_text("DJI_train_D.JPG\n", encoding="utf-8")
        test.write_text("DJI_20260526170850_0001_D.JPG\n", encoding="utf-8")
        camera_manifest = root / "heldout.json"
        split_manifest = root / "split.json"
        camera_manifest.write_text(
            json.dumps({"heldout_images": [{"image_id": 1, "image_name": "DJI_20260526170850_0001_D.JPG"}]}),
            encoding="utf-8",
        )
        split_manifest.write_text(
            json.dumps({"split": {"train_file": str(train), "test_file": str(test)}}),
            encoding="utf-8",
        )
        args = argparse.Namespace(
            camera_manifest=str(camera_manifest),
            camera_manifest_sha256=sha256_file(camera_manifest),
            split_manifest=str(split_manifest),
            split_manifest_sha256=sha256_file(split_manifest),
            train_file_sha256=sha256_file(train),
            test_file_sha256="0" * 64,
            target="DJI_20260526170850_0001_D",
        )
        try:
            verify_heldout_manifest(args)
        except ValueError as exc:
            assert "test split file SHA256 mismatch" in str(exc)
        else:
            raise AssertionError("test SHA mismatch must fail")


def _write_sparse_manifest(root: Path, bad_file_hash: bool = False) -> Path:
    sparse = root / "sparse" / "0"
    sparse.mkdir(parents=True)
    rows = []
    for name in ["cameras.bin", "images.bin", "points3D.bin", "train.txt", "test.txt"]:
        p = sparse / name
        p.write_bytes(f"{name}\n".encode("utf-8"))
        rows.append({"file": name, "path": str(p), "sha256": "0" * 64 if bad_file_hash and name == "images.bin" else sha256_file(p)})
    manifest = root / "COLMAP_SPARSE_HASHES.csv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "path", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def test_sparse_manifest_sha_mismatch_fails() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        manifest = _write_sparse_manifest(root)
        args = argparse.Namespace(
            source_path=str(root),
            colmap_sparse_hash_manifest=str(manifest),
            colmap_sparse_hash_manifest_sha256="0" * 64,
        )
        try:
            verify_colmap_sparse_hash_manifest(args)
        except ValueError as exc:
            assert "COLMAP sparse hash manifest SHA256 mismatch" in str(exc)
        else:
            raise AssertionError("sparse manifest SHA mismatch must fail")


def test_sparse_file_hash_mismatch_fails() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        manifest = _write_sparse_manifest(root, bad_file_hash=True)
        args = argparse.Namespace(
            source_path=str(root),
            colmap_sparse_hash_manifest=str(manifest),
            colmap_sparse_hash_manifest_sha256=sha256_file(manifest),
        )
        try:
            verify_colmap_sparse_hash_manifest(args)
        except ValueError as exc:
            assert "COLMAP sparse file images.bin SHA256 mismatch" in str(exc)
        else:
            raise AssertionError("sparse file hash mismatch must fail")


def test_runtime_source_hash_manifest_requires_core_files() -> None:
    with tempfile.TemporaryDirectory() as td:
        manifest = Path(td) / "runtime_sources.csv"
        with manifest.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "sha256"])
            writer.writeheader()
            for rel in REQUIRED_RUNTIME_SOURCE_PATHS:
                writer.writerow({"path": rel, "sha256": sha256_file(REPO_ROOT / rel)})
        rows = verify_runtime_source_hash_manifest(manifest, sha256_file(manifest), REPO_ROOT)
        assert {r["path"] for r in rows} >= set(REQUIRED_RUNTIME_SOURCE_PATHS)


class DummyCamera:
    image_name = "DJI_20260526170850_0001_D.JPG"
    colmap_id = 1
    uid = 0
    image_width = 1200
    image_height = 869
    FoVx = 1.3
    FoVy = 1.0
    znear = 0.01
    zfar = 100.0
    R = np.eye(3, dtype=np.float32)
    T = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    world_view_transform = torch.eye(4, dtype=torch.float32)
    projection_matrix = torch.eye(4, dtype=torch.float32)
    full_proj_transform = torch.eye(4, dtype=torch.float32)


def _write_expected_fingerprint(path: Path, fingerprint: dict) -> Path:
    path.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
    return path


def test_camera_fingerprint_exact_match_passes() -> None:
    with tempfile.TemporaryDirectory() as td:
        actual = camera_fingerprint(DummyCamera())
        fp_path = _write_expected_fingerprint(Path(td) / "fingerprint.json", actual)
        args = argparse.Namespace(
            expected_camera_fingerprint=str(fp_path),
            expected_camera_fingerprint_sha256=sha256_file(fp_path),
            camera_fingerprint_atol=1e-8,
        )
        comparison = verify_expected_camera_fingerprint(actual, args)
        assert comparison["matched"] is True


def test_expected_camera_fingerprint_file_sha_mismatch_fails() -> None:
    with tempfile.TemporaryDirectory() as td:
        actual = camera_fingerprint(DummyCamera())
        fp_path = _write_expected_fingerprint(Path(td) / "fingerprint.json", actual)
        args = argparse.Namespace(
            expected_camera_fingerprint=str(fp_path),
            expected_camera_fingerprint_sha256="0" * 64,
            camera_fingerprint_atol=1e-8,
        )
        try:
            verify_expected_camera_fingerprint(actual, args)
        except ValueError as exc:
            assert "expected camera fingerprint JSON SHA256 mismatch" in str(exc)
        else:
            raise AssertionError("expected camera fingerprint file SHA mismatch must fail")


def test_camera_discrete_field_mismatch_fails() -> None:
    with tempfile.TemporaryDirectory() as td:
        actual = camera_fingerprint(DummyCamera())
        expected = json.loads(json.dumps(actual))
        expected["payload"]["image_name"] = "DIFFERENT.JPG"
        expected["fingerprint_sha256"] = canonical_json_sha256(expected["payload"])
        fp_path = _write_expected_fingerprint(Path(td) / "fingerprint.json", expected)
        args = argparse.Namespace(
            expected_camera_fingerprint=str(fp_path),
            expected_camera_fingerprint_sha256=sha256_file(fp_path),
            camera_fingerprint_atol=1e-8,
        )
        try:
            verify_expected_camera_fingerprint(actual, args)
        except ValueError as exc:
            assert "target camera fingerprint mismatch" in str(exc)
        else:
            raise AssertionError("camera discrete mismatch must fail")


def test_camera_floating_field_over_tolerance_fails() -> None:
    with tempfile.TemporaryDirectory() as td:
        actual = camera_fingerprint(DummyCamera())
        expected = json.loads(json.dumps(actual))
        expected["payload"]["FoVx"] = expected["payload"]["FoVx"] + 1e-4
        expected["fingerprint_sha256"] = canonical_json_sha256(expected["payload"])
        fp_path = _write_expected_fingerprint(Path(td) / "fingerprint.json", expected)
        args = argparse.Namespace(
            expected_camera_fingerprint=str(fp_path),
            expected_camera_fingerprint_sha256=sha256_file(fp_path),
            camera_fingerprint_atol=1e-8,
        )
        try:
            verify_expected_camera_fingerprint(actual, args)
        except ValueError as exc:
            assert "target camera fingerprint mismatch" in str(exc)
        else:
            raise AssertionError("camera floating mismatch over tolerance must fail")


def test_public_renderer_interface_is_opt_in_and_not_metric_named() -> None:
    text = (REPO_ROOT / "gaussian_renderer" / "__init__.py").read_text(encoding="utf-8")
    assert "return_expected_camera_z_packet=False" in text
    assert "expected_camera_z_packet" in text
    assert "normalization_epsilon" not in text
    assert "return_metric_depth_packet" not in text
    assert "metric_depth_packet" not in text


def test_parser_requires_runtime_binary_binding_arguments() -> None:
    help_text = build_parser().format_help()
    assert "--rasterizer-extension-path" in help_text
    assert "--rasterizer-extension-sha256" in help_text
    assert "--runtime-source-hash-manifest" in help_text
    assert "--train-file-sha256" in help_text
    assert "--test-file-sha256" in help_text
    assert "--colmap-sparse-hash-manifest" in help_text
    assert "--colmap-sparse-hash-manifest-sha256" in help_text
    assert "--expected-camera-fingerprint" in help_text
    assert "--expected-camera-fingerprint-sha256" in help_text


def test_rasterizer_python_binding_is_opt_in() -> None:
    text = (
        REPO_ROOT
        / "submodules"
        / "diff-gaussian-rasterization"
        / "diff_gaussian_rasterization"
        / "__init__.py"
    ).read_text(encoding="utf-8")
    assert "return_expected_camera_z_packet : bool = False" in text
    assert "opacity_epsilon : float = 1e-6" in text
    assert "variance_clamp_tolerance : float = 1e-6" in text
    assert "normalization_epsilon" not in text
    assert "return_metric_depth_packet" not in text


if __name__ == "__main__":
    test_packet_schema_uses_named_expected_camera_z_arrays()
    test_packet_tensor_to_named_arrays_recomputes_numeric_valid_and_variance()
    test_manifest_hash_and_heldout_membership_preflight()
    test_split_files_missing_fail_fast()
    test_train_sha_mismatch_fails_before_membership()
    test_test_sha_mismatch_fails_before_membership()
    test_sparse_manifest_sha_mismatch_fails()
    test_sparse_file_hash_mismatch_fails()
    test_runtime_source_hash_manifest_requires_core_files()
    test_camera_fingerprint_exact_match_passes()
    test_expected_camera_fingerprint_file_sha_mismatch_fails()
    test_camera_discrete_field_mismatch_fails()
    test_camera_floating_field_over_tolerance_fails()
    test_public_renderer_interface_is_opt_in_and_not_metric_named()
    test_parser_requires_runtime_binary_binding_arguments()
    test_rasterizer_python_binding_is_opt_in()
    print("expected-camera-z exporter A0 guard static tests passed")
