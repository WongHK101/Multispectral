"""Synthetic tests for the single-packet expected-camera-z validator."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.depth_reference_geometry_v2.validate_umgs_expected_camera_z_single_packet import (  # noqa: E402
    sha256_file,
    validate_packet_dir,
)


EXPECTED = {
    "scene": "maize_02_20260526_1658",
    "target": "DJI_20260526170850_0001_D.JPG",
    "checkpoint": "c" * 64,
    "camera": "a" * 64,
    "split": "b" * 64,
    "train": "1" * 64,
    "test": "2" * 64,
    "sparse": "3" * 64,
    "runtime": "4" * 64,
    "rasterizer": "5" * 64,
    "fingerprint_file": "6" * 64,
    "fingerprint_payload": "7" * 64,
}


class Args:
    packet_dir = ""
    expected_height = 2
    expected_width = 3
    expected_scene = EXPECTED["scene"]
    expected_target = EXPECTED["target"]
    checkpoint_sha256 = EXPECTED["checkpoint"]
    camera_manifest_sha256 = EXPECTED["camera"]
    split_manifest_sha256 = EXPECTED["split"]
    train_file_sha256 = EXPECTED["train"]
    test_file_sha256 = EXPECTED["test"]
    colmap_sparse_hash_manifest_sha256 = EXPECTED["sparse"]
    runtime_source_hash_manifest_sha256 = EXPECTED["runtime"]
    rasterizer_extension_sha256 = EXPECTED["rasterizer"]
    expected_camera_fingerprint_sha256 = EXPECTED["fingerprint_file"]
    expected_camera_fingerprint_payload_sha256 = EXPECTED["fingerprint_payload"]


def base_arrays() -> dict[str, np.ndarray]:
    A = np.array([[0.5, 0.2, 0.9], [0.0, 0.6, 0.7]], dtype=np.float32)
    z = np.array([[2.0, 3.0, 4.0], [np.nan, 5.0, 6.0]], dtype=np.float32)
    valid = np.array([[1, 1, 1], [0, 1, 1]], dtype=np.uint8)
    variance = np.array([[0.1, 0.2, 0.3], [np.nan, 0.4, 0.5]], dtype=np.float32)
    return {
        "accumulated_opacity": A,
        "weighted_camera_z_sum": np.where(valid.astype(bool), A * z, 0.0).astype(np.float32),
        "expected_camera_z": z,
        "numeric_valid": valid,
        "weighted_camera_z2_sum": np.where(valid.astype(bool), A * z * z, 0.0).astype(np.float32),
        "camera_z_variance": variance,
    }


def write_packet(root: Path, arrays: dict[str, np.ndarray] | None = None, manifest_override: dict | None = None) -> Args:
    packet_dir = root / "packet"
    packet_dir.mkdir()
    arr = base_arrays() if arrays is None else arrays
    npz_path = packet_dir / "synthetic_expected_camera_z_packet.npz"
    np.savez_compressed(npz_path, **arr)
    manifest = {
        "schema": {"schema_version": "umgs_expected_camera_z_packet_v1"},
        "scene": EXPECTED["scene"],
        "target": EXPECTED["target"],
        "raster_height": 2,
        "raster_width": 3,
        "checkpoint_sha256": EXPECTED["checkpoint"],
        "target_camera_fingerprint": {"fingerprint_sha256": EXPECTED["fingerprint_payload"]},
        "target_camera_fingerprint_comparison": {
            "expected_file_sha256": EXPECTED["fingerprint_file"],
            "expected_payload_sha256": EXPECTED["fingerprint_payload"],
            "actual_payload_sha256": EXPECTED["fingerprint_payload"],
            "matched": True,
        },
        "preflight": {
            "manifest_info": {
                "camera_manifest": {"sha256": EXPECTED["camera"]},
                "split_manifest": {
                    "sha256": EXPECTED["split"],
                    "train_file_expected_sha256": EXPECTED["train"],
                    "train_file_actual_sha256": EXPECTED["train"],
                    "test_file_expected_sha256": EXPECTED["test"],
                    "test_file_actual_sha256": EXPECTED["test"],
                },
            },
            "colmap_sparse_hash_manifest": {"sha256": EXPECTED["sparse"]},
            "runtime_source_hash_manifest": {"sha256": EXPECTED["runtime"]},
        },
        "rasterizer_extension": {"imported_extension_sha256": EXPECTED["rasterizer"]},
        "npz_sha256": sha256_file(npz_path),
    }
    if manifest_override:
        for key, value in manifest_override.items():
            manifest[key] = value
    (packet_dir / "expected_camera_z_packet_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    args = Args()
    args.packet_dir = str(packet_dir)
    return args


def failed_checks(summary: dict) -> set[str]:
    return {row["check"] for row in summary["checks"] if not row["passed"]}


def test_valid_packet_passes() -> None:
    with tempfile.TemporaryDirectory() as td:
        summary = validate_packet_dir(write_packet(Path(td)))
        assert summary["passed"] is True


def test_missing_key_fails() -> None:
    with tempfile.TemporaryDirectory() as td:
        arr = base_arrays()
        arr.pop("camera_z_variance")
        summary = validate_packet_dir(write_packet(Path(td), arrays=arr))
        assert summary["passed"] is False
        assert "required_npz_arrays_present" in failed_checks(summary)


def test_wrong_raster_fails() -> None:
    with tempfile.TemporaryDirectory() as td:
        args = write_packet(Path(td))
        args.expected_height = 3
        summary = validate_packet_dir(args)
        assert "expected_raster_shape" in failed_checks(summary)


def test_identity_mismatch_fails() -> None:
    with tempfile.TemporaryDirectory() as td:
        arr = base_arrays()
        arr["expected_camera_z"][0, 0] = 99.0
        summary = validate_packet_dir(write_packet(Path(td), arrays=arr))
        assert "expected_z_identity_valid_pixels" in failed_checks(summary)


def test_negative_valid_depth_fails() -> None:
    with tempfile.TemporaryDirectory() as td:
        arr = base_arrays()
        arr["expected_camera_z"][0, 0] = -2.0
        arr["weighted_camera_z_sum"][0, 0] = arr["accumulated_opacity"][0, 0] * -2.0
        summary = validate_packet_dir(write_packet(Path(td), arrays=arr))
        assert "valid_expected_z_positive" in failed_checks(summary)


def test_invalid_opacity_fails() -> None:
    with tempfile.TemporaryDirectory() as td:
        arr = base_arrays()
        arr["accumulated_opacity"][0, 0] = 1.2
        arr["weighted_camera_z_sum"][0, 0] = arr["accumulated_opacity"][0, 0] * arr["expected_camera_z"][0, 0]
        summary = validate_packet_dir(write_packet(Path(td), arrays=arr))
        assert "opacity_bounds_all_pixels" in failed_checks(summary)


def test_variance_below_tolerance_fails() -> None:
    with tempfile.TemporaryDirectory() as td:
        arr = base_arrays()
        arr["camera_z_variance"][0, 0] = -1e-4
        summary = validate_packet_dir(write_packet(Path(td), arrays=arr))
        assert "valid_variance_above_tolerance" in failed_checks(summary)


def test_provenance_hash_mismatch_fails() -> None:
    with tempfile.TemporaryDirectory() as td:
        args = write_packet(Path(td))
        args.checkpoint_sha256 = "0" * 64
        summary = validate_packet_dir(args)
        assert "checkpoint_sha256" in failed_checks(summary)


if __name__ == "__main__":
    test_valid_packet_passes()
    test_missing_key_fails()
    test_wrong_raster_fails()
    test_identity_mismatch_fails()
    test_negative_valid_depth_fails()
    test_invalid_opacity_fails()
    test_variance_below_tolerance_fails()
    test_provenance_hash_mismatch_fails()
    print("single-packet expected-camera-z validator synthetic tests passed")
