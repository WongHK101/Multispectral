#!/usr/bin/env python3
"""Synthetic tests for the UMGS-OpenMVS camera-z V1.2 evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import evaluate_umgs_openmvs_camera_z_proxy_alignment as ev


SCENE = "road_01_20260602_1648_40m"
TARGET = "DJI_20260602165038_0001_D.JPG"
CHECKPOINT_SHA = "e3c79257692ca8197e208560d2e001c54bbdef4425d39a4fc11c0301b9d3394a"
UMGS_RUNTIME_SHA = "d06a2331199c2ef44773d31c4cdabcf40cefa054d1206655a7ad1e93bf7753ed"
OPENMVS_RUNTIME_SHA = "a964ce1af9e92449ae48e4cb8e6d1acfd0240d9bd105efa88a3782ac6f3f93e0"
MESH_PLY_SHA = "eb5a4ebe0d3d0053588047437f4c7a9205081c261507ae4b54af43f1e09cf9d9"
MESH_MVS_SHA = "0cf80bdf5b5e41bcdad6339a16578f13f7a556ccfe38662e9309beb5c1ffead0"
ADAPTER_SHA = "1aca514d157740f1dd5bc84d9b8b37a32a5b8732b4556c0ddfbc11bc13238beb"
CORE_SHA = "a2b5e2380199e47b17b5c5b08364885b37510048109fe3f09812142955fbe35e"
SOURCE_SPARSE_ROOT = "/synthetic/source/sparse/0"
MATERIALIZED_SPARSE_ROOT = "/synthetic/materialized/sparse/0"
SOURCE_SPARSE_HASHES = {
    "cameras.bin": "cam-sha",
    "images.bin": "images-sha",
    "points3D.bin": "points-sha",
}
MATERIALIZED_SPARSE_HASHES = {
    "cameras.bin": "cam-sha",
    "images.bin": "materialized-images-sha",
    "points3D.bin": "materialized-points-sha",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def base_grid(shape: tuple[int, int] = (160, 160)) -> np.ndarray:
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    z = 20.0 + 0.05 * xx + 0.02 * yy
    z[:, shape[1] // 2 :] += 3.0
    return z.astype(np.float32)


def make_fingerprint(path: Path, *, image_width: int, image_height: int, target: str = TARGET, tweak: float = 0.0) -> tuple[str, str]:
    payload = {
        "image_name": target,
        "colmap_id": 1,
        "uid": 0,
        "image_width": image_width,
        "image_height": image_height,
        "FoVx": 1.3040666631082152 + tweak,
        "FoVy": 1.0107811730688179,
        "R": [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        "T": [1.0, 2.0, 3.0],
        "world_view_transform": [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [1.0, 2.0, 3.0, 1.0]],
        "projection_matrix": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, -0.01, 0.0]],
        "full_proj_transform": [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, -1.0], [1.0, 2.0, 2.99, 3.0]],
    }
    payload_sha = ev.canonical_payload_hash(payload)
    write_json(path, {"fingerprint_sha256": payload_sha, "payload": payload})
    return sha256(path), payload_sha


def make_runtime_manifest(path: Path, repo_root: Path) -> tuple[str, dict[str, str]]:
    src_dir = repo_root / "synthetic_sources"
    src_dir.mkdir(parents=True, exist_ok=True)
    evaluator = src_dir / "evaluate_umgs_openmvs_camera_z_proxy_alignment.py"
    control = src_dir / "openmvs_da3_overlap_corrected.py"
    evaluator.write_text("synthetic evaluator\n", encoding="utf-8")
    control.write_text("synthetic corrected control\n", encoding="utf-8")
    shas = {"evaluator": sha256(evaluator), "negative_control_source": sha256(control)}
    write_json(
        path,
        {
            "schema": ev.RUNTIME_MANIFEST_SCHEMA,
            "repo_root": str(repo_root),
            "environment": {"python": "synthetic", "numpy": np.__version__, "scipy": ev.scipy.__version__},
            "sources": [
                {"role": "evaluator", "repo_relative_path": str(evaluator.relative_to(repo_root)), "sha256": shas["evaluator"]},
                {"role": "negative_control_source", "repo_relative_path": str(control.relative_to(repo_root)), "sha256": shas["negative_control_source"]},
            ],
        },
    )
    return sha256(path), shas


def make_frame_decision(path: Path, shape: tuple[int, int], fp_file_sha: str, fp_payload_sha: str) -> str:
    write_json(
        path,
        {
            "schema": ev.FRAME_SCALE_DECISION_SCHEMA,
            "status": "correspondence_pass",
            "scene": SCENE,
            "target": TARGET,
            "umgs_checkpoint_sparse_root": SOURCE_SPARSE_ROOT,
            "openmvs_materialization_source_sparse_root": SOURCE_SPARSE_ROOT,
            "openmvs_source_image_only_materialized_sparse_root": MATERIALIZED_SPARSE_ROOT,
            "source_sparse_hashes": SOURCE_SPARSE_HASHES,
            "source_image_only_materialized_sparse_hashes": MATERIALIZED_SPARSE_HASHES,
            "pruning_explanation": "held-out image records and observations removed for source-image-only OpenMVS.",
            "transform_audit": {
                "similarity_transform": "not_recorded",
                "coordinate_normalization": "not_recorded",
                "scene_rescale": "not_recorded",
                "axis_conversion": "not_recorded",
                "mesh_export_transform": "not_recorded",
            },
            "mesh_provenance": {
                "mesh_ply_sha256": MESH_PLY_SHA,
                "mesh_mvs_sha256": MESH_MVS_SHA,
                "sparse_to_mesh_distance_audited": True,
            },
            "camera_qualification": {
                "package_sha256": "e213185d0d14084c42f586a0ec6d551d14edc6561879caaa341bc32c2b126e9b",
                "canonical_camera_file_sha256": fp_file_sha,
                "canonical_camera_payload_sha256": fp_payload_sha,
                "width": shape[1],
                "height": shape[0],
            },
        },
    )
    return sha256(path)


def make_case(
    root: Path,
    *,
    umgs_z: np.ndarray,
    openmvs_z: np.ndarray,
    umgs_valid: np.ndarray | None = None,
    openmvs_valid: np.ndarray | None = None,
    umgs_dtype: np.dtype = np.dtype("float32"),
    openmvs_depth_dtype: np.dtype = np.dtype("float32"),
    openmvs_bary_shape_bad: bool = False,
) -> SimpleNamespace:
    root.mkdir(parents=True, exist_ok=True)
    shape = umgs_z.shape
    if umgs_valid is None:
        umgs_valid = np.ones(shape, dtype=np.uint8)
    if openmvs_valid is None:
        openmvs_valid = np.ones(shape, dtype=np.uint8)

    umgs_npz = root / "umgs_packet.npz"
    openmvs_npz = root / "openmvs_packet.npz"
    np.savez_compressed(
        umgs_npz,
        accumulated_opacity=np.full(shape, 0.5, dtype=np.float32),
        weighted_camera_z_sum=(umgs_z * 0.5).astype(np.float32),
        expected_camera_z=umgs_z.astype(umgs_dtype),
        numeric_valid=umgs_valid.astype(np.uint8),
        weighted_camera_z2_sum=((umgs_z * umgs_z + 0.1) * 0.5).astype(np.float32),
        camera_z_variance=np.full(shape, 0.1, dtype=np.float32),
    )
    bary_shape = (shape[0], shape[1], 2) if openmvs_bary_shape_bad else (shape[0], shape[1], 3)
    bary = np.zeros(bary_shape, dtype=np.float32)
    bary[..., 0] = 1.0
    np.savez_compressed(
        openmvs_npz,
        depth=openmvs_z.astype(openmvs_depth_dtype),
        valid=openmvs_valid.astype(np.uint8),
        triangle_id=np.ones(shape, dtype=np.int32),
        barycentric=bary,
    )

    fingerprint = root / "fingerprint.json"
    fp_file_sha, fp_payload_sha = make_fingerprint(fingerprint, image_width=shape[1], image_height=shape[0])
    runtime_manifest = root / "layer2_runtime_source_manifest.json"
    runtime_manifest_sha, runtime_source_shas = make_runtime_manifest(runtime_manifest, root)
    frame_decision = root / "ROAD0001_FRAME_SCALE_CORRESPONDENCE_DECISION.json"
    frame_decision_sha = make_frame_decision(frame_decision, shape, fp_file_sha, fp_payload_sha)

    umgs_manifest = root / "umgs_manifest.json"
    openmvs_manifest = root / "openmvs_manifest.json"
    write_json(
        umgs_manifest,
        {
            "schema": {"schema_version": "umgs_expected_camera_z_packet_v1"},
            "scene": SCENE,
            "target": TARGET,
            "raster_height": shape[0],
            "raster_width": shape[1],
            "checkpoint_sha256": CHECKPOINT_SHA,
            "npz_sha256": sha256(umgs_npz),
            "target_camera_fingerprint": {"fingerprint_sha256": fp_payload_sha},
            "target_camera_fingerprint_comparison": {
                "expected_file_sha256": fp_file_sha,
                "expected_payload_sha256": fp_payload_sha,
                "actual_payload_sha256": fp_payload_sha,
                "matched": True,
            },
            "preflight": {
                "runtime_source_hash_manifest": {"sha256": UMGS_RUNTIME_SHA},
                "colmap_sparse_hash_manifest": {
                    "sha256": "synthetic_sparse_manifest_sha",
                    "files": [
                        {"file": "cameras.bin", "sha256": SOURCE_SPARSE_HASHES["cameras.bin"]},
                        {"file": "images.bin", "sha256": SOURCE_SPARSE_HASHES["images.bin"]},
                        {"file": "points3D.bin", "sha256": SOURCE_SPARSE_HASHES["points3D.bin"]},
                    ],
                },
                "manifest_info": {
                    "split_manifest": {
                        "train_file_expected_sha256": "train-sha",
                        "train_file_actual_sha256": "train-sha",
                        "test_file_expected_sha256": "test-sha",
                        "test_file_actual_sha256": "test-sha",
                    }
                },
            },
        },
    )
    write_json(
        openmvs_manifest,
        {
            "schema": "openmvs_canonical_camera_triangle_render_v1",
            "scene": SCENE,
            "target": TARGET,
            "output": {"height": shape[0], "width": shape[1], "npz_sha256": sha256(openmvs_npz)},
            "fingerprint": {"file_sha256": fp_file_sha, "payload_sha256": fp_payload_sha},
            "mesh": {"ply_sha256": MESH_PLY_SHA, "mvs_sha256": MESH_MVS_SHA},
            "runtime_source_manifest": {
                "sha256": OPENMVS_RUNTIME_SHA,
                "source_files": [
                    {"role": "adapter", "sha256": ADAPTER_SHA},
                    {"role": "core_rasterizer_mesh_loader", "sha256": CORE_SHA},
                ],
            },
            "no_proxy_metric": True,
            "not_ground_truth": True,
            "valid_definition": "triangle_hit AND finite_camera_z AND camera_z_gt_0",
            "pixel_center_convention": "corner-origin_pixel-centers_at_index_plus_0.5",
            "principal_point_policy": "centered_fov_projection_cx_width_over_2_cy_height_over_2",
        },
    )

    return SimpleNamespace(
        umgs_npz=umgs_npz,
        umgs_npz_sha256=sha256(umgs_npz),
        umgs_manifest=umgs_manifest,
        umgs_manifest_sha256=sha256(umgs_manifest),
        umgs_checkpoint_sha256=CHECKPOINT_SHA,
        umgs_fingerprint=fingerprint,
        umgs_fingerprint_sha256=fp_file_sha,
        umgs_fingerprint_payload_sha256=fp_payload_sha,
        umgs_runtime_manifest_sha256=UMGS_RUNTIME_SHA,
        openmvs_npz=openmvs_npz,
        openmvs_npz_sha256=sha256(openmvs_npz),
        openmvs_manifest=openmvs_manifest,
        openmvs_manifest_sha256=sha256(openmvs_manifest),
        openmvs_mesh_ply_sha256=MESH_PLY_SHA,
        openmvs_mesh_mvs_sha256=MESH_MVS_SHA,
        openmvs_fingerprint=fingerprint,
        openmvs_fingerprint_sha256=fp_file_sha,
        openmvs_fingerprint_payload_sha256=fp_payload_sha,
        openmvs_runtime_manifest_sha256=OPENMVS_RUNTIME_SHA,
        openmvs_adapter_sha256=ADAPTER_SHA,
        openmvs_core_sha256=CORE_SHA,
        frame_scale_decision_json=frame_decision,
        frame_scale_decision_sha256=frame_decision_sha,
        layer2_runtime_source_manifest=runtime_manifest,
        layer2_runtime_source_manifest_sha256=runtime_manifest_sha,
        expected_scene=SCENE,
        expected_target=TARGET,
        expected_height=shape[0],
        expected_width=shape[1],
        negative_control_status=ev.CONTROL_PROTOCOL_STATUS,
        output_json=root / "out.json",
        output_csv=root / "out.csv",
        runtime_source_shas=runtime_source_shas,
    )


def evaluate_case(tmp: Path, name: str, umgs: np.ndarray, openmvs: np.ndarray, **kwargs) -> dict:
    args = make_case(tmp / name, umgs_z=umgs, openmvs_z=openmvs, **kwargs)
    return ev.evaluate(args)


def test_identical_arrays_strong() -> None:
    with tempfile.TemporaryDirectory() as td:
        z = base_grid()
        out = evaluate_case(Path(td), "identical", z, z)
        assert out["interpretation_status"] == "proxy_alignment_strong"
        assert out["true_descriptive_metrics"]["mean_absolute_camera_z_disagreement"] == 0.0


def test_constant_positive_offset() -> None:
    with tempfile.TemporaryDirectory() as td:
        ref = base_grid()
        out = evaluate_case(Path(td), "offset", ref + 2.0, ref)
        assert out["true_descriptive_metrics"]["median_signed_camera_z_disagreement"] > 1.9


def test_constant_multiplicative_scale() -> None:
    with tempfile.TemporaryDirectory() as td:
        ref = base_grid()
        out = evaluate_case(Path(td), "scale", ref * 1.1, ref)
        assert out["true_core_metrics_on_shared"]["openmvs_denominated_relative_camera_z_disagreement_median"] > 0.09


def test_rank_preserving_nonlinear_transform() -> None:
    with tempfile.TemporaryDirectory() as td:
        ref = base_grid()
        umgs = ref + 0.001 * (ref - ref.min()) ** 2
        out = evaluate_case(Path(td), "rank", umgs, ref)
        assert out["true_core_metrics_on_shared"]["spearman"] > 0.99


def test_spatially_shifted_edge_weakens_structure() -> None:
    with tempfile.TemporaryDirectory() as td:
        ref = base_grid()
        umgs = np.roll(ref, 8, axis=1)
        out = evaluate_case(Path(td), "shift", umgs, ref)
        assert out["true_core_metrics_on_shared"]["high_gradient_cosine_median"] < 1.0


def test_partial_invalid_support_and_insufficient_support_no_metrics() -> None:
    with tempfile.TemporaryDirectory() as td:
        ref = base_grid()
        valid = np.zeros_like(ref, dtype=np.uint8)
        valid[:20, :20] = 1
        out = evaluate_case(Path(td), "insufficient", ref, ref, umgs_valid=valid)
        assert out["support_status"] == "insufficient_common_support"
        assert out["interpretation_status"] == "not_evaluated"
        assert "true_descriptive_metrics" not in out
        assert "true_core_metrics_on_shared" not in out


def test_nonpositive_nan_inf_are_excluded() -> None:
    with tempfile.TemporaryDirectory() as td:
        ref = base_grid()
        umgs = ref.copy()
        umgs[0, 0] = np.nan
        umgs[0, 1] = np.inf
        umgs[0, 2] = -1
        out = evaluate_case(Path(td), "invalid_values", umgs, ref)
        assert out["support_counts"]["primary_common_count"] == ref.size - 3


def test_opacity_and_variance_do_not_change_primary_mask_or_control() -> None:
    with tempfile.TemporaryDirectory() as td:
        ref = base_grid()
        a1 = make_case(Path(td) / "a1", umgs_z=ref, openmvs_z=ref)
        a2 = make_case(Path(td) / "a2", umgs_z=ref, openmvs_z=ref)
        np.savez_compressed(
            a2.umgs_npz,
            accumulated_opacity=np.full(ref.shape, 0.9, dtype=np.float32),
            weighted_camera_z_sum=(ref * 0.9).astype(np.float32),
            expected_camera_z=ref.astype(np.float32),
            numeric_valid=np.ones(ref.shape, dtype=np.uint8),
            weighted_camera_z2_sum=((ref * ref + 99) * 0.9).astype(np.float32),
            camera_z_variance=np.full(ref.shape, 99, dtype=np.float32),
        )
        a2.umgs_npz_sha256 = sha256(a2.umgs_npz)
        m = json.loads(a2.umgs_manifest.read_text(encoding="utf-8"))
        m["npz_sha256"] = a2.umgs_npz_sha256
        write_json(a2.umgs_manifest, m)
        a2.umgs_manifest_sha256 = sha256(a2.umgs_manifest)
        o1 = ev.evaluate(a1)
        o2 = ev.evaluate(a2)
        assert o1["support_counts"] == o2["support_counts"]
        assert o1["core_comparisons"] == o2["core_comparisons"]


def test_deterministic_control_repeats_identically() -> None:
    z = base_grid((20, 20))
    mask = np.ones_like(z, dtype=bool)
    d1, m1 = ev.deterministic_shuffle(z, mask)
    d2, m2 = ev.deterministic_shuffle(z, mask)
    assert np.array_equal(d1, d2)
    assert np.array_equal(m1, m2)


def test_taxonomy_strong_case() -> None:
    true = {"openmvs_denominated_relative_camera_z_disagreement_median": 0.01, "spearman": 0.95, "high_gradient_cosine_median": 0.9}
    ctrl = {"openmvs_denominated_relative_camera_z_disagreement_median": 0.5, "spearman": 0.1, "high_gradient_cosine_median": 0.2}
    assert ev.classify_interpretation(true, ev.compare_core(true, ctrl))[0] == "proxy_alignment_strong"


def test_taxonomy_relative_improves_structure_degrades_contradictory() -> None:
    true = {"openmvs_denominated_relative_camera_z_disagreement_median": 0.01, "spearman": 0.2, "high_gradient_cosine_median": 0.1}
    ctrl = {"openmvs_denominated_relative_camera_z_disagreement_median": 0.5, "spearman": 0.8, "high_gradient_cosine_median": 0.7}
    assert ev.classify_interpretation(true, ev.compare_core(true, ctrl))[0] == "proxy_alignment_contradictory"


def test_taxonomy_structure_improves_relative_degrades_contradictory() -> None:
    true = {"openmvs_denominated_relative_camera_z_disagreement_median": 0.9, "spearman": 0.9, "high_gradient_cosine_median": 0.8}
    ctrl = {"openmvs_denominated_relative_camera_z_disagreement_median": 0.1, "spearman": 0.2, "high_gradient_cosine_median": 0.1}
    assert ev.classify_interpretation(true, ev.compare_core(true, ctrl))[0] == "proxy_alignment_contradictory"


def test_taxonomy_partial_improvement_weak_or_mixed() -> None:
    true = {"openmvs_denominated_relative_camera_z_disagreement_median": 0.01, "spearman": 0.5, "high_gradient_cosine_median": 0.5}
    ctrl = {"openmvs_denominated_relative_camera_z_disagreement_median": 0.5, "spearman": 0.4, "high_gradient_cosine_median": 0.5}
    assert ev.classify_interpretation(true, ev.compare_core(true, ctrl))[0] == "proxy_alignment_weak_or_mixed"


def test_missing_core_metric_is_not_degrade() -> None:
    true = {"openmvs_denominated_relative_camera_z_disagreement_median": 0.01, "spearman": None, "high_gradient_cosine_median": 0.8}
    ctrl = {"openmvs_denominated_relative_camera_z_disagreement_median": 0.5, "spearman": 0.9, "high_gradient_cosine_median": 0.7}
    comp = ev.compare_core(true, ctrl)
    assert comp["spearman"] == "invalid"
    assert "degrade" not in comp.values()


def test_fewer_than_two_valid_core_comparisons_not_evaluated() -> None:
    true = {"openmvs_denominated_relative_camera_z_disagreement_median": 0.01, "spearman": None, "high_gradient_cosine_median": None}
    ctrl = {"openmvs_denominated_relative_camera_z_disagreement_median": 0.5, "spearman": None, "high_gradient_cosine_median": None}
    assert ev.classify_interpretation(true, ev.compare_core(true, ctrl))[0] == "not_evaluated"


def test_control_unavailable_not_evaluated_but_descriptive_metrics_exist() -> None:
    with tempfile.TemporaryDirectory() as td:
        ref = base_grid()
        args = make_case(Path(td), umgs_z=ref, openmvs_z=ref)
        args.negative_control_status = "negative_control_not_yet_approved"
        out = ev.evaluate(args)
        assert out["true_descriptive_metrics"]["mean_absolute_camera_z_disagreement"] == 0.0
        assert out["interpretation_status"] == "not_evaluated"


def test_native_mae_does_not_directly_change_taxonomy() -> None:
    true = {"openmvs_denominated_relative_camera_z_disagreement_median": 0.01, "spearman": 0.95, "high_gradient_cosine_median": 0.9, "mean_absolute_camera_z_disagreement": 999}
    ctrl = {"openmvs_denominated_relative_camera_z_disagreement_median": 0.5, "spearman": 0.1, "high_gradient_cosine_median": 0.2, "mean_absolute_camera_z_disagreement": 0}
    assert ev.classify_interpretation(true, ev.compare_core(true, ctrl))[0] == "proxy_alignment_strong"


def test_wrong_checkpoint_sha_fails() -> None:
    with tempfile.TemporaryDirectory() as td:
        ref = base_grid()
        args = make_case(Path(td), umgs_z=ref, openmvs_z=ref)
        args.umgs_checkpoint_sha256 = "bad"
        out = ev.evaluate(args)
        assert out["input_status"] == "input_hash_mismatch"


def test_wrong_fingerprint_payload_sha_fails() -> None:
    with tempfile.TemporaryDirectory() as td:
        ref = base_grid()
        args = make_case(Path(td), umgs_z=ref, openmvs_z=ref)
        args.umgs_fingerprint_payload_sha256 = "bad"
        out = ev.evaluate(args)
        assert out["input_status"] == "input_hash_mismatch"


def test_different_umgs_openmvs_fingerprint_payload_fails() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ref = base_grid()
        args = make_case(root, umgs_z=ref, openmvs_z=ref)
        other = root / "openmvs_fingerprint_other.json"
        file_sha, payload_sha = make_fingerprint(other, image_width=ref.shape[1], image_height=ref.shape[0], tweak=0.1)
        args.openmvs_fingerprint = other
        args.openmvs_fingerprint_sha256 = file_sha
        args.openmvs_fingerprint_payload_sha256 = payload_sha
        m = json.loads(args.openmvs_manifest.read_text(encoding="utf-8"))
        m["fingerprint"] = {"file_sha256": file_sha, "payload_sha256": payload_sha}
        write_json(args.openmvs_manifest, m)
        args.openmvs_manifest_sha256 = sha256(args.openmvs_manifest)
        out = ev.evaluate(args)
        assert out["input_status"] == "input_hash_mismatch"


def test_wrong_umgs_runtime_manifest_sha_fails() -> None:
    with tempfile.TemporaryDirectory() as td:
        args = make_case(Path(td), umgs_z=base_grid(), openmvs_z=base_grid())
        args.umgs_runtime_manifest_sha256 = "bad"
        out = ev.evaluate(args)
        assert out["input_status"] == "input_hash_mismatch"


def test_wrong_openmvs_runtime_manifest_sha_fails() -> None:
    with tempfile.TemporaryDirectory() as td:
        args = make_case(Path(td), umgs_z=base_grid(), openmvs_z=base_grid())
        args.openmvs_runtime_manifest_sha256 = "bad"
        out = ev.evaluate(args)
        assert out["input_status"] == "input_hash_mismatch"


def test_wrong_mesh_ply_sha_fails() -> None:
    with tempfile.TemporaryDirectory() as td:
        args = make_case(Path(td), umgs_z=base_grid(), openmvs_z=base_grid())
        args.openmvs_mesh_ply_sha256 = "bad"
        out = ev.evaluate(args)
        assert out["input_status"] == "input_hash_mismatch"


def test_wrong_mesh_mvs_sha_fails() -> None:
    with tempfile.TemporaryDirectory() as td:
        args = make_case(Path(td), umgs_z=base_grid(), openmvs_z=base_grid())
        args.openmvs_mesh_mvs_sha256 = "bad"
        out = ev.evaluate(args)
        assert out["input_status"] == "input_hash_mismatch"


def test_wrong_adapter_sha_fails() -> None:
    with tempfile.TemporaryDirectory() as td:
        args = make_case(Path(td), umgs_z=base_grid(), openmvs_z=base_grid())
        args.openmvs_adapter_sha256 = "bad"
        out = ev.evaluate(args)
        assert out["input_status"] == "input_hash_mismatch"


def test_wrong_core_sha_fails() -> None:
    with tempfile.TemporaryDirectory() as td:
        args = make_case(Path(td), umgs_z=base_grid(), openmvs_z=base_grid())
        args.openmvs_core_sha256 = "bad"
        out = ev.evaluate(args)
        assert out["input_status"] == "input_hash_mismatch"


def test_wrong_manifest_schema_fails() -> None:
    with tempfile.TemporaryDirectory() as td:
        args = make_case(Path(td), umgs_z=base_grid(), openmvs_z=base_grid())
        m = json.loads(args.openmvs_manifest.read_text(encoding="utf-8"))
        m["schema"] = "bad_schema"
        write_json(args.openmvs_manifest, m)
        args.openmvs_manifest_sha256 = sha256(args.openmvs_manifest)
        out = ev.evaluate(args)
        assert out["input_status"] == "input_hash_mismatch"


def test_wrong_array_dtype_fails() -> None:
    with tempfile.TemporaryDirectory() as td:
        ref = base_grid()
        args = make_case(Path(td), umgs_z=ref, openmvs_z=ref, umgs_dtype=np.dtype("float64"))
        args.umgs_npz_sha256 = sha256(args.umgs_npz)
        m = json.loads(args.umgs_manifest.read_text(encoding="utf-8"))
        m["npz_sha256"] = args.umgs_npz_sha256
        write_json(args.umgs_manifest, m)
        args.umgs_manifest_sha256 = sha256(args.umgs_manifest)
        out = ev.evaluate(args)
        assert out["inconclusive_reason"] == "schema_or_dtype_mismatch"


def test_wrong_barycentric_shape_fails() -> None:
    with tempfile.TemporaryDirectory() as td:
        ref = base_grid()
        args = make_case(Path(td), umgs_z=ref, openmvs_z=ref, openmvs_bary_shape_bad=True)
        args.openmvs_npz_sha256 = sha256(args.openmvs_npz)
        m = json.loads(args.openmvs_manifest.read_text(encoding="utf-8"))
        m["output"]["npz_sha256"] = args.openmvs_npz_sha256
        write_json(args.openmvs_manifest, m)
        args.openmvs_manifest_sha256 = sha256(args.openmvs_manifest)
        out = ev.evaluate(args)
        assert out["inconclusive_reason"] == "schema_or_dtype_mismatch"


def test_insufficient_shared_control_support_not_evaluated() -> None:
    with tempfile.TemporaryDirectory() as td:
        ref = base_grid((300, 300))
        valid = np.zeros_like(ref, dtype=np.uint8)
        valid.reshape(-1)[:18000] = 1
        out = evaluate_case(Path(td), "shared_low", ref, ref, umgs_valid=valid)
        assert out["support_status"] == "common_support_sufficient"
        assert out["control_shared_support_status"] == "insufficient_shared_control_support"
        assert out["interpretation_status"] == "not_evaluated"
        assert out["inconclusive_reason"] == "negative_control_inconclusive_due_to_shared_support"
        assert "true_core_metrics_on_shared" not in out


def test_control_source_sha_mismatch_fails() -> None:
    with tempfile.TemporaryDirectory() as td:
        args = make_case(Path(td), umgs_z=base_grid(), openmvs_z=base_grid())
        args.layer2_runtime_source_manifest_sha256 = sha256(args.layer2_runtime_source_manifest)
        m = json.loads(args.layer2_runtime_source_manifest.read_text(encoding="utf-8"))
        for row in m["sources"]:
            if row["role"] == "negative_control_source":
                row["sha256"] = "bad"
        write_json(args.layer2_runtime_source_manifest, m)
        args.layer2_runtime_source_manifest_sha256 = sha256(args.layer2_runtime_source_manifest)
        out = ev.evaluate(args)
        assert out["input_status"] == "input_hash_mismatch"


def test_corrected_core_metrics_match_direct_calls() -> None:
    with tempfile.TemporaryDirectory() as td:
        ref = base_grid()
        args = make_case(Path(td), umgs_z=ref, openmvs_z=ref)
        out = ev.evaluate(args)
        with np.load(args.umgs_npz) as umgs_data, np.load(args.openmvs_npz) as openmvs_data:
            umgs_z = np.array(umgs_data["expected_camera_z"], copy=True)
            umgs_valid = np.array(umgs_data["numeric_valid"].astype(bool), copy=True)
            openmvs_z = np.array(openmvs_data["depth"], copy=True)
            openmvs_valid = np.array(openmvs_data["valid"].astype(bool), copy=True)
        true_mask = ev.build_primary_mask(umgs_z, umgs_valid, openmvs_z, openmvs_valid)
        control_z, control_valid = ev.deterministic_shuffle(umgs_z, umgs_valid)
        control_mask = ev.build_primary_mask(control_z, control_valid, openmvs_z, openmvs_valid)
        shared = true_mask & control_mask
        gd = ev.corrected.reference_high_gradient_domain(openmvs_z, shared)
        direct = ev.corrected.metrics_on_mask(openmvs_z, umgs_z, shared, gradient_domain=gd)
        assert out["true_core_metrics_on_shared"]["openmvs_denominated_relative_camera_z_disagreement_median"] == direct["absrel_median"]
        assert out["true_core_metrics_on_shared"]["high_gradient_cosine_median"] == direct["high_gradient_cosine_median"]


def test_exact_future_command_parser_smoke_passes() -> None:
    with tempfile.TemporaryDirectory() as td:
        args = make_case(Path(td), umgs_z=base_grid(), openmvs_z=base_grid())
        argv = [
            "--umgs-npz", str(args.umgs_npz),
            "--umgs-npz-sha256", args.umgs_npz_sha256,
            "--umgs-manifest", str(args.umgs_manifest),
            "--umgs-manifest-sha256", args.umgs_manifest_sha256,
            "--umgs-checkpoint-sha256", args.umgs_checkpoint_sha256,
            "--umgs-fingerprint", str(args.umgs_fingerprint),
            "--umgs-fingerprint-sha256", args.umgs_fingerprint_sha256,
            "--umgs-fingerprint-payload-sha256", args.umgs_fingerprint_payload_sha256,
            "--umgs-runtime-manifest-sha256", args.umgs_runtime_manifest_sha256,
            "--openmvs-npz", str(args.openmvs_npz),
            "--openmvs-npz-sha256", args.openmvs_npz_sha256,
            "--openmvs-manifest", str(args.openmvs_manifest),
            "--openmvs-manifest-sha256", args.openmvs_manifest_sha256,
            "--openmvs-mesh-ply-sha256", args.openmvs_mesh_ply_sha256,
            "--openmvs-mesh-mvs-sha256", args.openmvs_mesh_mvs_sha256,
            "--openmvs-fingerprint", str(args.openmvs_fingerprint),
            "--openmvs-fingerprint-sha256", args.openmvs_fingerprint_sha256,
            "--openmvs-fingerprint-payload-sha256", args.openmvs_fingerprint_payload_sha256,
            "--openmvs-runtime-manifest-sha256", args.openmvs_runtime_manifest_sha256,
            "--openmvs-adapter-sha256", args.openmvs_adapter_sha256,
            "--openmvs-core-sha256", args.openmvs_core_sha256,
            "--frame-scale-decision-json", str(args.frame_scale_decision_json),
            "--frame-scale-decision-sha256", args.frame_scale_decision_sha256,
            "--layer2-runtime-source-manifest", str(args.layer2_runtime_source_manifest),
            "--layer2-runtime-source-manifest-sha256", args.layer2_runtime_source_manifest_sha256,
            "--expected-scene", args.expected_scene,
            "--expected-target", args.expected_target,
            "--expected-height", str(args.expected_height),
            "--expected-width", str(args.expected_width),
            "--output-json", str(args.output_json),
            "--output-csv", str(args.output_csv),
        ]
        parsed = ev.build_parser().parse_args(argv)
        assert parsed.openmvs_adapter_sha256 == ADAPTER_SHA
        assert parsed.openmvs_core_sha256 == CORE_SHA


def test_frame_scale_decision_file_missing_or_wrong_sha_fails() -> None:
    with tempfile.TemporaryDirectory() as td:
        args = make_case(Path(td), umgs_z=base_grid(), openmvs_z=base_grid())
        args.frame_scale_decision_sha256 = "bad"
        out = ev.evaluate(args)
        assert out["input_status"] == "input_hash_mismatch"
        args = make_case(Path(td) / "missing", umgs_z=base_grid(), openmvs_z=base_grid())
        args.frame_scale_decision_json = Path(td) / "missing.json"
        out = ev.evaluate(args)
        assert out["input_status"] == "input_hash_mismatch"


def test_formal_runtime_manifest_schema_passes() -> None:
    with tempfile.TemporaryDirectory() as td:
        args = make_case(Path(td), umgs_z=base_grid(), openmvs_z=base_grid())
        checks = []
        ev.verify_layer2_runtime_manifest(args.layer2_runtime_source_manifest, args.layer2_runtime_source_manifest_sha256, checks)
        assert any(row["check"] == "runtime_source_manifest_has_evaluator" and row["status"] == "pass" for row in checks)


def test_formal_frame_scale_decision_schema_passes() -> None:
    with tempfile.TemporaryDirectory() as td:
        args = make_case(Path(td), umgs_z=base_grid(), openmvs_z=base_grid())
        checks = []
        manifest = json.loads(args.umgs_manifest.read_text(encoding="utf-8"))
        ev.verify_frame_scale_decision(args.frame_scale_decision_json, args.frame_scale_decision_sha256, args, manifest, checks)
        assert any(row["check"] == "frame_scale_decision_schema" and row["status"] == "pass" for row in checks)


def test_runtime_manifest_old_wrong_field_names_fail() -> None:
    with tempfile.TemporaryDirectory() as td:
        args = make_case(Path(td), umgs_z=base_grid(), openmvs_z=base_grid())
        m = json.loads(args.layer2_runtime_source_manifest.read_text(encoding="utf-8"))
        m["files"] = [{"role": row["role"], "path": row["repo_relative_path"], "sha256": row["sha256"]} for row in m.pop("sources")]
        write_json(args.layer2_runtime_source_manifest, m)
        args.layer2_runtime_source_manifest_sha256 = sha256(args.layer2_runtime_source_manifest)
        out = ev.evaluate(args)
        assert out["input_status"] == "input_hash_mismatch"


def test_frame_decision_missing_required_formal_field_fails() -> None:
    with tempfile.TemporaryDirectory() as td:
        args = make_case(Path(td), umgs_z=base_grid(), openmvs_z=base_grid())
        m = json.loads(args.frame_scale_decision_json.read_text(encoding="utf-8"))
        m.pop("transform_audit")
        write_json(args.frame_scale_decision_json, m)
        args.frame_scale_decision_sha256 = sha256(args.frame_scale_decision_json)
        out = ev.evaluate(args)
        assert out["input_status"] == "input_hash_mismatch"


def test_source_sparse_root_mismatch_fails() -> None:
    with tempfile.TemporaryDirectory() as td:
        args = make_case(Path(td), umgs_z=base_grid(), openmvs_z=base_grid())
        m = json.loads(args.frame_scale_decision_json.read_text(encoding="utf-8"))
        m["openmvs_materialization_source_sparse_root"] = "/different/source/sparse/0"
        write_json(args.frame_scale_decision_json, m)
        args.frame_scale_decision_sha256 = sha256(args.frame_scale_decision_json)
        out = ev.evaluate(args)
        assert out["input_status"] == "input_hash_mismatch"


def test_source_sparse_hash_mismatch_with_umgs_manifest_fails() -> None:
    with tempfile.TemporaryDirectory() as td:
        args = make_case(Path(td), umgs_z=base_grid(), openmvs_z=base_grid())
        m = json.loads(args.frame_scale_decision_json.read_text(encoding="utf-8"))
        m["source_sparse_hashes"]["images.bin"] = "bad-images-sha"
        write_json(args.frame_scale_decision_json, m)
        args.frame_scale_decision_sha256 = sha256(args.frame_scale_decision_json)
        out = ev.evaluate(args)
        assert out["input_status"] == "input_hash_mismatch"


def test_decision_mesh_hash_mismatch_with_cli_fails() -> None:
    with tempfile.TemporaryDirectory() as td:
        args = make_case(Path(td), umgs_z=base_grid(), openmvs_z=base_grid())
        m = json.loads(args.frame_scale_decision_json.read_text(encoding="utf-8"))
        m["mesh_provenance"]["mesh_ply_sha256"] = "bad"
        write_json(args.frame_scale_decision_json, m)
        args.frame_scale_decision_sha256 = sha256(args.frame_scale_decision_json)
        out = ev.evaluate(args)
        assert out["input_status"] == "input_hash_mismatch"


def test_decision_camera_qualification_hash_mismatch_fails() -> None:
    with tempfile.TemporaryDirectory() as td:
        args = make_case(Path(td), umgs_z=base_grid(), openmvs_z=base_grid())
        m = json.loads(args.frame_scale_decision_json.read_text(encoding="utf-8"))
        m["camera_qualification"]["canonical_camera_payload_sha256"] = "bad"
        write_json(args.frame_scale_decision_json, m)
        args.frame_scale_decision_sha256 = sha256(args.frame_scale_decision_json)
        out = ev.evaluate(args)
        assert out["input_status"] == "input_hash_mismatch"


def test_preflight_only_passes_without_metrics_or_support_count() -> None:
    with tempfile.TemporaryDirectory() as td:
        args = make_case(Path(td), umgs_z=base_grid(), openmvs_z=base_grid())
        argv = [
            "--umgs-npz", str(args.umgs_npz),
            "--umgs-npz-sha256", args.umgs_npz_sha256,
            "--umgs-manifest", str(args.umgs_manifest),
            "--umgs-manifest-sha256", args.umgs_manifest_sha256,
            "--umgs-checkpoint-sha256", args.umgs_checkpoint_sha256,
            "--umgs-fingerprint", str(args.umgs_fingerprint),
            "--umgs-fingerprint-sha256", args.umgs_fingerprint_sha256,
            "--umgs-fingerprint-payload-sha256", args.umgs_fingerprint_payload_sha256,
            "--umgs-runtime-manifest-sha256", args.umgs_runtime_manifest_sha256,
            "--openmvs-npz", str(args.openmvs_npz),
            "--openmvs-npz-sha256", args.openmvs_npz_sha256,
            "--openmvs-manifest", str(args.openmvs_manifest),
            "--openmvs-manifest-sha256", args.openmvs_manifest_sha256,
            "--openmvs-mesh-ply-sha256", args.openmvs_mesh_ply_sha256,
            "--openmvs-mesh-mvs-sha256", args.openmvs_mesh_mvs_sha256,
            "--openmvs-fingerprint", str(args.openmvs_fingerprint),
            "--openmvs-fingerprint-sha256", args.openmvs_fingerprint_sha256,
            "--openmvs-fingerprint-payload-sha256", args.openmvs_fingerprint_payload_sha256,
            "--openmvs-runtime-manifest-sha256", args.openmvs_runtime_manifest_sha256,
            "--openmvs-adapter-sha256", args.openmvs_adapter_sha256,
            "--openmvs-core-sha256", args.openmvs_core_sha256,
            "--frame-scale-decision-json", str(args.frame_scale_decision_json),
            "--frame-scale-decision-sha256", args.frame_scale_decision_sha256,
            "--layer2-runtime-source-manifest", str(args.layer2_runtime_source_manifest),
            "--layer2-runtime-source-manifest-sha256", args.layer2_runtime_source_manifest_sha256,
            "--expected-scene", args.expected_scene,
            "--expected-target", args.expected_target,
            "--expected-height", str(args.expected_height),
            "--expected-width", str(args.expected_width),
            "--preflight-only",
            "--output-json", str(args.output_json),
            "--output-csv", str(args.output_csv),
        ]
        rc = ev.main(argv)
        assert rc == 0
        payload = json.loads(args.output_json.read_text(encoding="utf-8"))
        assert payload["real_metrics_computed"] is False
        assert payload["real_common_mask_constructed"] is False
        assert payload["support_count_computed"] is False
        assert "support_counts" not in payload
        assert "true_descriptive_metrics" not in payload


def test_output_directory_guard_refuses_overwrite() -> None:
    with tempfile.TemporaryDirectory() as td:
        args = make_case(Path(td), umgs_z=base_grid(), openmvs_z=base_grid())
        args.output_json.write_text("existing", encoding="utf-8")
        rc = ev.main([
            "--umgs-npz", str(args.umgs_npz),
            "--umgs-npz-sha256", args.umgs_npz_sha256,
            "--umgs-manifest", str(args.umgs_manifest),
            "--umgs-manifest-sha256", args.umgs_manifest_sha256,
            "--umgs-checkpoint-sha256", args.umgs_checkpoint_sha256,
            "--umgs-fingerprint", str(args.umgs_fingerprint),
            "--umgs-fingerprint-sha256", args.umgs_fingerprint_sha256,
            "--umgs-fingerprint-payload-sha256", args.umgs_fingerprint_payload_sha256,
            "--umgs-runtime-manifest-sha256", args.umgs_runtime_manifest_sha256,
            "--openmvs-npz", str(args.openmvs_npz),
            "--openmvs-npz-sha256", args.openmvs_npz_sha256,
            "--openmvs-manifest", str(args.openmvs_manifest),
            "--openmvs-manifest-sha256", args.openmvs_manifest_sha256,
            "--openmvs-mesh-ply-sha256", args.openmvs_mesh_ply_sha256,
            "--openmvs-mesh-mvs-sha256", args.openmvs_mesh_mvs_sha256,
            "--openmvs-fingerprint", str(args.openmvs_fingerprint),
            "--openmvs-fingerprint-sha256", args.openmvs_fingerprint_sha256,
            "--openmvs-fingerprint-payload-sha256", args.openmvs_fingerprint_payload_sha256,
            "--openmvs-runtime-manifest-sha256", args.openmvs_runtime_manifest_sha256,
            "--openmvs-adapter-sha256", args.openmvs_adapter_sha256,
            "--openmvs-core-sha256", args.openmvs_core_sha256,
            "--frame-scale-decision-json", str(args.frame_scale_decision_json),
            "--frame-scale-decision-sha256", args.frame_scale_decision_sha256,
            "--layer2-runtime-source-manifest", str(args.layer2_runtime_source_manifest),
            "--layer2-runtime-source-manifest-sha256", args.layer2_runtime_source_manifest_sha256,
            "--expected-scene", args.expected_scene,
            "--expected-target", args.expected_target,
            "--expected-height", str(args.expected_height),
            "--expected-width", str(args.expected_width),
            "--preflight-only",
            "--output-json", str(args.output_json),
            "--output-csv", str(args.output_csv),
        ])
        assert rc == 9


TESTS = [
    test_identical_arrays_strong,
    test_constant_positive_offset,
    test_constant_multiplicative_scale,
    test_rank_preserving_nonlinear_transform,
    test_spatially_shifted_edge_weakens_structure,
    test_partial_invalid_support_and_insufficient_support_no_metrics,
    test_nonpositive_nan_inf_are_excluded,
    test_opacity_and_variance_do_not_change_primary_mask_or_control,
    test_deterministic_control_repeats_identically,
    test_taxonomy_strong_case,
    test_taxonomy_relative_improves_structure_degrades_contradictory,
    test_taxonomy_structure_improves_relative_degrades_contradictory,
    test_taxonomy_partial_improvement_weak_or_mixed,
    test_missing_core_metric_is_not_degrade,
    test_fewer_than_two_valid_core_comparisons_not_evaluated,
    test_control_unavailable_not_evaluated_but_descriptive_metrics_exist,
    test_native_mae_does_not_directly_change_taxonomy,
    test_wrong_checkpoint_sha_fails,
    test_wrong_fingerprint_payload_sha_fails,
    test_different_umgs_openmvs_fingerprint_payload_fails,
    test_wrong_umgs_runtime_manifest_sha_fails,
    test_wrong_openmvs_runtime_manifest_sha_fails,
    test_wrong_mesh_ply_sha_fails,
    test_wrong_mesh_mvs_sha_fails,
    test_wrong_adapter_sha_fails,
    test_wrong_core_sha_fails,
    test_wrong_manifest_schema_fails,
    test_wrong_array_dtype_fails,
    test_wrong_barycentric_shape_fails,
    test_insufficient_shared_control_support_not_evaluated,
    test_control_source_sha_mismatch_fails,
    test_corrected_core_metrics_match_direct_calls,
    test_exact_future_command_parser_smoke_passes,
    test_frame_scale_decision_file_missing_or_wrong_sha_fails,
    test_formal_runtime_manifest_schema_passes,
    test_formal_frame_scale_decision_schema_passes,
    test_runtime_manifest_old_wrong_field_names_fail,
    test_frame_decision_missing_required_formal_field_fails,
    test_source_sparse_root_mismatch_fails,
    test_source_sparse_hash_mismatch_with_umgs_manifest_fails,
    test_decision_mesh_hash_mismatch_with_cli_fails,
    test_decision_camera_qualification_hash_mismatch_fails,
    test_preflight_only_passes_without_metrics_or_support_count,
    test_output_directory_guard_refuses_overwrite,
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", type=Path)
    args = parser.parse_args()
    rows = []
    for test in TESTS:
        test()
        rows.append({"test": test.__name__, "status": "pass"})
        print(f"PASS {test.__name__}")
    print("ALL_UMGS_OPENMVS_CAMERA_Z_PROXY_ALIGNMENT_V12_TESTS_PASSED")
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps({"tests": rows}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
