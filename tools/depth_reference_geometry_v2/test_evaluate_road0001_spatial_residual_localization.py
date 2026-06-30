#!/usr/bin/env python3
"""Synthetic tests for the Road-0001 spatial residual localization evaluator."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_road0001_spatial_residual_localization as ev


def sha(path: Path) -> str:
    return ev.sha256_file(path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_csv(path: Path) -> None:
    path.write_text("status\nok\n", encoding="utf-8")


@contextlib.contextmanager
def patched_constants(**kwargs: Any):
    old = {k: getattr(ev, k) for k in kwargs}
    for k, v in kwargs.items():
        setattr(ev, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(ev, k, v)


@contextlib.contextmanager
def patched_attr(obj: Any, name: str, value: Any):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def make_fingerprint(path: Path, width: int = ev.EXPECTED_WIDTH, height: int = ev.EXPECTED_HEIGHT) -> tuple[str, str]:
    payload = {"image_name": ev.EXPECTED_TARGET, "image_width": width, "image_height": height}
    payload_sha = ev.sha256_json_payload(payload)
    write_json(path, {"payload": payload, "fingerprint_sha256": payload_sha})
    return sha(path), payload_sha


def make_runtime_manifest(path: Path, roles: dict[str, Path], *, schema: str = ev.LOCALIZATION_RUNTIME_MANIFEST_SCHEMA) -> str:
    write_json(
        path,
        {
            "schema": schema,
            "repo_root": str(Path.cwd()),
            "sources": [
                {"role": role, "repo_relative_path": str(src), "sha256": sha(src)}
                for role, src in roles.items()
            ],
        },
    )
    return sha(path)


def make_npzs(root: Path, shape: tuple[int, int] = (ev.EXPECTED_HEIGHT, ev.EXPECTED_WIDTH)) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    y, x = np.indices(shape, dtype=np.float32)
    openmvs_depth = (20.0 + 0.02 * x + 0.01 * y).astype(np.float32)
    umgs_depth = (openmvs_depth + 0.1 + 0.001 * np.sin(x / 20.0)).astype(np.float32)
    opacity = np.full(shape, 0.9, dtype=np.float32)
    variance = np.full(shape, 0.05, dtype=np.float32)
    umgs_path = root / "umgs_packet.npz"
    openmvs_path = root / "openmvs_packet.npz"
    np.savez_compressed(
        umgs_path,
        accumulated_opacity=opacity,
        weighted_camera_z_sum=(umgs_depth * opacity).astype(np.float32),
        expected_camera_z=umgs_depth,
        numeric_valid=np.ones(shape, dtype=np.uint8),
        weighted_camera_z2_sum=((umgs_depth**2 + variance) * opacity).astype(np.float32),
        camera_z_variance=variance,
    )
    bary = np.zeros((*shape, 3), dtype=np.float32)
    bary[..., 0] = 1.0
    np.savez_compressed(
        openmvs_path,
        depth=openmvs_depth,
        valid=np.ones(shape, dtype=np.uint8),
        triangle_id=np.ones(shape, dtype=np.int32),
        barycentric=bary,
    )
    return umgs_path, openmvs_path


def build_args(root: Path, result: dict[str, Any], *, output_name: str = "out", runtime_schema: str = ev.LOCALIZATION_RUNTIME_MANIFEST_SCHEMA) -> argparse.Namespace:
    umgs_npz, openmvs_npz = make_npzs(root)
    umgs_manifest = root / "umgs_manifest.json"
    openmvs_manifest = root / "openmvs_manifest.json"
    write_json(umgs_manifest, {"schema": "synthetic_umgs_manifest"})
    write_json(openmvs_manifest, {"schema": "synthetic_openmvs_manifest"})
    result_json = root / "formal_result.json"
    result_csv = root / "formal_result.csv"
    write_json(result_json, result)
    write_csv(result_csv)
    frame = root / "frame.json"
    write_json(frame, {"status": "correspondence_pass"})
    fingerprint = root / "fingerprint.json"
    fp_sha, fp_payload_sha = make_fingerprint(fingerprint)
    layer2_manifest = root / "layer2_runtime.json"
    localization_manifest = root / "localization_runtime.json"
    layer2_sha = make_runtime_manifest(
        layer2_manifest,
        {
            "evaluator": Path("tools/depth_reference_geometry_v2/evaluate_umgs_openmvs_camera_z_proxy_alignment.py"),
            "negative_control_source": Path("tools/depth_reference_geometry_v2/openmvs_da3_overlap_corrected.py"),
        },
        schema=ev.LAYER2_RUNTIME_MANIFEST_SCHEMA,
    )
    localization_sha = make_runtime_manifest(
        localization_manifest,
        {
            "localization_evaluator": Path("tools/depth_reference_geometry_v2/evaluate_road0001_spatial_residual_localization.py"),
            "localization_tests": Path("tools/depth_reference_geometry_v2/test_evaluate_road0001_spatial_residual_localization.py"),
            "layer2_evaluator_v1_2": Path("tools/depth_reference_geometry_v2/evaluate_umgs_openmvs_camera_z_proxy_alignment.py"),
            "corrected_gradient_source": Path("tools/depth_reference_geometry_v2/openmvs_da3_overlap_corrected.py"),
        },
        schema=runtime_schema,
    )
    return argparse.Namespace(
        preflight_only=False,
        execute_localization=False,
        expected_scene=ev.EXPECTED_SCENE,
        expected_target=ev.EXPECTED_TARGET,
        expected_height=ev.EXPECTED_HEIGHT,
        expected_width=ev.EXPECTED_WIDTH,
        umgs_npz=umgs_npz,
        umgs_npz_sha256=sha(umgs_npz),
        umgs_manifest=umgs_manifest,
        umgs_manifest_sha256=sha(umgs_manifest),
        openmvs_npz=openmvs_npz,
        openmvs_npz_sha256=sha(openmvs_npz),
        openmvs_manifest=openmvs_manifest,
        openmvs_manifest_sha256=sha(openmvs_manifest),
        formal_result_json=result_json,
        formal_result_json_sha256=sha(result_json),
        formal_result_csv=result_csv,
        formal_result_csv_sha256=sha(result_csv),
        frame_scale_decision_json=frame,
        frame_scale_decision_sha256=sha(frame),
        layer2_runtime_source_manifest=layer2_manifest,
        layer2_runtime_source_manifest_sha256=layer2_sha,
        layer2_evaluator=Path("tools/depth_reference_geometry_v2/evaluate_umgs_openmvs_camera_z_proxy_alignment.py"),
        layer2_evaluator_sha256=sha(Path("tools/depth_reference_geometry_v2/evaluate_umgs_openmvs_camera_z_proxy_alignment.py")),
        corrected_source=Path("tools/depth_reference_geometry_v2/openmvs_da3_overlap_corrected.py"),
        corrected_source_sha256=sha(Path("tools/depth_reference_geometry_v2/openmvs_da3_overlap_corrected.py")),
        camera_fingerprint=fingerprint,
        camera_fingerprint_sha256=fp_sha,
        camera_fingerprint_payload_sha256=fp_payload_sha,
        localization_evaluator_sha256=sha(Path("tools/depth_reference_geometry_v2/evaluate_road0001_spatial_residual_localization.py")),
        localization_runtime_source_manifest=localization_manifest,
        localization_runtime_source_manifest_sha256=localization_sha,
        expected_primary_common_count=ev.EXPECTED_PRIMARY_COMMON_COUNT,
        expected_primary_common_ratio=ev.EXPECTED_PRIMARY_COMMON_RATIO,
        expected_gradient_local_valid_count=ev.EXPECTED_GRADIENT_LOCAL_VALID_COUNT,
        expected_high_gradient_threshold=ev.EXPECTED_HIGH_GRADIENT_THRESHOLD,
        expected_high_gradient_pixels=ev.EXPECTED_HIGH_GRADIENT_PIXELS,
        expected_high_gradient_packbits_sha256=ev.EXPECTED_HIGH_GRADIENT_PACKBITS_SHA256,
        output_root=root / output_name,
    )


def synthetic_result_from_arrays(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    umgs_npz, openmvs_npz = make_npzs(root)
    umgs = ev.load_umgs(umgs_npz)
    openmvs = ev.load_openmvs(openmvs_npz)
    primary = ev.build_primary_mask(umgs, openmvs)
    desc = ev.descriptive_metrics(umgs["expected_camera_z"], openmvs["depth"], primary)
    gd = ev.corrected.reference_high_gradient_domain(openmvs["depth"], primary)
    true_core = ev.corrected.metrics_on_mask(openmvs["depth"], umgs["expected_camera_z"], primary, gradient_domain=gd)
    high_sha = ev.mask_packbits_sha256(gd.high_mask)
    result = {
        **ev.EXPECTED_STATUS,
        "shuffle_seed": ev.SHUFFLE_SEED,
        "support_counts": {"primary_common_count": int(primary.sum()), "primary_common_ratio": float(primary.sum() / primary.size)},
        "true_descriptive_metrics": desc,
        "true_core_metrics_on_shared": {
            "spearman": true_core["spearman"],
            "high_gradient_cosine_median": true_core["high_gradient_cosine_median"],
            "high_gradient_pixels": int(gd.high_count),
            "high_gradient_threshold": float(gd.threshold),
            "gradient_local_valid_pixels": int(gd.local_valid_count),
        },
    }
    constants = {
        "EXPECTED_PRIMARY_COMMON_COUNT": int(primary.sum()),
        "EXPECTED_PRIMARY_COMMON_RATIO": float(primary.sum() / primary.size),
        "EXPECTED_GRADIENT_LOCAL_VALID_COUNT": int(gd.local_valid_count),
        "EXPECTED_HIGH_GRADIENT_THRESHOLD": float(gd.threshold),
        "EXPECTED_HIGH_GRADIENT_PIXELS": int(gd.high_count),
        "EXPECTED_HIGH_GRADIENT_PACKBITS_SHA256": high_sha,
        "EXPECTED_SCALARS": {
            "mean_absolute_camera_z_disagreement": desc["mean_absolute_camera_z_disagreement"],
            "median_absolute_camera_z_disagreement": desc["median_absolute_camera_z_disagreement"],
            "rmse_camera_z_disagreement": desc["rmse_camera_z_disagreement"],
            "spearman": true_core["spearman"],
            "high_gradient_cosine_median": true_core["high_gradient_cosine_median"],
        },
    }
    return result, constants


def expect_protocol_error(fn, code: str) -> ev.ProtocolError:
    try:
        fn()
    except ev.ProtocolError as exc:
        assert code in exc.code or code in str(exc), (code, exc.code, exc)
        return exc
    raise AssertionError(f"expected {code}")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_synthetic_execution(tmp_path: Path) -> tuple[argparse.Namespace, dict[str, Any]]:
    result, constants = synthetic_result_from_arrays(tmp_path / "probe")
    with patched_constants(**constants):
        args = build_args(tmp_path, result, output_name="execute")
        args.expected_primary_common_count = constants["EXPECTED_PRIMARY_COMMON_COUNT"]
        args.expected_primary_common_ratio = constants["EXPECTED_PRIMARY_COMMON_RATIO"]
        args.expected_gradient_local_valid_count = constants["EXPECTED_GRADIENT_LOCAL_VALID_COUNT"]
        args.expected_high_gradient_threshold = constants["EXPECTED_HIGH_GRADIENT_THRESHOLD"]
        args.expected_high_gradient_pixels = constants["EXPECTED_HIGH_GRADIENT_PIXELS"]
        args.expected_high_gradient_packbits_sha256 = constants["EXPECTED_HIGH_GRADIENT_PACKBITS_SHA256"]
        ev.execute_localization(args)
        return args, constants


def test_preflight_and_no_spatial_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        result, constants = synthetic_result_from_arrays(tmp_path / "probe")
        with patched_constants(**constants):
            args = build_args(tmp_path, result, output_name="preflight")
            payload = ev.run_preflight(args, write_outputs=True)
            assert payload["preflight_status"] == "pass"
            assert payload["real_localization_computed"] is False
            assert payload["spatial_statistics_computed"] is False
            assert payload["png_generated"] is False
            assert payload["high_gradient_mask_generated"] is False
            assert not any(args.output_root.glob("*.npz"))
            assert not any(args.output_root.rglob("*.png"))


def test_execute_localization_synthetic_end_to_end_outputs_and_summaries() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        args, constants = run_synthetic_execution(Path(tmp))
        npz = args.output_root / "road0001_spatial_residual_localization_arrays.npz"
        metadata = args.output_root / "road0001_spatial_residual_localization_metadata.json"
        manifest = args.output_root / "ROAD0001_SPATIAL_LOCALIZATION_OUTPUT_MANIFEST.csv"
        summary_json = args.output_root / "road0001_spatial_residual_localization_execution_summary.json"
        summary_csv = args.output_root / "road0001_spatial_residual_localization_execution_summary.csv"
        grid = args.output_root / "road0001_spatial_residual_grid_stats.csv"
        border = args.output_root / "road0001_spatial_residual_border_stats.csv"
        assert npz.exists() and metadata.exists() and manifest.exists() and summary_json.exists() and summary_csv.exists()
        assert len(read_csv_rows(grid)) == 64
        assert len(read_csv_rows(border)) == 4
        with np.load(npz) as arr:
            assert arr["primary_common_mask"].dtype == np.uint8
            assert arr["grid_id_map"].dtype == np.int16
            assert arr["border_band_id_map"].dtype == np.int8
            assert arr["signed_camera_z_disagreement"].shape == (ev.EXPECTED_HEIGHT, ev.EXPECTED_WIDTH)
            cosine = arr["per_pixel_gradient_cosine"]
            high = arr["frozen_high_gradient_mask"].astype(bool)
            actual = float(np.median(cosine[high & np.isfinite(cosine)].astype(np.float64)))
            expected = constants["EXPECTED_SCALARS"]["high_gradient_cosine_median"]
            assert abs(actual - expected) <= ev.SCALAR_ATOL + ev.SCALAR_RTOL * abs(expected)
        meta = json.loads(metadata.read_text(encoding="utf-8"))
        assert meta["high_gradient"]["packbits_sha256"] == constants["EXPECTED_HIGH_GRADIENT_PACKBITS_SHA256"]
        assert "road0001_spatial_residual_localization_metadata.json" not in meta["outputs"]
        assert meta["visualization_dependencies"]["matplotlib"]
        assert meta["visualization_dependencies"]["Pillow"]
        rows = read_csv_rows(manifest)
        manifest_paths = {r["path"] for r in rows}
        assert "road0001_spatial_residual_localization_metadata.json" in manifest_paths
        for row in rows:
            p = args.output_root / row["path"]
            assert p.exists()
            assert row["sha256"] == ev.sha256_file(p)
            assert int(row["size_bytes"]) == p.stat().st_size
        summary = json.loads(summary_json.read_text(encoding="utf-8"))
        assert summary["execution_status"] == "pass"
        assert summary["real_localization_computed"] is True
        assert summary["spatial_statistics_computed"] is True
        assert summary["high_gradient_mask_generated"] is True
        assert summary["png_generated"] is True
        assert summary["formal_result_consistency"] == "pass"
        assert summary["high_gradient_identity"] == "pass"
        assert summary["does_not_change_layer2_taxonomy"] is True
        assert summary["phase"] == ev.PHASE
        assert summary["grid_row_count"] == 64
        assert summary["border_row_count"] == 4
        assert summary["official_layer2_interpretation_status"] == "proxy_alignment_weak_or_mixed"


def test_all_eight_registered_pngs_and_real_colormaps() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        args, _ = run_synthetic_execution(Path(tmp))
        pngs = sorted((args.output_root / "png_previews").glob("*.png"))
        assert len(pngs) == 8
        assert {p.name for p in pngs} == {name for _, name in ev.PNG_SPECS}
        assert "RdBu_r" in ev.colormaps
        assert "viridis" in ev.colormaps
        meta = json.loads((args.output_root / "road0001_spatial_residual_localization_metadata.json").read_text(encoding="utf-8"))
        assert meta["display_ranges"]["signed_camera_z_disagreement"]["colormap"] == "RdBu_r"
        assert meta["display_ranges"]["absolute_camera_z_disagreement"]["colormap"] == "viridis"


def test_formal_spearman_and_high_gradient_mismatch_stop_before_output() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        result, constants = synthetic_result_from_arrays(tmp_path / "probe")
        bad_spearman = json.loads(json.dumps(result))
        bad_spearman["true_core_metrics_on_shared"]["spearman"] += 1e-4
        with patched_constants(**constants):
            args = build_args(tmp_path, bad_spearman, output_name="bad_spearman")
            expect_protocol_error(lambda: ev.run_preflight(args, write_outputs=True), "formal_result_scalar_mismatch")
            assert not args.output_root.exists()
        bad_hg = json.loads(json.dumps(result))
        bad_hg["true_core_metrics_on_shared"]["high_gradient_cosine_median"] += 1e-4
        with patched_constants(**constants):
            args = build_args(tmp_path, bad_hg, output_name="bad_hg")
            expect_protocol_error(lambda: ev.run_preflight(args, write_outputs=True), "formal_result_scalar_mismatch")
            assert not args.output_root.exists()


def test_runtime_manifest_source_sha_and_schema_fail_fast() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        result, constants = synthetic_result_from_arrays(tmp_path / "probe")
        with patched_constants(**constants):
            args = build_args(tmp_path, result, output_name="bad_runtime")
            manifest = json.loads(args.localization_runtime_source_manifest.read_text(encoding="utf-8"))
            manifest["sources"][0]["sha256"] = "0" * 64
            write_json(args.localization_runtime_source_manifest, manifest)
            args.localization_runtime_source_manifest_sha256 = sha(args.localization_runtime_source_manifest)
            expect_protocol_error(lambda: ev.run_preflight(args, write_outputs=False), "input_hash_mismatch")
            args = build_args(tmp_path, result, output_name="bad_schema", runtime_schema="wrong_schema")
            expect_protocol_error(lambda: ev.run_preflight(args, write_outputs=False), "runtime_manifest_schema_mismatch")


def test_output_and_staging_overwrite_guards() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        result, constants = synthetic_result_from_arrays(tmp_path / "probe")
        with patched_constants(**constants):
            args = build_args(tmp_path, result, output_name="exists")
            args.output_root.mkdir()
            expect_protocol_error(lambda: ev.run_preflight(args, write_outputs=True), "output_overwrite_guard")
            args = build_args(tmp_path, result, output_name="stage_exists")
            staging = args.output_root.with_name(args.output_root.name + "_STAGING")
            staging.mkdir()
            expect_protocol_error(lambda: ev.execute_localization(args), "staging_overwrite_guard")
            assert staging.exists()


def test_injected_npz_and_png_failures_preserve_failure_staging_no_final_root() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        result, constants = synthetic_result_from_arrays(tmp_path / "probe")
        with patched_constants(**constants):
            args = build_args(tmp_path, result, output_name="npz_fail")

            def fail_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
                raise ev.ProtocolError("injected_npz_failure", "synthetic npz failure")

            with patched_attr(ev, "write_localization_npz", fail_npz):
                exc = expect_protocol_error(lambda: ev.execute_localization(args), "injected_npz_failure")
            assert not args.output_root.exists()
            failure_root = Path(exc.details["failure_root"])
            assert failure_root.exists()
            assert (failure_root / "road0001_spatial_residual_localization_failure.json").exists()

            args = build_args(tmp_path, result, output_name="png_fail")

            def fail_png(path: Path, values: np.ndarray, valid: np.ndarray, display: dict[str, Any]) -> None:
                raise ev.ProtocolError("injected_png_failure", "synthetic png failure")

            with patched_attr(ev, "save_png", fail_png):
                exc = expect_protocol_error(lambda: ev.execute_localization(args), "injected_png_failure")
            assert not args.output_root.exists()
            failure_root = Path(exc.details["failure_root"])
            assert failure_root.exists()
            assert (failure_root / "road0001_spatial_residual_localization_failure.json").exists()


def test_missing_image_dependency_fail_fast() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        result, constants = synthetic_result_from_arrays(tmp_path / "probe")
        with patched_constants(**constants):
            args = build_args(tmp_path, result, output_name="image_dep_fail")

            class EmptyColormaps(dict):
                def __contains__(self, key: object) -> bool:
                    return False

            with patched_attr(ev, "colormaps", EmptyColormaps()):
                exc = expect_protocol_error(lambda: ev.execute_localization(args), "visualization_dependency_or_colormap_missing")
            assert not args.output_root.exists()
            assert Path(exc.details["failure_root"]).exists()


def test_future_command_no_rm_rf_not_executed_and_parser_smoke() -> None:
    base = [
        "--umgs-npz", "u.npz", "--umgs-npz-sha256", "0", "--umgs-manifest", "u.json", "--umgs-manifest-sha256", "0",
        "--openmvs-npz", "o.npz", "--openmvs-npz-sha256", "0", "--openmvs-manifest", "o.json", "--openmvs-manifest-sha256", "0",
        "--formal-result-json", "r.json", "--formal-result-json-sha256", "0", "--formal-result-csv", "r.csv", "--formal-result-csv-sha256", "0",
        "--frame-scale-decision-json", "f.json", "--frame-scale-decision-sha256", "0", "--layer2-runtime-source-manifest", "l.json", "--layer2-runtime-source-manifest-sha256", "0",
        "--layer2-evaluator-sha256", "0", "--corrected-source-sha256", "0", "--camera-fingerprint", "c.json", "--camera-fingerprint-sha256", "0",
        "--camera-fingerprint-payload-sha256", "0", "--localization-evaluator-sha256", "0", "--localization-runtime-source-manifest", "m.json",
        "--localization-runtime-source-manifest-sha256", "0", "--output-root", "out",
    ]
    args = ev.parse_args(["--preflight-only", *base])
    assert args.preflight_only and not args.execute_localization
    args = ev.parse_args(["--execute-localization", *base])
    assert args.execute_localization and not args.preflight_only
    cmd = ev.build_future_command_not_executed(args, python_exe="python")
    assert "NOT_EXECUTED" in cmd
    assert "rm -rf" not in cmd
    assert 'test ! -e "$OUTPUT_ROOT"' in cmd


def test_output_manifest_and_metadata_no_stale_self_sha() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        args, _ = run_synthetic_execution(Path(tmp))
        meta_path = args.output_root / "road0001_spatial_residual_localization_metadata.json"
        manifest_path = args.output_root / "ROAD0001_SPATIAL_LOCALIZATION_OUTPUT_MANIFEST.csv"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert "metadata_sha256" not in json.dumps(meta)
        assert meta["metadata_self_sha256_recorded"] is False
        assert meta_path.name not in meta["outputs"]
        rows = read_csv_rows(manifest_path)
        assert any(r["path"] == meta_path.name for r in rows)
        for row in rows:
            assert ev.sha256_file(args.output_root / row["path"]) == row["sha256"]


def test_corrected_spearman_and_high_gradient_recomputed_consistently() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        result, constants = synthetic_result_from_arrays(tmp_path / "probe")
        with patched_constants(**constants):
            args = build_args(tmp_path, result, output_name="execute")
            ev.execute_localization(args)
            meta = json.loads((args.output_root / "road0001_spatial_residual_localization_metadata.json").read_text(encoding="utf-8"))
            assert abs(meta["formal_recomputed_scalars"]["spearman"] - constants["EXPECTED_SCALARS"]["spearman"]) <= ev.SCALAR_ATOL + ev.SCALAR_RTOL * abs(constants["EXPECTED_SCALARS"]["spearman"])
            assert abs(meta["formal_recomputed_scalars"]["high_gradient_cosine_median"] - constants["EXPECTED_SCALARS"]["high_gradient_cosine_median"]) <= ev.SCALAR_ATOL + ev.SCALAR_RTOL * abs(constants["EXPECTED_SCALARS"]["high_gradient_cosine_median"])


def run_all_tests() -> None:
    tests = [
        test_preflight_and_no_spatial_outputs,
        test_execute_localization_synthetic_end_to_end_outputs_and_summaries,
        test_all_eight_registered_pngs_and_real_colormaps,
        test_formal_spearman_and_high_gradient_mismatch_stop_before_output,
        test_runtime_manifest_source_sha_and_schema_fail_fast,
        test_output_and_staging_overwrite_guards,
        test_injected_npz_and_png_failures_preserve_failure_staging_no_final_root,
        test_missing_image_dependency_fail_fast,
        test_future_command_no_rm_rf_not_executed_and_parser_smoke,
        test_output_manifest_and_metadata_no_stale_self_sha,
        test_corrected_spearman_and_high_gradient_recomputed_consistently,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"ALL_PASS {len(tests)}/{len(tests)}")


if __name__ == "__main__":
    run_all_tests()
