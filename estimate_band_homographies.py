from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from utils.minima_bridge import build_minima_matcher, check_backend_available
from utils.minima_match_utils import (
    accept_frame_for_global_pool,
    build_candidate_pool_schedule,
    compute_match_spatial_coverage,
    estimate_homography_ransac,
    filter_matches_by_confidence,
    homography_pair_displacement_summary,
    homography_stability_diagnostics,
    normalize_homography_matrix,
    robust_aggregate_homographies_weighted,
    score_frame_alignment_quality,
)
from utils.rectification_utils import (
    build_metadata_assisted_h0,
    build_naive_h0,
    determine_pass_from_summary,
    determine_qa_status_from_summary,
    evaluate_transform_on_frames,
    metadata_has_alignment_prior,
    optimize_global_transform_opencv_search,
    optimize_global_transform_scipy,
    prepare_frame_batch,
    scale_transform_from_alignment,
    scale_transform_to_alignment,
    select_representative_frames,
    write_rectification_diagnostics_json,
)
from utils.spectral_image_utils import load_image_preserve_dtype


DEFAULT_BANDS = ("G", "R", "RE", "NIR")


def _parse_bands(bands) -> List[str]:
    if bands is None:
        return list(DEFAULT_BANDS)
    if isinstance(bands, str):
        items = [item.strip() for item in bands.split(",")]
    else:
        items = [str(item).strip() for item in bands]
    parsed: List[str] = []
    for item in items:
        if not item:
            continue
        if item not in parsed:
            parsed.append(item)
    if not parsed:
        raise ValueError("No valid bands provided.")
    return parsed


def _load_manifest(scene_root: Path) -> Dict[str, object]:
    manifest_path = scene_root / "spectral_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing spectral_manifest.json: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _image_map(manifest: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    return {
        str(item.get("image_name", "")).strip(): item
        for item in manifest.get("images", [])
        if str(item.get("image_name", "")).strip()
    }


def _safe_float(value, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def load_frame_records(prepared_root: Path, band_name: str) -> List[dict]:
    rgb_scene_root = prepared_root / "RGB"
    raw_scene_root = prepared_root / f"{band_name}_raw"
    rgb_map = _image_map(_load_manifest(rgb_scene_root))
    band_map = _image_map(_load_manifest(raw_scene_root))
    common_names = sorted(set(rgb_map.keys()) & set(band_map.keys()))
    records = []
    skipped_missing_rgb_plane = 0
    skipped_missing_band_source = 0
    for image_name in common_names:
        rgb_item = rgb_map[image_name]
        band_item = band_map[image_name]
        rgb_path = rgb_scene_root / "images" / image_name
        band_path = Path(str(band_item.get("source_path") or (raw_scene_root / "images" / image_name)))
        if not rgb_path.exists():
            skipped_missing_rgb_plane += 1
            continue
        if not band_path.exists():
            skipped_missing_band_source += 1
            continue
        records.append(
            {
                "frame_id": str(band_item.get("frame_id", image_name)),
                "image_name": image_name,
                "rgb_path": str(rgb_path),
                "band_path": str(band_path),
                "rgb_meta": rgb_item.get("metadata", {}) if isinstance(rgb_item, dict) else {},
                "band_meta": band_item.get("metadata", {}) if isinstance(band_item, dict) else {},
            }
        )
    if skipped_missing_rgb_plane or skipped_missing_band_source:
        print(
            f"[rectification:{band_name}] filtered unavailable frames: "
            f"missing_rgb_plane={skipped_missing_rgb_plane}, "
            f"missing_band_source={skipped_missing_band_source}, "
            f"usable={len(records)}"
        )
    return records


def _build_min_good_frames(total_frames: int, requested: int) -> int:
    if requested and requested > 0:
        return int(max(4, min(total_frames, requested)))
    default_target = max(8, min(20, int(round(0.15 * max(total_frames, 1)))))
    return int(max(4, min(total_frames, default_target)))


def _candidate_records_by_count(records: Sequence[dict], count: int, min_structure_score: float | None) -> List[dict]:
    return select_representative_frames(
        frame_records=list(records),
        max_frames=int(count),
        min_structure_score=min_structure_score,
    )


def _summarize_numeric(values: Sequence[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "min": None,
            "mean": None,
            "median": None,
            "max": None,
            "std": None,
        }
    return {
        "n": int(arr.size),
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
    }


def _summarize_match_attempts(match_attempts: Sequence[dict]) -> dict:
    accepted = [item for item in match_attempts if bool(item.get("accepted", False))]
    keys = (
        "num_matches_raw",
        "num_matches_after_conf",
        "num_inliers",
        "inlier_ratio",
        "reproj_error",
        "coverage",
        "conf_mean",
        "quality_score",
    )

    def summarize_subset(items: Sequence[dict]) -> dict:
        return {
            key: _summarize_numeric([float(item.get(key, np.nan)) for item in items])
            for key in keys
        }

    total = int(len(match_attempts))
    return {
        "num_attempted": total,
        "num_accepted": int(len(accepted)),
        "accepted_ratio": float(len(accepted)) / float(max(total, 1)),
        "all": summarize_subset(match_attempts),
        "accepted": summarize_subset(accepted),
    }


def _summary_value(summary: dict, key: str, default: float) -> float:
    value = summary.get(key, default) if isinstance(summary, dict) else default
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _build_candidate_diagnostics(
    band_name: str,
    records: Sequence[dict],
    accepted_items: Sequence[dict],
    all_diags: Sequence[dict],
    min_good_frames: int,
    config: dict,
) -> dict:
    accepted_sorted = sorted(accepted_items, key=lambda item: item["quality_score"], reverse=True)
    t_full = robust_aggregate_homographies_weighted(
        [item["homography"] for item in accepted_sorted],
        [max(item["quality_score"], 1e-6) for item in accepted_sorted],
    )

    refine_result = _maybe_refine_transform(
        t_full=t_full,
        accepted_records=[item["record"] for item in accepted_sorted],
        config=config,
    )
    t_opt_full = np.asarray(refine_result["T_opt"], dtype=np.float64)

    eval_records = select_representative_frames(
        frame_records=records,
        max_frames=int(config.get("rectification_estimation_frames", 12)),
        min_structure_score=config.get("rectification_min_structure_score", None),
    )
    if not eval_records:
        eval_records = [item["record"] for item in accepted_sorted[: max(6, min(12, len(accepted_sorted)))]]  # type: ignore[index]

    frame_batch = prepare_frame_batch(
        frame_records=eval_records,
        input_dynamic_range=str(config["input_dynamic_range"]),
        radiometric_mode=str(config["radiometric_mode"]),
        alignment_scale=float(config.get("rectification_alignment_scale", 0.25)),
        alignment_max_dim=int(config.get("rectification_alignment_max_dim", 640)),
    )
    reference_frame = frame_batch[0]
    t_opt_align = scale_transform_to_alignment(t_opt_full, reference_frame["scale_adapters"])
    evaluation = evaluate_transform_on_frames(t_opt_align, frame_batch, config)

    legacy_pass = determine_pass_from_summary(
        evaluation,
        min_improved_ratio=float(config.get("rectification_min_improved_ratio", 0.6)),
        max_severe_outliers=int(config.get("rectification_max_severe_outliers", 0)),
    )

    rgb_shape_full = reference_frame["rgb_full"].shape
    band_shape_full = reference_frame["band_full"].shape
    rgb_meta = reference_frame.get("rgb_meta", {})
    band_meta = reference_frame.get("band_meta", {})
    naive_h0 = build_naive_h0(rgb_shape_full, band_shape_full)
    metadata_h0 = build_metadata_assisted_h0(rgb_meta, band_meta, rgb_shape_full, band_shape_full)
    use_metadata = bool(config.get("rectification_use_metadata_h0", True))
    h0_source = "metadata_assisted" if use_metadata and metadata_has_alignment_prior(rgb_meta, band_meta) else "naive_fallback"
    h0 = metadata_h0 if use_metadata else naive_h0
    stability_grid = int(config.get("rectification_stability_grid", 5))
    accepted_weights = [max(item["quality_score"], 1e-6) for item in accepted_sorted]
    homography_stability = homography_stability_diagnostics(
        [item["homography"] for item in accepted_sorted],
        accepted_weights,
        aggregate_homography=t_full,
        source_shape=band_shape_full,
        grid_size=stability_grid,
    )
    for item, stability_item in zip(accepted_sorted, homography_stability.get("per_homography", [])):
        item["diag"].update(
            {
                "aggregation_weight": float(stability_item.get("aggregation_weight", 0.0)),
                "robust_keep": bool(stability_item.get("robust_keep", False)),
                "param_distance_to_median": float(stability_item.get("param_distance_to_median", 0.0)),
                "disp_vs_aggregate_grid_mean_px": float(stability_item.get("disp_vs_aggregate_grid_mean_px", 0.0)),
                "disp_vs_aggregate_grid_median_px": float(stability_item.get("disp_vs_aggregate_grid_median_px", 0.0)),
                "disp_vs_aggregate_grid_max_px": float(stability_item.get("disp_vs_aggregate_grid_max_px", 0.0)),
            }
        )
    h0_vs_topt = homography_pair_displacement_summary(
        h0,
        t_opt_full,
        source_shape=band_shape_full,
        grid_size=stability_grid,
    )
    h0_vs_aggregate = homography_pair_displacement_summary(
        h0,
        t_full,
        source_shape=band_shape_full,
        grid_size=stability_grid,
    )
    match_summary = _summarize_match_attempts(all_diags)
    qa_info = determine_qa_status_from_summary(
        evaluation,
        legacy_pass=bool(legacy_pass),
        band_name=band_name,
        h0_source=h0_source,
        match_summary=match_summary,
        homography_stability=homography_stability,
        config=config,
        min_improved_ratio=float(config.get("rectification_min_improved_ratio", 0.6)),
        max_severe_outliers=int(config.get("rectification_max_severe_outliers", 0)),
    )

    stability_warn_px = float(config.get("rectification_stability_warn_mean_px", 25.0))
    max_reject_ratio = float(config.get("rectification_stability_max_reject_ratio", 0.25))
    median_disp = _summary_value(
        homography_stability.get("disp_vs_aggregate_grid_mean_px", {}),
        "median",
        float("inf"),
    )
    robust_reject_ratio = float(homography_stability.get("robust_reject_ratio", 1.0))
    accepted_enough = len(accepted_items) >= int(min_good_frames)
    stability_ok = median_disp <= stability_warn_px and robust_reject_ratio <= max_reject_ratio
    qa_ok = str(qa_info["qa_status"]) in {"pass", "pass_with_warning"}
    stop_criteria = {
        "accepted_enough": bool(accepted_enough),
        "stability_ok": bool(stability_ok),
        "qa_ok": bool(qa_ok),
        "should_stop": bool(accepted_enough and stability_ok and qa_ok),
        "median_disp_to_aggregate_mean_px": float(median_disp),
        "median_disp_to_aggregate_mean_rel": _summary_value(
            homography_stability.get("disp_vs_aggregate_grid_mean_rel", {}),
            "median",
            float("inf"),
        ),
        "stability_warn_mean_px": float(stability_warn_px),
        "robust_reject_ratio": float(robust_reject_ratio),
        "max_robust_reject_ratio": float(max_reject_ratio),
        "qa_status": qa_info["qa_status"],
        "qa_warning_codes": qa_info["qa_warning_codes"],
        "qa_decision_inputs": qa_info["decision_inputs"],
        "warning_path_reason": qa_info["warning_path_reason"],
    }

    return {
        "accepted_sorted": accepted_sorted,
        "t_full": np.asarray(t_full, dtype=np.float64),
        "refine_result": refine_result,
        "t_opt_full": t_opt_full,
        "eval_records": eval_records,
        "evaluation": evaluation,
        "band_pass": bool(qa_info["qa_pass"]),
        "legacy_pass": bool(legacy_pass),
        "rgb_shape_full": rgb_shape_full,
        "band_shape_full": band_shape_full,
        "naive_h0": naive_h0,
        "h0": h0,
        "h0_source": h0_source,
        "homography_stability": homography_stability,
        "h0_vs_topt": h0_vs_topt,
        "h0_vs_aggregate": h0_vs_aggregate,
        "match_summary": match_summary,
        "qa_info": qa_info,
        "stop_criteria": stop_criteria,
    }


def _run_matching_for_record(matcher, record: dict, config: dict) -> dict:
    match_res = matcher.match(record["rgb_path"], record["band_path"])
    mkpts0 = np.asarray(match_res.get("mkpts0", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32).reshape(-1, 2)
    mkpts1 = np.asarray(match_res.get("mkpts1", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32).reshape(-1, 2)
    mconf = np.asarray(match_res.get("mconf", np.ones((mkpts0.shape[0],), dtype=np.float32)), dtype=np.float32).reshape(-1)

    conf_thresh = float(config.get("minima_match_conf_thresh", 0.20))
    filtered = filter_matches_by_confidence(mkpts0, mkpts1, mconf, conf_thresh=conf_thresh)

    h_stats = estimate_homography_ransac(
        filtered["mkpts0"],
        filtered["mkpts1"],
        method=str(config.get("minima_ransac_method", "usac_magsac")),
        ransac_thresh=float(config.get("minima_ransac_thresh", 3.0)),
        confidence=float(config.get("minima_ransac_confidence", 0.999)),
        max_iters=int(config.get("minima_ransac_max_iters", 10000)),
    )

    loaded_band = load_image_preserve_dtype(record["band_path"])
    band_shape = (int(loaded_band.height), int(loaded_band.width))
    coverage = compute_match_spatial_coverage(
        filtered["mkpts1"],
        image_shape=band_shape,
        grid_size=int(config.get("minima_coverage_grid", 4)),
    )
    conf_stats = {
        "mean": float(np.mean(filtered["mconf"])) if filtered["mconf"].size else 0.0,
        "median": float(np.median(filtered["mconf"])) if filtered["mconf"].size else 0.0,
    }
    quality = score_frame_alignment_quality(h_stats, coverage=coverage, conf_stats=conf_stats)
    accepted = accept_frame_for_global_pool(h_stats, coverage=coverage, cfg=config)

    diag = {
        "frame_id": record["frame_id"],
        "image_name": record["image_name"],
        "num_matches_raw": int(mkpts0.shape[0]),
        "num_matches_after_conf": int(filtered["mkpts0"].shape[0]),
        "num_inliers": int(h_stats["num_inliers"]),
        "inlier_ratio": float(h_stats["inlier_ratio"]),
        "reproj_error": float(h_stats["reproj_error"]),
        "coverage": float(coverage),
        "conf_mean": conf_stats["mean"],
        "conf_median": conf_stats["median"],
        "quality_score": float(quality),
        "accepted": bool(accepted),
        "matcher_debug": match_res.get("debug", {}),
    }
    return {
        "record": record,
        "diag": diag,
        "homography": np.asarray(h_stats["H"], dtype=np.float64),
        "quality_score": float(quality),
        "accepted": bool(accepted),
    }


def _maybe_refine_transform(
    t_full: np.ndarray,
    accepted_records: List[dict],
    config: dict,
) -> dict:
    if not bool(config.get("rectification_enable_residual_refine", False)):
        return {
            "T_opt": np.asarray(t_full, dtype=np.float64),
            "optimizer": {
                "backend": "none",
                "num_restarts": 0,
                "num_iterations": 0,
                "history": [],
            },
            "theta_opt": [],
        }

    refine_count = int(max(4, min(len(accepted_records), int(config.get("rectification_refine_frames", 6)))))
    refine_records = accepted_records[:refine_count]
    print(
        f"[rectification:refine] enabled with frames={len(refine_records)}, "
        f"alignment_scale={float(config.get('rectification_alignment_scale', 0.25))}, "
        f"alignment_max_dim={int(config.get('rectification_alignment_max_dim', 640))}, "
        f"backend={config.get('rectification_optimizer_backend', 'opencv_search')}",
        flush=True,
    )
    frame_batch = prepare_frame_batch(
        frame_records=refine_records,
        input_dynamic_range=str(config["input_dynamic_range"]),
        radiometric_mode=str(config["radiometric_mode"]),
        alignment_scale=float(config.get("rectification_alignment_scale", 0.25)),
        alignment_max_dim=int(config.get("rectification_alignment_max_dim", 640)),
    )
    reference = frame_batch[0]
    h0_align = scale_transform_to_alignment(np.asarray(t_full, dtype=np.float64), reference["scale_adapters"])

    optimizer_backend = str(config.get("rectification_optimizer_backend", "opencv_search"))
    if optimizer_backend == "scipy_minimize":
        opt_result = optimize_global_transform_scipy(h0_align, frame_batch, config)
    else:
        opt_result = optimize_global_transform_opencv_search(h0_align, frame_batch, config)

    t_align = np.asarray(opt_result["best_T"], dtype=np.float64)
    t_refined = scale_transform_from_alignment(t_align, reference["scale_adapters"])
    print(
        f"[rectification:refine] finished score={float(opt_result.get('best_score', 0.0)):.4f} "
        f"iterations={opt_result.get('num_iterations', 0)} restarts={opt_result.get('num_restarts', 0)}",
        flush=True,
    )
    return {
        "T_opt": np.asarray(t_refined, dtype=np.float64),
        "optimizer": {
            "backend": opt_result["backend"],
            "num_restarts": opt_result["num_restarts"],
            "num_iterations": opt_result["num_iterations"],
            "history": opt_result.get("history", []),
        },
        "theta_opt": opt_result.get("best_theta", []),
    }


def _estimate_for_band(prepared_root: Path, band_name: str, config: dict, matcher) -> Dict[str, object]:
    records = load_frame_records(prepared_root, band_name)
    if not records:
        raise RuntimeError(f"No paired frames found for band {band_name}")

    n_total = len(records)
    min_good_frames = _build_min_good_frames(
        total_frames=n_total,
        requested=int(config.get("minima_min_good_frames", 0)),
    )
    full_if_frames_le = int(config.get("minima_full_if_frames_le", 300))
    if full_if_frames_le > 0 and n_total <= full_if_frames_le:
        candidate_schedule = [int(n_total)]
        candidate_policy = "all_small_dataset"
    else:
        candidate_schedule = build_candidate_pool_schedule(
            total_frames=n_total,
            base_frames=int(config.get("rectification_estimation_frames", 12)),
            initial_ratio=float(config.get("minima_initial_candidate_ratio", 0.15)),
            ratio_step=float(config.get("minima_candidate_ratio_step", 0.15)),
            max_ratio=float(config.get("minima_max_candidate_ratio", 0.50)),
            include_all=bool(config.get("minima_use_all_if_needed", True)),
        )
        candidate_policy = "adaptive"

    attempted_names = set()
    accepted_items: List[dict] = []
    rejected_items: List[dict] = []
    all_diags: List[dict] = []
    stage_diags: List[dict] = []
    selected_diagnostics: dict | None = None

    print(
        f"[rectification:{band_name}] start MINIMA estimation: total_frames={n_total}, "
        f"min_good_frames={min_good_frames}, policy={candidate_policy}, schedule={candidate_schedule}",
        flush=True,
    )
    for count in candidate_schedule:
        stage_records = _candidate_records_by_count(
            records=records,
            count=count,
            min_structure_score=config.get("rectification_min_structure_score", None),
        )
        new_records = [item for item in stage_records if item["image_name"] not in attempted_names]
        stage_info = {
            "candidate_count": int(count),
            "new_records": len(new_records),
            "accepted_before": len(accepted_items),
            "candidate_policy": candidate_policy,
        }
        print(
            f"[rectification:{band_name}] candidate_count={count}, "
            f"new_records={len(new_records)}, accepted={len(accepted_items)}/{min_good_frames}",
            flush=True,
        )
        for idx, record in enumerate(new_records, start=1):
            attempted_names.add(record["image_name"])
            print(
                f"[rectification:{band_name}] matching {idx}/{len(new_records)} "
                f"frame_id={record.get('frame_id')} image={record.get('image_name')}",
                flush=True,
            )
            result = _run_matching_for_record(matcher, record, config)
            diag = result["diag"]
            diag.update(
                {
                    "candidate_stage": int(count),
                    "candidate_stage_index": int(idx),
                    "homography": normalize_homography_matrix(result["homography"]).tolist(),
                    "aggregation_weight": 0.0,
                    "robust_keep": False,
                }
            )
            all_diags.append(diag)
            if result["accepted"]:
                accepted_items.append(result)
            else:
                rejected_items.append(result)
            print(
                f"[rectification:{band_name}] result accepted={diag['accepted']} "
                f"raw={diag['num_matches_raw']} conf={diag['num_matches_after_conf']} "
                f"inliers={diag['num_inliers']} inlier_ratio={diag['inlier_ratio']:.3f} "
                f"reproj={diag['reproj_error']:.3f} coverage={diag['coverage']:.3f} "
                f"accepted_total={len(accepted_items)}/{min_good_frames}",
                flush=True,
            )
        stage_info["accepted_after"] = len(accepted_items)
        stage_info["attempted_total"] = len(attempted_names)
        stage_info["stop_decision"] = "continue_insufficient_accepted"
        if len(accepted_items) >= min_good_frames:
            selected_diagnostics = _build_candidate_diagnostics(
                band_name=band_name,
                records=records,
                accepted_items=accepted_items,
                all_diags=all_diags,
                min_good_frames=min_good_frames,
                config=config,
            )
            stop_criteria = selected_diagnostics["stop_criteria"]
            stage_info["stop_criteria"] = stop_criteria
            stage_info["qa_status"] = stop_criteria["qa_status"]
            stage_info["qa_warning_codes"] = stop_criteria["qa_warning_codes"]
            stage_info["delta_edge_f1"] = float(selected_diagnostics["evaluation"].get("delta_edge_f1", 0.0))
            stage_info["delta_grad_ncc"] = float(selected_diagnostics["evaluation"].get("delta_grad_ncc", 0.0))
            stage_info["homography_stability"] = {
                "robust_keep_count": int(selected_diagnostics["homography_stability"].get("robust_keep_count", 0)),
                "robust_reject_count": int(selected_diagnostics["homography_stability"].get("robust_reject_count", 0)),
                "robust_reject_ratio": float(selected_diagnostics["homography_stability"].get("robust_reject_ratio", 1.0)),
                "disp_vs_aggregate_grid_mean_px": selected_diagnostics["homography_stability"].get("disp_vs_aggregate_grid_mean_px", {}),
            }
            if bool(stop_criteria["should_stop"]):
                stage_info["stop_decision"] = "stop_accepted_stable_qa"
                stage_diags.append(stage_info)
                print(
                    f"[rectification:{band_name}] stop at candidate_count={count}: "
                    f"qa_status={stop_criteria['qa_status']} "
                    f"median_disp={stop_criteria['median_disp_to_aggregate_mean_px']:.3f}px "
                    f"reject_ratio={stop_criteria['robust_reject_ratio']:.3f}",
                    flush=True,
                )
                break
            stage_info["stop_decision"] = "expand_unstable_or_qa"
            print(
                f"[rectification:{band_name}] expanding candidates: "
                f"qa_status={stop_criteria['qa_status']} "
                f"accepted_enough={stop_criteria['accepted_enough']} "
                f"stability_ok={stop_criteria['stability_ok']} "
                f"qa_ok={stop_criteria['qa_ok']}",
                flush=True,
            )
        stage_diags.append(stage_info)

    if len(accepted_items) < min_good_frames:
        raise RuntimeError(
            f"[{band_name}] MINIMA matching produced insufficient high-quality frames: "
            f"accepted={len(accepted_items)} < min_good_frames={min_good_frames}, attempted={len(attempted_names)}, total={n_total}"
        )
    if selected_diagnostics is None:
        selected_diagnostics = _build_candidate_diagnostics(
            band_name=band_name,
            records=records,
            accepted_items=accepted_items,
            all_diags=all_diags,
            min_good_frames=min_good_frames,
            config=config,
        )
    print(
        f"[rectification:{band_name}] accepted {len(accepted_items)} high-quality frames "
        f"after attempting {len(attempted_names)}/{n_total}",
        flush=True,
    )

    accepted_sorted = selected_diagnostics["accepted_sorted"]
    t_full = selected_diagnostics["t_full"]
    refine_result = selected_diagnostics["refine_result"]
    t_opt_full = selected_diagnostics["t_opt_full"]
    eval_records = selected_diagnostics["eval_records"]
    evaluation = selected_diagnostics["evaluation"]
    band_pass = selected_diagnostics["band_pass"]
    h0_source = selected_diagnostics["h0_source"]
    h0 = selected_diagnostics["h0"]
    naive_h0 = selected_diagnostics["naive_h0"]
    homography_stability = selected_diagnostics["homography_stability"]
    h0_vs_topt = selected_diagnostics["h0_vs_topt"]
    h0_vs_aggregate = selected_diagnostics["h0_vs_aggregate"]
    match_summary = selected_diagnostics["match_summary"]
    qa_info = selected_diagnostics["qa_info"]
    stop_criteria = selected_diagnostics["stop_criteria"]
    print(
        f"[rectification:{band_name}] QA pass={bool(band_pass)} "
        f"legacy_pass={qa_info['legacy_pass']} "
        f"qa_status={qa_info['qa_status']} "
        f"delta_edge={evaluation.get('delta_edge_f1', 0.0):.4f} "
        f"delta_grad={evaluation.get('delta_grad_ncc', 0.0):.4f} "
        f"stop_decision={stop_criteria['should_stop']}",
        flush=True,
    )

    payload = {
        "num_candidate_frames": int(n_total),
        "num_attempted_frames": int(len(attempted_names)),
        "num_accepted_frames": int(len(accepted_items)),
        "candidate_policy": candidate_policy,
        "final_stop_criteria": stop_criteria,
        "selected_frame_ids": [item["record"]["frame_id"] for item in accepted_sorted],
        "selected_image_names": [item["image_name"] for item in eval_records],
        "qa_representative_frame_ids": evaluation.get("representative_frame_ids", []),
        "qa_representative_image_names": evaluation.get("representative_image_names", []),
        "h0_source": h0_source,
        "H0": np.asarray(h0, dtype=np.float64).tolist(),
        "naive_H0": np.asarray(naive_h0, dtype=np.float64).tolist(),
        "theta_opt": refine_result["theta_opt"],
        "T_aggregate": np.asarray(t_full, dtype=np.float64).tolist(),
        "T_opt": t_opt_full.tolist(),
        "optimizer": refine_result["optimizer"],
        "matcher_backend": str(config.get("rectification_backend", "minima")),
        "minima_method": str(config.get("minima_method", "roma")),
        "scores": {k: v for k, v in evaluation.items() if k != "per_frame"},
        "per_frame": evaluation["per_frame"],
        "match_summary": match_summary,
        "homography_stability": homography_stability,
        "H0_vs_Topt_displacement": h0_vs_topt,
        "H0_vs_Taggregate_displacement": h0_vs_aggregate,
        "qa_status": qa_info["qa_status"],
        "qa_warning_codes": qa_info["qa_warning_codes"],
        "legacy_pass": qa_info["legacy_pass"],
        "qa_decision_inputs": qa_info["decision_inputs"],
        "warning_path_reason": qa_info["warning_path_reason"],
        "qa_notes": qa_info["notes"],
        "match_attempts": all_diags,
        "adaptive_schedule": stage_diags,
        "accepted_ratio": float(len(accepted_items)) / float(max(len(attempted_names), 1)),
        "pass": bool(band_pass),
        "notes": [],
    }
    return payload


def estimate_band_homographies(
    prepared_root: Path,
    out_json: Path,
    bands: Sequence[str] | str | None = None,
    frame_count: int = 12,
    input_dynamic_range: str = "uint16",
    radiometric_mode: str = "exposure_normalized",
    rectification_use_metadata_h0: bool = True,
    rectification_global_mode: str = "affine_residual_over_h0",
    rectification_optimizer_backend: str = "opencv_search",
    rectification_search_restarts: int = 5,
    rectification_search_steps: int = 50,
    rectification_edge_dilate_radius: int = 1,
    rectification_residual_reg: float = 1e-3,
    rectification_alignment_scale: float = 0.25,
    rectification_alignment_max_dim: int = 640,
    rectification_refine_frames: int = 6,
    rectification_min_structure_score: float | None = None,
    rectification_debug_use_legacy_ecc: bool = False,
    rectification_min_improved_ratio: float = 0.6,
    rectification_max_severe_outliers: int = 0,
    rectification_high_modality_bands: str = "NIR",
    rectification_warning_min_accepted_ratio: float = 0.80,
    rectification_warning_min_delta_edge_f1: float = -0.05,
    rectification_warning_min_grad_improved_ratio: float = 0.60,
    rectification_warning_max_severe_outlier_ratio: float = 0.10,
    rectification_warning_max_severe_outlier_count: int = 1,
    rectification_stability_grid: int = 5,
    rectification_stability_warn_mean_px: float = 25.0,
    rectification_stability_max_reject_ratio: float = 0.25,
    rectification_backend: str = "minima",
    minima_method: str = "roma",
    minima_root: str = r"G:\2DSOTA\MINIMA",
    minima_device: str = "cuda",
    minima_ckpt: str = "",
    minima_roma_size: str = "large",
    minima_match_threshold: float = 0.3,
    minima_fine_threshold: float = 0.1,
    minima_match_max_dim: int = 1600,
    minima_match_conf_thresh: float = 0.2,
    minima_min_matches: int = 80,
    minima_min_inlier_ratio: float = 0.30,
    minima_max_reproj_error: float = 4.0,
    minima_min_coverage: float = 0.25,
    minima_coverage_grid: int = 4,
    minima_ransac_method: str = "usac_magsac",
    minima_ransac_thresh: float = 3.0,
    minima_ransac_confidence: float = 0.999,
    minima_ransac_max_iters: int = 10000,
    minima_min_good_frames: int = 0,
    minima_initial_candidate_ratio: float = 0.15,
    minima_candidate_ratio_step: float = 0.15,
    minima_max_candidate_ratio: float = 0.50,
    minima_use_all_if_needed: bool = True,
    minima_full_if_frames_le: int = 300,
    rectification_enable_residual_refine: bool = False,
) -> Dict[str, object]:
    prepared_root = prepared_root.resolve()
    out_json = out_json.resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    backend = str(rectification_backend).strip().lower()
    if backend != "minima":
        raise RuntimeError(
            f"Legacy/non-learning rectification backends are disabled in MINIMA-V2. "
            f"Expected rectification_backend='minima', got {rectification_backend!r}."
        )
    if not check_backend_available(minima_method, minima_root=minima_root):
        raise RuntimeError(
            f"MINIMA backend unavailable: method={minima_method}, minima_root={Path(minima_root).resolve()}. "
            f"Please verify dependencies/checkpoints."
        )

    config = {
        "rectification_estimation_frames": int(frame_count),
        "input_dynamic_range": input_dynamic_range,
        "radiometric_mode": radiometric_mode,
        "rectification_use_metadata_h0": bool(rectification_use_metadata_h0),
        "rectification_global_mode": rectification_global_mode,
        "rectification_optimizer_backend": rectification_optimizer_backend,
        "rectification_search_restarts": int(rectification_search_restarts),
        "rectification_search_steps": int(rectification_search_steps),
        "rectification_edge_dilate_radius": int(rectification_edge_dilate_radius),
        "rectification_residual_reg": float(rectification_residual_reg),
        "rectification_alignment_scale": float(rectification_alignment_scale),
        "rectification_alignment_max_dim": int(rectification_alignment_max_dim),
        "rectification_refine_frames": int(rectification_refine_frames),
        "rectification_min_structure_score": rectification_min_structure_score,
        "rectification_debug_use_legacy_ecc": bool(rectification_debug_use_legacy_ecc),
        "rectification_min_improved_ratio": float(rectification_min_improved_ratio),
        "rectification_max_severe_outliers": int(rectification_max_severe_outliers),
        "rectification_high_modality_bands": str(rectification_high_modality_bands),
        "rectification_warning_min_accepted_ratio": float(rectification_warning_min_accepted_ratio),
        "rectification_warning_min_delta_edge_f1": float(rectification_warning_min_delta_edge_f1),
        "rectification_warning_min_grad_improved_ratio": float(rectification_warning_min_grad_improved_ratio),
        "rectification_warning_max_severe_outlier_ratio": float(rectification_warning_max_severe_outlier_ratio),
        "rectification_warning_max_severe_outlier_count": int(rectification_warning_max_severe_outlier_count),
        "rectification_stability_grid": int(rectification_stability_grid),
        "rectification_stability_warn_mean_px": float(rectification_stability_warn_mean_px),
        "rectification_stability_max_reject_ratio": float(rectification_stability_max_reject_ratio),
        "rectification_backend": backend,
        "minima_method": str(minima_method).strip().lower(),
        "minima_root": str(Path(minima_root).resolve()),
        "minima_device": str(minima_device),
        "minima_ckpt": str(minima_ckpt),
        "minima_roma_size": str(minima_roma_size),
        "minima_match_threshold": float(minima_match_threshold),
        "minima_fine_threshold": float(minima_fine_threshold),
        "minima_match_max_dim": int(minima_match_max_dim),
        "minima_match_conf_thresh": float(minima_match_conf_thresh),
        "minima_min_matches": int(minima_min_matches),
        "minima_min_inlier_ratio": float(minima_min_inlier_ratio),
        "minima_max_reproj_error": float(minima_max_reproj_error),
        "minima_min_coverage": float(minima_min_coverage),
        "minima_coverage_grid": int(minima_coverage_grid),
        "minima_ransac_method": str(minima_ransac_method),
        "minima_ransac_thresh": float(minima_ransac_thresh),
        "minima_ransac_confidence": float(minima_ransac_confidence),
        "minima_ransac_max_iters": int(minima_ransac_max_iters),
        "minima_min_good_frames": int(minima_min_good_frames),
        "minima_initial_candidate_ratio": float(minima_initial_candidate_ratio),
        "minima_candidate_ratio_step": float(minima_candidate_ratio_step),
        "minima_max_candidate_ratio": float(minima_max_candidate_ratio),
        "minima_use_all_if_needed": bool(minima_use_all_if_needed),
        "minima_full_if_frames_le": int(minima_full_if_frames_le),
        "rectification_enable_residual_refine": bool(rectification_enable_residual_refine),
    }
    band_list = _parse_bands(bands)

    payload = {
        "version": 6,
        "rectification_method": "minima_assisted_global_homography",
        "transform_mode": rectification_global_mode,
        "target_plane_root": str((prepared_root / "RGB").resolve()),
        "input_dynamic_range": input_dynamic_range,
        "radiometric_mode": radiometric_mode,
        "bands_order": band_list,
        "config": config,
        "bands": {},
    }

    matcher = build_minima_matcher(
        backend=str(config["minima_method"]),
        minima_root=str(config["minima_root"]),
        device=str(config["minima_device"]),
        ckpt=str(config["minima_ckpt"]),
        roma_size=str(config["minima_roma_size"]),
        match_threshold=float(config["minima_match_threshold"]),
        fine_threshold=float(config["minima_fine_threshold"]),
        match_max_dim=int(config["minima_match_max_dim"]),
    )

    for band_name in band_list:
        payload["bands"][band_name] = _estimate_for_band(prepared_root, band_name, config, matcher=matcher)

    status_counts: Dict[str, int] = {}
    for band_payload in payload["bands"].values():
        status = str(band_payload.get("qa_status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    for status in ("pass", "pass_with_warning", "fail"):
        status_counts.setdefault(status, 0)
    payload["qa_status_counts"] = status_counts
    payload["num_pass"] = int(status_counts.get("pass", 0))
    payload["num_pass_with_warning"] = int(status_counts.get("pass_with_warning", 0))
    payload["num_fail"] = int(status_counts.get("fail", 0))

    write_rectification_diagnostics_json(payload, out_json)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Estimate one global band transform per band to the RGB training plane.")
    ap.add_argument("--prepared_root", required=True, help="Prepared raw root containing RGB and *_raw scenes.")
    ap.add_argument("--out_json", required=True, help="Output JSON file for band-global rectification transforms.")
    ap.add_argument("--bands", default="G,R,RE,NIR", help="Comma-separated list of modalities to estimate (e.g., G,R,RE,NIR or T).")
    ap.add_argument("--frame_count", type=int, default=12)
    ap.add_argument("--input_dynamic_range", default="uint16", choices=["uint8", "uint16", "float"])
    ap.add_argument("--radiometric_mode", default="exposure_normalized", choices=["raw_dn", "exposure_normalized", "reflectance_ready_stub"])
    ap.add_argument("--rectification_use_metadata_h0", type=str, default="true")
    ap.add_argument("--rectification_global_mode", default="affine_residual_over_h0", choices=["affine_residual_over_h0", "projective_residual_over_h0"])
    ap.add_argument("--rectification_optimizer_backend", default="opencv_search", choices=["opencv_search", "scipy_minimize"])
    ap.add_argument("--rectification_search_restarts", type=int, default=5)
    ap.add_argument("--rectification_search_steps", type=int, default=50)
    ap.add_argument("--rectification_edge_dilate_radius", type=int, default=1)
    ap.add_argument("--rectification_residual_reg", type=float, default=1e-3)
    ap.add_argument("--rectification_alignment_scale", type=float, default=0.25)
    ap.add_argument("--rectification_alignment_max_dim", type=int, default=640)
    ap.add_argument("--rectification_refine_frames", type=int, default=6)
    ap.add_argument("--rectification_min_structure_score", type=float, default=None)
    ap.add_argument("--rectification_debug_use_legacy_ecc", type=str, default="false")
    ap.add_argument("--rectification_min_improved_ratio", type=float, default=0.6)
    ap.add_argument("--rectification_max_severe_outliers", type=int, default=0)
    ap.add_argument("--rectification_high_modality_bands", default="NIR")
    ap.add_argument("--rectification_warning_min_accepted_ratio", type=float, default=0.80)
    ap.add_argument("--rectification_warning_min_delta_edge_f1", type=float, default=-0.05)
    ap.add_argument("--rectification_warning_min_grad_improved_ratio", type=float, default=0.60)
    ap.add_argument("--rectification_warning_max_severe_outlier_ratio", type=float, default=0.10)
    ap.add_argument("--rectification_warning_max_severe_outlier_count", type=int, default=1)
    ap.add_argument("--rectification_stability_grid", type=int, default=5)
    ap.add_argument("--rectification_stability_warn_mean_px", type=float, default=25.0)
    ap.add_argument("--rectification_stability_max_reject_ratio", type=float, default=0.25)

    ap.add_argument("--rectification_backend", default="minima", choices=["minima"])
    ap.add_argument("--minima_method", default="roma", choices=["roma", "xoftr"])
    ap.add_argument("--minima_root", default=r"G:\2DSOTA\MINIMA")
    ap.add_argument("--minima_device", default="cuda")
    ap.add_argument("--minima_ckpt", default="")
    ap.add_argument("--minima_roma_size", default="large", choices=["large", "tiny"])
    ap.add_argument("--minima_match_threshold", type=float, default=0.3)
    ap.add_argument("--minima_fine_threshold", type=float, default=0.1)
    ap.add_argument("--minima_match_max_dim", type=int, default=1600)
    ap.add_argument("--minima_match_conf_thresh", type=float, default=0.2)
    ap.add_argument("--minima_min_matches", type=int, default=80)
    ap.add_argument("--minima_min_inlier_ratio", type=float, default=0.30)
    ap.add_argument("--minima_max_reproj_error", type=float, default=4.0)
    ap.add_argument("--minima_min_coverage", type=float, default=0.25)
    ap.add_argument("--minima_coverage_grid", type=int, default=4)
    ap.add_argument("--minima_ransac_method", default="usac_magsac", choices=["usac_magsac", "ransac"])
    ap.add_argument("--minima_ransac_thresh", type=float, default=3.0)
    ap.add_argument("--minima_ransac_confidence", type=float, default=0.999)
    ap.add_argument("--minima_ransac_max_iters", type=int, default=10000)
    ap.add_argument("--minima_min_good_frames", type=int, default=0)
    ap.add_argument("--minima_initial_candidate_ratio", type=float, default=0.15)
    ap.add_argument("--minima_candidate_ratio_step", type=float, default=0.15)
    ap.add_argument("--minima_max_candidate_ratio", type=float, default=0.50)
    ap.add_argument("--minima_use_all_if_needed", type=str, default="true")
    ap.add_argument("--minima_full_if_frames_le", type=int, default=300)

    ap.add_argument("--rectification_enable_residual_refine", type=str, default="false")
    args = ap.parse_args()

    payload = estimate_band_homographies(
        prepared_root=Path(args.prepared_root),
        out_json=Path(args.out_json),
        bands=args.bands,
        frame_count=args.frame_count,
        input_dynamic_range=args.input_dynamic_range,
        radiometric_mode=args.radiometric_mode,
        rectification_use_metadata_h0=str(args.rectification_use_metadata_h0).strip().lower() not in ("0", "false", "no", "off"),
        rectification_global_mode=args.rectification_global_mode,
        rectification_optimizer_backend=args.rectification_optimizer_backend,
        rectification_search_restarts=args.rectification_search_restarts,
        rectification_search_steps=args.rectification_search_steps,
        rectification_edge_dilate_radius=args.rectification_edge_dilate_radius,
        rectification_residual_reg=args.rectification_residual_reg,
        rectification_alignment_scale=args.rectification_alignment_scale,
        rectification_alignment_max_dim=args.rectification_alignment_max_dim,
        rectification_refine_frames=args.rectification_refine_frames,
        rectification_min_structure_score=args.rectification_min_structure_score,
        rectification_debug_use_legacy_ecc=str(args.rectification_debug_use_legacy_ecc).strip().lower() not in ("0", "false", "no", "off"),
        rectification_min_improved_ratio=args.rectification_min_improved_ratio,
        rectification_max_severe_outliers=args.rectification_max_severe_outliers,
        rectification_high_modality_bands=args.rectification_high_modality_bands,
        rectification_warning_min_accepted_ratio=args.rectification_warning_min_accepted_ratio,
        rectification_warning_min_delta_edge_f1=args.rectification_warning_min_delta_edge_f1,
        rectification_warning_min_grad_improved_ratio=args.rectification_warning_min_grad_improved_ratio,
        rectification_warning_max_severe_outlier_ratio=args.rectification_warning_max_severe_outlier_ratio,
        rectification_warning_max_severe_outlier_count=args.rectification_warning_max_severe_outlier_count,
        rectification_stability_grid=args.rectification_stability_grid,
        rectification_stability_warn_mean_px=args.rectification_stability_warn_mean_px,
        rectification_stability_max_reject_ratio=args.rectification_stability_max_reject_ratio,
        rectification_backend=args.rectification_backend,
        minima_method=args.minima_method,
        minima_root=args.minima_root,
        minima_device=args.minima_device,
        minima_ckpt=args.minima_ckpt,
        minima_roma_size=args.minima_roma_size,
        minima_match_threshold=args.minima_match_threshold,
        minima_fine_threshold=args.minima_fine_threshold,
        minima_match_max_dim=args.minima_match_max_dim,
        minima_match_conf_thresh=args.minima_match_conf_thresh,
        minima_min_matches=args.minima_min_matches,
        minima_min_inlier_ratio=args.minima_min_inlier_ratio,
        minima_max_reproj_error=args.minima_max_reproj_error,
        minima_min_coverage=args.minima_min_coverage,
        minima_coverage_grid=args.minima_coverage_grid,
        minima_ransac_method=args.minima_ransac_method,
        minima_ransac_thresh=args.minima_ransac_thresh,
        minima_ransac_confidence=args.minima_ransac_confidence,
        minima_ransac_max_iters=args.minima_ransac_max_iters,
        minima_min_good_frames=args.minima_min_good_frames,
        minima_initial_candidate_ratio=args.minima_initial_candidate_ratio,
        minima_candidate_ratio_step=args.minima_candidate_ratio_step,
        minima_max_candidate_ratio=args.minima_max_candidate_ratio,
        minima_use_all_if_needed=str(args.minima_use_all_if_needed).strip().lower() not in ("0", "false", "no", "off"),
        minima_full_if_frames_le=args.minima_full_if_frames_le,
        rectification_enable_residual_refine=str(args.rectification_enable_residual_refine).strip().lower() not in ("0", "false", "no", "off"),
    )
    print(
        json.dumps(
            {
                "out_json": str(Path(args.out_json).resolve()),
                "bands": {
                    band_name: {
                        "pass": payload["bands"][band_name]["pass"],
                        "legacy_pass": payload["bands"][band_name].get("legacy_pass", None),
                        "qa_status": payload["bands"][band_name].get("qa_status", "unknown"),
                        "qa_warning_codes": payload["bands"][band_name].get("qa_warning_codes", []),
                        "num_attempted_frames": payload["bands"][band_name]["num_attempted_frames"],
                        "num_accepted_frames": payload["bands"][band_name]["num_accepted_frames"],
                        "accepted_ratio": _safe_float(payload["bands"][band_name].get("accepted_ratio", 0.0), 0.0),
                        "scores": payload["bands"][band_name]["scores"],
                    }
                    for band_name in payload.get("bands", {}).keys()
                },
                "qa_status_counts": payload.get("qa_status_counts", {}),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
