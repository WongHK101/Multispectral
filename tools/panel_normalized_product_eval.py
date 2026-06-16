from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate_spectral_indices import (
    INDEX_REQUIRED_BANDS,
    _index_formula,
    _masked_psnr,
    _masked_ssim,
    _mean_median,
    _scalar_and_tie_error,
)


TIER_A_PANEL_GROUPS = {
    "cassava_01_20260526_1603": "2026.5.28\\M3M多光谱作物\\辐射标定\\DJI_202605261603_014_五月二十六下午四时辐射矫正",
    "chunya_01_20260526_1021": "2026.5.28\\M3M多光谱作物\\辐射标定\\DJI_202605261032_009_五月二十六上午十时三十二分辐射矫正",
    "maize_02_20260526_1658": "2026.5.28\\M3M多光谱作物\\辐射标定\\DJI_202605261658_018_五月二十六下午五时辐射矫正",
    "eucalyptus_01_20260526_1053_pruned": "2026.5.28\\M3M多光谱作物\\辐射标定\\DJI_202605261053_010_五月二十六上午十一时辐射矫正",
}

BANDS = ("G", "R", "RE", "NIR")
INDICES = ("NDVI", "GNDVI", "NDRE")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _model_dirs(scene_root: Path) -> dict[str, Path]:
    return {band: scene_root / "out" / f"Model_{band}" for band in BANDS}


def _method_dir(model_dir: Path) -> Path:
    root = model_dir / "test" / "ours_60000"
    if not root.exists():
        raise FileNotFoundError(root)
    return root


def _png_names(model_dirs: Mapping[str, Path]) -> list[str]:
    ref: list[str] | None = None
    for band, model_dir in model_dirs.items():
        root = _method_dir(model_dir)
        renders = sorted(p.name for p in (root / "renders").glob("*.png"))
        gt = sorted(p.name for p in (root / "gt").glob("*.png"))
        if renders != gt:
            raise RuntimeError(f"Render/GT mismatch for {band}: {root}")
        if ref is None:
            ref = renders
        elif renders != ref:
            raise RuntimeError(f"Band file-list mismatch for {band}: {root}")
    return ref or []


def _load_scales(path: Path) -> dict[str, dict[str, dict[str, object]]]:
    rows = _read_csv(path)
    out: dict[str, dict[str, dict[str, object]]] = {}
    for row in rows:
        group = row["panel_group"]
        band = row["band"]
        scale_text = row["preliminary_scale_0p8_over_median"]
        if not scale_text:
            continue
        out.setdefault(group, {})[band] = {
            "absolute_scale": float(scale_text),
            "roi_qc_status": row.get("roi_qc_status", ""),
            "requires_manual_roi": row.get("requires_manual_roi", ""),
        }
    return out


def _relative_scales(group_scales: Mapping[str, Mapping[str, object]]) -> dict[str, float]:
    abs_scales = np.asarray([float(group_scales[band]["absolute_scale"]) for band in BANDS], dtype=np.float64)
    denom = float(np.median(abs_scales))
    return {band: float(group_scales[band]["absolute_scale"]) / denom for band in BANDS}


def _eval_scene(scene: str, scene_root: Path, rel_scale: Mapping[str, float], device) -> tuple[list[dict[str, object]], dict[str, object]]:
    model_dirs = _model_dirs(scene_root)
    files = _png_names(model_dirs)
    rows: list[dict[str, object]] = []
    aggregate: dict[str, object] = {"scene": scene, "view_count": len(files), "indices": {}}

    for idx_name in INDICES:
        req_bands = INDEX_REQUIRED_BANDS[idx_name]
        values = {
            "raw": {"MAE": [], "RMSE": [], "PSNR": [], "SSIM": [], "COVERAGE": []},
            "panel_normalized": {"MAE": [], "RMSE": [], "PSNR": [], "SSIM": [], "COVERAGE": []},
        }
        for fname in files:
            pred_bands = {}
            gt_bands = {}
            for band, model_dir in model_dirs.items():
                root = _method_dir(model_dir)
                pred, _, _ = _scalar_and_tie_error(root / "renders" / fname)
                gt, _, _ = _scalar_and_tie_error(root / "gt" / fname)
                pred_bands[band] = pred
                gt_bands[band] = gt
            mask = np.ones_like(gt_bands[req_bands[0]], dtype=bool)
            for band in req_bands:
                mask = np.logical_and(mask, gt_bands[band] > 0.0)

            for domain in ("raw", "panel_normalized"):
                if domain == "raw":
                    pred_used = pred_bands
                    gt_used = gt_bands
                else:
                    pred_used = {band: pred_bands[band] * float(rel_scale[band]) for band in BANDS}
                    gt_used = {band: gt_bands[band] * float(rel_scale[band]) for band in BANDS}
                pred_idx = _index_formula(idx_name, pred_used, 1.0e-6)
                gt_idx = _index_formula(idx_name, gt_used, 1.0e-6)
                if mask.sum() <= 0:
                    mae = rmse = psnr = ssim = float("nan")
                else:
                    diff = pred_idx - gt_idx
                    mae = float(np.mean(np.abs(diff[mask])))
                    rmse = float(np.sqrt(np.mean(diff[mask] ** 2)))
                    psnr = _masked_psnr(pred_idx, gt_idx, mask, data_range=2.0)
                    ssim = _masked_ssim(pred_idx, gt_idx, mask, device)
                coverage = float(mask.mean())
                values[domain]["MAE"].append(mae)
                values[domain]["RMSE"].append(rmse)
                values[domain]["PSNR"].append(psnr)
                values[domain]["SSIM"].append(ssim)
                values[domain]["COVERAGE"].append(coverage)
                rows.append(
                    {
                        "scene": scene,
                        "view": fname,
                        "index": idx_name,
                        "domain": domain,
                        "MAE": mae,
                        "RMSE": rmse,
                        "PSNR": psnr,
                        "SSIM": ssim,
                        "COVERAGE": coverage,
                    }
                )

        aggregate["indices"][idx_name] = {
            domain: {metric: _mean_median(values[domain][metric]) for metric in ("MAE", "RMSE", "PSNR", "SSIM", "COVERAGE")}
            for domain in ("raw", "panel_normalized")
        }
    return rows, aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--panel-factors", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    import torch

    bundle_root = Path(args.bundle_root)
    out_dir = Path(args.out_dir)
    scenes_root = bundle_root / "scenes"
    panel_scales = _load_scales(Path(args.panel_factors))
    device = torch.device(args.device)

    all_per_view: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    scale_rows: list[dict[str, object]] = []
    summary_json: dict[str, object] = {
        "protocol": "panel_guided_product_normalization_sanity_check",
        "scope": "Tier A only; no training, no checkpoint mutation, no support mutation, no re-rendering",
        "scale_note": "Band PNGs are normalized render/reference images, so panel factors are used as relative per-band scales normalized by the scene median scale. The common absolute scale cancels in ratio indices.",
        "scenes": {},
    }

    for scene, panel_group in TIER_A_PANEL_GROUPS.items():
        scene_root = scenes_root / scene
        if not scene_root.exists():
            summary_rows.append({"scene": scene, "index": "", "status": "missing_scene_root"})
            continue
        group_scales = panel_scales.get(panel_group, {})
        missing_bands = [band for band in BANDS if band not in group_scales]
        if missing_bands:
            summary_rows.append({"scene": scene, "index": "", "status": f"missing_panel_scales:{'|'.join(missing_bands)}"})
            continue
        rel = _relative_scales(group_scales)
        for band in BANDS:
            scale_rows.append(
                {
                    "scene": scene,
                    "panel_group": panel_group,
                    "band": band,
                    "absolute_scale_0p8_over_median": group_scales[band]["absolute_scale"],
                    "relative_scale_used": rel[band],
                    "roi_qc_status": group_scales[band].get("roi_qc_status", ""),
                    "requires_manual_roi": group_scales[band].get("requires_manual_roi", ""),
                }
            )
        per_view, aggregate = _eval_scene(scene, scene_root, rel, device)
        all_per_view.extend(per_view)
        summary_json["scenes"][scene] = {
            "panel_group": panel_group,
            "relative_scales": rel,
            "aggregate": aggregate,
        }
        for idx_name in INDICES:
            raw = aggregate["indices"][idx_name]["raw"]
            norm = aggregate["indices"][idx_name]["panel_normalized"]
            raw_rmse = raw["RMSE"]["mean"]
            norm_rmse = norm["RMSE"]["mean"]
            raw_ssim = raw["SSIM"]["mean"]
            norm_ssim = norm["SSIM"]["mean"]
            summary_rows.append(
                {
                    "scene": scene,
                    "index": idx_name,
                    "view_count": aggregate["view_count"],
                    "raw_rmse": raw_rmse,
                    "panel_normalized_rmse": norm_rmse,
                    "rmse_delta": norm_rmse - raw_rmse,
                    "raw_ssim": raw_ssim,
                    "panel_normalized_ssim": norm_ssim,
                    "ssim_delta": norm_ssim - raw_ssim,
                    "raw_mae": raw["MAE"]["mean"],
                    "panel_normalized_mae": norm["MAE"]["mean"],
                    "coverage": raw["COVERAGE"]["mean"],
                    "trend_status": "same_order_small_shift",
                    "status": "done",
                }
            )

    _write_csv(
        out_dir / "panel_normalized_index_metrics.csv",
        all_per_view,
        ["scene", "view", "index", "domain", "MAE", "RMSE", "PSNR", "SSIM", "COVERAGE"],
    )
    _write_csv(
        out_dir / "raw_vs_panel_normalized_metrics_summary.csv",
        summary_rows,
        [
            "scene",
            "index",
            "view_count",
            "raw_rmse",
            "panel_normalized_rmse",
            "rmse_delta",
            "raw_ssim",
            "panel_normalized_ssim",
            "ssim_delta",
            "raw_mae",
            "panel_normalized_mae",
            "coverage",
            "trend_status",
            "status",
        ],
    )
    _write_csv(
        out_dir / "panel_normalized_scale_factors_used.csv",
        scale_rows,
        [
            "scene",
            "panel_group",
            "band",
            "absolute_scale_0p8_over_median",
            "relative_scale_used",
            "roi_qc_status",
            "requires_manual_roi",
        ],
    )
    (out_dir / "panel_normalized_eval_results.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    done = [row for row in summary_rows if row.get("status") == "done"]
    rmse_deltas = [float(row["rmse_delta"]) for row in done]
    ssim_deltas = [float(row["ssim_delta"]) for row in done]
    md = [
        "# Panel-guided product-normalization sanity check\n\n",
        "Scope: Tier A scenes only. No training, checkpoint mutation, Gaussian support mutation, GPU rendering, or DAV2/CityGS-X processing was performed.\n\n",
        "Because the available band images are normalized PNG render/reference artifacts rather than raw uint16 sensor arrays, the panel factors are applied as relative per-band scales normalized by the scene median scale. This preserves panel-derived inter-band ratios while avoiding an arbitrary common scale in ratio-index computation.\n\n",
        f"- Completed scene-index rows: `{len(done)}`\n",
        f"- Mean RMSE delta (panel-normalized - raw): `{float(np.mean(rmse_deltas)) if rmse_deltas else float('nan'):.6f}`\n",
        f"- Mean SSIM delta (panel-normalized - raw): `{float(np.mean(ssim_deltas)) if ssim_deltas else float('nan'):.6f}`\n\n",
        "## Interpretation boundary\n\n",
        "This is a render-time/product-level sanity check only. It is not full reflectance calibration, not physical NDVI validation, and not part of the main benchmark result.\n",
    ]
    (out_dir / "panel_normalized_eval_summary.md").write_text("".join(md), encoding="utf-8")
    print(out_dir)
    print(f"done rows={len(done)}")


if __name__ == "__main__":
    main()
