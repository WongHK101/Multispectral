#!/usr/bin/env python3
"""Collect UAV-MultiSpec3D benchmark16 UMGS-I summary on AutoDL.

This reporting script reads only small metric/provenance files from existing
run roots. It does not train, render, evaluate, or modify model outputs.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_DATA_ROOT = Path(
    "/root/autodl-tmp/datasets/UAV-MultiSpec3D/"
    "UAV-MultiSpec3D_Benchmark_16scenes_20260602"
)

RUN_MAP = {
    "banana_01_20260530_1616": "/root/autodl-tmp/runs/uav_multispec3d_active17_registered100_umgs_i_20260602_120858",
    "banana_02_20260530_1641": "/root/autodl-tmp/runs/uav_multispec3d_active17_registered100_umgs_i_20260602_120858",
    "cassava_01_20260526_1603": "/root/autodl-tmp/runs/uav_multispec3d_release15_clean8_umgs_i_20260529_030125",
    "chunya_01_20260526_1021": "/root/autodl-tmp/runs/uav_multispec3d_release15_clean8_umgs_i_20260529_030125",
    "eucalyptus_01_20260526_1053_pruned": "/root/autodl-tmp/runs/uav_multispec3d_benchmark16_missing_eucalyptus_umgs_i_20260604_125304",
    "eucalyptus_02_20260526_1108_pruned": "/root/autodl-tmp/runs/uav_multispec3d_benchmark16_missing_eucalyptus_umgs_i_20260604_125304",
    "maize_01_20260526_0959": "/root/autodl-tmp/runs/uav_multispec3d_active17_registered100_umgs_i_20260602_120858",
    "maize_02_20260526_1658": "/root/autodl-tmp/runs/uav_multispec3d_active17_registered100_umgs_i_20260602_120858",
    "maize_03_20260527_1103": "/root/autodl-tmp/runs/uav_multispec3d_active17_registered100_umgs_i_20260602_120858",
    "papaya_01_20251217": "/root/autodl-tmp/runs/uav_multispec3d_benchmark16_umgs_i_consolidated_20260604",
    "road_01_20260602_1648_40m": "/root/autodl-tmp/runs/uav_multispec3d_road40_umgs_i_20260603_231629",
    "wogan_mandarin_01_20260525_1533": "/root/autodl-tmp/runs/uav_multispec3d_wogan_fix_after_current_batch_20260602_180641",
    "wogan_mandarin_02_20260527_0955": "/root/autodl-tmp/runs/uav_multispec3d_release15_clean8_umgs_i_20260529_030125",
    "wogan_mandarin_03_20260528_1441": "/root/autodl-tmp/runs/uav_multispec3d_active17_registered100_umgs_i_20260602_120858",
    "wogan_mandarin_04_20260528_1558": "/root/autodl-tmp/runs/uav_multispec3d_wogan_fix_after_current_batch_20260602_180641",
    "wogan_mandarin_05_20260528_1621": "/root/autodl-tmp/runs/uav_multispec3d_active17_registered100_umgs_i_20260602_120858",
}

SEQ_PATTERNS = [
    re.compile(r"_(\d{4})_D\.JPG$", re.IGNORECASE),
    re.compile(r"_(\d{4})_MS_(G|R|RE|NIR)\.TIF$", re.IGNORECASE),
]


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def image_count_from_images_txt(path: Path) -> int | None:
    if not path.exists():
        return None
    lines = [
        line
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    return len(lines) // 2


def names_sha_from_images_txt(path: Path) -> str | None:
    if not path.exists():
        return None
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    names = []
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        if len(parts) >= 10:
            names.append(parts[9])
    return hashlib.sha256("\n".join(sorted(names)).encode("utf-8")).hexdigest()


def raw_counts(scene_dir: Path) -> dict[str, Any]:
    jpgs = sorted(p for p in scene_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jpg")
    tifs = sorted(p for p in scene_dir.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"})
    mrks = sorted(p for p in scene_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mrk")
    rgb_seq = set()
    band_seq: dict[str, set[str]] = {b: set() for b in ["G", "R", "RE", "NIR"]}
    for p in jpgs:
        m = SEQ_PATTERNS[0].search(p.name)
        if m:
            rgb_seq.add(m.group(1))
    for p in tifs:
        m = SEQ_PATTERNS[1].search(p.name)
        if m:
            band_seq[m.group(2).upper()].add(m.group(1))
    complete = rgb_seq.copy()
    for seqs in band_seq.values():
        complete &= seqs
    return {
        "raw_rgb_jpg_count": len(jpgs),
        "raw_ms_tif_count": len(tifs),
        "raw_mrk_count": len(mrks),
        "complete_frame_groups": len(complete),
    }


def metric_value(results: dict[str, Any] | None, key: str) -> Any:
    if not results:
        return None
    # results.json is usually {"ours_30000": {...}} or {"ours_60000": {...}}.
    for _, values in sorted(results.items()):
        if isinstance(values, dict) and key in values:
            return values.get(key)
    return None


def index_metric(idx: dict[str, Any] | None, index_name: str, metric: str) -> Any:
    if not idx:
        return None
    values = idx.get("metrics_by_index", {}).get(index_name, {}).get(metric)
    if isinstance(values, dict):
        return values.get("mean")
    return None


def support_status(support: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "support_audit_exists": support is not None,
        "support_zero_drift": None,
        "support_same_shape": None,
        "support_num_gaussians_G": None,
        "support_max_delta": None,
    }
    if not support:
        return out
    bands = support.get("bands", {})
    g = bands.get("G", {})
    out["support_num_gaussians_G"] = g.get("num_gaussians")
    deltas = []
    shapes = []
    for band in ["R", "RE", "NIR"]:
        b = bands.get(band, {})
        if "same_shape_as_G" in b:
            shapes.append(bool(b.get("same_shape_as_G")))
        if "max_support_delta_vs_G" in b:
            deltas.append(float(b.get("max_support_delta_vs_G")))
    out["support_same_shape"] = all(shapes) if shapes else None
    out["support_max_delta"] = max(deltas) if deltas else None
    out["support_zero_drift"] = (out["support_same_shape"] is True and out["support_max_delta"] == 0.0)
    return out


def split_count(split: dict[str, Any] | None, key: str, list_key: str) -> int | None:
    if split is None:
        return None
    if split.get(key) is not None:
        return int(split[key])
    values = split.get(list_key)
    if isinstance(values, list):
        return len(values)
    return None


def collect(data_root: Path, out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scenes_root = data_root / "scenes"
    scenes = sorted(p.name for p in scenes_root.iterdir() if p.is_dir())
    rows: list[dict[str, Any]] = []
    for scene in scenes:
        scene_dir = scenes_root / scene
        root = Path(RUN_MAP.get(scene, ""))
        run_scene = root / scene if root else Path("")
        idx = read_json(run_scene / "out" / "index_metrics_summary.json")
        support = read_json(run_scene / "summary" / "support_audit.json")
        run_split = read_json(run_scene / "summary" / "registered_llffhold8_split_v1.json")
        metadata_split = read_json(
            data_root / "metadata" / "splits" / "llffhold8_complete_frames" / scene / "split_v1.json"
        )
        if run_split is not None:
            split = run_split
            split_source = "run_registered_split"
        elif metadata_split is not None:
            split = metadata_split
            split_source = "dataset_metadata_split"
        else:
            split = None
            split_source = None
        alias = read_json(run_scene / "summary" / "papaya_alias_audit.json")
        sparse = run_scene / "prepared" / "RGB" / "sparse" / "0" / "images.txt"

        row: dict[str, Any] = {
            "scene_id": scene,
            "dataset_root": str(data_root),
            "run_root": str(root) if root else "",
            "run_scene_dir": str(run_scene) if root else "",
            "done_train_eval": (run_scene / "DONE_TRAIN_EVAL").exists(),
            "done_train_eval_alias": (run_scene / "DONE_TRAIN_EVAL_ALIAS").exists(),
            "has_index_metrics": idx is not None,
            "has_support_audit": support is not None,
            "has_rgb_results": (run_scene / "out" / "Model_RGB" / "results.json").exists(),
            "sparse_images_txt_exists": sparse.exists(),
            "registered_image_count": image_count_from_images_txt(sparse),
            "registered_names_sha256": names_sha_from_images_txt(sparse),
            "split_total_count": split_count(split, "total_count", "images"),
            "split_train_count": split_count(split, "train_count", "train"),
            "split_test_count": split_count(split, "test_count", "test")
            if split is not None
            else idx.get("view_count") if idx else None,
            "split_name": split.get("split_name") if split is not None else None,
            "split_source": split_source,
            "index_view_count": idx.get("view_count") if idx else None,
            "index_mask_mode": idx.get("mask_mode") if idx else None,
            "papaya_alias_source_run": alias.get("source_run") if alias else None,
        }
        row.update(raw_counts(scene_dir))

        rgb = read_json(run_scene / "out" / "Model_RGB" / "results.json")
        row.update(
            {
                "rgb_psnr": metric_value(rgb, "PSNR"),
                "rgb_ssim": metric_value(rgb, "SSIM"),
                "rgb_lpips": metric_value(rgb, "LPIPS"),
            }
        )
        for band in ["G", "R", "RE", "NIR"]:
            res = read_json(run_scene / "out" / f"Model_{band}" / "results.json")
            row[f"{band.lower()}_psnr"] = metric_value(res, "PSNR")
            row[f"{band.lower()}_ssim"] = metric_value(res, "SSIM")
            row[f"{band.lower()}_lpips"] = metric_value(res, "LPIPS")
        for index_name in ["NDVI", "GNDVI", "NDRE"]:
            prefix = index_name.lower()
            row[f"{prefix}_rmse"] = index_metric(idx, index_name, "RMSE")
            row[f"{prefix}_ssim"] = index_metric(idx, index_name, "SSIM")
            row[f"{prefix}_psnr"] = index_metric(idx, index_name, "PSNR")
            row[f"{prefix}_mae"] = index_metric(idx, index_name, "MAE")
            cov = idx.get("coverage", {}).get(index_name, {}) if idx else {}
            row[f"{prefix}_coverage_mean"] = cov.get("mean")
        row.update(support_status(support))
        rows.append(row)

    summary = {
        "schema": "uav_multispec3d_benchmark16_umgs_i_summary_v1",
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "data_root": str(data_root),
        "scene_count": len(scenes),
        "scene_ids": scenes,
        "complete_train_eval_or_alias_count": sum(
            bool(r["done_train_eval"] or r["done_train_eval_alias"]) for r in rows
        ),
        "index_metrics_count": sum(bool(r["has_index_metrics"]) for r in rows),
        "support_audit_count": sum(bool(r["has_support_audit"]) for r in rows),
        "all_support_zero_drift": all(r["support_zero_drift"] is True for r in rows),
        "run_map": RUN_MAP,
        "notes": [
            "Papaya uses a non-destructive consolidated alias to the historical manuscript raw_self run; no retraining was performed for papaya.",
            "This report reads existing small metrics/provenance files only and does not touch model checkpoints or render outputs.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark16_summary.json").write_text(
        json.dumps({"summary": summary, "scenes": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return rows, summary


def write_csv(rows: list[dict[str, Any]], out_dir: Path) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with (out_dir / "benchmark16_scene_metrics_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], summary: dict[str, Any], out_dir: Path) -> None:
    lines = [
        "# UAV-MultiSpec3D Benchmark16 UMGS-I Summary",
        "",
        f"- Created at: `{summary['created_at']}`",
        f"- Data root: `{summary['data_root']}`",
        f"- Scene count: `{summary['scene_count']}`",
        f"- Train/eval or alias complete: `{summary['complete_train_eval_or_alias_count']}/{summary['scene_count']}`",
        f"- Index metrics available: `{summary['index_metrics_count']}/{summary['scene_count']}`",
        f"- Support audits available: `{summary['support_audit_count']}/{summary['scene_count']}`",
        f"- All support audits zero-drift: `{summary['all_support_zero_drift']}`",
        "",
        "## Scene Metrics",
        "",
        "| Scene | Train/Test | RGB PSNR/SSIM | NDVI RMSE/SSIM | GNDVI RMSE/SSIM | NDRE RMSE/SSIM | Support | Run root |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        train = r.get("split_train_count")
        test = r.get("split_test_count")
        train_test = f"{train}/{test}" if train is not None or test is not None else ""
        def pair(a: str, b: str, n1: int = 3, n2: int = 3) -> str:
            va, vb = r.get(a), r.get(b)
            if va is None or vb is None:
                return ""
            return f"{float(va):.{n1}f}/{float(vb):.{n2}f}"
        support = "pass" if r.get("support_zero_drift") is True else "check"
        lines.append(
            "| {scene_id} | {train_test} | {rgb} | {ndvi} | {gndvi} | {ndre} | {support} | `{run}` |".format(
                scene_id=r["scene_id"],
                train_test=train_test,
                rgb=pair("rgb_psnr", "rgb_ssim", 2, 3),
                ndvi=pair("ndvi_rmse", "ndvi_ssim", 4, 3),
                gndvi=pair("gndvi_rmse", "gndvi_ssim", 4, 3),
                ndre=pair("ndre_rmse", "ndre_ssim", 4, 3),
                support=support,
                run=Path(r["run_root"]).name,
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Metrics are existing UMGS-I outputs; this collection step did not run training, rendering, or evaluation.",
            "- Papaya is represented through a consolidated alias to the historical manuscript `raw_self` run because the image list and LLFF hold-8 split match the current benchmark papaya scene.",
            "- Support status is based on exported support-invariant checks: R/RE/NIR must have the same shape as G and zero max support delta vs G.",
        ]
    )
    (out_dir / "benchmark16_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.out_dir is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = Path(f"/root/autodl-tmp/runs/uav_multispec3d_benchmark16_summary_{stamp}")
    rows, summary = collect(args.data_root, args.out_dir)
    write_csv(rows, args.out_dir)
    write_markdown(rows, summary, args.out_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"OUT_DIR={args.out_dir}")


if __name__ == "__main__":
    main()
