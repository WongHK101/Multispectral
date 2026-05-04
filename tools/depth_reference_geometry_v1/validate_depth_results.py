from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
        f.write("\n")


def iter_metrics(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    yield from root.rglob("metrics_summary.json")


def view_map(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(v["image_name"]): v for v in manifest.get("views", [])}


def max_pose_delta(a: Any, b: Any) -> float:
    try:
        if len(a) != len(b):
            return float("inf")
        out = 0.0
        for row_a, row_b in zip(a, b):
            if len(row_a) != len(row_b):
                return float("inf")
            for va, vb in zip(row_a, row_b):
                out = max(out, abs(float(va) - float(vb)))
        return out
    except Exception:
        return float("inf")


def compare_manifests(reference_manifest: Path, model_manifest: Path) -> Dict[str, Any]:
    ref = load_json(reference_manifest)
    model = load_json(model_manifest)
    ref_views = view_map(ref)
    model_views = view_map(model)
    common = sorted(set(ref_views) & set(model_views))
    missing = sorted(set(ref_views) - set(model_views))
    extra = sorted(set(model_views) - set(ref_views))

    max_intrinsic_delta = 0.0
    max_pose_delta_value = 0.0
    max_resolution_delta = 0
    for image_name in common:
        rv = ref_views[image_name]
        mv = model_views[image_name]
        max_resolution_delta = max(
            max_resolution_delta,
            abs(int(rv["width"]) - int(mv["width"])),
            abs(int(rv["height"]) - int(mv["height"])),
        )
        for key in ("fx", "fy", "cx", "cy"):
            max_intrinsic_delta = max(max_intrinsic_delta, abs(float(rv[key]) - float(mv[key])))
        max_pose_delta_value = max(
            max_pose_delta_value,
            max_pose_delta(rv.get("camera_to_world"), mv.get("camera_to_world")),
        )

    return {
        "reference_view_count": len(ref_views),
        "model_view_count": len(model_views),
        "common_view_count": len(common),
        "missing_view_count": len(missing),
        "extra_view_count": len(extra),
        "missing_view_sample": missing[:8],
        "extra_view_sample": extra[:8],
        "max_camera_intrinsic_abs_delta": max_intrinsic_delta,
        "max_camera_pose_abs_delta": max_pose_delta_value,
        "max_resolution_delta_px": max_resolution_delta,
    }


def classify_metrics(
    metrics_path: Path,
    intrinsic_tolerance: float,
    pose_tolerance: float,
    recheck_manifests: bool,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "metrics_path": str(metrics_path),
        "status": "unknown",
        "reason": "",
        "scene_name": "",
        "method_name": "",
        "reference_manifest": "",
        "model_manifest": "",
        "adapter_manifest": "",
        "has_camera_audit": False,
        "max_camera_intrinsic_abs_delta": "",
        "max_camera_pose_abs_delta": "",
        "max_resolution_delta_px": "",
    }
    try:
        metrics = load_json(metrics_path)
    except Exception as exc:
        row["status"] = "error"
        row["reason"] = f"cannot_read_metrics: {exc}"
        return row

    if metrics.get("protocol_name") != "reference-depth-based-geometric-evaluation-v1":
        row["status"] = "skipped"
        row["reason"] = "not_depth_reference_protocol"
        return row

    row["scene_name"] = str(metrics.get("scene_name", ""))
    row["method_name"] = str(metrics.get("method_name", ""))
    row["reference_manifest"] = str(metrics.get("reference_manifest", ""))
    row["model_manifest"] = str(metrics.get("model_manifest", ""))
    row["adapter_manifest"] = str(metrics.get("adapter_manifest", ""))

    opts = metrics.get("evaluation_options") or {}
    intrinsic = opts.get("max_camera_intrinsic_abs_delta")
    pose = opts.get("max_camera_pose_abs_delta")
    if intrinsic is not None and pose is not None:
        row["has_camera_audit"] = True
        row["max_camera_intrinsic_abs_delta"] = float(intrinsic)
        row["max_camera_pose_abs_delta"] = float(pose)
        if float(intrinsic) > intrinsic_tolerance:
            row["status"] = "fail"
            row["reason"] = "intrinsic_delta_exceeds_tolerance"
        elif float(pose) > pose_tolerance:
            row["status"] = "fail"
            row["reason"] = "pose_delta_exceeds_tolerance"
        else:
            row["status"] = "pass"
            row["reason"] = "camera_audit_present_and_within_tolerance"
        return row

    if not recheck_manifests:
        row["status"] = "needs_recheck"
        row["reason"] = "missing_camera_audit_fields"
        return row

    ref_path = Path(row["reference_manifest"])
    model_path = Path(row["model_manifest"])
    if not ref_path.exists() or not model_path.exists():
        row["status"] = "needs_recheck"
        row["reason"] = "missing_camera_audit_fields_and_manifest_not_found"
        return row

    try:
        cmp = compare_manifests(ref_path, model_path)
    except Exception as exc:
        row["status"] = "error"
        row["reason"] = f"manifest_compare_failed: {exc}"
        return row

    row["max_camera_intrinsic_abs_delta"] = cmp["max_camera_intrinsic_abs_delta"]
    row["max_camera_pose_abs_delta"] = cmp["max_camera_pose_abs_delta"]
    row["max_resolution_delta_px"] = cmp["max_resolution_delta_px"]
    row.update({f"manifest_{k}": v for k, v in cmp.items() if k.endswith("_count")})

    if cmp["missing_view_count"] or cmp["extra_view_count"]:
        row["status"] = "fail"
        row["reason"] = "view_set_mismatch"
    elif cmp["max_resolution_delta_px"] != 0:
        row["status"] = "fail"
        row["reason"] = "resolution_mismatch"
    elif cmp["max_camera_intrinsic_abs_delta"] > intrinsic_tolerance:
        row["status"] = "fail"
        row["reason"] = "intrinsic_delta_exceeds_tolerance"
    elif cmp["max_camera_pose_abs_delta"] > pose_tolerance:
        row["status"] = "fail"
        row["reason"] = "pose_delta_exceeds_tolerance"
    else:
        row["status"] = "pass_rechecked"
        row["reason"] = "missing_camera_audit_but_manifest_recheck_passed"
    return row


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate reference-depth metrics summaries before paper aggregation. "
            "The validator checks that camera audit fields exist or, optionally, "
            "recomputes reference/model manifest alignment."
        )
    )
    parser.add_argument("--root", required=True, help="Metrics summary file or directory to scan")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--camera_intrinsic_tolerance", type=float, default=1e-4)
    parser.add_argument("--camera_pose_tolerance", type=float, default=1e-5)
    parser.add_argument("--recheck_manifests", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve()
    rows = [
        classify_metrics(
            metrics_path=p,
            intrinsic_tolerance=float(args.camera_intrinsic_tolerance),
            pose_tolerance=float(args.camera_pose_tolerance),
            recheck_manifests=bool(args.recheck_manifests),
        )
        for p in sorted(iter_metrics(root))
    ]
    rows = [r for r in rows if r["status"] != "skipped"]

    status_counts: Dict[str, int] = {}
    reason_counts: Dict[str, int] = {}
    for row in rows:
        status_counts[str(row["status"])] = status_counts.get(str(row["status"]), 0) + 1
        reason_counts[str(row["reason"])] = reason_counts.get(str(row["reason"]), 0) + 1

    summary = {
        "root": str(root),
        "num_depth_metrics": len(rows),
        "status_counts": status_counts,
        "reason_counts": reason_counts,
        "camera_intrinsic_tolerance": float(args.camera_intrinsic_tolerance),
        "camera_pose_tolerance": float(args.camera_pose_tolerance),
        "recheck_manifests": bool(args.recheck_manifests),
        "rows": rows,
    }
    save_json(out_dir / "depth_result_validation_summary.json", summary)
    write_csv(out_dir / "depth_result_validation_rows.csv", rows)
    print(f"DEPTH_RESULT_VALIDATION {out_dir / 'depth_result_validation_summary.json'}")
    print(json.dumps({"num_depth_metrics": len(rows), "status_counts": status_counts}, indent=2))


if __name__ == "__main__":
    main()
