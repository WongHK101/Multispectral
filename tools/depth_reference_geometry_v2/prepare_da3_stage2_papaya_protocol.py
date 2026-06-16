#!/usr/bin/env python3
"""Prepare the leakage-safe Papaya DA3 Stage-2 protocol manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FIG3_FRAME_IDS = ("0001", "0025", "0049", "0065", "0113")


def sha256_lines(lines: list[str]) -> str:
    payload = "".join(f"{line}\n" for line in lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def make_window_starts(count: int, window_size: int, stride: int) -> list[int]:
    if count < window_size:
        return [0]
    starts = list(range(0, count - window_size + 1, stride))
    final_start = count - window_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--window-overlap", type=int, default=8)
    args = parser.parse_args()

    split = json.loads(args.split_json.read_text(encoding="utf-8"))
    train_names = [entry["image_name"] for entry in split["train"]]
    test_names = [entry["image_name"] for entry in split["test"]]
    train_ids = {entry["frame_id"] for entry in split["train"]}
    test_ids = {entry["frame_id"] for entry in split["test"]}

    if set(train_names) & set(test_names):
        raise RuntimeError("Train/test image lists overlap")
    if not set(FIG3_FRAME_IDS).issubset(test_ids):
        raise RuntimeError("One or more Fig. 3 views are not held-out test views")
    if set(FIG3_FRAME_IDS) & train_ids:
        raise RuntimeError("Fig. 3 views leaked into the training list")

    stride = args.window_size - args.window_overlap
    if stride <= 0:
        raise ValueError("window-overlap must be smaller than window-size")
    starts = make_window_starts(len(train_names), args.window_size, stride)
    windows = []
    covered: set[str] = set()
    for window_index, start in enumerate(starts):
        names = train_names[start : start + args.window_size]
        covered.update(names)
        windows.append(
            {
                "window_id": f"W{window_index:02d}",
                "start_index": start,
                "end_index_exclusive": start + len(names),
                "frame_count": len(names),
                "first_image": names[0],
                "last_image": names[-1],
                "image_names": names,
            }
        )

    missing_coverage = sorted(set(train_names) - covered)
    if missing_coverage:
        raise RuntimeError(f"Training window coverage is incomplete: {missing_coverage}")

    test_by_id = {entry["frame_id"]: entry["image_name"] for entry in split["test"]}
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "papaya_stage2_training_frames.txt").write_text(
        "".join(f"{name}\n" for name in train_names),
        encoding="utf-8",
    )
    (output_dir / "papaya_stage2_training_windows.json").write_text(
        json.dumps(
            {
                "schema": "da3_stage2_training_windows_v1",
                "scene": split["scene_id"],
                "split_name": split["split_name"],
                "training_count": len(train_names),
                "window_size": args.window_size,
                "window_overlap": args.window_overlap,
                "window_stride": stride,
                "window_count": len(windows),
                "training_image_sha256": sha256_lines(train_names),
                "windows": windows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "papaya_stage2_split_leakage_audit.json").write_text(
        json.dumps(
            {
                "schema": "da3_stage2_split_leakage_audit_v1",
                "scene": split["scene_id"],
                "train_count": len(train_names),
                "test_count": len(test_names),
                "train_test_overlap_count": len(set(train_names) & set(test_names)),
                "all_training_frames_covered_by_windows": True,
                "fig3_held_out_views": [
                    {"frame_id": frame_id, "image_name": test_by_id[frame_id]}
                    for frame_id in FIG3_FRAME_IDS
                ],
                "fig3_views_in_training_count": len(set(FIG3_FRAME_IDS) & train_ids),
                "policy": {
                    "da3_inference_inputs": "training RGB views only",
                    "threshold_selection": "training-side evidence only",
                    "roi_selection": "training-side evidence only",
                    "held_out_use": "evaluation camera parameters and final evaluation only",
                },
                "status": "pass",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
