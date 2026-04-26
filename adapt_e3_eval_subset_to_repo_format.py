from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

from PIL import Image

from build_mmsplat_raw_json_split import _collect_complete_groups


BANDS = ("G", "R", "RE", "NIR")
CHANNEL_TO_BAND = {
    "MS_G": "G",
    "MS_R": "R",
    "MS_RE": "RE",
    "MS_NIR": "NIR",
}
PROCESSED_D_RE = re.compile(r"^images/D/D_(?P<idx>\d{5})\.JPG$", re.IGNORECASE)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _entry_value(item: object) -> str:
    if isinstance(item, dict):
        value = item.get("official_path") or item.get("images2_path") or item.get("path") or item.get("file_path")
    else:
        value = item
    return str(value).replace("\\", "/").strip()


def _band_from_entry(rel: str) -> str | None:
    parts = Path(rel).parts
    if len(parts) < 2:
        return None
    return CHANNEL_TO_BAND.get(str(parts[1]))


def _retained_d_rel_paths(split_json: Path) -> List[str]:
    split = _load_json(split_json)
    eval_items = split.get("eval", split.get("test", []))
    if not eval_items:
        raise RuntimeError(f"Split has no eval/test entries: {split_json}")

    grouped: Dict[str, Dict[str, str]] = {}
    order: List[str] = []
    for item in eval_items:
        rel = _entry_value(item)
        d_match = PROCESSED_D_RE.match(rel)
        if d_match:
            group_key = d_match.group("idx")
        else:
            stem = Path(rel).stem
            m = re.search(r"_(\d{5})$", stem)
            if not m:
                raise RuntimeError(f"Cannot infer capture group from split entry: {rel}")
            group_key = m.group(1)
        if group_key not in grouped:
            grouped[group_key] = {}
            order.append(group_key)
        band = _band_from_entry(rel)
        if band is not None:
            grouped[group_key][band] = rel
        elif d_match:
            grouped[group_key]["D"] = rel

    retained_d: List[str] = []
    for group_key in order:
        group = grouped[group_key]
        missing = [k for k in ("D", *BANDS) if k not in group]
        if missing:
            raise RuntimeError(f"Retained split group {group_key} is not five-band complete; missing={missing}")
        retained_d.append(group["D"])
    return retained_d


def _processed_d_to_original_name(raw_root: Path, group_mode: str) -> Dict[str, str]:
    complete = _collect_complete_groups(raw_root.resolve(), group_mode=group_mode)
    out: Dict[str, str] = {}
    for i, row in enumerate(complete, start=1):
        processed = f"images/D/D_{i:05d}.JPG"
        out[processed] = str(row["d_name"])
    if not out:
        raise RuntimeError(f"No complete raw groups found under {raw_root}")
    return out


def _list_pngs(path: Path) -> List[str]:
    files = sorted([f for f in os.listdir(path) if f.lower().endswith(".png")])
    if not files:
        raise RuntimeError(f"No png files found in {path}")
    return files


def _copy_or_resize(src: Path, dst: Path, target_size: tuple[int, int] | None) -> tuple[int, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if target_size is None:
        shutil.copy2(src, dst)
        return Image.open(src).size
    img = Image.open(src).convert("RGB")
    if img.size != target_size:
        img = img.resize(target_size, Image.BILINEAR)
    img.save(dst)
    return img.size


def _target_sizes(match_root: Path | None, band: str, iteration: int, names: Iterable[str]) -> Dict[str, tuple[int, int]]:
    if match_root is None:
        return {}
    render_dir = match_root / f"Model_{band}" / "test" / f"ours_{int(iteration)}" / "renders"
    sizes = {}
    for name in names:
        path = render_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing match render for size reference: {path}")
        sizes[name] = Image.open(path).size
    return sizes


def adapt_e3_subset(
    reference_run_root: Path,
    split_json: Path,
    raw_root: Path,
    out_root: Path,
    iteration: int,
    group_mode: str,
    match_root: Path | None,
) -> dict:
    reference_run_root = reference_run_root.resolve()
    split_json = split_json.resolve()
    raw_root = raw_root.resolve()
    out_root = out_root.resolve()
    match_root = match_root.resolve() if match_root else None

    retained_d_rel = _retained_d_rel_paths(split_json)
    processed_to_original = _processed_d_to_original_name(raw_root, group_mode=group_mode)
    retained_d_names = []
    for rel in retained_d_rel:
        if rel not in processed_to_original:
            raise RuntimeError(f"Cannot map processed D path to raw D name: {rel}")
        retained_d_names.append(processed_to_original[rel])

    if out_root.exists():
        shutil.rmtree(out_root)

    audit = {
        "source_e3_root": str(reference_run_root),
        "split_json": str(split_json),
        "raw_root": str(raw_root),
        "output_root": str(out_root),
        "policy": "subset_existing_e3_test_renders_by_mms_retained_eval_groups",
        "resize_policy": "resize_to_match_root_bilinear" if match_root else "copy_native_resolution",
        "match_root": str(match_root) if match_root else "",
        "iteration": int(iteration),
        "eval_group_count": len(retained_d_names),
        "retained_processed_d_paths": retained_d_rel,
        "retained_d_names": retained_d_names,
        "bands": {},
    }

    for band in BANDS:
        src_model = reference_run_root / f"Model_{band}"
        dst_model = out_root / f"Model_{band}"
        src_ours = src_model / "test" / f"ours_{int(iteration)}"
        src_render_dir = src_ours / "renders"
        src_gt_dir = src_ours / "gt"
        src_files = _list_pngs(src_render_dir)
        src_gt_files = _list_pngs(src_gt_dir)
        if src_files != src_gt_files:
            raise RuntimeError(f"E3 render/gt file mismatch for {band}: {src_render_dir} vs {src_gt_dir}")

        cameras = json.loads((src_model / "cameras.json").read_text(encoding="utf-8"))
        if len(cameras) < len(src_files):
            raise RuntimeError(f"{src_model}/cameras.json has fewer rows than rendered test files")
        name_to_file = {}
        name_to_camera = {}
        for file_name, camera in zip(src_files, cameras):
            img_name = str(camera.get("img_name", ""))
            if not img_name:
                raise RuntimeError(f"Missing img_name in {src_model}/cameras.json")
            name_to_file[img_name] = file_name
            name_to_camera[img_name] = camera

        dst_render_dir = dst_model / "test" / f"ours_{int(iteration)}" / "renders"
        dst_gt_dir = dst_model / "test" / f"ours_{int(iteration)}" / "gt"
        dst_render_dir.mkdir(parents=True, exist_ok=True)
        dst_gt_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_model / "cfg_args", dst_model / "cfg_args")

        dst_names = [f"{i:05d}.png" for i in range(len(retained_d_names))]
        size_refs = _target_sizes(match_root, band, iteration, dst_names)

        camera_rows = []
        copied = []
        for i, d_name in enumerate(retained_d_names):
            if d_name not in name_to_file:
                raise RuntimeError(f"Retained D view not found in E3 test cameras for band={band}: {d_name}")
            src_name = name_to_file[d_name]
            dst_name = f"{i:05d}.png"
            target_size = size_refs.get(dst_name)
            render_size = _copy_or_resize(src_render_dir / src_name, dst_render_dir / dst_name, target_size)
            gt_size = _copy_or_resize(src_gt_dir / src_name, dst_gt_dir / dst_name, target_size)
            row = dict(name_to_camera[d_name])
            row["img_name"] = d_name
            camera_rows.append(row)
            copied.append(
                {
                    "retained_d_name": d_name,
                    "source_file": src_name,
                    "output_file": dst_name,
                    "render_size": list(render_size),
                    "gt_size": list(gt_size),
                }
            )

        (dst_model / "cameras.json").write_text(json.dumps(camera_rows, indent=2), encoding="utf-8")
        audit["bands"][band] = {
            "view_count": len(camera_rows),
            "source_model": str(src_model),
            "output_model": str(dst_model),
            "copied_preview": copied[:8],
        }

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "subset_shim_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a repo-native E3 subset run root matching an MMSplat retained json-list eval subset."
    )
    ap.add_argument("--reference_run_root", required=True, help="E3 run root containing Model_G/R/RE/NIR.")
    ap.add_argument("--split_json", required=True, help="Retained/sanitized MMSplat json-list split.")
    ap.add_argument("--raw_root", required=True, help="Original flat raw root used to map processed D ids to raw D names.")
    ap.add_argument("--out_root", required=True, help="Synthetic E3 subset output root.")
    ap.add_argument("--iteration", type=int, default=60000)
    ap.add_argument("--group_mode", choices=["prefix_frame", "frame_only"], default="frame_only")
    ap.add_argument(
        "--match_root",
        default="",
        help="Optional repo-native run root whose render dimensions should be matched by bilinear resizing.",
    )
    args = ap.parse_args()

    audit = adapt_e3_subset(
        reference_run_root=Path(args.reference_run_root),
        split_json=Path(args.split_json),
        raw_root=Path(args.raw_root),
        out_root=Path(args.out_root),
        iteration=int(args.iteration),
        group_mode=args.group_mode,
        match_root=Path(args.match_root) if args.match_root else None,
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
