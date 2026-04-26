from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image


CHANNEL_TO_BAND = {
    "MS_G": "G",
    "MS_R": "R",
    "MS_RE": "RE",
    "MS_NIR": "NIR",
}


def _load_loose_json(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r",(?=\s*[\]}])", "", text)
    return json.loads(text)


def _path_without_ext(value: str) -> str:
    p = str(value).replace("\\", "/").strip()
    suffix = Path(p).suffix
    if suffix:
        return p[: -len(suffix)]
    return p


def _has_matching_file(path_without_ext: str, paths_with_ext: Iterable[str]) -> bool:
    for item in paths_with_ext:
        if _path_without_ext(item) == path_without_ext:
            return True
    return False


def _downscaled_rel_path(frame_path: str, downscale_factor: int) -> str:
    rel = str(frame_path).replace("\\", "/")
    if downscale_factor <= 0:
        return rel
    power = 2 ** int(downscale_factor)
    if rel.startswith("images/"):
        return rel.replace("images/", f"images_{power}/", 1)
    return rel


def _safe_unlink(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        _safe_unlink(dst)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        raise ValueError(f"Unsupported link mode: {mode}")


def _parse_source_path_from_cfg(cfg_path: Path) -> Path:
    text = cfg_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"source_path\s*=\s*['\"]([^'\"]+)['\"]", text)
    if not m:
        raise RuntimeError(f"Cannot parse source_path from {cfg_path}")
    return Path(m.group(1)).resolve()


def _write_minimal_cfg_args(dst: Path, source_path: Path, use_validity_mask: bool) -> None:
    dst.write_text(
        "Namespace("
        f"source_path='{source_path.as_posix()}', "
        f"use_validity_mask={'True' if use_validity_mask else 'False'}"
        ")\n",
        encoding="utf-8",
    )


def _channel_from_split_item(item: object) -> str:
    if isinstance(item, dict):
        value = item.get("official_path") or item.get("images2_path") or item.get("path") or item.get("file_path")
    else:
        value = item
    rel = str(value).replace("\\", "/").strip()
    parts = Path(rel).parts
    if len(parts) < 2:
        raise RuntimeError(f"Cannot infer channel from split entry: {item!r}")
    return parts[1]


def _eval_entries_by_band(split_json: Path) -> Dict[str, List[str]]:
    split = _load_loose_json(split_json)
    eval_items = split.get("eval", split.get("test", []))
    if not eval_items:
        raise RuntimeError(f"Split has no eval/test entries: {split_json}")
    out: Dict[str, List[str]] = {band: [] for band in CHANNEL_TO_BAND.values()}
    for item in eval_items:
        if isinstance(item, dict):
            value = item.get("official_path") or item.get("images2_path") or item.get("path") or item.get("file_path")
        else:
            value = item
        rel = str(value).replace("\\", "/").strip()
        channel = _channel_from_split_item(item)
        if channel in CHANNEL_TO_BAND:
            out[CHANNEL_TO_BAND[channel]].append(rel)
    return out


def _band_entries(
    eval_items_by_band: Dict[str, List[str]],
    renders_root: Path,
) -> Dict[str, List[Tuple[str, Path, str]]]:
    out: Dict[str, List[Tuple[str, Path, str]]] = {band: [] for band in CHANNEL_TO_BAND.values()}
    for channel, band in CHANNEL_TO_BAND.items():
        render_dir = renders_root / channel
        render_files = sorted([p.name for p in render_dir.glob("*.png")])
        eval_items = eval_items_by_band[band]
        if len(render_files) != len(eval_items):
            raise RuntimeError(
                f"View count mismatch for {channel}/{band}: renders={len(render_files)} vs split_eval={len(eval_items)}. "
                "This usually means the json-list split contains entries missing from transforms.json, "
                "so the external dataparser silently evaluated fewer views than the copied split declares."
            )
        for render_name, rel in zip(render_files, eval_items):
            render_path = render_dir / render_name
            out[band].append((render_name, render_path, rel))
    return out


def _split_combined_render(combined_src: Path, render_dst: Path, gt_dst: Path) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    img = Image.open(combined_src).convert("RGB")
    width, height = img.size
    if width % 2 != 0:
        raise RuntimeError(f"Combined render width is not even: {combined_src} -> {width}x{height}")
    half = width // 2
    gt_img = img.crop((0, 0, half, height))
    pred_img = img.crop((half, 0, width, height))
    render_dst.parent.mkdir(parents=True, exist_ok=True)
    gt_dst.parent.mkdir(parents=True, exist_ok=True)
    pred_img.save(render_dst)
    gt_img.save(gt_dst)
    return pred_img.size, gt_img.size


def adapt_run(
    mmsplat_run_root: Path,
    split_json: Path,
    reference_run_root: Path,
    out_root: Path,
    synthetic_iteration: int,
    link_mode: str,
    gt_source: str,
) -> dict:
    run_root = mmsplat_run_root.resolve()
    split_json = split_json.resolve()
    reference_run_root = reference_run_root.resolve()
    out_root = out_root.resolve()
    renders_root = run_root / "eval" / "renders"
    eval_json = run_root / "eval" / "eval.json"
    if not renders_root.exists():
        raise FileNotFoundError(f"Missing renders root: {renders_root}")
    if not eval_json.exists():
        raise FileNotFoundError(f"Missing eval.json: {eval_json}")

    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    eval_items_by_band = _eval_entries_by_band(split_json)
    entries_by_band = _band_entries(eval_items_by_band, renders_root)

    audit = {
        "source_mmsplat_run_root": str(run_root),
        "split_json": str(split_json),
        "reference_run_root": str(reference_run_root),
        "synthetic_out_root": str(out_root),
        "synthetic_iteration": int(synthetic_iteration),
        "link_mode": link_mode,
        "gt_source": gt_source,
        "eval_frame_count": sum(len(v) for v in eval_items_by_band.values()),
        "bands": {},
    }

    for band in ("G", "R", "RE", "NIR"):
        model_root = out_root / f"Model_{band}"
        ours_root = model_root / "test" / f"ours_{int(synthetic_iteration)}"
        renders_dir = ours_root / "renders"
        gt_dir = ours_root / "gt"
        renders_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)

        ref_cfg = reference_run_root / f"Model_{band}" / "cfg_args"
        if not ref_cfg.exists():
            raise FileNotFoundError(f"Missing reference cfg_args: {ref_cfg}")
        ref_source_path = _parse_source_path_from_cfg(ref_cfg)
        _write_minimal_cfg_args(model_root / "cfg_args", ref_source_path, use_validity_mask=False)

        camera_rows = []
        sample_render_files: List[str] = []
        sample_gt_files: List[str] = []
        render_size = None
        gt_size = None
        for view_idx, (render_name, render_src, rel_item) in enumerate(entries_by_band[band]):
            canonical_name = f"{view_idx:05d}.png"
            if gt_source == "split_combined_render":
                render_size, gt_size = _split_combined_render(
                    render_src,
                    renders_dir / canonical_name,
                    gt_dir / canonical_name,
                )
                gt_src = render_src
            else:
                raise ValueError(f"Unsupported gt_source: {gt_source}")
            sample_render_files.append(str(render_src))
            sample_gt_files.append(str(gt_src))
            camera_rows.append({"img_name": Path(rel_item).name})

        (model_root / "cameras.json").write_text(
            json.dumps(camera_rows, indent=2),
            encoding="utf-8",
        )

        audit["bands"][band] = {
            "view_count": len(camera_rows),
            "reference_source_path": str(ref_source_path),
            "sample_render_sources": sample_render_files[:5],
            "sample_gt_sources": sample_gt_files[:5],
            "render_size": list(render_size) if render_size is not None else None,
            "gt_size": list(gt_size) if gt_size is not None else None,
        }

    audit_path = out_root / "mmsplat_adapter_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Adapt MMSplat eval renders into a synthetic repo-native run root for compare scripts."
    )
    ap.add_argument("--mmsplat_run_root", required=True)
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--reference_run_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--synthetic_iteration", type=int, default=60000)
    ap.add_argument("--link_mode", choices=["copy", "symlink", "hardlink"], default="symlink")
    ap.add_argument("--gt_source", choices=["split_combined_render"], default="split_combined_render")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    audit = adapt_run(
        mmsplat_run_root=Path(args.mmsplat_run_root),
        split_json=Path(args.split_json),
        reference_run_root=Path(args.reference_run_root),
        out_root=Path(args.out_root),
        synthetic_iteration=int(args.synthetic_iteration),
        link_mode=str(args.link_mode),
        gt_source=str(args.gt_source),
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
