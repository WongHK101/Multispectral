from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


PROTOCOL_ID = "UNIFIED_EXPERIMENT_PROTOCOL_v1_20260416"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
M3M_RGB_RE = re.compile(r"^(?P<prefix>.+?)_(?P<frame>\d{4})_D\.(?P<ext>jpe?g)$", re.IGNORECASE)
M3M_BAND_RE = re.compile(
    r"^(?P<prefix>.+?)_(?P<frame>\d{4})_MS_(?P<band>G|R|RE|NIR)\.(?P<ext>tiff?|tif)$",
    re.IGNORECASE,
)
M3M_REQUIRED_KEYS = ("RGB", "G", "R", "RE", "NIR")


M3M_SCENES = [
    ("self_m3m", Path(r"G:\Multispectral\dataset"), "multispectral_main"),
    ("vineyard_001_reynoldsTR01crossrtk", Path(r"G:\Multispectral\othersdatasets\20240528\20240528\DJI_202405281154_001_reynoldsTR01crossrtk"), "multispectral_main"),
    ("vineyard_002_reynoldsAB02crossrtk", Path(r"G:\Multispectral\othersdatasets\20240528\20240528\DJI_202405281154_002_reynoldsAB02crossrtk"), "multispectral_main"),
    ("vineyard_003_reynoldsAb01crossrtk", Path(r"G:\Multispectral\othersdatasets\20240528\20240528\DJI_202405281220_003_reynoldsAb01crossrtk"), "multispectral_main"),
    ("vineyard_004_jprAb01crossrtk", Path(r"G:\Multispectral\othersdatasets\20240528\20240528\DJI_202405281326_004_jprAb01crossrtk"), "multispectral_main"),
    ("vineyard_005_jprAr01crossrtk", Path(r"G:\Multispectral\othersdatasets\20240528\20240528\DJI_202405281358_005_jprAr01crossrtk"), "multispectral_main"),
    ("vineyard_006_jprAb02crossrtk", Path(r"G:\Multispectral\othersdatasets\20240528\20240528\DJI_202405281358_006_jprAb02crossrtk"), "multispectral_main"),
]

OFFICIAL_MS_SCENES = [
    ("ms_golf", Path(r"G:\Multispectral\othersdatasets\ms-splatting-dataset\dataset\ms-golf")),
    ("ms_lake", Path(r"G:\Multispectral\othersdatasets\ms-splatting-dataset\dataset\ms-lake")),
    ("ms_solar", Path(r"G:\Multispectral\othersdatasets\ms-splatting-dataset\dataset\ms-solar")),
]

RGBT_SELF_SCENES = [
    ("Cropland", Path(r"G:\Multispectral\othersdatasets\Cropland")),
    ("Orchard", Path(r"G:\Multispectral\othersdatasets\Orchard")),
]

MTV_SCENES = [
    ("MTV_building", Path(r"G:\Multispectral\othersdatasets\MTV\building")),
    ("MTV_contryroad", Path(r"G:\Multispectral\othersdatasets\MTV\contryroad")),
    ("MTV_park", Path(r"G:\Multispectral\othersdatasets\MTV\park")),
    ("MTV_race_track", Path(r"G:\Multispectral\othersdatasets\MTV\race track")),
    ("MTV_villa", Path(r"G:\Multispectral\othersdatasets\MTV\villa")),
]


def _write_text_list(path: Path, items: Iterable[str]) -> None:
    path.write_text("\n".join(items) + "\n", encoding="utf-8")


def _split_stride(names: List[str], stride: int = 8) -> Tuple[List[str], List[str]]:
    names = sorted(names)
    test = [name for idx, name in enumerate(names) if idx % stride == 0]
    train = [name for name in names if name not in set(test)]
    return train, test


def _write_split(scene_dir: Path, payload: Dict[str, object]) -> Dict[str, str]:
    scene_dir.mkdir(parents=True, exist_ok=True)
    split_path = scene_dir / "split_v1.json"
    split_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_text_list(scene_dir / "train.txt", [_entry_name(item) for item in payload.get("train", [])])
    _write_text_list(scene_dir / "test.txt", [_entry_name(item) for item in payload.get("test", payload.get("eval", []))])
    (scene_dir / "README.md").write_text(
        "# Frozen Split\n\n"
        f"- protocol_id: `{payload.get('protocol_id')}`\n"
        f"- scene_name: `{payload.get('scene_name')}`\n"
        f"- source_root: `{payload.get('source_root')}`\n"
        f"- split_policy: `{payload.get('split_policy')}`\n"
        f"- pose_policy: `{payload.get('pose_policy')}`\n"
        f"- train_count: `{len(payload.get('train', []))}`\n"
        f"- test_count: `{len(payload.get('test', payload.get('eval', [])))}`\n",
        encoding="utf-8",
    )
    return {
        "split_json": str(split_path),
        "train_txt": str(scene_dir / "train.txt"),
        "test_txt": str(scene_dir / "test.txt"),
    }


def _entry_name(entry) -> str:
    if isinstance(entry, dict):
        value = entry.get("image_name") or entry.get("rgb_image") or entry.get("path") or entry.get("file_path")
    else:
        value = entry
    return Path(str(value).replace("\\", "/")).name


def _m3m_complete_rgb_names(root: Path) -> List[str]:
    """Return RGB names whose capture-group frame id has all M3M bands.

    DJI Mavic 3M files from the same capture can differ by one or more
    seconds in the timestamp portion of the filename.  The stable pairing key
    is the four-digit frame id before `_D` / `_MS_*`, which is also what the
    dataset preparation script uses.  Do not require the timestamp stems to be
    byte-identical here, otherwise valid capture groups are silently excluded
    from the frozen split.
    """
    groups: Dict[str, Dict[str, Path]] = {}
    for path in _list_images(root):
        rgb_match = M3M_RGB_RE.match(path.name)
        if rgb_match:
            groups.setdefault(rgb_match.group("frame"), {})["RGB"] = path
            continue
        band_match = M3M_BAND_RE.match(path.name)
        if band_match:
            groups.setdefault(band_match.group("frame"), {})[band_match.group("band").upper()] = path

    complete = []
    for frame_id, items in sorted(groups.items()):
        if all(key in items for key in M3M_REQUIRED_KEYS):
            complete.append(items["RGB"].name)
    if not complete:
        raise FileNotFoundError(f"No complete M3M RGB/Band groups found under {root}")
    return complete


def _list_images(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS], key=lambda p: p.name)


def _rgbt_self_pairs(root: Path) -> List[Dict[str, str]]:
    rgb_dir = root / "rgb"
    thermal_dir = root / "thermal"
    if not rgb_dir.exists() or not thermal_dir.exists():
        raise FileNotFoundError(f"Expected rgb/ and thermal/ under {root}")
    rgb_by_stem = {p.stem: p.name for p in _list_images(rgb_dir)}
    thermal_by_stem = {p.stem: p.name for p in _list_images(thermal_dir)}
    common = sorted(set(rgb_by_stem) & set(thermal_by_stem))
    if not common:
        raise RuntimeError(f"No RGB/Thermal stem pairs under {root}")
    return [
        {
            "image_name": rgb_by_stem[stem],
            "rgb_image": rgb_by_stem[stem],
            "thermal_image": thermal_by_stem[stem],
            "paired_group_id": stem,
        }
        for stem in common
    ]


def _colmap_image_names_from_text(path: Path) -> List[str]:
    names = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            try:
                int(parts[0])
            except ValueError:
                continue
            name = parts[9]
            if Path(name).suffix.lower() in IMAGE_EXTS:
                names.append(name)
    if not names:
        raise RuntimeError(f"No COLMAP image records found in {path}")
    return sorted(set(names))


def _load_loose_json(path: Path) -> Dict[str, object]:
    text = path.read_text(encoding="utf-8")
    # Some downloaded official split files contain trailing commas.
    text = re.sub(r",(\s*[\]}])", r"\1", text)
    return json.loads(text)


def _official_ms_split(root: Path) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    payload = _load_loose_json(root / "train_split.json")
    def convert(items: List[str]) -> List[Dict[str, str]]:
        out = []
        for item in items:
            rel = str(item).replace("\\", "/")
            image_name = Path(rel).name
            out.append(
                {
                    "image_name": image_name,
                    "official_path": rel,
                    "images2_path": rel.replace("images/", "images_2/", 1),
                }
            )
        return out
    return convert(payload.get("train", [])), convert(payload.get("eval", []))


def _append_index_row(rows: List[Dict[str, object]], split_paths: Dict[str, str], payload: Dict[str, object]) -> None:
    rows.append(
        {
            "scene_name": payload["scene_name"],
            "track": payload["track"],
            "source_root": payload["source_root"],
            "split_policy": payload["split_policy"],
            "pose_policy": payload["pose_policy"],
            "train_count": len(payload.get("train", [])),
            "test_count": len(payload.get("test", payload.get("eval", []))),
            "split_json": split_paths["split_json"],
            "train_txt": split_paths["train_txt"],
            "test_txt": split_paths["test_txt"],
        }
    )


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_official_ms_templates(protocols_root: Path, split_index_rows: List[Dict[str, object]]) -> None:
    csv_rows = []
    ps_lines = [
        "# Official MS special-case command templates.",
        "# These scenes use dataset-provided images_2/splits and MUST override internal resize to r1.",
        "# The prepare step converts official transforms into repo-readable aligned scene roots.",
        "$env:PYTHONPATH='G:\\Multispectral\\repo-B735'",
        "",
    ]
    for scene_name, root in OFFICIAL_MS_SCENES:
        split_json = next(row["split_json"] for row in split_index_rows if row["scene_name"] == scene_name)
        prepared = rf"G:\Multispectral\runs\protocol_packs\{PROTOCOL_ID}\official_ms\{scene_name}"
        out_root = rf"G:\Multispectral\runs\experiments\{PROTOCOL_ID}\multispectral_main\{scene_name}\A2_MINIMA_plus_residual"
        cmd = (
            "python G:\\Multispectral\\repo-B735\\prepare_official_ms_scene.py "
            f"--source_root \"{root}\" "
            f"--out_root \"{prepared}\" "
            "--image_root images_2 "
            f"--split_json \"{split_json}\""
        )
        rgb_cmd = (
            "python G:\\Multispectral\\repo-B735\\train.py "
            f"-s \"{prepared}\\RGB\" -m \"{out_root}\\Model_RGB\" -r 1 --eval "
            "--iterations 30000 --checkpoint_iterations 30000 --save_iterations 30000 --test_iterations 30000 "
            "--disable_viewer --modality_kind rgb"
        )
        band_cmds = []
        for band, scene_dir in [("G", "G_aligned"), ("R", "R_aligned"), ("RE", "RE_aligned"), ("NIR", "NIR_aligned")]:
            band_cmds.append(
                "python G:\\Multispectral\\repo-B735\\train.py "
                f"-s \"{prepared}\\{scene_dir}\" -m \"{out_root}\\Model_{band}\" -r 1 --eval "
                f"--iterations 60000 --checkpoint_iterations 60000 --save_iterations 60000 --test_iterations 60000 "
                f"--start_checkpoint \"{out_root}\\Model_RGB\\chkpnt30000.pth\" "
                f"--modality_kind band --target_band {band} --single_band_mode true --single_band_replicate_to_rgb true "
                "--input_dynamic_range uint8 --radiometric_mode raw_dn "
                "--stage2_mode band_transfer --reset_appearance_features true --freeze_geometry true "
                "--freeze_opacity true --tied_scalar_carrier true --feature_lr 0.001 --lambda_dssim 0 "
                "--require_rectified_band_scene false --use_validity_mask false --disable_viewer"
            )
        train_note = "Official MS uses images_2 and explicit -r 1; do not use the pipeline r8 defaults."
        csv_rows.append(
            {
                "scene_name": scene_name,
                "source_root": str(root),
                "split_json": split_json,
                "resolution_override": "rgb_res=1;band_res=1",
                "prepared_root": prepared,
                "out_root": out_root,
                "prepare_template": cmd,
                "train_rgb_template": rgb_cmd,
                "train_band_templates": " || ".join(band_cmds),
                "train_note": train_note,
            }
        )
        ps_lines.extend(
            [
                f"# {scene_name}",
                cmd,
                rgb_cmd,
                *band_cmds,
                "",
            ]
        )
    _write_csv(protocols_root / "official_ms_command_templates_20260416.csv", csv_rows)
    (protocols_root / "official_ms_command_templates_20260416.ps1").write_text("\n".join(ps_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze protocol split files and command templates without touching raw data.")
    parser.add_argument("--out_root", default=rf"G:\Multispectral\runs\protocol_splits\{PROTOCOL_ID}")
    parser.add_argument("--pack_root", default=rf"G:\Multispectral\runs\protocol_packs\{PROTOCOL_ID}")
    parser.add_argument("--protocols_root", default=r"G:\Multispectral\protocols")
    parser.add_argument("--stride", type=int, default=8)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    pack_root = Path(args.pack_root)
    protocols_root = Path(args.protocols_root)
    out_root.mkdir(parents=True, exist_ok=True)
    pack_root.mkdir(parents=True, exist_ok=True)
    protocols_root.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now().isoformat(timespec="seconds")
    index_rows: List[Dict[str, object]] = []

    for scene_name, root, track in M3M_SCENES:
        names = _m3m_complete_rgb_names(root)
        train, test = _split_stride(names, stride=int(args.stride))
        payload = {
            "protocol_id": PROTOCOL_ID,
            "created_at": created_at,
            "scene_name": scene_name,
            "track": track,
            "source_root": str(root),
            "split_policy": (
                f"m3m_complete_capture_group_by_frame_id_then_uniform_stride_holdout_v1_"
                f"every_{args.stride}th_sorted_by_rgb_filename"
            ),
            "pose_policy": "train_split_controls_supervision_and_eval;SfM_may_use_train_images_only_unless_sparse_source_is_explicitly_provided",
            "train": train,
            "test": test,
        }
        split_paths = _write_split(out_root / "multispectral" / scene_name, payload)
        _append_index_row(index_rows, split_paths, payload)

    for scene_name, root in OFFICIAL_MS_SCENES:
        train, test = _official_ms_split(root)
        payload = {
            "protocol_id": PROTOCOL_ID,
            "created_at": created_at,
            "scene_name": scene_name,
            "track": "multispectral_main_official_ms",
            "source_root": str(root),
            "split_policy": "dataset_provided_train_split_json",
            "pose_policy": "dataset_provided_transforms_and_sparse;split_controls_supervision_and_eval",
            "image_root": "images_2",
            "internal_resize": 1,
            "train": train,
            "test": test,
        }
        split_paths = _write_split(out_root / "multispectral" / scene_name, payload)
        _append_index_row(index_rows, split_paths, payload)

    for scene_name, root in RGBT_SELF_SCENES:
        pairs = _rgbt_self_pairs(root)
        train_names, test_names = _split_stride([item["image_name"] for item in pairs], stride=int(args.stride))
        pair_by_name = {item["image_name"]: item for item in pairs}
        train = [pair_by_name[name] for name in train_names]
        test = [pair_by_name[name] for name in test_names]
        payload = {
            "protocol_id": PROTOCOL_ID,
            "created_at": created_at,
            "scene_name": scene_name,
            "track": "rgbt_main",
            "source_root": str(root),
            "split_policy": f"uniform_stride_holdout_v1_every_{args.stride}th_sorted_by_rgb_filename",
            "pose_policy": "train_split_controls_supervision_and_eval;SfM_should_be_computed_from_shared_protocol_scene_not_raw_root",
            "train": train,
            "test": test,
        }
        split_paths = _write_split(out_root / "rgbt" / scene_name, payload)
        _append_index_row(index_rows, split_paths, payload)

    for scene_name, root in MTV_SCENES:
        names = _colmap_image_names_from_text(root / "sparse" / "images.txt")
        train, test = _split_stride(names, stride=int(args.stride))
        payload = {
            "protocol_id": PROTOCOL_ID,
            "created_at": created_at,
            "scene_name": scene_name,
            "track": "rgbt_main_mtv",
            "source_root": str(root),
            "split_policy": f"uniform_stride_holdout_v1_every_{args.stride}th_sorted_by_colmap_image_name",
            "pose_policy": "known_pose_full_sparse;dataset_sparse_may_use_all_views;split_controls_supervision_and_eval_only",
            "train": train,
            "test": test,
        }
        split_paths = _write_split(out_root / "rgbt" / scene_name, payload)
        _append_index_row(index_rows, split_paths, payload)

    _write_csv(out_root / "split_index.csv", index_rows)
    _write_csv(protocols_root / "frozen_split_index_20260416.csv", index_rows)

    scene_index = {
        "protocol_id": PROTOCOL_ID,
        "created_at": created_at,
        "raw_data_policy": "read_only",
        "split_root": str(out_root),
        "pack_root": str(pack_root),
        "notes": [
            "This pack intentionally stores manifests and frozen splits only.",
            "Large images remain in their raw source roots and must not be modified.",
            "Derived prepared/rectified scenes should be written under runs/protocol_packs or runs/experiments.",
        ],
        "scenes": index_rows,
    }
    (pack_root / "scene_index.json").write_text(json.dumps(scene_index, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_official_ms_templates(protocols_root, index_rows)

    print(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "split_root": str(out_root),
        "pack_root": str(pack_root),
        "scene_count": len(index_rows),
        "split_index": str(out_root / "split_index.csv"),
        "official_ms_templates": str(protocols_root / "official_ms_command_templates_20260416.csv"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
