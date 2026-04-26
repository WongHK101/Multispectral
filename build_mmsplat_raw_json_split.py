from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


RAW_D_RE = re.compile(
    r"^(?P<prefix>.+?)_(?P<frame>\d{4})_D\.(?P<ext>jpe?g)$",
    re.IGNORECASE,
)
RAW_MS_RE = re.compile(
    r"^(?P<prefix>.+?)_(?P<frame>\d{4})_MS_(?P<band>G|R|RE|NIR)\.(?P<ext>tiff?|png)$",
    re.IGNORECASE,
)

CHANNEL_ORDER = ("D", "MS_G", "MS_R", "MS_RE", "MS_NIR")
MS_CHANNELS = ("MS_G", "MS_R", "MS_RE", "MS_NIR")


def _capture_tokens(path: Path) -> Tuple[str, str] | None:
    name = path.name
    m = RAW_D_RE.match(name)
    if m:
        return m.group("prefix"), m.group("frame")
    m = RAW_MS_RE.match(name)
    if m:
        return m.group("prefix"), m.group("frame")
    return None


def _channel_name(path: Path) -> str | None:
    name = path.name
    m = RAW_D_RE.match(name)
    if m:
        return "D"
    m = RAW_MS_RE.match(name)
    if m:
        return f"MS_{m.group('band').upper()}"
    return None


def _load_eval_d_names_from_cameras(cameras_json: Path) -> List[str]:
    payload = json.loads(cameras_json.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected camera list in {cameras_json}")
    names: List[str] = []
    for row in payload:
        if not isinstance(row, dict) or "img_name" not in row:
            raise RuntimeError(f"Malformed cameras row in {cameras_json}: {row!r}")
        names.append(str(row["img_name"]))
    if not names:
        raise RuntimeError(f"No img_name entries found in {cameras_json}")
    return names


def _group_key(path: Path, group_mode: str) -> str | None:
    tokens = _capture_tokens(path)
    if tokens is None:
        return None
    prefix, frame = tokens
    if group_mode == "prefix_frame":
        return f"{prefix}_{frame}"
    if group_mode == "frame_only":
        return frame
    raise ValueError(f"Unsupported group_mode: {group_mode}")


def _collect_complete_groups(raw_root: Path, group_mode: str) -> List[dict]:
    grouped: Dict[str, Dict[str, Path]] = {}
    for path in sorted(raw_root.iterdir()):
        if not path.is_file():
            continue
        key = _group_key(path, group_mode=group_mode)
        channel = _channel_name(path)
        if key is None or channel is None:
            continue
        grouped.setdefault(key, {})[channel] = path

    complete: List[dict] = []
    for key, channels in grouped.items():
        if not all(ch in channels for ch in CHANNEL_ORDER):
            continue
        complete.append(
            {
                "capture_key": key,
                "paths": channels,
                "d_name": channels["D"].name,
            }
        )
    complete.sort(key=lambda row: row["d_name"])
    return complete


def _processed_rel_paths(index_1based: int) -> Dict[str, str]:
    idx = f"{index_1based:05d}"
    return {
        "D": f"images/D/D_{idx}.JPG",
        "MS_G": f"images/MS_G/MS_G_{idx}.png",
        "MS_R": f"images/MS_R/MS_R_{idx}.png",
        "MS_RE": f"images/MS_RE/MS_RE_{idx}.png",
        "MS_NIR": f"images/MS_NIR/MS_NIR_{idx}.png",
    }


def _eval_d_names_from_hold(complete: List[dict], eval_hold: int) -> List[str]:
    if eval_hold <= 0:
        raise RuntimeError(f"eval_hold must be positive, got {eval_hold}")
    ordered = [row["d_name"] for row in complete]
    return [name for idx, name in enumerate(ordered) if idx % eval_hold == 0]


def build_split(
    raw_root: Path,
    cameras_json: Path | None,
    eval_hold: int | None,
    out_json: Path,
    audit_json: Path | None,
    group_mode: str,
) -> dict:
    complete = _collect_complete_groups(raw_root, group_mode=group_mode)
    if not complete:
        raise RuntimeError(f"No complete D/MS_G/MS_R/MS_RE/MS_NIR groups found under {raw_root}")
    if cameras_json is not None:
        eval_d_names = set(_load_eval_d_names_from_cameras(cameras_json))
        eval_source = f"cameras_json:{cameras_json}"
    elif eval_hold is not None:
        eval_d_names = set(_eval_d_names_from_hold(complete, eval_hold=eval_hold))
        eval_source = f"llffhold:{eval_hold}"
    else:
        raise RuntimeError("Either cameras_json or eval_hold must be provided.")

    missing_eval = sorted(eval_d_names - {row["d_name"] for row in complete})
    if missing_eval:
        raise RuntimeError(
            f"{len(missing_eval)} eval D images from {cameras_json} are not present as complete groups under {raw_root}. "
            f"First missing: {missing_eval[:8]}"
        )

    train_entries: List[str] = []
    eval_entries: List[str] = []
    audit_rows: List[dict] = []
    for idx, row in enumerate(complete, start=1):
        rels = _processed_rel_paths(idx)
        target = eval_entries if row["d_name"] in eval_d_names else train_entries
        for channel in CHANNEL_ORDER:
            target.append(rels[channel])
        audit_rows.append(
            {
                "index_1based": idx,
                "capture_key": row["capture_key"],
                "original_d_name": row["d_name"],
                "split": "eval" if row["d_name"] in eval_d_names else "train",
                "processed_rel_paths": rels,
            }
        )

    payload = {
        "train": train_entries,
        "eval": eval_entries,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    audit = {
        "raw_root": str(raw_root),
        "cameras_json": str(cameras_json) if cameras_json is not None else "",
        "eval_hold": int(eval_hold) if eval_hold is not None else None,
        "eval_source": eval_source,
        "output_json": str(out_json),
        "group_mode": group_mode,
        "complete_group_count": len(complete),
        "eval_d_count": len(eval_d_names),
        "train_entry_count": len(train_entries),
        "eval_entry_count": len(eval_entries),
        "train_capture_count": len(train_entries) // len(CHANNEL_ORDER),
        "eval_capture_count": len(eval_entries) // len(CHANNEL_ORDER),
        "channel_order": list(CHANNEL_ORDER),
        "eval_d_names_preview": sorted(eval_d_names)[:16],
        "rows_preview": audit_rows[:16],
    }
    if audit_json is not None:
        audit_json.parent.mkdir(parents=True, exist_ok=True)
        audit_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build a strict MMSplat json-list split for a flat raw UAV multispectral directory "
            "using either the reference E3 test-view cameras.json or an llffhold rule as the eval set."
        )
    )
    ap.add_argument("--raw_root", required=True, help="Flat raw directory containing D and MS_* files.")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--cameras_json", help="Reference E3 Model_*/cameras.json listing eval D views.")
    group.add_argument("--eval_hold", type=int, help="Use legacy llffhold on lexicographically sorted D names.")
    ap.add_argument("--out_json", required=True, help="Strict json-list split output path.")
    ap.add_argument("--audit_json", default="", help="Optional audit JSON path.")
    ap.add_argument(
        "--group_mode",
        choices=["prefix_frame", "frame_only"],
        default="prefix_frame",
        help="How to group raw D/MS files into a capture unit.",
    )
    args = ap.parse_args()

    audit = build_split(
        raw_root=Path(args.raw_root).resolve(),
        cameras_json=Path(args.cameras_json).resolve() if args.cameras_json else None,
        eval_hold=args.eval_hold,
        out_json=Path(args.out_json).resolve(),
        audit_json=Path(args.audit_json).resolve() if args.audit_json else None,
        group_mode=args.group_mode,
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
