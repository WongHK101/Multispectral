from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple


OFFICIAL_CAPTURE_RE = re.compile(r"^(?:D|MS_[A-Z]+)_(?P<frame>\d+)$")


def _load_loose_json(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r",(?=\s*[\]}])", "", text)
    return json.loads(text)


def _entry_value(item: object) -> str:
    if isinstance(item, dict):
        value = item.get("official_path") or item.get("images2_path") or item.get("path") or item.get("file_path")
    else:
        value = item
    return str(value).replace("\\", "/").strip()


def _channel_name(rel_path: str) -> str:
    parts = Path(rel_path).parts
    return str(parts[1]) if len(parts) >= 2 else "UNKNOWN"


def _capture_group_key(rel_path: str) -> str:
    stem = Path(rel_path).stem
    match = OFFICIAL_CAPTURE_RE.match(stem)
    if match:
        return match.group("frame")
    return _path_key(rel_path)


def _path_key(rel_path: str) -> str:
    return str(rel_path).replace("\\", "/").strip()


def _valid_frame_paths(transforms_json: Path) -> set[str]:
    payload = _load_loose_json(transforms_json)
    frames = payload.get("frames", [])
    valid = set()
    for frame in frames:
        rel = str(frame.get("file_path", "")).replace("\\", "/").strip()
        if rel:
            valid.add(rel)
    if not valid:
        raise RuntimeError(f"No frame file_path entries found in {transforms_json}")
    return valid


def _sanitize_section(
    items: Iterable[object],
    valid_paths: set[str],
) -> Tuple[list[object], list[str], Dict[str, int], Dict[str, int]]:
    groups: Dict[str, list[Tuple[object, str, str]]] = defaultdict(list)
    group_order: list[str] = []
    for item in items:
        rel = _entry_value(item)
        group_key = _capture_group_key(rel)
        if group_key not in groups:
            group_order.append(group_key)
        groups[group_key].append((item, rel, _channel_name(rel)))

    kept: list[object] = []
    removed: list[str] = []
    kept_by_channel: Dict[str, int] = defaultdict(int)
    removed_by_channel: Dict[str, int] = defaultdict(int)
    for group_key in group_order:
        group_items = groups[group_key]
        group_valid = all(rel in valid_paths for _, rel, _ in group_items)
        if group_valid:
            for item, rel, channel in group_items:
                kept.append(item)
                kept_by_channel[channel] += 1
        else:
            for _, rel, channel in group_items:
                removed.append(rel)
                removed_by_channel[channel] += 1
    return kept, removed, dict(kept_by_channel), dict(removed_by_channel)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sanitize an MMSplat json-list split so train/eval entries match transforms.json frame file_path entries."
    )
    ap.add_argument("--src_json", required=True, help="Source train_split.json (loose JSON allowed).")
    ap.add_argument("--transforms_json", required=True, help="transforms.json used by the external dataset root.")
    ap.add_argument("--dst_json", required=True, help="Strict sanitized output JSON path.")
    ap.add_argument("--audit_json", default="", help="Optional audit JSON path.")
    ap.add_argument(
        "--fail_on_removed_eval",
        action="store_true",
        help="Fail if any eval/test entries are removed during sanitization.",
    )
    args = ap.parse_args()

    src_json = Path(args.src_json).resolve()
    transforms_json = Path(args.transforms_json).resolve()
    dst_json = Path(args.dst_json).resolve()
    audit_json = Path(args.audit_json).resolve() if args.audit_json else None

    payload = _load_loose_json(src_json)
    valid_paths = _valid_frame_paths(transforms_json)

    eval_key = "eval" if "eval" in payload else "test"
    if eval_key not in payload:
        raise RuntimeError(f"Expected eval/test list in {src_json}")

    train_items = payload.get("train", [])
    eval_items = payload.get(eval_key, [])

    train_kept, train_removed, train_kept_by_channel, train_removed_by_channel = _sanitize_section(train_items, valid_paths)
    eval_kept, eval_removed, eval_kept_by_channel, eval_removed_by_channel = _sanitize_section(eval_items, valid_paths)

    sanitized = dict(payload)
    sanitized["train"] = train_kept
    sanitized[eval_key] = eval_kept

    dst_json.parent.mkdir(parents=True, exist_ok=True)
    dst_json.write_text(json.dumps(sanitized, indent=2, ensure_ascii=False), encoding="utf-8")

    audit = {
        "source_json": str(src_json),
        "transforms_json": str(transforms_json),
        "output_json": str(dst_json),
        "valid_frame_count": len(valid_paths),
        "train_before": len(train_items),
        "train_after": len(train_kept),
        "train_removed_count": len(train_removed),
        "train_removed_preview": train_removed[:16],
        "train_kept_by_channel": train_kept_by_channel,
        "train_removed_by_channel": train_removed_by_channel,
        "eval_key": eval_key,
        "eval_before": len(eval_items),
        "eval_after": len(eval_kept),
        "eval_removed_count": len(eval_removed),
        "eval_removed_preview": eval_removed[:16],
        "eval_kept_by_channel": eval_kept_by_channel,
        "eval_removed_by_channel": eval_removed_by_channel,
    }

    if audit_json:
        audit_json.parent.mkdir(parents=True, exist_ok=True)
        audit_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.fail_on_removed_eval and eval_removed:
        raise RuntimeError(
            f"Removed {len(eval_removed)} eval/test entries not present in transforms.json. "
            f"First removed: {eval_removed[:8]}"
        )

    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
