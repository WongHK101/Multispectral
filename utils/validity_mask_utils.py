import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


def _read_cfg_text(cfg_path: Path) -> str:
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing cfg_args: {cfg_path}")
    return cfg_path.read_text(encoding="utf-8", errors="ignore")


def parse_source_path_from_cfg(cfg_path: Path) -> Path:
    text = _read_cfg_text(cfg_path)
    m = re.search(r"source_path\s*=\s*['\"]([^'\"]+)['\"]", text)
    if not m:
        raise ValueError(f"Cannot parse source_path from {cfg_path}")
    return Path(m.group(1)).resolve()


def parse_bool_flag_from_cfg(cfg_path: Path, flag_name: str, default: bool) -> bool:
    text = _read_cfg_text(cfg_path)
    m = re.search(rf"{re.escape(flag_name)}\s*=\s*(True|False|true|false)", text)
    if not m:
        return bool(default)
    return m.group(1).lower() == "true"


def resolve_validity_mask_policy(cfg_path: Path) -> Tuple[Path, Path, bool]:
    source_path = parse_source_path_from_cfg(cfg_path)
    mask_dir = source_path / "validity_masks"
    use_validity_mask = parse_bool_flag_from_cfg(cfg_path, "use_validity_mask", default=True)
    return source_path, mask_dir, use_validity_mask


def load_validity_mask_image(mask_path: Path, target_hw: Tuple[int, int]) -> np.ndarray:
    h, w = target_hw
    mask_img = Image.open(mask_path).convert("L")
    if mask_img.size != (w, h):
        mask_img = mask_img.resize((w, h), resample=Image.Resampling.NEAREST)
    mask = np.asarray(mask_img, dtype=np.float32) / 255.0
    return mask > 0.5


def load_validity_mask_or_ones(
    cfg_path: Path,
    image_name: str,
    target_hw: Tuple[int, int],
) -> Tuple[np.ndarray, Optional[Path], bool]:
    _, mask_dir, use_validity_mask = resolve_validity_mask_policy(cfg_path)
    mask_path = mask_dir / f"{Path(image_name).stem}.png"
    if mask_path.exists():
        return load_validity_mask_image(mask_path, target_hw), mask_path, use_validity_mask
    if use_validity_mask:
        raise FileNotFoundError(f"Missing validity mask: {mask_path}")
    return np.ones(target_hw, dtype=np.bool_), None, use_validity_mask


def load_pair_keys_for_image_names(cfg_path: Path, image_names: List[str]) -> List[str]:
    source_path = parse_source_path_from_cfg(cfg_path)
    manifest_path = source_path / "spectral_manifest.json"
    default_keys = [Path(name).stem for name in image_names]
    if not manifest_path.exists():
        return default_keys

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return default_keys

    key_by_name = {}
    for item in manifest.get("images", []):
        image_name = str(item.get("image_name", "")).strip()
        if not image_name:
            continue
        pair_key = str(item.get("paired_group_id") or Path(image_name).stem)
        key_by_name[image_name] = pair_key
        key_by_name[Path(image_name).stem] = pair_key

    out = []
    for name in image_names:
        stem = Path(name).stem
        out.append(key_by_name.get(name, key_by_name.get(stem, stem)))
    return out
