import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image


@dataclass
class LoadedSpectralImage:
    path: str
    array: np.ndarray
    mode: str
    width: int
    height: int
    dtype_name: str
    channel_count: int
    metadata: Dict[str, Any]


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, (tuple, list)) and value:
        try:
            return float(value[0])
        except Exception:
            return None
    s = str(value).strip()
    if not s:
        return None
    if "/" in s:
        try:
            num, den = s.split("/", 1)
            den_v = float(den)
            if den_v == 0:
                return None
            return float(num) / den_v
        except Exception:
            return None
    try:
        return float(s)
    except Exception:
        return None


def _first_numeric(metadata: Dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        if key in metadata:
            value = _safe_float(metadata.get(key))
            if value is not None:
                return value
    return None


def _load_sidecar_metadata(path: Path) -> Dict[str, Any]:
    sidecar_paths = [
        path.with_suffix(path.suffix + ".meta.json"),
        path.with_suffix(".meta.json"),
        path.with_name(path.name + ".meta.json"),
    ]
    for sidecar in sidecar_paths:
        if sidecar.exists():
            try:
                return json.loads(sidecar.read_text(encoding="utf-8"))
            except Exception:
                return {}
    return {}


def extract_band_metadata(path, pil_or_array, exif_blob_if_any=None):
    path = Path(path)
    if isinstance(pil_or_array, Image.Image):
        arr = np.array(pil_or_array)
        mode = pil_or_array.mode
        width, height = pil_or_array.size
    else:
        arr = np.asarray(pil_or_array)
        mode = getattr(pil_or_array, "mode", "")
        if arr.ndim == 2:
            height, width = arr.shape
            channel_count = 1
        else:
            height, width = arr.shape[:2]
            channel_count = int(arr.shape[2])
        meta = {
            "source_path": str(path),
            "width": int(width),
            "height": int(height),
            "dtype": str(arr.dtype),
            "channel_count": channel_count,
            "mode": mode,
        }
        if isinstance(exif_blob_if_any, dict):
            meta.update(exif_blob_if_any)
        sidecar = _load_sidecar_metadata(path)
        if sidecar:
            meta.update(sidecar)
        return meta

    if arr.ndim == 2:
        channel_count = 1
    else:
        channel_count = int(arr.shape[2])

    meta = {
        "source_path": str(path),
        "width": int(width),
        "height": int(height),
        "dtype": str(arr.dtype),
        "channel_count": channel_count,
        "mode": mode,
    }
    if isinstance(exif_blob_if_any, dict):
        meta.update(exif_blob_if_any)
    sidecar = _load_sidecar_metadata(path)
    if sidecar:
        meta.update(sidecar)
    return meta


def load_image_preserve_dtype(path):
    path = Path(path)
    with Image.open(path) as image:
        image.load()
        arr = np.array(image)
        meta = extract_band_metadata(path, image, None)
        return LoadedSpectralImage(
            path=str(path),
            array=arr,
            mode=image.mode,
            width=int(image.size[0]),
            height=int(image.size[1]),
            dtype_name=str(arr.dtype),
            channel_count=1 if arr.ndim == 2 else int(arr.shape[2]),
            metadata=meta,
        )


def maybe_apply_black_level(image, metadata):
    arr = np.asarray(image, dtype=np.float32)
    black_level = _first_numeric(
        metadata,
        "black_level",
        "BlackLevel",
        "blacklevel",
        "Black Level",
    )
    if black_level is None:
        return arr
    arr = arr - float(black_level)
    arr[arr < 0.0] = 0.0
    return arr


def maybe_apply_exposure_gain_normalization(image, metadata):
    arr = np.asarray(image, dtype=np.float32)
    exposure_time = _first_numeric(
        metadata,
        "exposure_time",
        "ExposureTime",
        "ShutterSpeed",
        "Exposure Time",
    )
    iso = _first_numeric(metadata, "iso", "ISO", "PhotographicSensitivity")
    gain = _first_numeric(metadata, "gain", "Gain", "AnalogGain", "DigitalGain")

    scale = 1.0
    if exposure_time is not None and exposure_time > 0:
        scale /= float(exposure_time)
    if gain is not None and gain > 0:
        scale /= float(gain)
    elif iso is not None and iso > 0:
        scale /= max(float(iso) / 100.0, 1e-6)
    if scale != 1.0:
        arr = arr * float(scale)
    return arr


def maybe_apply_irradiance_normalization(image, metadata):
    arr = np.asarray(image, dtype=np.float32)
    irradiance = _first_numeric(
        metadata,
        "irradiance",
        "Irradiance",
        "solar_irradiance",
        "SolarIrradiance",
    )
    if irradiance is None or irradiance <= 0:
        return arr
    return arr / float(irradiance)


def normalize_scalar_band_image(image, metadata, mode, dynamic_range):
    if isinstance(image, LoadedSpectralImage):
        arr = np.asarray(image.array)
    else:
        arr = np.asarray(image)

    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        # Some aligned benchmark bands are stored as RGB/JPG grayscale proxies.
        # Keep the scalar-band semantics by collapsing the carrier channels late.
        arr = arr[..., :3].astype(np.float32, copy=False).mean(axis=2)
    if arr.ndim != 2:
        raise ValueError(f"Expected a single-band image, got shape={arr.shape}")

    arr = arr.astype(np.float32, copy=False)
    mode = str(mode or "raw_dn").strip().lower()
    dynamic_range = str(dynamic_range or "uint16").strip().lower()

    dtype_max = 1.0
    if dynamic_range == "uint8":
        dtype_max = 255.0
    elif dynamic_range == "uint16":
        dtype_max = 65535.0
    elif dynamic_range == "float":
        dtype_max = 1.0
    else:
        if np.issubdtype(arr.dtype, np.integer):
            dtype_max = float(np.iinfo(arr.dtype).max)

    if mode in ("exposure_normalized", "reflectance_ready_stub"):
        arr = maybe_apply_black_level(arr, metadata)
        arr = arr / max(dtype_max, 1.0)
        arr = maybe_apply_exposure_gain_normalization(arr, metadata)
        arr = maybe_apply_irradiance_normalization(arr, metadata)
    else:
        arr = arr / max(dtype_max, 1.0)

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    return arr.astype(np.float32, copy=False)


def replicate_single_band_to_rgb(image):
    arr = np.asarray(image)
    if arr.ndim == 2:
        return np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[0] == 1:
        return np.repeat(arr, 3, axis=0)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return np.repeat(arr, 3, axis=2)
    raise ValueError(f"Expected a scalar-band image, got shape={arr.shape}")
