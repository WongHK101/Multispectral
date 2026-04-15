from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

import numpy as np

# Compatibility shim for older MINIMA third-party code paths on modern NumPy.
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]


@contextmanager
def _temporary_cwd(path: Path):
    old = Path.cwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(old))


def _append_sys_path(path: Path) -> None:
    path_str = str(path.resolve())
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def check_backend_available(
    backend: str,
    minima_root: str | Path,
) -> bool:
    try:
        _prepare_minima_imports(minima_root)
    except Exception:
        return False
    backend_name = str(backend).strip().lower()
    if backend_name not in {"roma", "xoftr"}:
        return False
    try:
        if backend_name == "roma":
            from load_model import load_roma  # noqa: F401
        else:
            from load_model import load_xoftr  # noqa: F401
    except Exception:
        return False
    return True


def build_minima_matcher(
    backend: str,
    minima_root: str | Path,
    device: str = "cuda",
    ckpt: str = "",
    roma_size: str = "large",
    match_threshold: float = 0.3,
    fine_threshold: float = 0.1,
) -> "MinimaMatcherBridge":
    return MinimaMatcherBridge(
        backend=backend,
        minima_root=minima_root,
        device=device,
        ckpt=ckpt,
        roma_size=roma_size,
        match_threshold=match_threshold,
        fine_threshold=fine_threshold,
    )


def _prepare_minima_imports(minima_root: str | Path) -> Path:
    root = Path(minima_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"MINIMA root does not exist: {root}")
    _append_sys_path(root)
    _append_sys_path(root / "third_party" / "RoMa_minima")
    return root


class MinimaMatcherBridge:
    def __init__(
        self,
        backend: str = "roma",
        minima_root: str | Path = r"G:\2DSOTA\MINIMA",
        device: str = "cuda",
        ckpt: str = "",
        roma_size: str = "large",
        match_threshold: float = 0.3,
        fine_threshold: float = 0.1,
    ):
        self.backend = str(backend).strip().lower()
        if self.backend not in {"roma", "xoftr"}:
            raise ValueError(f"Unsupported MINIMA backend: {backend}")
        self.minima_root = _prepare_minima_imports(minima_root)
        self.device = str(device)
        self.ckpt = str(ckpt).strip()
        self.roma_size = str(roma_size).strip().lower()
        self.match_threshold = float(match_threshold)
        self.fine_threshold = float(fine_threshold)
        self._matcher_from_paths = self._build_matcher_from_paths()

    def _resolve_ckpt(self, default_name: str) -> Optional[str]:
        if self.ckpt:
            ckpt_path = Path(self.ckpt).resolve()
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Specified MINIMA checkpoint not found: {ckpt_path}")
            return str(ckpt_path)
        default_path = (self.minima_root / "weights" / default_name).resolve()
        if default_path.exists():
            return str(default_path)
        return None

    def _build_matcher_from_paths(self):
        with _temporary_cwd(self.minima_root):
            if self.backend == "roma":
                from load_model import load_roma

                args = SimpleNamespace(
                    ckpt2=self.roma_size if self.roma_size in {"large", "tiny"} else "large",
                    ckpt=self._resolve_ckpt("minima_roma.pth"),
                    device=self.device,
                )
                matcher_obj = load_roma(args, test_orginal_megadepth=False)
                return matcher_obj.from_paths
            if self.backend == "xoftr":
                from load_model import load_xoftr

                xoftr_ckpt = self._resolve_ckpt("minima_xoftr.ckpt")
                if not xoftr_ckpt:
                    raise FileNotFoundError(
                        "XoFTR backend requested but checkpoint is missing. "
                        "Expected minima_xoftr.ckpt in MINIMA weights or pass --minima_ckpt."
                    )
                args = SimpleNamespace(
                    match_threshold=self.match_threshold,
                    fine_threshold=self.fine_threshold,
                    ckpt=xoftr_ckpt,
                    device=self.device,
                )
                matcher_obj = load_xoftr(args)
                return matcher_obj.from_paths
        raise RuntimeError(f"Unsupported MINIMA backend: {self.backend}")

    @staticmethod
    def _to_numpy_matches(match_result: Dict[str, object]) -> Dict[str, np.ndarray]:
        if "mkpts0" in match_result and "mkpts1" in match_result:
            mkpts0 = np.asarray(match_result["mkpts0"], dtype=np.float32).reshape(-1, 2)
            mkpts1 = np.asarray(match_result["mkpts1"], dtype=np.float32).reshape(-1, 2)
        elif "keypoints0" in match_result and "keypoints1" in match_result:
            mkpts0 = np.asarray(match_result["keypoints0"], dtype=np.float32).reshape(-1, 2)
            mkpts1 = np.asarray(match_result["keypoints1"], dtype=np.float32).reshape(-1, 2)
        else:
            mkpts0 = np.zeros((0, 2), dtype=np.float32)
            mkpts1 = np.zeros((0, 2), dtype=np.float32)

        if "mconf" in match_result:
            mconf = np.asarray(match_result["mconf"], dtype=np.float32).reshape(-1)
        elif "matching_scores" in match_result:
            mconf = np.asarray(match_result["matching_scores"], dtype=np.float32).reshape(-1)
        else:
            mconf = np.ones((mkpts0.shape[0],), dtype=np.float32)

        if mconf.shape[0] != mkpts0.shape[0]:
            if mconf.shape[0] == 0:
                mconf = np.ones((mkpts0.shape[0],), dtype=np.float32)
            else:
                size = min(mconf.shape[0], mkpts0.shape[0])
                mkpts0 = mkpts0[:size]
                mkpts1 = mkpts1[:size]
                mconf = mconf[:size]
        return {"mkpts0": mkpts0, "mkpts1": mkpts1, "mconf": mconf}

    def match(self, rgb_path: str | Path, band_path: str | Path) -> Dict[str, object]:
        rgb = str(Path(rgb_path).resolve())
        band = str(Path(band_path).resolve())
        with _temporary_cwd(self.minima_root):
            result = self._matcher_from_paths(rgb, band)
        parsed = self._to_numpy_matches(result if isinstance(result, dict) else {})
        parsed.update(
            {
                "backend": self.backend,
                "success": parsed["mkpts0"].shape[0] > 0,
            }
        )
        return parsed
