"""CPU-only safety and filename-grouping tests for raw Mavic 3M preparation."""

from __future__ import annotations

from pathlib import Path

import pytest

from prepare_m3m_multispectral import _assert_disjoint_roots, _scan_groups


def test_rejects_equal_or_nested_raw_and_output_roots(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()

    with pytest.raises(ValueError):
        _assert_disjoint_roots(raw, raw)
    with pytest.raises(ValueError):
        _assert_disjoint_roots(raw, raw / "prepared")
    with pytest.raises(ValueError):
        _assert_disjoint_roots(raw, tmp_path)


def test_accepts_disjoint_roots(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    prepared = tmp_path / "prepared"
    raw.mkdir()

    resolved_raw, resolved_prepared = _assert_disjoint_roots(raw, prepared)

    assert resolved_raw == raw.resolve()
    assert resolved_prepared == prepared.resolve()


def test_dji_mavic_3m_groups_are_paired_by_frame_id(tmp_path: Path) -> None:
    names = [
        "DJI_20260526160312_0001_D.JPG",
        "DJI_20260526160312_0001_MS_G.TIF",
        "DJI_20260526160312_0001_MS_R.TIF",
        "DJI_20260526160312_0001_MS_RE.TIF",
        "DJI_20260526160312_0001_MS_NIR.TIF",
    ]
    for name in names:
        (tmp_path / name).write_bytes(b"test")

    groups = _scan_groups(tmp_path)

    assert sorted(groups) == ["0001"]
    assert set(groups["0001"]) == {"RGB", "G", "R", "RE", "NIR"}
