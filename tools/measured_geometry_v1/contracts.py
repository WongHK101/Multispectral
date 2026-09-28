"""Local integrity and header inspection. Never import a renderer or load depth values."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import struct
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np


PACKET_SCHEMA = "ms_gcp_metric_depth_packet_v2"
FLOAT_TENSORS = (
    "accumulated_alpha", "weighted_camera_z_sum",
    "weighted_camera_z_second_moment", "weighted_inverse_camera_z_sum",
    "alpha_normalized_expected_camera_z", "alpha_normalized_expected_inverse_camera_z",
    "harmonic_camera_z", "camera_z_variance",
)
PRIMARY_TENSOR = "alpha_normalized_expected_camera_z"


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def record_hash(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sha(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or any(
        c not in "0123456789abcdef" for c in expected
    ):
        raise ValueError("Expected an explicit lowercase SHA-256")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")
    return actual


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"Non-finite JSON value: {value}")

    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                      object_pairs_hook=pairs, parse_constant=reject_constant)


def safe_member(root, relative):
    rel = PurePosixPath(relative)
    if (not relative or "\\" in relative or ":" in relative or rel.is_absolute()
            or ".." in rel.parts or str(rel) != relative):
        raise ValueError(f"Unsafe or noncanonical relative path: {relative}")
    root = Path(root).resolve()
    target = root.joinpath(*rel.parts)
    if target.resolve() == root or root not in target.resolve().parents:
        raise ValueError(f"Path escapes root: {relative}")
    if any(p.is_symlink() for p in [target, *target.parents] if p != root and root in p.parents):
        raise ValueError(f"Symlink input member: {relative}")
    return target


def verify_source_snapshot(root, expected_manifest_sha):
    root = Path(root)
    manifest = root / "REFERENCE_MANIFEST.json"
    verify_sha(manifest, expected_manifest_sha)
    data = read_json(manifest)
    if data["schema"] != "gs_gcp_evaluation_reference_source_bundle_v1":
        raise ValueError("Unknown reference snapshot schema")
    listed = set()
    for row in data["files"]:
        rel = row["path"]
        if rel in listed or rel == manifest.name:
            raise ValueError("Duplicate or self-referencing source manifest")
        listed.add(rel)
        path = safe_member(root, rel)
        verify_sha(path, row["sha256"])
        if path.stat().st_size != row["bytes"]:
            raise ValueError(f"Source size mismatch: {rel}")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and "__pycache__" not in p.parts}
    if actual != listed | {manifest.name}:
        raise ValueError("Unregistered or missing reference source")
    return {"status": "PASS", "file_count": len(listed),
            "manifest_sha256": expected_manifest_sha,
            "source_git_head": data["source_git_head"]}


def read_npz_headers(path):
    """Read only NPY headers; array bodies are neither decoded nor sampled."""
    result = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            name = member.filename
            if (not name.endswith(".npy") or "/" in name or "\\" in name
                    or name[:-4] in result or member.flag_bits & 1):
                raise ValueError(f"Invalid/duplicate/encrypted NPZ member: {name}")
            with archive.open(member) as stream:
                if stream.read(6) != b"\x93NUMPY":
                    raise ValueError("Invalid NPY magic")
                version = tuple(stream.read(2))
                if version not in {(1, 0), (2, 0), (3, 0)}:
                    raise ValueError("Unsupported NPY version")
                nbytes = 2 if version == (1, 0) else 4
                raw_length = stream.read(nbytes)
                if len(raw_length) != nbytes:
                    raise ValueError("Truncated NPY header")
                length = struct.unpack("<H" if nbytes == 2 else "<I", raw_length)[0]
                if not 0 < length <= 65536:
                    raise ValueError("Unsafe NPY header length")
                raw = stream.read(length)
                if len(raw) != length:
                    raise ValueError("Truncated NPY header")
                header = ast.literal_eval(raw.decode("utf-8" if version == (3, 0) else "latin1"))
                if set(header) != {"descr", "fortran_order", "shape"}:
                    raise ValueError("Unexpected NPY header fields")
                dtype = np.dtype(header["descr"])
                shape = header["shape"]
                if (dtype.hasobject or dtype.fields or dtype.subdtype
                        or not isinstance(shape, tuple) or len(shape) != 2
                        or any(type(n) is not int or n <= 0 for n in shape)
                        or header["fortran_order"] is not False):
                    raise ValueError("Packet tensors must be numeric, C-order, two-dimensional")
                header_bytes = 8 + nbytes + length
                if member.file_size != header_bytes + math.prod(shape) * dtype.itemsize:
                    raise ValueError("NPY body size disagrees with header")
                result[name[:-4]] = {"dtype": dtype.str, "shape": list(shape),
                                     "header_bytes_read": header_bytes,
                                     "member_bytes": member.file_size}
    return result


def validate_packet_headers(path, *, expected_sha, contract, width, height):
    """Metadata qualification only. This cannot establish packet/ref numeric parity."""
    required = {
        "schema": PACKET_SCHEMA, "version": 2, "primary_tensor": PRIMARY_TENSOR,
        "semantics": "camera_z", "formula": "M1/A",
    }
    for key, value in required.items():
        if contract.get(key) != value or type(contract.get(key)) is not type(value):
            raise ValueError(f"Packet contract mismatch: {key}")
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError("Invalid packet dimensions")
    verify_sha(path, expected_sha)
    headers = read_npz_headers(path)
    tensor_dtypes = {name: "<f4" for name in FLOAT_TENSORS}
    tensor_dtypes["metric_depth_valid_mask"] = "|b1"
    if contract.get("required_tensor_dtypes") != tensor_dtypes:
        raise ValueError("Required tensor declaration mismatch")
    for name, dtype in tensor_dtypes.items():
        if name not in headers:
            raise ValueError(f"Missing required tensor: {name}")
        if headers[name]["dtype"] != dtype or headers[name]["shape"] != [height, width]:
            raise ValueError(f"Tensor dtype/shape mismatch: {name}")
    # No legacy H synthesis and no promotion from header validity to formal readiness.
    return {"status": "PASS_HEADERS_ONLY", "headers": headers,
            "tensor_values_loaded": False, "packet_ref_numeric_validation": "NOT_RUN",
            "formal_ready": False}
