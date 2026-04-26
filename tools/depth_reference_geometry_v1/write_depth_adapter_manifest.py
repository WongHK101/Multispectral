from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
        f.write("\n")


def _argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write a depth-reference adapter manifest for Gaussian renderer bundles")
    parser.add_argument("--method_name", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model_path", default="")
    parser.add_argument("--source_path", default="")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--depth_semantics", default="inverse_camera_z_from_renderer")
    parser.add_argument("--opacity_threshold", type=float, default=0.5)
    parser.add_argument("--depth_min", type=float, default=1e-6)
    parser.add_argument("--notes", default="")
    return parser


def main() -> None:
    args = _argparser().parse_args()
    _save_json(
        Path(args.out).resolve(),
        {
            "protocol_name": "reference-depth-based-geometric-evaluation-v1",
            "method_name": str(args.method_name),
            "model_path": str(args.model_path),
            "source_path": str(args.source_path),
            "iteration": int(args.iteration),
            "depth_semantics": str(args.depth_semantics),
            "validity_rule": {
                "mode": "opacity_threshold",
                "opacity_threshold": float(args.opacity_threshold),
                "depth_min": float(args.depth_min),
            },
            "notes": str(args.notes),
        },
    )


if __name__ == "__main__":
    main()
