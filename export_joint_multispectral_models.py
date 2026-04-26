import argparse
import json
import subprocess
import sys
from pathlib import Path


def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {v}")


def _parse_bands(text: str):
    return [x.strip() for x in str(text).replace(",", " ").split() if x.strip()]


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Export E4b joint multispectral checkpoint into render-compatible per-band models.")
    ap.add_argument("--joint_checkpoint", required=True)
    ap.add_argument("--rectified_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--iteration", type=int, default=60000)
    ap.add_argument("--bands", default="G,R,RE,NIR")
    ap.add_argument("--band_res", type=int, default=8)
    ap.add_argument("--input_dynamic_range", default="uint16", choices=["uint8", "uint16", "float"])
    ap.add_argument("--radiometric_mode", default="exposure_normalized", choices=["raw_dn", "exposure_normalized", "reflectance_ready_stub"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--render_smoke", type=_str2bool, nargs="?", const=True, default=False)
    ap.add_argument("--python_executable", default=sys.executable)
    return ap


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_render_smoke(python_executable: str, model_dir: Path, scene_root: Path, band_res: int) -> tuple[bool, str]:
    cmd = [
        python_executable,
        "render.py",
        "-m",
        str(model_dir),
        "-s",
        str(scene_root),
        "-r",
        str(band_res),
        "--skip_train",
    ]
    try:
        proc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent), text=True, capture_output=True, check=False)
        tail = (proc.stdout + "\n" + proc.stderr)[-4000:]
        return proc.returncode == 0, tail
    except Exception as exc:
        return False, repr(exc)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    from utils.joint_multispectral_utils import (
        compute_export_topology_audit,
        export_band_model,
        load_unified_checkpoint,
    )

    bands = _parse_bands(args.bands)
    if not bands:
        raise ValueError("At least one band is required")

    joint_checkpoint = Path(args.joint_checkpoint).resolve()
    rectified_root = Path(args.rectified_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    payload = load_unified_checkpoint(joint_checkpoint, device=args.device)

    fragments = {}
    for band in bands:
        fragments[band] = export_band_model(
            joint_payload=payload,
            band=band,
            out_model_dir=out_root / f"Model_{band}",
            scene_root=rectified_root / f"{band}_rectified",
            iteration=int(args.iteration),
            resolution=int(args.band_res),
            input_dynamic_range=args.input_dynamic_range,
            radiometric_mode=args.radiometric_mode,
            device=args.device,
        )

    topology_audit = compute_export_topology_audit(
        {band: out_root / f"Model_{band}" for band in bands},
        iteration=int(args.iteration),
    )

    smoke = {"requested": bool(args.render_smoke), "passed": None, "by_band": {}}
    if bool(args.render_smoke):
        all_ok = True
        for band in bands:
            ok, output_tail = _run_render_smoke(
                args.python_executable,
                out_root / f"Model_{band}",
                rectified_root / f"{band}_rectified",
                int(args.band_res),
            )
            smoke["by_band"][band] = {"passed": bool(ok), "output_tail": output_tail}
            all_ok = all_ok and ok
        smoke["passed"] = bool(all_ok)

    audit = {
        "method": "E4b_geometry_locked_joint_multispectral_banks",
        "joint_checkpoint": str(joint_checkpoint),
        "out_root": str(out_root),
        "bands": bands,
        "export_iteration": int(args.iteration),
        "export_iteration_note": "Aligned with E3 band_iter evaluation iteration; joint_total_updates is stored in the unified checkpoint.",
        "exported_band_models": fragments,
        "render_smoke": smoke,
        **topology_audit,
    }
    audit_path = out_root / "Model_MSJoint_exports" / "export_audit.json"
    _write_json(audit_path, audit)
    print(f"[E4b] Export audit: {audit_path}")


if __name__ == "__main__":
    main()
