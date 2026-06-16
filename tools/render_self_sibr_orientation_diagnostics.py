import json
import math
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_renderer import GaussianModel, render
from scene.cameras import MiniCam
from utils.graphics_utils import getProjectionMatrix


CAMERA_PATH = Path("/root/autodl-tmp/umgs_runs/visualization/self_sibr_camera.bin")
OUT_DIR = Path("/root/autodl-tmp/umgs_runs/visualization/self_sibr_orientation_diagnostics_20260508")
MODEL_PLY = "/root/autodl-tmp/runs/paper_autodl_full_20260429/e3_raw_self_clean_cost_20260501_230446/raw_self/out/Model_G/point_cloud/iteration_60000/point_cloud.ply"
STRICT_TO_NATIVE = np.array(
    [
        [-0.8447658030599412, -0.009008857058253912, 0.07036153491215513, 0.6576625559061127],
        [0.07049842279068755, -0.19986974028905793, 0.8208186318676058, -2.7688087801079546],
        [0.007866224298603995, 0.8237912930660815, 0.1999179709380715, 2.2706296640630197],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def decode_sibr_camera(path: Path):
    b = path.read_bytes()
    o = 1
    focal, _k1, _k2 = struct.unpack_from(">fff", b, o)
    o += 12
    width, height = struct.unpack_from(">HH", b, o)
    o += 4
    pos = np.array(struct.unpack_from(">fff", b, o), dtype=np.float64)
    o += 12
    qw, qx, qy, qz = struct.unpack_from(">ffff", b, o)
    o += 16
    fovy, aspect, znear, zfar = struct.unpack_from(">ffff", b, o)
    r = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )
    fovx = 2.0 * math.atan(width / (2.0 * focal))
    return {
        "width": int(width),
        "height": int(height),
        "fovy": float(fovy),
        "fovx": float(fovx),
        "znear": float(znear),
        "zfar": float(zfar),
        "position": pos,
        "rotation_matrix_from_quat": r,
        "focal": float(focal),
        "aspect": float(aspect),
    }


def transform_gaussians_to_reference(gaussians):
    native_to_strict = np.linalg.inv(STRICT_TO_NATIVE)
    linear = native_to_strict[:3, :3]
    trans = native_to_strict[:3, 3]
    scale = float(abs(np.linalg.det(linear)) ** (1.0 / 3.0))
    xyz = gaussians._xyz.detach()
    l = torch.tensor(linear.T, dtype=xyz.dtype, device=xyz.device)
    t = torch.tensor(trans, dtype=xyz.dtype, device=xyz.device)
    gaussians._xyz = torch.nn.Parameter(xyz @ l + t, requires_grad=False)
    gaussians._scaling = torch.nn.Parameter(gaussians._scaling.detach() + math.log(scale), requires_grad=False)


def make_w2c(cam, variant):
    r = cam["rotation_matrix_from_quat"]
    c = cam["position"]
    if variant["transpose"]:
        m0 = r.T
    else:
        m0 = r
    m = np.diag(variant["axis"]) @ m0
    t = -m @ c
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = m.astype(np.float32)
    w2c[:3, 3] = t.astype(np.float32)
    return w2c


def make_camera(cam, w2c):
    world_view = torch.tensor(w2c, dtype=torch.float32, device="cuda").transpose(0, 1)
    proj = getProjectionMatrix(cam["znear"], cam["zfar"], cam["fovx"], cam["fovy"]).transpose(0, 1).cuda()
    full = world_view.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    return MiniCam(cam["width"], cam["height"], cam["fovy"], cam["fovx"], cam["znear"], cam["zfar"], world_view, full)


def render_variant(gaussians, cam, variant):
    camera = make_camera(cam, make_w2c(cam, variant))
    bg = torch.zeros(3, dtype=torch.float32, device="cuda")
    pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False, antialiasing=False)
    with torch.no_grad():
        pkg = render(camera, gaussians, pipe, bg, separate_sh=False)
        image = pkg["render"].clamp(0.0, 1.0)
        visible = int((pkg["radii"] > 0).sum().item())
    out = OUT_DIR / f"{variant['name']}.png"
    torchvision.utils.save_image(image, out)
    return str(out), visible


def make_contact_sheet(entries):
    thumbs = []
    for name, path, visible in entries:
        im = Image.open(path).convert("RGB")
        tw = 480
        th = int(round(im.height * tw / im.width))
        im = im.resize((tw, th), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(im)
        draw.rectangle([0, 0, tw, 32], fill=(0, 0, 0))
        draw.text((8, 9), f"{name} | vis {visible:,}", fill=(255, 255, 255))
        thumbs.append(im)
    cols = 2
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * thumbs[0].width, rows * thumbs[0].height), (20, 20, 20))
    for i, im in enumerate(thumbs):
        sheet.paste(im, ((i % cols) * im.width, (i // cols) * im.height))
    path = OUT_DIR / "orientation_variants_contact_sheet.png"
    sheet.save(path)
    return str(path)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cam = decode_sibr_camera(CAMERA_PATH)
    variants = [
        {"name": "A_Rt_axis_1_1_neg1", "transpose": True, "axis": [1.0, 1.0, -1.0]},
        {"name": "B_Rt_axis_neg1_neg1_1_previous_corrected", "transpose": True, "axis": [-1.0, -1.0, 1.0]},
        {"name": "C_Rt_axis_neg1_1_neg1_hflip_from_A", "transpose": True, "axis": [-1.0, 1.0, -1.0]},
        {"name": "D_Rt_axis_1_neg1_neg1_vflip_from_A", "transpose": True, "axis": [1.0, -1.0, -1.0]},
        {"name": "E_R_axis_1_1_neg1", "transpose": False, "axis": [1.0, 1.0, -1.0]},
        {"name": "F_R_axis_neg1_neg1_1", "transpose": False, "axis": [-1.0, -1.0, 1.0]},
        {"name": "G_R_axis_neg1_1_neg1", "transpose": False, "axis": [-1.0, 1.0, -1.0]},
        {"name": "H_R_axis_1_neg1_neg1", "transpose": False, "axis": [1.0, -1.0, -1.0]},
    ]
    gaussians = GaussianModel(3)
    gaussians.load_ply(MODEL_PLY)
    transform_gaussians_to_reference(gaussians)
    entries = []
    manifest_entries = []
    for variant in variants:
        print("[variant]", variant["name"], flush=True)
        path, visible = render_variant(gaussians, cam, variant)
        entries.append((variant["name"], path, visible))
        manifest_entries.append({**variant, "output": path, "visible_gaussians": visible})
    panel = make_contact_sheet(entries)
    manifest = {"camera_path": str(CAMERA_PATH), "panel": panel, "variants": manifest_entries}
    (OUT_DIR / "orientation_diagnostics_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
