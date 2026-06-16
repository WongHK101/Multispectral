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
OUT_DIR = Path("/root/autodl-tmp/umgs_runs/visualization/self_same_sibr_camera_splats_20260508")
SIBR_TO_3DGS_AXIS = np.diag([1.0, -1.0, -1.0])

TARGET_STRICT_TO_NATIVE = [
    [-0.8447658030599412, -0.009008857058253912, 0.07036153491215513, 0.6576625559061127],
    [0.07049842279068755, -0.19986974028905793, 0.8208186318676058, -2.7688087801079546],
    [0.007866224298603995, 0.8237912930660815, 0.1999179709380715, 2.2706296640630197],
    [0.0, 0.0, 0.0, 1.0],
]


MODELS = [
    {
        "label": "UMGS-I_E3_clean",
        "source": "/root/autodl-tmp/runs/paper_autodl_full_20260429/e3_raw_self_clean_cost_20260501_230446/raw_self/out/Model_G/point_cloud/iteration_60000/point_cloud.ply",
        "strict_to_native": [
            [-0.8447658030599412, -0.009008857058253912, 0.07036153491215513, 0.6576625559061127],
            [0.07049842279068755, -0.19986974028905793, 0.8208186318676058, -2.7688087801079546],
            [0.007866224298603995, 0.8237912930660815, 0.1999179709380715, 2.2706296640630197],
            [0.0, 0.0, 0.0, 1.0],
        ],
    },
    {
        "label": "UMGS-J_E4b_zero_clean",
        "source": "/root/autodl-tmp/runs/paper_autodl_full_20260429/e4b_zero_clean_raw_self_20260502_105052/raw_self/out/Model_G/point_cloud/iteration_60000/point_cloud.ply",
        "strict_to_native": [
            [-0.8447658030599412, -0.009008857058253912, 0.07036153491215513, 0.6576625559061127],
            [0.07049842279068755, -0.19986974028905793, 0.8208186318676058, -2.7688087801079546],
            [0.007866224298603995, 0.8237912930660815, 0.1999179709380715, 2.2706296640630197],
            [0.0, 0.0, 0.0, 1.0],
        ],
    },
    {
        "label": "FromScratch_clean",
        "source": "/root/autodl-tmp/runs/paper_autodl_full_20260429/post_e3self_mms_ablation_retained_20260502_000018/ablations/fromscratch_raw_self/out/Model_G/point_cloud/iteration_60000/point_cloud.ply",
        "strict_to_native": [
            [-0.8447658030599412, -0.009008857058253912, 0.07036153491215513, 0.6576625559061127],
            [0.07049842279068755, -0.19986974028905793, 0.8208186318676058, -2.7688087801079546],
            [0.007866224298603995, 0.8237912930660815, 0.1999179709380715, 2.2706296640630197],
            [0.0, 0.0, 0.0, 1.0],
        ],
    },
    {
        "label": "GeometryUnfrozen_clean",
        "source": "/root/autodl-tmp/runs/paper_autodl_full_20260429/post_e3self_mms_ablation_retained_20260502_000018/ablations/geom_unfrozen_raw_self/out/Model_G/point_cloud/iteration_60000/point_cloud.ply",
        "strict_to_native": [
            [-0.8447658030599412, -0.009008857058253912, 0.07036153491215513, 0.6576625559061127],
            [0.07049842279068755, -0.19986974028905793, 0.8208186318676058, -2.7688087801079546],
            [0.007866224298603995, 0.8237912930660815, 0.1999179709380715, 2.2706296640630197],
            [0.0, 0.0, 0.0, 1.0],
        ],
    },
    {
        "label": "MMS_retained_self",
        "source": "/root/autodl-tmp/umgs_runs/sibr_view_packages/mms_retained_self_20260508/MMS_retained_self_Model_G_sibr/point_cloud/iteration_30000/point_cloud.ply",
        "strict_to_native": [
            [-0.8489844541708844, 0.019383818900909407, 0.048318318202503915, 1.3871780273436813],
            [-0.00043034486962409115, 0.7867844341446671, -0.32319508771620487, 4.135721140669586],
            [-0.05205964956081496, -0.3226135730024526, -0.7852994786290973, 1.091892801877765],
            [0.0, 0.0, 0.0, 1.0],
        ],
    },
]


def decode_sibr_camera(path: Path):
    b = path.read_bytes()
    if len(b) != 61:
        raise ValueError(f"Unexpected SIBR camera byte length: {len(b)}")
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

    # SIBR stores a camera-to-world quaternion in the native model coordinate
    # system used by the viewer. Convert to the 3DGS rasterizer's
    # world-to-camera convention. The axis signs were selected by matching the
    # native RGB orientation diagnostic sheet against the user-saved SIBR
    # overview screenshot.
    r_c2w = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )
    m = SIBR_TO_3DGS_AXIS @ r_c2w.T
    t = -m @ pos
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = m.astype(np.float32)
    w2c[:3, 3] = t.astype(np.float32)

    fovx = 2.0 * math.atan(width / (2.0 * focal))
    return {
        "width": int(width),
        "height": int(height),
        "fovy": float(fovy),
        "fovx": float(fovx),
        "znear": float(znear),
        "zfar": float(zfar),
        "w2c": w2c,
        "position": pos.tolist(),
        "focal": float(focal),
        "aspect": float(aspect),
    }


def transform_gaussians_to_target_native(gaussians: GaussianModel, strict_to_native):
    """Map a method's native coordinates into the E3 RGB native viewer frame.

    The SIBR camera file was saved while viewing the E3/E4 shared RGB model,
    so it is expressed in that native coordinate frame. Each method has a
    strict/reference-to-native similarity transform recorded in the
    CloudCompare package. We use the strict/reference frame only as a bridge:

        p_target_native = T_target_strict_to_native * inv(T_source_strict_to_native) * p_source_native
    """
    source = np.asarray(strict_to_native, dtype=np.float64)
    target = np.asarray(TARGET_STRICT_TO_NATIVE, dtype=np.float64)
    source_to_target = target @ np.linalg.inv(source)
    linear = source_to_target[:3, :3]
    trans = source_to_target[:3, 3]
    scale = float(abs(np.linalg.det(linear)) ** (1.0 / 3.0))

    xyz = gaussians._xyz.detach()
    l = torch.tensor(linear.T, dtype=xyz.dtype, device=xyz.device)
    t = torch.tensor(trans, dtype=xyz.dtype, device=xyz.device)
    gaussians._xyz = torch.nn.Parameter(xyz @ l + t, requires_grad=False)

    # The similarity scale must also be applied to Gaussian radii.
    gaussians._scaling = torch.nn.Parameter(
        gaussians._scaling.detach() + math.log(scale),
        requires_grad=False,
    )
    return {
        "source_native_to_target_native_scale": scale,
        "source_native_to_target_native_transform": source_to_target.tolist(),
    }


def make_camera(cam):
    world_view = torch.tensor(cam["w2c"], dtype=torch.float32, device="cuda").transpose(0, 1)
    proj = getProjectionMatrix(cam["znear"], cam["zfar"], cam["fovx"], cam["fovy"]).transpose(0, 1).cuda()
    full = world_view.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    return MiniCam(cam["width"], cam["height"], cam["fovy"], cam["fovx"], cam["znear"], cam["zfar"], world_view, full)


def render_one(item, cam):
    gaussians = GaussianModel(3)
    gaussians.load_ply(item["source"])
    align_meta = transform_gaussians_to_target_native(gaussians, item["strict_to_native"])
    camera = make_camera(cam)
    bg = torch.zeros(3, dtype=torch.float32, device="cuda")
    pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False, antialiasing=False)
    with torch.no_grad():
        render_pkg = render(camera, gaussians, pipe, bg, separate_sh=False)
        out = render_pkg["render"].clamp(0.0, 1.0)
    out_path = OUT_DIR / f"{item['label']}_splat_same_sibr_camera.png"
    torchvision.utils.save_image(out, out_path)
    radii_count = int((render_pkg["radii"] > 0).sum().item())
    del gaussians
    torch.cuda.empty_cache()
    return {
        "label": item["label"],
        "source": item["source"],
        "output": str(out_path),
        "visible_gaussians": radii_count,
        **align_meta,
    }


def make_contact_sheet(image_paths):
    thumbs = []
    for label, path in image_paths:
        im = Image.open(path).convert("RGB")
        tw = 600
        th = int(round(im.height * tw / im.width))
        im = im.resize((tw, th), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(im)
        draw.rectangle([0, 0, min(360, tw), 28], fill=(0, 0, 0))
        draw.text((8, 7), label, fill=(255, 255, 255))
        thumbs.append(im)
    cols = 2
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * thumbs[0].width, rows * thumbs[0].height), (20, 20, 20))
    for i, im in enumerate(thumbs):
        sheet.paste(im, ((i % cols) * im.width, (i // cols) * im.height))
    path = OUT_DIR / "all_methods_splat_same_sibr_camera_contact_sheet.png"
    sheet.save(path)
    return str(path)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cam = decode_sibr_camera(CAMERA_PATH)
    results = []
    image_paths = []
    for item in MODELS:
        print(f"[render] {item['label']}", flush=True)
        result = render_one(item, cam)
        results.append(result)
        image_paths.append((item["label"], result["output"]))
    panel = make_contact_sheet(image_paths)
    manifest = {
        "camera_path": str(CAMERA_PATH),
        "camera": {k: v for k, v in cam.items() if k != "w2c"},
        "note": "True Gaussian splat rasterization. The decoded SIBR camera is in the E3/E4 shared RGB native viewer frame. Method-native Gaussian models are transformed into that target native frame via the recorded strict/reference similarity transforms before rendering.",
        "contact_sheet": panel,
        "renders": results,
    }
    (OUT_DIR / "splat_render_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
