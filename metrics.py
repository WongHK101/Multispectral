#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, XXXX
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser


def readImages(renders_dir, gt_dir, device, mask_mode="gt_nonzero", mask_threshold=0.0):
    samples = []
    render_names = {x for x in os.listdir(renders_dir)}
    gt_names = {x for x in os.listdir(gt_dir)}
    for fname in sorted(render_names & gt_names):
        render = Image.open(renders_dir / fname).convert("RGB")
        gt = Image.open(gt_dir / fname).convert("RGB")
        render_t = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].to(device)
        gt_t = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].to(device)

        if mask_mode == "none":
            mask = torch.ones((1, 1, gt_t.shape[-2], gt_t.shape[-1]), device=device, dtype=gt_t.dtype)
        else:
            # Default masked protocol: remove black-border / invalid pixels from GT.
            gt_gray = gt_t.mean(dim=1, keepdim=True)
            mask = (gt_gray > mask_threshold).to(gt_t.dtype)
            if float(mask.sum().item()) <= 0:
                mask = torch.ones_like(mask)

        samples.append(
            {
                "name": fname,
                "render": render_t,
                "gt": gt_t,
                "mask": mask,
                "coverage": float(mask.mean().item()),
            }
        )
    return samples


def masked_psnr(render, gt, mask):
    mask3 = mask.expand(-1, render.shape[1], -1, -1)
    diff2 = ((render - gt) ** 2) * mask3
    denom = torch.clamp(mask.sum() * render.shape[1], min=1.0)
    mse = diff2.sum() / denom
    return (-10.0 * torch.log10(torch.clamp(mse, min=1e-12))).item()


def evaluate(model_paths, mask_mode="gt_nonzero", mask_threshold=0.0):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                samples = readImages(
                    renders_dir,
                    gt_dir,
                    device=device,
                    mask_mode=mask_mode,
                    mask_threshold=mask_threshold,
                )

                ssims = []
                psnrs = []
                lpipss = []
                coverages = []

                for idx in tqdm(range(len(samples)), desc="Metric evaluation progress"):
                    render = samples[idx]["render"]
                    gt = samples[idx]["gt"]
                    mask = samples[idx]["mask"]
                    mask3 = mask.expand(-1, render.shape[1], -1, -1)

                    if mask_mode == "none":
                        ssim_v = ssim(render, gt)
                        psnr_v = psnr(render, gt).item()
                        lpips_v = lpips(render, gt, net_type="vgg").item()
                    else:
                        masked_render = render * mask3
                        masked_gt = gt * mask3
                        ssim_v = ssim(masked_render, masked_gt)
                        psnr_v = masked_psnr(render, gt, mask)
                        lpips_v = lpips(masked_render, masked_gt, net_type="vgg").item()

                    ssims.append(float(ssim_v.item() if hasattr(ssim_v, "item") else ssim_v))
                    psnrs.append(float(psnr_v))
                    lpipss.append(float(lpips_v))
                    coverages.append(float(samples[idx]["coverage"]))

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                if coverages:
                    print("  MASK COVERAGE (mean/median): {:>12.7f} / {:>12.7f}".format(
                        torch.tensor(coverages).mean(), torch.tensor(coverages).median()
                    ))
                print("")

                image_names = [s["name"] for s in samples]
                full_dict[scene_dir][method].update(
                    {
                        "SSIM": torch.tensor(ssims).mean().item(),
                        "PSNR": torch.tensor(psnrs).mean().item(),
                        "LPIPS": torch.tensor(lpipss).mean().item(),
                        "MASK_MODE": mask_mode,
                        "MASK_THRESHOLD": float(mask_threshold),
                        "MASK_COVERAGE_MEAN": torch.tensor(coverages).mean().item() if coverages else 1.0,
                        "MASK_COVERAGE_MEDIAN": torch.tensor(coverages).median().item() if coverages else 1.0,
                    }
                )
                per_view_dict[scene_dir][method].update(
                    {
                        "SSIM": {name: ssim_val for ssim_val, name in zip(torch.tensor(ssims).tolist(), image_names)},
                        "PSNR": {name: psnr_val for psnr_val, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                        "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                        "COVERAGE": {name: cov for cov, name in zip(coverages, image_names)},
                    }
                )

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument(
        "--mask_mode",
        type=str,
        default="gt_nonzero",
        choices=["gt_nonzero", "none"],
        help="Metric masking mode. Default 'gt_nonzero' ignores black-border/invalid GT regions.",
    )
    parser.add_argument(
        "--mask_threshold",
        type=float,
        default=0.0,
        help="Threshold in [0,1] for gt_nonzero mask generation on GT grayscale.",
    )
    args = parser.parse_args()
    evaluate(args.model_paths, mask_mode=args.mask_mode, mask_threshold=args.mask_threshold)
