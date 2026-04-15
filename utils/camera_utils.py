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

from scene.cameras import Camera
import numpy as np
from utils.graphics_utils import fov2focal
from PIL import Image
import cv2
from utils.spectral_image_utils import load_image_preserve_dtype

WARNED = False


def _merge_metadata(primary, secondary):
    out = {}
    if isinstance(primary, dict):
        out.update(primary)
    if isinstance(secondary, dict):
        out.update(secondary)
    return out


def _resolve_camera_modality(args, cam_info):
    modality_type = getattr(cam_info, "modality_type", "") or ""
    if modality_type == "scalar_band":
        source_modality = "scalar_band"
    elif getattr(args, "modality_kind", "rgb") == "band" or bool(getattr(args, "single_band_mode", False)):
        source_modality = "scalar_band"
    else:
        source_modality = "rgb"

    band_name = getattr(cam_info, "band_name", "") or getattr(args, "target_band", "") or ""
    carrier_mode = getattr(cam_info, "carrier_mode", "") or ""
    if not carrier_mode:
        if source_modality == "scalar_band":
            carrier_mode = "replicated_scalar_rgb" if getattr(args, "single_band_replicate_to_rgb", None) is not False else "native_rgb"
        else:
            carrier_mode = "native_rgb"
    return source_modality, band_name, carrier_mode


def _load_validity_mask(mask_path, resolution):
    if not mask_path:
        return None
    path = str(mask_path)
    with Image.open(path) as mask_image:
        mask_image.load()
        mask = np.asarray(mask_image.convert("L"), dtype=np.float32) / 255.0
    mask = cv2.resize(mask, resolution, interpolation=cv2.INTER_NEAREST)
    mask = np.ascontiguousarray(mask[None, ...].astype(np.float32))
    return mask

def loadCam(args, id, cam_info, resolution_scale, is_nerf_synthetic, is_test_dataset):
    source_modality, band_name, carrier_mode = _resolve_camera_modality(args, cam_info)
    image_metadata = getattr(cam_info, "image_meta", {}) or {}
    if source_modality == "scalar_band":
        loaded = load_image_preserve_dtype(cam_info.image_path)
        loaded.metadata = _merge_metadata(loaded.metadata, image_metadata)
        image = loaded
        orig_w, orig_h = loaded.width, loaded.height
    else:
        image = Image.open(cam_info.image_path)
        orig_w, orig_h = image.size

    if cam_info.depth_path != "":
        try:
            if is_nerf_synthetic:
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / 512
            else:
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / float(2**16)

        except FileNotFoundError:
            print(f"Error: The depth file at path '{cam_info.depth_path}' was not found.")
            raise
        except IOError:
            print(f"Error: Unable to open the image file '{cam_info.depth_path}'. It may be corrupted or an unsupported format.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred when trying to read depth at {cam_info.depth_path}: {e}")
            raise
    else:
        invdepthmap = None

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution
    

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    validity_mask = _load_validity_mask(getattr(cam_info, "validity_mask_path", ""), resolution)

    return Camera(resolution, colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, depth_params=cam_info.depth_params,
                  image=image, invdepthmap=invdepthmap,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  train_test_exp=args.train_test_exp, is_test_dataset=is_test_dataset, is_test_view=cam_info.is_test,
                  modality_type=source_modality, band_name=band_name, carrier_mode=carrier_mode,
                  image_metadata=image_metadata, input_dynamic_range=getattr(args, "input_dynamic_range", "uint8"),
                  radiometric_mode=getattr(args, "radiometric_mode", "raw_dn"),
                  single_band_mode=bool(getattr(args, "single_band_mode", False)),
                  single_band_replicate_to_rgb=getattr(args, "single_band_replicate_to_rgb", None),
                  validity_mask=validity_mask)

def cameraList_from_camInfos(cam_infos, resolution_scale, args, is_nerf_synthetic, is_test_dataset):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale, is_nerf_synthetic, is_test_dataset))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
