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

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
import cv2
from utils.spectral_image_utils import LoadedSpectralImage, normalize_scalar_band_image, replicate_single_band_to_rgb

class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image, invdepthmap,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False,
                 modality_type = "rgb", band_name = "", carrier_mode = "native_rgb",
                 image_metadata = None, input_dynamic_range = "uint8", radiometric_mode = "raw_dn",
                 single_band_mode = False, single_band_replicate_to_rgb = None,
                 validity_mask = None, scene_kind = "", rectification_status = ""
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.source_modality = modality_type or "rgb"
        self.band_name = band_name or ""
        self.carrier_mode = carrier_mode or "native_rgb"
        self.image_metadata = dict(image_metadata or {})
        self.scene_kind = scene_kind or ""
        self.rectification_status = rectification_status or ""
        self.original_scalar_image = None
        self.validity_mask = None
        self.validity_mask_source = "none"

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.alpha_mask = None
        if self.source_modality == "scalar_band" or isinstance(image, LoadedSpectralImage):
            if not isinstance(image, LoadedSpectralImage):
                raise TypeError("scalar-band camera path expects LoadedSpectralImage input.")
            replicate_to_rgb = True if single_band_replicate_to_rgb is None else bool(single_band_replicate_to_rgb)
            if not replicate_to_rgb:
                raise ValueError("single_band_replicate_to_rgb=false is not supported by the legacy 3-channel renderer.")

            merged_meta = dict(image.metadata or {})
            merged_meta.update(self.image_metadata)
            scalar_image = normalize_scalar_band_image(
                image=image,
                metadata=merged_meta,
                mode=radiometric_mode,
                dynamic_range=input_dynamic_range,
            )
            scalar_image = cv2.resize(scalar_image, resolution, interpolation=cv2.INTER_LINEAR)
            scalar_image = np.ascontiguousarray(scalar_image.astype(np.float32))
            scalar_tensor = torch.from_numpy(scalar_image[None, ...])
            self.original_scalar_image = scalar_tensor.clamp(0.0, 1.0).to(self.data_device)
            gt_rgb = replicate_single_band_to_rgb(scalar_image)
            gt_image = torch.from_numpy(np.ascontiguousarray(gt_rgb)).permute(2, 0, 1)
            self.alpha_mask = torch.ones_like(gt_image[0:1, ...].to(self.data_device))
            self.carrier_mode = "replicated_scalar_rgb"
        else:
            resized_image_rgb = PILtoTorch(image, resolution)
            gt_image = resized_image_rgb[:3, ...]
            if resized_image_rgb.shape[0] == 4:
                self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
            else: 
                self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))
            self.carrier_mode = "native_rgb"

        if train_test_exp and is_test_view:
            if is_test_dataset:
                self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
            else:
                self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0

        if validity_mask is not None:
            mask_tensor = torch.as_tensor(validity_mask, dtype=torch.float32)
            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
            if mask_tensor.ndim != 3 or mask_tensor.shape[0] != 1:
                raise ValueError(f"validity_mask must have shape [1,H,W] or [H,W], got {tuple(mask_tensor.shape)}")
            self.validity_mask = mask_tensor.clamp(0.0, 1.0).to(self.data_device)
            if self.alpha_mask is not None:
                self.validity_mask = self.validity_mask * self.alpha_mask
            self.validity_mask_source = "external"
        elif self.alpha_mask is not None:
            self.validity_mask = self.alpha_mask.clone()
            self.validity_mask_source = "alpha_fallback"

        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None:
            self.depth_mask = torch.ones_like(self.alpha_mask)
            self.invdepthmap = cv2.resize(invdepthmap, resolution)
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True

            if depth_params is not None:
                if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                    self.depth_reliable = False
                    self.depth_mask *= 0
                
                if depth_params["scale"] > 0:
                    self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
