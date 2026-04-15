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

from argparse import ArgumentParser, Namespace
import sys
import os


def _str2bool(v):
    if isinstance(v, bool) or v is None:
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise ValueError(f"invalid bool: {v}")

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        arg_meta = getattr(self, "_arg_meta", {})
        for key, value in vars(self).items():
            if key == "_arg_meta":
                continue
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            meta = arg_meta.get(key, {})
            kwargs = {}
            if "choices" in meta:
                kwargs["choices"] = meta["choices"]
            if "help" in meta:
                kwargs["help"] = meta["help"]
            if shorthand:
                if meta.get("explicit_bool", False):
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=_str2bool, nargs="?", const=True, **kwargs)
                elif t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true", **kwargs)
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t, **kwargs)
            else:
                if meta.get("explicit_bool", False):
                    group.add_argument("--" + key, default=value, type=_str2bool, nargs="?", const=True, **kwargs)
                elif t == bool:
                    group.add_argument("--" + key, default=value, action="store_true", **kwargs)
                else:
                    group.add_argument("--" + key, default=value, type=t, **kwargs)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False
        self.modality_kind = "rgb"
        self.target_band = ""
        self.single_band_mode = False
        self.single_band_replicate_to_rgb = None
        self.input_dynamic_range = "uint8"
        self.radiometric_mode = "raw_dn"
        self.rectification_config = ""
        self.rectified_root = ""
        self.require_rectified_band_scene = None
        self.use_validity_mask = None
        self.rectification_method = "none"
        self.reset_appearance_features = None
        self.freeze_geometry = None
        self.freeze_opacity = None
        self.tied_scalar_carrier = None
        self.stage2_mode = "none"
        self.rectification_backend = "none"
        self.minima_method = "roma"
        self.minima_root = ""
        self.minima_device = "cuda"
        self.minima_ckpt = ""
        self.rectification_enable_residual_refine = None
        self._arg_meta = {
            "modality_kind": {
                "choices": ["rgb", "band", "thermal"],
                "help": "Input modality kind for the current scene or transfer target.",
            },
            "target_band": {
                "help": "Target scalar band name, e.g. G, R, RE, NIR, or thermal for legacy runs.",
            },
            "single_band_mode": {
                "explicit_bool": True,
                "help": "Treat source images as scalar-band imagery instead of native RGB.",
            },
            "single_band_replicate_to_rgb": {
                "explicit_bool": True,
                "help": "Replicate scalar-band supervision into a 3-channel RGB carrier for legacy renderer compatibility.",
            },
            "input_dynamic_range": {
                "choices": ["uint8", "uint16", "float"],
                "help": "Expected input image dynamic range for loading and normalization.",
            },
            "radiometric_mode": {
                "choices": ["raw_dn", "exposure_normalized", "reflectance_ready_stub"],
                "help": "Radiometric preprocessing mode for scalar-band imagery.",
            },
            "rectification_config": {
                "help": "Optional rectification config JSON used to build or validate rectified band scenes.",
            },
            "rectified_root": {
                "help": "Optional rectified dataset root for band-stage runs.",
            },
            "require_rectified_band_scene": {
                "explicit_bool": True,
                "help": "Require the scene manifest to declare a rectified band dataset before training scalar-band stage-2.",
            },
            "use_validity_mask": {
                "explicit_bool": True,
                "help": "Use rectification validity masks when computing scalar-band photometric loss.",
            },
            "rectification_method": {
                "help": "Rectification method identifier recorded in cfg_args for rectified band runs.",
            },
            "reset_appearance_features": {
                "explicit_bool": True,
                "help": "Reset appearance features after restoring a checkpoint for stage-2 transfer.",
            },
            "freeze_geometry": {
                "explicit_bool": True,
                "help": "Freeze geometry-related parameters during stage-2 transfer.",
            },
            "freeze_opacity": {
                "explicit_bool": True,
                "help": "Freeze opacity during stage-2 transfer to preserve shared Gaussian topology.",
            },
            "tied_scalar_carrier": {
                "explicit_bool": True,
                "help": "Project RGB feature channels to a tied scalar carrier after each optimization step.",
            },
            "stage2_mode": {
                "choices": ["none", "band_transfer"],
                "help": "Optional stage-2 preset. band_transfer enables the shared-geometry scalar-band transfer recipe.",
            },
            "rectification_backend": {
                "choices": ["none", "minima"],
                "help": "Rectification backend identifier used by pipeline metadata and downstream bookkeeping.",
            },
            "minima_method": {
                "choices": ["roma", "xoftr"],
                "help": "MINIMA matcher backend name for rectification runs.",
            },
            "minima_root": {
                "help": "Path to local MINIMA repository root used by the matcher bridge.",
            },
            "minima_device": {
                "help": "Device string for MINIMA inference, e.g. cuda or cpu.",
            },
            "minima_ckpt": {
                "help": "Optional explicit checkpoint for MINIMA backend.",
            },
            "rectification_enable_residual_refine": {
                "explicit_bool": True,
                "help": "Enable optional residual refinement after MINIMA global transform aggregation.",
            },
        }
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        # Backward-compatibility: older cfg_args may miss newly added fields.
        # Ensure ModelParams always has a complete attribute set.
        defaults = {
            "sh_degree": self.sh_degree,
            "source_path": self._source_path,
            "model_path": self._model_path,
            "images": self._images,
            "depths": self._depths,
            "resolution": self._resolution,
            "white_background": self._white_background,
            "train_test_exp": self.train_test_exp,
            "data_device": self.data_device,
            "eval": self.eval,
            "modality_kind": self.modality_kind,
            "target_band": self.target_band,
            "single_band_mode": self.single_band_mode,
            "single_band_replicate_to_rgb": self.single_band_replicate_to_rgb,
            "input_dynamic_range": self.input_dynamic_range,
            "radiometric_mode": self.radiometric_mode,
            "rectification_config": self.rectification_config,
            "rectified_root": self.rectified_root,
            "require_rectified_band_scene": self.require_rectified_band_scene,
            "use_validity_mask": self.use_validity_mask,
            "rectification_method": self.rectification_method,
            "reset_appearance_features": self.reset_appearance_features,
            "freeze_geometry": self.freeze_geometry,
            "freeze_opacity": self.freeze_opacity,
            "tied_scalar_carrier": self.tied_scalar_carrier,
            "stage2_mode": self.stage2_mode,
            "rectification_backend": self.rectification_backend,
            "minima_method": self.minima_method,
            "minima_root": self.minima_root,
            "minima_device": self.minima_device,
            "minima_ckpt": self.minima_ckpt,
            "rectification_enable_residual_refine": self.rectification_enable_residual_refine,
        }
        for k, v in defaults.items():
            if not hasattr(g, k):
                setattr(g, k, v)

        if g.source_path is not None:
            g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.optimizer_type = "default"
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
