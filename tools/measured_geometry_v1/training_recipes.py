"""Reviewed MS-family command construction, never execution or qualification.

The shared-camera comparison is not an author-exact original-dataset replay.
No command here changes the frozen native input resolution or chooses a score.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import PurePosixPath

from .contracts import record_hash

UPSTREAM_COMMIT = "3723dd7ca9c8134c416d7574b12144a61928ee00"
CHANNELS = ["D", "MS_G", "MS_R", "MS_RE", "MS_NIR"]
SPECTRAL = CHANNELS[1:]


def recipe(method):
    if method not in {"jo", "sig_mechanism", "ms_splatting_neural"}:
        raise ValueError("Unknown reviewed MS-family method")
    model = {
        "refine_every": 300, "densification_strategy": "max_average",
        "densify_grad_thresh": .0008, "stop_split_at": 60000,
        "pos_optim_delay_channels": ["D", "500", *SPECTRAL, "32000"],
        "opacity_optim_delay_channels": [*SPECTRAL, "32000"],
        "use_sh_channels": CHANNELS.copy(), "use_mlp_channels": [], "sh_degree": 3,
        "densification_pause_iterations": [29000, 32001],
        "camera_optimizer_rgb": {"mode": "off"}, "camera_optimizer_ms": {"mode": "off"},
        "ssim_lambda": .2, "use_scale_regularization": True,
        "opacity_correction_flag": False,
        "use_neighbouring_features": False, "use_cosine_features": False,
    }
    manager = {"delay_channels": [*SPECTRAL, "30000"], "channel_oversampling": [],
               "channel_size": ["D", "3", *SPECTRAL, "1"], "equal_channel_sampling": True,
               "camera_sampling_seed": 42, "cache_images": "cpu", "camera_res_scale_factor": 1.0}
    notes = ["Fixed shared SfM/split/native pixels; not the authors' independent-band calibration.",
             "Remaining unspecified model/optimizer defaults are bound to the upstream commit."]
    if method == "sig_mechanism":
        model["pos_optim_delay_channels"] = ["D", "500", *SPECTRAL, "-1"]
        model["opacity_optim_delay_channels"] = [*SPECTRAL, "-1"]
        notes.append("Isolates direct non-RGB photometric structure gradients only. Scale regularization, optimizer state and MS densification may still change structure/support.")
    if method == "ms_splatting_neural":
        model.update({
            "stop_split_at": 50000, "use_sh_channels": [], "use_mlp_channels": CHANNELS.copy(),
            "pos_optim_delay_channels": [*CHANNELS, "500"],
            "opacity_optim_delay_channels": [*CHANNELS, "500"],
            "densification_pause_iterations": [], "mlp_type": "standard",
            "feature_input_dim": 8, "mlp_hidden_depth": 32, "mlp_hidden_layers": 1,
            "mlp_hidden_activation_fn": "ELU", "direction_encoding_flag": True,
            "direction_encoding_num_frequencies": 0, "positional_encoding_flag": False,
            "use_feature_norm_regularization": True, "lambda_norm": .1,
            "use_neighbouring_features": False, "use_cosine_features": False,
            "use_smoothness_loss": False,
        })
        manager.update(delay_channels=[], channel_oversampling=["D", "4"])
        notes.extend([
            "Effective decoder 11->32->32->7, 1671 parameters, three affine layers. Source layer counting is retained; no architecture sweep.",
            "Direction is the upstream normalized (mean-camera) 3-vector, no Fourier frequencies or position input.",
            "stop_split_at=50000 and scale regularization are disclosed upstream defaults, not recovered author argv.",
        ])
    result = {"schema": "umgs_tgrs_ms_family_recipe_v1", "method": method,
              "upstream_commit": UPSTREAM_COMMIT, "seed": 42, "max_num_iterations": 120000,
              "pipeline": {"model": model, "datamanager": manager},
              "dataparser": {"downscale_factor": 0, "eval_mode": "json-list",
                             "orientation_method": "up", "center_method": "poses", "scale_factor": 1.0},
              "notes": notes, "gpu_launch_authorized": False,
              "historical_joint_config_recovered": method == "jo",
              "author_exact_command_recovered": False}
    return deepcopy(result)


def _path(value):
    if (not isinstance(value, str) or not value.startswith("/") or "\\" in value
            or "\0" in value or any(c in value for c in "\r\n")
            or ".." in PurePosixPath(value).parts or str(PurePosixPath(value)) != value):
        raise ValueError("Require a canonical absolute POSIX path")
    return value


def command(method, *, entrypoint, data, output, split):
    """Return an argument vector. No shell interpolation or free CLI overrides."""
    spec = recipe(method)
    for value in (entrypoint, data, output, split):
        _path(value)
    d, o = PurePosixPath(data), PurePosixPath(output)
    if d == o or d in o.parents or o in d.parents:
        raise ValueError("Input and output trees must be separate")
    argv = [entrypoint, "mmsplat", "--data", data, "--output-dir", output,
            "--machine.seed", "42", "--max-num-iterations", "120000",
            "--steps-per-save", "60000", "--steps-per-eval-image", "0",
            "--steps-per-eval-batch", "0", "--steps-per-eval-all-images", "0",
            "--vis", "tensorboard"]

    def append(prefix, fields):
        for key, value in fields.items():
            flag = prefix + key.replace("_", "-")
            if isinstance(value, dict):
                append(flag + ".", value)
            else:
                argv.append(flag)
                values = value if isinstance(value, list) else [value]
                argv.extend(str(v) for v in values)

    append("--pipeline.model.", spec["pipeline"]["model"])
    append("--pipeline.datamanager.", spec["pipeline"]["datamanager"])
    # Parser-specific options must follow the tyro subcommand.
    argv.append("mmsplat-dataparser")
    append("--", spec["dataparser"])
    argv.extend(["--json-list-path", split])
    return {"argv": argv, "recipe": spec, "recipe_sha256": record_hash(spec),
            "execution_performed": False, "resolved_cli_verification": "PENDING"}


def validate_resolved(config, expected_recipe):
    """Check scientific settings in a JSON-safe resolved config, including types."""
    if expected_recipe != recipe(expected_recipe["method"]):
        raise ValueError("Recipe differs from the reviewed fixed definition")
    expected = {"machine": {"seed": expected_recipe["seed"]},
                "max_num_iterations": expected_recipe["max_num_iterations"],
                "steps_per_eval_image": 0, "steps_per_eval_batch": 0, "steps_per_eval_all_images": 0,
                "pipeline": deepcopy(expected_recipe["pipeline"])}
    expected["pipeline"]["datamanager"]["dataparser"] = expected_recipe["dataparser"]
    checks = []

    def check(actual, wanted, path):
        if isinstance(wanted, dict):
            if not isinstance(actual, dict):
                raise ValueError("Resolved configuration mismatch: " + path)
            for key, value in wanted.items():
                check(actual.get(key), value, path + "." + key)
        else:
            if type(actual) is not type(wanted) or actual != wanted:
                raise ValueError("Resolved configuration mismatch: " + path)
            checks.append(path.lstrip("."))

    check(config, expected, "")
    return {"status": "PASS_RESOLVED_RECIPE_ONLY", "checked_fields": checks,
            "recipe_sha256": record_hash(expected_recipe), "gpu_launch_authorized": False}
