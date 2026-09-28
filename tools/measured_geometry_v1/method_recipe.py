"""Inspect resolved MS-Splatting configuration without importing the method.

Results identify configured mechanisms, not paper reproduction or permission
to launch. Callers must bind the full recipe and its original file hash.
"""

from __future__ import annotations


def channel_delays(tokens, channels):
    if not isinstance(tokens, list) or len(set(channels)) != len(channels):
        raise ValueError("Invalid channel/delay declaration")
    result = {name: 0 for name in channels}
    pending, assigned = [], set()
    for token in tokens:
        text = str(token)
        if text in channels:
            if text in pending or text in assigned:
                raise ValueError("Duplicate delay channel")
            pending.append(text)
        else:
            try:
                value = int(text)
            except ValueError as exc:
                raise ValueError(f"Unknown channel or integer delay: {text}") from exc
            if not pending or str(value) != text:
                raise ValueError("Delay requires a preceding channel group and an integer")
            for name in pending:
                result[name] = value
                assigned.add(name)
            pending = []
    if pending:
        raise ValueError("Delay channel group has no value")
    return result


def inspect_resolved_recipe(config, *, channels, rgb_channel):
    if not isinstance(channels, list) or rgb_channel not in channels or len(channels) < 2:
        raise ValueError("Explicit RGB and spectral channel identities required")
    model = config["pipeline"]["model"]
    iterations = config["max_num_iterations"]
    if type(iterations) is not int or iterations <= 0:
        raise ValueError("Invalid iteration count")
    for key in ("camera_optimizer_rgb", "camera_optimizer_ms"):
        if model[key]["mode"] != "off":
            raise ValueError("Fixed-camera comparison rejects camera optimization")
    positions = channel_delays(model["pos_optim_delay_channels"], channels)
    opacities = channel_delays(model["opacity_optim_delay_channels"], channels)
    mlp, sh = model["use_mlp_channels"], model["use_sh_channels"]
    if any(not isinstance(values, list) or len(set(values)) != len(values)
           or not set(values) <= set(channels) for values in (mlp, sh)):
        raise ValueError("Unknown or duplicate color channel")
    spectral = set(channels) - {rgb_channel}
    detached = all(positions[name] < 0 and opacities[name] < 0 for name in spectral)
    joint = all(0 <= positions[name] < iterations and 0 <= opacities[name] < iterations for name in spectral)
    if spectral <= set(mlp):
        mechanism = "neural_spectral_color_enabled"
    elif detached and not mlp:
        mechanism = "non_rgb_structure_gradient_isolation"
    elif joint and not mlp:
        mechanism = "joint_structure_optimization"
    else:
        mechanism = "other_explicit_configuration_requires_review"
    return {"mechanism": mechanism, "position_scale_quaternion_delays": positions,
            "opacity_delays": opacities, "mlp_channels": mlp, "sh_channels": sh,
            "rgb_structure_can_update": 0 <= positions[rgb_channel] < iterations,
            "full_support_lock_proven": False, "paper_recipe_verified": False,
            "gpu_launch_authorized": False}


def verify_training_split(train_names, test_names, available_names):
    for label, values in (("train", train_names), ("test", test_names), ("available", available_names)):
        if not values or any(not isinstance(v, str) or not v for v in values) or len(values) != len(set(values)):
            raise ValueError(f"Empty/duplicate/invalid {label} image identity")
    if set(train_names) & set(test_names):
        raise ValueError("Train/test image leakage")
    if set(train_names) | set(test_names) != set(available_names):
        raise ValueError("Image split must exactly cover declared input")
    return {"train_count": len(train_names), "test_count": len(test_names), "disjoint": True,
            "note": "Identity check only; source hashes and per-band grouping remain required"}
