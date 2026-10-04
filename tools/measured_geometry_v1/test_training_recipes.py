"""Synthetic command/recipe tests; no method import or subprocess launch."""
import unittest
from copy import deepcopy

from .training_recipes import CHANNELS, command, recipe, validate_resolved


def resolved(method):
    spec = recipe(method)
    config = {"machine": {"seed": 42}, "max_num_iterations": 120000,
              "steps_per_eval_image": 0, "steps_per_eval_batch": 0, "steps_per_eval_all_images": 0,
              "pipeline": deepcopy(spec["pipeline"])}
    config["pipeline"]["datamanager"]["dataparser"] = spec["dataparser"].copy()
    return config


class RecipeTests(unittest.TestCase):
    def test_three_scientific_configs_validate(self):
        for method in ("jo", "sig_mechanism", "ms_splatting_neural"):
            with self.subTest(method=method):
                self.assertFalse(validate_resolved(resolved(method), recipe(method))["gpu_launch_authorized"])

    def test_sig_changes_only_two_gradient_routes(self):
        a, b = recipe("jo")["pipeline"], recipe("sig_mechanism")["pipeline"]
        changed = {key for key in a["model"] if a["model"][key] != b["model"][key]}
        self.assertEqual(changed, {"pos_optim_delay_channels", "opacity_optim_delay_channels"})
        self.assertEqual(a["datamanager"], b["datamanager"])
        self.assertTrue(b["model"]["use_scale_regularization"])

    def test_neural_effective_shape_and_direction(self):
        model = recipe("ms_splatting_neural")["pipeline"]["model"]
        self.assertEqual(model["use_mlp_channels"], CHANNELS)
        self.assertEqual(model["use_sh_channels"], [])
        self.assertEqual((model["feature_input_dim"], model["mlp_hidden_depth"], model["mlp_hidden_layers"]), (8, 32, 1))
        self.assertEqual((11+1)*32 + (32+1)*32 + (32+1)*7, 1671)
        self.assertTrue(model["direction_encoding_flag"])
        self.assertEqual(model["direction_encoding_num_frequencies"], 0)
        self.assertFalse(model["positional_encoding_flag"])

    def test_recipes_do_not_share_mutable_state(self):
        a = recipe("jo")
        a["pipeline"]["model"]["use_sh_channels"].clear()
        self.assertEqual(recipe("jo")["pipeline"]["model"]["use_sh_channels"], CHANNELS)

    def test_command_is_argv_with_bound_parser(self):
        output = command("ms_splatting_neural", entrypoint="/isolated/bin/ns-train", data="/inputs/road",
                         output="/runs/test ; literal", split="/inputs/road/train_split.json")
        args = output["argv"]
        self.assertIn("/runs/test ; literal", args)
        self.assertNotIn("--resolution", args)
        self.assertEqual(args[args.index("--downscale-factor")+1], "0")
        self.assertLess(args.index("mmsplat-dataparser"), args.index("--downscale-factor"))
        self.assertFalse(output["execution_performed"])

    def test_bad_paths_and_input_output_overlap_fail(self):
        for out in ("relative", "/inputs/road/new", "/inputs", "/tmp/../run", "/runs\nother"):
            with self.subTest(out=out), self.assertRaises(ValueError):
                command("jo", entrypoint="/env/ns-train", data="/inputs/road", output=out, split="/inputs/split.json")

    def test_unknown_method_fails(self):
        with self.assertRaises(ValueError):
            recipe("SIG_is_JO_renamed")

    def test_tampered_expected_recipe_fails(self):
        spec = recipe("jo"); spec["seed"] = 0
        with self.assertRaisesRegex(ValueError, "reviewed"):
            validate_resolved(resolved("jo"), spec)

    def test_wrong_neural_or_gradient_settings_fail(self):
        changes = {"feature_input_dim": 16, "mlp_hidden_layers": 0, "direction_encoding_num_frequencies": 10,
                   "positional_encoding_flag": True, "use_mlp_channels": [], "stop_split_at": 60000,
                   "camera_optimizer_rgb": {"mode": "SO3xR3"}}
        for key, value in changes.items():
            cfg = resolved("ms_splatting_neural"); cfg["pipeline"]["model"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                validate_resolved(cfg, recipe("ms_splatting_neural"))

    def test_hidden_resize_holdout_seed_changes_fail(self):
        for section, key, value in (("machine", "seed", 0), ("dataparser", "downscale_factor", 2),
                                    ("dataparser", "eval_mode", "fraction")):
            cfg = resolved("jo")
            target = cfg["machine"] if section == "machine" else cfg["pipeline"]["datamanager"]["dataparser"]
            target[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                validate_resolved(cfg, recipe("jo"))

    def test_type_confusion_rejected(self):
        cfg = resolved("jo"); cfg["max_num_iterations"] = "120000"
        with self.assertRaises(ValueError):
            validate_resolved(cfg, recipe("jo"))


if __name__ == "__main__":
    unittest.main()
