import unittest

from .ms_checkpoint_export import checkpoint_model_state, image_lookup


class ExportIdentityTests(unittest.TestCase):
    def test_channel_identity_and_duplicate_path(self):
        group = {"image_name": "a.JPG", "split": "eval", "output_images": {"D": {"relative_path": "images/a.png"}}}
        self.assertEqual(image_lookup({"groups": [group]})["images/a.png"][1], "D")
        with self.assertRaises(ValueError):
            image_lookup({"groups": [group, group]})

    def test_checkpoint_model_prefix(self):
        result = checkpoint_model_state({"_model.module.gauss_params.means": 1, "unrelated": 2})
        self.assertEqual(result, {"gauss_params.means": 1})
        with self.assertRaises(ValueError):
            checkpoint_model_state({"_model.gauss_params.means": 1, "_model.module.gauss_params.means": 2})
        with self.assertRaises(ValueError):
            checkpoint_model_state({"_model.other": 1})


if __name__ == "__main__":
    unittest.main()
