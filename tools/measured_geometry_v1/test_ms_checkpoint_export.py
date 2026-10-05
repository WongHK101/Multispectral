import unittest

from .ms_checkpoint_export import checkpoint_model_state, image_lookup, validate_export_counts


class ExportIdentityTests(unittest.TestCase):
    def test_appearance_only_does_not_claim_depth_export(self):
        validate_export_counts(0, 95, 150, 95, True)
        for packets, appearance in ((150, 95), (1, 95), (0, 94), (0, 96)):
            with self.assertRaises(ValueError):
                validate_export_counts(packets, appearance, 150, 95, True)

    def test_native_depth_export_still_requires_all_views(self):
        validate_export_counts(94, 60, 94, 60, False)
        with self.assertRaises(ValueError):
            validate_export_counts(0, 60, 94, 60, False)

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
