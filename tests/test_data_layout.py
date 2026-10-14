"""Check renamed data paths and compatibility with pinned earlier releases."""
from pathlib import Path
import tempfile
import unittest
from preprocessing.layout import generated_object_directory, object_pose_directory


class DataLayoutTests(unittest.TestCase):
    def test_current_and_pinned_legacy_inputs_resolve_without_modification(self):
        cases = [
            (generated_object_directory, 'object', 'nano_final'),
            (object_pose_directory, 'object_pose',
             'foundationpose/foundationpose_moge_scaled_nano_final_preg_5_30.0_global_scaleall'),
        ]
        for resolve, current, legacy in cases:
            with self.subTest(current=current), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.assertRaises(FileNotFoundError):
                    resolve(root)
                (root / legacy).mkdir(parents=True)
                self.assertEqual(resolve(root), root / legacy)
                (root / current).mkdir()
                # A re-download may leave the immutable old layout beside the new one.
                self.assertEqual(resolve(root), root / current)
                self.assertTrue((root / legacy).is_dir())

    def test_incomplete_current_layout_does_not_silently_use_other_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'object').mkdir()
            (root / 'nano_final').mkdir()
            (root / 'nano_final/textured_mesh.obj').write_text('legacy mesh')
            selected = generated_object_directory(root) / 'textured_mesh.obj'
            self.assertEqual(selected, root / 'object/textured_mesh.obj')
            self.assertFalse(selected.exists())


if __name__ == '__main__':
    unittest.main()
