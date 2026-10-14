import tempfile
import unittest
from pathlib import Path
import numpy as np
from preprocessing.single_hand import require_one_hand_axis, single_hand_mask_directory


class SingleHandScopeTests(unittest.TestCase):
    def test_one_hand_keeps_all_frames_and_values(self):
        values = np.arange(2 * 21 * 2).reshape(2, 1, 21, 2)
        self.assertIs(require_one_hand_axis(values, 'joints'), values)

    def test_two_hands_are_not_silently_sliced_to_the_first_hand(self):
        with self.assertRaisesRegex(ValueError, 'exactly one selected hand'):
            require_one_hand_axis(np.zeros((3, 2, 21, 2)), 'joints')

    def test_ambiguous_masks_fail_even_when_a_hand_directory_exists(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); (root / 'hand').mkdir()
            self.assertEqual(single_hand_mask_directory(root), root / 'hand')
            (root / 'hand2').mkdir()
            with self.assertRaisesRegex(ValueError, 'unsupported'):
                single_hand_mask_directory(root)

    def test_missing_masks_report_the_required_input(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                single_hand_mask_directory(td)
