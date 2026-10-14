import unittest
import numpy as np
from preprocessing.depth_contract import alignment_depth_mask


class AlignmentDepthTests(unittest.TestCase):
    def test_missing_saturated_nonfinite_and_background_are_excluded(self):
        depth = np.array([[0, 0.7, 65.535, np.inf, np.nan, -1, 0.9, 0.8]], dtype=np.float32)
        hand = np.ones_like(depth); hand[0, 6] = 0
        mano = np.ones_like(depth); mano[0, 7] = 0
        np.testing.assert_array_equal(
            alignment_depth_mask(depth, hand, mano),
            [[False, True, False, False, False, False, False, False]],
        )

    def test_shapes_must_match_before_backprojection(self):
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            alignment_depth_mask(np.ones((2, 3)), np.ones((3, 2)), np.ones((2, 3)))


if __name__ == "__main__":
    unittest.main()
