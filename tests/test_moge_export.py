"""Depth export contract tests independent of MoGe/PyTorch installations."""
import unittest

import numpy as np

from preprocessing.export_moge import (
    encode_depth_mm,
    normalized_intrinsics_to_pixels,
    prepare_prediction,
    resize_nearest_2d,
)


class MoGeExportTests(unittest.TestCase):
    def test_invalid_and_saturated_depth_cannot_become_surfaces(self):
        values = np.array([[np.inf, np.nan, -1, 0, 65.535, 100, 65.5349, .542, .0001]])
        actual = encode_depth_mm(values)
        np.testing.assert_array_equal(actual, [[0, 0, 0, 0, 0, 0, 0, 542, 0]])
        self.assertEqual(actual.dtype, np.uint16)

    def test_model_validity_is_respected(self):
        actual = encode_depth_mm(np.ones((2, 2)), [[True, False], [np.nan, True]])
        np.testing.assert_array_equal(actual, [[1000, 0], [0, 1000]])
        with self.assertRaisesRegex(ValueError, "same shape"):
            encode_depth_mm(np.ones((2, 2)), np.ones((2, 1)))

    def test_nearest_resize_preserves_holes_and_edges(self):
        actual = resize_nearest_2d(np.array([[0, 200], [300, 400]], dtype=np.uint16), (4, 4))
        np.testing.assert_array_equal(actual, [[0, 0, 200, 200], [0, 0, 200, 200], [300, 300, 400, 400], [300, 300, 400, 400]])

    def test_intrinsics_are_scaled_to_original_rgb_grid(self):
        actual = normalized_intrinsics_to_pixels([[1, .1, .5], [0, 1.5, .5], [0, 0, 1]], 640, 480)
        np.testing.assert_array_equal(actual, [[640, 64, 320], [0, 720, 240], [0, 0, 1]])
        with self.assertRaises(ValueError):
            normalized_intrinsics_to_pixels(np.full((3, 3), np.nan), 640, 480)

    def test_prediction_uses_matching_final_camera_and_validity(self):
        depth, camera = prepare_prediction({"depth": np.array([[1, np.inf]]), "mask": np.array([[True, False]]), "intrinsics": np.array([[1, 0, .5], [0, 1, .5], [0, 0, 1]])}, (2, 4))
        np.testing.assert_array_equal(depth, [[1000, 1000, 0, 0], [1000, 1000, 0, 0]])
        np.testing.assert_array_equal(camera, [[4, 0, 2], [0, 2, 1], [0, 0, 1]])
        with self.assertRaisesRegex(ValueError, "valid-pixel mask"):
            prepare_prediction({"depth": depth, "intrinsics": camera}, (2, 4))


if __name__ == "__main__":
    unittest.main()
