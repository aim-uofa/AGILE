import tempfile
from pathlib import Path
import unittest
import zipfile

import numpy as np
import torch

from optimization.data.loader import _load_npz_array


class TensorNPZTests(unittest.TestCase):
    def test_pickled_tensor_joints_keep_frame_and_hand_axes(self):
        values = np.empty((2, 1), dtype=object)
        values[0, 0] = torch.arange(42, dtype=torch.float32).reshape(21, 2)
        values[1, 0] = torch.arange(42, dtype=torch.float32).reshape(21, 2) + 100
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'joints.npz'
            np.savez(path, mano_joints_2d=values)
            loaded = _load_npz_array(str(path), 'mano_joints_2d')
        self.assertEqual(loaded.shape, (2, 1, 21, 2))
        self.assertEqual(loaded.dtype, np.float32)
        np.testing.assert_array_equal(loaded[1, 0], values[1, 0].numpy())

    def test_regular_arrays_are_preserved(self):
        values = np.array([[[True, False]], [[False, True]]])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'mask.npz'
            np.savez(path, mask=values)
            loaded = _load_npz_array(str(path), 'mask')
        self.assertEqual(loaded.dtype, values.dtype)
        np.testing.assert_array_equal(loaded, values)

    def test_numpy2_module_paths_remain_readable_on_numpy1(self):
        values = np.array([[[1., 2.]]], dtype=object)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'legacy.npz'
            np.savez(path, values=values)
            with zipfile.ZipFile(path) as archive:
                content = archive.read('values.npy')
            # NumPy's object-array writer uses GLOBAL pickle module names.
            content = content.replace(b'numpy.core.multiarray\n', b'numpy._core.multiarray\n')
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('values.npy', content)
            loaded = _load_npz_array(str(path), 'values')
        np.testing.assert_array_equal(loaded, values)


if __name__ == '__main__':
    unittest.main()
