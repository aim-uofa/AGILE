"""Regression for the public WiLoR fork's actual pickled 2D-joint output."""
import contextlib
import hashlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from preprocessing.filter_wilor import filter_sequence


class FilterWiLoRTensorTests(unittest.TestCase):
    def test_expands_pickled_joints_without_modifying_raw_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            masks = root / 'mask/hand'
            masks.mkdir(parents=True)
            Image.new('L', (16, 12), 255).save(masks / '00000.png')
            raw = root / 'wilor_raw'
            raw.mkdir()
            shapes = {'mano_verts': (1, 1, 778, 3), 'mano_joints': (1, 1, 21, 3),
                      'mano_global_orient': (1, 1, 1, 3, 3),
                      'mano_hand_pose': (1, 1, 15, 3, 3), 'mano_betas': (1, 1, 10),
                      'mano_masks': (1, 12, 16)}
            for key, shape in shapes.items():
                np.savez(raw / f'{key}.npz', **{key: np.ones(shape, dtype=object)})
            joints = np.empty((1, 1), dtype=object)
            joints[0, 0] = torch.arange(42, dtype=torch.float32).reshape(21, 2)
            np.savez(raw / 'mano_joints_2d.npz', mano_joints_2d=joints)
            hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.iterdir()}
            with contextlib.redirect_stdout(io.StringIO()):
                target = filter_sequence(root, raw, root / 'wilor')
            with np.load(target / 'mano_joints_2d.npz', allow_pickle=False) as data:
                self.assertEqual(data['mano_joints_2d'].shape, (1, 1, 21, 2))
                np.testing.assert_array_equal(data['mano_joints_2d'][0, 0], joints[0, 0].numpy())
            self.assertEqual(hashes, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.iterdir()})


if __name__ == '__main__':
    unittest.main()
