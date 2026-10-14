"""Check public-tool handoffs with synthetic files; no neural inference."""
import contextlib
import hashlib
import io
import logging
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from preprocessing.filter_wilor import filter_sequence
from preprocessing.check_inputs import check_sequence


def images(directory, count=2):
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        Image.new('RGB', (16, 12), 'white').save(directory / f'{i:05d}.png')


def wilor(directory):
    directory.mkdir(parents=True)
    shapes = {'mano_verts': (2, 1, 778, 3), 'mano_joints': (2, 1, 21, 3),
              'mano_joints_2d': (2, 1, 21, 2), 'mano_global_orient': (2, 1, 1, 3, 3),
              'mano_hand_pose': (2, 1, 15, 3, 3), 'mano_betas': (2, 1, 10),
              'mano_masks': (2, 12, 16)}
    for key, shape in shapes.items():
        np.savez(directory / (key + '.npz'), **{key: np.ones(shape, dtype=object)})


class InputPreparationTests(unittest.TestCase):
    def test_fork_style_npz_filter_preserves_raw_and_rejects_output_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); images(root / 'rgb'); images(root / 'mask/hand'); images(root / 'mask/object')
            wilor(root / 'wilor_raw')
            raw = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / 'wilor_raw').iterdir()}
            with contextlib.redirect_stdout(io.StringIO()):
                filter_sequence(root, root / 'wilor_raw', root / 'wilor')
            for filename, digest in raw.items():
                self.assertEqual(hashlib.sha256((root / 'wilor_raw' / filename).read_bytes()).hexdigest(), digest)
            with self.assertRaises(FileExistsError):
                filter_sequence(root, root / 'wilor_raw', root / 'wilor')

    def test_filter_refuses_mismatched_frame_count_before_creating_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); images(root / 'mask/hand', 1); wilor(root / 'raw')
            with self.assertRaisesRegex(ValueError, 'frame count'):
                filter_sequence(root, root / 'raw', root / 'filtered')
            self.assertFalse((root / 'filtered').exists())

    def test_input_check_requires_hand_alignment_and_exact_pose_frame_ids(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for relative in ('rgb', 'mask/hand', 'mask/object', 'moge/depth'):
                images(root / relative)
            np.savetxt(root / 'moge/cam_K.txt', np.eye(3)); wilor(root / 'wilor')
            pose_root = root / 'foundationpose/test'; (pose_root / 'ob_in_cam').mkdir(parents=True)
            (pose_root / 'scaled_mesh.obj').write_text('v 0 0 0\n')
            for i in range(2):np.savetxt(pose_root / 'ob_in_cam' / f'{i:05d}.txt', np.eye(4))
            with self.assertRaises(FileNotFoundError):check_sequence(root, pose_root)
            hand = root / 'hand_processed'; hand.mkdir()
            np.savez(hand / 'optimized_mano_verts.npz', mano_verts=np.zeros((2, 1, 778, 3)))
            np.savez(hand / 'optimized_mano_joints.npz', mano_joints=np.zeros((2, 1, 21, 3)))
            np.savez(hand / 'mano_transforms.npz', scales=1., rotations=np.eye(3), translations=np.zeros((2, 3)))
            self.assertTrue(check_sequence(root, pose_root)['input_contract_passed'])
            (pose_root / 'ob_in_cam/00001.txt').rename(pose_root / 'ob_in_cam/00002.txt')
            with self.assertRaisesRegex(ValueError, 'frame IDs'):check_sequence(root, pose_root)


if __name__ == '__main__':
    unittest.main()
