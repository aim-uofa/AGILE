"""Check bidirectional reader selection with real NPZ inputs and CPU fixtures."""
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import optimization.pipelines.trainer as trainer
from optimization.data.megasam import Parser, Dataset
from tests import test_periodic_export as fixtures

FixtureDataset = fixtures.FixtureDataset
parser_for = fixtures.parser_for


class BidirectionalReaderTests(unittest.TestCase):
    setUp = fixtures.PeriodicExportTests.setUp
    runner = fixtures.PeriodicExportTests.runner

    def make_npz_runner(self, name, start=10):
        runner = self.runner(list(range(start, 26, 5)))
        inputs = Path(self.directory) / name
        inputs.mkdir()
        (inputs / 'wilor').mkdir()
        for kind in ('hand', 'object'):
            (inputs / kind).mkdir()
            for frame in range(26):
                Image.fromarray(np.full((2, 2), 255, dtype=np.uint8)).save(inputs / kind / f'{frame:06d}.png')
        p = parser_for(range(26))
        np.savez(inputs / 'camera_data.npz', intrinsic=p.K, rgbd=p.rgbd, cam_c2w=p.cam_c2w)
        for key in ('mano_betas', 'mano_hand_pose', 'mano_global_orient', 'mano_faces'):
            np.savez(inputs / (key + '.npz'), **{key: getattr(p, key)})
        np.savez(inputs / 'mano_transforms.npz', **p.mano_transforms)
        np.savez(inputs / 'wilor/mano_joints_2d.npz', mano_joints_2d=np.zeros((26, 1, 21, 2)))
        np.savez(inputs / 'optimized_mano_verts.npz', mano_verts=np.zeros((26, 1, 778, 3)))
        np.savez(inputs / 'optimized_mano_joints.npz', mano_joints=np.zeros((26, 1, 21, 3)))
        np.savez(inputs / 'mano_masks.npz', mano_masks=np.ones((26, 2, 2), dtype=bool))
        np.savez(inputs / 'object_poses.npz', object_poses_in_cam=np.tile(np.eye(4), (26, 1, 1)))
        p.object_mesh.export(inputs / 'object_mesh.obj')
        runner.cfg.data_dir = str(inputs)
        runner.parser, runner.trainset = runner._build_parser_and_dataset(str(inputs), start, -1)
        self.assertIsInstance(runner.parser, Parser)
        self.assertIsInstance(runner.trainset, Dataset)
        return runner

    def test_npz_backward_without_folder_marker(self):
        runner = self.make_npz_runner('MDF12')
        runner.train_bidirectional()
        self.assertEqual(runner.hand_calls, [10, 15, 20, 25, 5, 0])
        self.assertEqual(runner.object_calls, [10, 15, 20, 25, 5, 0])
        self.assertEqual(runner.saved[-1]['ids'], [0, 5, 10, 15, 20, 25])
        np.testing.assert_array_equal(runner.optimized_hand_R.numpy(), parser_for([0, 5, 10, 15, 20, 25]).mano_global_orient[:, 0, 0])

    def test_npz_folder_names_produce_identical_results(self):
        old = self.make_npz_runner('old_aligned')
        old.train_bidirectional()
        new = self.make_npz_runner('portable_scene')
        new.train_bidirectional()
        self.assertEqual(old.hand_calls, new.hand_calls)
        self.assertEqual(old.object_calls, new.object_calls)
        self.assertEqual(old.saved, new.saved)
        for attr in ('optimized_hand_R', 'optimized_hand_t', 'optimized_obj_R', 'optimized_obj_t'):
            torch.testing.assert_close(getattr(old, attr), getattr(new, attr), rtol=0, atol=0)

    def test_dataset_factory_contract_survives_misleading_parent_name(self):
        runner = self.runner([10, 15, 20, 25])
        runner.cfg.data_dir = 'portable_parent_containing_aligned/scene'
        all_ids = [0, 5, 10, 15, 20, 25]
        def build(data_dir, start_frame, end_frame):
            parser = parser_for([i for i in all_ids if i >= start_frame and (end_frame < 0 or i < end_frame)])
            return parser, FixtureDataset(parser)
        runner._build_parser_and_dataset = build
        with patch.object(trainer, 'OldDataset', side_effect=AssertionError('Must retain the factory-selected reader')):
            runner.train_bidirectional()
        self.assertEqual(runner.saved[-1]['ids'], all_ids)


    def test_zero_contact_npz_only_optimizes_forward(self):
        runner = self.make_npz_runner('SMu40', start=0)
        runner.train_bidirectional()
        ids = [0, 5, 10, 15, 20, 25]
        self.assertEqual(runner.hand_calls, ids)
        self.assertEqual(runner.object_calls, ids)
        self.assertEqual(runner.saved[-1]['ids'], ids)

if __name__ == '__main__':
    unittest.main()
