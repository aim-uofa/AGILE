"""Exercise partial snapshots and bidirectional exports without CUDA training."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import torch
import trimesh

from optimization.config.config import Config
from optimization.evaluation.ho3d_io import load_results_data
from optimization.pipelines.trainer import Runner
from tests.test_rotation_geometry import MC1_HALF_TURN


class QuietProgress:
    def __init__(self, values, **kwargs):
        self.values = values

    def __iter__(self):
        return iter(self.values)

    def set_description(self, *args):
        pass

    def write(self, *args):
        pass


def parser_for(frame_ids):
    ids = np.asarray(frame_ids, dtype=np.int64)
    n = len(ids)
    rotations = np.stack([cv2.Rodrigues(np.array([0., .1 + f / 1000., 0.]))[0] for f in ids]).astype(np.float32)
    return SimpleNamespace(
        frame_ids=ids, K=np.eye(3, dtype=np.float32),
        rgbd=np.zeros((n, 2, 2, 4), dtype=np.float32),
        cam_c2w=np.repeat(np.eye(4, dtype=np.float32)[None], n, axis=0),
        mano_betas=np.zeros((n, 1, 10), dtype=np.float32),
        mano_hand_pose=np.tile(np.eye(3, dtype=np.float32), (n, 1, 15, 1, 1)),
        mano_global_orient=rotations[:, None, None],
        mano_faces=np.array([[0, 1, 2]]),
        mano_transforms=dict(scales=1., rotations=np.eye(3), translations=np.zeros((n, 3))),
        object_mesh=trimesh.creation.box())


class FixtureDataset:
    def __init__(self, parser):
        self.parser = parser

    def __len__(self):
        return len(self.parser.frame_ids)

    def __getitem__(self, i):
        p = self.parser
        return dict(frame_id=int(p.frame_ids[i]), image=torch.zeros(2, 2, 3),
                    K=torch.eye(3), mano_betas=torch.from_numpy(p.mano_betas[i]),
                    mano_hand_pose=torch.from_numpy(p.mano_hand_pose[i]),
                    mano_global_orient=torch.from_numpy(p.mano_global_orient[i]),
                    mano_transform_scale=torch.tensor(1.))


class PeriodicExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = self.temp.name
        for target, value in [('optimization.pipelines.trainer.tqdm.tqdm', QuietProgress),
                              ('optimization.pipelines.trainer.imageio.mimwrite', Mock()),
                              ('optimization.pipelines.trainer.overlay_rgb', lambda a, b, c: b)]:
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def runner(self, ids, reverse=False):
        runner = Runner.__new__(Runner)
        runner.device = 'cpu'
        runner.cfg = Config(result_dir=self.directory, data_dir='fixture/aligned',
                            start_frame=int(ids[0]), fix_hand_rotation_to_wilor=True,
                            reverse_frame_order=reverse, show_opt_progress=False,
                            optimize_hand_pose=False, optimize_hand_scale=False)
        runner.parser = parser_for(ids)
        runner.trainset = FixtureDataset(runner.parser)
        runner.num_frames = len(ids)
        runner.optimized_hand_R = torch.zeros(len(ids), 3, 3)
        runner.optimized_hand_t = torch.zeros(len(ids), 3)
        runner.optimized_obj_R = torch.zeros(len(ids), 3, 3)
        runner.optimized_obj_t = torch.zeros(len(ids), 3)
        runner.optimized_hand_scale = None
        runner.optimized_obj_scale = None
        runner.hand_appeared = False
        runner.get_actual_frame_id = lambda i: runner.trainset[i]['frame_id']
        runner.set_hand_base_pose = Mock()
        runner.set_obj_base_pose = Mock()
        runner.align_object_to_hand = Mock()
        runner.save_ho_pcd = Mock(return_value='fixture.ply')
        runner._print_final_summary = Mock()
        hand = SimpleNamespace(get_transformed_vertices=lambda: torch.zeros(3, 3))
        runner.scene_renderer = SimpleNamespace(get_mesh_object=lambda name: hand if name == 'hand' else None)
        runner.render_meshes = lambda **kw: dict(colors=torch.zeros(1, 2, 2, 3),
                                                alpha=torch.ones(1, 2, 2, 1),
                                                zbuf=torch.zeros(1, 2, 2, 1))
        output = SimpleNamespace(vertices=torch.zeros(1, 778, 3),
                                 joints=torch.arange(63).float().reshape(1, 21, 3) * .001)
        runner.mano_model = lambda **kw: output
        runner.hand_calls = []
        runner.object_calls = []

        def hand_train(i, **kwargs):
            frame = runner.trainset[i]
            runner.hand_calls.append(frame['frame_id'])
            runner.optimized_hand_R[i] = frame['mano_global_orient'][0, 0]
            runner.optimized_hand_t[i] = torch.tensor([frame['frame_id'] * .001, 0., .5])
            return {}

        def object_train(i, **kwargs):
            runner.object_calls.append(runner.trainset[i]['frame_id'])
            runner.optimized_obj_R[i] = torch.eye(3)
            runner.optimized_obj_t[i] = torch.tensor([0., .05, .5])

        runner.hand_optimizer = SimpleNamespace(train_one_frame=hand_train)
        runner.object_optimizer = SimpleNamespace(train_one_frame=object_train)
        runner.saved = []
        export = runner.save_optimized_data

        def save(*args, **kwargs):
            folder = Path(export(*args, **kwargs))
            prediction, n = load_results_data(str(folder))
            camera = np.load(folder / 'camera_data.npz')
            lengths = {key: len(np.load(folder / (key + '.npz'))[key])
                       for key in ['mano_betas', 'mano_hand_pose', 'mano_global_orient']}
            lengths.update(rgbd=len(camera['rgbd']), cam_c2w=len(camera['cam_c2w']))
            self.assertTrue(all(length == n for length in lengths.values()), lengths)
            transforms = np.load(folder / 'mano_transforms.npz')
            np.testing.assert_array_equal(transforms['rotations'], np.zeros((n, 3)))
            runner.saved.append(dict(n=n, ids=prediction['frame_ids'].cpu().numpy().tolist(),
                                     lengths=lengths))
            return str(folder)

        runner.save_optimized_data = save
        return runner

    def test_export_preserves_texture_rgb_and_uv_topology(self):
        runner = self.runner([0])
        runner.cfg.optimize_object_texture = True  # preserve the explicitly fitted appearance
        runner.mano_model = None
        runner.optimized_hand_R[:] = torch.from_numpy(runner.parser.mano_global_orient[:, 0, 0])
        runner.optimized_obj_R[:] = torch.eye(3)
        texture = np.tile(np.array([.2, .4, .8], dtype=np.float32), (2, 2, 1))
        obj = SimpleNamespace(canonical_vertices=torch.tensor([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.]]),
            faces=torch.tensor([[0,1,2]]), texture_image=torch.from_numpy(texture),
            uvs=torch.tensor([[0.,0.], [1.,0.], [0.,1.]]))
        runner.scene_renderer = SimpleNamespace(get_mesh_object=lambda name: obj if name == 'object' else None)
        Runner.save_optimized_data(runner, self.directory, end_frame=1)
        mesh = trimesh.load(Path(self.directory) / 'aligned/object_mesh.obj', process=False)
        np.testing.assert_array_equal(np.asarray(mesh.visual.material.image)[0,0,:3], [51,102,204])
        np.testing.assert_array_equal(mesh.visual.uv, obj.uvs.numpy())
        np.testing.assert_array_equal(mesh.faces, obj.faces.numpy())

    def test_object_half_turn_export_preserves_optimized_matrix_exactly(self):
        runner = self.runner([610])
        runner.mano_model = None
        runner.optimized_hand_R[:] = torch.from_numpy(runner.parser.mano_global_orient[:, 0, 0])
        runner.optimized_obj_R[0] = torch.from_numpy(MC1_HALF_TURN)
        runner.optimized_obj_t[0] = torch.tensor([.01, -.02, .5])
        # No object matrix-to-vector conversion is needed for native export.
        with patch('optimization.pipelines.trainer.rotation_matrix_to_rvec',
                   side_effect=AssertionError('export must retain object matrices')):
            Runner.save_optimized_data(runner, self.directory, end_frame=1)
        poses = np.load(Path(self.directory) / 'aligned/object_poses.npz')['object_poses_in_cam']
        np.testing.assert_array_equal(poses[0, :3, :3], runner.optimized_obj_R[0].numpy())
        np.testing.assert_array_equal(poses[0, :3, 3], runner.optimized_obj_t[0].numpy())
        np.testing.assert_array_equal(poses[0, 3], [0., 0., 0., 1.])

    def test_hand_half_turn_geometry_uses_relative_matrix_directly(self):
        runner = self.runner([610])
        runner.cfg.fix_hand_rotation_to_wilor = False
        runner.parser.mano_global_orient[:] = np.eye(3)
        runner.optimized_hand_R[0] = torch.from_numpy(MC1_HALF_TURN)
        runner.optimized_hand_t[0] = torch.tensor([.01, -.02, .5])
        runner.optimized_obj_R[0] = torch.eye(3)
        vertices = torch.arange(778 * 3).reshape(1, 778, 3).float() * .0001
        joints = torch.arange(21 * 3).reshape(1, 21, 3).float() * .001
        runner.mano_model = lambda **kw: SimpleNamespace(vertices=vertices, joints=joints)
        Runner.save_optimized_data(runner, self.directory, end_frame=1)
        folder = Path(self.directory) / 'aligned'
        transform = np.load(folder / 'mano_transforms.npz')
        np.testing.assert_array_equal(transform['rotation_matrices'][0], MC1_HALF_TURN)
        expected = (vertices[0] - joints[0, 0]) @ runner.optimized_hand_R[0].T + runner.optimized_hand_t[0]
        saved = np.load(folder / 'optimized_mano_verts.npz')['mano_verts'][0, 0]
        np.testing.assert_array_equal(saved, expected.numpy())

    def test_forward_snapshot_stops_before_unprocessed_zero_rotations(self):
        ids = list(range(100, 365, 5))  # 53 frames; snapshot after frame index 50.
        runner = self.runner(ids)
        runner._run_train_loop([], self.directory)
        self.assertEqual([s['n'] for s in runner.saved], [51, 53])
        self.assertEqual(runner.saved[0]['ids'], ids[:51])
        self.assertEqual(runner.saved[-1]['ids'], ids)
        self.assertEqual(runner.hand_calls, ids)
        self.assertEqual(runner.object_calls, ids)

    def test_runner_exports_original_surface_and_proxy_separately(self):
        from tests.test_object_assets import textured_box
        from optimization.data.result_bundle import ResultBundle
        runner = self.runner([0])
        runner.mano_model = None
        runner.parser.object_mesh = textured_box()
        runner.cfg.optimize_object_scale = False
        runner.cfg.optimize_object_scale_not_aligning = True
        runner.optimized_obj_scale = torch.tensor([2., 3., 4.])
        runner.optimized_hand_R[:] = torch.from_numpy(runner.parser.mano_global_orient[:, 0, 0])
        runner.optimized_obj_R[:] = torch.eye(3)
        obj = SimpleNamespace(canonical_vertices=torch.tensor([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.]]),
            faces=torch.tensor([[0,1,2]]), texture_image=torch.full((2,2,3), .25),
            uvs=torch.tensor([[0.,0.], [1.,0.], [0.,1.]]))
        runner.scene_renderer = SimpleNamespace(get_mesh_object=lambda name: obj if name == 'object' else None)
        Runner.save_optimized_data(runner, self.directory, end_frame=1)
        bundle = ResultBundle(Path(self.directory)/'aligned')
        self.assertEqual(len(bundle.object_mesh.faces), len(runner.parser.object_mesh.faces))
        self.assertEqual(bundle.object_mesh.visual.material.baseColorTexture.size, (48,32))
        np.testing.assert_allclose(bundle.object_mesh.vertices, runner.parser.object_mesh.vertices * [2,3,4], atol=1e-8)
        proxy = bundle.load_object_variant('optimization_proxy')
        self.assertEqual(len(proxy.faces), 1)
        self.assertEqual(proxy.visual.material.baseColorTexture.size, (2,2))

    def test_reverse_snapshot_waits_until_all_frames_are_processed(self):
        ids = list(range(0, 275, 5))
        runner = self.runner(ids, reverse=True)
        runner._run_train_loop([], self.directory)
        self.assertEqual([s['n'] for s in runner.saved], [55])
        self.assertEqual(runner.saved[0]['ids'], ids)
        self.assertEqual(runner.hand_calls, list(reversed(ids)))

    def test_bidirectional_seed_is_not_retrained_and_combined_export_is_complete(self):
        runner = self.runner([10, 15, 20, 25])
        all_ids = [0, 5, 10, 15, 20, 25]

        def build(data_dir, start_frame, end_frame):
            ids = [i for i in all_ids if i >= start_frame and (end_frame < 0 or i < end_frame)]
            parser = parser_for(ids)
            return parser, FixtureDataset(parser)

        runner._build_parser_and_dataset = build
        with patch('optimization.pipelines.trainer.OldDataset', FixtureDataset):
            runner.train_bidirectional()
        self.assertEqual(runner.hand_calls, [10, 15, 20, 25, 5, 0])
        self.assertEqual(runner.object_calls, [10, 15, 20, 25, 5, 0])
        self.assertEqual(runner.saved[-1]['ids'], all_ids)
        np.testing.assert_array_equal(runner.optimized_hand_R.numpy(),
                                      parser_for(all_ids).mano_global_orient[:, 0, 0])

    def test_processed_rotation_corruption_is_still_rejected(self):
        runner = self.runner([0, 5, 10])
        runner.optimized_hand_R[:] = torch.from_numpy(runner.parser.mano_global_orient[:, 0, 0])
        runner.optimized_hand_R[1] = torch.eye(3)
        with self.assertRaisesRegex(ValueError, 'rotation changed'):
            runner.save_optimized_data(self.directory, start_frame=0, end_frame=3)

    def test_backward_temporal_cache_rebuilds_contact_frame_geometry(self):
        runner = self.runner([10, 15, 20, 25])
        ids = [0, 5, 10, 15, 20, 25]
        def build(data_dir, start_frame, end_frame):
            parser = parser_for([f for f in ids if f >= start_frame and (end_frame < 0 or f < end_frame)])
            return parser, FixtureDataset(parser)
        runner._build_parser_and_dataset = build
        runner.cached_shape = -1
        def set_hand(i, *a, compute_mano=False, **kw):
            if compute_mano:
                runner.cached_shape = runner.trainset[i]['frame_id']
        runner.set_hand_base_pose = Mock(side_effect=set_hand)
        hand = runner.scene_renderer.get_mesh_object('hand')
        hand.get_transformed_vertices = lambda: torch.full((3, 3), float(runner.cached_shape))
        train = runner.hand_optimizer.train_one_frame
        def hand_train(i, **kw):
            frame = runner.trainset[i]['frame_id']
            if frame == 5:
                torch.testing.assert_close(runner.last_hand_verts_incam, torch.full((3,3), 10.))
            runner.cached_shape = frame
            return train(i, **kw)
        runner.hand_optimizer.train_one_frame = hand_train
        with patch('optimization.pipelines.trainer.OldDataset', FixtureDataset):
            runner.train_bidirectional()


if __name__ == '__main__':
    unittest.main()
