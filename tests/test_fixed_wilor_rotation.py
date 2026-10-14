from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import torch
import tyro

from optimization.config.config import Config
from optimization.optim.hand_optimizer import HandOptimizer


class FixedWiLoRRotationTests(unittest.TestCase):
    def runner(self):
        near_pi = torch.from_numpy(cv2.Rodrigues(np.array([np.pi - 1e-4, 0., 0.]))[0]).float()
        other = torch.from_numpy(cv2.Rodrigues(np.array([0., .7, 0.]))[0]).float()
        frames = [{'mano_global_orient': R.reshape(1, 1, 3, 3)} for R in [other, near_pi]]
        runner = SimpleNamespace(device='cpu', trainset=frames, num_frames=2,
                                 optimized_hand_R=torch.stack([other, torch.zeros(3, 3)]),
                                 optimized_hand_t=torch.zeros(2, 3),
                                 optimized_hand_pose_offsets=torch.zeros(2, 6))
        runner.get_actual_frame_id = lambda i: i * 5
        runner.init_hand_pose_pnp = Mock(return_value=(other, torch.zeros(3)))
        def set_pose(frame, R, t, **kwargs):
            runner.current_R = frames[frame]['mano_global_orient'][0, 0] if R is None else R
            runner.current_t = torch.zeros(3) if t is None else t
            return runner.current_R, runner.current_t
        runner.set_hand_base_pose = Mock(side_effect=set_pose)
        return runner, near_pi

    def test_default_and_cli_are_backward_compatible(self):
        self.assertFalse(Config().fix_hand_rotation_to_wilor)
        cfg = tyro.cli(Config, args=['--fix-hand-rotation-to-wilor'])
        self.assertTrue(cfg.fix_hand_rotation_to_wilor)
        self.assertEqual(cfg.hand_rot_lr, Config().hand_rot_lr)

    def test_fixed_rotation_comes_from_current_frame_not_pnp_or_previous_frame(self):
        runner, expected = self.runner()
        previous_t = torch.tensor([1., 2., 3.])
        cfg = Config(fix_hand_rotation_to_wilor=True, hand_pose_from_last_frame=True)
        R, t = HandOptimizer(cfg, runner)._initialize_hand_pose(
            1, torch.eye(3), torch.zeros(21, 2), torch.eye(3), previous_t)
        self.assertTrue(torch.equal(R, expected))
        self.assertTrue(torch.equal(t, previous_t))
        runner.init_hand_pose_pnp.assert_not_called()

    def test_legacy_mode_still_uses_pnp(self):
        runner, _ = self.runner()
        R, _ = HandOptimizer(Config(), runner)._initialize_hand_pose(
            1, torch.eye(3), torch.zeros(21, 2), None, None)
        runner.init_hand_pose_pnp.assert_called_once()
        self.assertTrue(torch.equal(R, runner.init_hand_pose_pnp.return_value[0]))

    def test_fixed_mode_rejects_external_pose_overrides(self):
        runner, _ = self.runner()
        cfg = Config(fix_hand_rotation_to_wilor=True, mano_transforms_path='external.npz')
        with self.assertRaisesRegex(ValueError, 'external'):
            HandOptimizer(cfg, runner)

    def test_nonzero_rotation_lr_cannot_change_near_pi_rotation_but_translation_updates(self):
        runner, expected = self.runner()
        with tempfile.TemporaryDirectory() as directory:
            cfg = Config(fix_hand_rotation_to_wilor=True, hand_rot_lr=.1,
                         hand_trans_lr=.1, hand_pose_from_last_frame=True,
                         result_dir=directory, show_opt_progress=False,
                         hand_sdf_loss=False, ho_relpose_loss=False,
                         hand_2d_joints_loss=False, hand_smoothness_loss=False)
            shape = (1, 2, 2, 1)
            prepared = dict(Ks=torch.eye(3).unsqueeze(0), images_gt=torch.zeros(1, 2, 2, 3),
                            depths_gt=torch.zeros(shape), depths_masked=torch.zeros(shape),
                            hand_mask=torch.ones(1, 2, 2), object_mask=torch.zeros(1, 2, 2),
                            mano_mask=torch.ones(1, 2, 2), normals_gt=None, H=2, W=2,
                            hand_pose=torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 15, 1, 1),
                            mano_joints_2d=torch.zeros(21, 2))
            runner.prepare_frame_data = lambda *a, **k: prepared
            runner.create_scheduler = lambda optimizer, *a, **k: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
            runner.render_meshes = lambda **k: dict(colors=torch.zeros(1, 2, 2, 3),
                                                     alpha=torch.ones(shape) + runner.current_t.sum() * .01,
                                                     depth=torch.zeros(shape), zbuf=torch.zeros(shape))
            optimizer = HandOptimizer(cfg, runner)
            optimizer._build_loss_context = lambda **k: SimpleNamespace()
            optimizer._to_legacy_loss_dict = lambda d: d
            target = torch.tensor([.5, -.2, 1.])
            def losses(*a, **k):
                # A real differentiable translation objective. Rotation LR stays
                # positive to ensure the feature, rather than a zero LR, freezes R.
                return {'translation_loss': ((runner.current_t - target) ** 2).sum()}
            with patch('optimization.optim.hand_optimizer.compute_losses', side_effect=losses),\
                 patch('optimization.optim.hand_optimizer.make_vis_image', return_value=np.zeros((2, 2, 3), dtype=np.uint8)),\
                 patch('optimization.optim.hand_optimizer.imageio.mimwrite'):
                result = optimizer.train_one_frame(1, train_max_iter=8)
            runner.init_hand_pose_pnp.assert_not_called()
            self.assertTrue(torch.equal(result['final_R_w2c'], expected))
            self.assertTrue(torch.equal(runner.optimized_hand_R[1], expected))
            self.assertEqual(float(torch.linalg.norm(runner.optimized_hand_pose_offsets[1, :3])), 0.)
            self.assertLess(float(torch.linalg.norm(result['final_t_w2c'] - target)), float(torch.linalg.norm(target)))
            # The exported pose must include the final Adam step, just as the
            # stored offset does; otherwise geometry and optimizer state differ.
            torch.testing.assert_close(result['final_t_w2c'],
                                       runner.optimized_hand_pose_offsets[1, 3:])
            torch.testing.assert_close(runner.current_t.detach(), result['final_t_w2c'])


if __name__ == '__main__':
    unittest.main()
