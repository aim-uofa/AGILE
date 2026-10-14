from types import SimpleNamespace
import unittest
import torch
from optimization.config.config import Config
from optimization.optim.recovery import fixed_rotation_translation, recover_object_initialization


class TranslationTests(unittest.TestCase):
    def test_fixed_scaled_skeleton_recovers_translation_without_changing_shape(self):
        generator = torch.Generator().manual_seed(7)
        joints = torch.randn(21,3,generator=generator,dtype=torch.float64)*.04
        K = torch.tensor([[610.,0.,305.],[0.,590.,235.],[0.,0.,1.]],dtype=torch.float64)
        expected = torch.tensor([.13,-.04,.72],dtype=torch.float64)
        points = (joints+expected) @ K.T
        pixels = points[:,:2]/points[:,2:]
        original = joints.clone()
        result = fixed_rotation_translation(joints,pixels,K,torch.tensor([.3,.1,1.4],dtype=torch.float64))
        torch.testing.assert_close(result,expected,atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(joints,original)

    def test_degenerate_hand_preserves_prepared_translation(self):
        initial=torch.tensor([.1,0.,1.])
        result=fixed_rotation_translation(torch.zeros(21,3),torch.zeros(21,2),torch.eye(3),initial)
        torch.testing.assert_close(result,initial)


class RecoveryTests(unittest.TestCase):
    def runner(self):
        candidate=torch.eye(4); candidate[:3,3]=torch.tensor([0.,0.,.5])
        runner=SimpleNamespace(cfg=Config(recover_object_pose_from_mask=True),
                               trainset=[{'object_pose':candidate}], optimized_obj_scale=torch.tensor(2.))
        runner.get_actual_frame_id=lambda i: 700
        def set_pose(i,R,t): runner.current_t=t
        def render(**kw):
            visible=abs(float(runner.current_t[0])) < .2
            return {'alpha':torch.full((1,2,2,1),float(visible))}
        runner.set_obj_base_pose=set_pose; runner.render_meshes=render
        return runner

    def test_empty_render_recovers_using_only_observed_mask_and_existing_pose(self):
        runner=self.runner()
        R,t,recovered=recover_object_initialization(runner,0,torch.eye(3),torch.tensor([1.,0.,1.]),
            torch.eye(3)[None],2,2,torch.ones(2,2),torch.zeros(2,2))
        self.assertTrue(recovered)
        torch.testing.assert_close(t,torch.tensor([0.,0.,1.]))
        self.assertEqual(runner.object_recovery_events[-1]['prepared_iou'],1.)

    def test_occluded_object_does_not_trigger_reinitialization(self):
        runner=self.runner(); original=torch.tensor([1.,0.,1.])
        _,t,recovered=recover_object_initialization(runner,0,torch.eye(3),original,
            torch.eye(3)[None],2,2,torch.ones(2,2),torch.ones(2,2))
        self.assertFalse(recovered); torch.testing.assert_close(t,original)

    def test_visible_track_preserved(self):
        runner=self.runner(); original=torch.tensor([.1,0.,.8])
        _,t,recovered=recover_object_initialization(runner,0,torch.eye(3),original,
            torch.eye(3)[None],2,2,torch.ones(2,2),torch.zeros(2,2))
        self.assertFalse(recovered); torch.testing.assert_close(t,original)


if __name__=='__main__': unittest.main()
