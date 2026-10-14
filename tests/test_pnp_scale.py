from types import SimpleNamespace
import unittest
import torch
from optimization.pipelines.trainer import Runner


class PnpScaleTests(unittest.TestCase):
    def test_pnp_translation_has_same_units_as_scaled_render_geometry(self):
        generator=torch.Generator().manual_seed(12)
        joints=torch.randn(21,3,generator=generator)*.03
        K=torch.tensor([[620.,0.,320.],[0.,620.,240.],[0.,0.,1.]])
        translation=torch.tensor([.1,-.04,.8])
        for scale in [.5,2.5]:
            with self.subTest(scale=scale):
                runner=Runner.__new__(Runner); runner.device='cpu'; runner.optimized_hand_scale=None
                runner.get_actual_frame_id=lambda i:0
                runner.trainset=[{'mano_transform_scale':torch.tensor(scale)}]
                runner.scene_renderer=SimpleNamespace(get_mesh_object=lambda _:SimpleNamespace(canonical_joints=joints))
                runner.set_hand_base_pose=lambda i,R,t,**kw:(R,t)
                projected=(joints*scale+translation)@K.T
                pixels=projected[:,:2]/projected[:,2:]
                _,actual=runner.init_hand_pose_pnp(0,K,pixels)
                torch.testing.assert_close(actual,translation,atol=1e-5,rtol=1e-5)


if __name__=='__main__': unittest.main()
