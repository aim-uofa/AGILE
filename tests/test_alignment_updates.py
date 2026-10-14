import tempfile, unittest
from unittest.mock import patch
import numpy as np
import torch
from optimization.config.config import Config
from optimization.pipelines.trainer import Runner


class AlignmentUpdateTests(unittest.TestCase):
    def test_scale_regularizer_reaches_optimizer_and_final_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = Runner.__new__(Runner)
            runner.device = 'cpu'
            runner.cfg = Config(result_dir=directory, object_scale_dim=1,
                                object_scale_reg_loss=True, object_scale_reg_lambda=1.,
                                initial_object_scale=1., object_scale_lr=.1, align_ho_max_iter=1,
                                show_opt_progress=False, visualize_optimization=False)
            runner.optimized_obj_scale = torch.nn.Parameter(torch.tensor(2.))
            runner.optimized_obj_R = torch.zeros(1,3,3)
            runner.optimized_obj_t = torch.zeros(1,3)
            runner.optimized_obj_pose_offsets = torch.zeros(1,6)
            runner.get_actual_frame_id = lambda i: 0
            runner.hand_zbuf = torch.ones(1,2,2,2)
            runner.prepare_frame_data = lambda *a,**k: dict(Ks=torch.eye(3)[None],images_gt=torch.zeros(1,2,2,3),
                depths_gt=None,normals_gt=None,depths_masked=None,hand_mask=torch.zeros(2,2),object_mask=torch.ones(2,2),H=2,W=2)
            def set_pose(i,R,t):
                runner.current_t=t; runner.current_scale=runner.optimized_obj_scale.clone()
            runner.set_obj_base_pose = set_pose
            def render(**kw):
                return dict(colors=torch.zeros(1,2,2,3),alpha=torch.ones(1,2,2,1)+runner.optimized_obj_scale*0,
                            depth=None,normals=None,zbuf=(runner.optimized_obj_scale+runner.current_t.sum()).expand(1,2,2,2))
            runner.render_meshes = render
            runner.create_scheduler = lambda opt,*a,**kw: torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
            with patch('optimization.pipelines.trainer.zbuf_contact_loss_simplified',side_effect=lambda h,o,*a,**kw:(o.sum()*0,{'debug_pcd':None})),\
                 patch('optimization.pipelines.trainer.make_vis_image',return_value=np.zeros((2,2,3),dtype=np.uint8)),\
                 patch('optimization.pipelines.trainer.imageio.mimwrite'):
                runner.align_object_to_hand(0,torch.eye(3),torch.tensor([0.,0.,.5]))
            self.assertLess(runner.optimized_obj_scale.item(),2.)
            torch.testing.assert_close(runner.current_scale,runner.optimized_obj_scale)
            torch.testing.assert_close(runner.current_t,runner.optimized_obj_t[0])


if __name__=='__main__': unittest.main()
