"""Exercise real rendering/export paths, not just API presence."""
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch
import trimesh
from PIL import Image,ImageSequence
from optimization.data.visualization_loader import VisualizationLoader
from scripts.vis.render_demo import FinalRenderer
from scripts.vis.run_viser_hoi import encode_calibrated_gif
from optimization.utils.visualization import distance_heatmap


class RenderingTests(unittest.TestCase):
    def test_diagnostic_surface_distance_and_same_raster(self):
        mesh=trimesh.Trimesh(vertices=[[-.2,-.2,0],[.2,-.2,0],[.2,.2,0],[-.2,.2,0]],
                             faces=[[0,1,2],[0,2,3]],process=False)
        pose=np.eye(4);pose[2,3]=.5
        hand=np.array([[-.02,-.02,.49],[.02,-.02,.49],[0,.02,.49]])
        b=SimpleNamespace(object_mesh=mesh,has_object=True,has_mano=True,valid_object=[True],valid_hand=[True],
            mano_faces=np.array([[0,1,2]]),get_mano_vertices=lambda i:hand,
            K=np.array([[60.,0,32.],[0,65.,24.],[0,0,1.]]),width=64,height=48,get_object_pose=lambda i:pose)
        renderer=FinalRenderer(b,'cpu')
        ordinary,alpha=renderer.render(0)
        colors,second,detail=renderer.render(0,diagnostics=True)
        np.testing.assert_array_equal(colors,ordinary)
        np.testing.assert_array_equal(alpha,second)
        self.assertAlmostEqual(detail['contact']['minimum_mm'],10.,places=3)
        self.assertEqual(detail['contact']['vertices_within_5mm'],0)
        self.assertFalse(np.any(detail['object_mask']&detail['hand_mask']))
        np.testing.assert_array_equal(detail['object_mask']|detail['hand_mask'],alpha[...,0].astype(bool))

    def test_render_saved_texture_and_full_camera_matrix(self):
        mesh=trimesh.Trimesh(vertices=[[-.1,-.1,0],[.1,-.1,0],[0,.1,0]],faces=[[0,1,2]],process=False)
        mesh.visual=trimesh.visual.TextureVisuals(uv=[[0,0],[1,0],[.5,1]],
            image=Image.fromarray(np.tile(np.array([51,102,204],np.uint8),(4,4,1))))
        pose=np.eye(4);pose[2,3]=.5
        b=SimpleNamespace(object_mesh=mesh,has_object=True,has_mano=False,valid_object=[True],
            K=np.array([[60.,0,23.],[0,65.,17.],[0,0,1.]]),width=64,height=48,
            get_object_pose=lambda i:pose,cam_c2w=np.diag([1.,1.,1.,1.])[None])
        b.cam_c2w[0,0,3]=3.  # Must not move camera-space geometry a second time.
        renderer=FinalRenderer(b,'cpu');colors,alpha=renderer.render(0)
        self.assertEqual(alpha[17,23,0],1)
        np.testing.assert_allclose(colors[17,23]*255,[51,102,204],atol=1)
        self.assertEqual(alpha[40,60,0],0)
        np.testing.assert_array_equal(renderer.object[0],mesh.vertices)

    def test_parametric_nonzero_start_uses_corresponding_rotation(self):
        from tests.test_result_bundle import BundleTests
        fixture=BundleTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        p=fixture.source;n=4
        np.savez(p/'mano_betas.npz',mano_betas=np.zeros((n,1,10)))
        np.savez(p/'mano_hand_pose.npz',mano_hand_pose=np.tile(np.eye(3),(n,1,15,1,1)))
        np.savez(p/'mano_global_orient.npz',mano_global_orient=np.tile(np.eye(3),(n,1,1,1,1)))
        np.savez(p/'mano_transforms.npz',scales=1.,rotations=[[0,0,0],[0,0,np.pi/2],[0,0,0],[0,0,0]],translations=np.zeros((n,3)))
        b=VisualizationLoader(p,start_frame=1,end_frame=2)
        b.device='cpu';b.use_parametric_mano=True
        b.mano_model=lambda **kw:SimpleNamespace(vertices=torch.tensor([[[1.,0.,1.]]]),joints=torch.zeros((1,21,3)))
        b._load_mano_data()
        np.testing.assert_allclose(b.get_mano_vertices(0),[[0,1,1]],atol=1e-6)

    def test_gif_advances_geometry_and_preserves_frame_durations(self):
        calls=[];b=SimpleNamespace(num_frames=3,timestamps=[0.,.04,.12])
        def render(i):
            calls.append(i);return np.full((8,8,3),i*90,np.uint8)
        data=encode_calibrated_gif(b,render)
        self.assertEqual(calls,[0,1,2])
        img=Image.open(io.BytesIO(data));durations=[];colors=[]
        for frame in ImageSequence.Iterator(img):
            durations.append(frame.info['duration']);colors.append(np.asarray(frame.convert('RGB'))[0,0,0])
        self.assertEqual(colors,[0,90,180]);self.assertEqual(durations,[40,80,80])

    def test_distance_colors_have_fixed_mm_scale_and_invalid_pixels(self):
        a=distance_heatmap(np.array([[.01,.03,np.nan,0.]]))
        b=distance_heatmap(np.array([[.01,.3,np.nan,0.]]))
        np.testing.assert_array_equal(a[0,0],b[0,0])
        np.testing.assert_array_equal(a[0,2:],np.zeros((2,3),dtype=np.uint8))
        zero=distance_heatmap(np.zeros((1,1)),valid_mask=np.ones((1,1),bool))
        self.assertGreater(zero.sum(),0)

if __name__=='__main__':unittest.main()
