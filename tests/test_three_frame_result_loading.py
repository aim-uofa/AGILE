"""Real NPZ loader regressions for the ambiguous three-frame axis-angle shape."""
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
import trimesh
from optimization.evaluation.ho3d_io import load_results_data

class ThreeFrameResultLoading(unittest.TestCase):
    def fixture(self,n,vectors,native=True):
        t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);p=Path(t.name)
        ids=np.arange(n)*5+60
        np.savez(p/'camera_data.npz',intrinsic=np.eye(3),cam_c2w=np.tile(np.eye(4),(n,1,1)),prediction_frame_ids=ids)
        np.savez(p/'mano_faces.npz',mano_faces=np.array([[0,1,2]]))
        data=dict(scales=1.,translations=np.zeros((n,3)),rotations=vectors)
        if native:data['rotation_matrices']=np.stack([cv2.Rodrigues(v)[0] for v in vectors])
        np.savez(p/'mano_transforms.npz',**data)
        np.savez(p/'object_poses.npz',object_poses_in_cam=np.tile(np.eye(4),(n,1,1)),object_scale=1.)
        np.savez(p/'optimized_mano_joints.npz',mano_joints=np.arange(n*21*3,dtype=np.float32).reshape(n,1,21,3)*.001)
        np.savez(p/'optimized_mano_verts.npz',mano_verts=np.arange(n*778*3,dtype=np.float32).reshape(n,1,778,3)*.001)
        trimesh.creation.box().export(p/'object_mesh.obj')
        return p
    def test_three_fixed_rotations_use_native_identity(self):
        p=self.fixture(3,np.zeros((3,3)))
        pred,n=load_results_data(str(p))
        self.assertEqual(n,3);self.assertEqual(pred['frame_ids'].tolist(),[60,65,70])
        self.assertTrue(np.isfinite(pred['ori_v3d_c.right'].numpy()).all())
    def test_three_nonzero_rotations_use_native_matrices(self):
        v=np.array([[0.,.2,0.],[0.,np.pi-1e-6,0.],[.3,0.,0.]])
        p=self.fixture(3,v);pred,n=load_results_data(str(p));self.assertEqual(n,3)
        with np.load(p/'optimized_mano_joints.npz') as z:joint=z['mano_joints'][1,0]
        expected=joint@np.linalg.inv(cv2.Rodrigues(v[1])[0])
        np.testing.assert_allclose(pred['ori_j3d_c.right'][1].numpy(),expected,atol=1e-7)
    def test_existing_fifteen_frame_metric_inputs_unchanged(self):
        v=np.tile([.01,.02,.03],(15,1));a,_=load_results_data(str(self.fixture(15,v,True)))
        b,_=load_results_data(str(self.fixture(15,v,False)))
        for key in ['v3d_right.object','v3d_c.object','j3d_c.right','j3d_ra.right','frame_ids']:
            np.testing.assert_array_equal(a[key].numpy(),b[key].numpy())
    def test_invalid_native_length_is_rejected(self):
        p=self.fixture(3,np.zeros((3,3)))
        np.savez(p/'mano_transforms.npz',scales=1.,translations=np.zeros((3,3)),rotations=np.zeros((3,3)),rotation_matrices=np.eye(3)[None])
        with self.assertRaisesRegex(ValueError,'Invalid native'):load_results_data(str(p))

if __name__=='__main__':unittest.main()
