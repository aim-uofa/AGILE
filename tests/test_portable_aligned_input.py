"""Aligned inputs must load without a lab folder name or external GT/cache files."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from optimization.data.megasam import Parser


class PortableAlignedInputTests(unittest.TestCase):
    def test_local_wilor_and_rgb_sampling(self):
        with tempfile.TemporaryDirectory(prefix='portable-input-') as temporary:
            root=Path(temporary)
            rgbd=np.arange(6*2*3*4,dtype=np.float32).reshape(6,2,3,4)
            np.savez(root/'camera_data.npz',intrinsic=np.eye(3),rgbd=rgbd,cam_c2w=np.repeat(np.eye(4)[None],6,axis=0))
            (root/'wilor').mkdir()
            joints=np.arange(6*21*2).reshape(6,1,21,2)
            np.savez(root/'wilor/mano_joints_2d.npz',mano_joints_2d=joints)
            parser=Parser(str(root),online_start_frame=0,online_end_frame=6,hold_mode=True,hold_mode_interval=5)
            np.testing.assert_array_equal(parser.frame_ids,[0,5])
            np.testing.assert_array_equal(parser.rgbd,rgbd[[0,5]])
            np.testing.assert_array_equal(parser.rgbs,rgbd[[0,5],...,:3])
            np.testing.assert_array_equal(parser.mano_joints_2d,joints[[0,5],0])
            self.assertIsNone(parser.normals)
            self.assertIsNone(parser.object_mesh)


if __name__=='__main__': unittest.main()
