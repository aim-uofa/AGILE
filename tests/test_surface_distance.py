"""Small real-world triangles must not collapse distant points to contact."""
import unittest
import numpy as np
import trimesh
from optimization.data.surface_distance import SurfaceDistance


class SurfaceDistanceTests(unittest.TestCase):
    def test_tiny_triangles_and_rigid_motion_match_independent_geometry(self):
        vertices=np.array([[0.,0.,0.],[.00001,0,0],[0,.00001,0], [.1,.1,0],[.1001,.1,0],[.1,.1001,0]])
        faces=np.array([[0,1,2],[3,4,5]])
        points=np.array([[.05,.05,.4],[.000001,.000001,.01],[.099,.099,.002]])
        mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
        _, expected, _=trimesh.proximity.closest_point_naive(mesh,points)
        pose=trimesh.transformations.rotation_matrix(.7,[1.,2.,3.]);pose[:3,3]=[.7,-.1,.9]
        camera=points@pose[:3,:3].T+pose[:3,3]
        actual=SurfaceDistance(vertices,faces).millimetres(camera,pose)
        np.testing.assert_allclose(actual,expected*1000,atol=.0001,rtol=1e-5)
        self.assertGreater(actual[0],400.)


if __name__=='__main__':unittest.main()
