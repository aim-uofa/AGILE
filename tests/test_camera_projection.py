import unittest
import torch
from optimization.utils.camera_utils import convert_camtoworld_to_pytorch3d_cameras


class CameraProjectionTests(unittest.TestCase):
    def test_renderer_matches_opencv_pixels_with_off_center_principal_point(self):
        K = torch.tensor([[[610., 0., 305.], [0., 615., 228.], [0., 0., 1.]]])
        points = torch.tensor([[[.1, -.1, .8], [-.1, .05, 1.2], [0., 0., 1.]]])
        camera = convert_camtoworld_to_pytorch3d_cameras(torch.eye(4)[None], K, 480, 640, 'cpu')
        expected = points @ K.transpose(1, 2)
        expected = expected[..., :2] / expected[..., 2:]
        actual = camera.transform_points_screen(points)[..., :2]
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-5)

    def test_nonidentity_extrinsics_and_intrinsic_gradients(self):
        angle = torch.tensor(.4)
        c, s = torch.cos(angle), torch.sin(angle)
        world_to_cam = torch.tensor([[c,0.,s,.1],[0.,1.,0.,-.05],[-s,0.,c,.2],[0.,0.,0.,1.]])[None]
        K = torch.tensor([[[610.,0.,305.],[0.,615.,228.],[0.,0.,1.]]], requires_grad=True)
        points = torch.tensor([[[.1,-.1,.8],[-.1,.05,1.2]]])
        camera = convert_camtoworld_to_pytorch3d_cameras(torch.linalg.inv(world_to_cam), K, 480, 640, 'cpu')
        camera_points = points @ world_to_cam[:,:3,:3].transpose(1,2) + world_to_cam[:,:3,3][:,None]
        expected = camera_points @ K.transpose(1,2)
        expected = expected[...,:2] / expected[...,2:]
        actual = camera.transform_points_screen(points)[...,:2]
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-5)
        grad_actual, = torch.autograd.grad(actual.sum(), K, retain_graph=True)
        grad_expected, = torch.autograd.grad(expected.sum(), K)
        # The supported four camera parameters retain the correct gradients.
        for row,col in [(0,0),(1,1),(0,2),(1,2)]:
            torch.testing.assert_close(grad_actual[:,row,col], grad_expected[:,row,col])


if __name__ == '__main__': unittest.main()
