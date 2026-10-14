import unittest

import torch

from optimization.utils.feature import feature_loss, feature_loss_global


class FeatureBoundaryTests(unittest.TestCase):
    def test_local_and_global_reject_coordinates_outside_the_image(self):
        points = torch.tensor([
            [1., 1.], [6., 1.], [1., -1.], [1., 6.], [-1., 1.],
            [-.2, 1.], [1., -.2], [3.999, 1.], [4., 1.],
        ])
        expected = [True, False, False, False, False, False, False, True, False]
        for function in (feature_loss, feature_loss_global):
            with self.subTest(function=function.__name__):
                _, info = function(torch.ones((len(points), 2, 2)), points, torch.zeros((4, 4)))
                self.assertEqual(info['valid_pts_mask'].bool().tolist(), expected)

    def test_occluded_points_do_not_contribute(self):
        mask = torch.zeros((4, 4))
        mask[1, 1] = 1
        for function in (feature_loss, feature_loss_global):
            loss, info = function(torch.ones((1, 2, 2)), torch.tensor([[1., 1.]]), mask)
            self.assertFalse(info['valid_pts_mask'][0])
            self.assertEqual(loss.item(), 0.)

    def test_empty_samples_have_finite_differentiable_zero_loss(self):
        for function in (feature_loss, feature_loss_global):
            points = torch.empty((0, 2), requires_grad=True)
            loss, info = function(torch.empty((0, 2, 2)), points, torch.zeros((4, 4)))
            self.assertEqual(loss.item(), 0.)
            self.assertEqual(tuple(info['valid_pts_mask'].shape), (0,))
            loss.backward()
            self.assertIsNotNone(points.grad)


if __name__ == '__main__':
    unittest.main()
