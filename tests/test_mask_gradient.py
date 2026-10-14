import unittest
import torch
from optimization.losses.mask import _mask_loss


class MaskGradientTests(unittest.TestCase):
    def test_iou_has_gradient_toward_foreground_and_away_from_background(self):
        alpha = torch.tensor([[.3,.6]], requires_grad=True)
        target = torch.tensor([[1.,0.]])
        loss = _mask_loss(alpha,target,torch.ones_like(alpha),
                          (alpha > 1e-3) & target.bool(),torch.zeros_like(alpha),'iou',1.)
        self.assertTrue(loss.requires_grad)
        loss.backward()
        self.assertLess(alpha.grad[0,0].item(),0.)
        self.assertGreater(alpha.grad[0,1].item(),0.)

    def test_iou_ignores_occluded_pixels(self):
        alpha = torch.tensor([[.3,.6]], requires_grad=True)
        target = torch.tensor([[1.,0.]])
        loss = _mask_loss(alpha,target,torch.ones_like(alpha),target.bool(),
                          torch.tensor([[0.,1.]]),'iou',1.)
        loss.backward()
        self.assertEqual(alpha.grad[0,1].item(),0.)


if __name__ == '__main__': unittest.main()
