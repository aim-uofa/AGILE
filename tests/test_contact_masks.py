import unittest
import torch
from optimization.losses.contact import zbuf_contact_loss_simplified


class ContactMaskTests(unittest.TestCase):
    def test_unrelated_occlusion_pixels_cannot_change_other_pixel_gradient(self):
        hand = torch.tensor([[[.50, .52], [.60, .62]]])
        obj = torch.tensor([[[.56, .58], [.55, .57]]], requires_grad=True)
        mask_a = torch.tensor([[True, False]])
        mask_b = torch.tensor([[False, True]])
        mixed, _ = zbuf_contact_loss_simplified(hand, obj, mask_a, mask_b)
        only_a, _ = zbuf_contact_loss_simplified(hand, obj, mask_a, torch.zeros_like(mask_a))
        only_b, _ = zbuf_contact_loss_simplified(hand, obj, torch.zeros_like(mask_a), mask_b)
        expected = (only_a + only_b) / 2
        torch.testing.assert_close(mixed, expected)
        grad_mix, = torch.autograd.grad(mixed, obj, retain_graph=True)
        grad_expected, = torch.autograd.grad(expected, obj)
        torch.testing.assert_close(grad_mix, grad_expected)

    def test_invalid_fragments_do_not_dilute_valid_contact(self):
        hand = torch.tensor([[[.50, .52], [-1., -1.]]])
        obj = torch.tensor([[[.56, .58], [.55, .57]]], requires_grad=True)
        loss, _ = zbuf_contact_loss_simplified(hand, obj, torch.tensor([[True, True]]), torch.tensor([[False, False]]))
        reference, _ = zbuf_contact_loss_simplified(hand[:,:1], obj[:,:1], torch.tensor([[True]]), torch.tensor([[False]]))
        torch.testing.assert_close(loss, reference)


if __name__ == '__main__':
    unittest.main()
