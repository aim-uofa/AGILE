from types import SimpleNamespace
import unittest

from optimization.config.config import Config
from optimization.optim.object_optimizer import ObjectOptimizer


class MissingHandTests(unittest.TestCase):
    def test_stale_hand_buffers_do_not_supervise_an_absent_hand(self):
        runner = SimpleNamespace(device='cpu', hand_valid_current_frame=False,
                                 hand_valid_previous_frame=True, hand_zbuf=object(),
                                 last_hand_verts_incam=object(), last_hand_verts_in_ob=object(),
                                 hand_verts_dist_weights=object())
        optimizer = ObjectOptimizer(Config(object_contact_loss=True), runner)
        config = optimizer._make_loss_cfg(1, 200, False, False, False)
        self.assertFalse(config.object_contact_loss)
        self.assertFalse(config.ho_relpose_loss)
        runner.hand_valid_current_frame = True
        runner.hand_valid_previous_frame = False
        config = optimizer._make_loss_cfg(1, 200, False, False, False)
        self.assertTrue(config.object_contact_loss)
        self.assertFalse(config.ho_relpose_loss)
        runner.hand_valid_previous_frame = True
        config = optimizer._make_loss_cfg(1, 200, False, False, False)
        self.assertTrue(config.object_contact_loss)
        self.assertTrue(config.ho_relpose_loss)


if __name__ == '__main__':
    unittest.main()
