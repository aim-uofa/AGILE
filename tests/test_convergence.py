"""Prevent small learning rates and objective switches from faking convergence."""
import unittest
import torch
from optimization.optim.convergence import ConvergenceMonitor, optimizer_snapshot, optimizer_step_norm
from optimization.utils.geometry import so3_log_batch, so3_exp_batch
from tests.test_rotation_geometry import MC1_HALF_TURN
from scipy.spatial.transform import Rotation
import numpy as np


class ConvergenceTests(unittest.TestCase):
    def test_tiny_steps_with_large_gradient_are_stalled(self):
        monitor = ConvergenceMonitor(1e-4)
        for _ in range(10):
            status = monitor.observe(12., 1e-9, 2.)
        self.assertEqual(status, 'stalled')

    def test_stationarity_requires_a_full_stable_phase(self):
        monitor = ConvergenceMonitor(1e-4)
        for _ in range(5):
            status = monitor.observe(.5, 1e-6, 1e-5, 'first')
        self.assertEqual(status, 'converged')
        self.assertEqual(monitor.observe(.5, 1e-6, 1e-5, 'second'), 'optimizing')
        monitor.reset()
        self.assertEqual(monitor.observe(.5, 1e-6, 1e-5), 'optimizing')

    def test_changing_objective_is_not_convergence(self):
        monitor = ConvergenceMonitor(1e-4)
        for value in range(5):
            status = monitor.observe(float(value), 1e-7, 1e-5)
        self.assertNotEqual(status, 'converged')
        with self.assertRaises(FloatingPointError):
            monitor.observe(float('nan'), 0., 0.)

    def test_optimizer_snapshot_measures_all_updates(self):
        x = torch.tensor([1., 2.], requires_grad=True)
        optimizer = torch.optim.SGD([x], lr=.1)
        x.square().sum().backward()
        before, gradient = optimizer_snapshot([optimizer])
        optimizer.step()
        self.assertEqual(gradient, 4.)
        self.assertAlmostEqual(optimizer_step_norm([optimizer], before), np.sqrt(.2**2+.4**2), places=6)

    def test_batch_log_preserves_the_real_half_turn(self):
        rotations = torch.tensor(MC1_HALF_TURN)[None]
        vectors = so3_log_batch(rotations)
        actual = Rotation.from_rotvec(vectors.numpy())
        expected = Rotation.from_matrix(MC1_HALF_TURN)
        self.assertLess(float(np.rad2deg((actual.inv()*expected).magnitude()).max()), .001)
        self.assertTrue(torch.isfinite(so3_log_batch(torch.eye(3)[None])).all())


if __name__ == '__main__':
    unittest.main()
