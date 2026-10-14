"""Regression coverage for the half-turn that corrupted MC1 frame 610."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from optimization.utils.geometry import rotation_matrix_to_rvec


# Recovered from the saved training geometry at frame 610. Its small float32
# orthogonality error is deliberate: a perfect analytic matrix missed the bug.
MC1_HALF_TURN = np.array([
    [-0.45582056116961106, -0.8812632549072186, -0.12490952893471935],
    [-0.8812625450443689, 0.4271481644042004, 0.20228843213199144],
    [-0.12491450597985608, 0.202285365352956, -0.9713271858398146],
], dtype=np.float32)


class RotationGeometryTests(unittest.TestCase):
    def test_real_half_turn_is_not_collapsed_to_identity(self):
        vector = rotation_matrix_to_rvec(torch.from_numpy(MC1_HALF_TURN))
        restored = Rotation.from_rotvec(vector.numpy())
        reference = Rotation.from_matrix(MC1_HALF_TURN)
        self.assertGreater(np.rad2deg(np.linalg.norm(vector.numpy())), 179.99)
        error = np.rad2deg((restored.inv() * reference).magnitude())
        self.assertLess(error, 0.001)

    def test_round_trip_across_half_turn_and_small_angles(self):
        axes = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., -1.],
                         [1., -2., 3.], [-2., -1., .5]])
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)
        degrees = [0., 1e-7, .01, 90., 179., 179.99, 179.999831,
                   180., 180.000169, 180.01, 181.]
        vectors = (axes[:, None] * np.deg2rad(degrees)[None, :, None]).reshape(-1, 3)
        matrices = Rotation.from_rotvec(vectors).as_matrix()
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            for dtype in (torch.float32, torch.float64):
                with self.subTest(device=device, dtype=dtype):
                    source = torch.tensor(matrices, dtype=dtype, device=device)
                    actual = rotation_matrix_to_rvec(source)
                    self.assertEqual(actual.dtype, source.dtype)
                    self.assertEqual(actual.device, source.device)
                    self.assertEqual(actual.shape, (len(matrices), 3))
                    reconstructed = Rotation.from_rotvec(actual.cpu().numpy())
                    error = (reconstructed.inv() * Rotation.from_matrix(matrices)).magnitude()
                    tolerance = 1e-6 if dtype == torch.float32 else 1e-12
                    self.assertLess(float(error.max()), tolerance)

    def test_gradients_are_finite_at_identity_and_near_half_turn(self):
        axis = np.array([1., -2., 3.]) / np.sqrt(14.)
        for angle in (0., 1e-7, np.pi - 1e-4, np.pi + 1e-4):
            with self.subTest(angle=angle):
                source = torch.tensor(Rotation.from_rotvec(axis * angle).as_matrix(),
                                      dtype=torch.float64, requires_grad=True)
                value = rotation_matrix_to_rvec(source)
                value.sum().backward()
                self.assertTrue(torch.isfinite(source.grad).all())
                # Away from the unavoidable axis-sign discontinuity at exactly
                # pi, this must remain a differentiable Torch operation.
                self.assertTrue(torch.autograd.gradcheck(rotation_matrix_to_rvec,
                                                         (source,), eps=1e-7,
                                                         atol=2e-5, rtol=1e-4))


if __name__ == '__main__':
    unittest.main()
