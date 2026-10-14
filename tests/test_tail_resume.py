"""Fail-closed checks for recovering a legacy optimization boundary."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from optimization.config.config import Config
from scripts.train.resume_tail import (anchor_index, fit_ordered_rigid_pose, load_resume_config,
                                       native_proxy_path, preserve_anchor_pcd)


class TailResumeTests(unittest.TestCase):
    def test_anchor_uses_original_frame_ids_and_excludes_anchor_from_work(self):
        frame_ids = np.arange(0, 900, 5)
        index = anchor_index(frame_ids, 600)
        self.assertEqual(index, 120)
        self.assertEqual(frame_ids[index + 1], 605)
        self.assertEqual(len(frame_ids[index + 1:]), 59)
        for invalid in (601, 895):
            with self.assertRaises(ValueError):
                anchor_index(frame_ids, invalid)
        with self.assertRaises(ValueError):
            anchor_index([0, 10, 5], 5)

    def test_recovers_half_turn_geometry_without_using_corrupt_export(self):
        points = np.random.default_rng(3).normal(size=(80, 3)) * [.04, .07, .02]
        rotation = Rotation.from_rotvec(np.array([1., 2., 3.]) / np.sqrt(14) * (np.pi - 3e-6)).as_matrix()
        translation = np.array([.04, -.1, .55])
        # The training PLY stores float32 coordinates; preserve that boundary.
        target = (points @ rotation.T + translation).astype(np.float32)
        pose, error = fit_ordered_rigid_pose(points, target)
        self.assertLess(error, 1e-7)
        self.assertLess(Rotation.from_matrix(pose[:3, :3].T @ rotation).magnitude(), 1e-6)
        np.testing.assert_allclose(pose[:3, 3], translation, atol=1e-7)
        self.assertAlmostEqual(np.linalg.det(pose[:3, :3]), 1.)

    def test_changed_vertex_order_or_scale_is_rejected(self):
        points = np.random.default_rng(9).normal(size=(80, 3)) * .1
        for bad_target in (points[::-1], points * 1.01, points[:20]):
            with self.assertRaises(ValueError):
                fit_ordered_rigid_pose(points, bad_target)

    def test_cannot_fit_an_ambiguous_planar_anchor(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0.]])
        with self.assertRaisesRegex(ValueError, 'non-planar'):
            fit_ordered_rigid_pose(points, points)

    def test_original_optimization_config_is_preserved(self):
        cfg = Config(start_frame=10, bidirectional_optimization=True,
                     object_max_iter_per_frame=400, object_max_iter_first_frame=800,
                     hand_max_iter_per_frame=200, fix_hand_rotation_to_wilor=True,
                     hand_contact_loss=False, object_contact_loss=False,
                     result_dir='/original')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            original = json.loads(json.dumps(asdict(cfg)))
            path.write_text(json.dumps(original))
            resumed, source = load_resume_config(path, Path(directory) / 'new')
            self.assertEqual(source, original)
            self.assertEqual(json.loads(path.read_text()), original)
            resumed_json = json.loads(json.dumps(asdict(resumed)))
            self.assertEqual({k: v for k, v in resumed_json.items() if k != 'result_dir'},
                             {k: v for k, v in original.items() if k != 'result_dir'})

    def test_unsaved_appearance_state_cannot_be_silently_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            cfg = Config(fix_hand_rotation_to_wilor=True, optimize_object_texture=True)
            path.write_text(json.dumps(asdict(cfg)))
            with self.assertRaisesRegex(ValueError, 'optimize_object_texture'):
                load_resume_config(path, Path(directory) / 'new')

    def test_asset_descriptor_selects_proxy_even_when_display_mesh_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            aligned = Path(directory).resolve()
            legacy = aligned / 'object_mesh.obj'
            legacy.write_text('different high-quality display geometry')
            self.assertEqual(native_proxy_path(aligned), legacy)
            proxy = aligned / 'object_assets' / 'optimization_proxy.glb'
            proxy.parent.mkdir()
            proxy.write_bytes(b'original optimization proxy')
            descriptor = dict(schema_version=1, object_scale_baked=False, variants={
                'display': dict(mesh='object_mesh.obj'),
                'optimization_proxy': dict(mesh='object_assets/optimization_proxy.glb')})
            path = aligned / 'object_assets.json'
            path.write_text(json.dumps(descriptor))
            self.assertEqual(native_proxy_path(aligned), proxy)
            descriptor['object_scale_baked'] = True
            path.write_text(json.dumps(descriptor))
            with self.assertRaisesRegex(ValueError, 'unscaled'):
                native_proxy_path(aligned)
            descriptor['object_scale_baked'] = False
            descriptor['variants']['optimization_proxy']['mesh'] = '../escape.glb'
            path.write_text(json.dumps(descriptor))
            with self.assertRaisesRegex(ValueError, 'contained'):
                native_proxy_path(aligned)

    def test_verified_anchor_is_retained_without_overwriting_another_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'frame_0600_ho_pcd.ply'
            source.write_bytes(b'verified ordered training points')
            target = preserve_anchor_pcd(source, Path(directory) / 'new')
            self.assertEqual(source.read_bytes(), target.read_bytes())
            self.assertEqual(preserve_anchor_pcd(source, Path(directory) / 'new'), target)
            target.write_bytes(b'different capture')
            with self.assertRaisesRegex(ValueError, 'different existing'):
                preserve_anchor_pcd(source, Path(directory) / 'new')
            self.assertEqual(target.read_bytes(), b'different capture')


if __name__ == '__main__':
    unittest.main()
