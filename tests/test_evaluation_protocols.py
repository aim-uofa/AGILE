"""Known-answer, failure and actual MANO camera-conversion regressions (CPU)."""
import copy
import json
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch
import sys
import runpy

import cv2
import numpy as np
import open3d as o3d
import torch
import trimesh
import yaml

from optimization.evaluation.eval_utils import (calculate_chamfer_f_scores, eval_cd_f_right,
    eval_cd_f_right_icp, eval_mpjpe_right, eval_mpjpe_right_scaled, eval_cd_icp_first_frame)
from optimization.evaluation.hold_tools.icp import compute_icp_metrics, RegistrationError
from optimization.evaluation.ho3d_io import load_gt_data, load_results_data
from optimization.evaluation.interaction import hand_similarity
from optimization.evaluation.protocols import validate_inputs, reduce_metrics, strict_mean, hand_sim3_frame_mask
from optimization.models.mano_wrapper import MANO
from optimization.utils.dataset_utils import Ho3dReader
from optimization.utils.geometry import so3_exp_batch
from optimization.utils.mano_coordinates import convert_mano_camera, OPENGL_TO_OPENCV as C
from scripts.experiments import evaluate_protocol as protocol_entry


def sequence(n=2):
    rng = np.random.default_rng(42)
    joints = torch.tensor(rng.normal(size=(n, 21, 3)) * .03, dtype=torch.float64)
    vertices = torch.tensor(rng.normal(size=(n, 60, 3)) * .01 + [.1, .03, .5], dtype=torch.float64)
    data = {'j3d_c.right': joints, 'j3d_ra.right': joints - joints[:, :1],
            'ori_j3d_c.right': joints.clone(), 'v3d_c.right': joints.clone(),
            'v3d_c.object': vertices, 'v3d_ra.object': vertices - vertices.mean(1, keepdim=True),
            'frame_ids': torch.arange(n) * 5, 'is_valid': torch.ones(n, dtype=torch.int32),
            'full_seq_name': 'synthetic'}
    return copy.deepcopy(data), data


class MetricTests(unittest.TestCase):
    def test_squared_bidirectional_cd_units(self):
        p, g = np.array([[0., 0., 0.]]), np.array([[.01, 0., 0.]])
        cd, f5, f10 = calculate_chamfer_f_scores(p, g)
        self.assertAlmostEqual(cd, 2.)
        self.assertEqual(f5, 0.)
        self.assertEqual(f10, 0.)  # HOLD uses strict '<', including threshold equality.

    def test_identical_prediction_has_zero_canonical_mpjpe(self):
        p, g = sequence()
        self.assertEqual(reduce_metrics(eval_mpjpe_right(p, g, {}), g)['mpjpe_ra_r'], 0.)

    def test_wrist_alignment_removes_only_common_translation(self):
        p, g = sequence()
        for key in ['v3d_c.object', 'j3d_c.right', 'v3d_c.right']:
            p[key] += torch.tensor([.3, -.1, .2])
        result = reduce_metrics(eval_cd_f_right(p, g, {}), g)
        self.assertLess(result['cd_right'], 1e-20)

    def test_hand_similarity_and_root_only_are_distinct(self):
        p, g = sequence()
        rotation = cv2.Rodrigues(np.array([.5, -.3, .2]))[0]
        for key in ['v3d_c.object', 'j3d_c.right', 'v3d_c.right']:
            p[key] = p[key] @ torch.tensor(rotation.T) * 1.8 + .2
        sim3 = reduce_metrics(eval_cd_f_right_icp(p, g, {}), g)
        root = reduce_metrics(eval_cd_f_right(p, g, {}), g)
        self.assertLess(sim3['cd_interaction_hand_sim3_per_frame'], 1e-6)
        self.assertGreater(root['cd_right'], 1.)
        self.assertEqual(sim3['cd_interaction_hand_sim3_per_frame'], sim3['cd_right_icp'])

    def test_interaction_sampling_matches_official_rng_including_invalid(self):
        p, g = sequence(3)
        vertices = torch.tensor(np.random.default_rng(27).normal(size=(3, 3500, 3)) * .01, dtype=torch.float64)
        p['v3d_c.object'] = vertices.clone()
        g['v3d_c.object'] = vertices.clone()
        p['v3d_c.object'] += .001
        g['is_valid'][0] = 0
        torch.manual_seed(1)
        expected = []
        for i in range(3):
            pi = torch.randperm(3500)[:3000]; gi = torch.randperm(3500)[:3000]
            expected.append(calculate_chamfer_f_scores(
                (p['v3d_c.object'][i, pi] - p['j3d_c.right'][i, :1]).numpy(),
                (g['v3d_c.object'][i, gi] - g['j3d_c.right'][i, :1]).numpy())[0])
        actual = eval_cd_f_right(p, g, {})['cd_right']
        self.assertTrue(np.isnan(actual[0]))
        np.testing.assert_allclose(actual[1:], expected[1:], atol=1e-12, rtol=0)

    def test_interaction_does_not_modify_global_torch_rng(self):
        p, g = sequence()
        torch.manual_seed(837)
        before = torch.get_rng_state().clone()
        eval_cd_f_right(p, g, {})
        eval_cd_f_right_icp(p, g, {})
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_sequence_scale_is_one_positive_scalar(self):
        p, g = sequence()
        p['j3d_ra.right'] *= 2
        result = reduce_metrics(eval_mpjpe_right_scaled(p, g, {}), g)
        self.assertLess(result['mpjpe_right_scaled'], 1e-5)

    def test_negative_auxiliary_scale_is_rejected(self):
        p, g = sequence()
        p['j3d_ra.right'] *= -1
        with self.assertRaisesRegex(ValueError, 'positive'):
            eval_mpjpe_right_scaled(p, g, {})

    def test_degenerate_hand_similarity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Degenerate'):
            hand_similarity(np.zeros((21, 3)), np.ones((21, 3)))

    def test_nan_prediction_on_valid_gt_fails(self):
        p, g = sequence()
        p['v3d_c.object'][0, 0, 0] = float('nan')
        for function in [eval_cd_f_right, eval_cd_f_right_icp, eval_mpjpe_right]:
            with self.assertRaisesRegex(ValueError, 'non-finite'):
                function(p, g, {})

    def test_inf_gt_on_valid_frame_fails(self):
        p, g = sequence()
        g['j3d_c.right'][0, 0, 0] = float('inf')
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            validate_inputs(p, g)

    def test_invalid_gt_nan_does_not_hide_valid_frame(self):
        p, g = sequence()
        g['is_valid'][0] = 0
        p['v3d_c.object'][0] = float('nan')
        coverage = validate_inputs(p, g)
        self.assertEqual(coverage['valid_GT_prediction_coverage'], 1.)
        self.assertEqual(coverage['finite_prediction_frames'], 1)
        self.assertEqual(reduce_metrics(eval_cd_f_right(p, g, {}), g)['cd_right'], 0.)

    def test_metric_nan_on_valid_frame_cannot_be_reduced(self):
        with self.assertRaisesRegex(ValueError, 'silently omit'):
            strict_mean(np.array([np.nan, 1.]), np.array([True, True]), 'CD')

    def test_no_valid_gt_fails(self):
        p, g = sequence()
        g['is_valid'][:] = 0
        with self.assertRaisesRegex(ValueError, 'at least one'):
            validate_inputs(p, g)

    def test_frame_ids_shift_fails(self):
        p, g = sequence()
        p['frame_ids'] += 5
        with self.assertRaisesRegex(ValueError, 'frame_ids mismatch'):
            validate_inputs(p, g, require_frame_ids=True)

    def test_duplicate_frame_id_fails(self):
        p, g = sequence()
        p['frame_ids'][1] = p['frame_ids'][0]
        with self.assertRaisesRegex(ValueError, 'strictly increasing'):
            validate_inputs(p, g, require_frame_ids=True)

    def test_ho3d_protocol_does_not_truncate(self):
        p, g = sequence()
        p['v3d_c.object'] = p['v3d_c.object'][:1]
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            validate_inputs(p, g, require_frame_ids=True)


    def test_geometry_uses_first_valid_anchor(self):
        p, g = sequence()
        g['is_valid'][0] = 0
        p['faces'] = {'object': torch.tensor([[0, 1, 2]])}
        g['faces.object'] = torch.tensor([[0, 1, 2]])
        diag = {}
        with patch('optimization.evaluation.hold_tools.icp.compute_icp_metrics', return_value=(0., 1., 1., {})):
            eval_cd_icp_first_frame(p, g, {}, registration_diagnostics=diag)
        self.assertEqual(diag['anchor_frame_id'], 5)


class RegistrationTests(unittest.TestCase):
    def test_scaled_rotated_translated_shape_recovers(self):
        target = o3d.geometry.TriangleMesh.create_box(.11, .07, .04)
        source = copy.deepcopy(target)
        tf = np.eye(4)
        tf[:3, :3] = cv2.Rodrigues(np.array([.7, -.5, .4]))[0] * 1.8
        tf[:3, 3] = [.3, -.2, .5]
        source.transform(tf)
        before_source = np.asarray(source.vertices).copy()
        before_target = np.asarray(target.vertices).copy()
        cd, _, _, details = compute_icp_metrics(target, source, num_iters=2, no_tqdm=True, return_details=True)
        self.assertLess(cd, 1e-12)
        self.assertAlmostEqual(details['fitted_scale'], 1 / 1.8, places=8)
        np.testing.assert_array_equal(source.vertices, before_source)
        np.testing.assert_array_equal(target.vertices, before_target)

    def test_registration_is_repeatable(self):
        target = o3d.geometry.TriangleMesh.create_box(.1, .07, .05)
        source = copy.deepcopy(target).scale(2, np.zeros(3))
        a = compute_icp_metrics(target, source, num_iters=3, no_tqdm=True, return_details=True)
        b = compute_icp_metrics(target, source, num_iters=3, no_tqdm=True, return_details=True)
        np.testing.assert_allclose(a[:3], b[:3], rtol=0, atol=1e-12)
        self.assertEqual(a[3].keys(), b[3].keys())
        for key, value in a[3].items():
            if isinstance(value, float):
                self.assertAlmostEqual(value, b[3][key], places=12)
            elif key in ('centered_transformation', 'transformation'):
                np.testing.assert_allclose(value, b[3][key], rtol=0, atol=1e-12)
            else:
                self.assertEqual(value, b[3][key])

    def test_failed_ransac_keeps_valid_pca_result_and_records_error(self):
        target = o3d.geometry.TriangleMesh.create_box(.1, .07, .05)
        source = copy.deepcopy(target).scale(2, np.zeros(3))
        with patch('optimization.evaluation.hold_tools.icp.execute_global_registration', side_effect=RuntimeError('fixture failure')):
            cd, _, _, d = compute_icp_metrics(target, source, num_iters=2, no_tqdm=True, return_details=True)
        self.assertLess(cd, 1e-12)
        self.assertEqual(d['failed_candidate_count'], 2)

    def test_all_icp_failures_raise_clear_error(self):
        mesh = o3d.geometry.TriangleMesh.create_box(.1, .07, .05)
        with patch('open3d.pipelines.registration.registration_icp', side_effect=RuntimeError('fixture failure')):
            with self.assertRaisesRegex(RegistrationError, 'All registration candidates failed'):
                compute_icp_metrics(mesh, mesh, num_iters=1, no_tqdm=True)

    def test_empty_mesh_fails(self):
        with self.assertRaisesRegex(RegistrationError, 'empty'):
            compute_icp_metrics(o3d.geometry.TriangleMesh(), o3d.geometry.TriangleMesh(), no_tqdm=True)

    def test_nonfinite_mesh_fails(self):
        mesh = o3d.geometry.TriangleMesh.create_box(.1, .07, .05)
        bad = copy.deepcopy(mesh)
        arr = np.asarray(bad.vertices).copy(); arr[0, 0] = np.nan
        bad.vertices = o3d.utility.Vector3dVector(arr)
        with self.assertRaisesRegex(RegistrationError, 'non-finite'):
            compute_icp_metrics(mesh, bad, no_tqdm=True)


class ManoCameraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.environ.get('HO3D_MANO_ROOT'):
            raise unittest.SkipTest('Set HO3D_MANO_ROOT to licensed MANO assets for camera regressions')
        root = Path(os.environ['HO3D_MANO_ROOT'])
        cfg = yaml.safe_load((root / 'mano_config.yaml').read_text())['MANO']
        cls.model = MANO(**{k.lower(): v for k, v in cfg.items()}).to('cpu')

    def test_camera_conversion_matches_transforming_all_generated_vertices(self):
        rng = np.random.default_rng(19)
        for _ in range(3):
            pose = rng.normal(size=48).astype(np.float32) * .3
            beta = rng.normal(size=10).astype(np.float32)
            translation = np.array([.03, -.1, -.5], dtype=np.float32)
            converted_pose, converted_t = convert_mano_camera(pose, translation, beta, self.model)
            def forward(p, t):
                rotations = so3_exp_batch(torch.tensor(p.reshape(1, 16, 3)))
                return self.model(betas=torch.tensor(beta)[None], global_orient=rotations[:, :1],
                                  hand_pose=rotations[:, 1:], transl=torch.tensor(t, dtype=torch.float32)[None])
            with torch.no_grad():
                raw, converted = forward(pose, translation), forward(converted_pose, converted_t)
            np.testing.assert_allclose(converted.vertices, raw.vertices.numpy() @ C.T, atol=3e-7, rtol=0)
            np.testing.assert_allclose(converted.joints, raw.joints.numpy() @ C.T, atol=3e-7, rtol=0)

    def test_reader_applies_rest_wrist_correction(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); (base / 'rgb').mkdir(); (base / 'meta').mkdir()
            meta = {'handBeta': np.zeros(10, dtype=np.float32), 'handPose': np.zeros(48, dtype=np.float32),
                    'handTrans': np.array([.1, -.1, -.5], dtype=np.float32), 'handJoints3D': np.zeros((21, 3))}
            (base / 'meta/0000.pkl').write_bytes(pickle.dumps(meta))
            reader = Ho3dReader.__new__(Ho3dReader); reader.color_files = [str(base / 'rgb/0000.jpg')]
            actual = reader.get_mano_data(0, mano_model=self.model)
            _, expected_t = convert_mano_camera(meta['handPose'], meta['handTrans'], meta['handBeta'], self.model)
            np.testing.assert_array_equal(actual['handTrans'], expected_t)
            self.assertGreater(np.linalg.norm(expected_t - C @ meta['handTrans']), .01)

    def test_first_invalid_gt_is_finite_and_covered(self):
        model = self.model
        class Reader:
            K = np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1.]])
            color_files = ['0000.jpg', '', '', '', '', '0005.jpg']
            id_strs = ['0000', '', '', '', '', '0005']
            def __init__(self, path): pass
            def __len__(self): return 6
            def get_gt_mesh(self): return trimesh.creation.box(extents=[.1, .07, .04])
            def get_mano_data(self, i, mano_model=None):
                if i == 0: return None
                pose, t = convert_mano_camera(np.zeros(48, dtype=np.float32), np.array([0., 0., -.5]), np.zeros(10), model)
                return {'handBeta': np.zeros(10), 'handPose': pose, 'handTrans': t}
            def get_gt_pose(self, i): return np.eye(4)
        with patch('optimization.evaluation.ho3d_io.Ho3dReader', Reader), patch('optimization.evaluation.ho3d_io.os.path.exists', return_value=True):
            gt = load_gt_data('fixture', end_frame=6, hold_mode=True)
        self.assertTrue(torch.isfinite(gt['j3d_c.right']).all())
        np.testing.assert_array_equal(gt['is_valid'], [0, 1])
        np.testing.assert_array_equal(gt['frame_ids'], [0, 5])


class SavedPredictionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.aligned = self.base / 'aligned'; self.aligned.mkdir()
        n = 2
        np.savez(self.aligned / 'camera_data.npz', intrinsic=np.eye(3), cam_c2w=np.repeat(np.eye(4)[None], n, axis=0))
        np.savez(self.aligned / 'mano_faces.npz', mano_faces=np.array([[0, 1, 2]]))
        np.savez(self.aligned / 'mano_transforms.npz', scales=1., rotations=np.zeros((n, 3)), translations=np.zeros((n, 3)))
        np.savez(self.aligned / 'object_poses.npz', object_poses_in_cam=np.repeat(np.eye(4)[None], n, axis=0))
        rng = np.random.default_rng(18)
        np.savez(self.aligned / 'optimized_mano_joints.npz', mano_joints=rng.normal(size=(n, 1, 21, 3)))
        np.savez(self.aligned / 'optimized_mano_verts.npz', mano_verts=rng.normal(size=(n, 1, 778, 3)))
        trimesh.creation.box().export(self.aligned / 'object_mesh.obj')
        (self.base / 'config.json').write_text(json.dumps({'start_frame': 5, 'hold_mode': True,
            'hold_mode_interval': 5, 'bidirectional_optimization': False, 'reverse_frame_order': False}))

    def test_inconsistent_saved_arrays_are_rejected(self):
        np.savez(self.aligned / 'object_poses.npz', object_poses_in_cam=np.repeat(np.eye(4)[None], 1, axis=0))
        with self.assertRaisesRegex(ValueError, 'lengths disagree'):
            load_results_data(str(self.aligned))

    def test_legacy_frame_ids_are_explicitly_labelled_as_inferred(self):
        prediction, n = load_results_data(str(self.aligned))
        self.assertEqual(n, 2)
        np.testing.assert_array_equal(prediction['frame_ids'], [5, 10])
        self.assertEqual(prediction['frame_ids_source'], 'inferred-from-saved-config-legacy-output')

    def test_saved_frame_ids_override_config_inference(self):
        np.savez(self.aligned / 'camera_data.npz', intrinsic=np.eye(3), cam_c2w=np.repeat(np.eye(4)[None], 2, axis=0),
                 prediction_frame_ids=np.array([40, 45]))
        prediction, _ = load_results_data(str(self.aligned))
        np.testing.assert_array_equal(prediction['frame_ids'], [40, 45])
        self.assertEqual(prediction['frame_ids_source'], 'saved-prediction-frame-ids')

    def test_invalid_saved_scale_is_rejected(self):
        np.savez(self.aligned / 'mano_transforms.npz', scales=-1., rotations=np.zeros((2, 3)), translations=np.zeros((2, 3)))
        with self.assertRaisesRegex(ValueError, 'scale must be finite and positive'):
            load_results_data(str(self.aligned))


class EvaluationEntryTests(unittest.TestCase):
    def test_failure_keeps_previous_success_and_writes_parseable_record(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest = root / 'manifest.json'; manifest.write_text('{}')
            output = root / 'metrics.json'; output.write_text('{"original": true}')
            argv = ['evaluate', '--dataset', 'ho3d', '--scene', 'fixture', '--result-dir', temp,
                    '--manifest-path', str(manifest), '--output-json', str(output)]
            with patch.object(sys, 'argv', argv), patch.object(protocol_entry, 'code_revision', return_value='fixture'),\
                    patch.object(protocol_entry, 'evaluate_sequence', side_effect=ValueError('bad frame IDs')):
                with self.assertRaises(ValueError): protocol_entry.main()
            self.assertEqual(json.loads(output.read_text()), {'original': True})
            failure = json.loads((root / 'metrics-failure.json').read_text())
            self.assertFalse(failure['success'])
            self.assertIn('bad frame IDs', failure['error'])

    def test_success_backs_up_old_metrics_and_writes_parseable_record(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest = root / 'manifest.json'; manifest.write_text('{}')
            output = root / 'metrics.json'; output.write_text('{"original": true}')
            argv = ['evaluate', '--dataset', 'ho3d', '--scene', 'fixture', '--result-dir', temp,
                    '--manifest-path', str(manifest), '--output-json', str(output)]
            with patch.object(sys, 'argv', argv), patch.object(protocol_entry, 'evaluate_sequence', return_value={'success': True}):
                protocol_entry.main()
            self.assertEqual(json.loads(output.read_text()), {'success': True})
            backups = list(root.glob('metrics-before-*.json'))
            self.assertEqual(len(backups), 1)
            self.assertEqual(json.loads(backups[0].read_text()), {'original': True})

    def test_partial_metric_calculation_retains_defined_scores(self):
        p, g = sequence()
        p['j3d_c.right'][0] = 0; p['j3d_ra.right'][0] = 0; p['v3d_c.right'][0] = 0
        errors = {}
        def geometry(*args, **kwargs): return {'cd_icp': .1, 'f5_icp': 90., 'f10_icp': 95.}
        with patch.object(protocol_entry, 'eval_cd_icp_first_frame', geometry):
            result = protocol_entry.metrics(p, g, metric_errors=errors)
        self.assertEqual(result['cd_icp'], .1)
        self.assertIsNotNone(result['mpjpe_ra_r'])
        self.assertIsNotNone(result['cd_right'])
        self.assertIsNone(result['cd_right_icp'])
        self.assertIn('Frame 0:', errors['eval_cd_f_right_icp']['error'])

    def test_partial_record_does_not_replace_previous_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest = root / 'manifest.json'; manifest.write_text('{}')
            output = root / 'metrics.json'; output.write_text('{"original": true}')
            argv = ['evaluate', '--dataset', 'ho3d', '--scene', 'fixture', '--result-dir', temp,
                    '--manifest-path', str(manifest), '--output-json', str(output)]
            result = {'success': False, 'full': {'cd_right': 1., 'cd_right_icp': None}, 'metric_errors': {'hand': 'degenerate'}}
            with patch.object(sys, 'argv', argv), patch.object(protocol_entry, 'code_revision', return_value='fixture'),\
                    patch.object(protocol_entry, 'evaluate_sequence', return_value=result):
                with self.assertRaises(SystemExit) as exit_result: protocol_entry.main()
            self.assertEqual(exit_result.exception.code, 1)
            self.assertEqual(json.loads(output.read_text()), {'original': True})
            self.assertEqual(json.loads((root / 'metrics-partial.json').read_text())['full'], result['full'])

    def test_nan_prediction_still_fails_before_partial_scoring(self):
        p, g = sequence(); p['v3d_c.object'][0, 0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            protocol_entry.metrics(p, g, metric_errors={})

    def test_explicit_exclusion_is_metric_specific_and_uses_video_id(self):
        p, g = sequence(3)
        p['j3d_c.right'][1] = 0; p['j3d_ra.right'][1] = 0; p['v3d_c.right'][1] = 0
        selected, coverage = hand_sim3_frame_mask(g, [5])
        self.assertEqual(selected['is_valid'].tolist(), [1, 0, 1])
        self.assertEqual(g['is_valid'].tolist(), [1, 1, 1])
        self.assertEqual(coverage['evaluated_frame_ids'], [0, 10])
        self.assertEqual(coverage['evaluated_frames'], 2)
        self.assertAlmostEqual(coverage['valid_GT_frame_coverage'], 2 / 3)
        def geometry(*args, **kwargs): return {'cd_icp': .1, 'f5_icp': 90., 'f10_icp': 95.}
        with patch.object(protocol_entry, 'eval_cd_icp_first_frame', geometry):
            before = protocol_entry.metrics(p, g, metric_errors={})
            after = protocol_entry.metrics(p, g, hand_sim3_excluded_frame_ids=[5])
        for key in ['mpjpe_ra_r', 'mpjpe_right_scaled', 'cd_icp', 'cd_right', 'f5_right', 'f10_right']:
            self.assertEqual(before[key], after[key])
        self.assertLess(after['cd_right_icp'], 1e-6)  # Float32 transform precision, cm².
        self.assertIsNone(before['cd_right_icp'])

    def test_explicit_exclusion_does_not_change_other_frame_samples(self):
        p, g = sequence(3)
        vertices = torch.tensor(np.random.default_rng(84).normal(size=(3, 3100, 3)) * .01,
                                dtype=torch.float64)
        p['v3d_c.object'] = vertices + .001; g['v3d_c.object'] = vertices
        original = eval_cd_f_right_icp(p, g, {})
        selected, _ = hand_sim3_frame_mask(g, [5])
        excluded = eval_cd_f_right_icp(p, selected, {})
        for key in original:
            np.testing.assert_array_equal(original[key][[0, 2]], excluded[key][[0, 2]])
            self.assertTrue(np.isnan(excluded[key][1]))

    def test_explicit_exclusion_rejects_unknown_duplicate_or_empty_coverage(self):
        _, g = sequence()
        for ids in ([1], [5, 5], [0, 5], [5.5]):
            with self.assertRaises(ValueError): hand_sim3_frame_mask(g, ids)
        g['is_valid'][0] = 0
        with self.assertRaisesRegex(ValueError, 'valid GT'): hand_sim3_frame_mask(g, [0])

    def test_explicit_exclusion_does_not_hide_other_degenerate_or_nonfinite_frames(self):
        p, g = sequence(3)
        p['j3d_c.right'][1] = 0; p['j3d_c.right'][2] = 0
        selected, _ = hand_sim3_frame_mask(g, [5])
        with self.assertRaisesRegex(ValueError, 'Frame 10:'):
            eval_cd_f_right_icp(p, selected, {})
        p['v3d_c.object'][1, 0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            protocol_entry.metrics(p, g, hand_sim3_excluded_frame_ids=[5])




if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
