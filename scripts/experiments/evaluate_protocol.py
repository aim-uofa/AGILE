"""Strict full-sequence evaluation with separately named interaction protocols."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT))
from optimization.evaluation.ho3d_io import load_results_data, load_gt_data
from optimization.evaluation.eval_utils import (
    eval_cd_icp_first_frame, eval_mpjpe_right, eval_mpjpe_right_scaled,
    eval_cd_f_right_icp, eval_cd_f_right,
)
from optimization.evaluation.protocols import (METRICS_VERSION, METRIC_PROTOCOLS, validate_inputs,
                                 reduce_metrics, hand_sim3_frame_mask)


def subset(data, indices, length):
    result = {}
    for key, value in data.items():
        # Constant mesh topology / intrinsics must not be mistaken for frames.
        per_frame = key not in ('faces.object', 'faces.right', 'K', 'faces')
        if per_frame and isinstance(value, (torch.Tensor, np.ndarray)) and value.ndim and value.shape[0] == length:
            result[key] = value[indices]
        elif per_frame and isinstance(value, list) and len(value) == length:
            result[key] = [value[int(i)] for i in indices]
        else:
            result[key] = value
    return result


def metrics(prediction, truth, registration_diagnostics=None, metric_errors=None,
            hand_sim3_excluded_frame_ids=()):
    validate_inputs(prediction, truth, require_frame_ids=True)
    sim3_truth, _ = hand_sim3_frame_mask(truth, hand_sim3_excluded_frame_ids)
    groups = [(eval_cd_icp_first_frame, ['cd_icp', 'f5_icp', 'f10_icp']),
              (eval_mpjpe_right, ['mpjpe_ra_r']), (eval_mpjpe_right_scaled, ['mpjpe_right_scaled']),
              (eval_cd_f_right_icp, ['cd_right_icp', 'f5_right_icp', 'f10_right_icp',
                  'cd_interaction_hand_sim3_per_frame', 'f5_interaction_hand_sim3_per_frame',
                  'f10_interaction_hand_sim3_per_frame']),
              (eval_cd_f_right, ['cd_right', 'f5_right', 'f10_right'])]
    result = {}
    for function, names in groups:
        try:
            kwargs = {'registration_diagnostics': registration_diagnostics} if function == eval_cd_icp_first_frame else {}
            group_truth = sim3_truth if function == eval_cd_f_right_icp else truth
            values = function(prediction, group_truth, {}, **kwargs)
            result.update(reduce_metrics(values, group_truth))
        except (ValueError, RuntimeError) as exc:
            if metric_errors is None:
                raise
            metric_errors[function.__name__] = {'error': type(exc).__name__ + ': ' + str(exc),
                                               'unavailable_metrics': names}
            result.update({name: None for name in names})
    return result


def code_revision():
    revision_file = PROJECT / 'REVISION'
    return revision_file.read_text().strip() if revision_file.exists() else subprocess.check_output(
        ['git', '-c', 'safe.directory=' + str(PROJECT), '-C', str(PROJECT), 'rev-parse', 'HEAD'],
        text=True).strip()


def evaluate_sequence(dataset, scene_id, result_dir, manifest, include_paper_window=True,
                      hand_sim3_excluded_frame_ids=(), hand_sim3_exclusion_reason=None):
    if hand_sim3_excluded_frame_ids and not hand_sim3_exclusion_reason:
        raise ValueError('Explicit hand-Sim(3) exclusions require a recorded reason')
    scene = next(s for s in manifest['datasets'][dataset]['scenes'] if s['id'] == scene_id)
    config = json.loads((result_dir / 'config.json').read_text())
    if config.get('test_first_frame'):
        raise ValueError('This evaluator requires full training outputs, not a first-frame smoke test')
    prediction, _ = load_results_data(str(result_dir / 'aligned'), preprocess_hold_mode=False)
    stride = config['hold_mode_interval'] if config['hold_mode'] else 1
    if stride != 5:
        raise ValueError('This experiment manifest requires the declared stride of 5')
    if dataset == 'ho3d':
        start = 0 if config.get('bidirectional_optimization') else config['start_frame']
        frame_ids = np.arange(start, scene['frame_count'], stride)
        truth = load_gt_data(scene_id, start, scene['frame_count'], hold_mode=True, device='cpu')
    else:
        raise ValueError('This release supports HO3D only')
    for side, data in [('prediction', prediction), ('GT', truth)]:
        actual = data['frame_ids'].cpu().numpy()
        if not np.array_equal(actual, frame_ids):
            raise ValueError(side + ' frame IDs do not match the declared declared full-sequence interval')
    coverage = validate_inputs(prediction, truth, require_frame_ids=True)
    _, sim3_coverage = hand_sim3_frame_mask(truth, hand_sim3_excluded_frame_ids)
    sim3_coverage['exclusion_reason'] = hand_sim3_exclusion_reason
    diagnostics = {}
    errors = {}
    result = {'paper_url': manifest.get('paper_url'), 'scene': scene_id,
              'evaluation_code_commit': code_revision(), 'metrics_version': METRICS_VERSION,
              'metric_protocols': METRIC_PROTOCOLS, 'paper_scene_match': scene.get('paper_scene_match', True),
              'frame_count': len(frame_ids), 'frame_ids': frame_ids.tolist(),
              'prediction_frame_ids_source': prediction['frame_ids_source'],
              'coverage': coverage, 'metric_coverage': {'hand_sim3': sim3_coverage},
              'full': metrics(prediction, truth, diagnostics, errors, hand_sim3_excluded_frame_ids),
              'object_registration': diagnostics, 'metric_errors': errors, 'success': not errors}
    if dataset == 'ho3d' and include_paper_window:
        indices = np.flatnonzero((frame_ids >= scene['start_frame']) & (frame_ids < 1024))
        if len(indices):
            window_diagnostics = {}
            window_errors = {}
            result['paper_window'] = metrics(subset(prediction, indices, len(frame_ids)),
                                             subset(truth, indices, len(frame_ids)), window_diagnostics, window_errors,
                                             [i for i in hand_sim3_excluded_frame_ids if i in frame_ids[indices]])
            _, window_coverage = hand_sim3_frame_mask(subset(truth, indices, len(frame_ids)),
                [i for i in hand_sim3_excluded_frame_ids if i in frame_ids[indices]])
            window_coverage['exclusion_reason'] = hand_sim3_exclusion_reason
            result['paper_window_metric_coverage'] = {'hand_sim3': window_coverage}
            result['paper_window_frame_ids'] = frame_ids[indices].tolist()
            result['paper_window_registration'] = window_diagnostics
            result['paper_window_metric_errors'] = window_errors
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['ho3d'])
    parser.add_argument('--scene', required=True)
    parser.add_argument('--result-dir', type=Path, required=True)
    parser.add_argument('--manifest-path', type=Path, required=True)
    parser.add_argument('--output-json', type=Path)
    parser.add_argument('--no-paper-window', action='store_true')
    parser.add_argument('--exclude-hand-sim3-frame-ids', nargs='+', type=int, default=[])
    parser.add_argument('--hand-sim3-exclusion-reason')
    args = parser.parse_args()
    manifest = json.loads(args.manifest_path.read_text())
    output = args.output_json or args.result_dir / 'metrics_protocol.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = evaluate_sequence(args.dataset, args.scene, args.result_dir, manifest,
                                   include_paper_window=not args.no_paper_window,
                                   hand_sim3_excluded_frame_ids=args.exclude_hand_sim3_frame_ids,
                                   hand_sim3_exclusion_reason=args.hand_sim3_exclusion_reason)
    except Exception as exc:
        # Failed evaluations retain coverage/protocol context and a clear error,
        # without replacing a previously successful metric file.
        failure = {'scene': args.scene, 'success': False, 'metrics_version': METRICS_VERSION,
                   'evaluation_code_commit': code_revision(), 'error': type(exc).__name__ + ': ' + str(exc)}
        output.with_name(output.stem + '-failure.json').write_text(json.dumps(failure, indent=2) + chr(10))
        raise
    output.parent.mkdir(parents=True, exist_ok=True)
    if not result['success']:
        partial = output.with_name(output.stem + '-partial.json')
        partial.write_text(json.dumps(result, indent=2, allow_nan=False) + chr(10))
        failure = {'scene': args.scene, 'success': False, 'metrics_version': METRICS_VERSION,
                   'evaluation_code_commit': code_revision(), 'metric_errors': result['metric_errors'],
                   'partial_metrics': str(partial)}
        output.with_name(output.stem + '-failure.json').write_text(json.dumps(failure, indent=2) + chr(10))
        print(json.dumps(result, indent=2, allow_nan=False))
        raise SystemExit(1)
    if output.exists():
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        output.rename(output.with_name(output.stem + '-before-' + stamp + output.suffix))
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + chr(10))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
