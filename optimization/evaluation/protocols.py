"""Strict frame, validity and reduction rules shared by evaluation entry points."""
import numpy as np
import torch

METRICS_VERSION = 'evaluation-v2-gt-cv-robust-icp'
METRIC_PROTOCOLS = {
    'cd_icp': {'unit': 'cm2', 'alignment': 'object Sim(3), deterministic radius/PCA + scale-normalized FPFH, 3cm then 1cm ICP',
               'seed': 1, 'registration_surface_samples': 1000, 'score_points': 'all mesh vertices',
               'anchor': 'first valid HO3D frame'},
    'cd_right': {'unit': 'cm2', 'alignment': 'HOLD: each object minus its own hand wrist; no fitted rotation or scale',
                 'seed': 1, 'max_vertices_per_frame': 3000},
    'cd_interaction_hand_sim3_per_frame': {'unit': 'cm2', 'alignment': 'per-frame positive Sim(3) fitted to corresponding 21 hand joints and applied to object',
                                          'seed': 1, 'max_vertices_per_frame': 3000, 'legacy_alias': 'cd_right_icp'},
    'mpjpe_ra_r': {'unit': 'mm', 'alignment': 'wrist translation only; canonical 21 joints including wrist'},
    'mpjpe_right_scaled': {'unit': 'mm', 'alignment': 'wrist translation plus one positive least-squares scale for the whole evaluated sequence; auxiliary'},
    'reduction': 'valid GT frames within each sequence; equal-weight sequence means across dataset',
    'cd_formula': 'sum of the two nearest-neighbor squared-distance means; no division by 2',
    'gt_camera_conversion': 'HO3D MANO rest-wrist pivot included; OpenGL to OpenCV',
}


def as_numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def valid_mask(truth):
    mask = as_numpy(truth['is_valid'])
    if mask.ndim != 1 or not np.isin(mask, [0, 1]).all() or not mask.any():
        raise ValueError('is_valid must be a nonempty binary frame mask with at least one valid GT frame')
    return mask.astype(bool)


def check_frame_ids(prediction, truth):
    for side, data in [('prediction', prediction), ('GT', truth)]:
        ids = as_numpy(data['frame_ids'])
        if ids.ndim != 1 or not np.isfinite(ids).all() or not np.equal(ids, np.floor(ids)).all():
            raise ValueError(side + ' frame_ids must be finite integer IDs')
        if len(ids) != len(data['j3d_c.right']) or (np.diff(ids) <= 0).any():
            raise ValueError(side + ' frame_ids must match data length and be strictly increasing')
    if not np.array_equal(as_numpy(prediction['frame_ids']), as_numpy(truth['frame_ids'])):
        raise ValueError('Prediction/GT frame_ids mismatch; automatic truncation is not permitted')


def validate_inputs(prediction, truth, require_frame_ids=False,
                    required_keys=('j3d_c.right', 'v3d_c.object')):
    mask = valid_mask(truth)
    n = len(mask)
    finite_prediction = np.ones(n, dtype=bool)
    for side, data in [('prediction', prediction), ('GT', truth)]:
        for key in required_keys:
            if key not in data:
                raise ValueError(side + ' is missing ' + key)
        for key, value in data.items():
            if not (key.startswith(('v3d_', 'j3d_', 'ori_v3d_', 'ori_j3d_')) or key == 'root.object'):
                continue
            arr = as_numpy(value)
            if not arr.ndim or len(arr) != n or arr.shape[-1] != 3:
                raise ValueError('%s %s frame/shape mismatch: %s; expected %s frames' % (side, key, arr.shape, n))
            if key.startswith(('j3d_', 'ori_j3d_')) and arr.shape != (n, 21, 3):
                raise ValueError(side + ' ' + key + ' must have canonical shape (N,21,3)')
            finite = np.isfinite(arr).reshape(n, -1).all(axis=1)
            if side == 'prediction':
                finite_prediction &= finite
            if not finite[mask].all():
                bad = np.flatnonzero(mask & ~finite).tolist()
                raise ValueError('%s %s non-finite on valid GT frames %s; evaluation failed' % (side, key, bad))
    if require_frame_ids or 'frame_ids' in prediction or 'frame_ids' in truth:
        if 'frame_ids' not in prediction or 'frame_ids' not in truth:
            raise ValueError('Both prediction and GT must identify their frame_ids')
        check_frame_ids(prediction, truth)
    return {'frame_count': n, 'valid_GT_frames': int(mask.sum()),
            'invalid_GT_frames': int((~mask).sum()),
            'finite_prediction_frames': int(finite_prediction.sum()),
            'valid_GT_prediction_coverage': float(finite_prediction[mask].mean())}


def strict_mean(values, mask, name):
    arr = as_numpy(values)
    if arr.ndim == 0:
        if not np.isfinite(arr):
            raise ValueError(name + ' is not finite')
        return float(arr)
    if arr.shape != mask.shape:
        raise ValueError(name + ' metric frame count does not match GT mask')
    if not np.isfinite(arr[mask]).all():
        raise ValueError(name + ' is non-finite on a valid GT frame; cannot silently omit it')
    return float(arr[mask].mean())


def reduce_metrics(values, truth):
    mask = valid_mask(truth)
    return {name: strict_mean(value, mask, name) for name, value in values.items()}


def hand_sim3_frame_mask(truth, excluded_frame_ids=()):
    """Apply an explicit, recorded exception only to the hand-Sim(3) metric."""
    ids = as_numpy(truth['frame_ids'])
    original = valid_mask(truth)
    requested = np.asarray(excluded_frame_ids)
    if requested.ndim != 1 or not np.isfinite(requested).all() or not np.equal(requested, np.floor(requested)).all():
        raise ValueError('Hand-Sim(3) exclusions must be integer frame IDs')
    if len(np.unique(requested)) != len(requested):
        raise ValueError('Duplicate hand-Sim(3) exclusion frame IDs')
    if not np.isin(requested, ids[original]).all():
        raise ValueError('Each hand-Sim(3) exclusion must name a present, valid GT frame')
    mask = original & ~np.isin(ids, requested)
    if not mask.any():
        raise ValueError('Hand-Sim(3) exclusions leave no evaluable frames')
    selected_truth = dict(truth)
    if isinstance(truth['is_valid'], torch.Tensor):
        selected_truth['is_valid'] = torch.as_tensor(mask, dtype=truth['is_valid'].dtype,
                                                    device=truth['is_valid'].device)
    else:
        selected_truth['is_valid'] = mask.astype(as_numpy(truth['is_valid']).dtype)
    coverage = {'valid_GT_frames': int(original.sum()),
                'excluded_frame_ids': sorted(int(i) for i in requested),
                'evaluated_frame_ids': ids[mask].astype(int).tolist(),
                'evaluated_frames': int(mask.sum()),
                'valid_GT_frame_coverage': float(mask.sum() / original.sum())}
    return selected_truth, coverage
