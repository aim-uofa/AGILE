"""Distinct wrist-only HOLD and per-frame hand Sim(3) interaction metrics."""
import numpy as np
import torch

from .protocols import as_numpy, validate_inputs, valid_mask


def hand_similarity(predicted_joints, gt_joints):
    pc, gc = predicted_joints.mean(0), gt_joints.mean(0)
    x, y = predicted_joints - pc, gt_joints - gc
    denom = np.sum(x * x)
    if denom <= 1e-12 or np.linalg.matrix_rank(x, tol=1e-8) < 2:
        raise ValueError('Degenerate predicted hand joints cannot define a hand Sim(3)')
    u, _, vt = np.linalg.svd(x.T @ y)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    scale = np.sum(y * (rotation @ x.T).T) / (denom + 1e-8)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Hand Sim(3) scale must be finite and positive')
    return rotation, scale, gc - scale * rotation @ pc


def eval_cd_f_right(data_pred, data_gt, metric_dict, visualize=False, metric_dir=None):
    """HOLD eval_cd_f_right sampling and squared CD; validate before reduction."""
    from .eval_utils import calculate_chamfer_f_scores
    validate_inputs(data_pred, data_gt)
    mask = valid_mask(data_gt)
    generator = torch.Generator(device='cpu').manual_seed(1)
    cd, f5, f10 = [], [], []
    for i, valid in enumerate(mask):
        pv = as_numpy(data_pred['v3d_c.object'][i])
        gv = as_numpy(data_gt['v3d_c.object'][i])
        pi = torch.randperm(len(pv), generator=generator)[:3000].numpy()
        gi = torch.randperm(len(gv), generator=generator)[:3000].numpy()
        if valid:
            p = pv[pi] - as_numpy(data_pred['j3d_c.right'][i, :1])
            g = gv[gi] - as_numpy(data_gt['j3d_c.right'][i, :1])
            values = calculate_chamfer_f_scores(p, g)
        else:
            values = (np.nan, np.nan, np.nan)
        cd.append(values[0]); f5.append(values[1]); f10.append(values[2])
    metric_dict.update(cd_right=np.asarray(cd), f5_right=np.asarray(f5) * 100,
                       f10_right=np.asarray(f10) * 100)
    return metric_dict
