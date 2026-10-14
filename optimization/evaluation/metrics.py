
import torch


def compute_joint3d_error(joints3d_cam_gt, joints3d_cam_pred, valid_jts):
    valid_jts = valid_jts.view(-1)
    assert joints3d_cam_gt.shape == joints3d_cam_pred.shape
    assert joints3d_cam_gt.shape[0] == valid_jts.shape[0]
    dist = ((joints3d_cam_gt - joints3d_cam_pred) ** 2).sum(dim=2).sqrt()
    invalid_idx = torch.nonzero((1 - valid_jts).long()).view(-1)
    dist[invalid_idx, :] = float("nan")
    return dist
