"""Joint-based and relpose losses, plus scale regularisation."""
from __future__ import annotations

import torch

from optimization.losses import register_loss, LossContext
from optimization.utils.geometry import project_points


# ---------------------------------------------------------------------------
# Hand 2D joints loss
# ---------------------------------------------------------------------------

def hand_2d_joints(cfg, ctx: LossContext) -> torch.Tensor:
    """Mirrors ``Runner.compute_hand_joints_2d_loss``."""
    hand_joints_3d = getattr(ctx, 'hand_joints_3d', None)
    if hand_joints_3d is None:
        return torch.tensor(0.0, device=ctx.rendered_rgb.device)

    K = ctx.K
    joints_2d_gt = ctx.gt_hand_joints_2d

    joints_2d = project_points(hand_joints_3d, K)
    joints_dist = torch.norm(joints_2d - joints_2d_gt.to(joints_2d.device), dim=1)
    return torch.mean(joints_dist) * cfg.hand_joints_2d_lambda


hand_2d_joints.is_enabled = lambda cfg: cfg.hand_2d_joints_loss
hand_2d_joints = register_loss("hand_2d_joints")(hand_2d_joints)


# ---------------------------------------------------------------------------
# Hand-object relative pose loss
# ---------------------------------------------------------------------------

def ho_relpose(cfg, ctx: LossContext) -> torch.Tensor:
    """Mirrors ``Runner.compute_ho_relative_pose_loss``."""
    last_hand_verts_in_ob = getattr(ctx, 'last_hand_verts_in_ob', None)
    hand_verts_dist_weights = getattr(ctx, 'hand_verts_dist_weights', None)

    if last_hand_verts_in_ob is None or hand_verts_dist_weights is None:
        return torch.tensor(0.0, device=ctx.rendered_rgb.device)

    hand_verts = ctx.hand_vertices.detach()
    R = ctx.obj_full_R
    t = ctx.obj_full_t

    hand_vert_in_ob = (hand_verts - t.unsqueeze(0)) @ R
    verts_dist = torch.norm(hand_vert_in_ob - last_hand_verts_in_ob, dim=1)
    verts_dist = verts_dist * hand_verts_dist_weights

    return torch.mean(verts_dist) * cfg.ho_relpose_loss_lambda


ho_relpose.is_enabled = lambda cfg: cfg.ho_relpose_loss
ho_relpose = register_loss("ho_relpose")(ho_relpose)


# ---------------------------------------------------------------------------
# Object scale regularisation
# ---------------------------------------------------------------------------

def object_scale_reg(cfg, ctx: LossContext) -> torch.Tensor:
    """Mirrors ``Runner.compute_object_scale_regularization``."""
    if ctx.optimized_obj_scale is None:
        return torch.tensor(0.0, device=ctx.rendered_rgb.device)

    target_scale = cfg.initial_object_scale

    if cfg.object_scale_dim == 1:
        scale_diff = ctx.optimized_obj_scale - target_scale
        reg_loss = (scale_diff ** 2).mean()
    elif cfg.object_scale_dim == 3:
        target = torch.ones(3, device=ctx.optimized_obj_scale.device) * target_scale
        scale_diff = ctx.optimized_obj_scale - target
        reg_loss = (scale_diff ** 2).mean()
    else:
        reg_loss = torch.tensor(0.0, device=ctx.rendered_rgb.device)

    return reg_loss * cfg.object_scale_reg_lambda


object_scale_reg.is_enabled = lambda cfg: cfg.object_scale_reg_loss
object_scale_reg = register_loss("object_scale_reg")(object_scale_reg)
