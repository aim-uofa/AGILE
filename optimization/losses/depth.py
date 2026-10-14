"""Depth, normal, edge, and depth-SDF losses."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from optimization.losses import register_loss, LossContext
from optimization.utils.image import compute_edge_sobel
from optimization.utils.depth import depths_to_points
from optimization.utils.sdf import query_sdf_differentiable


# ---------------------------------------------------------------------------
# Depth loss (shared core)
# ---------------------------------------------------------------------------

def _depth_loss_fn(cfg, ctx: LossContext, is_hand: bool = False) -> torch.Tensor:
    rendered_depth = ctx.rendered_depth
    depth_gt = ctx.gt_depth
    occ_mask = (ctx.gt_hand_mask if not is_hand else ctx.gt_object_mask).float()

    supervision_mask = ctx.gt_object_mask if not is_hand else ctx.gt_hand_mask
    alpha = ctx.rendered_object_alpha if not is_hand else ctx.rendered_hand_alpha

    loss_mask_union = (supervision_mask.bool() | (alpha[0, ..., 0] > 1e-3))
    loss_mask_inter = (supervision_mask.bool() & (alpha[0, ..., 0] > 1e-3))
    loss_mask_union = ((occ_mask <= 0) & loss_mask_union).float()
    loss_mask_inter = ((occ_mask <= 0) & loss_mask_inter).float()

    if cfg.depth_loss_type == 'l1':
        depth_diff = torch.abs(rendered_depth - depth_gt)
    elif cfg.depth_loss_type == 'l2':
        depth_diff = (rendered_depth - depth_gt) ** 2
    elif cfg.depth_loss_type == 'huber':
        depth_diff = F.smooth_l1_loss(rendered_depth, depth_gt,
                                      beta=cfg.depth_huber_beta, reduction='none')
    else:
        raise NotImplementedError(f"Depth loss type {cfg.depth_loss_type} not implemented.")

    valid_depth_mask = (depth_gt > 0).squeeze(0).squeeze(-1).float()

    if cfg.depth_mask_type == 'union':
        depth_loss_mask = loss_mask_union * valid_depth_mask
    elif cfg.depth_mask_type == 'intersection':
        depth_loss_mask = loss_mask_inter * valid_depth_mask
    else:
        raise NotImplementedError(f"Depth mask type {cfg.depth_mask_type} not implemented.")

    depth_diff = depth_diff * depth_loss_mask[None, ..., None]
    loss = torch.sum(depth_diff) / (torch.sum(depth_loss_mask) + 1e-8)
    return loss * cfg.depth_lambda


# ---------------------------------------------------------------------------
# Object depth
# ---------------------------------------------------------------------------

def object_depth_loss(cfg, ctx: LossContext) -> torch.Tensor:
    return _depth_loss_fn(cfg, ctx, is_hand=False)

object_depth_loss.is_enabled = lambda cfg: cfg.object_depth_loss
object_depth_loss = register_loss("object_depth")(object_depth_loss)


# ---------------------------------------------------------------------------
# Hand depth
# ---------------------------------------------------------------------------

def hand_depth_loss(cfg, ctx: LossContext) -> torch.Tensor:
    return _depth_loss_fn(cfg, ctx, is_hand=True)

hand_depth_loss.is_enabled = lambda cfg: cfg.hand_depth_loss
hand_depth_loss = register_loss("hand_depth")(hand_depth_loss)


# ---------------------------------------------------------------------------
# Normal loss (object only in the original code)
# ---------------------------------------------------------------------------

def normal_loss(cfg, ctx: LossContext) -> torch.Tensor:
    rendered_normal = ctx.rendered_normals
    if rendered_normal is None:
        return torch.tensor(0.0, device=ctx.rendered_rgb.device)

    normal_gt = getattr(ctx, 'gt_normals', None)
    if normal_gt is None:
        return torch.tensor(0.0, device=ctx.rendered_rgb.device)

    occ_mask = ctx.gt_hand_mask.float()
    alpha = ctx.rendered_object_alpha
    loss_mask_union = (ctx.gt_object_mask.bool() | (alpha[0, ..., 0] > 1e-3))
    loss_mask_inter = (ctx.gt_object_mask.bool() & (alpha[0, ..., 0] > 1e-3))
    loss_mask_union = ((occ_mask <= 0) & loss_mask_union).float()
    loss_mask_inter = ((occ_mask <= 0) & loss_mask_inter).float()

    if cfg.normal_loss_type == 'l1':
        normal_diff = torch.abs(rendered_normal - normal_gt).sum(dim=-1, keepdim=True)
    elif cfg.normal_loss_type == 'l2':
        normal_diff = ((rendered_normal - normal_gt) ** 2).sum(dim=-1, keepdim=True)
    elif cfg.normal_loss_type == 'cosine':
        normal_diff = 1 - (rendered_normal * normal_gt).sum(dim=-1, keepdim=True)
    else:
        raise NotImplementedError(f"Normal loss type {cfg.normal_loss_type} not implemented.")

    valid_normal_mask = (normal_gt.abs().sum(dim=-1, keepdim=True) > 0).squeeze(0).squeeze(-1).float()

    if cfg.normal_mask_type == 'union':
        normal_loss_mask = loss_mask_union * valid_normal_mask
    elif cfg.normal_mask_type == 'intersection':
        normal_loss_mask = loss_mask_inter * valid_normal_mask
    else:
        raise NotImplementedError(f"Normal mask type {cfg.normal_mask_type} not implemented.")

    normal_diff = normal_diff * normal_loss_mask[None, ..., None]
    loss = torch.sum(normal_diff) / (torch.sum(normal_loss_mask) + 1e-8)
    return loss * cfg.normal_lambda

normal_loss.is_enabled = lambda cfg: cfg.normal_loss
normal_loss = register_loss("normal")(normal_loss)


# ---------------------------------------------------------------------------
# Object edge loss
# ---------------------------------------------------------------------------

def object_edge_loss(cfg, ctx: LossContext) -> torch.Tensor:
    rendered_depth = ctx.rendered_depth
    rendered_edge = compute_edge_sobel(rendered_depth[0, ..., 0])

    gt_object_mask = ctx.gt_object_mask.float()
    edge_gt = compute_edge_sobel(gt_object_mask).to(rendered_edge.device)

    occ_mask = ctx.gt_hand_mask.float()
    alpha = ctx.rendered_object_alpha
    loss_mask_union = (ctx.gt_object_mask.bool() | (alpha[0, ..., 0] > 1e-3))
    loss_mask_union = ((occ_mask <= 0) & loss_mask_union).float()

    edge_diff = torch.abs(rendered_edge - edge_gt)
    loss = torch.sum(edge_diff) / (torch.sum(loss_mask_union) + 1e-8)
    return loss * cfg.edge_lambda

object_edge_loss.is_enabled = lambda cfg: cfg.object_edge_loss
object_edge_loss = register_loss("object_edge")(object_edge_loss)


# ---------------------------------------------------------------------------
# Depth-SDF loss
# ---------------------------------------------------------------------------

def object_depth_sdf_loss(cfg, ctx: LossContext) -> torch.Tensor:
    depth = ctx.rendered_depth[0, ..., 0]
    K = ctx.K
    point_map = depths_to_points(K, depth)
    points = point_map.reshape(-1, 3)
    points_valid = points[points[:, 2] > 1e-6]

    if ctx.object_sdf_grid is None:
        return torch.tensor(0.0, device=depth.device)

    sdf_grid = ctx.object_sdf_grid['sdf']
    norm_trans = ctx.object_sdf_grid['norm_trans']
    scale = ctx.object_sdf_grid['scale']

    sdf_values, _ = query_sdf_differentiable(
        sdf_grid, points_valid, norm_trans, scale, get_vis_pcd=False
    )
    valid_mask = sdf_values.abs() < 1.0
    sdf_values = sdf_values[valid_mask]

    if cfg.depth_sdf_loss_type == 'l1':
        loss = torch.mean(torch.abs(sdf_values))
    elif cfg.depth_sdf_loss_type == 'l2':
        loss = torch.mean(torch.square(sdf_values))
    else:
        loss = torch.mean(torch.abs(sdf_values))

    return loss * cfg.depth_sdf_lambda

object_depth_sdf_loss.is_enabled = lambda cfg: cfg.object_depth_sdf_loss
object_depth_sdf_loss = register_loss("object_depth_sdf")(object_depth_sdf_loss)
