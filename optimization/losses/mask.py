"""Mask losses (object and hand silhouette supervision)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from optimization.losses import register_loss, LossContext


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _mask_loss(
    rendered_alpha,   # [H, W]
    alpha_gt,         # [H, W]
    loss_mask_union,  # [H, W] float
    loss_mask_inter,  # [H, W] float
    occ_mask,         # [H, W] float
    mask_loss_type: str,
    weight: float,
) -> torch.Tensor:
    """Compute a single mask loss term."""
    if mask_loss_type == 'l1':
        alpha_diff = torch.abs(rendered_alpha - alpha_gt.float())
        alpha_diff = alpha_diff * loss_mask_union * (1 - occ_mask)
        alpha_loss = torch.sum(alpha_diff) / (torch.sum(loss_mask_union) + 1e-8)
    elif mask_loss_type == 'l2':
        alpha_diff = (rendered_alpha - alpha_gt.float()) ** 2
        alpha_diff = alpha_diff * loss_mask_union * (1 - occ_mask)
        alpha_loss = torch.sum(alpha_diff) / (torch.sum(loss_mask_union) + 1e-8)
    elif mask_loss_type == 'iou':
        # Boolean thresholding alpha severs the renderer's gradient entirely.
        visible = 1 - occ_mask
        inter_sum = torch.sum(rendered_alpha * alpha_gt.float() * visible)
        union_sum = torch.sum((rendered_alpha + alpha_gt.float()
                               - rendered_alpha * alpha_gt.float()) * visible)
        iou = inter_sum / (union_sum + 1e-8)
        alpha_loss = 1.0 - iou
    elif mask_loss_type == 'cross_entropy':
        bce_loss = F.binary_cross_entropy(rendered_alpha, alpha_gt.float(), reduction='none')
        bce_loss = bce_loss * loss_mask_union * (1 - occ_mask)
        alpha_loss = torch.sum(bce_loss) / (torch.sum(loss_mask_union) + 1e-8)
    else:
        raise NotImplementedError(f"Mask loss type {mask_loss_type} not implemented.")

    return alpha_loss * weight


# ---------------------------------------------------------------------------
# Object mask loss
# ---------------------------------------------------------------------------

def object_mask_loss(cfg, ctx: LossContext) -> torch.Tensor:
    alpha = ctx.rendered_object_alpha[0, ..., 0]      # [H, W]
    alpha_gt = ctx.gt_object_mask                      # [H, W]
    occ_mask = ctx.gt_hand_mask.float()                # [H, W]

    loss_mask_union = (alpha_gt.bool() | (alpha > 1e-3))
    loss_mask_inter = (alpha_gt.bool() & (alpha > 1e-3))
    loss_mask_union = ((occ_mask <= 0) & loss_mask_union).float()
    loss_mask_inter = ((occ_mask <= 0) & loss_mask_inter).float()

    return _mask_loss(
        alpha, alpha_gt,
        loss_mask_union, loss_mask_inter, occ_mask,
        cfg.mask_loss_type, cfg.object_mask_lambda,
    )


object_mask_loss.is_enabled = lambda cfg: cfg.object_mask_loss
object_mask_loss = register_loss("object_mask")(object_mask_loss)


# ---------------------------------------------------------------------------
# Hand mask loss
# ---------------------------------------------------------------------------

def hand_mask_loss(cfg, ctx: LossContext) -> torch.Tensor:
    alpha = ctx.rendered_hand_alpha[0, ..., 0]         # [H, W]
    alpha_gt = getattr(ctx, 'gt_hand_mask_alpha', None)
    if alpha_gt is None:
        alpha_gt = ctx.gt_hand_mask                    # [H, W]
    occ_mask = ctx.gt_object_mask.float()              # [H, W]

    loss_mask_union = ((alpha > 1e-3) | alpha_gt.bool()).float()
    loss_mask_inter = ((alpha > 1e-3) & alpha_gt.bool()).float()

    return _mask_loss(
        alpha, alpha_gt,
        loss_mask_union, loss_mask_inter, occ_mask,
        cfg.mask_loss_type, cfg.hand_mask_lambda,
    )


hand_mask_loss.is_enabled = lambda cfg: cfg.hand_mask_loss
hand_mask_loss = register_loss("hand_mask")(hand_mask_loss)
