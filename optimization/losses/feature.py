"""DINO feature-based loss for object pose optimisation."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from optimization.losses import register_loss, LossContext
from optimization.utils.feature import feature_loss, feature_loss_global
from optimization.utils.geometry import project_points


# ---------------------------------------------------------------------------
# Object feature loss
# ---------------------------------------------------------------------------

def object_feature_loss(cfg, ctx: LossContext) -> torch.Tensor:
    """Compute feature matching loss between rendered and GT DINO features.

    Mirrors ``Runner.compute_object_feature_loss`` from mesh_trainer.py.
    The caller must populate ctx with the sampled feature data.
    """
    if ctx.dino_features_gt is None or ctx.face_visibility is None:
        return torch.tensor(0.0, device=ctx.rendered_rgb.device)

    feature_pts_canonical = getattr(ctx, 'feature_pts_canonical', None)
    feature_pts_similarity_maps = getattr(ctx, 'feature_pts_similarity_maps', None)
    feature_pts_to_face = getattr(ctx, 'feature_pts_to_face', None)

    if feature_pts_canonical is None or feature_pts_similarity_maps is None:
        return torch.tensor(0.0, device=ctx.rendered_rgb.device)

    face_visibility_mask = ctx.face_visibility
    pts_visibility_mask = face_visibility_mask[feature_pts_to_face]
    visible_pts = feature_pts_canonical[pts_visibility_mask]
    sim_maps_visible = feature_pts_similarity_maps[pts_visibility_mask]

    scale = ctx.optimized_obj_scale if ctx.optimized_obj_scale is not None else 1.0
    R = ctx.obj_full_R
    t = ctx.obj_full_t
    K = ctx.K

    pts_scaled = visible_pts * scale
    pts_rotated = pts_scaled @ R.T
    pts_transformed = pts_rotated + t.unsqueeze(0)

    v2d = project_points(pts_transformed, K)
    sample_coords = v2d

    occ_mask = ctx.gt_hand_mask.float()

    if cfg.object_feature_loss_type == 'local':
        feat_loss, info = feature_loss(
            similarity_maps=sim_maps_visible,
            xys_image=sample_coords,
            occ_mask=occ_mask,
        )
    elif cfg.object_feature_loss_type == 'global':
        feat_loss, info = feature_loss_global(
            similarity_maps=sim_maps_visible,
            xys_image=sample_coords,
            occ_mask=occ_mask,
        )
    else:
        raise ValueError(f"Unknown object feature loss type: {cfg.object_feature_loss_type}")

    return feat_loss * cfg.object_feature_loss_lambda


object_feature_loss.is_enabled = lambda cfg: cfg.object_feature_loss
object_feature_loss = register_loss("object_feature")(object_feature_loss)
