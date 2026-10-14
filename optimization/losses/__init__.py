from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any
import torch

_LOSS_REGISTRY: dict[str, Callable] = {}


def register_loss(name: str):
    """Decorator that registers a loss function in the global registry.

    The decorated function **must** already have an ``is_enabled`` attribute
    (a callable ``cfg -> bool``) before ``@register_loss`` is applied.
    """
    def decorator(fn):
        if not hasattr(fn, 'is_enabled'):
            raise AttributeError(
                f"Loss '{name}' must set is_enabled before @register_loss"
            )
        _LOSS_REGISTRY[name] = fn
        return fn
    return decorator


def compute_losses(cfg, ctx: "LossContext") -> dict[str, torch.Tensor]:
    """Call all enabled losses against shared LossContext.  Returns {name: tensor}."""
    losses = {}
    for name, fn in _LOSS_REGISTRY.items():
        if fn.is_enabled(cfg):
            losses[name] = fn(cfg, ctx)
    return losses


@dataclass
class LossContext:
    """Shared context passed to every loss function."""

    # Rendered outputs (one forward pass, shared by all losses)
    rendered_rgb: torch.Tensor
    rendered_object_alpha: torch.Tensor
    rendered_hand_alpha: torch.Tensor
    rendered_zbuf: torch.Tensor
    rendered_depth: torch.Tensor
    rendered_normals: Any  # may be None

    # GT data
    gt_rgb: torch.Tensor
    gt_object_mask: torch.Tensor
    gt_hand_mask: torch.Tensor
    gt_depth: torch.Tensor
    gt_hand_joints_2d: torch.Tensor

    # Frame state
    frame_idx: int
    iteration: int
    object_vertices: torch.Tensor
    hand_vertices: torch.Tensor
    camera: Any

    # Optional fields
    pix_to_face: Any = None
    vertices_visibility: Any = None
    face_visibility: Any = None
    prev_obj_R_seq: Any = None
    prev_obj_t_seq: Any = None
    prev_hand_verts_seq: Any = None
    dino_features_gt: Any = None
    gt_normals: Any = None
    gt_hand_mask_alpha: Any = None
    hand_sdf_grid: Any = None
    object_sdf_grid: Any = None
    hand_sdf_verts: Any = None

    # Extra context for SDF / contact losses
    obj_full_R: Any = None
    obj_full_t: Any = None
    hand_full_R: Any = None
    hand_full_t: Any = None
    hand_zbuf: Any = None
    K: Any = None  # camera intrinsics [3,3]
    in_contact: bool = False
    finger_mask: Any = None

    # For feature loss
    rendered_object_alpha_hard: Any = None
    feature_pts_canonical: Any = None
    feature_pts_similarity_maps: Any = None
    feature_pts_to_face: Any = None

    # For scale regularisation
    optimized_obj_scale: Any = None
    hand_joints_3d: Any = None
    last_hand_verts_in_ob: Any = None
    hand_verts_dist_weights: Any = None


# Trigger registration of all concrete loss modules
from optimization.losses import mask, depth, feature, contact, smoothness, joints  # noqa: F401
