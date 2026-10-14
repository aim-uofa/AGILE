"""Contact losses: z-buffer contact, SDF-based hand/object contact."""
from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np

from optimization.losses import register_loss, LossContext
from optimization.losses._utils import huber_loss

EPS = 1e-6


def zbuf_contact_loss(
    hand_zbuf: torch.Tensor,       # [H, W, K]
    obj_zbuf: torch.Tensor,        # [H, W, K]
    hand_occ_obj_mask: torch.Tensor,  # [H, W] bool
    obj_occ_hand_mask: torch.Tensor,  # [H, W] bool
):
    """Original multi-layer z-buffer contact loss."""
    H, W, K = hand_zbuf.shape
    device = hand_zbuf.device

    loss_map = torch.zeros((H, W), dtype=torch.float32, device=device)
    debug_color = torch.zeros((H, W, 3), dtype=torch.float32, device=device)

    if hand_occ_obj_mask.any():
        for i in range(1, K):
            hand_z = hand_zbuf[:, :, i]
            obj_z = obj_zbuf[:, :, i - 1]
            if (hand_z <= 0.0).all() or (obj_z <= 0.0).all():
                break
            valid_mask = hand_occ_obj_mask & (hand_z > 0.0) & (obj_z > 0.0)
            dist_map = torch.abs(obj_z - hand_z) * valid_mask.float()
            loss_map += dist_map
            debug_color[valid_mask] = torch.tensor([1.0, 0.0, 0.0], device=device)

    if obj_occ_hand_mask.any():
        for i in range(1, K):
            obj_z = obj_zbuf[:, :, i]
            hand_z = hand_zbuf[:, :, i - 1]
            if (obj_z <= 0.0).all() or (hand_z <= 0.0).all():
                break
            valid_mask = obj_occ_hand_mask & (obj_z > 0.0) & (hand_z > 0.0)
            dist_map = torch.abs(obj_z - hand_z) * valid_mask.float()
            loss_map += dist_map
            debug_color[valid_mask] = torch.tensor([0.0, 1.0, 0.0], device=device)

    loss_mask = hand_occ_obj_mask | obj_occ_hand_mask
    total_loss = loss_map.sum() / (loss_mask.sum() + 1e-8)
    info = {'loss_map': loss_map, 'debug_color': debug_color}
    return total_loss, info


def zbuf_contact_loss_simplified(
    hand_zbuf: torch.Tensor,       # [H, W, K]
    obj_zbuf: torch.Tensor,        # [H, W, K]
    hand_occ_obj_mask: torch.Tensor,  # [H, W] bool
    obj_occ_hand_mask: torch.Tensor,  # [H, W] bool
    cam_K: torch.Tensor = None,
):
    """Simplified z-buffer contact loss using front/back faces only."""
    H, W, K = hand_zbuf.shape
    device = hand_zbuf.device

    total_dist_map = torch.zeros((H, W), dtype=torch.float32, device=device)
    debug_color = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
    penetration_weights = torch.zeros((H, W), dtype=torch.float32, device=device)
    penetration_weight_value = 2.0
    valid_contact = torch.zeros((H, W), dtype=torch.bool, device=device)

    # Local vars needed for optional debug PCD
    hand_back_z = hand_front_z = obj_front_z = obj_back_z = None

    if hand_occ_obj_mask.any():
        hand_back_z = hand_zbuf[:, :, 1]
        obj_front_z = obj_zbuf[:, :, 0]
        valid_mask = hand_occ_obj_mask & (hand_back_z > EPS) & (obj_front_z > EPS)
        if valid_mask.any():
            dist_map = (obj_front_z - hand_back_z) * valid_mask.float()
            penetration_weights += ((dist_map < EPS) & valid_mask).float() * penetration_weight_value
            valid_contact |= valid_mask
            total_dist_map += dist_map
            debug_color[valid_mask] = torch.tensor([1.0, 0.0, 0.0], device=device)

    if obj_occ_hand_mask.any():
        hand_front_z = hand_zbuf[:, :, 0]
        obj_back_z = obj_zbuf.max(dim=2).values
        valid_mask = obj_occ_hand_mask & (obj_back_z > EPS) & (hand_front_z > EPS)
        if valid_mask.any():
            dist_map = (obj_back_z - hand_front_z) * valid_mask.float()
            penetration_weights += ((dist_map > EPS) & valid_mask).float() * penetration_weight_value
            valid_contact |= valid_mask
            total_dist_map += dist_map
            debug_color[valid_mask] = torch.tensor([0.0, 1.0, 0.0], device=device)

    pcd = None
    if cam_K is not None:
        try:
            from optimization.utils.depth import depth2pts_torch
            import open3d as o3d
            pts_list = []
            colors_list = []
            if hand_occ_obj_mask.any() and hand_back_z is not None and obj_front_z is not None:
                hand_back_z_pts = depth2pts_torch(hand_back_z, cam_K).detach().cpu().numpy()
                obj_front_z_pts = depth2pts_torch(obj_front_z, cam_K).detach().cpu().numpy()
                pts_list.extend([hand_back_z_pts, obj_front_z_pts])
                colors_list.extend([
                    np.array([[1.0, 0.0, 0.0]] * hand_back_z_pts.shape[0]),
                    np.array([[0.0, 1.0, 0.0]] * obj_front_z_pts.shape[0]),
                ])
            if obj_occ_hand_mask.any() and hand_front_z is not None and obj_back_z is not None:
                hand_front_z_pts = depth2pts_torch(hand_front_z, cam_K).detach().cpu().numpy()
                obj_back_z_pts = depth2pts_torch(obj_back_z, cam_K).detach().cpu().numpy()
                pts_list.extend([hand_front_z_pts, obj_back_z_pts])
                colors_list.extend([
                    np.array([[1.0, 1.0, 0.0]] * hand_front_z_pts.shape[0]),
                    np.array([[0.0, 1.0, 1.0]] * obj_back_z_pts.shape[0]),
                ])
            if len(pts_list) > 0:
                all_pts = np.concatenate(pts_list, axis=0)
                all_pts[:, 1:] *= -1
                all_colors = np.concatenate(colors_list, axis=0)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(all_pts)
                pcd.colors = o3d.utility.Vector3dVector(all_colors)
        except Exception:
            pass

    dist_map_weighted = total_dist_map * (1.0 + penetration_weights)
    total_loss = huber_loss(dist_map_weighted, delta=0.001).sum() / (valid_contact.sum() + 1e-8)

    info = {
        'dist_map': total_dist_map,
        'debug_color': debug_color,
        'debug_pcd': pcd,
    }
    return total_loss, info


def repel_and_contact_loss_p3d(
    P_h: torch.Tensor,
    P_o: torch.Tensor,
    contact_target: float = None,
    min_gap: float = None,
    band_width: float = None,
    q: float = 0.02,
    margin_scale: float = 0.10,
    min_gap_ratio: float = 0.2,
    contact_ratio: float = 0.8,
    band_ratio: float = 0.4,
    K: int = 8,
    softmin_temp: float = 0.02,
    repel_weight: float = 0.0,
    contact_weight: float = 1.0,
    use_huber: bool = True,
    huber_delta: float = 1.0,
    sanitize: bool = True,
    eps: float = 1e-12,
):
    """Repel-and-contact loss using PyTorch3D KNN.

    Moved verbatim from ``utils/general_utils.py``.
    """
    from pytorch3d.ops import knn_points

    assert P_h.ndim == 2 and P_h.shape[-1] == 3
    assert P_o.ndim == 2 and P_o.shape[-1] == 3
    if sanitize:
        P_h = torch.nan_to_num(P_h, nan=0.0, posinf=1e6, neginf=-1e6)
        P_o = torch.nan_to_num(P_o, nan=0.0, posinf=1e6, neginf=-1e6)

    dh2, idx_h, _ = knn_points(P_h[None], P_o[None], K=max(1, K))
    do2, idx_o, _ = knn_points(P_o[None], P_h[None], K=max(1, K))
    if K == 1:
        dh = torch.sqrt(dh2[0, :, 0].clamp_min(eps))
        do = torch.sqrt(do2[0, :, 0].clamp_min(eps))
    else:
        t = max(softmin_temp, 1e-6)
        dhK = torch.sqrt(dh2[0].clamp_min(eps))
        doK = torch.sqrt(do2[0].clamp_min(eps))
        w_h = F.softmax(-dhK / t, dim=1)
        w_o = F.softmax(-doK / t, dim=1)
        dh = (w_h * dhK).sum(dim=1)
        do = (w_o * doK).sum(dim=1)

    all_d = torch.cat([dh, do], 0)
    base = torch.quantile(all_d, float(min(max(q, 0.0), 0.5)))
    mean = all_d.mean()
    delta_adapt = float(base + margin_scale * (mean - base))

    g_min = float(min_gap if min_gap is not None else max(1e-8, min_gap_ratio * delta_adapt))
    g_contact = float(contact_target if contact_target is not None else max(g_min * 1.001, contact_ratio * delta_adapt))
    bw = float(band_width if band_width is not None else max(1e-8, band_ratio * delta_adapt))

    def hinge_repel(d):
        z = torch.relu(g_min - d)
        return z.mean(), float((d < g_min).float().mean())

    Lr_h, frac_r_h = hinge_repel(dh)
    Lr_o, frac_r_o = hinge_repel(do)
    L_repel = Lr_h + Lr_o

    def contact_keep(d):
        w = torch.exp(-0.5 * ((d - g_contact) / max(bw, 1e-8)) ** 2)
        if use_huber:
            res = d - g_contact
            delta_abs = huber_delta * max(bw, 1e-8)
            abs_res = torch.abs(res)
            quad = 0.5 * (abs_res ** 2) / max(delta_abs, 1e-8)
            lin = abs_res - 0.5 * delta_abs
            huber_val = torch.where(abs_res <= delta_abs, quad, lin)
            loss = (w * huber_val).sum() / (w.sum() + 1e-8)
        else:
            loss = (w * (d - g_contact) ** 2).sum() / (w.sum() + 1e-8)
        return loss, float((w > 1e-3).float().mean())

    Lc_h, frac_c_h = contact_keep(dh)
    Lc_o, frac_c_o = contact_keep(do)
    L_contact = Lc_h + Lc_o

    loss = repel_weight * L_repel + contact_weight * L_contact

    extras = {
        "min_d": float(all_d.min()), "mean_d": float(all_d.mean()),
        "delta_adapt": delta_adapt,
        "g_min": g_min, "g_contact": g_contact, "band_width": bw,
        "frac_repel_h": frac_r_h, "frac_repel_o": frac_r_o,
        "frac_contactband_h": frac_c_h, "frac_contactband_o": frac_c_o,
        "L_repel": L_repel.detach(), "L_contact": L_contact.detach(),
        "K": K, "softmin_temp": (softmin_temp if K > 1 else None),
        "use_huber": use_huber, "repel_weight": repel_weight,
        "contact_weight": contact_weight,
    }
    return loss, extras


# ====================================================================
# Registered losses
# ====================================================================

# ---------------------------------------------------------------------------
# Z-buffer contact loss (object optimisation)
# ---------------------------------------------------------------------------

def zbuf_contact_registered(cfg, ctx: LossContext) -> torch.Tensor:
    """Wrapper that mirrors ``Runner.compute_zbuf_contact_loss``."""
    hand_zbuf = getattr(ctx, 'hand_zbuf', None)
    obj_zbuf = ctx.rendered_zbuf

    if hand_zbuf is None or obj_zbuf is None:
        return torch.tensor(0.0, device=ctx.rendered_rgb.device)

    hand_occ_obj = ctx.gt_hand_mask.bool() & (ctx.rendered_object_alpha[0, ..., 0] > 1e-3)
    obj_occ_hand = ctx.gt_object_mask.bool() & (hand_zbuf[0, ..., 0] > 0)

    contact_loss_val, _info = zbuf_contact_loss_simplified(
        hand_zbuf[0], obj_zbuf[0],
        hand_occ_obj.detach(), obj_occ_hand.detach(),
    )
    return contact_loss_val * cfg.contact_lambda


zbuf_contact_registered.is_enabled = lambda cfg: cfg.object_contact_loss
zbuf_contact_registered = register_loss("zbuf_contact")(zbuf_contact_registered)


# ---------------------------------------------------------------------------
# SDF: object vertices queried against hand SDF
# ---------------------------------------------------------------------------

def object_sdf_loss(cfg, ctx: LossContext) -> torch.Tensor:
    """Mirrors ``Runner.compute_object_sdf_loss``."""
    from optimization.utils.sdf import query_sdf_differentiable

    if ctx.hand_sdf_grid is None:
        return torch.tensor(0.0, device=ctx.rendered_rgb.device)

    sdf_grid = ctx.hand_sdf_grid['sdf']
    norm_trans = ctx.hand_sdf_grid['norm_trans']
    scale = ctx.hand_sdf_grid['scale']

    obj_vert_canonical = ctx.object_vertices
    obj_scale = ctx.optimized_obj_scale if ctx.optimized_obj_scale is not None else 1.0
    R = ctx.obj_full_R
    t = ctx.obj_full_t

    v_scaled = obj_vert_canonical * obj_scale
    v_rotated = v_scaled @ R.T
    v_transformed = v_rotated + t.unsqueeze(0)

    sdf_values, _ = query_sdf_differentiable(
        sdf_grid, v_transformed, norm_trans, scale, get_vis_pcd=False
    )
    valid_mask = sdf_values.abs() < 1.0
    inside_mask = (sdf_values < 0.0) & valid_mask
    outside_mask = (sdf_values > 0.0) & valid_mask

    device = ctx.rendered_rgb.device
    inside_loss = torch.mean(-sdf_values[inside_mask]) if inside_mask.any() else torch.tensor(0.0, device=device)
    outside_loss = torch.mean(sdf_values[outside_mask]) if outside_mask.any() else torch.tensor(0.0, device=device)

    return inside_loss * cfg.sdf_inside_lambda + outside_loss * cfg.sdf_outside_lambda


# Disabled by default: main branch replaced this path with ``hand_sdf_for_object``
# (object optimiser queries hand vertices against object SDF, not the reverse),
# and in the current object-optimisation context ``ctx.hand_sdf_grid`` is never
# populated so this loss returns 0 while still paying the dispatch cost. Keep the
# implementation registered for future opt-in via an explicit flag, but do not
# share ``cfg.object_sdf_loss`` with ``hand_sdf_for_object_loss``.
object_sdf_loss.is_enabled = lambda cfg: False
object_sdf_loss = register_loss("object_sdf")(object_sdf_loss)


# ---------------------------------------------------------------------------
# SDF: hand vertices queried against object SDF (for object optimisation)
# ---------------------------------------------------------------------------

def hand_sdf_for_object_loss(cfg, ctx: LossContext) -> torch.Tensor:
    """Mirrors ``Runner.compute_hand_sdf_loss_for_object``."""
    from optimization.utils.sdf import query_sdf_differentiable

    if ctx.object_sdf_grid is None:
        return torch.tensor(0.0, device=ctx.rendered_rgb.device)

    sdf_grid = ctx.object_sdf_grid['sdf']
    norm_trans = ctx.object_sdf_grid['norm_trans']
    scale = ctx.object_sdf_grid['scale']

    hand_verts = ctx.hand_vertices.detach()
    R = ctx.obj_full_R
    t = ctx.obj_full_t
    hand_vert_in_ob = (hand_verts - t.unsqueeze(0)) @ R

    sdf_values, _ = query_sdf_differentiable(
        sdf_grid, hand_vert_in_ob, norm_trans, scale, get_vis_pcd=False
    )
    valid_mask = sdf_values.abs() < 1.0
    inside_mask = (sdf_values < 0.0) & valid_mask
    outside_mask = (sdf_values > 0.0) & valid_mask

    device = ctx.rendered_rgb.device
    inside_loss = torch.mean(-sdf_values[inside_mask]) if inside_mask.any() else torch.tensor(0.0, device=device)
    outside_loss = torch.mean(sdf_values[outside_mask]) if outside_mask.any() else torch.tensor(0.0, device=device)

    return inside_loss * cfg.sdf_inside_lambda + outside_loss * cfg.sdf_outside_lambda


hand_sdf_for_object_loss.is_enabled = lambda cfg: cfg.object_sdf_loss
hand_sdf_for_object_loss = register_loss("hand_sdf_for_object")(hand_sdf_for_object_loss)


# ---------------------------------------------------------------------------
# SDF: hand vertices queried against object SDF (for hand optimisation)
# ---------------------------------------------------------------------------

def hand_sdf_loss(cfg, ctx: LossContext) -> torch.Tensor:
    """Mirrors ``Runner.compute_hand_sdf_loss``."""
    from optimization.utils.sdf import query_sdf_differentiable

    if ctx.object_sdf_grid is None:
        return torch.tensor(0.0, device=ctx.rendered_rgb.device)

    sdf_grid = ctx.object_sdf_grid['sdf']
    norm_trans = ctx.object_sdf_grid['norm_trans']
    scale = ctx.object_sdf_grid['scale']

    hand_verts = ctx.hand_vertices

    sdf_values, _ = query_sdf_differentiable(
        sdf_grid, hand_verts, norm_trans, scale, get_vis_pcd=False
    )
    valid_mask = sdf_values.abs() < 1.0
    inside_mask = (sdf_values < 0.0) & valid_mask
    outside_mask = (sdf_values > 0.0) & valid_mask

    device = ctx.rendered_rgb.device
    inside_loss = torch.mean(-sdf_values[inside_mask]) if inside_mask.any() else torch.tensor(0.0, device=device)
    outside_loss = torch.mean(sdf_values[outside_mask]) if outside_mask.any() else torch.tensor(0.0, device=device)

    return inside_loss * cfg.sdf_inside_lambda + outside_loss * cfg.sdf_outside_lambda


hand_sdf_loss.is_enabled = lambda cfg: cfg.hand_sdf_loss
hand_sdf_loss = register_loss("hand_sdf")(hand_sdf_loss)
