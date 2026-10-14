"""Temporal smoothness losses for object and hand pose sequences."""
from __future__ import annotations

import torch

from optimization.losses import register_loss, LossContext
from optimization.losses._utils import huber_loss
from optimization.utils.geometry import so3_log_batch


def temporal_smoothness_loss_se3(
    prev_R_seq: torch.Tensor,  # [T-1, 3, 3]
    prev_t_seq: torch.Tensor,  # [T-1, 3]
    curr_R: torch.Tensor,      # [3, 3]
    curr_t: torch.Tensor,      # [3]
    w_vel_r: float = 1.0,
    w_vel_t: float = 1.0,
    w_acc_r: float = 0.2,
    w_acc_t: float = 0.2,
    use_acc: bool = True,
    dt: float = 1 / 24,
):
    """SE(3) temporal smoothness using velocity / acceleration penalties."""
    def huber(x, delta):
        ax = x.abs()
        return torch.where(ax <= delta, 0.5 * (x ** 2) / (delta + 1e-12), ax - 0.5 * delta)

    R = torch.cat([prev_R_seq, curr_R.unsqueeze(0)], dim=0)
    t = torch.cat([prev_t_seq, curr_t.unsqueeze(0)], dim=0)
    T_len = R.shape[0]
    if T_len < 3:
        raise ValueError("prev_R_seq and prev_t_seq must contain at least 3 frames.")

    v_t_last = (t[-1] - t[-2]) / dt
    dR_last = R[-1] @ R[-2].transpose(-1, -2)
    v_r_last = so3_log_batch(dR_last.unsqueeze(0))[0] / dt

    delta_vel_t = 0.01
    delta_vel_r = 0.02
    Lv_t = huber(v_t_last.norm(), delta_vel_t)
    Lv_r = huber(v_r_last.norm(), delta_vel_r)

    if use_acc and T_len >= 3:
        v_t_prev = (t[-2] - t[-3]) / dt
        dR_prev = R[-2] @ R[-3].transpose(-1, -2)
        v_r_prev = so3_log_batch(dR_prev.unsqueeze(0))[0] / dt

        a_t_last = (v_t_last - v_t_prev) / dt
        a_r_last = (v_r_last - v_r_prev) / dt

        delta_acc_t = 0.005
        delta_acc_r = 0.01
        La_t = huber(a_t_last.norm(), delta_acc_t)
        La_r = huber(a_r_last.norm(), delta_acc_r)
    else:
        La_t = La_r = R.new_tensor(0.0)

    loss = w_vel_r * Lv_r + w_vel_t * Lv_t + w_acc_r * La_r + w_acc_t * La_t

    info = {
        "Lv_r": Lv_r.item(), "Lv_t": Lv_t.item(),
        "La_r": La_r.item(), "La_t": La_t.item(),
        "speed_r_last(rad/s)": v_r_last.norm().item(),
        "speed_t_last": v_t_last.norm().item(),
    }
    return loss, info


def temporal_smoothness_loss_vertices(
    vertices,  # [N, 3]
    prev_R,    # [3, 3]
    prev_t,    # [3]
    curr_R,    # [3, 3]
    curr_t,    # [3]
):
    """Vertex-based temporal smoothness."""
    dR = prev_R - curr_R
    dt = prev_t - curr_t
    vert_rotated = vertices @ dR.T
    vert_transformed = vert_rotated + dt
    return torch.norm(vert_transformed, p=2, dim=1).mean()


# ====================================================================
# Registered losses
# ====================================================================

# ---------------------------------------------------------------------------
# Object smoothness
# ---------------------------------------------------------------------------

def object_smoothness(cfg, ctx: LossContext) -> torch.Tensor:
    """Mirrors ``Runner.compute_obj_smoothness_loss``."""
    prev_R_seq = ctx.prev_obj_R_seq
    prev_t_seq = ctx.prev_obj_t_seq

    if prev_R_seq is None or prev_t_seq is None:
        return torch.tensor(0.0, device=ctx.rendered_rgb.device)
    if len(prev_R_seq) == 0 or len(prev_t_seq) == 0:
        return torch.tensor(0.0, device=ctx.rendered_rgb.device)

    curr_R = ctx.obj_full_R
    curr_t = ctx.obj_full_t

    if cfg.object_smoothness_type == 'se3':
        smoothness_loss, _info = temporal_smoothness_loss_se3(
            prev_R_seq, prev_t_seq,
            curr_R, curr_t,
            w_vel_r=0.2, w_vel_t=0.2,
            w_acc_r=1.0, w_acc_t=1.0,
        )
    elif cfg.object_smoothness_type == 'vertices':
        vertices = ctx.object_vertices
        assert vertices is not None, "Vertices must be provided for vertex-based smoothness loss."
        smoothness_loss = temporal_smoothness_loss_vertices(
            vertices,
            prev_R_seq[-1], prev_t_seq[-1],
            curr_R, curr_t,
        )
    else:
        raise NotImplementedError(
            f"Smoothness type {cfg.object_smoothness_type} not implemented."
        )

    return smoothness_loss * cfg.object_smoothness_lambda


object_smoothness.is_enabled = lambda cfg: cfg.object_smoothness_loss
object_smoothness = register_loss("object_smoothness")(object_smoothness)


# ---------------------------------------------------------------------------
# Hand smoothness
# ---------------------------------------------------------------------------

def hand_smoothness(cfg, ctx: LossContext) -> torch.Tensor:
    """Mirrors ``Runner.compute_hand_smoothness_loss``."""
    prev_hand_verts = ctx.prev_hand_verts_seq
    curr_hand_verts = ctx.hand_vertices

    if prev_hand_verts is None or curr_hand_verts is None:
        return torch.tensor(0.0, device=ctx.rendered_rgb.device)

    dist = torch.norm(prev_hand_verts - curr_hand_verts, p=2, dim=1).mean()
    return dist * cfg.hand_smoothness_lambda


hand_smoothness.is_enabled = lambda cfg: cfg.hand_smoothness_loss
hand_smoothness = register_loss("hand_smoothness")(hand_smoothness)
