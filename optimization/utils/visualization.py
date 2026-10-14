import numpy as np
import cv2
import torch
import open3d as o3d
import os
import imageio
from .geometry import project_points

def overlay_rgb(front,     # [H, W, 3], float32, [0, 1]
                back,      # [H, W, 3], float32, [0, 1]
                front_mask # [H, W, 1], float32, [0, 1]
    ):
    alpha_back = 1.0
    canvas = front * front_mask + back * alpha_back * (1 - front_mask)
    return canvas

def distance_heatmap(distance, max_distance=0.03, valid_mask=None):
    """RGB diagnostic with a fixed metric scale; legacy zero means background."""
    d = np.asarray(distance)
    if d.ndim == 3:
        d = d[..., 0]
    if max_distance <= 0:
        raise ValueError("max_distance must be positive")
    valid = np.isfinite(d) & (d > 0) if valid_mask is None else np.isfinite(d) & np.asarray(valid_mask).reshape(d.shape)
    values = np.rint(np.clip(np.nan_to_num(d) / max_distance, 0, 1) * 255).astype(np.uint8)
    rgb = cv2.applyColorMap(values, cv2.COLORMAP_JET)[..., ::-1].copy()
    rgb[~valid] = 0
    return rgb


def make_vis_image(
        frame_id: int,

        # Rendering info
        rendered_color: torch.Tensor,    # [H, W, 3]
        color_gt: torch.Tensor,          # [H, W, 3]
        rendered_alpha: torch.Tensor,    # [H, W, 1]
        mask_gt: torch.Tensor,           # [H, W, 1]
        depth_diff: torch.Tensor = None, # [H, W, 1] or None
        max_depth_diff: float = 0.1,

        # Info of other modules

        # 'contact_map': torch.Tensor [H, W],
        # 'contact_debug_color': torch.Tensor [H, W, 3] in [0,1]
        contact_info: dict = None,

        # 'hand_joints': torch.Tensor [N, 2],
        # 'hand_joints_gt': torch.Tensor [N, 2]
        hand_joints_info: dict = None,

        # 'feature_similarity': torch.Tensor [N,],
        # 'feature_visible_pts': torch.Tensor [N, 2]
        feature_info: dict = None,

        # 'weights': torch.Tensor [N,],
        # 'curr_v_cam': torch.Tensor [N, 3],
        # 'last_v_cam': torch.Tensor [N, 3],
        # 'cam_K': torch.Tensor [3, 3]
        ho_pose_info: dict = None,
    ) -> np.ndarray:
    assert rendered_color.ndim == 3 and rendered_color.shape[2] == 3
    assert color_gt.ndim == 3 and color_gt.shape[2] == 3
    assert rendered_alpha.ndim == 3 and rendered_alpha.shape[2] == 1
    assert mask_gt.ndim == 3 and mask_gt.shape[2] == 1

    rendered_color = rendered_color.detach().cpu().numpy()
    color_gt = color_gt.detach().cpu().numpy()
    rendered_alpha = rendered_alpha.detach().cpu().numpy()
    mask_gt = mask_gt.detach().cpu().numpy()

    # RGB visualization (left top)
    front = rendered_color
    back = color_gt
    front_mask = rendered_alpha
    rgb_overlay = (overlay_rgb(front, back, front_mask) * 255).astype(np.uint8)

    # Alpha visualization with edges (left bottom)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_edge = cv2.morphologyEx(mask_gt[..., 0].astype(np.uint8),
                                 cv2.MORPH_GRADIENT, morph_kernel) # [H, W]
    object_alpha_vis = rendered_alpha.repeat(3, axis=2)  # [H, W, 3]
    object_alpha_vis[mask_edge > 0] = np.array([0.0, 1.0, 0.0])  # Green edges
    object_alpha_vis = (object_alpha_vis * 255).astype(np.uint8)

    # Depth visualization or feature similarity (right top)
    if depth_diff is not None:
        depth_diff = depth_diff.detach().cpu().numpy()
        diff_max = depth_diff.max() if max_depth_diff is None else max_depth_diff
        diff_min = depth_diff.min()
        depth_diff = np.clip(depth_diff, diff_min, diff_max)
        depth_vis = (depth_diff - diff_min) / (diff_max - diff_min + 1e-8)
        depth_vis = (depth_vis * 255).astype(np.uint8)[..., 0] # [H, W]
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET) # [H, W, 3]
        depth_vis = cv2.putText(depth_vis, f"Max Err: {diff_max*1000:.1f} mm", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)[..., ::-1]
        right_up = depth_vis
    elif feature_info is not None:
        feature_similarity = feature_info['similarity'].detach().cpu().numpy()
        feature_visible_pts = feature_info['xys_image'].detach().cpu().numpy()
        vis_image = np.zeros_like(rgb_overlay)
        sim_values = (feature_similarity * 255).astype(np.uint8)  # [N,]
        if sim_values.size > 0:
            sim_colors = cv2.applyColorMap(sim_values, cv2.COLORMAP_JET)[..., ::-1]  # [N, 1, 3], BGR->RGB
            random_indices = np.random.permutation(feature_visible_pts.shape[0])
            for i in random_indices:
                x, y = feature_visible_pts[i]
                x, y = int(x), int(y)
                cv2.circle(vis_image, (x, y), 2, sim_colors[i, 0].tolist(), -2)
        right_up = vis_image
    else:
        right_up = np.zeros_like(rgb_overlay)

    # Contact visualization or relpose visualization (right bottom)
    if contact_info is not None:
        debug = contact_info.get('debug_color')
        distance = contact_info.get('dist_map')
        if debug is not None:
            right_down = (debug.detach().cpu().numpy() * 255).astype(np.uint8)
            cv2.putText(right_down, "Contact debug colors (not distance scale)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        elif distance is not None:
            right_down = distance_heatmap(distance.detach().cpu().numpy())
            cv2.putText(right_down, "Distance: 0 - 30 mm", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1)
        else:
            right_down = np.zeros_like(rgb_overlay)
    else:
        right_down = np.zeros_like(rgb_overlay)

    # Hand joints visualization
    if hand_joints_info is not None:
        hand_joints_gt = hand_joints_info['joints_2d_gt'].detach().cpu().numpy()
        hand_joints = hand_joints_info['joints_2d'].detach().cpu().numpy()
        rgb_overlay = visualize_2d_points(rgb_overlay, hand_joints_gt, color=(0, 255, 0), radius=5) # Green
        rgb_overlay = visualize_2d_points(rgb_overlay, hand_joints, color=(255, 0, 0), radius=3) # Red

    # HO pose visualization
    if ho_pose_info is not None:
        curr_v_cam = ho_pose_info['curr_v_cam']
        last_v_cam = ho_pose_info['last_v_cam']
        cam_K = ho_pose_info['cam_K']
        weights = ho_pose_info['weights']
        lightness = weights.detach().cpu().numpy()

        curr_v_2d = project_points(curr_v_cam, cam_K).detach().cpu().numpy()  # [N, 2]
        last_v_2d = project_points(last_v_cam, cam_K).detach().cpu().numpy()  # [N, 2]
        rgb_overlay = visualize_2d_points(rgb_overlay, curr_v_2d, color=(0, 255, 0), radius=4, lightness=lightness) # Green
        rgb_overlay = visualize_2d_points(rgb_overlay, last_v_2d, color=(255, 0, 0), radius=2, lightness=lightness) # Red


    left_half = np.concatenate([rgb_overlay, object_alpha_vis], axis=0)
    right_half = np.concatenate([right_up, right_down], axis=0)
    vis_image = np.concatenate([left_half, right_half], axis=1)
    vis_image = cv2.putText(vis_image, f"Frame {frame_id}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return vis_image


    # o3d.visualization.draw_geometries([pcd1, pcd2])


def visualize_2d_points(image: np.ndarray, # [H, W, 3], [0, 255]
                        pts_2d: np.ndarray, # [N, 2]
                        color=(0, 255, 0), radius=3,
                        lightness=None):
    vis_image = image.copy()
    for i, pt in enumerate(pts_2d):
        x, y = int(pt[0]), int(pt[1])
        if lightness is not None:
            color = np.array(color)
            color_vis = (color * lightness[i]).astype(np.int32).tolist()
        else:
            color_vis = color
        cv2.circle(vis_image, (x, y), radius, color_vis, -1)
    return vis_image


"""
Simple visualization data loader for HOI scenes.
This loader is designed to work directly with processed data from train_mesh.py,
providing a clean interface for visualization without optimization components.
"""

import numpy as np
from pathlib import Path
import trimesh
import torch
import yaml
import sys
import os
from optimization.utils.geometry import rotation_vector_to_matrix_safe
from optimization.models.mano_wrapper import MANO


# Backward-compatible import; rendering and packaging share one data contract.
from optimization.data.visualization_loader import VisualizationLoader
