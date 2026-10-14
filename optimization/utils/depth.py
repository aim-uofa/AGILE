"""
Utility functions for depth map processing and normal map computation.
"""

import torch
import torch.nn.functional as F
import numpy as np

from .image import erode_depth
from .geometry import compute_pcd_diameter


def depths_to_points(K, depth, c2w=None):
    """
    Convert depth map to 3D points in camera or world space.

    Args:
        K: Camera intrinsic matrix [3, 3] or [B, 3, 3]
        depth: Depth map [H, W] or [B, H, W] or [B, 1, H, W]
        c2w: Optional camera-to-world transform [4, 4] or [B, 4, 4]
             If None, returns points in camera space

    Returns:
        points: 3D points [H, W, 3] or [B, H, W, 3] or [B, 3, H, W]
    """
    # Handle batch dimension and squeeze single channel
    if depth.ndim == 4 and depth.shape[1] == 1:  # [B, 1, H, W]
        depth = depth.squeeze(1)  # [B, H, W]

    batch_mode = depth.ndim == 3  # [B, H, W]

    if not batch_mode:
        depth = depth.unsqueeze(0)  # [1, H, W]
        K = K.unsqueeze(0) if K.ndim == 2 else K
        if c2w is not None:
            c2w = c2w.unsqueeze(0) if c2w.ndim == 2 else c2w

    B, H, W = depth.shape
    device = depth.device

    # Create pixel grid
    u, v = torch.meshgrid(
        torch.arange(W, device=device, dtype=depth.dtype),
        torch.arange(H, device=device, dtype=depth.dtype),
        indexing='xy'
    )  # [H, W]

    # Expand to batch
    u = u.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
    v = v.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]

    # Get intrinsic parameters
    if K.ndim == 2:
        K = K.unsqueeze(0).expand(B, -1, -1)

    fx = K[:, 0, 0].view(B, 1, 1)
    fy = K[:, 1, 1].view(B, 1, 1)
    cx = K[:, 0, 2].view(B, 1, 1)
    cy = K[:, 1, 2].view(B, 1, 1)

    # Unproject to camera space
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth

    # Stack to points [B, H, W, 3]
    points = torch.stack([x, y, z], dim=-1)

    # Transform to world space if c2w is provided
    if c2w is not None:
        # points: [B, H, W, 3] -> [B, H*W, 3]
        points_flat = points.reshape(B, -1, 3)
        # Add homogeneous coordinate
        ones = torch.ones(B, H*W, 1, device=device, dtype=points.dtype)
        points_hom = torch.cat([points_flat, ones], dim=-1)  # [B, H*W, 4]

        # Transform: [B, 4, 4] @ [B, 4, H*W] -> [B, 4, H*W]
        points_world = torch.bmm(c2w, points_hom.transpose(1, 2)).transpose(1, 2)
        points = points_world[..., :3].reshape(B, H, W, 3)

    if not batch_mode:
        points = points.squeeze(0)  # [H, W, 3]

    return points


def depth_to_normal(K, depth, c2w=None, valid_mask=None, handle_boundaries=True):
    """
    Convert depth map to normal map using central differences.

    Args:
        K: Camera intrinsic matrix [3, 3] or [B, 3, 3]
        depth: Depth map [H, W] or [B, H, W] or [B, 1, H, W]
        c2w: Optional camera-to-world transform [4, 4] or [B, 4, 4]
        valid_mask: Optional mask for valid depth values [H, W] or [B, H, W]
        handle_boundaries: If True, replicate normals at boundaries

    Returns:
        normal_map: Normal map [H, W, 3] or [B, 3, H, W]
                   Normals are in camera space (or world space if c2w provided)
    """
    # Get 3D points in camera (or world) space
    points = depths_to_points(K, depth, c2w)  # [H, W, 3] or [B, H, W, 3]

    original_batch = points.ndim == 4
    if not original_batch:
        points = points.unsqueeze(0)  # [1, H, W, 3]
        if valid_mask is not None and valid_mask.ndim == 2:
            valid_mask = valid_mask.unsqueeze(0)

    B, H, W, _ = points.shape
    device = points.device
    dtype = points.dtype

    # Use central differences (more accurate than forward/backward)
    # dx: change in x direction (horizontal)
    # dy: change in y direction (vertical)
    dx = points[:, 1:-1, 2:] - points[:, 1:-1, :-2]  # [B, H-2, W-2, 3]
    dy = points[:, 2:, 1:-1] - points[:, :-2, 1:-1]  # [B, H-2, W-2, 3]

    # Cross product: dy × dx (for correct orientation)
    # In camera space: camera looks at +Z, so we want normals pointing toward camera (-Z)
    normal_map = torch.cross(dy, dx, dim=-1)  # [B, H-2, W-2, 3]

    # Normalize
    normal_map = F.normalize(normal_map, dim=-1, eps=1e-6)

    # Create output with boundaries
    output = torch.zeros(B, H, W, 3, device=device, dtype=dtype)
    output[:, 1:-1, 1:-1, :] = normal_map

    if handle_boundaries:
        # Replicate nearest valid normal to boundaries
        # Top and bottom rows
        output[:, 0, :, :] = output[:, 1, :, :]
        output[:, -1, :, :] = output[:, -2, :, :]
        # Left and right columns
        output[:, :, 0, :] = output[:, :, 1, :]
        output[:, :, -1, :] = output[:, :, -2, :]

    # Apply valid mask if provided
    if valid_mask is not None:
        if valid_mask.ndim == 3:  # [B, H, W]
            valid_mask = valid_mask.unsqueeze(-1)  # [B, H, W, 1]
        output = output * valid_mask.float()

    # Convert to [B, 3, H, W] format for compatibility with image tensors
    output = output.permute(0, 3, 1, 2)  # [B, 3, H, W]

    if not original_batch:
        output = output.squeeze(0)  # [3, H, W]

    return output


def depth2xyzmap(depth, K, uvs=None):
  invalid_mask = (depth<0.001)
  H,W = depth.shape[:2]
  if uvs is None:
    vs,us = np.meshgrid(np.arange(0,H),np.arange(0,W), sparse=False, indexing='ij')
    vs = vs.reshape(-1)
    us = us.reshape(-1)
  else:
    uvs = uvs.round().astype(int)
    us = uvs[:,0]
    vs = uvs[:,1]
  zs = depth[vs,us]
  xs = (us-K[0,2])*zs/K[0,0]
  ys = (vs-K[1,2])*zs/K[1,1]
  pts = np.stack((xs.reshape(-1),ys.reshape(-1),zs.reshape(-1)), 1)  #(N,3)
  xyz_map = np.zeros((H,W,3), dtype=np.float32)
  xyz_map[vs,us] = pts
  xyz_map[invalid_mask] = 0
  return xyz_map


def depth2pts_torch(
        depth: torch.Tensor, # [H, W]
        K: torch.Tensor,     # [3, 3]
    ):
    H,W = depth.shape
    device = depth.device
    vs,us = torch.meshgrid(
        torch.arange(H, device=device, dtype=depth.dtype),
        torch.arange(W, device=device, dtype=depth.dtype),
        indexing='ij'
    )  # [H, W]
    us = us.reshape(-1).long()
    vs = vs.reshape(-1).long()
    zs = depth[vs,us]  # [N]
    xs = (us - K[0,2]) * zs / K[0,0]
    ys = (vs - K[1,2]) * zs / K[1,1]
    pts = torch.stack((xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)), dim=1)  # (N, 3)
    valid_mask = zs > 0.001
    pts = pts[valid_mask]  # (N_valid, 3)
    return pts
