import torch
import torch.nn.functional as F
import numpy as np
import random
from matplotlib import pyplot as plt
import open3d as o3d
import math
from pytorch3d.ops import knn_points
from pytorch3d.transforms import (
    matrix_to_quaternion as p3d_matrix_to_quaternion,
    quaternion_to_axis_angle as p3d_quaternion_to_axis_angle,
)

glcam_in_cvcam = np.array([[1,0,0,0],
                          [0,-1,0,0],
                          [0,0,-1,0],
                          [0,0,0,1]]).astype(float)


def rotation_matrix_to_rvec(R: torch.Tensor) -> torch.Tensor:
    """Convert SO(3) matrices (..., 3, 3) to Rodrigues vectors (..., 3).

    The quaternion path avoids dividing the antisymmetric part by sin(theta),
    which is ill-conditioned near 180 degrees, especially for float32 matrices.
    It preserves dtype, device, batching, and autograd. At exactly pi either
    axis sign represents the same rotation; the axis-angle chart is not globally
    continuous there, so stored poses should retain their rotation matrices.
    """
    return p3d_quaternion_to_axis_angle(p3d_matrix_to_quaternion(R))


def rotation_vector_to_matrix_safe(rotation_vector: torch.Tensor) -> torch.Tensor:
    """ Differentiable and numerically stable conversion from rotation vector to rotation matrix using Rodriguez formula."""
    device = rotation_vector.device
    dtype = rotation_vector.dtype


    angle = torch.linalg.vector_norm(rotation_vector)


    angle_sq = angle * angle


    small_angle_threshold = 1e-4


    sin_over_angle = torch.where(
        angle > small_angle_threshold,
        torch.sin(angle) / torch.clamp(angle, min=1e-8),
        1.0 - angle_sq / 6.0 + angle_sq * angle_sq / 120.0
    )


    one_minus_cos_over_angle_sq = torch.where(
        angle > small_angle_threshold,
        (1.0 - torch.cos(angle)) / torch.clamp(angle_sq, min=1e-12),
        0.5 - angle_sq / 24.0 + angle_sq * angle_sq / 720.0
    )


    rx, ry, rz = rotation_vector[0], rotation_vector[1], rotation_vector[2]
    K = torch.stack([
        torch.stack([torch.zeros_like(rx), -rz, ry]),
        torch.stack([rz, torch.zeros_like(rx), -rx]),
        torch.stack([-ry, rx, torch.zeros_like(rx)])
    ])


    I = torch.eye(3, device=device, dtype=dtype)
    R = I + sin_over_angle * K + one_minus_cos_over_angle_sq * torch.matmul(K, K)

    return R


def matrix_to_quaternion(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Convert rotation matrix (3x3 or Nx3x3) to quaternion (w, x, y, z).
    Args:
        R: (..., 3, 3) rotation matrix
        eps: small value to avoid numerical issues
    Returns:
        q: (..., 4) quaternion, normalized
    """
    assert R.shape[-2:] == (3, 3), "R must be (..., 3, 3)"
    single_input = R.ndim == 2
    if single_input:
        R = R.unsqueeze(0)  # [1, 3, 3]

    # --- 1. Ensure R is a valid rotation matrix via SVD orthogonalization ---
    with torch.no_grad():
        U, _, Vt = torch.linalg.svd(R)
        R = U @ Vt
        # Handle improper rotation (det = -1) by flipping sign
        det = torch.det(R)
        mask = det < 0
        if mask.any():
            U[mask, :, -1] *= -1
            R = U @ Vt

    # --- 2. Extract matrix elements ---
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    # --- 3. Compute quaternion based on max diagonal element ---
    q = torch.zeros((*R.shape[:-2], 4), device=R.device, dtype=R.dtype)

    # Case 1: m00 is largest
    mask = (m00 > m11) & (m00 > m22)
    s = torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=eps)) * 2
    q[mask, 0] = (m21[mask] - m12[mask]) / s
    q[mask, 1] = 0.25 * s
    q[mask, 2] = (m01[mask] + m10[mask]) / s
    q[mask, 3] = (m02[mask] + m20[mask]) / s

    # Case 2: m11 is largest
    mask = (m11 > m22) & ~(m00 > m11)
    s = torch.sqrt(torch.clamp(1.0 + m11 - m00 - m22, min=eps)) * 2
    q[mask, 0] = (m02[mask] - m20[mask]) / s
    q[mask, 1] = (m01[mask] + m10[mask]) / s
    q[mask, 2] = 0.25 * s
    q[mask, 3] = (m12[mask] + m21[mask]) / s

    # Case 3: m22 is largest
    mask = ~(m00 > m11) & ~(m11 > m22)
    s = torch.sqrt(torch.clamp(1.0 + m22 - m00 - m11, min=eps)) * 2
    q[mask, 0] = (m10[mask] - m01[mask]) / s
    q[mask, 1] = (m02[mask] + m20[mask]) / s
    q[mask, 2] = (m12[mask] + m21[mask]) / s
    q[mask, 3] = 0.25 * s

    # --- 4. Normalize quaternion ---
    q = q / (torch.norm(q, dim=-1, keepdim=True) + eps)

    return q[0] if single_input else q


def compute_vertex_normals(vertices, faces):
    """Compute vertex normals from mesh vertices and faces."""
    # Compute face normals
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    face_normals = np.cross(v1 - v0, v2 - v0)
    face_normals = face_normals / (np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-6)

    # Average face normals to get vertex normals
    vertex_normals = np.zeros_like(vertices)
    for i, face in enumerate(faces):
        vertex_normals[face] += face_normals[i]

    vertex_normals = vertex_normals / (np.linalg.norm(vertex_normals, axis=1, keepdims=True) + 1e-6)
    return vertex_normals


def _skew(v):  # v:[...,3]
    vx, vy, vz = v[...,0], v[...,1], v[...,2]
    O = torch.zeros_like(vx)
    return torch.stack([
        torch.stack([ O,   -vz,  vy], dim=-1),
        torch.stack([ vz,   O,  -vx], dim=-1),
        torch.stack([-vy,   vx,   O], dim=-1),
    ], dim=-2)  # [...,3,3]

def rodrigues_batch(r):  # r:[...,3]
    orig = r.shape[:-1]
    r = r.reshape(-1, 3)
    T = r.shape[0]
    I = torch.eye(3, dtype=r.dtype, device=r.device).expand(T, 3, 3)
    theta = torch.norm(r, dim=-1, keepdim=True)             # [T,1]
    K = _skew(r)                                            # [T,3,3]
    th2 = theta**2
    s_over_t = torch.where(theta > 1e-4,
                           torch.sin(theta) / (theta + 1e-12),
                           1.0 - th2/6.0 + th2*th2/120.0)[..., None]
    one_m_c_over_t2 = torch.where(theta > 1e-4,
                           (1.0 - torch.cos(theta)) / (th2 + 1e-12),
                           0.5 - th2/24.0 + th2*th2/720.0)[..., None]
    R = I + s_over_t * K + one_m_c_over_t2 * (K @ K)
    return R.reshape(*orig, 3, 3)

def so3_log_batch(R):
    """Batched SO(3) logarithm through the stable quaternion conversion.

    The antisymmetric/sin(theta) formula loses the rotation at a half-turn.
    Keep the same implementation for pose fitting and temporal SE(3) losses.
    """
    return rotation_matrix_to_rvec(R)

def so3_exp_batch(w):  # w:[...,3] -> R:[...,3,3]
    return rodrigues_batch(w)


def compute_pcd_diameter(points, downsample_num=5000):
    if not isinstance(points, torch.Tensor):
        pcd = torch.tensor(points)
    else:
        pcd = points
    n_pts = pcd.shape[0]
    if n_pts > downsample_num:
        print(f"Too many points, randomly sample {downsample_num} points for diameter computation.")
        pcd = pcd[torch.randperm(n_pts)[:downsample_num]]
        n_pts = downsample_num
    print(pcd.shape)

    max_dist = 0.0
    batch_size = 2048
    for i in range(0, n_pts, batch_size):
        end_i = min(i + batch_size, n_pts)
        batch_points = pcd[i:end_i]  # (batch_size, 3)
        diff = batch_points.unsqueeze(1) - pcd.unsqueeze(0)  # (batch_size, N, 3)
        distances = torch.norm(diff, dim=2)  # (batch_size, N)
        batch_max = torch.max(distances)

        max_dist = max(max_dist, batch_max.item())

    return max_dist


def sample_2d_3d_pts_from_depth(depth, cam_K, max_num_pts=1000):
    """
    depth: [H,W]
    cam_K: [3,3]
    return: pts_cam: [H,W,3] or [N,3] if flatten=True
    """
    H, W = depth.shape
    device = depth.device

    ys, xs = torch.meshgrid(torch.arange(H, device=device),
                            torch.arange(W, device=device), indexing='ij') # [H,W]
    fx, fy = cam_K[0,0], cam_K[1,1]
    cx, cy = cam_K[0,2], cam_K[1,2]

    X = (xs - cx) * depth / fx
    Y = (ys - cy) * depth / fy
    Z = depth

    p3d_cam = torch.stack([X,Y,Z], dim=-1)  # [H, W, 3]

    valid_mask = depth.reshape(-1) > 0.0
    valid_indices = torch.where(valid_mask)[0] # [N_valid], in (0, H*W-1)

    if max_num_pts > 0 and len(valid_indices) > max_num_pts:
        selected_indices = torch.randperm(len(valid_indices), device=device)[:max_num_pts]
        valid_indices = valid_indices[selected_indices]

    p3d_cam = p3d_cam.reshape(-1,3)[valid_indices]  # [N_valid, 3]
    xs = xs.reshape(-1)[valid_indices]  # [N_valid]
    ys = ys.reshape(-1)[valid_indices]  # [N_valid]
    p2d = torch.stack([xs, ys], dim=-1)  # [N_valid, 2]

    return p2d, p3d_cam, valid_indices


def project_points(
    p3d: torch.Tensor, # [N, 3]
    K: torch.Tensor, # [3, 3]
) -> torch.Tensor:
    z = p3d[:, 2:3].clamp(min=1e-8)
    p3d_norm = p3d / z
    p2d = p3d_norm @ K.T
    p2d = p2d[:, :2]
    return p2d
