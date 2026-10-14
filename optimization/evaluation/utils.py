import numpy as np
import cv2
import torch



def to_xy_batch(x_homo):
    assert isinstance(x_homo, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert x_homo.shape[2] == 3
    assert len(x_homo.shape) == 3
    batch_size = x_homo.shape[0]
    num_pts = x_homo.shape[1]
    x = torch.ones(batch_size, num_pts, 2, device=x_homo.device)
    zz = x_homo[:, :, 2:3]


    x = x_homo[:, :, :2] / zz
    return x


def project2d_batch(K, pts_cam):
    """
    K: (B, 3, 3)
    pts_cam: (B, N, 3)
    """

    assert isinstance(K, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert isinstance(pts_cam, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert K.shape[1:] == (3, 3)
    assert pts_cam.shape[2] == 3
    assert len(pts_cam.shape) == 3
    pts2d_homo = torch.bmm(K, pts_cam.permute(0, 2, 1)).permute(0, 2, 1)
    pts2d = to_xy_batch(pts2d_homo)
    return pts2d






def rotation_vector_to_matrix_safe(rotation_vector):
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


def compute_bounding_box_centers(vertices):
    """
    Compute the centers of the tight bounding box for a moving point cloud.

    Parameters:
    - vertices: A numpy array of shape (frames, num_verts, 3) representing the vertices of the object over time.

    Returns:
    - A numpy array of shape (frames, 3) where each row represents the center of the bounding box for each frame.
    """

    if isinstance(vertices, list):
        bbox_centers = []
        for verts in vertices:
            assert verts.shape[1] == 3
            bmin = np.min(verts, axis=0)
            bmax = np.max(verts, axis=0)
            bbox_center = (bmin + bmax) / 2
            bbox_centers.append(bbox_center)
        bbox_centers = np.stack(bbox_centers, axis=0)
    else:
        bbox_min = np.min(vertices, axis=1)
        bbox_max = np.max(vertices, axis=1)
        bbox_centers = (bbox_min + bbox_max) / 2
    return bbox_centers
