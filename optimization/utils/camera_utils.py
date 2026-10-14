"""Convert camera-to-world poses and OpenCV intrinsics for rasterization."""
import torch
from pytorch3d.utils import cameras_from_opencv_projection


def convert_camtoworld_to_pytorch3d_cameras(camtoworld_matrices, Ks, image_height, image_width, device):
    """Match OpenCV pixel projections, including off-center principal points.

    The official conversion handles PyTorch3D row-vector extrinsics, XY axis
    signs and NDC scaling by the shorter image dimension. Keep tensors attached
    so camera/pose optimization receives gradients.
    """
    world_to_cam = torch.linalg.inv(camtoworld_matrices.to(device))
    image_size = torch.tensor([image_height, image_width], dtype=Ks.dtype,
                              device=device)[None].expand(len(Ks), -1)
    return cameras_from_opencv_projection(
        R=world_to_cam[:, :3, :3], tvec=world_to_cam[:, :3, 3],
        camera_matrix=Ks.to(device), image_size=image_size,
    )
