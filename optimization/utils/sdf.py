import torch
import torchcumesh2sdf
import torch.nn.functional as F
import open3d as o3d
from matplotlib import cm
import numpy as np


def compute_sdf(vertices: torch.Tensor, # [N, 3], not normalized
                faces: torch.Tensor,
                resolution: int = 1024,
                band: float = 0.1,
                dist_th_wld: float = None):

    # Normalize vertices to fit within the unit cube
    v_min = vertices.min(dim=0)[0]
    v_max = vertices.max(dim=0)[0]
    if dist_th_wld is not None:
        scale = (v_max - v_min + 2 * dist_th_wld).max()
        norm_trans = -(v_min - dist_th_wld)
    else:
        scale = (v_max - v_min).max() / (1.0 - 2 * band)
        norm_trans = -v_min
    v_normalized = (vertices + norm_trans) / scale

    triangles = vertices[faces]  # [F, 3, 3]
    t_pts = triangles.reshape(-1, 3)  # [F*3, 3]

    # o3d.io.write_point_cloud("sdf_temp_pts.ply", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(t_pts.detach().cpu().numpy())))

    # sdf_grid = torchcumesh2sdf.get_sdf(
    #     tris=v_normalized[faces], # [F, 3, 3]
    #     R=resolution,
    #     # R=64,
    #     band=band
    # )  # [R, R, R]

    faces_swapped = faces[:, [0, 2, 1]]
    sdf_grid1 = torchcumesh2sdf.get_sdf(
        tris=v_normalized[faces_swapped], # [F, 3, 3]
        R=resolution,
        # R=64,
        band=band
    )  # [R, R, R]

    return sdf_grid1, norm_trans, scale


def query_sdf_differentiable(sdf_grid: torch.Tensor,  # [R, R, R]
                             vertices: torch.Tensor,  # [N, 3], requires_grad=True
                             norm_trans: torch.Tensor,  # [3]
                             norm_scale: float,
                             get_vis_pcd: bool = False):
    """
    Gradients supported SDF query
    """
    R = sdf_grid.shape[0]
    # Normalize to [0, 1]^3
    vertices_normalized = (vertices + norm_trans) / norm_scale  # [N, 3]
    coords = vertices_normalized.clamp(0., 1.)  # Prevent out-of-bounds

    # Reverse XYZ to ZYX
    coords_sample = torch.stack([
        coords[:, 2],  # Z → dim0 (D)
        coords[:, 1],  # Y → dim1 (H)
        coords[:, 0],  # X → dim2 (W)
    ], dim=-1)  # [N, 3], (Z, Y, X)
    coords_sample = coords_sample * 2 - 1
    coords_sample = coords_sample.view(1, 1, 1, -1, 3)
    sdf_grid_5d = sdf_grid.unsqueeze(0).unsqueeze(0)

    # grid_sample SDF values
    sdf_values = F.grid_sample(
        sdf_grid_5d, coords_sample,
        mode='bilinear',
        align_corners=True
    ).squeeze()


    if get_vis_pcd:
        vis_threshold = 0.01
        sdf_pcd = vis_sdf_as_pointcloud(
            sdf_grid,
            threshold = vis_threshold
        )

        # points_np = coords.detach().cpu().numpy() # normalized to [0, 1]
        points_np = vertices_normalized.detach().cpu().numpy() # normalized to [0, 1]
        sdf_values_np = sdf_values.detach().cpu().numpy()


        norm_sdf = np.clip(sdf_values_np / vis_threshold, -1, 1)
        colors = cm.RdBu_r((norm_sdf + 1) / 2)[:, :3]

        outliers_mask = np.abs(sdf_values_np) > 1e8
        points_np = points_np[~outliers_mask]
        colors = colors[~outliers_mask]

        query_pcd = o3d.geometry.PointCloud()
        query_pcd.points = o3d.utility.Vector3dVector(points_np)
        query_pcd.colors = o3d.utility.Vector3dVector(colors)

        pcd_combined = sdf_pcd + query_pcd
    else:
        pcd_combined = None

    return sdf_values, pcd_combined


def vis_sdf_as_pointcloud(
    sdf_grid: torch.Tensor,  # [R, R, R]
    threshold: float = 0.005,
    subsample: int = 1,
    scale: torch.Tensor = None,
    translate: torch.Tensor = None
):
    'Convert an SDF grid to an Open3D point cloud, retaining values below threshold and applying spatial subsampling.'
    R = sdf_grid.shape[0]


    if subsample > 1:
        sdf_grid = sdf_grid[::subsample, ::subsample, ::subsample]
        R = sdf_grid.shape[0]


    coords = torch.linspace(0, 1, R, device=sdf_grid.device)
    # zz, yy, xx = torch.meshgrid(coords, coords, coords, indexing='ij')
    xx, yy, zz = torch.meshgrid(coords, coords, coords, indexing='ij')


    mask = torch.abs(sdf_grid) < threshold
    # mask = sdf_grid < 0

    points = torch.stack([xx[mask], yy[mask], zz[mask]], dim=-1)  # [N, 3]
    sdf_values = sdf_grid[mask]  # [N]


    points_np = points.cpu().numpy() # normalized to [0, 1]
    sdf_values_np = sdf_values.cpu().numpy()


    norm_sdf = np.clip(sdf_values_np / threshold, -1, 1)
    colors = cm.RdBu_r((norm_sdf + 1) / 2)[:, :3]

    if scale is not None:
        points_np = points_np * scale.detach().cpu().numpy()
    if translate is not None:
        points_np -= translate.detach().cpu().numpy()

    # print(f"points_np min: {points_np.min(axis=0)}, max: {points_np.max(axis=0)}")


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    pcd.colors = o3d.utility.Vector3dVector(colors)


    return pcd


    # print("SDF values point cloud saved to sdf_values_pointcloud.ply")
