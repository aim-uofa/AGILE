import torch
import torch.nn.functional as F
import numpy as np
import warp as wp

@wp.kernel(enable_backward=False)
def erode_depth_kernel(depth:wp.array(dtype=float, ndim=2),
                       out:wp.array(dtype=float, ndim=2),
                       radius:int, depth_diff_thres:float, ratio_thres:float, zfar:float):
    h,w = wp.tid()
    H = depth.shape[0]
    W = depth.shape[1]
    if w>=W or h>=H:
        return
    d_ori = depth[h,w]
    if d_ori<0.001 or d_ori>=zfar:
        out[h,w] = 0.0
    bad_cnt = float(0)
    total = float(0)
    for u in range(w-radius, w+radius+1):
        if u<0 or u>=W:
            continue
        for v in range(h-radius, h+radius+1):
            if v<0 or v>=H:
                continue
            cur_depth = depth[v,u]
            total += 1.0
            if cur_depth<0.001 or cur_depth>=zfar or abs(cur_depth-d_ori)>depth_diff_thres:
                bad_cnt += 1.0
    if bad_cnt/total>ratio_thres:
        out[h,w] = 0.0
    else:
        out[h,w] = d_ori


def erode_depth(depth, radius=2, depth_diff_thres=0.001, ratio_thres=0.8, zfar=100, device='cuda'):
    'Erode depth.'
    depth_wp = wp.from_torch(torch.as_tensor(depth, dtype=torch.float, device=device))
    out_wp = wp.zeros(depth.shape, dtype=float, device=device)
    wp.launch(kernel=erode_depth_kernel, device=device, dim=[depth.shape[0], depth.shape[1]], inputs=[depth_wp, out_wp, radius, depth_diff_thres, ratio_thres, zfar],)
    depth_out = wp.to_torch(out_wp)

    if isinstance(depth, np.ndarray):
      depth_out = depth_out.data.cpu().numpy()
    return depth_out

def compute_edge_sobel(mask: torch.Tensor):
    """
    Args:
        mask: [H, W], (0, 1) float

    Returns:
        edge_magnitude: [H, W]
    """

    assert mask.ndim == 2
    device = mask.device


    sobel_x = torch.tensor([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=mask.dtype, device=device).view(1, 1, 3, 3)

    sobel_y = torch.tensor([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=mask.dtype, device=device).view(1, 1, 3, 3)


    grad_x = F.conv2d(mask[None, None, ...], sobel_x, padding=1)
    grad_y = F.conv2d(mask[None, None, ...], sobel_y, padding=1)


    edge_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8) # [1, 1, H, W]

    return edge_magnitude.squeeze(0).squeeze(0) # [H, W]


def alpha2pts(mask, max_num_pts=1000):
    """
    mask: [H,W], float (0,1)
    return: pts_cam: [N,2] (y,x)
    """
    H, W = mask.shape
    device = mask.device
    ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    ys = ys.reshape(-1)
    xs = xs.reshape(-1)
    mask_flat = mask.reshape(-1)
    valid_idx = (mask_flat > 0.5)
    pts_cam = torch.stack([ys[valid_idx], xs[valid_idx]], dim=-1)  # [N,2]
    if max_num_pts > 0 and len(pts_cam) > max_num_pts:
        indices = torch.randperm(len(pts_cam), device=device)[:max_num_pts]
        pts_cam = pts_cam[indices]
    return pts_cam
