from transformers import AutoModel, AutoImageProcessor
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import cv2


DEFAULT_MODEL_PATH = os.environ.get(
    "DINO_MODEL_PATH",
    "assets/dinov3",
)
class DINOWrapper(torch.nn.Module):
    def __init__(self, device: str = 'cuda:0',
                 model_path: str = DEFAULT_MODEL_PATH,
                 input_size: int = 518):
        super().__init__()
        self.model_path = model_path
        print(f"Loading DINO model from {model_path}...")
        self.model = AutoModel.from_pretrained(model_path).to(device)
        print("DINO model loaded.")
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_path,
            size={'height': input_size, 'width': input_size},
        )

    @torch.no_grad()
    def forward(self, images: np.ndarray): # (B, H, W, 3) in [0, 1]
        assert images.max() <= 1.0 and images.min() >= 0.0,\
            "Input images should be in [0, 1] range."
        images_255 = (images * 255).astype(np.uint8)
        inputs = self.processor(images=list(images_255), return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        # normalized features
        features = outputs.last_hidden_state  # (B, N, D)
        num_register_tokens = 4
        features = features[:, 1+num_register_tokens:, :]  # remove CLS and register tokens
        features = torch.nn.functional.normalize(features, dim=-1)
        return features  # (B, N, D)

    def get_features(self, images: np.ndarray):
        self.eval()
        feature_tensors = self.forward(images)  # (B, N, D)
        return feature_tensors  # (B, N, D)


def query_similarity(similarity_maps: torch.Tensor, # (N, HW, HW)
                     xys_norm: torch.Tensor # (N, 2)
                     ):
    query_coords = xys_norm * 2 - 1  # (N, 2) in [-1, 1]
    sim_4d = similarity_maps.unsqueeze(1)  # (N, 1, H, W)
    coords_4d = query_coords.unsqueeze(1).unsqueeze(1)  # (N, 1, 1, 2)

    values = F.grid_sample(
        sim_4d, coords_4d,
        mode='bilinear',
        align_corners=True,
        padding_mode='border'
    )[:, 0, 0, 0]  # (N,)

    return values


def query_feature(feature_map: torch.Tensor,  # (HW, HW, D)
                  xy_norm: torch.Tensor):      # (Np, 2) in [0, 1]
    HW, _, D = feature_map.shape

    # To grid_sample coords [-1, 1]
    query_coords = xy_norm * 2 - 1  # (Np, 2)

    # Reshape for grid_sample: (B, C, H, W) and (B, H_out, W_out, 2)
    feat_4d = feature_map.permute(2, 0, 1).unsqueeze(0)  # (1, D, HW, HW)
    coords_4d = query_coords.unsqueeze(0).unsqueeze(0)   # (1, 1, Np, 2)

    # Interpolate
    sampled_features = F.grid_sample(
        feat_4d, coords_4d,
        mode='bilinear',
        align_corners=True,
        padding_mode='border'
    )  # (1, D, 1, Np)

    # Reshape to (Np, D)
    sampled_features = sampled_features.squeeze(0).squeeze(1).T  # (Np, D)

    return sampled_features


def query_feature_no_interpolation(feature_map: torch.Tensor,  # (HW, HW, D)
                                  xy_norm: torch.Tensor):      # (Np, 2) in [0, 1]
    HW, _, D = feature_map.shape
    Np = xy_norm.shape[0]
    # Convert normalized coords to indices
    xs = (xy_norm[:, 0] * (HW - 1)).long().clamp(0, HW - 1)  # (Np,)
    ys = (xy_norm[:, 1] * (HW - 1)).long().clamp(0, HW - 1)  # (Np,)
    sampled_features = feature_map[ys, xs, :]  # (Np, D)
    return sampled_features


def sample_feature_at_pixel_coords(feature_tensor: torch.Tensor, # (N, D)
                                   xys_image: torch.Tensor, # (Np, 2)
                                   image_size_xy: torch.Tensor, # (2,): W, H
):
    N = feature_tensor.shape[0]
    HW = int(N ** 0.5) # Size of feature_map
    assert HW ** 2 == N,\
        "Feature tensor shape is not a perfect square number of patches."
    D = feature_tensor.shape[1]

    feature_map = feature_tensor.view(HW, HW, D) # (HW, HW, D)
    xy_norm = xys_image / image_size_xy
    xy_norm = xy_norm.clamp(0.0, 1.0)

    sampled_features = query_feature(feature_map, xy_norm) # (Np, D)
    # sampled_features = query_feature_no_interpolation(feature_map, xy_norm) # (Np, D)
    return sampled_features  # (Np, D)


def feature_loss(
    similarity_maps: torch.Tensor, # (Np, HW, HW)
    xys_image: torch.Tensor, # (Np, 2)
    occ_mask: torch.Tensor, # (H, W), in [0, 1]
    weighted_by_max_similarity: bool = False
):
    if xys_image.shape[0] == 0:
        loss = (similarity_maps.sum() + xys_image.sum()) * 0.0
        info = {
            'similarity': similarity_maps.new_empty((0,)),
            'valid_pts_mask': torch.empty((0,), dtype=torch.bool, device=xys_image.device),
        }
        if False:
            info.update(
                max_similarity=similarity_maps.new_empty((0,)),
                max_coords_norm=xys_image.new_empty((0, 2)),
                max_coords_image=xys_image.new_empty((0, 2)),
            )
        return loss, info
    device = similarity_maps.device
    valid_mask = occ_mask < 1e-3 # (H, W), bool
    H, W = valid_mask.shape
    image_size_xy = torch.tensor([W, H], device=device)  # (2,)

    # Filter occluded points
    xs = xys_image[:, 0].long() # (Np,)
    ys = xys_image[:, 1].long() # (Np,)
    xys_in_boundary = (
        (xys_image[:, 0] >= 0) & (xys_image[:, 0] < W)
        & (xys_image[:, 1] >= 0) & (xys_image[:, 1] < H)
    )
    xs = xs.clamp(0, W - 1)
    ys = ys.clamp(0, H - 1)
    xys_mask_value = valid_mask[ys, xs] * xys_in_boundary  # (Np,)

    # Normalize xys to [0, 1]
    xys_norm = xys_image.clone() / image_size_xy # (Np, 2)
    xys_norm = xys_norm.clamp(0.0, 1.0)

    # Query similarity values
    similarity_values = query_similarity(
        similarity_maps, # (Np, HW, HW)
        xys_norm         # (Np, 2)
    ) # (Np,)

    # Compute loss
    target = torch.ones_like(similarity_values)
    diff = (similarity_values - target)

    if weighted_by_max_similarity:
        NP, HW, _ = similarity_maps.shape
        flat_sim_maps = similarity_maps.reshape(NP, -1)  # (Np, HW*HW)
        max_sim_val, _ = torch.max(flat_sim_maps, dim=1)  # (Np,)
        max_sim_val = max_sim_val.detach()  # (Np,)
        weights = torch.clamp((max_sim_val - 0.5) * 2, min=0.0) ** 2
        diff = diff * weights

    diff *= xys_mask_value.float()
    loss = diff.abs().sum() / (xys_mask_value.sum() + 1e-8)

    return loss, {
        'similarity': similarity_values, # (Np,)
        'valid_pts_mask': xys_mask_value, # (Np,)
    }


def analyze_feature_stats(
    dist: torch.Tensor,  # (Np,)
    max_sim_val: torch.Tensor,  # (Np,)
    valid_mask: torch.Tensor = None,  # (Np,)
):
    'Analyze feature stats.'
    import scipy.stats as stats


    if valid_mask is not None:
        valid = valid_mask.bool()
        dist = dist[valid]
        max_sim_val = max_sim_val[valid]

    dist_np = dist.detach().cpu().numpy()
    sim_np = max_sim_val.detach().cpu().numpy()


    slope, intercept, r_value, p_value, std_err = stats.linregress(sim_np, dist_np)


    plt.figure(figsize=(8, 6))
    plt.scatter(sim_np, dist_np, alpha=0.3, s=10, label='Data')
    plt.plot(sim_np, slope * sim_np + intercept, 'r-', linewidth=2,
             label=f'y={slope:.3f}x+{intercept:.3f}\nR²={r_value**2:.3f}')
    plt.xlabel('Max Similarity', fontsize=12)
    plt.ylabel('Distance (normalized)', fontsize=12)
    plt.title(f'Feature Correspondence (n={len(dist_np)})', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"Linear regression: dist = {slope:.4f} * sim + {intercept:.4f}")
    print(f"R² = {r_value**2:.4f}, p-value = {p_value:.2e}")


def feature_loss_global(
    similarity_maps: torch.Tensor, # (Np, HW, HW)
    xys_image: torch.Tensor, # (Np, 2)
    occ_mask: torch.Tensor, # (H, W), in [0, 1]
):
    if xys_image.shape[0] == 0:
        loss = (similarity_maps.sum() + xys_image.sum()) * 0.0
        info = {
            'similarity': similarity_maps.new_empty((0,)),
            'valid_pts_mask': torch.empty((0,), dtype=torch.bool, device=xys_image.device),
        }
        if True:
            info.update(
                max_similarity=similarity_maps.new_empty((0,)),
                max_coords_norm=xys_image.new_empty((0, 2)),
                max_coords_image=xys_image.new_empty((0, 2)),
            )
        return loss, info
    device = similarity_maps.device
    valid_mask = occ_mask < 1e-3 # (H, W), bool
    H, W = valid_mask.shape
    image_size_xy = torch.tensor([W, H], device=device)  # (2,)

    # Filter occluded points
    xs = xys_image[:, 0].long() # (Np,)
    ys = xys_image[:, 1].long() # (Np,)
    xys_in_boundary = (
        (xys_image[:, 0] >= 0) & (xys_image[:, 0] < W)
        & (xys_image[:, 1] >= 0) & (xys_image[:, 1] < H)
    )
    xs = xs.clamp(0, W - 1)
    ys = ys.clamp(0, H - 1)
    xys_mask_value = valid_mask[ys, xs] * xys_in_boundary  # (Np,)

    # Normalize xys to [0, 1]
    xys_norm = xys_image.clone() / image_size_xy # (Np, 2)
    xys_norm = xys_norm.clamp(0.0, 1.0)

    sim_values = query_similarity(
        similarity_maps, # (HW, HW, Np)
        xys_norm         # (Np, 2)
    ) # (Np,)

    NP, HW, _ = similarity_maps.shape
    flat_sim_maps = similarity_maps.reshape(NP, -1)  # (Np, HW*HW)
    max_sim_val, max_indices_flat = torch.max(flat_sim_maps, dim=1)  # (Np,), (Np,)
    max_xs = (max_indices_flat % HW).float()  # (Np,)
    max_ys = (max_indices_flat // HW).float()  # (Np,)
    max_coords_norm = torch.stack([max_xs / (HW - 1), max_ys / (HW - 1)], dim=1)  # (Np, 2)
    max_coords_image = max_coords_norm * image_size_xy  # (Np, 2)

    weights = torch.clamp((max_sim_val - 0.5) * 2, min=0.0) ** 2

    coords_diff = xys_norm - max_coords_norm  # (Np, 2)
    distl1 = coords_diff.norm(dim=1, p=1)  # (Np,)
    distl2 = coords_diff.norm(dim=1, p=2)  # (Np,)

    dist = distl2

    # analyze_feature_stats(distl1, max_sim_val, xys_mask_value)

    dist = dist * xys_mask_value.float()
    dist = dist * weights
    loss = dist.sum() / (xys_mask_value.sum() + 1e-8)

    return loss, {
        'similarity': sim_values,
        'max_similarity': max_sim_val,
        'max_coords_norm': max_coords_norm,
        'max_coords_image': max_coords_image,
        'valid_pts_mask': xys_mask_value,
    }
