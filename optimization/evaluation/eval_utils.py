import torch
import numpy as np
import os
import open3d as o3d

from scipy.spatial import cKDTree
from matplotlib import pyplot as plt
from .metrics import compute_joint3d_error
from .protocols import validate_inputs, valid_mask
from .interaction import hand_similarity, eval_cd_f_right


def calculate_chamfer_f_scores(vertices_source, vertices_target):
    for vertices in (vertices_source, vertices_target):
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
            raise ValueError('Chamfer inputs must be nonempty finite (V,3) point sets')
    vertices_source = vertices_source * 100
    vertices_target = vertices_target * 100

    gen_points_kd_tree = cKDTree(vertices_source)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(vertices_target)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))
    # other direction
    gt_points_kd_tree = cKDTree(vertices_target)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(vertices_source)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))
    chamfer_obj = gt_to_gen_chamfer + gen_to_gt_chamfer
    threshold = 0.5  # 5 mm
    precision_1 = np.mean(one_distances < threshold).astype(np.float32)
    precision_2 = np.mean(two_distances < threshold).astype(np.float32)
    fscore_obj_5 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)

    threshold = 1.0  # 10 mm
    precision_1 = np.mean(one_distances < threshold).astype(np.float32)
    precision_2 = np.mean(two_distances < threshold).astype(np.float32)
    fscore_obj_10 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)
    return chamfer_obj, fscore_obj_5, fscore_obj_10


def eval_cd_icp_first_frame(data_pred, data_gt, metric_dict, visualize=False, metric_dir=None,
                            registration_diagnostics=None):
    validate_inputs(data_pred, data_gt)
    faces = data_pred["faces"]["object"]
    from .hold_tools.icp import compute_icp_metrics

    use_last_frame = False  # HO3D anchors registration at the first valid frame.
    print(f"use_last_frame:{use_last_frame}, full_seq_name: {data_gt['full_seq_name']}")
    valid_ids = np.flatnonzero(valid_mask(data_gt))
    idx = int(valid_ids[-1] if use_last_frame else valid_ids[0])
    v3d_o_ra = o3d.utility.Vector3dVector(data_pred["v3d_ra.object"][idx].numpy())
    faces_o = o3d.utility.Vector3iVector(faces.cpu().numpy())
    v3d_o_ra_gt = o3d.utility.Vector3dVector(data_gt["v3d_ra.object"][idx].numpy())
    faces_o_gt = o3d.utility.Vector3iVector(data_gt["faces.object"].numpy())
    source_mesh = o3d.geometry.TriangleMesh(v3d_o_ra, faces_o)
    target_mesh = o3d.geometry.TriangleMesh(v3d_o_ra_gt, faces_o_gt)

    if visualize:
        os.makedirs(metric_dir, exist_ok=True)
        combined_mesh = source_mesh + target_mesh
        o3d.io.write_triangle_mesh(os.path.join(metric_dir, "combined_mesh.ply"), combined_mesh)
        print(f"Saved combined mesh to: {os.path.join(metric_dir, 'combined_mesh.ply')}")

    best_cd, best_f5, best_f10, details = compute_icp_metrics(
        target_mesh, source_mesh, num_iters=600, return_details=True
    )
    if registration_diagnostics is not None:
        registration_diagnostics.update(details, anchor_index=idx,
            anchor_frame_id=int(data_gt['frame_ids'][idx]) if 'frame_ids' in data_gt else None)
    metric_dict["cd_icp"] = best_cd
    metric_dict["f5_icp"] = best_f5 * 100.0
    metric_dict["f10_icp"] = best_f10 * 100.0
    return metric_dict


def eval_cd_f_right_icp(data_pred, data_gt, metric_dict, visualize=False, metric_dir=None):
    """
    Evaluate chamfer distance using Procrustes alignment on hand joints.

    For each frame:
    1. Perform Procrustes alignment on hand joints (j3d_h_c_pred -> j3d_h_c_gt)
    2. Extract rotation (R), scale (s), and translation (t) from Procrustes result
    3. Apply transformation to object vertices (v3d_o_c_pred)
    4. Compute chamfer distance with v3d_o_c_gt

    Args:
        visualize: If True, generate visualization images. Default False.
        metric_dir: Directory to save metric visualizations. If None, uses default 'enhanced_visuals_icp'
    """

    validate_inputs(data_pred, data_gt)
    v3d_o_c_pred = data_pred["v3d_c.object"]
    v3d_o_c_gt = data_gt["v3d_c.object"]

    j3d_h_c_pred = data_pred["j3d_c.right"]
    j3d_h_c_gt = data_gt["j3d_c.right"]


    v3d_h_c_pred = data_pred["v3d_c.right"]  # (N, 778, 3)
    v3d_h_c_gt = data_gt["v3d_c.right"]      # (N, 778, 3)

    is_valid = data_gt["is_valid"]

    generator = torch.Generator(device='cpu').manual_seed(1)

    cd_list = []
    f5_list = []
    f10_list = []

    # Store data for batch visualization
    # pred_points_for_vis = [] if visualize else None
    pred_points_for_vis = None
    gt_points_for_vis = [] if visualize else None
    aligned_points_for_vis = [] if visualize else None
    frame_indices_for_vis = [] if visualize else None

    if visualize:
        debug_dir = "debug/debug_cd_f_right_icp"
        os.makedirs(debug_dir, exist_ok=True)

    for idx in range(len(v3d_o_c_pred)):
        v3d_o_pred = v3d_o_c_pred[idx]
        v3d_o_gt = v3d_o_c_gt[idx]

        j3d_pred = j3d_h_c_pred[idx]
        j3d_gt = j3d_h_c_gt[idx]

        v3d_h_pred = v3d_h_c_pred[idx]  # (778, 3)
        v3d_h_gt = v3d_h_c_gt[idx]      # (778, 3)

        if not bool(is_valid[idx]):
            # Match HOLD's RNG consumption on invalid GT frames, without
            # fitting a meaningless transform to placeholder hand joints.
            torch.randperm(v3d_o_pred.shape[0], generator=generator)
            torch.randperm(v3d_o_gt.shape[0], generator=generator)
            cd_error = float("nan")
            f5 = float("nan")
            f10 = float("nan")
        else:
            rand_pred_idx = torch.randperm(v3d_o_pred.shape[0], generator=generator)[:3000]
            rand_gt_idx = torch.randperm(v3d_o_gt.shape[0], generator=generator)[:3000]

            pred_o_pts = v3d_o_pred[rand_pred_idx].numpy()
            gt_o_pts = v3d_o_gt[rand_gt_idx].numpy()

            # Perform alignment on hand joints using Procrustes
            # Since hand joints are in one-to-one correspondence, use Procrustes instead of ICP
            j3d_pred_np = j3d_pred.numpy() if hasattr(j3d_pred, 'numpy') else j3d_pred
            j3d_gt_np = j3d_gt.numpy() if hasattr(j3d_gt, 'numpy') else j3d_gt

            try:
                R, s, t = hand_similarity(j3d_pred_np, j3d_gt_np)
            except ValueError as exc:
                frame = int(data_gt['frame_ids'][idx]) if 'frame_ids' in data_gt else idx
                raise ValueError(f'Frame {frame}: {exc}') from exc

            # Verify alignment quality
            j3d_pred_aligned_test = s * (R @ j3d_pred_np.T).T + t
            alignment_error = np.mean(np.linalg.norm(j3d_pred_aligned_test - j3d_gt_np, axis=1))

            # print(f"Frame {idx}: Procrustes scale={s:.4f}, alignment_error={alignment_error*1000:.2f}mm")

            # Apply transformation to object vertices: v_aligned = s * R * v + t
            v3d_pred_full = v3d_o_pred.numpy()
            v3d_pred_aligned = (s * (R @ v3d_pred_full.T).T + t)


            v3d_h_pred_np = v3d_h_pred.numpy() if hasattr(v3d_h_pred, 'numpy') else v3d_h_pred
            v3d_h_pred_aligned = (s * (R @ v3d_h_pred_np.T).T + t)

            v3d_h_gt_np = v3d_h_gt.numpy() if hasattr(v3d_h_gt, 'numpy') else v3d_h_gt  # (778, 3)

            # Apply the same transformation to hand joints for visualization
            j3d_pred_aligned = (s * (R @ j3d_pred_np.T).T + t)

            # Sample aligned points for chamfer distance calculation
            pred_pts_aligned = v3d_pred_aligned[rand_pred_idx]

            # Calculate chamfer distance with Procrustes-aligned points
            cd_error, f5, f10 = calculate_chamfer_f_scores(
                pred_pts_aligned, gt_o_pts
            )


            if visualize and is_valid[idx]:
                v3d_o_gt_np = v3d_o_gt.numpy() if hasattr(v3d_o_gt, 'numpy') else v3d_o_gt


                gt_hand_colors = np.tile([0, 0, 1], (v3d_h_gt_np.shape[0], 1))
                pred_hand_colors = np.tile([1, 0, 0], (v3d_h_pred_aligned.shape[0], 1))
                gt_obj_colors = np.tile([0, 1, 0], (v3d_o_gt_np.shape[0], 1))
                pred_obj_colors = np.tile([1, 1, 0], (v3d_pred_aligned.shape[0], 1))


                combined_points = np.vstack([
                    v3d_h_gt_np,
                    v3d_h_pred_aligned,
                    v3d_o_gt_np,
                    v3d_pred_aligned
                ])

                combined_colors = np.vstack([
                    gt_hand_colors,
                    pred_hand_colors,
                    gt_obj_colors,
                    pred_obj_colors
                ])


                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(combined_points)
                pcd.colors = o3d.utility.Vector3dVector(combined_colors)


                ply_path = os.path.join(metric_dir, f"combined_4colors_{idx:04d}_{cd_error:.4f}.ply")
                o3d.io.write_point_cloud(ply_path, pcd)
                # print(f"Saved 4-color point cloud to: {ply_path}")
                # print(f"  - GT hand (blue): {v3d_h_gt_np.shape[0]} points")
                # print(f"  - Pred hand (red): {v3d_h_pred_aligned.shape[0]} points")
                # print(f"  - GT object (green): {v3d_o_gt_np.shape[0]} points")
                # print(f"  - Pred object (yellow): {v3d_pred_aligned.shape[0]} points")


        cd_list.append(cd_error)
        f5_list.append(f5)
        f10_list.append(f10)

    cd_list = np.array(cd_list)
    f5_list = np.array(f5_list)
    f10_list = np.array(f10_list)

    not_valid = (1 - is_valid).numpy().astype(bool)
    cd_list[not_valid] = np.nan
    f5_list[not_valid] = np.nan
    f10_list[not_valid] = np.nan

    metric_dict["cd_right_icp"] = cd_list
    metric_dict["f5_right_icp"] = f5_list * 100.0
    metric_dict["f10_right_icp"] = f10_list * 100.0
    metric_dict['cd_interaction_hand_sim3_per_frame'] = cd_list
    metric_dict['f5_interaction_hand_sim3_per_frame'] = f5_list * 100.0
    metric_dict['f10_interaction_hand_sim3_per_frame'] = f10_list * 100.0

    # Create summary analysis for selected frames (only if visualize=True)
    if visualize and pred_points_for_vis:
        center_distances_before = []
        center_distances_after = []
        scale_ratios_all = []

        not_valid = (1 - is_valid).numpy().astype(bool)

        for frame_idx, pred_o_pts, aligned_pts, gt_o_pts in zip(
            frame_indices_for_vis, pred_points_for_vis, aligned_points_for_vis, gt_points_for_vis
        ):
            if not_valid[frame_idx]:
                center_distances_before.append(np.nan)
                center_distances_after.append(np.nan)
                scale_ratios_all.append(np.nan)
                continue

            # Collect statistics for summary (before alignment)
            pred_center = np.mean(pred_o_pts, axis=0)
            gt_center = np.mean(gt_o_pts, axis=0)
            center_distance_before = np.linalg.norm(pred_center - gt_center)
            center_distances_before.append(center_distance_before)

            # After alignment
            aligned_center = np.mean(aligned_pts, axis=0)
            center_distance_after = np.linalg.norm(aligned_center - gt_center)
            center_distances_after.append(center_distance_after)

            pred_size = np.max(pred_o_pts, axis=0) - np.min(pred_o_pts, axis=0)
            gt_size = np.max(gt_o_pts, axis=0) - np.min(gt_o_pts, axis=0)
            scale_ratios = pred_size / (gt_size + 1e-8)
            scale_ratios_all.append(np.mean(scale_ratios))

        # Create summary plot - now with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        # Plot 1: Center Distance Over Time (Before vs After)
        axes[0, 0].plot(frame_indices_for_vis, center_distances_before, 'r-o',
                       linewidth=2, markersize=6, label='Before ICP')
        axes[0, 0].plot(frame_indices_for_vis, center_distances_after, 'b-s',
                       linewidth=2, markersize=6, label='After ICP')
        axes[0, 0].set_xlabel('Frame Index')
        axes[0, 0].set_ylabel('Center Distance (m)')
        axes[0, 0].set_title('Center Distance Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Scale Ratio Over Time
        axes[0, 1].plot(frame_indices_for_vis, scale_ratios_all, 'r-s', linewidth=2, markersize=6)
        axes[0, 1].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Perfect Scale Match')
        axes[0, 1].set_xlabel('Frame Index')
        axes[0, 1].set_ylabel('Mean Scale Ratio (Pred/GT)')
        axes[0, 1].set_title('Scale Ratio Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Chamfer Distance (ICP-Aligned) Over Time
        axes[1, 0].plot(frame_indices_for_vis, cd_list, 'g-^', linewidth=2, markersize=6)
        axes[1, 0].set_xlabel('Frame Index')
        axes[1, 0].set_ylabel('Chamfer Distance (cm²)')
        axes[1, 0].set_title('Chamfer Distance (ICP-Aligned) Over Time')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: F-Score Over Time
        axes[1, 1].plot(frame_indices_for_vis, f5_list, 'm-D', linewidth=2, markersize=6, label='F5')
        axes[1, 1].plot(frame_indices_for_vis, f10_list, 'c-x', linewidth=2, markersize=6, label='F10')
        axes[1, 1].set_xlabel('Frame Index')
        axes[1, 1].set_ylabel('F-Score (%)')
        axes[1, 1].set_title('F-Score Over Time')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Use metric_dir for summary path as well
        vis_dir = metric_dir if metric_dir is not None else 'enhanced_visuals_icp'
        summary_path = f'{vis_dir}/summary_analysis_icp.png'
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"Summary analysis saved to: {summary_path}")
        print(f"Average center distance (before ICP): {np.nanmean(center_distances_before):.6f} m")
        print(f"Average center distance (after ICP): {np.nanmean(center_distances_after):.6f} m")
        print(f"Average scale ratio: {np.nanmean(scale_ratios_all):.4f}")
        print(f"Average CD (ICP-aligned): {np.nanmean(cd_list):.4f} cm²")
        print(f"Average F5: {np.nanmean(f5_list):.2f}%")
        print(f"Average F10: {np.nanmean(f10_list):.2f}%")

    return metric_dict


def eval_mpjpe_right(data_pred, data_gt, metric_dict, visualize=False, metric_dir=None):
    validate_inputs(data_pred, data_gt, required_keys=('j3d_ra.right',))
    # Hand joints
    j3d_h_c_pred_original = data_pred['ori_j3d_c.right']
    j3d_h_c_gt_original = data_gt['j3d_c.right']
    j3d_h_c_pred_ra = data_pred["j3d_ra.right"]
    j3d_h_c_gt_ra = data_gt["j3d_ra.right"]
    is_valid = data_gt["is_valid"]

    # The WiLoR/MANO wrapper and the manopth GT loader both return the same
    # canonical 21-joint order. Reapplying MANO-to-OpenPose here permutes fingers
    # that are already ordered, so even identical predictions have nonzero error.

    mpjpe_ra_r = compute_joint3d_error(j3d_h_c_gt_ra, j3d_h_c_pred_ra, is_valid)
    mpjpe_ra_r = mpjpe_ra_r.mean(dim=1) * 1000  # Use dim instead of axis for PyTorch

    # mpjpe_r = compute_joint3d_error(j3d_h_c_gt_original, j3d_h_c_pred_ra, is_valid)
    # mpjpe_r = mpjpe_r.mean(dim=1) * 1000

    if visualize and False:
        # max_mpjpe_idx = np.nanargmax(mpjpe_ra_r.detach().cpu().numpy())
        # max_mpjpe_idx = torch.argmax(mpjpe_ra_r).item()
        v3d_h_pred = data_pred['v3d_c.right']
        v3d_h_gt = data_gt["v3d_c.right"]
        v3d_h_pred_ra = data_pred['v3d_ra.right']
        v3d_h_gt_ra = data_gt["v3d_ra.right"]


        os.makedirs("debug", exist_ok=True)

        def save_pcd_pair(pred, gt, path):
            num_verts = len(pred)
            hue = np.linspace(0, 1, num_verts)
            colors = plt.cm.rainbow(hue)[:, :3]


            pcd_pred = o3d.geometry.PointCloud()
            pcd_pred.points = o3d.utility.Vector3dVector(pred)
            pcd_pred.colors = o3d.utility.Vector3dVector(colors)

            pcd_gt = o3d.geometry.PointCloud()
            pcd_gt.points = o3d.utility.Vector3dVector(gt)
            pcd_gt.colors = o3d.utility.Vector3dVector(colors)

            pcd = pcd_pred + pcd_gt

            os.makedirs(os.path.dirname(path), exist_ok=True)
            o3d.io.write_point_cloud(path, pcd)
            print(f"Saved rainbow point cloud to: {path}")


        for idx in range(len(v3d_h_pred)):

            # idx = max_mpjpe_idx
            pred_verts = v3d_h_pred[idx].numpy()  # (778, 3)
            gt_verts = v3d_h_gt[idx].numpy()      # (778, 3)

            ply_path = f"debug/debug_mpjpe/{idx:04d}.ply"
            save_pcd_pair(pred_verts, gt_verts, ply_path)

            pred_verts_ra = v3d_h_pred_ra[idx].numpy()  # (778, 3)
            gt_verts_ra = v3d_h_gt_ra[idx].numpy()      # (778, 3)

            ply_path = f"debug/debug_mpjpe/{idx:04d}_ra.ply"
            save_pcd_pair(pred_verts_ra, gt_verts_ra, ply_path)

    metric_dict["mpjpe_ra_r"] = mpjpe_ra_r
    # metric_dict["mpjpe_r"] = mpjpe_r
    return metric_dict


def eval_mpjpe_right_scaled(data_pred, data_gt, metric_dict, visualize=False, metric_dir=None):
    """Auxiliary wrist-relative MPJPE with one positive scale for the sequence.

    It fits neither per-frame scales nor rotations and is not PA-MPJPE.
    """

    validate_inputs(data_pred, data_gt, required_keys=('j3d_ra.right',))
    # Hand joints
    j3d_h_c_pred_ra = data_pred["j3d_ra.right"]
    j3d_h_c_gt_ra = data_gt["j3d_ra.right"]
    is_valid = data_gt["is_valid"]

    # Collect all valid frames for global scale computation
    all_pred = []
    all_gt = []

    for idx in range(len(j3d_h_c_pred_ra)):
        if is_valid[idx] and not torch.isnan(j3d_h_c_pred_ra[idx].mean()):
            all_pred.append(j3d_h_c_pred_ra[idx])
            all_gt.append(j3d_h_c_gt_ra[idx])

    # Stack all data
    all_pred = torch.stack(all_pred)  # (N_frames, num_joints, 3)
    all_gt = torch.stack(all_gt)      # (N_frames, num_joints, 3)

    # Flatten to compute global scale
    all_pred_flat = all_pred.reshape(-1, 3)  # (N_total_joints, 3)
    all_gt_flat = all_gt.reshape(-1, 3)      # (N_total_joints, 3)

    # Compute global scale: s = (X^T * Y) / (X^T * X)
    numerator = torch.sum(all_pred_flat * all_gt_flat)
    denominator = torch.sum(all_pred_flat * all_pred_flat)
    if denominator.item() <= 1e-12:
        raise ValueError('Degenerate hand joints cannot define a sequence scale')
    global_scale = (numerator / (denominator + 1e-8)).item()
    if not np.isfinite(global_scale) or global_scale <= 0:
        raise ValueError('Auxiliary sequence scale must be finite and positive')

    print(f"[eval_mpjpe_right_scaled] Global scale factor: {global_scale:.4f}")

    # Apply global scale to all predictions
    j3d_h_c_pred_ra_scaled = j3d_h_c_pred_ra * global_scale

    # Use compute_joint3d_error to calculate MPJPE (same as eval_mpjpe_right)
    mpjpe_ra_r_scaled = compute_joint3d_error(j3d_h_c_gt_ra, j3d_h_c_pred_ra_scaled, is_valid)
    mpjpe_ra_r_scaled = mpjpe_ra_r_scaled.mean(dim=1) * 1000  # Convert to mm


    metric_dict["mpjpe_right_scaled"] = mpjpe_ra_r_scaled


    print(f"Average MPJPE (global scale): {np.nanmean(mpjpe_ra_r_scaled):.2f}mm")

    return metric_dict
