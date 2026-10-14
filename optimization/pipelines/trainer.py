"""Slim Runner class that orchestrates HOI mesh optimization.

Delegates per-frame optimisation to :class:`optim.ObjectOptimizer` and
:class:`optim.HandOptimizer`, keeping the training loop, data loading,
result saving and visualisation logic here.

This is a *structural* refactor only -- no algorithmic changes.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import shutil
import time
import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation
import yaml
from torch.utils.tensorboard import SummaryWriter

from optimization.config.config import Config
from optimization.data.loader import Parser, Dataset
from optimization.data.megasam import (
    Parser as OldParser,
    Dataset as OldDataset,
    CombinedDataset,
    reverse_parser_time_arrays,
)
from optimization.models.mesh_objects import MeshObject, HandMesh
from optimization.models.mesh_renderer import create_scene_renderer
from optimization.models.mano_wrapper import MANO
from optimization.utils.io import set_random_seed, PrintColors as pc
from optimization.utils.geometry import (
    rotation_vector_to_matrix_safe,
    rotation_matrix_to_rvec,
    so3_log_batch,
    so3_exp_batch,
    sample_2d_3d_pts_from_depth,
)
from optimization.utils.image import erode_depth, compute_edge_sobel, alpha2pts
from optimization.utils.visualization import overlay_rgb, make_vis_image
from optimization.utils.camera_utils import convert_camtoworld_to_pytorch3d_cameras
from optimization.utils.feature import DINOWrapper, sample_feature_at_pixel_coords
from optimization.utils.sdf import compute_sdf, query_sdf_differentiable
from optimization.losses.contact import zbuf_contact_loss_simplified

from optimization.optim.object_optimizer import ObjectOptimizer
from optimization.optim.hand_optimizer import HandOptimizer

import open3d as o3d


class Runner:
    """Engine for training and testing with PyTorch3D mesh rendering.

    This is a slim orchestrator: heavy-lifting per-frame optimisation is
    delegated to :class:`ObjectOptimizer` and :class:`HandOptimizer`.
    """

    def __init__(self, cfg: Config) -> None:
        set_random_seed(42)

        self.cfg = cfg
        self.device = "cuda"

        if cfg.test_first_frame:
            if cfg.reverse_frame_order:
                cfg.start_frame = cfg.end_frame - 1
            else:
                cfg.end_frame = cfg.start_frame + 1

        result_dir = cfg.get_result_dir()
        os.makedirs(result_dir, exist_ok=True)

        if cfg.use_tensorboard:
            self.tb_dir = cfg.get_result_dir() + "/tensorboard"
            if os.path.exists(self.tb_dir):
                shutil.rmtree(self.tb_dir)
            os.makedirs(self.tb_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(self.tb_dir)
        else:
            self.tb_writer = None

        cfg.save_config()

        self.render_video_dir = f"{result_dir}/videos"
        os.makedirs(self.render_video_dir, exist_ok=True)
        self.pcd_vis_dir = f"{result_dir}/pcd_vis"
        os.makedirs(self.pcd_vis_dir, exist_ok=True)

        # ---- Data ----
        self.parser, self.trainset = self._build_parser_and_dataset(
            cfg.data_dir,
            start_frame=cfg.start_frame,
            end_frame=cfg.end_frame,
        )
        # Validate the immutable display asset before starting any optimization.
        from optimization.data.object_assets import read_object_asset
        asset_manifest = cfg.object_asset_manifest
        automatic_asset = Path(cfg.data_dir) / 'object_asset' / 'asset.json'
        if asset_manifest is None and automatic_asset.is_file():
            asset_manifest = automatic_asset
        if asset_manifest:
            read_object_asset(asset_manifest, self.parser.object_mesh)

        # ---- MANO ----
        mano_config_path = Path(cfg.mano_model_path) / "mano_config.yaml"
        assert mano_config_path.exists(), f"MANO config not found at {mano_config_path}"
        with open(mano_config_path, "r") as f:
            mano_cfg = yaml.safe_load(f)
        mano_cfg = {k.lower(): v for k, v in mano_cfg["MANO"].items()}
        self.mano_model = MANO(**mano_cfg).to(self.device)
        print(f"Initialized MANO model from {cfg.mano_model_path}")

        # ---- Renderer ----
        self.scene_renderer = create_scene_renderer(
            self.parser, cfg=cfg,
            mano_model=self.mano_model,
            device=self.device,
        )

        # Appearance optimiser
        self.appearance_optimizer = None
        params_to_optimize = []
        if cfg.optimize_vertex_colors:
            for name, param in self.scene_renderer.named_parameters():
                if 'vertex_colors' in name:
                    params_to_optimize.append({'params': param, 'lr': cfg.vertex_color_lr})
        if cfg.optimize_object_texture:
            for name, param in self.scene_renderer.named_parameters():
                if 'texture' in name:
                    params_to_optimize.append({'params': param, 'lr': cfg.texture_lr})
        if params_to_optimize:
            self.appearance_optimizer = torch.optim.Adam(params_to_optimize, eps=1e-15, betas=(0.9, 0.999))

        total_vertices = self.scene_renderer.num_total_vertices
        print(f"Model initialized. Total vertices: {total_vertices}")

        # ---- Scales ----
        if cfg.optimize_hand_scale:
            self.optimized_hand_scale = torch.nn.Parameter(
                torch.tensor(cfg.initial_hand_scale, device=self.device, dtype=torch.float32)
            )
        else:
            self.optimized_hand_scale = None

        if cfg.object_scale_dim == 1:
            scale_tensor = torch.ones(1) * cfg.initial_object_scale
        elif cfg.object_scale_dim == 3:
            scale_tensor = torch.ones(3) * cfg.initial_object_scale
        else:
            raise ValueError(f"Invalid object_scale_dim: {cfg.object_scale_dim}")
        self.optimized_obj_scale = torch.nn.Parameter(
            torch.tensor(scale_tensor, device=self.device, dtype=torch.float32)
        )

        if cfg.use_finger_mask:
            finger_mask_path = f"{cfg.mano_model_path}/finger_vertices_mask.npz"
            finger_mask = np.load(finger_mask_path)['finger_vertices_mask']
            self.finger_mask = torch.tensor(finger_mask, device=self.device, dtype=torch.float32).detach()

        if cfg.object_feature_loss:
            self.dino_model = DINOWrapper(input_size=cfg.object_feature_input_size)

        self.in_contact = False

        # ---- Per-frame optimizers ----
        self.object_optimizer = ObjectOptimizer(cfg, self)
        self.hand_optimizer = HandOptimizer(cfg, self)

    def _build_parser_and_dataset(
        self,
        data_dir: str,
        start_frame: int,
        end_frame: int,
    ):
        cfg = self.cfg
        if os.path.isfile(os.path.join(data_dir, 'camera_data.npz')):
            parser = OldParser(
                data_dir=data_dir,
                online_start_frame=start_frame,
                online_end_frame=end_frame,
                hold_mode=cfg.hold_mode,
                hold_mode_interval=cfg.hold_mode_interval,
                mano_transforms_path=cfg.mano_transforms_path,
            )
            dataset = OldDataset(parser)
        else:
            parser = Parser(
                data_dir=data_dir,
                mano_model_path=cfg.mano_model_path,
                start_frame=start_frame,
                end_frame=end_frame,
                frame_stride=cfg.hold_mode_interval if cfg.hold_mode else 1,
                mano_transforms_path=cfg.mano_transforms_path,
            )
            dataset = Dataset(parser)
        return parser, dataset

    # ==================================================================
    # Methods that the optimizers call back into (unchanged from original)
    # ==================================================================

    def set_hand_base_pose(self, frame_id, R, t, compute_mano, hand_pose_vecs=None):
        hand_mesh: HandMesh = self.scene_renderer.get_mesh_object("hand")
        frame_data = self.trainset[frame_id]
        mano_betas = frame_data['mano_betas'].to(self.device)
        if hand_pose_vecs is None:
            hand_pose = frame_data['mano_hand_pose'].to(self.device)
        else:
            hand_pose_matrices = so3_exp_batch(hand_pose_vecs)
            hand_pose = hand_pose_matrices.unsqueeze(0)
        global_orient = frame_data['mano_global_orient'].to(self.device)

        if self.optimized_hand_scale is not None:
            transform_scale = self.optimized_hand_scale
        else:
            transform_scale = frame_data['mano_transform_scale'].to(self.device)

        transform_R = torch.eye(3, device=self.device)
        transform_t = frame_data['mano_transform_translation'].to(self.device)
        if transform_t.isnan().any():
            for i in range(frame_id, len(self.trainset)):
                frame_data_i = self.trainset[i]
                if not frame_data_i['mano_transform_translation'].isnan().any():
                    transform_t = frame_data_i['mano_transform_translation'].to(self.device)
                    break
        if R is not None and t is not None:
            hand_full_R = R
            hand_full_t = t
        else:
            hand_full_R = None
            hand_full_t = None

        R, t = hand_mesh.compute_one_frame_vertices(
            mano_model=self.mano_model,
            betas=mano_betas,
            hand_pose=hand_pose,
            global_orient=global_orient,
            mano_transform_R=transform_R,
            mano_transform_t=transform_t,
            transform_scale=transform_scale,
            full_R=hand_full_R,
            full_t=hand_full_t,
            compute_mano=compute_mano,
        )
        return R, t

    def set_obj_base_pose(self, frame_id, R, t):
        obj_mesh: MeshObject = self.scene_renderer.get_mesh_object("object")
        if R is None or t is None:
            object_pose_matrix = self.trainset[frame_id]['object_pose']
            R = object_pose_matrix[:3, :3].to(self.device)
            t = object_pose_matrix[:3, 3].to(self.device)
        if self.cfg.optimize_object_scale or self.cfg.optimize_object_scale_not_aligning:
            obj_mesh.apply_transform(R, t, scale=self.optimized_obj_scale)
        else:
            obj_mesh.apply_transform(R, t)
        return R, t

    def render_meshes(self, camtoworlds, Ks, width, height, frame_id,
                      random_bkgd=False, render_objects=None, require_normal=False,
                      hard_rasterization=False):
        cameras = convert_camtoworld_to_pytorch3d_cameras(
            camtoworlds, Ks, height, width, self.device
        )
        render_output = self.scene_renderer.render(
            cameras, height, width, render_objects,
            require_normal=require_normal,
            force_hard_rasterization=hard_rasterization,
        )

        colors = render_output['colors']
        alpha = render_output['alpha']
        depth = render_output['depth']
        zbuf = render_output['zbuf']
        normals = render_output.get('normals', None)
        vertices_visibility = render_output.get('vertices_visibility', {})
        face_visibility = render_output.get('face_visibility', None)
        pix_to_face = render_output.get('pix_to_face', None)

        if random_bkgd:
            bkgd_color = torch.rand(1, 3, device=self.device)
            colors = colors + bkgd_color * (1 - alpha)

        result = {
            'colors': colors,
            'alpha': alpha,
            'depth': depth,
            'normals': normals,
            'hand_points': None,
            'object_points': None,
            'zbuf': zbuf,
            'pix_to_face': pix_to_face,
            'face_visibility': face_visibility,
            'vertices_visibility': vertices_visibility,
        }
        return result

    def save_ho_pcd(self, frame_id):
        actual_frame_id = self.get_actual_frame_id(frame_id)
        hand_mesh: HandMesh = self.scene_renderer.get_mesh_object("hand")
        hand_verts = hand_mesh.get_transformed_vertices()
        object_mesh: MeshObject = self.scene_renderer.get_mesh_object("object")
        object_verts = object_mesh.get_transformed_vertices()

        hand_colors = torch.ones_like(hand_verts) * torch.tensor([1.0, 0, 0], device=self.device)
        object_colors = torch.ones_like(object_verts) * torch.tensor([0, 1.0, 0], device=self.device)
        all_verts = torch.cat([hand_verts, object_verts], dim=0).detach().cpu().numpy()
        all_colors = torch.cat([hand_colors, object_colors], dim=0).detach().cpu().numpy()
        all_verts[:, 1:] *= -1
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_verts)
        pcd.colors = o3d.utility.Vector3dVector(all_colors)
        save_path = f"{self.pcd_vis_dir}/frame_{actual_frame_id:04d}_ho_pcd.ply"
        o3d.io.write_point_cloud(save_path, pcd)
        return save_path

    def get_actual_frame_id(self, dataset_idx: int) -> int:
        frame_id = self.trainset[dataset_idx].get('frame_id', dataset_idx)
        if torch.is_tensor(frame_id):
            frame_id = frame_id.item()
        return int(frame_id)

    def create_scheduler(self, optimizer, train_max_iter: int, scheduler_type: str = None):
        if scheduler_type is None:
            scheduler_type = self.cfg.online_scheduler_type
        if scheduler_type == 'constant':
            return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=train_max_iter)
        elif scheduler_type == 'sequential':
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[
                    torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=train_max_iter // 4 * 3),
                    torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01,
                                                       total_iters=train_max_iter - train_max_iter // 4 * 3),
                ],
                milestones=[train_max_iter // 4 * 3],
            )
        elif scheduler_type == 'linear':
            return torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1,
                                                      total_iters=train_max_iter)
        elif scheduler_type == 'exponential':
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.01 ** (1.0 / train_max_iter))
        elif scheduler_type == 'sequential2':
            def lr_lambda(epoch):
                return 1.0 if epoch < 200 else 5.0
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        else:
            raise NotImplementedError(f"Scheduler {scheduler_type} not implemented.")

    def sample_points_feature(
        self,
        gt_feature_tensor,
        rendered_color,
        rendered_depth,
        K,
        R_w2c,
        t_w2c,
        pix_to_face,
        images_gt=None,
    ):
        cfg = self.cfg
        depth = rendered_depth[0, ..., 0]
        H, W = depth.shape

        p2d, p3d_cam, indices_flat = sample_2d_3d_pts_from_depth(
            depth, K, max_num_pts=cfg.num_sampled_feature_points
        )

        p3d_canonical = p3d_cam - t_w2c.unsqueeze(0)
        p3d_canonical = p3d_canonical @ R_w2c
        if cfg.optimize_object_scale:
            p3d_canonical = p3d_canonical / self.optimized_obj_scale

        pix_to_face_flat = pix_to_face.view(-1)
        pts_to_face = pix_to_face_flat[indices_flat]

        image_size_xy = torch.tensor([W, H], device=self.device)
        input_image = rendered_color.detach().cpu().numpy()
        rendered_feature_tensor = self.dino_model(input_image).to(self.device)[0]

        features = sample_feature_at_pixel_coords(
            feature_tensor=rendered_feature_tensor,
            xys_image=p2d,
            image_size_xy=image_size_xy,
        )

        Nf = gt_feature_tensor.shape[1]
        HWf = int(Nf ** 0.5)

        gt_feature_tensor_normalized = F.normalize(gt_feature_tensor[0], dim=-1)
        pt_feature_tensor_normalized = F.normalize(features, dim=-1)

        similarity_tensors = torch.matmul(
            pt_feature_tensor_normalized,
            gt_feature_tensor_normalized.T,
        )
        similarity_maps = similarity_tensors.view(-1, HWf, HWf)

        self.feature_pts_canonical = p3d_canonical.detach()
        self.feature_pts_similarity_maps = similarity_maps.detach()
        self.feature_pts_to_face = pts_to_face.detach()

    def prepare_frame_data(self, frame_id: int, data_type: str):
        assert data_type in ['hand', 'object'], "data_type must be 'hand' or 'object'"
        cfg = self.cfg
        device = self.device
        data = self.trainset[frame_id]
        Ks = data['K'].unsqueeze(0).to(device)
        images_gt = data['image'].unsqueeze(0).to(device)
        depths_gt = data['depth'].unsqueeze(0).unsqueeze(-1).to(device)
        hand_mask = data.get("hand_mask", None).to(device)
        object_mask = data.get("object_mask", None).to(device)
        mano_mask = data.get("mano_mask", None).to(device)
        H, W = images_gt.shape[1:3]

        if cfg.erode_depth:
            depths_gt_eroded = erode_depth(
                depths_gt[0, ..., 0],
                radius=2,
                depth_diff_thres=0.01,
                ratio_thres=0.8,
                zfar=100.0,
                device=device,
            ).unsqueeze(0).unsqueeze(-1)
        else:
            depths_gt_eroded = depths_gt

        depths_masked = depths_gt_eroded[0, ..., 0] * object_mask.float()
        depths_masked = depths_masked[None, ..., None]

        if 'normal' in data:
            normal_gt = data['normal'].to(device)
            normals_gt = normal_gt.unsqueeze(0)
        else:
            normals_gt = None

        assert 'mano_hand_pose' in data
        mano_hand_pose = data['mano_hand_pose'].to(device)

        assert 'mano_joints_2d' in data
        mano_joints_2d = data['mano_joints_2d'].to(device)
        object_edge = compute_edge_sobel(object_mask.float())

        return {
            'Ks': Ks,
            'images_gt': images_gt,
            'depths_gt': depths_gt_eroded,
            'depths_masked': depths_masked,
            'hand_mask': hand_mask,
            'object_mask': object_mask,
            'object_edge': object_edge,
            'mano_mask': mano_mask,
            'normals_gt': normals_gt,
            'H': H,
            'W': W,
            'hand_pose': mano_hand_pose,
            'mano_joints_2d': mano_joints_2d
        }

    def precompute_hand_sdf(self):
        cfg = self.cfg
        mesh = self.scene_renderer.get_mesh_object("hand")
        vertices = mesh.get_transformed_vertices()
        faces = mesh.faces

        sdf_grid, norm_trans, scale = compute_sdf(
            vertices, faces,
            resolution=cfg.sdf_resolution,
            dist_th_wld=cfg.sdf_contact_threshold,
        )
        self.hand_sdf = sdf_grid.detach()
        self.hand_sdf_norm_trans = norm_trans.detach()
        self.hand_sdf_scale = scale.detach()

    def precompute_object_sdf(self, canonical: bool = False, contact_th: float = None):
        cfg = self.cfg
        mesh = self.scene_renderer.get_mesh_object("object")
        if canonical:
            vertices = mesh.canonical_vertices
            if cfg.optimize_object_scale or cfg.optimize_object_scale_not_aligning:
                vertices = vertices * self.optimized_obj_scale
        else:
            vertices = mesh.get_transformed_vertices()
        faces = mesh.faces

        contact_threshold = contact_th if contact_th is not None else cfg.sdf_contact_threshold
        sdf_grid, norm_trans, scale = compute_sdf(
            vertices, faces,
            resolution=cfg.sdf_resolution,
            dist_th_wld=contact_threshold,
        )
        self.object_sdf = sdf_grid.detach()
        self.object_sdf_norm_trans = norm_trans.detach()
        self.object_sdf_scale = scale.detach()

    def compute_last_hands_in_ob(self, frame_id):
        last_hand_verts_incam = self.last_hand_verts_incam
        last_frame_id = frame_id + 1 if self.cfg.reverse_frame_order else frame_id - 1
        last_R = self.optimized_obj_R[last_frame_id]
        last_t = self.optimized_obj_t[last_frame_id]
        last_hand_verts_in_ob = (last_hand_verts_incam - last_t.unsqueeze(0)) @ last_R
        self.last_hand_verts_in_ob = last_hand_verts_in_ob.detach()
        assert hasattr(self, 'object_sdf'), "Object SDF must be precomputed before computing hand pose loss."
        last_frame_verts_sdf_value, _ = query_sdf_differentiable(
            self.object_sdf,
            last_hand_verts_in_ob,
            self.object_sdf_norm_trans,
            self.object_sdf_scale,
            get_vis_pcd=False,
        )
        hand_dist_wld = last_frame_verts_sdf_value.clamp(min=0.0) * self.object_sdf_scale
        scale = self.cfg.ho_weight_tanh_scale
        verts_dist_weights = 1.0 - torch.tanh(hand_dist_wld * scale)
        if self.cfg.use_finger_mask:
            verts_dist_weights = verts_dist_weights * self.finger_mask
        self.hand_verts_dist_weights = verts_dist_weights.detach()

    def init_hand_pose_pnp(
        self, frame_id: int,
        K: torch.Tensor,
        mano_joints_2d: torch.Tensor,
    ):
        device = self.device
        actual_frame_id = self.get_actual_frame_id(frame_id)

        hand_mesh: HandMesh = self.scene_renderer.get_mesh_object("hand")
        mano_joints_3d = hand_mesh.canonical_joints
        hand_scale = (self.optimized_hand_scale if self.optimized_hand_scale is not None
                      else self.trainset[frame_id]['mano_transform_scale'].to(device))
        # PnP translation must share the units of the rendered, scaled skeleton.
        mano_joints_3d = mano_joints_3d * hand_scale

        joints_2d = mano_joints_2d.detach().cpu().numpy()
        joints_3d = mano_joints_3d.detach().cpu().numpy()
        K_np = K.detach().cpu().numpy()

        success, rvec, tvec = cv2.solvePnP(
            objectPoints=joints_3d,
            imagePoints=joints_2d,
            cameraMatrix=K_np,
            distCoeffs=None,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not success:
            print(f"Frame {actual_frame_id}: PnP failed.")
            return None, None

        rvec = torch.from_numpy(rvec.flatten()).to(device).float()
        tvec = torch.from_numpy(tvec.flatten()).to(device).float()

        R_pnp = rotation_vector_to_matrix_safe(rvec)
        R = R_pnp
        t = tvec

        base_R, base_t = self.set_hand_base_pose(frame_id, R, t, compute_mano=False)
        return base_R, base_t

    def align_object_to_hand(
        self, frame_id: int,
        object_R: torch.Tensor,
        object_t: torch.Tensor,
    ):
        cfg = self.cfg
        device = self.device
        actual_frame_id = self.get_actual_frame_id(frame_id)

        prepared_data = self.prepare_frame_data(frame_id, data_type='hand')
        Ks = prepared_data['Ks']
        images_gt = prepared_data['images_gt']
        depths_gt = prepared_data['depths_gt']
        normals_gt = prepared_data['normals_gt']
        depths_masked = prepared_data['depths_masked']
        hand_mask = prepared_data['hand_mask']
        object_mask = prepared_data['object_mask']
        H = prepared_data['H']
        W = prepared_data['W']

        self.set_obj_base_pose(frame_id, object_R, object_t)

        max_iter = cfg.align_ho_max_iter
        tvec_z = torch.nn.Parameter(torch.zeros(1, device=device))
        trans_optimizer = torch.optim.Adam([{'params': tvec_z, 'lr': cfg.object_trans_lr}])
        trans_scheduler = self.create_scheduler(
            trans_optimizer, max_iter, scheduler_type=cfg.object_scheduler_type
        )

        if cfg.object_scale_dim == 1:
            scale_optimizer = torch.optim.Adam([
                {'params': self.optimized_obj_scale, 'lr': cfg.object_scale_lr}
            ])
            target_scale = None
        else:
            scale_1d = torch.nn.Parameter(torch.ones(1, device=device))
            scale_optimizer = torch.optim.Adam([
                {'params': scale_1d, 'lr': cfg.object_scale_lr}
            ])
            optimized_scale_original = self.optimized_obj_scale.detach().clone()
            target_scale = optimized_scale_original
        scale_scheduler = self.create_scheduler(
            scale_optimizer, max_iter, scheduler_type=cfg.object_scheduler_type
        )

        optimizers = [trans_optimizer, scale_optimizer]
        schedulers = [trans_scheduler, scale_scheduler]

        vis_images_list = []
        pbar = tqdm.trange(max_iter, position=1, leave=False, disable=not cfg.show_opt_progress)
        for i in pbar:
            curr_t = object_t.clone()
            trans_dir = curr_t / curr_t.norm()
            t_offset = trans_dir * tvec_z
            curr_t = object_t + t_offset
            if cfg.object_scale_dim == 1:
                self.set_obj_base_pose(frame_id, object_R, curr_t)
            else:
                self.optimized_obj_scale = optimized_scale_original * scale_1d
                self.set_obj_base_pose(frame_id, object_R, curr_t)
            pytorch3d_c2w = torch.eye(4, device=device).unsqueeze(0)
            render_output = self.render_meshes(
                camtoworlds=pytorch3d_c2w,
                Ks=Ks, height=H, width=W,
                frame_id=frame_id,
                random_bkgd=cfg.random_bkgd,
                render_objects=['object'],
                require_normal=False,
                hard_rasterization=True,
            )

            rendered_color = render_output['colors']
            rendered_alpha = render_output['alpha']
            rendered_depth = render_output['depth']
            rendered_normal = render_output['normals']
            obj_zbuf = render_output['zbuf']

            supervision_mask = object_mask
            occ_mask = hand_mask

            loss_mask_union = (supervision_mask.bool() | (rendered_alpha[0, ..., 0] > 1e-3))
            loss_mask_inter = (supervision_mask.bool() & (rendered_alpha[0, ..., 0] > 1e-3))
            loss_mask_union = ((occ_mask <= 0) & loss_mask_union).float()
            loss_mask_inter = ((occ_mask <= 0) & loss_mask_inter).float()

            loss = torch.tensor(0.0, device=device)
            alpha = rendered_alpha[0, ..., 0]
            alpha_gt = supervision_mask
            if cfg.mask_loss_type == 'l1':
                alpha_diff = torch.abs(alpha - alpha_gt.float())
                alpha_diff = alpha_diff * loss_mask_union * (1 - occ_mask.float())
                alpha_loss = torch.sum(alpha_diff) / (torch.sum(loss_mask_union) + 1e-8)
            elif cfg.mask_loss_type == 'l2':
                alpha_diff = (alpha - alpha_gt.float()) ** 2
                alpha_diff = alpha_diff * loss_mask_union * (1 - occ_mask.float())
                alpha_loss = torch.sum(alpha_diff) / (torch.sum(loss_mask_union) + 1e-8)
            elif cfg.mask_loss_type == 'iou':
                visible = 1 - occ_mask.float()
                inter_sum = torch.sum(alpha * alpha_gt.float() * visible)
                union_sum = torch.sum((alpha + alpha_gt.float() - alpha * alpha_gt.float()) * visible)
                iou = inter_sum / (union_sum + 1e-8)
                alpha_loss = 1.0 - iou
            elif cfg.mask_loss_type == 'cross_entropy':
                bce_loss = F.binary_cross_entropy(alpha, alpha_gt.float(), reduction='none')
                bce_loss = bce_loss * loss_mask_union * (1 - occ_mask.float())
                alpha_loss = torch.sum(bce_loss) / (torch.sum(loss_mask_union) + 1e-8)
            alpha_loss = alpha_loss * cfg.object_mask_lambda
            loss += alpha_loss
            depth_diff = None

            hand_occ_obj = hand_mask.bool() & (rendered_alpha[0, ..., 0] > 1e-3)
            obj_occ_hand = object_mask.bool() & (self.hand_zbuf[0, ..., 0] > 0)
            obj_zbuf.retain_grad()
            contact_loss, contact_info = zbuf_contact_loss_simplified(
                self.hand_zbuf[0], obj_zbuf[0],
                hand_occ_obj.detach(), obj_occ_hand.detach(),
                cam_K=Ks[0] if i == max_iter - 1 else None,
            )
            if cfg.align_ho_stage2_from_iter is not None and i >= cfg.align_ho_stage2_from_iter:
                contact_loss *= cfg.contact_lambda * 0.5
                loss *= 4.0
            else:
                contact_loss *= cfg.contact_lambda * 5.0
            loss += contact_loss
            loss_dict = {'contact_loss': contact_loss}

            if cfg.object_scale_reg_loss:
                if target_scale is None:
                    scale_target = self.cfg.initial_object_scale
                else:
                    scale_target = target_scale
                if self.cfg.object_scale_dim == 1:
                    scale_diff = self.optimized_obj_scale - scale_target
                    scale_reg_loss = (scale_diff ** 2).mean()
                else:
                    scale_diff = self.optimized_obj_scale - scale_target
                    scale_reg_loss = (scale_diff ** 2).mean()
                scale_reg_loss = scale_reg_loss * self.cfg.object_scale_reg_lambda
                loss_dict['object_scale_reg_loss'] = scale_reg_loss
                loss += scale_reg_loss
            else:
                scale_reg_loss = None

            loss.backward()

            desc = f"Frame {actual_frame_id} [Align Object] Iter {i}: "
            for k, v in loss_dict.items():
                if v is not None and 'loss' in k:
                    desc += f"{k}={v.item():.4f} "
            if cfg.object_scale_dim == 3:
                scale_log = self.optimized_obj_scale.tolist()
                scale_log = ','.join([f'{s:.4f}' for s in scale_log])
            elif cfg.object_scale_dim == 1:
                scale_log = self.optimized_obj_scale.item()
                scale_log = f'{scale_log:.4f}'
            desc += f" object_scale={scale_log} "
            pbar.set_description(desc)

            for optimizer, scheduler in zip(optimizers, schedulers):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if i % 10 == 0 or i == max_iter - 1:
                vis_image = make_vis_image(
                    actual_frame_id,
                    rendered_color[0],
                    images_gt[0],
                    rendered_alpha[0],
                    supervision_mask.unsqueeze(-1),
                    contact_info=contact_info,
                )

                if cfg.visualize_optimization:
                    window_name = f"{os.getpid()}_Align_Object"
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    vis_h, vis_w = vis_image.shape[:2]
                    cv2.resizeWindow(window_name, vis_w // 2, vis_h // 2)
                    cv2.imshow(window_name, vis_image[..., ::-1])
                    cv2.waitKey(1)

                vis_images_list.append(vis_image.astype(np.uint8))

        if cfg.visualize_optimization:
            cv2.destroyWindow(window_name)

        video_dir = os.path.join(cfg.get_result_dir(), "object_optimization_videos")
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, f"object_{actual_frame_id:04d}_align_ho.mp4")
        imageio.mimwrite(video_path, vis_images_list, fps=10)
        pbar.write(f"[Object] Saved HO alignment optimization video to {pc.GREEN}{video_path}{pc.RESET}")

        final_R = object_R.detach().clone()
        final_t = (object_t + object_t / object_t.norm() * tvec_z).detach().clone()
        if cfg.object_scale_dim != 1:
            self.optimized_obj_scale = optimized_scale_original * scale_1d
        delta_pose_6dof = torch.cat([rotation_matrix_to_rvec(final_R), final_t - object_t], dim=0)
        self.optimized_obj_R[frame_id] = final_R
        self.optimized_obj_t[frame_id] = final_t
        self.optimized_obj_pose_offsets[frame_id] = delta_pose_6dof

        self.optimized_obj_scale = self.optimized_obj_scale.detach()
        self.set_obj_base_pose(frame_id, final_R, final_t)

        contact_pcd_path = f"{self.cfg.result_dir}/contact_pcd.ply"
        if contact_info['debug_pcd'] is not None:
            o3d.io.write_point_cloud(
                contact_pcd_path,
                contact_info['debug_pcd']
            )
            pbar.write(f"Saved contact debug point cloud to {pc.CYAN}{contact_pcd_path}{pc.RESET}")
        else:
            pbar.write("Contact pcd is None, skip.")

    def save_optimized_data(
        self, result_dir: str, start_frame: int = 0, end_frame: int = None, frame_step: int = None
    ):
        aligned_dir = f"{result_dir}/aligned{'' if not frame_step else frame_step}"
        os.makedirs(aligned_dir, exist_ok=True)

        if end_frame is None:
            end_frame = len(self.trainset)

        print(f"Saving optimized data to {aligned_dir}...")
        print(f"Optimized frame range: {start_frame} to {end_frame - 1} (total: {end_frame - start_frame} frames)")

        num_optimized_frames = end_frame - start_frame

        final_hand_poses = []
        final_hand_rotations = []
        final_obj_poses = []

        for dataset_idx in range(num_optimized_frames):
            if hasattr(self, 'optimized_hand_R') and hasattr(self, 'optimized_hand_t'):
                hand_R = self.optimized_hand_R[dataset_idx].detach().cpu()
                mano_global_orient = self.trainset[dataset_idx]['mano_global_orient'].squeeze().detach().cpu()
                if (mano_global_orient == 0).all():
                    hand_t = torch.zeros(3, dtype=hand_R.dtype)
                    hand_rvec = torch.zeros(3, dtype=hand_R.dtype)
                    hand_R = torch.eye(3, dtype=hand_R.dtype)
                else:
                    hand_t = self.optimized_hand_t[dataset_idx].detach().cpu()
                    if self.cfg.fix_hand_rotation_to_wilor:
                        if not torch.equal(hand_R, mano_global_orient):
                            raise ValueError('Fixed WiLoR hand rotation changed before export')
                        # The saved transform is relative to the original WiLoR
                        # orientation. Identity preserves it without conversion.
                        hand_rvec = torch.zeros(3, dtype=hand_R.dtype)
                        hand_R = torch.eye(3, dtype=hand_R.dtype)
                    else:
                        hand_R = hand_R @ torch.linalg.inv(mano_global_orient)
                        hand_rvec = rotation_matrix_to_rvec(hand_R)
                hand_pose_6dof = torch.cat([hand_rvec, hand_t], dim=0).numpy()
                final_hand_poses.append(hand_pose_6dof)
                final_hand_rotations.append(hand_R.numpy())

            if hasattr(self, 'optimized_obj_R') and hasattr(self, 'optimized_obj_t'):
                obj_R = self.optimized_obj_R[dataset_idx].detach().cpu()
                obj_t = self.optimized_obj_t[dataset_idx].detach().cpu()

                # Keep the optimized matrix exactly: an axis-angle round trip
                # can alter an otherwise correct pose close to 180 degrees.
                obj_pose = np.eye(4, dtype=obj_R.numpy().dtype)
                obj_pose[:3, :3] = obj_R.numpy()
                obj_pose[:3, 3] = obj_t.numpy()
                final_obj_poses.append(obj_pose)

        if final_hand_poses:
            final_hand_poses = np.array(final_hand_poses)
        else:
            final_hand_poses = np.array([])

        if final_obj_poses:
            final_obj_poses = np.array(final_obj_poses)
        else:
            final_obj_poses = np.array([])

        if self.mano_model is not None and hasattr(self.parser, 'mano_betas'):
            if hasattr(self.parser, 'mano_betas') and self.parser.mano_betas is not None:
                mano_betas = self.parser.mano_betas[:num_optimized_frames]
                np.savez(f"{aligned_dir}/mano_betas.npz", mano_betas=mano_betas)

            if hasattr(self.parser, 'mano_hand_pose') and self.parser.mano_hand_pose is not None:
                if self.cfg.optimize_hand_pose:
                    mano_hand_pose = self.optimized_mano_hand_pose[:num_optimized_frames, None, ...].detach().cpu().numpy()
                else:
                    mano_hand_pose = self.parser.mano_hand_pose[:num_optimized_frames]
                np.savez(f"{aligned_dir}/mano_hand_pose.npz", mano_hand_pose=mano_hand_pose)

            if hasattr(self.parser, 'mano_global_orient') and self.parser.mano_global_orient is not None:
                mano_global_orient = self.parser.mano_global_orient[:num_optimized_frames]
                np.savez(f"{aligned_dir}/mano_global_orient.npz", mano_global_orient=mano_global_orient)

            if hasattr(self.parser, 'mano_transforms') and self.parser.mano_transforms is not None:
                original_transforms = self.parser.mano_transforms

                optimized_translations = []
                optimized_rotations = []
                optimized_rotation_matrices = []

                for frame_idx, frame_id in enumerate(range(start_frame, end_frame)):
                    if frame_idx < len(final_hand_poses):
                        hand_pose_6dof = final_hand_poses[frame_idx]
                        rotation_vector = hand_pose_6dof[:3]
                        translation = hand_pose_6dof[3:]

                        optimized_rotations.append(rotation_vector)
                        optimized_rotation_matrices.append(final_hand_rotations[frame_idx])
                        optimized_translations.append(translation)
                    else:
                        if frame_id < len(original_transforms['translations']):
                            optimized_translations.append(original_transforms['translations'][frame_id])
                        else:
                            optimized_translations.append(np.zeros(3))

                        original_rotations = original_transforms['rotations']
                        if original_rotations.shape == (3, 3):
                            rot_matrix = original_rotations
                        elif original_rotations.ndim == 3 and frame_id < len(original_rotations):
                            rot_matrix = original_rotations[frame_id]
                        else:
                            rot_matrix = np.eye(3)
                        rot_vec = Rotation.from_matrix(rot_matrix).as_rotvec()
                        optimized_rotations.append(rot_vec)
                        optimized_rotation_matrices.append(rot_matrix)

                mano_transforms = {
                    'scales': self.optimized_hand_scale.item() if self.optimized_hand_scale is not None else original_transforms.get('scales', 1.0),
                    'rotation_representation': np.asarray('per-frame-rotvec'),
                    'rotations': np.array(optimized_rotations if len(optimized_rotations) > 0 else np.zeros(3)),
                    # Retain the legacy rotvec field for existing consumers,
                    # and an exact matrix field for restoration and validation.
                    'rotation_matrices': np.array(optimized_rotation_matrices),
                    'translations': np.array(optimized_translations)
                }
                np.savez(f"{aligned_dir}/mano_transforms.npz", **mano_transforms)

            if hasattr(self.parser, 'mano_faces'):
                mano_faces = self.parser.mano_faces
                if torch.is_tensor(mano_faces):
                    mano_faces = mano_faces.cpu().numpy()
                np.savez(f"{aligned_dir}/mano_faces.npz", mano_faces=mano_faces)

            optimized_verts = []
            optimized_joints = []

            for dataset_idx in range(num_optimized_frames):
                frame_data = self.trainset[dataset_idx]

                mano_betas = frame_data['mano_betas'].to(self.device)
                if self.cfg.optimize_hand_pose:
                    hand_pose = self.optimized_mano_hand_pose[dataset_idx].unsqueeze(0).to(self.device)
                else:
                    hand_pose = frame_data['mano_hand_pose'].to(self.device)
                global_orient = frame_data['mano_global_orient'].to(self.device)

                if self.optimized_hand_scale is not None:
                    transform_scale = self.optimized_hand_scale
                else:
                    transform_scale = frame_data['mano_transform_scale'].to(self.device)

                if dataset_idx < len(final_hand_poses):
                    hand_pose_6dof = torch.from_numpy(final_hand_poses[dataset_idx]).float().to(self.device)
                    transform_R = torch.from_numpy(final_hand_rotations[dataset_idx]).to(self.device)
                    transform_t = hand_pose_6dof[3:]
                else:
                    transform_R = frame_data['mano_transform_rotation'].to(self.device)
                    transform_t = frame_data['mano_transform_translation'].to(self.device)

                mano_output = self.mano_model(
                    hand_pose=hand_pose,
                    betas=mano_betas,
                    pose2rot=False
                )

                vertices = mano_output.vertices[0]
                joints = mano_output.joints[0]

                wrist_pos = joints[0]
                R_global = global_orient[0, 0]

                vertices_centered = vertices - wrist_pos
                vertices_rotated = vertices_centered @ R_global.T

                joints_centered = joints - wrist_pos
                joints_rotated = joints_centered @ R_global.T

                vertices_scaled = transform_scale * vertices_rotated
                vertices_transformed = vertices_scaled @ transform_R.T + transform_t

                joints_scaled = transform_scale * joints_rotated
                joints_transformed = joints_scaled @ transform_R.T + transform_t

                optimized_verts.append(vertices_transformed.detach().cpu().numpy())
                optimized_joints.append(joints_transformed.detach().cpu().numpy())

            if optimized_verts:
                optimized_verts = np.array(optimized_verts)
                optimized_verts = optimized_verts[:, np.newaxis, :, :]
                np.savez(f"{aligned_dir}/optimized_mano_verts.npz", mano_verts=optimized_verts)

            if optimized_joints:
                optimized_joints = np.array(optimized_joints)
                optimized_joints = optimized_joints[:, np.newaxis, :, :]
                np.savez(f"{aligned_dir}/optimized_mano_joints.npz", mano_joints=optimized_joints)

        if len(final_obj_poses) > 0:
            if (self.cfg.optimize_object_scale or self.cfg.optimize_object_scale_not_aligning) and self.optimized_obj_scale is not None:
                obj_scale = self.optimized_obj_scale.detach().cpu().numpy()
            else:
                obj_scale = 1.0

            np.savez(
                f"{aligned_dir}/object_poses.npz",
                object_poses_in_cam=final_obj_poses,
                object_scale=obj_scale,
            )

        object_mesh_obj = self.scene_renderer.get_mesh_object("object")
        if object_mesh_obj is not None:
            print("Saving optimized object mesh from scene renderer...")

            canonical_vertices = object_mesh_obj.canonical_vertices.detach().cpu().numpy()
            faces = object_mesh_obj.faces.detach().cpu().numpy()

            optimized_vertices = canonical_vertices

            optimized_obj_mesh = trimesh.Trimesh(vertices=optimized_vertices, faces=faces, process=False)
            optimized_obj_mesh.export(f"{aligned_dir}/object_mesh.obj")
            print(f"Saved optimized object mesh with {len(optimized_vertices)} vertices and {len(faces)} faces")

            if hasattr(object_mesh_obj, 'texture_image') and object_mesh_obj.texture_image is not None:
                optimized_texture = object_mesh_obj.texture_image
                if isinstance(optimized_texture, torch.nn.Parameter):
                    optimized_texture = optimized_texture.detach()

                texture_np = optimized_texture.cpu().numpy()
                if texture_np.ndim == 4 and texture_np.shape[0] == 1:
                    texture_np = texture_np[0]

                from optimization.data.result_bundle import texture_uint8
                texture_np = texture_uint8(texture_np)

                print(f"Prepared optimized texture with shape {texture_np.shape}")

                if hasattr(object_mesh_obj, 'uvs') and object_mesh_obj.uvs is not None:
                    uvs_np = object_mesh_obj.uvs.detach().cpu().numpy()

                    from trimesh.visual import TextureVisuals
                    from trimesh.visual.material import SimpleMaterial
                    from PIL import Image

                    texture_pil = Image.fromarray(texture_np)
                    visual = TextureVisuals(uv=uvs_np, material=SimpleMaterial(
                        image=texture_pil, diffuse=[255, 255, 255, 255]))

                    textured_mesh = trimesh.Trimesh(
                        vertices=optimized_vertices,
                        faces=faces,
                        visual=visual, process=False
                    )

                    textured_mesh.export(f"{aligned_dir}/object_mesh.obj")
                    print("Saved textured mesh with UV coordinates, texture exported as material_0.png")
                else:
                    print("Note: No UV coordinates available, exporting mesh without texture")

        elif hasattr(self.parser, 'object_mesh') and self.parser.object_mesh is not None:
            print("Warning: Using original mesh from dataloader as fallback")
            obj_mesh = self.parser.object_mesh.copy()
            obj_mesh.export(f"{aligned_dir}/object_mesh.obj")

            try:
                if hasattr(self.parser, 'data_dir'):
                    for ext in ['.mtl', '.jpg', '.jpeg', '.png']:
                        for filename in os.listdir(self.parser.data_dir):
                            if filename.endswith(ext):
                                src = os.path.join(self.parser.data_dir, filename)
                                dst = os.path.join(aligned_dir, filename)
                                shutil.copy2(src, dst)
            except Exception as e:
                print(f"Warning: Could not copy material files: {e}")

        # The renderer owns an optimization proxy. The deliverable surface keeps
        # the final input geometry, UVs, and full-resolution source material.
        mesh_path = Path(aligned_dir) / 'object_mesh.obj'
        if mesh_path.exists():
            from optimization.data.object_assets import write_native_object_assets, load_mesh
            asset_manifest = self.cfg.object_asset_manifest
            if asset_manifest is None:
                automatic = Path(self.cfg.data_dir) / 'object_asset' / 'asset.json'
                if automatic.is_file():
                    asset_manifest = automatic
            write_native_object_assets(aligned_dir, load_mesh(mesh_path),
                getattr(self.parser, 'object_mesh', None), asset_manifest=asset_manifest,
                appearance_optimized=self.cfg.optimize_object_texture or self.cfg.optimize_vertex_colors)

        cam_c2w = getattr(self.parser, 'cam_c2w', None)
        if cam_c2w is None:
            cam_c2w = getattr(self.parser, 'camtoworlds', None)
        if cam_c2w is None:
            num_camera_frames = len(self.parser.rgbd)
            cam_c2w = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], num_camera_frames, axis=0)
            print("Warning: parser has no camera poses; saving identity cam_c2w.")

        camera_data = {
            'intrinsic': self.parser.K,
            'rgbd': self.parser.rgbd[:num_optimized_frames],
            'cam_c2w': cam_c2w[:num_optimized_frames],
            'prediction_frame_ids': np.asarray(
                [int(self.trainset[i]['frame_id']) for i in range(num_optimized_frames)],
                dtype=np.int64),
        }
        np.savez(f"{aligned_dir}/camera_data.npz", **camera_data)
        # This sidecar describes the native export; pack_result creates the
        # portable result.json bundle without retaining dataset dependencies.
        with open(f"{aligned_dir}/export_metadata.json", "w") as stream:
            json.dump({"schema_version": 1, "coordinates": "opencv-camera-metres",
                       "object_scale_baked": False,
                       "texture_encoding": "srgb-unmodified",
                       "rgb_encoding": "byte-range-0-255",
                       "handedness": "unknown",
                       "frame_ids_source": "saved-prediction-frame-ids",
                       "time_source": "unknown"}, stream, indent=2)


        print(f"Optimized data saved to {pc.GREEN}{aligned_dir}{pc.RESET}")
        print("This data can be used with run_viser_hoi and benchmark tools")
        print(f"Run \" {pc.BLUE}python -m scripts.vis.run_viser_hoi --data-dir {aligned_dir}{pc.RESET} \" to visualize.")
        return aligned_dir

    # ==================================================================
    # Main training loop
    # ==================================================================

    def _run_train_loop(
        self,
        combined_vis_frames: list,
        result_dir: str,
        skip_align_ho: bool = False,
        first_frame_max_iter: Optional[int] = None,
        save_periodic: bool = True,
        start_dataset_idx: int = 0,
    ) -> None:
        """Run the per-frame optimization loop over the current trainset.

        Uses ``self.trainset``, ``self.num_frames``, and ``self.optimized_*``
        arrays which must be initialised by the caller.

        Args:
            combined_vis_frames: list to accumulate visualization frames.
            result_dir: directory for saving progress videos / checkpoints.
            skip_align_ho: if True, skip ``align_object_to_hand`` for the
                first frame.
            first_frame_max_iter: override max iterations for the first frame;
                when None uses ``cfg.object_max_iter_first_frame``.
            save_periodic: if False, skip periodic checkpoint / video saves.
            start_dataset_idx: skip indices below this value (forward mode
                only).  Used by bidirectional backward pass to skip the
                pre-seeded contact frame at index 0.
        """
        cfg = self.cfg
        num_frames = self.num_frames

        if cfg.reverse_frame_order:
            frame_indices = range(num_frames - 1, -1, -1)
            first_frame_id = num_frames - 1
            last_frame_id = 0
        else:
            frame_indices = range(start_dataset_idx, num_frames)
            first_frame_id = start_dataset_idx
            last_frame_id = num_frames - 1

        pbar = tqdm.tqdm(frame_indices, total=len(frame_indices), position=0)
        pbar.set_description("Optimizing video")

        for dataset_idx in pbar:
            actual_frame_id = self.get_actual_frame_id(dataset_idx)
            is_first_frame = (dataset_idx == first_frame_id)
            is_last_frame = (dataset_idx == last_frame_id)

            # ---- Hand ----
            self.hand_valid_previous_frame = getattr(self, 'hand_valid_current_frame', True)
            ret = self.hand_optimizer.train_one_frame(
                dataset_idx, train_max_iter=cfg.hand_max_iter_per_frame, optimize_scale=False
            )
            self.hand_valid_current_frame = ret is not None
            if ret is not None:
                self.hand_appeared = True
                optimized_mano_pose_vecs = (
                    so3_log_batch(self.optimized_mano_hand_pose[dataset_idx])
                    if cfg.optimize_hand_pose else None
                )
                self.set_hand_base_pose(
                    dataset_idx,
                    self.optimized_hand_R[dataset_idx],
                    self.optimized_hand_t[dataset_idx],
                    compute_mano=False,
                    hand_pose_vecs=optimized_mano_pose_vecs,
                )

            # ---- Object ----
            if is_first_frame:
                max_iter = (
                    first_frame_max_iter
                    if first_frame_max_iter is not None
                    else cfg.object_max_iter_first_frame
                )
            else:
                max_iter = cfg.object_max_iter_per_frame
            self.object_optimizer.train_one_frame(
                dataset_idx, train_max_iter=max_iter,
                optimize_scale=is_first_frame and cfg.optimize_object_scale_not_aligning,
            )
            self.set_obj_base_pose(dataset_idx, self.optimized_obj_R[dataset_idx],
                                   self.optimized_obj_t[dataset_idx])

            if is_first_frame and not skip_align_ho and self.hand_valid_current_frame:
                self.align_object_to_hand(
                    dataset_idx,
                    self.optimized_obj_R[dataset_idx],
                    self.optimized_obj_t[dataset_idx],
                )

            # Save hand vertices
            hand_mesh_object: HandMesh = self.scene_renderer.get_mesh_object("hand")
            self.last_hand_verts_incam = hand_mesh_object.get_transformed_vertices().detach().clone()

            # Render combined visualization
            render_output = self.render_meshes(
                camtoworlds=torch.eye(4, device=self.device).unsqueeze(0),
                Ks=self.trainset[dataset_idx]['K'].unsqueeze(0).to(self.device),
                height=self.trainset[dataset_idx]['image'].shape[0],
                width=self.trainset[dataset_idx]['image'].shape[1],
                frame_id=dataset_idx,
                render_objects=None if self.hand_appeared else ['object'],
            )

            rendered_rgb = render_output['colors'][0].detach().cpu().numpy()
            rendered_alpha = render_output['alpha'][0].detach().cpu().numpy()
            viz_image = overlay_rgb(
                rendered_rgb,
                self.trainset[dataset_idx]['image'].numpy(),
                rendered_alpha,
            )
            viz_image = (viz_image * 255).astype(np.uint8)
            cv2.putText(viz_image, f"Frame {actual_frame_id}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            combined_vis_frames.append(viz_image)

            ho_pcd_path = self.save_ho_pcd(dataset_idx)
            pbar.write(f"[save_ho_pcd] Saved hand-object point cloud to {pc.BLUE}{ho_pcd_path}{pc.RESET}")

            if save_periodic:
                if (not is_first_frame and dataset_idx % 10 == 0) or is_last_frame:
                    video_path = f"{result_dir}/videos/optimization_progress_frame_{actual_frame_id:04d}.mp4"
                    imageio.mimwrite(video_path, combined_vis_frames, fps=12)
                    pbar.write(f"\nSaved optimization progress video to {pc.CYAN}{video_path}{pc.RESET}\n")

                # Forward snapshots contain only the optimized prefix. A reverse
                # pass has optimized a suffix, so defer export until it is complete.
                if (not cfg.reverse_frame_order and not is_first_frame and dataset_idx % 50 == 0) or is_last_frame:
                    save_idx = None if is_last_frame else actual_frame_id
                    num_saved_frames = num_frames if cfg.reverse_frame_order else dataset_idx + 1
                    if hasattr(self, 'last_save_dir'):
                        if self.last_save_dir is not None and os.path.exists(self.last_save_dir):
                            shutil.rmtree(self.last_save_dir)
                    self.last_save_dir = self.save_optimized_data(
                        result_dir, cfg.start_frame, cfg.start_frame + num_saved_frames, save_idx
                    )

    def _print_final_summary(self) -> None:
        cfg = self.cfg
        if self.optimized_hand_scale is not None:
            print(f"Final optimized hand scale: {self.optimized_hand_scale.item():.4f}")
        if self.optimized_obj_scale is not None:
            if cfg.object_scale_dim == 1:
                print(f"Final optimized object scale: {self.optimized_obj_scale.item():.4f} (isotropic)")
            else:
                print(f"Final optimized object scale: {self.optimized_obj_scale.detach().cpu().numpy()} (anisotropic)")
        if cfg.use_tensorboard:
            self.tb_writer.close()
            print(f"View training logs with: {pc.BLUE}tensorboard --logdir {self.tb_dir}{pc.RESET}")

    def train(self):
        cfg = self.cfg

        if cfg.bidirectional_optimization and cfg.start_frame > 0:
            self.train_bidirectional()
            return

        result_dir = cfg.get_result_dir()
        num_frames = len(self.trainset)
        self.num_frames = num_frames

        # Pose storage
        self.optimized_hand_pose_offsets = torch.zeros((num_frames, 6), device=self.device)
        self.optimized_obj_pose_offsets = torch.zeros((num_frames, 6), device=self.device)
        self.optimized_hand_R = torch.zeros((num_frames, 3, 3), device=self.device)
        self.optimized_hand_t = torch.zeros((num_frames, 3), device=self.device)
        self.optimized_obj_R = torch.zeros((num_frames, 3, 3), device=self.device)
        self.optimized_obj_t = torch.zeros((num_frames, 3), device=self.device)

        if cfg.optimize_hand_pose:
            self.optimized_mano_hand_pose = torch.zeros((num_frames, 15, 3, 3), device=self.device)

        self.shared_hand_rvec = torch.nn.Parameter(torch.zeros(3, device=self.device))

        combined_vis_frames = []
        self.hand_appeared = False

        self._run_train_loop(combined_vis_frames, result_dir)
        self._print_final_summary()

    def train_bidirectional(self) -> None:
        """Bidirectional optimization from contact frame.

        Runs the forward optimizer twice:
        1. Forward pass  – frames [contact_frame, end_frame).
        2. Backward pass – frames [0, contact_frame] loaded with **reversed**
           time order.  Index 0 of the reversed dataset is contact_frame itself
           (the "seed"); its already-optimized pose is pre-populated so that
           ``object_pose_from_last_frame`` naturally seeds contact_frame-1 from
           the forward result.  The seed frame is skipped by the optimizer
           (``start_dataset_idx=1``); only frames [1, n_bwd] are optimized.
           Results are flipped back to natural order afterward.
        3. Backward (frames 0..contact_frame-1) + forward results are
           concatenated and saved as a complete sequence.
        """
        cfg = self.cfg
        contact_frame = cfg.start_frame
        result_dir = cfg.get_result_dir()

        # ── Pre-condition checks ───────────────────────────────────────────
        if cfg.reverse_frame_order:
            raise ValueError(
                "bidirectional_optimization and reverse_frame_order cannot both be True. "
                "Disable reverse_frame_order when using bidirectional_optimization."
            )
        if cfg.hold_mode and contact_frame % cfg.hold_mode_interval != 0:
            raise ValueError(
                f"bidirectional_optimization requires start_frame ({contact_frame}) "
                f"to be a multiple of hold_mode_interval ({cfg.hold_mode_interval}) "
                "so that the contact frame is included in hold-mode sampling."
            )

        # ── 1. FORWARD PASS (contact_frame → end) ─────────────────────────
        print(f"\n{'='*60}")
        print(f"[Bidirectional] FORWARD pass: frames {contact_frame} → end")
        print(f"{'='*60}\n")

        n_fwd = len(self.trainset)
        self.num_frames = n_fwd
        self.optimized_hand_pose_offsets = torch.zeros((n_fwd, 6), device=self.device)
        self.optimized_obj_pose_offsets  = torch.zeros((n_fwd, 6), device=self.device)
        self.optimized_hand_R = torch.zeros((n_fwd, 3, 3), device=self.device)
        self.optimized_hand_t = torch.zeros((n_fwd, 3),    device=self.device)
        self.optimized_obj_R  = torch.zeros((n_fwd, 3, 3), device=self.device)
        self.optimized_obj_t  = torch.zeros((n_fwd, 3),    device=self.device)
        if cfg.optimize_hand_pose:
            self.optimized_mano_hand_pose = torch.zeros((n_fwd, 15, 3, 3), device=self.device)
        self.shared_hand_rvec = torch.nn.Parameter(torch.zeros(3, device=self.device))

        fwd_vis = []
        self.hand_appeared = False
        self._run_train_loop(fwd_vis, result_dir, skip_align_ho=False, save_periodic=True)

        # contact_frame optimized result (dataset index 0 in forward trainset)
        fwd_obj_R  = self.optimized_obj_R[:n_fwd].clone()
        fwd_obj_t  = self.optimized_obj_t[:n_fwd].clone()
        fwd_hand_R = self.optimized_hand_R[:n_fwd].clone()
        fwd_hand_t = self.optimized_hand_t[:n_fwd].clone()
        if cfg.optimize_hand_pose:
            fwd_hand_pose = self.optimized_mano_hand_pose[:n_fwd].clone()
        fwd_trainset = self.trainset

        # ── 2. BACKWARD PASS (frames [0, contact_frame] via reversed data) ─
        print(f"\n{'='*60}")
        print(f"[Bidirectional] BACKWARD pass: frames 0 → {contact_frame - 1} (seeded from contact frame)")
        print(f"{'='*60}\n")

        # Load frames [0, contact_frame) in natural order – used for the
        # combined save (n_bwd frames, no seed).
        bwd_parser_orig, bwd_trainset_orig = self._build_parser_and_dataset(
            cfg.data_dir,
            start_frame=0,
            end_frame=contact_frame,
        )
        n_bwd = len(bwd_trainset_orig)  # frames 0 .. contact_frame-1

        # Load [0, contact_frame+1) so that contact_frame itself is the last
        # entry; after reversing it becomes index 0 (the seed).
        # Note: contact_frame is expected to be a multiple of hold_mode_interval
        # when hold_mode is enabled (which is the standard usage).
        bwd_parser_with_seed, bwd_dataset_with_seed = self._build_parser_and_dataset(
            cfg.data_dir,
            start_frame=0,
            end_frame=contact_frame + 1,
        )
        # Shallow-copy, then reverse time arrays.  [::-1].copy() creates new
        # arrays so bwd_parser_with_seed remains untouched.
        bwd_parser_rev = copy.copy(bwd_parser_with_seed)
        reverse_parser_time_arrays(bwd_parser_rev)
        # Preserve the reader selected from the input contents, not its folder name.
        bwd_trainset_rev = type(bwd_dataset_with_seed)(bwd_parser_rev)

        # n_total_bwd = n_bwd + 1 (the extra slot is the seed at index 0)
        n_total_bwd = len(bwd_trainset_rev)
        assert n_total_bwd == n_bwd + 1, (
            f"Expected {n_bwd + 1} backward frames with seed, got {n_total_bwd}. "
            "Ensure contact_frame (start_frame) is a multiple of hold_mode_interval."
        )

        self.trainset   = bwd_trainset_rev
        self.parser     = bwd_parser_rev
        self.num_frames = n_total_bwd

        self.optimized_hand_pose_offsets = torch.zeros((n_total_bwd, 6), device=self.device)
        self.optimized_obj_pose_offsets  = torch.zeros((n_total_bwd, 6), device=self.device)
        self.optimized_hand_R = torch.zeros((n_total_bwd, 3, 3), device=self.device)
        self.optimized_hand_t = torch.zeros((n_total_bwd, 3),    device=self.device)
        self.optimized_obj_R  = torch.zeros((n_total_bwd, 3, 3), device=self.device)
        self.optimized_obj_t  = torch.zeros((n_total_bwd, 3),    device=self.device)
        if cfg.optimize_hand_pose:
            self.optimized_mano_hand_pose = torch.zeros((n_total_bwd, 15, 3, 3), device=self.device)
        self.shared_hand_rvec = torch.nn.Parameter(torch.zeros(3, device=self.device))

        # ── Pre-populate seed slot (index 0 = contact_frame) ──────────────
        # This is the already-optimised contact_frame result from the forward
        # pass.  The optimizer will use optimized_obj_R[0] / [1] etc. as the
        # "last frame" initialisation for index 1 (actual contact_frame-1).
        self.optimized_obj_R[0]  = fwd_obj_R[0]
        self.optimized_obj_t[0]  = fwd_obj_t[0]
        self.optimized_hand_R[0] = fwd_hand_R[0]
        self.optimized_hand_t[0] = fwd_hand_t[0]
        if cfg.optimize_hand_pose:
            self.optimized_mano_hand_pose[0] = fwd_hand_pose[0]

        # Set object and hand mesh state for the seed frame so that any loss
        # requiring last-frame geometry (ho_relpose_loss, last_hand_verts_incam)
        # is correctly initialised before frame 1 starts.
        self.set_obj_base_pose(0, fwd_obj_R[0], fwd_obj_t[0])
        self.set_hand_base_pose(
            0, fwd_hand_R[0], fwd_hand_t[0], compute_mano=True,
            hand_pose_vecs=so3_log_batch(fwd_hand_pose[0]) if cfg.optimize_hand_pose else None,
        )
        seed_valid = bool(torch.any(self.trainset[0]['mano_hand_pose'] != 0))
        self.hand_valid_current_frame = seed_valid
        self.hand_valid_previous_frame = seed_valid
        seed_render = self.render_meshes(
            camtoworlds=torch.eye(4, device=self.device).unsqueeze(0),
            Ks=self.trainset[0]['K'].unsqueeze(0).to(self.device),
            height=self.trainset[0]['image'].shape[0],
            width=self.trainset[0]['image'].shape[1],
            frame_id=0,
            render_objects=['hand'],
            hard_rasterization=True,
        )
        self.hand_zbuf = seed_render['zbuf'].detach().clone()
        hand_mesh_obj: HandMesh = self.scene_renderer.get_mesh_object("hand")
        self.last_hand_verts_incam = hand_mesh_obj.get_transformed_vertices().detach().clone()
        self.hand_appeared = seed_valid

        # ── Run backward optimization (skip the pre-seeded index 0) ───────
        # save_periodic=False: only the combined result is saved at the end.
        # first_frame_max_iter: normal per-frame budget (no special first-frame).
        bwd_vis = []
        self._run_train_loop(
            bwd_vis, result_dir,
            skip_align_ho=True,
            first_frame_max_iter=cfg.object_max_iter_per_frame,
            save_periodic=False,
            start_dataset_idx=1,
        )

        # Slice off the seed slot [0] and flip so index 0 = actual frame 0:
        #   optimized_obj_R[1]           = actual contact_frame-1 (first optimised)
        #   optimized_obj_R[n_total_bwd-1] = actual frame 0       (last optimised)
        # After flip:
        #   bwd_obj_R[0]         = actual frame 0
        #   bwd_obj_R[n_bwd-1]   = actual frame contact_frame-1
        bwd_obj_R  = self.optimized_obj_R[1:n_total_bwd].flip(0).clone()
        bwd_obj_t  = self.optimized_obj_t[1:n_total_bwd].flip(0).clone()
        bwd_hand_R = self.optimized_hand_R[1:n_total_bwd].flip(0).clone()
        bwd_hand_t = self.optimized_hand_t[1:n_total_bwd].flip(0).clone()
        if cfg.optimize_hand_pose:
            bwd_hand_pose = self.optimized_mano_hand_pose[1:n_total_bwd].flip(0).clone()

        # ── 3. COMBINE & SAVE ──────────────────────────────────────────────
        n_total = n_bwd + n_fwd
        self.optimized_obj_R  = torch.cat([bwd_obj_R,  fwd_obj_R],  dim=0)
        self.optimized_obj_t  = torch.cat([bwd_obj_t,  fwd_obj_t],  dim=0)
        self.optimized_hand_R = torch.cat([bwd_hand_R, fwd_hand_R], dim=0)
        self.optimized_hand_t = torch.cat([bwd_hand_t, fwd_hand_t], dim=0)
        if cfg.optimize_hand_pose:
            self.optimized_mano_hand_pose = torch.cat([bwd_hand_pose, fwd_hand_pose], dim=0)
        self.num_frames = n_total

        # Full-range parser so save_optimized_data gets correct .mano_* and
        # .rgbd arrays covering the complete [0, end_frame) range.
        full_parser, _ = self._build_parser_and_dataset(
            cfg.data_dir,
            start_frame=0,
            end_frame=-1,
        )
        self.parser   = full_parser
        # CombinedDataset: indices [0, n_bwd) → bwd frames in natural order,
        #                  indices [n_bwd, n_total) → forward frames.
        self.trainset = CombinedDataset(bwd_trainset_orig, fwd_trainset)

        print(f"\n[Bidirectional] Saving combined result: "
              f"{n_bwd} backward + {n_fwd} forward = {n_total} frames")
        self.save_optimized_data(result_dir, start_frame=0, end_frame=n_total)

        self._print_final_summary()
