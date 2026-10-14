"""Object pose optimizer for per-frame HOI mesh optimization; orchestrated by pipelines/trainer.py."""
from __future__ import annotations
from optimization.optim.convergence import ConvergenceMonitor, optimizer_snapshot, optimizer_step_norm

from optimization.optim.recovery import recover_object_initialization

from copy import copy
import os
import cv2
import imageio
import numpy as np
import torch
import tqdm

from optimization.utils.geometry import project_points, rotation_vector_to_matrix_safe, rotation_matrix_to_rvec
from optimization.utils.io import PrintColors as pc
from optimization.utils.visualization import make_vis_image
from optimization.utils.feature import feature_loss, feature_loss_global
from optimization.losses import LossContext, compute_losses
from optimization.losses.contact import zbuf_contact_loss_simplified


class ObjectOptimizer:
    """Encapsulates per-frame object pose optimization logic.

    Uses the loss registry (losses.compute_losses) for all loss computation.
    Orchestrated by pipelines.trainer.Runner.
    """

    def __init__(self, cfg, runner):
        """
        Args:
            cfg: Config dataclass.
            runner: Reference to the Runner instance (provides scene_renderer,
                    trainset, mano_model, etc.).
        """
        self.cfg = cfg
        self.runner = runner
        self.device = runner.device

    def _make_loss_cfg(
        self,
        frame_id: int,
        iter_id: int,
        is_first_frame: bool,
        optimize_scale: bool,
        sdf_loss_on: bool,
    ):
        cfg = copy(self.cfg)
        cfg.hand_mask_loss = False
        cfg.hand_depth_loss = False
        cfg.hand_smoothness_loss = False
        cfg.hand_2d_joints_loss = False
        cfg.hand_sdf_loss = False
        cfg.object_depth_loss = False
        cfg.object_depth_sdf_loss = False

        start_smooth_id = 3 if cfg.object_smoothness_type == 'se3' else 1
        if cfg.object_smoothness_loss and frame_id >= start_smooth_id and cfg.reverse_frame_order:
            raise NotImplementedError("object_smoothness_loss not implemented yet.")
        cfg.object_smoothness_loss = cfg.object_smoothness_loss and frame_id >= start_smooth_id
        cfg.object_contact_loss = (
            cfg.object_contact_loss
            and iter_id > cfg.contact_from_iter
            and iter_id < cfg.contact_until_iter
            and hasattr(self.runner, 'hand_zbuf')
            and getattr(self.runner, 'hand_valid_current_frame', True)
        )
        cfg.object_sdf_loss = cfg.object_sdf_loss and (iter_id > cfg.sdf_from_iter or sdf_loss_on)
        cfg.ho_relpose_loss = (
            cfg.ho_relpose_loss
            and not is_first_frame
            and getattr(self.runner, 'hand_valid_current_frame', True)
            and getattr(self.runner, 'hand_valid_previous_frame', True)
            and hasattr(self.runner, 'last_hand_verts_incam')
            and hasattr(self.runner, 'last_hand_verts_in_ob')
            and hasattr(self.runner, 'hand_verts_dist_weights')
        )
        cfg.object_feature_loss = (
            cfg.object_feature_loss
            and hasattr(self.runner, 'feature_pts_canonical')
            and hasattr(self.runner, 'feature_pts_similarity_maps')
            and hasattr(self.runner, 'feature_pts_to_face')
        )
        cfg.object_scale_reg_loss = cfg.object_scale_reg_loss and optimize_scale
        return cfg

    def _build_loss_context(
        self,
        frame_id: int,
        iter_id: int,
        Ks: torch.Tensor,
        images_gt: torch.Tensor,
        object_mask: torch.Tensor,
        hand_mask: torch.Tensor,
        depths_gt: torch.Tensor,
        normals_gt: torch.Tensor,
        render_output: dict,
        curr_R: torch.Tensor,
        curr_t: torch.Tensor,
        loss_cfg,
    ) -> LossContext:
        runner = self.runner
        device = self.device
        object_mesh = runner.scene_renderer.get_mesh_object("object")
        hand_mesh = runner.scene_renderer.get_mesh_object("hand")

        ctx = LossContext(
            rendered_rgb=render_output['colors'],
            rendered_object_alpha=render_output['alpha'],
            rendered_hand_alpha=torch.zeros_like(render_output['alpha']),
            rendered_zbuf=render_output['zbuf'],
            rendered_depth=render_output['depth'],
            rendered_normals=render_output['normals'],
            gt_rgb=images_gt,
            gt_object_mask=object_mask,
            gt_hand_mask=hand_mask,
            gt_depth=depths_gt,
            gt_hand_joints_2d=torch.empty((0, 2), device=device),
            frame_idx=frame_id,
            iteration=iter_id,
            object_vertices=object_mesh.canonical_vertices,
            hand_vertices=hand_mesh.get_transformed_vertices(),
            camera=Ks[0],
        )
        ctx.gt_normals = normals_gt
        ctx.K = Ks[0]
        ctx.obj_full_R = curr_R
        ctx.obj_full_t = curr_t
        ctx.hand_zbuf = getattr(runner, 'hand_zbuf', None)
        ctx.optimized_obj_scale = getattr(runner, 'optimized_obj_scale', None)

        if loss_cfg.object_smoothness_loss:
            start_smooth_id = 3 if self.cfg.object_smoothness_type == 'se3' else 1
            start_idx = max(0, frame_id - start_smooth_id)
            ctx.prev_obj_R_seq = runner.optimized_obj_R[start_idx:frame_id]
            ctx.prev_obj_t_seq = runner.optimized_obj_t[start_idx:frame_id]

        if loss_cfg.object_sdf_loss and all(
            hasattr(runner, attr) for attr in ('object_sdf', 'object_sdf_norm_trans', 'object_sdf_scale')
        ):
            ctx.object_sdf_grid = {
                'sdf': runner.object_sdf,
                'norm_trans': runner.object_sdf_norm_trans,
                'scale': runner.object_sdf_scale,
            }

        if loss_cfg.ho_relpose_loss:
            ctx.last_hand_verts_in_ob = runner.last_hand_verts_in_ob
            ctx.hand_verts_dist_weights = runner.hand_verts_dist_weights

        if loss_cfg.object_feature_loss:
            ctx.dino_features_gt = getattr(self, '_gt_feature', None)
            ctx.face_visibility = render_output.get('face_visibility')
            ctx.feature_pts_canonical = runner.feature_pts_canonical
            ctx.feature_pts_similarity_maps = runner.feature_pts_similarity_maps
            ctx.feature_pts_to_face = runner.feature_pts_to_face

        return ctx

    def _compute_feature_info(self, ctx: LossContext):
        if (
            ctx.face_visibility is None
            or ctx.feature_pts_canonical is None
            or ctx.feature_pts_similarity_maps is None
            or ctx.feature_pts_to_face is None
        ):
            return None

        pts_visibility_mask = ctx.face_visibility[ctx.feature_pts_to_face]
        visible_pts = ctx.feature_pts_canonical[pts_visibility_mask]
        sim_maps_visible = ctx.feature_pts_similarity_maps[pts_visibility_mask]
        scale = ctx.optimized_obj_scale if ctx.optimized_obj_scale is not None else 1.0
        pts_scaled = visible_pts * scale
        pts_cam = pts_scaled @ ctx.obj_full_R.T
        pts_cam = pts_cam + ctx.obj_full_t.unsqueeze(0)
        sample_coords = project_points(pts_cam, ctx.K)

        if self.cfg.object_feature_loss_type == 'local':
            _, info = feature_loss(
                similarity_maps=sim_maps_visible,
                xys_image=sample_coords,
                occ_mask=ctx.gt_hand_mask.float(),
            )
            max_similarity = None
            xys_target = None
        elif self.cfg.object_feature_loss_type == 'global':
            _, info = feature_loss_global(
                similarity_maps=sim_maps_visible,
                xys_image=sample_coords,
                occ_mask=ctx.gt_hand_mask.float(),
            )
            max_similarity = info.get('max_similarity')
            xys_target = info.get('max_coords_image')
        else:
            raise ValueError(f"Unknown object feature loss type: {self.cfg.object_feature_loss_type}")

        return {
            'similarity': info['similarity'],
            'max_similarity': max_similarity,
            'xys_image': sample_coords,
            'valid_mask': info['valid_pts_mask'],
            'xys_target': xys_target,
        }

    def _compute_ho_pose_info(self, ctx: LossContext):
        if ctx.last_hand_verts_in_ob is None or ctx.hand_verts_dist_weights is None:
            return None

        hand_vert_in_cam = ctx.hand_vertices.detach()
        hand_vert_in_ob = (hand_vert_in_cam - ctx.obj_full_t.unsqueeze(0)) @ ctx.obj_full_R
        last_hand_vert_in_cam = (ctx.last_hand_verts_in_ob @ ctx.obj_full_R.T) + ctx.obj_full_t.unsqueeze(0)
        verts_dist = torch.norm(hand_vert_in_ob - ctx.last_hand_verts_in_ob, dim=1)
        verts_dist = verts_dist * ctx.hand_verts_dist_weights
        return {
            'pose_dist': verts_dist,
            'weights': ctx.hand_verts_dist_weights,
            'curr_v': hand_vert_in_ob,
            'curr_v_cam': hand_vert_in_cam,
            'last_v': ctx.last_hand_verts_in_ob,
            'last_v_cam': last_hand_vert_in_cam,
        }

    @staticmethod
    def _to_legacy_loss_dict(raw_losses: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        key_map = {
            'object_mask': 'alpha_loss',
            'object_depth': 'depth_loss',
            'normal': 'normal_loss',
            'object_edge': 'edge_loss',
            'object_smoothness': 'smoothness_loss',
            'zbuf_contact': 'contact_loss',
            'hand_sdf_for_object': 'sdf_loss',
            'ho_relpose': 'ho_loss',
            'object_feature': 'feature_loss',
            'object_scale_reg': 'object_scale_reg_loss',
        }
        return {key_map.get(name, name): value for name, value in raw_losses.items() if name in key_map}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train_one_frame(self, frame_id: int, train_max_iter: int = 1000,
                        optimize_scale: bool = False) -> dict:
        """Optimize object pose for a single frame.

        This is a direct extraction of ``Runner.train_one_frame_object``.
        """
        cfg = self.cfg
        device = self.device
        runner = self.runner
        actual_frame_id = runner.get_actual_frame_id(frame_id)

        optimize_scale = optimize_scale and (runner.optimized_obj_scale is not None)

        prepared_data = runner.prepare_frame_data(frame_id, data_type='hand')
        Ks = prepared_data['Ks']
        images_gt = prepared_data['images_gt']
        depths_gt = prepared_data['depths_gt']
        depths_masked = prepared_data['depths_masked']
        hand_mask = prepared_data['hand_mask']
        object_mask = prepared_data['object_mask']
        mano_mask = prepared_data['mano_mask']
        edge_gt = prepared_data['object_edge']
        normals_gt = prepared_data['normals_gt']
        H = prepared_data['H']
        W = prepared_data['W']

        is_first_frame = frame_id == runner.num_frames - 1 if cfg.reverse_frame_order else frame_id == 0
        last_frame_id = frame_id + 1 if cfg.reverse_frame_order else frame_id - 1

        if cfg.object_pose_from_last_frame and not is_first_frame:
            base_R = runner.optimized_obj_R[last_frame_id].detach().clone()
            base_t = runner.optimized_obj_t[last_frame_id].detach().clone()
            rvec = torch.zeros(3, device=device)
            tvec = torch.zeros(3, device=device)
        else:
            object_pose_matrix = runner.trainset[frame_id]['object_pose']
            base_R = object_pose_matrix[:3, :3].to(device).detach()
            base_t = object_pose_matrix[:3, 3].to(device).detach()
            rvec = torch.zeros(3, dtype=torch.float32, device=device)
            tvec = torch.zeros(3, dtype=torch.float32, device=device)

        recovered = False
        if cfg.recover_object_pose_from_mask:
            base_R, base_t, recovered = recover_object_initialization(
                runner, frame_id, base_R, base_t, Ks, H, W, object_mask, hand_mask)

        # Setup Optimizers
        rvec = torch.nn.Parameter(rvec)
        tvec = torch.nn.Parameter(tvec)
        rot_optimizer = torch.optim.Adam([{'params': rvec, 'lr': cfg.object_rot_lr}])
        trans_optimizer = torch.optim.Adam([{'params': tvec, 'lr': cfg.object_trans_lr}])
        optimizers = [rot_optimizer, trans_optimizer]

        scheduler_type = cfg.object_scheduler_type
        schedulers = [runner.create_scheduler(opt, train_max_iter, scheduler_type=scheduler_type)
                      for opt in optimizers]

        if optimize_scale:
            obj_scale_optimizer = torch.optim.Adam([
                {'params': runner.optimized_obj_scale, 'lr': cfg.object_scale_lr}
            ])
            obj_scale_scheduler = runner.create_scheduler(
                obj_scale_optimizer, train_max_iter, scheduler_type=scheduler_type
            )
            optimizers.append(obj_scale_optimizer)
            schedulers.append(obj_scale_scheduler)

        vis_images_list = []
        convergence = ConvergenceMonitor(cfg.online_converge_thresh, cfg.converge_loss_rtol, cfg.converge_grad_tol)

        if cfg.object_sdf_loss:
            runner.precompute_hand_sdf()
        sdf_loss_on = False

        if (cfg.ho_relpose_loss and not is_first_frame
                and getattr(runner, 'hand_valid_current_frame', True)
                and getattr(runner, 'hand_valid_previous_frame', True)):
            runner.compute_last_hands_in_ob(frame_id)

        if cfg.object_feature_loss:
            images_gt_np = images_gt.detach().cpu().numpy()
            self._gt_feature = runner.dino_model(images_gt_np).detach()

        base_rvec = rotation_matrix_to_rvec(base_R).to(device)

        pbar = tqdm.trange(train_max_iter, position=1, leave=False,
                           disable=not cfg.show_opt_progress)
        for iter in pbar:
            curr_R = rotation_vector_to_matrix_safe(base_rvec + rvec).to(device)
            curr_t = base_t + tvec
            supervision_mask = object_mask
            runner.set_obj_base_pose(frame_id, curr_R, curr_t)
            pytorch3d_c2w = torch.eye(4, device=device).unsqueeze(0)
            render_output = runner.render_meshes(
                camtoworlds=pytorch3d_c2w,
                Ks=Ks, height=H, width=W,
                frame_id=frame_id,
                random_bkgd=cfg.random_bkgd,
                render_objects=['object'],
                require_normal=True,
            )

            rendered_color = render_output['colors']
            rendered_alpha = render_output['alpha']
            rendered_depth = render_output['depth']
            zbuf = render_output['zbuf']

            occ_mask = hand_mask

            if cfg.object_feature_loss:
                pix_to_face = render_output['pix_to_face'][0, ..., 0]
                compute_feature_every = cfg.compute_feature_every_iter
                if iter % compute_feature_every == 0:
                    runner.sample_points_feature(
                        self._gt_feature,
                        rendered_color,
                        rendered_depth,
                        Ks[0],
                        curr_R, curr_t,
                        pix_to_face,
                        images_gt=images_gt,
                    )
            loss_cfg = self._make_loss_cfg(
                frame_id=frame_id,
                iter_id=iter,
                is_first_frame=is_first_frame,
                optimize_scale=optimize_scale,
                sdf_loss_on=sdf_loss_on,
            )
            if recovered:
                loss_cfg.ho_relpose_loss = False
                loss_cfg.object_smoothness_loss = False
            ctx = self._build_loss_context(
                frame_id=frame_id,
                iter_id=iter,
                Ks=Ks,
                images_gt=images_gt,
                object_mask=object_mask,
                hand_mask=hand_mask,
                depths_gt=depths_gt,
                normals_gt=normals_gt,
                render_output=render_output,
                curr_R=curr_R,
                curr_t=curr_t,
                loss_cfg=loss_cfg,
            )
            raw_loss_dict = compute_losses(loss_cfg, ctx)
            loss_dict = self._to_legacy_loss_dict(raw_loss_dict)
            if 'sdf_loss' in loss_dict:
                sdf_loss_on = True

            loss = rendered_alpha.sum() * 0.0
            for loss_value in loss_dict.values():
                loss = loss + loss_value

            depth_diff = None
            normal_diff = None
            contact_info = None
            ho_pose_info = None
            feature_info = None

            if 'contact_loss' in loss_dict:
                hand_occ_obj = hand_mask.bool() & (rendered_alpha[0, ..., 0] > 1e-3)
                obj_occ_hand = object_mask.bool() & (runner.hand_zbuf[0, ..., 0] > 0)
                _, contact_info = zbuf_contact_loss_simplified(
                    runner.hand_zbuf[0],
                    zbuf[0],
                    hand_occ_obj.detach(),
                    obj_occ_hand.detach(),
                )

            if 'sdf_loss' in loss_dict and loss_dict['sdf_loss'].item() > 0:
                runner.in_contact = True

            if 'ho_loss' in loss_dict:
                ho_pose_info = self._compute_ho_pose_info(ctx)
                if ho_pose_info is not None:
                    ho_pose_info['cam_K'] = Ks[0]

            if 'feature_loss' in loss_dict:
                feature_info = self._compute_feature_info(ctx)

            loss.backward()

            if 'contact_loss' in loss_dict and tvec.grad is not None and rvec.grad is not None:
                tvec_grad_norm = tvec.grad.detach().clone().norm().item()
                rvec_grad_norm = rvec.grad.detach().clone().norm().item()
                abnormal_grad = (
                    tvec_grad_norm > cfg.contact_grad_threshold
                    or rvec_grad_norm > cfg.contact_grad_threshold
                )
                if abnormal_grad:
                    # Preserve the descent direction instead of replacing it
                    # with unrelated random translation/rotation directions.
                    torch.nn.utils.clip_grad_norm_([tvec, rvec], cfg.contact_grad_clip)
                    pbar.write(
                        f"Abnormal contact gradient detected at iter {iter}, "
                        f"rvec_grad_norm={rvec_grad_norm:.4f}, "
                        f"tvec_grad_norm={tvec_grad_norm:.4f}. "
                    )

            # Logging
            desc = f"Frame {actual_frame_id} [Object] Iter {iter}: "
            for k, v in loss_dict.items():
                if v is not None and 'loss' in k:
                    desc += f"{k}={v.item():.4f} "
            if optimize_scale:
                if cfg.object_scale_dim == 3:
                    scale_log = runner.optimized_obj_scale.tolist()
                    scale_log = ','.join([f'{s:.4f}' for s in scale_log])
                elif cfg.object_scale_dim == 1:
                    scale_log = runner.optimized_obj_scale.item()
                    scale_log = f'{scale_log:.4f}'
                desc += f" object_scale={scale_log} "
            pbar.set_description(desc)

            # Update pose
            step_before, grad_norm = optimizer_snapshot(optimizers)
            prev_pose = torch.cat([rvec.detach(), tvec.detach()], dim=0)
            if optimize_scale:
                prev_scale = runner.optimized_obj_scale.detach().clone()
            for optimizer, scheduler in zip(optimizers, schedulers):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            curr_pose = torch.cat([rvec.detach(), tvec.detach()], dim=0)
            pose_change = torch.norm(curr_pose - prev_pose).item()
            if optimize_scale:
                curr_scale = runner.optimized_obj_scale.detach()
                scale_diff = curr_scale - prev_scale
                scale_change = torch.abs(scale_diff).sum().item()
                pose_change += scale_change

            # LR decay can make updates tiny while the objective gradient is large.
            # Report that as stalled and retain the configured iteration budget.
            convergence_status = convergence.observe(
                float(loss.detach()), optimizer_step_norm(optimizers, step_before),
                grad_norm, phase=(tuple(sorted(loss_dict)), sdf_loss_on))
            converged = convergence_status == 'converged'
            if cfg.object_sdf_loss and converged and not sdf_loss_on:
                sdf_loss_on = True
                converged = False
                convergence.reset()

            # Visualize
            if iter % 10 == 0 or iter == train_max_iter - 1 or converged:
                vis_image = make_vis_image(
                    actual_frame_id,
                    rendered_color[0],
                    images_gt[0],
                    rendered_alpha[0],
                    supervision_mask.unsqueeze(-1),
                    depth_diff[0] if depth_diff is not None else None,
                    contact_info=contact_info,
                    feature_info=feature_info,
                    ho_pose_info=ho_pose_info,
                )
                vis_images_list.append(vis_image.astype(np.uint8))

                if cfg.visualize_optimization:
                    window_name = f"{os.getpid()}_Object"
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    vis_h, vis_w = vis_image.shape[:2]
                    cv2.resizeWindow(window_name, vis_w // 2, vis_h // 2)
                    cv2.imshow(window_name, vis_image[..., ::-1])
                    cv2.waitKey(1)

            # TensorBoard
            if runner.tb_writer is not None:
                for loss_name, loss_value in loss_dict.items():
                    if loss_value is not None and 'loss' in loss_name:
                        runner.tb_writer.add_scalar(
                            f'Object/Frame_{frame_id}/{loss_name}',
                            loss_value.item(), iter,
                        )
                runner.tb_writer.add_scalar(
                    f'Object/Frame_{frame_id}/total_loss', loss.item(), iter
                )
                runner.tb_writer.add_scalar(
                    f'Object/Frame_{frame_id}/pose_change', pose_change, iter
                )
                if optimize_scale:
                    if cfg.object_scale_dim == 1:
                        runner.tb_writer.add_scalar(
                            f'Object/Frame_{frame_id}/scale',
                            runner.optimized_obj_scale.item(), iter,
                        )
                    elif cfg.object_scale_dim == 3:
                        for dim_idx, scale_val in enumerate(runner.optimized_obj_scale):
                            runner.tb_writer.add_scalar(
                                f'Object/Frame_{frame_id}/scale_dim_{dim_idx}',
                                scale_val.item(), iter,
                            )

            if converged:
                pbar.write(f"Frame {actual_frame_id} [Object] converged at iteration {iter}/{train_max_iter}.")
                break

        event = dict(frame_id=actual_frame_id, component='object',
                     iterations=iter + 1, budget=train_max_iter,
                     status='converged' if converged else ('stalled_at_budget' if convergence_status == 'stalled' else 'budget_exhausted'),
                     loss=float(loss.detach()), gradient_inf=grad_norm)
        if not hasattr(runner, 'convergence_events'):
            runner.convergence_events = []
        runner.convergence_events.append(event)
        print('[Convergence]', event, flush=True)

        # Store the post-step pose and matching depth for subsequent contact.
        with torch.no_grad():
            curr_R = rotation_vector_to_matrix_safe(base_rvec + rvec).to(device)
            curr_t = base_t + tvec
            runner.set_obj_base_pose(frame_id, curr_R, curr_t)
            final_render = runner.render_meshes(
                camtoworlds=pytorch3d_c2w, Ks=Ks, height=H, width=W,
                frame_id=frame_id, random_bkgd=cfg.random_bkgd,
                render_objects=['object'], require_normal=False,
            )
        runner.obj_zbuf = final_render['zbuf'].detach().clone()

        # Save video
        video_dir = os.path.join(cfg.get_result_dir(), "object_optimization_videos")
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, f"object_{actual_frame_id:04d}.mp4")
        imageio.mimwrite(video_path, vis_images_list, fps=10)
        pbar.write(f"[Object] Saved optimization video to {pc.GREEN}{video_path}{pc.RESET}")

        # Save final optimized pose
        final_R = curr_R.detach().clone()
        final_t = curr_t.detach().clone()

        delta_pose_6dof = torch.cat([rvec.detach(), tvec.detach()], dim=0)
        runner.optimized_obj_R[frame_id] = final_R
        runner.optimized_obj_t[frame_id] = final_t
        runner.optimized_obj_pose_offsets[frame_id] = delta_pose_6dof

        vis_final = vis_image
        ret = {
            'final_R_w2c': final_R,
            'final_t_w2c': final_t,
            'delta_pose_6dof': delta_pose_6dof,
            'vis_final': vis_final,
            'train_step': iter + 1,
        }
        return ret
