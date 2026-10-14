"""Hand pose optimizer for per-frame HOI mesh optimization; orchestrated by pipelines/trainer.py."""
from __future__ import annotations
from optimization.optim.convergence import ConvergenceMonitor, optimizer_snapshot, optimizer_step_norm

from copy import copy
import os
import cv2
import imageio
import numpy as np
import torch
import tqdm

from optimization.losses import LossContext, compute_losses
from optimization.utils.geometry import (
    project_points,
    rotation_vector_to_matrix_safe,
    rotation_matrix_to_rvec,
    so3_log_batch,
    so3_exp_batch,
)
from optimization.utils.io import PrintColors as pc
from optimization.utils.visualization import make_vis_image
from optimization.optim.recovery import fixed_rotation_translation


class HandOptimizer:
    """Encapsulates per-frame hand pose optimization logic.

    Uses the loss registry (losses.compute_losses) for all loss computation.
    Orchestrated by pipelines.trainer.Runner.
    """

    def __init__(self, cfg, runner):
        """
        Args:
            cfg: Config dataclass.
            runner: Reference to the Runner instance.
        """
        self.cfg = cfg
        self.runner = runner
        self.device = runner.device
        if cfg.fix_hand_rotation_to_wilor and cfg.mano_transforms_path is not None:
            raise ValueError('fix_hand_rotation_to_wilor cannot use external mano_transforms_path')
        if cfg.fixed_wilor_translation_init != 'prepared' and not cfg.fix_hand_rotation_to_wilor:
            raise ValueError('fixed_wilor_translation_init requires fix_hand_rotation_to_wilor')

    def _initialize_hand_pose(self, frame_id, K, joints_2d, base_R, base_t):
        runner = self.runner
        if self.cfg.fix_hand_rotation_to_wilor:
            # Rotation comes from this frame, even when translation is propagated.
            original_R = runner.trainset[frame_id]['mano_global_orient']
            original_R = original_R.reshape(-1, 3, 3)[0].to(self.device).detach().clone()
            _, initial_t = runner.set_hand_base_pose(frame_id, None, None, compute_mano=True)
            if base_t is not None:
                initial_t = base_t
            runner.set_hand_base_pose(frame_id, original_R, initial_t, compute_mano=False)
            if self.cfg.fixed_wilor_translation_init == 'joints_2d':
                mesh = runner.scene_renderer.get_mesh_object('hand')
                centered = mesh.current_joints.detach() - initial_t
                initial_t = fixed_rotation_translation(centered, joints_2d, K, initial_t)
                runner.set_hand_base_pose(frame_id, original_R, initial_t, compute_mano=False)
            return original_R, initial_t
        runner.set_hand_base_pose(frame_id, base_R, base_t, compute_mano=True)
        return runner.init_hand_pose_pnp(frame_id, K, joints_2d)

    def _make_loss_cfg(self, is_first_frame: bool, sdf_loss_on: bool):
        cfg = copy(self.cfg)
        cfg.object_mask_loss = False
        cfg.object_depth_loss = False
        cfg.normal_loss = False
        cfg.object_edge_loss = False
        cfg.object_depth_sdf_loss = False
        cfg.object_contact_loss = False
        cfg.object_sdf_loss = False
        cfg.object_feature_loss = False
        cfg.object_scale_reg_loss = False
        cfg.ho_relpose_loss = False
        cfg.hand_sdf_loss = False
        cfg.hand_depth_loss = cfg.hand_depth_loss and not sdf_loss_on
        cfg.hand_smoothness_loss = (
            cfg.hand_smoothness_loss
            and not is_first_frame
            and getattr(self.runner, 'last_hand_verts_incam', None) is not None
            and getattr(self.runner, 'hand_valid_previous_frame', True)
        )
        return cfg

    def _build_loss_context(
        self,
        frame_id: int,
        iter_id: int,
        Ks: torch.Tensor,
        images_gt: torch.Tensor,
        hand_mask: torch.Tensor,
        hand_mask_alpha: torch.Tensor,
        object_mask: torch.Tensor,
        depths_gt: torch.Tensor,
        normals_gt: torch.Tensor,
        mano_joints_2d: torch.Tensor,
        render_output: dict,
        loss_cfg,
    ) -> LossContext:
        runner = self.runner
        hand_mesh = runner.scene_renderer.get_mesh_object("hand")
        object_mesh = runner.scene_renderer.get_mesh_object("object")
        ctx = LossContext(
            rendered_rgb=render_output['colors'],
            rendered_object_alpha=torch.zeros_like(render_output['alpha']),
            rendered_hand_alpha=render_output['alpha'],
            rendered_zbuf=render_output['zbuf'],
            rendered_depth=render_output['depth'],
            rendered_normals=render_output['normals'],
            gt_rgb=images_gt,
            gt_object_mask=object_mask,
            gt_hand_mask=hand_mask,
            gt_depth=depths_gt,
            gt_hand_joints_2d=mano_joints_2d,
            frame_idx=frame_id,
            iteration=iter_id,
            object_vertices=object_mesh.canonical_vertices,
            hand_vertices=hand_mesh.get_transformed_vertices(),
            camera=Ks[0],
        )
        ctx.gt_normals = normals_gt
        ctx.gt_hand_mask_alpha = hand_mask_alpha
        ctx.K = Ks[0]
        ctx.hand_joints_3d = hand_mesh.current_joints
        if loss_cfg.hand_smoothness_loss:
            ctx.prev_hand_verts_seq = runner.last_hand_verts_incam
        return ctx

    @staticmethod
    def _to_legacy_loss_dict(raw_losses: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        key_map = {
            'hand_mask': 'alpha_loss',
            'hand_depth': 'depth_loss',
            'hand_smoothness': 'smoothness_loss',
            'hand_2d_joints': 'joints_2d_loss',
        }
        return {key_map.get(name, name): value for name, value in raw_losses.items() if name in key_map}

    def _compute_depth_diff(self, rendered_depth: torch.Tensor, depths_gt: torch.Tensor):
        if self.cfg.depth_loss_type == 'l1':
            return torch.abs(rendered_depth - depths_gt)
        if self.cfg.depth_loss_type == 'l2':
            return (rendered_depth - depths_gt) ** 2
        if self.cfg.depth_loss_type == 'huber':
            return torch.nn.functional.smooth_l1_loss(
                rendered_depth,
                depths_gt,
                beta=self.cfg.depth_huber_beta,
                reduction='none',
            )
        raise NotImplementedError(f"Depth loss type {self.cfg.depth_loss_type} not implemented.")

    @staticmethod
    def _compute_joints_2d_info(ctx: LossContext):
        joints_2d = project_points(ctx.hand_joints_3d, ctx.K)
        joints_dist = torch.norm(joints_2d - ctx.gt_hand_joints_2d.to(joints_2d.device), dim=1)
        return {
            'joints_2d': joints_2d,
            'joints_2d_gt': ctx.gt_hand_joints_2d,
            'joints_2d_dist': joints_dist,
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train_one_frame(self, frame_id: int, train_max_iter: int,
                        optimize_scale: bool = False) -> dict | None:
        """Optimize hand pose for a single frame.

        This is a direct extraction of ``Runner.train_one_frame_hand``.
        """
        cfg = self.cfg
        device = self.device
        runner = self.runner
        actual_frame_id = runner.get_actual_frame_id(frame_id)

        optimize_scale = optimize_scale and (runner.optimized_hand_scale is not None)

        prepared_data = runner.prepare_frame_data(frame_id, data_type='hand')
        Ks = prepared_data['Ks']
        images_gt = prepared_data['images_gt']
        depths_gt = prepared_data['depths_gt']
        depths_masked = prepared_data['depths_masked']
        hand_mask = prepared_data['hand_mask']
        object_mask = prepared_data['object_mask']
        mano_mask = prepared_data['mano_mask']
        normals_gt = prepared_data['normals_gt']
        H = prepared_data['H']
        W = prepared_data['W']
        hand_pose_matrices = prepared_data['hand_pose']
        mano_joints_2d = prepared_data['mano_joints_2d']

        ### Check Invalid Hand Parameter ###
        if (hand_pose_matrices == 0).all():
            tqdm.tqdm.write("Hand not exists, skip this frame.")
            return None

        # Precompute SDF for hand based on object
        if cfg.hand_sdf_loss or cfg.ho_relpose_loss:
            runner.precompute_object_sdf(canonical=True)

        use_precomputed_hand = cfg.mano_transforms_path is not None
        if use_precomputed_hand:
            R, t = runner.set_hand_base_pose(frame_id, None, None, compute_mano=True)
            render_output = runner.render_meshes(
                camtoworlds=torch.eye(4, device=device).unsqueeze(0),
                Ks=Ks, height=H, width=W,
                frame_id=frame_id,
                random_bkgd=cfg.random_bkgd,
                render_objects=['hand'],
                hard_rasterization=True,
            )
            runner.hand_zbuf = render_output['zbuf'].detach().clone()
            runner.optimized_hand_R[frame_id] = R.detach().clone()
            runner.optimized_hand_t[frame_id] = t.detach().clone()
            return {'final_R_w2c': R, 'final_t_w2c': t}

        if cfg.reverse_frame_order:
            is_first_frame = frame_id == (runner.num_frames - 1)
            last_frame_id = frame_id + 1
        else:
            is_first_frame = frame_id == 0
            last_frame_id = frame_id - 1

        if cfg.hand_pose_from_last_frame:
            base_R = runner.optimized_hand_R[last_frame_id].detach().clone() if not is_first_frame else None
            base_t = runner.optimized_hand_t[last_frame_id].detach().clone() if not is_first_frame else None
        else:
            base_R = None
            base_t = None

        tvec = torch.nn.Parameter(torch.zeros(3, device=device))
        rvec = torch.nn.Parameter(torch.zeros(3, device=device),
                                  requires_grad=not cfg.fix_hand_rotation_to_wilor)

        trans_optimizer = torch.optim.Adam([{'params': tvec, 'lr': cfg.hand_trans_lr}])

        scheduler_type = cfg.hand_scheduler_type
        trans_scheduler = runner.create_scheduler(trans_optimizer, train_max_iter, scheduler_type=scheduler_type)
        optimizers = [trans_optimizer]
        schedulers = [trans_scheduler]
        if not cfg.fix_hand_rotation_to_wilor:
            rot_optimizer = torch.optim.Adam([{'params': rvec, 'lr': cfg.hand_rot_lr}])
            rot_scheduler = runner.create_scheduler(rot_optimizer, train_max_iter, scheduler_type=scheduler_type)
            optimizers.append(rot_optimizer)
            schedulers.append(rot_scheduler)

        if optimize_scale:
            hand_scale_optimizer = torch.optim.Adam([
                {'params': runner.optimized_hand_scale, 'lr': cfg.hand_scale_lr}
            ])
            hand_scale_scheduler = runner.create_scheduler(hand_scale_optimizer, train_max_iter)
            optimizers.append(hand_scale_optimizer)
            schedulers.append(hand_scale_scheduler)

        if cfg.optimize_intrinsics:
            original_fx = Ks[0, 0, 0].item()
            original_fy = Ks[0, 1, 1].item()
            runner.focal_aspect_ratio = original_fy / original_fx
            runner.focal_length = torch.nn.Parameter(torch.tensor(original_fx, device=device))
            intrinsics_optimizer = torch.optim.Adam([
                {'params': runner.focal_length, 'lr': cfg.intrinsics_lr}
            ])
            intrinsics_scheduler = runner.create_scheduler(intrinsics_optimizer, train_max_iter)
            optimizers.append(intrinsics_optimizer)
            schedulers.append(intrinsics_scheduler)

        # Prepare hand pose vectors if needed
        if cfg.optimize_hand_pose:
            hand_pose_vecs = so3_log_batch(hand_pose_matrices[0])
            hand_pose_vecs = hand_pose_vecs.to(device)
            hand_pose_vecs = torch.nn.Parameter(hand_pose_vecs)
            hand_pose_optimizer = torch.optim.Adam([
                {'params': hand_pose_vecs, 'lr': cfg.hand_pose_lr}
            ])
            hand_pose_scheduler = runner.create_scheduler(hand_pose_optimizer, train_max_iter)
            optimizers.append(hand_pose_optimizer)
            schedulers.append(hand_pose_scheduler)
        else:
            hand_pose_vecs = None

        sdf_loss_on = False

        # Get hand base pose
        base_R, base_t = self._initialize_hand_pose(frame_id, Ks[0], mano_joints_2d, base_R, base_t)
        base_rvec = None if cfg.fix_hand_rotation_to_wilor else rotation_matrix_to_rvec(base_R).to(device)

        vis_images_list = []
        convergence = ConvergenceMonitor(cfg.online_converge_thresh, cfg.converge_loss_rtol, cfg.converge_grad_tol)
        pbar = tqdm.trange(train_max_iter, position=1, leave=False,
                           disable=not cfg.show_opt_progress)
        for iter in pbar:
            # Avoid an axis-angle round trip, especially near 180 degrees.
            new_R = base_R if cfg.fix_hand_rotation_to_wilor else rotation_vector_to_matrix_safe(base_rvec + rvec).to(device)
            new_t = base_t + tvec
            optimize_hand_pose_this_iter = cfg.optimize_hand_pose and (iter >= cfg.hand_pose_from_iter)
            if optimize_hand_pose_this_iter:
                curr_hand_pose_vecs = hand_pose_vecs
            else:
                curr_hand_pose_vecs = None

            runner.set_hand_base_pose(
                frame_id, new_R, new_t,
                hand_pose_vecs=curr_hand_pose_vecs,
                compute_mano=optimize_hand_pose_this_iter,
            )

            if cfg.optimize_intrinsics:
                curr_fx = runner.focal_length
                curr_fy = runner.focal_length * runner.focal_aspect_ratio
                Ks = Ks.detach().clone()
                Ks[0, 0, 0] = curr_fx
                Ks[0, 1, 1] = curr_fy

            pytorch3d_c2w = torch.eye(4, device=device).unsqueeze(0)
            render_output = runner.render_meshes(
                camtoworlds=pytorch3d_c2w,
                Ks=Ks, height=H, width=W,
                frame_id=frame_id,
                random_bkgd=cfg.random_bkgd,
                render_objects=['hand'],
                hard_rasterization=cfg.no_blur_for_hand,
            )

            rendered_color = render_output['colors']
            rendered_alpha = render_output['alpha']
            rendered_depth = render_output['depth']
            zbuf = render_output['zbuf']

            supervision_mask = hand_mask
            supervision_hand_mask = mano_mask if cfg.use_mano_mask else supervision_mask
            loss_cfg = self._make_loss_cfg(is_first_frame=is_first_frame, sdf_loss_on=sdf_loss_on)
            ctx = self._build_loss_context(
                frame_id=frame_id,
                iter_id=iter,
                Ks=Ks,
                images_gt=images_gt,
                hand_mask=hand_mask,
                hand_mask_alpha=supervision_hand_mask,
                object_mask=object_mask,
                depths_gt=depths_gt,
                normals_gt=normals_gt,
                mano_joints_2d=mano_joints_2d,
                render_output=render_output,
                loss_cfg=loss_cfg,
            )
            raw_loss_dict = compute_losses(loss_cfg, ctx)
            loss_dict = self._to_legacy_loss_dict(raw_loss_dict)

            if cfg.hand_two_stage_optimization and iter > cfg.hand_stage_one_iters:
                for loss_name in ('alpha_loss', 'depth_loss'):
                    if loss_name in loss_dict:
                        loss_dict[loss_name] = loss_dict[loss_name] * cfg.hand_stage_weight_scale

            joints_2d_info = None
            if 'joints_2d_loss' in loss_dict:
                if cfg.hand_two_stage_optimization and iter <= cfg.hand_stage_one_iters:
                    loss_dict['joints_2d_loss'] = loss_dict['joints_2d_loss'] * cfg.hand_stage_weight_scale
                joints_2d_info = self._compute_joints_2d_info(ctx)

            loss = rendered_alpha.sum() * 0.0
            for loss_value in loss_dict.values():
                loss = loss + loss_value

            depth_diff = self._compute_depth_diff(rendered_depth, depths_gt) if 'depth_loss' in loss_dict else None

            loss.backward()

            ### Logging ###
            desc = f"Frame {actual_frame_id} [Hand] Iter {iter}: "
            for k, v in loss_dict.items():
                if v is not None and 'loss' in k:
                    desc += f"{k}={v.item():.4f} "
            if optimize_scale:
                desc += f" hand_scale={runner.optimized_hand_scale.item():.4f} "
            if cfg.optimize_intrinsics:
                desc += f" fx={Ks[0,0,0].item():.2f} fy={Ks[0,1,1].item():.2f} "
            pbar.set_description(desc)

            ### Optimization step ###
            step_before, grad_norm = optimizer_snapshot(optimizers)
            prev_pose = torch.cat([rvec.detach(), tvec.detach()], dim=0)
            if optimize_scale:
                prev_scale = runner.optimized_hand_scale.detach().clone()
            for optimizer, scheduler in zip(optimizers, schedulers):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            curr_pose = torch.cat([rvec.detach(), tvec.detach()], dim=0)
            pose_change = torch.norm(curr_pose - prev_pose).item()
            if optimize_scale:
                scale_change = torch.abs(runner.optimized_hand_scale.detach() - prev_scale).item()
                pose_change += scale_change

            # LR decay can make updates tiny while the objective gradient is large.
            # Report that as stalled and retain the configured iteration budget.
            convergence_status = convergence.observe(
                float(loss.detach()), optimizer_step_norm(optimizers, step_before),
                grad_norm, phase=(tuple(sorted(loss_dict)), sdf_loss_on, optimize_hand_pose_this_iter, cfg.hand_two_stage_optimization and iter > cfg.hand_stage_one_iters))
            converged = convergence_status == 'converged'
            if cfg.optimize_hand_pose and iter < cfg.hand_pose_from_iter:
                converged = False
            if cfg.object_sdf_loss and converged and not sdf_loss_on:
                sdf_loss_on = True
                converged = False
                convergence.reset()

            # Visualize
            if iter % 10 == 0 or iter == train_max_iter - 1 or converged:
                vis_hand_mask = mano_mask if cfg.use_mano_mask else hand_mask
                vis_image = make_vis_image(
                    actual_frame_id,
                    rendered_color[0],
                    images_gt[0],
                    rendered_alpha[0],
                    vis_hand_mask.unsqueeze(-1),
                    depth_diff[0] if depth_diff is not None else None,
                    max_depth_diff=0.05,
                    hand_joints_info=joints_2d_info,
                )
                vis_images_list.append(vis_image.astype(np.uint8))

                if cfg.visualize_optimization:
                    window_name = f"{os.getpid()}_Hand"
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    vis_h, vis_w = vis_image.shape[:2]
                    cv2.resizeWindow(window_name, vis_w // 2, vis_h // 2)
                    cv2.imshow(window_name, vis_image[..., ::-1])
                    cv2.waitKey(1)

            if converged:
                pbar.write(f"Frame {actual_frame_id} [Hand] converged at iteration {iter}/{train_max_iter}.")
                break

        event = dict(frame_id=actual_frame_id, component='hand',
                     iterations=iter + 1, budget=train_max_iter,
                     status='converged' if converged else ('stalled_at_budget' if convergence_status == 'stalled' else 'budget_exhausted'),
                     loss=float(loss.detach()), gradient_inf=grad_norm)
        if not hasattr(runner, 'convergence_events'):
            runner.convergence_events = []
        runner.convergence_events.append(event)
        print('[Convergence]', event, flush=True)

        # Synchronize geometry after the last optimizer step. Contact alignment
        # must use the same hand pose/scale/shape that will be exported.
        with torch.no_grad():
            new_R = base_R if cfg.fix_hand_rotation_to_wilor else rotation_vector_to_matrix_safe(base_rvec + rvec).to(device)
            new_t = base_t + tvec
            runner.set_hand_base_pose(
                frame_id, new_R, new_t,
                hand_pose_vecs=hand_pose_vecs if cfg.optimize_hand_pose else None,
                compute_mano=cfg.optimize_hand_pose,
            )
            render_output = runner.render_meshes(
                camtoworlds=pytorch3d_c2w,
                Ks=Ks, height=H, width=W,
                frame_id=frame_id,
                random_bkgd=cfg.random_bkgd,
                render_objects=['hand'],
                hard_rasterization=True,
            )
            zbuf = render_output['zbuf']

        runner.hand_zbuf = zbuf.detach().clone()

        if cfg.optimize_hand_pose:
            runner.optimized_mano_hand_pose[frame_id] = so3_exp_batch(hand_pose_vecs).detach().clone()

        # Save video
        video_dir = os.path.join(cfg.get_result_dir(), "hand_optimization_videos")
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, f"hand_{actual_frame_id:04d}.mp4")
        imageio.mimwrite(video_path, vis_images_list, fps=10)
        pbar.write(f"[Hand] Saved optimization video to {pc.GREEN}{video_path}{pc.RESET}")
        if cfg.optimize_intrinsics:
            pbar.write(f"[Hand] Initial intrinsics: fx={original_fx:.2f}, fy={original_fy:.2f}")
            pbar.write(f"[Hand] Final intrinsics: fx={Ks[0,0,0].item():.2f}, fy={Ks[0,1,1].item():.2f}")

        final_R = new_R.detach().clone()
        final_t = new_t.detach().clone()

        delta_pose_6dof = torch.cat([rvec.detach(), tvec.detach()], dim=0)
        runner.optimized_hand_R[frame_id] = final_R
        runner.optimized_hand_t[frame_id] = final_t
        runner.optimized_hand_pose_offsets[frame_id] = delta_pose_6dof

        vis_final = vis_image
        ret = {
            'final_R_w2c': final_R,
            'final_t_w2c': final_t,
            'delta_pose_6dof': delta_pose_6dof,
            'vis_final': vis_final,
            'train_step': iter + 1,
        }
        return ret
