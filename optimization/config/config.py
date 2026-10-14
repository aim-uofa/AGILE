"""
Configuration module for mesh training.
"""
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple
import os
from datetime import datetime

@dataclass
class Config:
    # Path to the HOI dataset
    data_dir: str = "datasets/HO3D/processed/train/example_sequence"
    # Path to MANO model directory (containing mano_config.yaml and mano_hand.obj)
    # mano_model_path: str = "data/MANO"
    mano_model_path: str = os.environ.get("HO3D_MANO_ROOT", "assets/mano")
    # Directory to save results (None for auto-generation based on data_dir, losses, and timestamp)
    result_dir: Optional[str] = None
    # Cached result directory (set on first call to get_result_dir())
    _cached_result_dir: Optional[str] = field(default=None, init=False, repr=False)
    # Use tensorboard to log training info
    use_tensorboard: bool = True

    # Optional mano transforms path for hand scaling and initial pose
    mano_transforms_path: Optional[str] = None

    # Mesh rendering parameters
    rasterization_image_size: int = 512  # Image size for rasterization
    rasterization_blur_radius: float = 1e-4  # Blur radius for soft rasterization
    rasterization_faces_per_pixel: int = 48  # Number of faces per pixel
    rasterization_bin_size: Optional[int] = None  # Bin size for rasterization
    rasterization_max_faces_per_bin: Optional[int] = 200000  # Max faces per bin
    rasterization_blend_sigma: float = 1e-5  # Sigma for blending, smaller for sharper depths
    rasterization_blend_gamma: float = 1e-5  # Gamma for blending, smaller for sharper edges
    no_blur_for_hand: bool = True  # No blur for hand mesh rasterization

    # Lighting parameters
    # Increase ambient light to maximum for better texture visibility
    # Ambient light provides uniform illumination regardless of surface orientation
    light_ambient_color: Tuple[float, float, float] = (0.8, 0.8, 0.8)
    light_diffuse_color: Tuple[float, float, float] = (0.2, 0.2, 0.2)
    light_specular_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    material_shininess: float = 10.0

    # Vertex color optimization
    optimize_vertex_colors: bool = False
    initial_vertex_color: Tuple[float, float, float] = (0.7, 0.7, 0.7)
    vertex_color_lr: float = 1e-3

    # Texture optimization for object mesh
    use_object_texture: bool = True  # Whether to use texture for object mesh
    optimize_object_texture: bool = False  # Whether to optimize texture
    texture_lr: float = 1e-3  # Learning rate for texture optimization

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # Use ground-truth object mesh from dataset (if available)
    use_gt_mesh: bool = False

    # Mesh downsample vertices number
    mesh_downsample_n_vert: int = 5000
    # Verified final postprocessed asset; retained separately from the proxy.
    # Its asset-to-input transform is established before mesh simplification.
    object_asset_manifest: Optional[str] = None

    ### MASK LOSS ###

    # Enable mask-based supervision (object only)
    object_mask_loss: bool = True
    # Enable mask-based supervision (hand only)
    hand_mask_loss: bool = False
    # Weight for hand mask loss
    hand_mask_lambda: float = 1.0
    # Weight for object mask loss
    object_mask_lambda: float = 5.0
    # mask_loss_type: 'l1', 'l2' or 'iou'
    mask_loss_type: Literal["l1", "l2", "iou", "cross_entropy"] = 'l2'
    # Use mano mask for hand supervision
    use_mano_mask: bool = True

    ### DEPTH LOSS ###

    # Enable depth loss
    hand_depth_loss: bool = False
    # Enable depth loss
    object_depth_loss: bool = False
    # Enable depth loss only for first frame
    object_first_frame_loss: bool = True
    # Weight for depth loss
    depth_lambda: float = 2.0
    # Depth loss type: 'l1' or 'l2'
    depth_loss_type: Literal["l1", "l2", "huber"] = 'l1'
    # Beta parameter for Huber loss (if used)
    depth_huber_beta: float = 0.05
    # Depth mask type: 'union' or 'intersection'
    depth_mask_type: Literal["union", "intersection"] = 'union'
    # Erode depth map to remove noisy values before computing depth loss
    erode_depth: bool = False

    ### NORMAL LOSS ###

    # Enable normal loss (computed from depth)
    normal_loss: bool = False
    # Weight for normal loss
    normal_lambda: float = 10.0
    # Type of normal loss: 'l1' or 'l2' or 'cosine'
    normal_loss_type: Literal["l1", "l2", "cosine"] = 'l1'
    # Normal mask type: 'union' or 'intersection'
    normal_mask_type: Literal["union", "intersection"] = 'intersection'

    ### SIMILARITY GUIDED VERTICES NORMAL LOSS ###

    # Enable similarity guided normal loss on hand vertices
    similarity_guided_normal_loss: bool = False
    # Weight for similarity guided normal loss
    similarity_guided_normal_lambda: float = 5.0
    # Omit normals with low similarity scores
    similarity_normal_threshold: float = 0.5

    ### MASK EDGE LOSS ###

    # Mask edge loss
    object_edge_loss: bool = False
    # Weight for mask edge loss
    edge_lambda: float = 0.1

    ### DEPTH SDF LOSS ###

    # SDF based depth loss for object mesh
    object_depth_sdf_loss: bool = False
    # SDF loss weight for depth points outside the mesh
    depth_sdf_lambda: float = 80.0
    # SDF loss type: 'l1' or 'l2'
    depth_sdf_loss_type: Literal["l1", "l2"] = 'l1'
    # From which iteration to enable depth SDF loss
    depth_sdf_from_iter: int = 150
    depth_sdf_until_iter: int = -1  # -1 means no limit
    # Depth SDF truncation threshold (in meters)
    depth_sdf_th: float = 0.3

    ### OBJECT SMOOTHNESS LOSS ###

    # Smoothness loss for object pose in SE3 space
    object_smoothness_loss: bool = False
    # Smoothness type
    object_smoothness_type: Literal["se3", "vertices"] = "vertices"
    # Weight for smoothness loss
    object_smoothness_lambda: float = 2e-4

    ### HAND SMOOTHNESS LOSS ###

    # Smoothness loss for hand vertices
    hand_smoothness_loss: bool = True
    # Weight for hand smoothness loss
    hand_smoothness_lambda: float = 0.1

    ### HAND TWO-STAGE OPTIMIZATION ###
    # Two-stage optimization for hand pose (optimize rotation first, then pose)
    hand_two_stage_optimization: bool = False
    # Iterations for first stage (rotation only)
    hand_stage_one_iters: int = 100
    # Weight scale for first stage
    hand_stage_weight_scale: float = 100.0

    ### ZBUF CONTACT LOSS ###

    # Compute contact loss when optimizing object pose
    object_contact_loss: bool = False
    # Compute contact loss when optimizing hand pose
    hand_contact_loss: bool = False
    # Weight for Contact loss
    contact_lambda: float = 1.0
    # Optimize from this iteration
    contact_from_iter: int = 100
    # Optimize until this iteration
    contact_until_iter: int = 300
    # Contact loss only for z translation
    contact_z_only: bool = False
    # Gradient thresholding for contact loss
    contact_grad_threshold: float = 100.0
    # Gradient clipping for contact loss
    contact_grad_clip: float = 5.0

    ### OBJ SDF CONTACT LOSS ###

    # Compute SDF loss when optimizing object pose
    object_sdf_loss: bool = False
    # Compute SDF loss when optimizing hand pose
    hand_sdf_loss: bool = False
    # Weight for SDF loss
    sdf_outside_lambda: float = 3.0
    sdf_inside_lambda: float = 5.0
    # SDF from this iteration
    sdf_from_iter: int = 200
    # SDF grid resolution
    sdf_resolution: int = 128
    # SDF truncation band (in normalized coordinates)
    sdf_truncation_band: float = 0.1
    # SDF contact threshold (in world units) meters
    sdf_contact_threshold: float = 0.05

    ### HAND JOINTS 2D LOSS ###

    # Enable 2D joints loss for hand
    hand_2d_joints_loss: bool = True
    # Weight for 2D joints loss
    hand_joints_2d_lambda: float = 1.0

    ### OBJECT FEATURE LOSS ###

    # Enable feature loss for object
    object_feature_loss: bool = True
    # Input size of DINO model
    object_feature_input_size: int = 1024
    # Weight for object feature loss
    object_feature_loss_lambda: float = 10.0
    # Compute feature every N iterations
    compute_feature_every_iter: int = 20
    # Visualize feature similarity maps
    visualize_feature_maps: bool = False
    # Sample feature from 'vertices' or 'alpha' (alpha blending)
    sample_feature_from: Literal['vertices', 'points'] = 'points'
    # Feature loss type: 'local' or 'global'
    object_feature_loss_type: Literal['local', 'global'] = 'local'
    # Number of sampled feature points (if sampling from vertices or alpha)
    num_sampled_feature_points: int = 3000

    ### OPTIMIZATION PARAMETERS ###

    # Learning rate for object rotation
    object_rot_lr: float = 2e-3
    # Learning rate for object translation
    object_trans_lr: float = 1e-3

    # Learning rate for hand rotation
    hand_rot_lr: float = 5e-3
    # Keep each frame's original WiLoR global rotation exactly: skip hand PnP
    # initialization and rotation updates, including previous-frame rotation reuse.
    # Translation remains optimizable. This overrides hand_rot_lr and is
    # incompatible with mano_transforms_path (which supplies external poses).
    fix_hand_rotation_to_wilor: bool = False
    # Optional image-only translation initialization; never changes WiLoR R.
    fixed_wilor_translation_init: Literal['prepared', 'joints_2d'] = 'prepared'
    # Learning rate for hand translation
    hand_trans_lr: float = 1e-3

    # Only optimize object z translation
    object_z_only_translation: bool = False
    # Z-only learning rate for object translation
    object_z_only_lr: float = 1e-2

    # Optimize hand pose (experimental)
    optimize_hand_pose: bool = False
    # Learning rate for hand pose optimization
    hand_pose_lr: float = 1e-2
    # Optimize hand pose from this iteration
    hand_pose_from_iter: int = 20

    # Optimize intrinsics by hand
    optimize_intrinsics: bool = False
    # Learning rate for intrinsics optimization
    intrinsics_lr: float = 1.0

    ### OBJECT HAND POSE LOSS ###

    # Enable object-hand pose prior loss
    ho_relpose_loss: bool = True
    # Weight for object-hand pose prior loss
    ho_relpose_loss_lambda: float = 400.0
    # Scale factor for tanh weighting in object-hand pose loss, larger means sharper transition
    ho_weight_tanh_scale: float = 40.0
    # Use finger mask for hand vertices in object-hand pose loss
    use_finger_mask: bool = True

    ### OBJECT SCALE REGULARIZATION LOSS ###
    # Enable object scale regularization loss
    object_scale_reg_loss: bool = False
    # Weight for object scale regularization loss
    object_scale_reg_lambda: float = 0.01

    ### TRAINING PARAMETERS ###

    # Enable visualization during online optimization
    visualize_optimization: bool = False
    # Start from this frame of video
    start_frame: int = 0
    # End at this frame of video (if -1, process all frames)
    end_frame: int = -1
    # Reverse frame order for optimization
    reverse_frame_order: bool = False
    # Bidirectional optimization: run forward from contact frame (start_frame) to end,
    # then backward from contact frame to 0 using reversed initial poses, and concatenate
    # both results into a complete sequence. start_frame must be > 0.
    bidirectional_optimization: bool = False
    # Max optimization iterations per frame for object
    object_max_iter_per_frame: int = 400
    # Max optimization iterations for the first frame for object
    object_max_iter_first_frame: int = 800
    # Max optimization iterations for object alignment
    align_ho_max_iter: int = 800
    # Start alignment stage 2 after N iterations
    align_ho_stage2_from_iter: int = int(1e8)
    # Max optimization iterations per frame for hand
    hand_max_iter_per_frame: int = 200

    # Hold mode - sample every N frames for training
    hold_mode: bool = True
    # Sample interval when hold_mode is enabled (e.g., 5 means take frames 0, 5, 10, 15, ...)
    hold_mode_interval: int = 5

    # Initialize object pose from last frame
    object_pose_from_last_frame: bool = True
    # Recover a lost track from its existing NAS pose using observed masks.
    recover_object_pose_from_mask: bool = False
    object_recovery_min_iou: float = 0.05
    # Initilize object pose with last frame velocity
    object_pose_with_velocity: bool = False
    # Initialize hand pose from last frame
    hand_pose_from_last_frame: bool = False
    # Threshold for online convergence
    online_converge_thresh: float = 1e-4
    converge_loss_rtol: float = 1e-5
    converge_grad_tol: float = 1e-3
    # Scheduler type for optimization
    online_scheduler_type: Literal["exponential", "linear", "sequential"] = "sequential"
    # Scheduler type for hand
    hand_scheduler_type: Literal["constant", "exponential", "linear", "sequential"] = "sequential"
    # Scheduler type for object
    object_scheduler_type: Literal["constant", "exponential", "linear", "sequential"] = "constant"


    # Use shared rotation across all frames for hand optimization
    hand_shared_rotation: bool = False

    # Hand scale optimization
    # Whether to optimize hand scale globally
    optimize_hand_scale: bool = False
    # Learning rate for hand scale optimization
    hand_scale_lr: float = 1e-3
    # Initial hand scale value
    initial_hand_scale: float = 1.0
    # Compute hand scale from depths
    compute_hand_scale_from_depth: bool = False

    # Object scale optimization
    # Whether to optimize object scale globally
    optimize_object_scale: bool = True
    # Whether to optimize object scale only in align ho optimization
    optimize_object_scale_not_aligning: bool = False
    # Scale dimension: isotropic (1) or anisotropic (3)
    object_scale_dim: int = 1
    # Learning rate for object scale optimization
    object_scale_lr: float = 8e-3
    # Initial object scale value
    initial_object_scale: float = 1.0

    ### DEBUG PARAMETERS ###

    # Whether to only test the first frame
    test_first_frame: bool = False

    # Show optimization progress bars
    show_opt_progress: bool = False

    # Code snapshot - save all source code to result directory
    save_code_snapshot: bool = True  # Whether to save code snapshot
    code_snapshot_name: str = "code_snapshot"  # Name of the snapshot subdirectory

    def generate_result_dir(self, prefix=None) -> str:
        """
        Generate result directory name based on data_dir, losses, weights, and timestamp.
        Format: results/{scene_name}_{timestamp}_{loss_info}
        """
        # Extract scene name from data_dir (parent directory name)
        # e.g., "data/ABF14_small/aligned" -> "ABF14_small"
        data_path = self.data_dir.rstrip('/')
        parent_dir = os.path.dirname(data_path)
        if 'aligned' in data_path:
            print("Old dataset format detected for result dir generation.")
            scene_name = os.path.basename(parent_dir) if parent_dir else os.path.basename(data_path)
        else:
            scene_name = os.path.basename(data_path)

        # Generate timestamp (MMDD_HHMMSS format)
        timestamp = datetime.now().strftime("%m%d_%H%M%S")

        # Collect enabled losses and their weights
        loss_components = []

        if self.object_mask_loss:
            loss_components.append(f"mask{self.object_mask_lambda}")

        if self.hand_depth_loss:
            loss_components.append(f"depth{self.depth_lambda}")

        if self.normal_loss:
            loss_components.append(f"normal{self.normal_lambda}")

        if self.object_contact_loss:
            loss_components.append(f"contact{self.contact_lambda}")

        if self.object_smoothness_loss:
            loss_components.append(f"smoothness{self.object_smoothness_lambda}")

        if self.object_feature_loss:
            loss_components.append(f"feature{self.object_feature_loss_lambda}")
            loss_components.append(f"type{self.object_feature_loss_type}")

        # Add optimization mode info
        mode_info = []

        if self.object_pose_from_last_frame:
            mode_info.append("lastframe")
        if self.optimize_hand_scale:
            mode_info.append("handscale")

        # Combine all components
        loss_info = "_".join(loss_components) if loss_components else "noloss"
        mode_info_str = "_".join(mode_info) if mode_info else ""

        # Build final directory name
        # parts = [scene_name, timestamp, loss_info]
        parts = [timestamp, loss_info]
        if mode_info_str:
            parts.append(mode_info_str)

        result_name = scene_name + "/" + "_".join(parts)
        if prefix is None:
            prefix = "results"
        return f"{prefix}/{result_name}"

    def get_result_dir(self, prefix=None) -> str:
        """
        Get the result directory, generating one automatically if not specified.
        The result directory is cached on first call to ensure consistency.
        """
        if self.result_dir is not None:
            return self.result_dir

        if self._cached_result_dir is None:
            self._cached_result_dir = self.generate_result_dir(prefix)
            self.result_dir = self._cached_result_dir

        return self._cached_result_dir

    def save_config(self, result_dir: Optional[str] = None, project_root: Optional[str] = None):
        """
        Save config to the result directory as JSON.
        Also saves a code snapshot if enabled.

        Args:
            result_dir: Directory to save the config (uses get_result_dir() if None)
            project_root: Root directory of the project for code snapshot (auto-detected if None)
        """
        import json
        import os

        if result_dir is None:
            result_dir = self.get_result_dir()

        os.makedirs(result_dir, exist_ok=True)
        config_path = f"{result_dir}/config.json"

        with open(config_path, "w") as f:
            json.dump(vars(self), f, indent=2)

        print(f"Config saved to {config_path}")

        # Save code snapshot if enabled
        if self.save_code_snapshot:
            try:
                from optimization.utils.io import save_code_snapshot

                # Auto-detect project root if not provided
                if project_root is None:
                    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

                print(f"\nSaving code snapshot...")
                snapshot_dir = save_code_snapshot(
                    project_root=project_root,
                    result_dir=result_dir,
                    snapshot_name=self.code_snapshot_name
                )
                print(f"✓ Code snapshot completed: {snapshot_dir}\n")
            except Exception as e:
                print(f"⚠ Warning: Failed to save code snapshot: {e}\n")
