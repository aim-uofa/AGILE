import json
import os
import glob
from typing import Any, Dict, List, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from typing_extensions import assert_never


from optimization.utils.depth import depth_to_normal

class Parser:
    """HOI (Hand-Object Interaction) parser for processed data."""

    def __init__(
        self,
        data_dir: str,
        online_start_frame: int = 0,
        online_end_frame: int = -1,
        hold_mode: bool = False,
        hold_mode_interval: int = 5,
        use_gt_mesh = False,
        mano_transforms_path = None,
    ):
        self.data_dir = data_dir
        self.online_start_frame = online_start_frame
        self.online_end_frame = online_end_frame
        self.hold_mode = hold_mode
        self.hold_mode_interval = hold_mode_interval

        # Check for aligned directory with processed data
        aligned_dir = self.data_dir
        print(f"Using aligned directory: {aligned_dir}")

        # Load processed camera data
        camera_data_path = os.path.join(aligned_dir, "camera_data.npz")
        if not os.path.exists(camera_data_path):
            raise FileNotFoundError(f"Camera data file {camera_data_path} does not exist.")

        camera_data = np.load(camera_data_path)
        print(f"[Parser] Loaded camera data from {camera_data_path}")
        self.K_original = camera_data["intrinsic"]
        self.K = self.K_original.copy()
        rgbd_full = camera_data["rgbd"]  # (N, H, W, 4) - N is time, not viewpoints
        cam_c2w_full = camera_data["cam_c2w"]  # (N, 4, 4) - camera poses over time

        # Determine frame range
        total_frames = rgbd_full.shape[0]
        start_frame = max(0, self.online_start_frame)
        end_frame = total_frames if self.online_end_frame == -1 else min(self.online_end_frame, total_frames)

        # Slice data to only include specified frame range
        self.rgbd = rgbd_full[start_frame:end_frame]  # (N', H, W, 4)
        self.rgbs = self.rgbd[..., :3]  # (N', H, W, 3)
        self.cam_c2w = cam_c2w_full[start_frame:end_frame]  # (N', 4, 4)
        self.frame_ids = np.arange(start_frame, end_frame, dtype=np.int32)

        # Apply hold mode sampling if enabled (take every Nth frame)
        if self.hold_mode:
            num_frames_before_hold = self.rgbd.shape[0]
            # Generate indices: 0, interval, 2*interval, ...
            hold_indices = np.arange(0, num_frames_before_hold, self.hold_mode_interval)
            self.rgbd = self.rgbd[hold_indices]
            self.cam_c2w = self.cam_c2w[hold_indices]
            self.frame_ids = self.frame_ids[hold_indices]
            print(f"[Parser] Hold mode enabled: sampling every {self.hold_mode_interval} frames ({num_frames_before_hold} -> {len(hold_indices)} frames)")
        else:
            hold_indices = None
        self.rgbs = self.rgbd[..., :3]

        # Normal supervision is disabled by the released HO3D preset.
        # Optional caches are resolved relative to the input, never a lab mount.
        self.normals = None
        normal_dir = os.path.join(aligned_dir, 'normal')
        normal_files = sorted(glob.glob(os.path.join(normal_dir, '*.npz')))
        if normal_files:
            selected = normal_files[start_frame:end_frame]
            if hold_indices is not None:
                selected = [selected[i] for i in hold_indices]
            self.normals = np.stack([np.load(p)['normal'] for p in selected])

        num_frames = self.rgbd.shape[0]
        height, width = self.rgbd.shape[1:3]

        print(f"[Parser] Loading frames {start_frame} to {end_frame-1} (total: {num_frames} frames out of {total_frames})")

        # Apply factor to intrinsics
        K = self.K_original.copy()

        # Load MANO data (time-varying)
        mano_verts_path = os.path.join(aligned_dir, "optimized_mano_verts.npz")
        mano_joints_path = os.path.join(aligned_dir, "optimized_mano_joints.npz")
        mano_faces_path = os.path.join(aligned_dir, "mano_faces.npz")

        if os.path.exists(mano_verts_path):
            mano_verts_full = np.load(mano_verts_path)["mano_verts"]  # (N, 1, 778, 3)
            mano_verts_sliced = mano_verts_full[start_frame:end_frame]
            self.mano_verts = mano_verts_sliced[hold_indices] if self.hold_mode else mano_verts_sliced
        else:
            self.mano_verts = None

        hand_joints_2d_path = os.path.join(aligned_dir, 'wilor', 'mano_joints_2d.npz')
        if not os.path.exists(hand_joints_2d_path):
            raise FileNotFoundError(f"Hand joints 2D file {hand_joints_2d_path} does not exist.")
        from preprocessing.single_hand import require_one_hand_axis
        joints_2d_full = require_one_hand_axis(np.load(hand_joints_2d_path)['mano_joints_2d'], 'mano_joints_2d')
        mano_joints_2d_sliced = joints_2d_full[start_frame:end_frame, 0]  # (N', 21, 2)
        self.mano_joints_2d = mano_joints_2d_sliced[hold_indices] if self.hold_mode else mano_joints_2d_sliced

        if os.path.exists(mano_joints_path):
            mano_joints_full = np.load(mano_joints_path)["mano_joints"]  # (N, 1, 21, 3)
            mano_joints_sliced = mano_joints_full[start_frame:end_frame]
            self.mano_joints = mano_joints_sliced[hold_indices] if self.hold_mode else mano_joints_sliced
        else:
            self.mano_joints = None

        if os.path.exists(mano_faces_path):
            self.mano_faces = np.load(mano_faces_path)["mano_faces"]  # (1552, 3) - static topology
        else:
            self.mano_faces = None

        # Load additional MANO parameters
        mano_betas_path = os.path.join(aligned_dir, "mano_betas.npz")
        mano_hand_pose_path = os.path.join(aligned_dir, "mano_hand_pose.npz")
        mano_global_orient_path = os.path.join(aligned_dir, "mano_global_orient.npz")
        mano_masks_path = os.path.join(aligned_dir, "mano_masks.npz")
        if mano_transforms_path is None:
            mano_transforms_path = os.path.join(aligned_dir, "mano_transforms.npz")
            use_precomputed_mano_transforms = False
        else:
            use_precomputed_mano_transforms = True

        if os.path.exists(mano_betas_path):
            mano_betas_full = np.load(mano_betas_path)["mano_betas"]  # (N, 1, 10) - shape parameters
            mano_betas_sliced = mano_betas_full[start_frame:end_frame]
            self.mano_betas = mano_betas_sliced[hold_indices] if self.hold_mode else mano_betas_sliced
        else:
            self.mano_betas = None

        if os.path.exists(mano_hand_pose_path):
            mano_hand_pose_full = np.load(mano_hand_pose_path)["mano_hand_pose"]  # (N, 1, 15, 3, 3) - hand pose as rotation matrices
            mano_hand_pose_sliced = mano_hand_pose_full[start_frame:end_frame]
            self.mano_hand_pose = mano_hand_pose_sliced[hold_indices] if self.hold_mode else mano_hand_pose_sliced
        else:
            self.mano_hand_pose = None

        if os.path.exists(mano_global_orient_path):
            mano_global_orient_full = np.load(mano_global_orient_path)["mano_global_orient"]  # (N, 1, 1, 3, 3) - global orientation as rotation matrix
            mano_global_orient_sliced = mano_global_orient_full[start_frame:end_frame]
            self.mano_global_orient = mano_global_orient_sliced[hold_indices] if self.hold_mode else mano_global_orient_sliced
        else:
            self.mano_global_orient = None

        if os.path.exists(mano_masks_path):
            mano_masks_data = np.load(mano_masks_path)
            loaded_mano_masks_full = mano_masks_data["mano_masks"]  # (N, H_orig, W_orig) - rendered MANO masks
            loaded_mano_masks_sliced = loaded_mano_masks_full[start_frame:end_frame]
            loaded_mano_masks = loaded_mano_masks_sliced[hold_indices] if self.hold_mode else loaded_mano_masks_sliced

            # Resize mano_masks to match image dimensions (height, width)
            if loaded_mano_masks.shape[1:3] != (height, width):
                resized_masks = []
                for i in range(len(loaded_mano_masks)):
                    mask = Image.fromarray(loaded_mano_masks[i].astype(np.uint8) * 255)
                    resized_mask = np.array(mask.resize((width, height), resample=Image.NEAREST)) > 0
                    resized_masks.append(resized_mask)
                self.mano_masks = np.array(resized_masks)  # (N', H, W)
            else:
                self.mano_masks = loaded_mano_masks
        else:
            self.mano_masks = None

        if os.path.exists(mano_transforms_path):
            mano_transforms_data = np.load(mano_transforms_path)
            if use_precomputed_mano_transforms:
                self.mano_transforms = mano_transforms_data
            else:
                translations_full = mano_transforms_data["translations"]  # (N, 3) - per-frame translations
                translations_sliced = translations_full[start_frame:end_frame]
                translations = translations_sliced[hold_indices] if self.hold_mode else translations_sliced
                self.mano_transforms = {
                    "scales": mano_transforms_data["scales"],  # () - scalar scale factor
                    "rotations": mano_transforms_data["rotations"],  # (3, 3) - global rotation matrix
                    "translations": translations,  # (N', 3) - per-frame translations
                }
        else:
            self.mano_transforms = None

        # Load object data
        object_poses_path = os.path.join(aligned_dir, "object_poses.npz")
        if os.path.exists(object_poses_path):
            object_data = np.load(object_poses_path)
            object_poses_full = object_data["object_poses_in_cam"]  # (N, 4, 4) - 6DOF over time
            object_poses_sliced = object_poses_full[start_frame:end_frame]
            self.object_poses_in_cam = object_poses_sliced[hold_indices] if self.hold_mode else object_poses_sliced
        else:
            self.object_poses_in_cam = None

        # Load object mesh (static)
        if use_gt_mesh:
            raise ValueError('The public inference loader does not use ground-truth object meshes')
        else:
            object_mesh_path = os.path.join(aligned_dir, "object_mesh.obj")
        if os.path.exists(object_mesh_path):
            import trimesh
            self.object_mesh = trimesh.load(object_mesh_path, process=False)
        else:
            self.object_mesh = None

        # Load object texture and material
        self.object_texture = None
        self.object_material = None

        # Look for material file (*.mtl)
        mesh_dir = aligned_dir
        mtl_files = sorted(glob.glob(os.path.join(aligned_dir, "*.mtl")))
        if mtl_files:
            mtl_path = mtl_files[0]  # Use the first mtl file found
            # Parse mtl file to find texture filename
            try:
                with open(mtl_path, 'r') as f:
                    mtl_content = f.read()
                    self.object_material = mtl_content

                    # Extract texture filename from mtl (look for map_Kd line)
                    for line in mtl_content.split('\n'):
                        if line.strip().startswith('map_Kd'):
                            texture_filename = line.split()[-1]
                            texture_path = os.path.join(mesh_dir, texture_filename)
                            if os.path.exists(texture_path):
                                # Load texture image
                                texture_image = Image.open(texture_path)
                                self.object_texture = np.array(texture_image)
                                print(f"[Parser] Loaded object texture from {texture_filename}, shape: {self.object_texture.shape}")
                            break
            except Exception as e:
                print(f"[Parser] Warning: Failed to load material/texture: {e}")
                self.object_material = None
                self.object_texture = None

        # Load hand masks and object masks
        hand_mask_dir = os.path.join(aligned_dir, "hand")
        object_mask_dir = os.path.join(aligned_dir, "object")

        self.hand_masks = None
        self.object_masks = None
        self.combined_masks = None

        if os.path.exists(hand_mask_dir):
            hand_mask_files = sorted(glob.glob(os.path.join(hand_mask_dir, "*.png")))
            if hand_mask_files:
                # Only load masks for the specified frame range
                hand_mask_files_range = hand_mask_files[start_frame:end_frame]
                # Apply hold mode sampling if enabled
                if self.hold_mode:
                    hand_mask_files_range = [hand_mask_files_range[i] for i in hold_indices]
                self.hand_masks = np.array([
                    np.array(Image.open(f).resize((width, height), resample=Image.NEAREST))
                    for f in hand_mask_files_range
                ])

        if os.path.exists(object_mask_dir):
            object_mask_files = sorted(glob.glob(os.path.join(object_mask_dir, "*.png")))
            if object_mask_files:
                # Only load masks for the specified frame range
                object_mask_files_range = object_mask_files[start_frame:end_frame]
                # Apply hold mode sampling if enabled
                if self.hold_mode:
                    object_mask_files_range = [object_mask_files_range[i] for i in hold_indices]
                self.object_masks = np.array([
                    np.array(Image.open(f).resize((width, height), resample=Image.NEAREST))
                    for f in object_mask_files_range
                ])

        # Create combined masks (union of hand and object masks)
        if self.hand_masks is not None and self.object_masks is not None:
            # Ensure both mask arrays have the same length
            min_frames = min(len(self.hand_masks), len(self.object_masks))
            self.combined_masks = np.logical_or(
                self.hand_masks[:min_frames] > 0,
                self.object_masks[:min_frames] > 0
            ).astype(np.uint8)
        elif self.hand_masks is not None:
            self.combined_masks = (self.hand_masks > 0).astype(np.uint8)
        elif self.object_masks is not None:
            self.combined_masks = (self.object_masks > 0).astype(np.uint8)


        # Generate image names for time sequence
        image_names = [f"{frame_id:06d}.jpg" for frame_id in self.frame_ids]

        # For compatibility, create single camera setup (since we're dealing with time, not multiple viewpoints)
        camera_id = 0
        camera_ids = [camera_id] * num_frames

        Ks_dict = {camera_id: K}
        params_dict = {camera_id: np.empty(0, dtype=np.float32)}
        imsize_dict = {camera_id: (width, height)}
        mask_dict = {camera_id: None}

        # Use camera-to-world poses as-is (time-varying camera poses)
        camtoworlds = self.cam_c2w.copy()
        transform = np.eye(4)

        print(f"[Parser] {num_frames} frames loaded from processed HOI data.")

        self.image_names = image_names  # List[str], (num_frames,)
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.mask_dict = mask_dict  # Dict of camera_id -> mask
        self.transform = transform  # np.ndarray, (4, 4)

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)


class Dataset:
    """HOI dataset class."""

    def __init__(
        self,
        parser: Parser,
        load_depths: bool = True,
        load_normals: bool = False,
    ):
        self.parser = parser
        self.load_depths = load_depths
        self.load_normals = load_normals
        self.indices = np.arange(len(self.parser.image_names))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]

        # Get RGBD data - images are stored in rgbd array
        rgbd = self.parser.rgbd[index]  # (H, W, 4)
        image = rgbd[..., :3]  # RGB channels
        depth = rgbd[..., 3]   # Depth channel

        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        camtoworlds = self.parser.camtoworlds[index]
        mask = self.parser.mask_dict[camera_id]

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "depth": torch.from_numpy(depth).float(),
            "image_id": item,  # the index of the frame in the dataset
            "frame_id": int(self.parser.frame_ids[index]),  # the actual frame number
        }

        # Compute normal map from depth if requested
        if self.load_normals:
            data["normal"] = torch.from_numpy(self.parser.normals[index]).float()

        if mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()

        # Add HOI-specific data
        if self.parser.hand_masks is not None and index < len(self.parser.hand_masks):
            hand_mask = self.parser.hand_masks[index]
            data["hand_mask"] = torch.from_numpy(hand_mask).bool()

        if self.parser.object_masks is not None and index < len(self.parser.object_masks):
            object_mask = self.parser.object_masks[index]
            data["object_mask"] = torch.from_numpy(object_mask).bool()

        if self.parser.combined_masks is not None and index < len(self.parser.combined_masks):
            combined_mask = self.parser.combined_masks[index]
            data["combined_mask"] = torch.from_numpy(combined_mask).bool()

        # Add MANO data
        if self.parser.mano_verts is not None and index < len(self.parser.mano_verts):
            mano_verts = self.parser.mano_verts[index]  # (1, 778, 3)
            data["mano_verts"] = torch.from_numpy(mano_verts).float()

        if self.parser.mano_joints is not None and index < len(self.parser.mano_joints):
            mano_joints = self.parser.mano_joints[index]  # (1, 21, 3)
            data["mano_joints"] = torch.from_numpy(mano_joints).float()

        if self.parser.mano_joints_2d is not None and index < len(self.parser.mano_joints_2d):
            mano_joints_2d = self.parser.mano_joints_2d[index]  # (21, 2)
            data["mano_joints_2d"] = torch.from_numpy(mano_joints_2d).float()

        if self.parser.mano_faces is not None:
            data["mano_faces"] = torch.from_numpy(self.parser.mano_faces).long()

        # Add additional MANO parameters
        if self.parser.mano_betas is not None and index < len(self.parser.mano_betas):
            mano_betas = self.parser.mano_betas[index]  # (1, 10)
            data["mano_betas"] = torch.from_numpy(mano_betas).float()

        if self.parser.mano_hand_pose is not None and index < len(self.parser.mano_hand_pose):
            mano_hand_pose = self.parser.mano_hand_pose[index]  # (1, 15, 3, 3)
            data["mano_hand_pose"] = torch.from_numpy(mano_hand_pose).float()

        if self.parser.mano_global_orient is not None and index < len(self.parser.mano_global_orient):
            mano_global_orient = self.parser.mano_global_orient[index]  # (1, 1, 3, 3)
            data["mano_global_orient"] = torch.from_numpy(mano_global_orient).float()

        if self.parser.mano_masks is not None and index < len(self.parser.mano_masks):
            mano_mask = self.parser.mano_masks[index]  # (H, W)
            data["mano_mask"] = torch.from_numpy(mano_mask).bool()

        if self.parser.mano_transforms is not None:
            # Transform data is mostly static except for per-frame translations
            data["mano_transform_scale"] = torch.tensor(self.parser.mano_transforms["scales"]).float()
            data["mano_transform_rotation"] = torch.from_numpy(self.parser.mano_transforms["rotations"]).float()
            if index < len(self.parser.mano_transforms["translations"]):
                data["mano_transform_translation"] = torch.from_numpy(
                    self.parser.mano_transforms["translations"][index]
                ).float()

        # Add object data
        if self.parser.object_poses_in_cam is not None and index < len(self.parser.object_poses_in_cam):
            object_pose = self.parser.object_poses_in_cam[index]  # (4, 4)
            data["object_pose"] = torch.from_numpy(object_pose).float()

        if self.parser.object_mesh is not None:
            # Object mesh is static, so we include the vertices and faces
            data["object_verts"] = torch.from_numpy(np.array(self.parser.object_mesh.vertices)).float()
            data["object_faces"] = torch.from_numpy(np.array(self.parser.object_mesh.faces)).long()

            # Add UV coordinates if available
            if hasattr(self.parser.object_mesh.visual, 'uv') and self.parser.object_mesh.visual.uv is not None:
                data["object_uvs"] = torch.from_numpy(np.array(self.parser.object_mesh.visual.uv)).float()

        # Add object texture and material if available
        if self.parser.object_texture is not None:
            data["object_texture"] = torch.from_numpy(self.parser.object_texture).float()

        if self.parser.object_material is not None:
            data["object_material"] = self.parser.object_material  # Keep as string

        return data


def reverse_parser_time_arrays(parser: "Parser") -> None:
    """Reverse all time-varying arrays in a Parser in-place.

    Used to present frames [0, contact_frame) in reverse order to the forward
    optimizer during the backward pass of bidirectional optimization.

    Important: every modification here **must** produce a new array (via
    ``[::-1].copy()`` or ``list(reversed(...))``) rather than mutating an
    existing one.  Callers such as ``train_bidirectional`` use ``copy.copy``
    to obtain a shallow copy of the Parser before calling this function; if
    arrays were modified in-place the original parser would be corrupted.
    """
    for attr in (
        'rgbd', 'cam_c2w', 'camtoworlds', 'frame_ids',
        'mano_joints_2d', 'mano_betas', 'mano_hand_pose',
        'mano_global_orient', 'hand_masks', 'object_masks',
        'combined_masks', 'object_poses_in_cam',
        'mano_verts', 'mano_joints', 'mano_masks', 'normals',
    ):
        val = getattr(parser, attr, None)
        if val is not None and isinstance(val, np.ndarray):
            setattr(parser, attr, val[::-1].copy())
    # rgbs is a slice of rgbd; re-derive it so it stays consistent.
    if getattr(parser, 'rgbs', None) is not None and parser.rgbd is not None:
        parser.rgbs = parser.rgbd[..., :3]
    for attr in ('image_names', 'camera_ids'):
        val = getattr(parser, attr, None)
        if val is not None:
            setattr(parser, attr, list(reversed(val)))
    if (parser.mano_transforms is not None
            and isinstance(parser.mano_transforms, dict)
            and 'translations' in parser.mano_transforms
            and isinstance(parser.mano_transforms['translations'], np.ndarray)):
        t = dict(parser.mano_transforms)
        t['translations'] = t['translations'][::-1].copy()
        parser.mano_transforms = t


class CombinedDataset:
    """Concatenates two Dataset instances sequentially.

    Used after bidirectional optimization to present the merged backward + forward
    datasets to save_optimized_data in natural time order.
    """

    def __init__(self, ds1: "Dataset", ds2: "Dataset") -> None:
        self.ds1 = ds1
        self.ds2 = ds2
        # Expose forward (ds2) parser as .parser for static attributes accessed by
        # save_optimized_data (object_mesh, mano_faces, K, etc.)
        self.parser = ds2.parser

    def __len__(self) -> int:
        return len(self.ds1) + len(self.ds2)

    def __getitem__(self, idx: int):
        n1 = len(self.ds1)
        return self.ds1[idx] if idx < n1 else self.ds2[idx - n1]
