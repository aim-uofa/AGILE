import json
import os
import glob
import sys
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import numpy.core as np_core
import torch
from PIL import Image
from tqdm import tqdm, trange
from typing_extensions import assert_never
import trimesh
from preprocessing.layout import object_pose_directory


@contextmanager
def _temporary_numpy_legacy_pickle_aliases():
    """Temporarily expose legacy numpy internal module paths for old pickles."""
    aliases = {
        "numpy._core": np_core,
        "numpy._core.multiarray": np_core.multiarray,
        "numpy._core.numeric": np_core.numeric,
        "numpy._core.umath": np_core.umath,
    }
    previous = {name: sys.modules.get(name) for name in aliases}
    try:
        for name, module in aliases.items():
            sys.modules[name] = module
        yield
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _load_npz_array(path: str, key: str):
    try:
        with np.load(path, allow_pickle=True) as archive:
            array = archive[key]
    except ModuleNotFoundError as error:
        if not (error.name or '').startswith('numpy._core'):
            raise
        # Only older NumPy reading a NumPy 2 pickle needs these aliases.
        # Installing them for every load can disturb Torch's deserialization.
        with _temporary_numpy_legacy_pickle_aliases():
            with np.load(path, allow_pickle=True) as archive:
                array = archive[key]
    # Legacy WiLoR files may store one Torch tensor per frame and hand.
    if array.dtype == object and array.size and isinstance(array.flat[0], torch.Tensor):
        tensors = [item.detach().cpu().numpy() for item in array.flat]
        array = np.stack(tensors).reshape(array.shape + tensors[0].shape)
    return array

class Parser:
    """ HOI parser for processed data """

    def __init__(
        self,
        data_dir: str,
        mano_model_path: str,
        start_frame: int = 0,
        end_frame: int = -1,
        frame_stride: int = 5,
        mano_transforms_path: Optional[str] = None,
    ):
        self.data_dir = data_dir
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.frame_stride = frame_stride
        self.seq_name = os.path.basename(data_dir.rstrip('/'))

        ### Directories
        rgb_dir = f"{data_dir}/rgb"
        fpose_dir = object_pose_directory(data_dir)
        gt_dir = f"{data_dir}/ho3d_gt"
        hand_mask_dir = f"{data_dir}/mask/hand"
        object_mask_dir = f"{data_dir}/mask/object"
        depth_dir = f"{data_dir}/moge/depth"
        wilor_dir = f"{data_dir}/wilor"
        processed_hand_pose_dir = f"{data_dir}/hand_processed"

        assert os.path.exists(fpose_dir), f"Foundation pose directory {fpose_dir} does not exist."

        ### Load camera intrinsics
        self.K = np.loadtxt(f"{data_dir}/moge/cam_K.txt")

        ### Data paths
        rgb_paths = sorted(glob.glob(f"{rgb_dir}/*.png"))
        if len(rgb_paths) == 0:
            rgb_paths = sorted(glob.glob(f"{rgb_dir}/*.jpg"))
        hand_mask_paths = sorted(glob.glob(f"{hand_mask_dir}/*.png"))
        object_mask_paths = sorted(glob.glob(f"{object_mask_dir}/*.png"))
        depth_paths = sorted(glob.glob(f"{depth_dir}/*.png"))

        print(f"Found {len(rgb_paths)} RGB frames.")
        print(f"Found {len(hand_mask_paths)} hand mask frames.")
        print(f"Found {len(object_mask_paths)} object mask frames.")
        print(f"Found {len(depth_paths)} depth frames.")

        num_frames_total = len(rgb_paths)
        if end_frame == -1:
            end_frame = num_frames_total
        if start_frame == -2:
            start_frame = num_frames_total - 1
        print(f"Using frames from {start_frame} to {end_frame} with stride {frame_stride}.")
        self.frame_ids = np.array(list(range(start_frame, end_frame, frame_stride)))

        ### Load data
        print("Loading RGB-D and masks ...")
        rgbs = []
        depths = []
        hand_masks = []
        object_masks = []
        for i in tqdm(self.frame_ids, desc="Loading frames"):
            rgb = imageio.imread(rgb_paths[i])
            depth = imageio.imread(depth_paths[i]).astype(np.float32) / 1000.0  # Convert mm to m
            hand_mask = imageio.imread(hand_mask_paths[i])
            object_mask = imageio.imread(object_mask_paths[i])

            rgbs.append(rgb)
            depths.append(depth)
            hand_masks.append(hand_mask)
            object_masks.append(object_mask)
        rgbs = np.stack(rgbs, 0)  # (N, H, W, 3)
        depths = np.stack(depths, 0)  # (N, H, W)
        self.rgbs = rgbs
        self.rgbd = np.concatenate([rgbs, depths[..., None]], axis=-1)  # (N, H, W, 4)
        self.hand_masks = np.stack(hand_masks, 0)  # (N, H, W)
        self.object_masks = np.stack(object_masks, 0)  # (N, H, W)

        self.H, self.W = rgbs[0].shape[:2]  # (H, W)
        num_loaded_frames = len(self.frame_ids)
        identity_pose = np.eye(4, dtype=np.float32)
        # Raw processed sequences are optimized in a single camera frame.
        # Expose the same camera-pose interface as the aligned parser so save
        # and reverse helpers can treat both parser types uniformly.
        self.cam_c2w = np.repeat(identity_pose[None, :, :], num_loaded_frames, axis=0)
        self.camtoworlds = self.cam_c2w.copy()
        self.camera_ids = [0] * num_loaded_frames
        self.image_names = [f"{frame_id:06d}.png" for frame_id in self.frame_ids]

        ### Load hand data
        print("Loading hand data ...", end=' ')
        # Raw outputs from wilor
        mano_betas_path = f"{wilor_dir}/mano_betas.npz"
        mano_global_orient_path = f"{wilor_dir}/mano_global_orient.npz"
        mano_hand_pose_path = f"{wilor_dir}/mano_hand_pose.npz"
        mano_masks_path = f"{wilor_dir}/mano_masks.npz"
        mano_joints_2d_path = f"{wilor_dir}/mano_joints_2d.npz"

        # Processed hand data
        if mano_transforms_path is None:
            use_precomputed_mano_transforms = False
            mano_transforms_path = f"{processed_hand_pose_dir}/mano_transforms.npz"
        else:
            use_precomputed_mano_transforms = True
        mano_joints_path = f"{processed_hand_pose_dir}/optimized_mano_joints.npz"
        mano_vertices_path = f"{processed_hand_pose_dir}/optimized_mano_verts.npz"

        # Load wilor data
        self.mano_betas = _load_npz_array(mano_betas_path, 'mano_betas')[self.frame_ids].astype(np.float32) # (N, 1, 10)
        self.mano_hand_pose = _load_npz_array(mano_hand_pose_path, 'mano_hand_pose')[self.frame_ids].astype(np.float32) # (N, 1, 45)
        self.mano_global_orient = _load_npz_array(mano_global_orient_path, 'mano_global_orient')[self.frame_ids].astype(np.float32) # (N, 1, 3)
        self.mano_masks = _load_npz_array(mano_masks_path, 'mano_masks')[self.frame_ids].astype(np.float32) # (N, H, W)
        self.mano_joints_2d = _load_npz_array(mano_joints_2d_path, 'mano_joints_2d')[self.frame_ids].astype(np.float32) # (N, 21, 2

        # Load processed hand data
        self.mano_joints = _load_npz_array(mano_joints_path, 'mano_joints')[self.frame_ids] # (N, 21, 3)
        self.mano_vertices = _load_npz_array(mano_vertices_path, 'mano_verts')[self.frame_ids] # (N, 778, 3)
        with _temporary_numpy_legacy_pickle_aliases():
            mano_transforms_data = np.load(mano_transforms_path, allow_pickle=True)
        mano_translations = mano_transforms_data['translations']
        if not use_precomputed_mano_transforms:
            mano_translations = mano_translations[self.frame_ids]
        self.mano_transforms = {
            'scales': mano_transforms_data['scales'], # ()
            'rotations': mano_transforms_data['rotations'], # (N, 3, 3), rotation matrix
            'translations': mano_translations # (N, 3)
        }

        from preprocessing.single_hand import require_one_hand_axis
        require_one_hand_axis(self.mano_betas, 'mano_betas')
        require_one_hand_axis(self.mano_joints_2d, 'mano_joints_2d')


        # Load mano faces
        mano_obj_path = f"{mano_model_path}/mano_hand.obj"
        mano_obj = trimesh.load(mano_obj_path, process=False)
        self.mano_faces = mano_obj.faces  # (F, 3)

        print("Done.")

        ### Load object mesh
        print("Loading object mesh and material ...", end=' ')
        original_object_mesh_path = f"{fpose_dir}/original_mesh.obj"
        scaled_object_mesh_path = f"{fpose_dir}/scaled_mesh.obj"
        if os.path.exists(scaled_object_mesh_path):
            self.object_mesh = trimesh.load(scaled_object_mesh_path, process=False)
        elif os.path.exists(original_object_mesh_path):
            self.object_mesh = trimesh.load(original_object_mesh_path, process=False)
            object_scales_dir = f'{fpose_dir}/scales'
            object_scale_paths = sorted(glob.glob(f'{object_scales_dir}/*.txt'))
            if 'unified_scale' in object_scale_paths[0]:
                object_scale = np.loadtxt(object_scale_paths[0])
            else:
                scale_idx = self.frame_ids[0] // 5 # It's not frame_stride here, it's fixed to 5
                object_scale = np.loadtxt(object_scale_paths[scale_idx])
            self.object_mesh.vertices *= object_scale
        else:
            raise FileNotFoundError("Object mesh file not found.")

        # Load object texture and material
        mtl_file = glob.glob(f"{fpose_dir}/*.mtl")[0]
        mtl = {}
        with open(mtl_file, 'r') as f:
            mtl_content = f.readlines()
            for line in mtl_content:
                line = line.strip()
                if line.startswith('map_Kd'):
                    texture_file = line.split(' ')[1]
                    texture_path = os.path.join(fpose_dir, texture_file)
                    self.object_texture = imageio.imread(texture_path)
                elif line.startswith('Kd'):
                    parts = line.split(' ')
                    diffuse = [float(parts[1]), float(parts[2]), float(parts[3])]
                    mtl['Kd'] = diffuse
                elif line.startswith('Ka'):
                    parts = line.split(' ')
                    ambient = [float(parts[1]), float(parts[2]), float(parts[3])]
                    mtl['Ka'] = ambient
                elif line.startswith('Ks'):
                    parts = line.split(' ')
                    specular = [float(parts[1]), float(parts[2]), float(parts[3])]
                    mtl['Ks'] = specular
        self.object_mtl = mtl
        print("Done.")
        ### Load object poses and scales
        print("Loading object poses ...", end=' ')
        object_poses_dir = f'{fpose_dir}/ob_in_cam'
        object_pose_paths = sorted(glob.glob(f'{object_poses_dir}/*.txt'))
        object_poses = [np.loadtxt(object_pose_paths[i]) for i in self.frame_ids]
        self.object_poses_in_cam = np.stack(object_poses, 0)  # (N, 4, 4)

        print("Done.")


class Dataset:
    """ HOI dataset for processed data """

    def __init__(
        self,
        parser: Parser,
    ):
        self.parser = parser

    def __len__(self) -> int:
        return len(self.parser.frame_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        K = torch.from_numpy(self.parser.K).float()  # (3, 3)

        rgb = torch.from_numpy(self.parser.rgbd[idx, ..., :3]).float() / 255.0  # (H, W, 3)
        depth = torch.from_numpy(self.parser.rgbd[idx, ..., 3]).float()  # (H, W)

        hand_mask = torch.from_numpy(self.parser.hand_masks[idx]).bool()  # (H, W)
        object_mask = torch.from_numpy(self.parser.object_masks[idx]).bool()  # (H, W)

        mano_betas = torch.from_numpy(self.parser.mano_betas[idx]).float() # (1, 10)
        mano_hand_pose = torch.from_numpy(self.parser.mano_hand_pose[idx]).float()  # (1, 45)
        mano_global_orient = torch.from_numpy(self.parser.mano_global_orient[idx]).float()  # (1, 3)
        mano_joints_2d = torch.from_numpy(self.parser.mano_joints_2d[idx]).float()[0]  # (1, 21, 2)
        mano_joints = torch.from_numpy(self.parser.mano_joints[idx]).float()  # (21, 3)
        mano_vertices = torch.from_numpy(self.parser.mano_vertices[idx]).float()  # (778, 3)
        mano_mask = torch.from_numpy(self.parser.mano_masks[idx]).bool()  # (H, W)

        mano_transform_scale = torch.tensor(self.parser.mano_transforms['scales']).float() # ()
        mano_transform_rotation = torch.from_numpy(self.parser.mano_transforms['rotations']).float() # (3, 3)
        mano_transform_translation = torch.from_numpy(self.parser.mano_transforms['translations'][idx]).float() # (3,)

        object_pose = torch.from_numpy(self.parser.object_poses_in_cam[idx]).float()  # (4, 4)

        data = {
            'K': K,

            'image': rgb,
            'depth': depth,

            'image_id': idx,
            'frame_id': int(self.parser.frame_ids[idx]),

            'object_mask': object_mask,
            'hand_mask': hand_mask,
            'mano_mask': mano_mask,

            'mano_verts': mano_vertices,
            'mano_joints': mano_joints,
            'mano_joints_2d': mano_joints_2d,
            'mano_betas': mano_betas,
            'mano_hand_pose': mano_hand_pose,
            'mano_global_orient': mano_global_orient,
            'mano_transform_scale': mano_transform_scale,
            'mano_transform_rotation': mano_transform_rotation,
            'mano_transform_translation': mano_transform_translation,

            'object_pose': object_pose,
        }
        return data
