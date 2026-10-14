import os.path as op
from .hold_tools.xdict import xdict
import numpy as np
import torch
import trimesh
import cv2
import yaml
from optimization.utils.dataset_utils import HO3D_ROOT, Ho3dReader
import open3d as o3d
import os
from .utils import (
    compute_bounding_box_centers,
    rotation_vector_to_matrix_safe,
    project2d_batch
)
from optimization.utils.geometry import so3_exp_batch
from tqdm import tqdm, trange
from matplotlib import pyplot as plt

SEGM_IDS = {"bg": 0, "object": 50, "right": 150, "left": 250}
HO3D_MANO_ROOT = os.environ.get("HO3D_MANO_ROOT", "assets/mano")

def load_results_data(path, preprocess_hold_mode=False):
    """
    Load HOI data from the specified path.

    Args:
        path: Path to the data directory
        preprocess_hold_mode: If True, sample every 5th frame (0, 5, 10, 15, ...)

    Returns:
        out_dict: Dictionary containing the loaded data
        N: Number of frames (after sampling if preprocess_hold_mode=True)
    """

    print(f"Loading results data from {path}")

    camera_data = np.load(op.join(path, "camera_data.npz")) # intrinsic(3,3), cam_c2w(N,4,4), rgbd(N,480,640,4)
    mano_faces = np.load(op.join(path, "mano_faces.npz"))['mano_faces'] # (1552, 3)
    # mano_masks = np.load(op.join(path, "mano_masks.npz"))['mano_masks'] # (N, 480, 640), bool
    mano_transforms = np.load(op.join(path, "mano_transforms.npz")) # scales, rotations, translations
    object_poses = np.load(op.join(path, 'object_poses.npz')) # object_poses_in_cam, object_poses_in_wld
    optimized_mano_joints = np.load(op.join(path, 'optimized_mano_joints.npz'))['mano_joints'] # (N, 1, 21, 3)
    optimized_mano_verts = np.load(op.join(path, 'optimized_mano_verts.npz'))['mano_verts'] # (N, 1 ,778, 3)

    out_dict = xdict()


    if 'n_original' in camera_data:
        N_original = camera_data['n_original']
    elif 'cam_c2w' in camera_data:
        N_original = len(camera_data['cam_c2w'])
    else:
        N_original = camera_data['rgbd'].shape[0]
    print(f"Original number of frames: {N_original}")

    # Some smoke-test / partial runs save full camera_data but only a subset of
    # optimized predictions. In that case, use the saved prediction length as
    # the authoritative result length instead of indexing by the full sequence.
    saved_lengths = (
        optimized_mano_verts.shape[0],
        optimized_mano_joints.shape[0],
        object_poses['object_poses_in_cam'].shape[0],
    )
    if len(set(saved_lengths)) != 1 or saved_lengths[0] == 0:
        raise ValueError(f"Saved prediction array lengths disagree or are empty: {saved_lengths}")
    N_saved = saved_lengths[0]
    if not np.isfinite(mano_transforms['scales']).all() or np.any(mano_transforms['scales'] <= 0):
        raise ValueError('Saved MANO normalization scale must be finite and positive')
    has_partial_results = N_saved != N_original
    if has_partial_results:
        print(
            f"Detected partial saved results: predictions={N_saved}, "
            f"camera_data={N_original}. Evaluating saved frames only."
        )

    # Apply hold mode sampling: select every 5th frame (0, 5, 10, 15, ...)
    # ``selected_frame_ids`` indexes into prediction arrays (optimized_mano_verts,
    # object_poses_in_cam, etc.), not into camera_data. When the run saved only a
    # prefix of predictions, the upper bound shrinks to N_saved; hold-mode striding
    # still applies on top of that so the returned frames stay aligned with the
    # strided GT produced by load_gt_data.
    print(f"Preprocess hold mode: {preprocess_hold_mode}")
    upper = N_saved if has_partial_results else N_original
    if preprocess_hold_mode:
        selected_frame_ids = np.arange(0, upper, 5)
        N = len(selected_frame_ids)
        print(f"Preprocess hold mode: sampling every 5th frame, {upper} -> {N} frames")
    else:
        selected_frame_ids = np.arange(upper)
        N = upper

    if 'prediction_frame_ids' in camera_data:
        saved_ids = camera_data['prediction_frame_ids']
        if saved_ids.shape != (N_saved,):
            raise ValueError('Saved prediction_frame_ids length disagrees with predictions')
        ids_source = 'saved-prediction-frame-ids'
    else:
        cfg_path = op.join(op.dirname(path), 'config.json')
        if not op.exists(cfg_path):
            raise ValueError('Legacy predictions need config.json to identify their frame IDs')
        import json
        with open(cfg_path) as f:
            cfg = json.load(f)
        stride = cfg.get('hold_mode_interval', 5) if cfg.get('hold_mode') else 1
        start = cfg['start_frame']
        if not cfg.get('test_first_frame') and (cfg.get('bidirectional_optimization') or cfg.get('reverse_frame_order')):
            start = 0
        saved_ids = start + np.arange(N_saved) * stride
        ids_source = 'inferred-from-saved-config-legacy-output'
    out_dict['frame_ids'] = saved_ids[selected_frame_ids]
    out_dict['frame_ids_source'] = ids_source

    fnames = [f"data/sequence/frame_{int(i):04d}.jpg" for i in out_dict['frame_ids']]
    out_dict["fnames"] = fnames
    out_dict["full_seq_name"] = "sequence"


    out_dict["K"] = np.stack([camera_data['intrinsic']] * N)


    # optimized_mano_verts = original_mano_verts * mano_transforms['scales'] @ mano_transforms['rotations'][:, None, :, None] + mano_transforms['translations'][:, None, :]


    # A three-frame axis-angle array also has shape (3, 3). Modern exports
    # explicitly store native matrices; prefer that unambiguous field instead
    # of mistaking the vectors for a single matrix (zero vectors are singular).
    if 'rotation_matrices' in mano_transforms:
        rotations_original = mano_transforms['rotation_matrices']
        if (rotations_original.ndim != 3 or rotations_original.shape[1:] != (3, 3)
                or len(rotations_original) < N_saved
                or not np.isfinite(rotations_original).all()):
            raise ValueError('Invalid native MANO rotation_matrices shape/values')
        rotations_per_frame = True
    elif mano_transforms['rotations'].shape == (3, 3):
        # Single rotation matrix for all frames
        rotations = np.linalg.inv(mano_transforms['rotations'])  # (3, 3)
        rotations_per_frame = False
    elif mano_transforms['rotations'].shape == (3,):
        # Single rotation vector for all frames - need to convert to matrix
        rot_vec = torch.FloatTensor(mano_transforms['rotations'])
        rot_mat = rotation_vector_to_matrix_safe(rot_vec).cpu().numpy()
        rotations = np.linalg.inv(rot_mat)  # (3, 3)
        rotations_per_frame = False
    elif len(mano_transforms['rotations'].shape) == 3 and mano_transforms['rotations'].shape[1:] == (3, 3):
        # Per-frame rotation matrices (N, 3, 3)
        rotations_original = mano_transforms['rotations']
        rotations_per_frame = True
    elif len(mano_transforms['rotations'].shape) == 2 and mano_transforms['rotations'].shape[1] == 3:
        # Per-frame rotation vectors (N, 3) - need to convert to matrices
        rotations_vectors = mano_transforms['rotations']  # (N_original, 3)
        rotations_matrices = []
        for i in range(rotations_vectors.shape[0]):
            rot_vec = torch.FloatTensor(rotations_vectors[i])
            rot_mat = rotation_vector_to_matrix_safe(rot_vec).cpu().numpy()
            rotations_matrices.append(rot_mat)
        rotations_original = np.stack(rotations_matrices, axis=0)  # (N_original, 3, 3)
        rotations_per_frame = True
    else:
        raise ValueError(f"Unexpected rotations shape: {mano_transforms['rotations'].shape}")

    # Sample data according to selected_frame_ids
    optimized_mano_verts_sampled = optimized_mano_verts[selected_frame_ids]
    optimized_mano_joints_sampled = optimized_mano_joints[selected_frame_ids]

    # Handle translations - could be per-frame or constant
    if mano_transforms['translations'].ndim == 2 and mano_transforms['translations'].shape[0] >= selected_frame_ids.max() + 1:
        translations_sampled = mano_transforms['translations'][selected_frame_ids]
    else:
        translations_sampled = mano_transforms['translations']

    # Apply inverse transformations to get original MANO output
    if rotations_per_frame:
        # Per-frame rotations: need to sample and invert per frame
        rotations_sampled = rotations_original[selected_frame_ids]  # (N, 3, 3)
        rotations_inv = np.linalg.inv(rotations_sampled)  # (N, 3, 3)

        # Apply per-frame inverse transform
        original_mano_verts = np.zeros_like(optimized_mano_verts_sampled.squeeze(1))
        original_mano_joints = np.zeros_like(optimized_mano_joints_sampled.squeeze(1))

        for i in range(N):
            original_mano_verts[i] = (optimized_mano_verts_sampled[i, 0] - translations_sampled[i]) @ rotations_inv[i] / mano_transforms['scales']
            original_mano_joints[i] = (optimized_mano_joints_sampled[i, 0] - translations_sampled[i]) @ rotations_inv[i] / mano_transforms['scales']
    else:
        # Single rotation for all frames
        original_mano_verts = (optimized_mano_verts_sampled.squeeze(1) - translations_sampled[:, None, :]) @ rotations / mano_transforms['scales']
        original_mano_joints = (optimized_mano_joints_sampled.squeeze(1) - translations_sampled[:, None, :]) @ rotations / mano_transforms['scales']

    out_dict["ori_v3d_c.right"] = original_mano_verts
    out_dict["ori_j3d_c.right"] = original_mano_joints


    original_mano_root = original_mano_joints[:, 0, :]
    optimized_mano_root = optimized_mano_joints_sampled.squeeze(1)[:, 0, :]


    # out_dict["v3d_ra.right"] = original_mano_verts - original_mano_root[:, None, :]
    # out_dict["j3d_ra.right"] = original_mano_joints - original_mano_root[:, None, :]


    mano_scale = mano_transforms['scales']
    out_dict["v3d_c.right"] = optimized_mano_verts_sampled.squeeze(1)  / mano_scale # (N, 778, 3)
    out_dict["j3d_c.right"] = optimized_mano_joints_sampled.squeeze(1) / mano_scale # (N, 21, 3)

    out_dict["v3d_ra.right"] = (optimized_mano_verts_sampled.squeeze(1) - optimized_mano_root[:, None, :]) / mano_scale
    out_dict["j3d_ra.right"] = (optimized_mano_joints_sampled.squeeze(1) - optimized_mano_root[:, None, :]) / mano_scale


    object_mesh = trimesh.load_mesh(op.join(path, "object_mesh.obj"))
    if "object_scale" in object_poses:
        object_scale = object_poses["object_scale"]
        if object_scale.size == 1:
            object_mesh.apply_scale(float(object_scale.reshape(-1)[0]))
        else:
            scale_matrix = np.diag(np.append(object_scale, 1.0))
            object_mesh.apply_transform(scale_matrix)
    # obj_vertices, obj_faces = load_obj_with_trimesh(op.join(path, "object_mesh.obj"))

    object_poses_in_cam = object_poses['object_poses_in_cam']  # (N_original, 4, 4)
    object_poses_in_cam_sampled = object_poses_in_cam[selected_frame_ids]  # (N, 4, 4)

    transformed_vertices = []
    for i in range(object_poses_in_cam_sampled.shape[0]):
        mesh_frame = object_mesh.copy()
        mesh_frame.apply_transform(object_poses_in_cam_sampled[i])
        transformed_vertices.append(mesh_frame.vertices)

    obj_vertices = np.array(transformed_vertices)  # (N, V, 3)
    obj_faces = np.array(object_mesh.faces)  # (F, 3)


    out_dict["v3d_c.object"] = obj_vertices / mano_transforms['scales']
    out_dict["root.object"] = compute_bounding_box_centers(out_dict["v3d_c.object"])
    out_dict["v3d_ra.object"] = (out_dict["v3d_c.object"] - out_dict["root.object"][:, None, :])

    # Apply inverse transformation to object vertices
    if rotations_per_frame:
        ori_obj_vertices = np.zeros_like(obj_vertices)
        for i in range(N):
            ori_obj_vertices[i] = (obj_vertices[i] - translations_sampled[i]) @ rotations_inv[i] / mano_transforms['scales']
    else:
        ori_obj_vertices = (obj_vertices - translations_sampled[:, None, :]) @ rotations / mano_transforms['scales']

    out_dict["ori_v3d_c.object"] = ori_obj_vertices

    # object: relative to right hand
    out_dict["v3d_right.object"] = out_dict["v3d_c.object"] - optimized_mano_root[:, None, :] / mano_transforms['scales']  # (N, V, 3)
    # out_dict["v3d_right.object"] = (out_dict["v3d_c.object"] - optimized_mano_root[:, None, :]) / mano_transforms['scales']  # (N, V, 3)

    out_dict["faces"] = {'right':mano_faces, 'object':obj_faces}

    out_dict = out_dict.to_torch()
    return out_dict, N




def load_gt_data(seq_name, start_frame=0, end_frame=-1, hold_mode=False, device='cpu'):
    """Load camera-space GT with a correct MANO rest-wrist camera transform."""
    if start_frame < 0 or (hold_mode and start_frame % 5):
        raise ValueError("Invalid GT start frame")
    train_dir, eval_dir = f"{HO3D_ROOT}/train/{seq_name}", f"{HO3D_ROOT}/evaluation/{seq_name}"
    data_dir = train_dir if os.path.exists(train_dir) else eval_dir
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Sequence {seq_name} not found")
    reader = Ho3dReader(data_dir)
    if end_frame == -1:
        end_frame = len(reader)
    if not start_frame < end_frame <= len(reader):
        raise ValueError("GT frame interval is empty or exceeds the sequence")
    from optimization.models.mano_wrapper import MANO
    with open(op.join(HO3D_MANO_ROOT, "mano_config.yaml")) as f:
        cfg = yaml.safe_load(f)["MANO"]
    model = MANO(**{k.lower(): v for k, v in cfg.items()}).to(device)
    ids = np.arange(start_frame, end_frame, 5 if hold_mode else 1)
    mesh = reader.get_gt_mesh()
    betas, poses, translations, objects, valid = [], [], [], [], []
    for idx in ids:
        hand = reader.get_mano_data(int(idx), mano_model=model)
        object_pose = reader.get_gt_pose(int(idx))
        ok = hand is not None and object_pose is not None
        valid.append(int(ok))
        if ok:
            betas.append(hand['handBeta'])
            poses.append(hand['handPose'].reshape(16, 3))
            translations.append(np.asarray(hand['handTrans']).reshape(3))
            objects.append(mesh.copy().apply_transform(object_pose).vertices.astype(np.float32))
        else:
            # A first-frame invalid annotation must never borrow an undefined
            # or stale translation from a different frame.
            betas.append(np.zeros(10, dtype=np.float32))
            poses.append(np.zeros((16, 3), dtype=np.float32))
            translations.append(np.zeros(3, dtype=np.float32))
            objects.append(np.zeros_like(mesh.vertices, dtype=np.float32))
    rotations = so3_exp_batch(torch.as_tensor(np.asarray(poses), dtype=torch.float32, device=device))
    with torch.no_grad():
        output = model(betas=torch.as_tensor(np.asarray(betas), dtype=torch.float32, device=device),
                       global_orient=rotations[:, :1], hand_pose=rotations[:, 1:],
                       transl=torch.as_tensor(np.asarray(translations), dtype=torch.float32, device=device))
    vertices, joints = output.vertices.cpu(), output.joints.cpu()
    object_vertices = torch.as_tensor(np.asarray(objects), dtype=torch.float32)
    wrist = joints[:, :1]
    object_roots = torch.as_tensor(compute_bounding_box_centers(object_vertices.numpy()))
    cameras = torch.as_tensor(reader.K, dtype=torch.float32)[None].repeat(len(ids), 1, 1)
    out = {'fnames': [reader.color_files[int(i)] for i in ids],
           'frame_ids': np.asarray([int(reader.id_strs[int(i)]) for i in ids]),
           'v3d_c.right': vertices, 'j3d_c.right': joints,
           'v3d_ra.right': vertices - wrist, 'j3d_ra.right': joints - wrist,
           'v3d_c.object': object_vertices,
           'v3d_ra.object': object_vertices - object_roots[:, None],
           'v3d_right.object': object_vertices - wrist, 'root.object': object_roots,
           'v2d_h': project2d_batch(cameras, vertices),
           'j2d_h': project2d_batch(cameras, joints),
           'v2d_o': project2d_batch(cameras, object_vertices),
           'faces.right': model.faces, 'faces.object': mesh.faces, 'K': reader.K,
           'is_valid': torch.tensor(valid, dtype=torch.int32), 'full_seq_name': seq_name}
    return xdict(out).to_torch()
