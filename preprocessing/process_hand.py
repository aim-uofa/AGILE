import os
import sys
import glob
import yaml
from pathlib import Path

import cv2
import trimesh
import torch
import open3d as o3d
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import tyro

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from optimization.models.mano_wrapper import MANO
from preprocessing.depth_contract import alignment_depth_mask

from optimization.utils.alignment import (
    compute_mano_normals,
    align_with_shared_scale_rotation,
    multi_frame_align_open3d,
    multi_frame_align_open3d_with_normals
)

def process_mano_simple(
    mano_data_path: Path,
    mano: MANO,
    depth_dir: Path,
    hand_mask_dir: Path,
    mano_model_path: Path,
) -> dict:
    'Align WiLoR MANO predictions to depth without normal supervision. Return aligned vertices, joints, shared scale/rotation, per-frame translations, faces, pose and shape parameters.'
    import yaml
    import trimesh
    from PIL import Image


    print("Loading depth images and camera intrinsics...")
    K = np.loadtxt(depth_dir.parent / "cam_K.txt", dtype=np.float32)
    depth_files = sorted(depth_dir.glob("*.png"))
    depths = np.array([
        cv2.imread(str(f), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0
        for f in depth_files
    ])
    num_frames = len(depths)


    print("Loading WiLoR MANO data...")
    mano_global_orient = np.load(mano_data_path / "mano_global_orient.npz", allow_pickle=True)["mano_global_orient"][:num_frames]
    mano_hand_pose = np.load(mano_data_path / "mano_hand_pose.npz", allow_pickle=True)["mano_hand_pose"][:num_frames]
    mano_betas = np.load(mano_data_path / "mano_betas.npz", allow_pickle=True)["mano_betas"][:num_frames]
    mano_masks = np.load(mano_data_path / "mano_masks.npz", allow_pickle=True)["mano_masks"][:num_frames]

    from preprocessing.single_hand import require_one_hand_axis
    for name, values in (("mano_global_orient", mano_global_orient),
                         ("mano_hand_pose", mano_hand_pose),
                         ("mano_betas", mano_betas)):
        require_one_hand_axis(values, name)

    mano_global_orient = mano_global_orient.astype(np.float32)
    mano_hand_pose = mano_hand_pose.astype(np.float32)
    mano_betas = mano_betas.astype(np.float32)
    mano_masks = mano_masks.astype(np.float32)

    print(f"Shape of mano_global_orient: {mano_global_orient.shape}")
    print(f"Shape of mano_hand_pose: {mano_hand_pose.shape}")
    print(f"Shape of mano_betas: {mano_betas.shape}")
    print(f"Shape of mano_masks: {mano_masks.shape}")


    print("Loading hand masks...")
    mask_files = sorted(hand_mask_dir.glob("*.png"))
    H, W = depths.shape[1:3]
    hand_masks = np.array([
        np.array(Image.open(f).resize((W, H), resample=Image.NEAREST))
        for f in mask_files
    ])


    print("Recomputing canonical MANO meshes...")
    mano_input = {
        "hand_pose": torch.from_numpy(mano_hand_pose.reshape(num_frames, -1, 3, 3)).float(),
        # "hand_pose": torch.from_numpy(mano_hand_pose).float(),
        "betas": torch.from_numpy(mano_betas.reshape(num_frames, -1)).float()
    }
    mano_output = mano(**mano_input, pose2rot=False)
    mano_verts = mano_output.vertices.detach().cpu().numpy().reshape(num_frames, 1, 778, 3)
    mano_joints = mano_output.joints.detach().cpu().numpy().reshape(num_frames, 1, 21, 3)


    print("Applying global orientations...")
    R = mano_global_orient[:, 0, 0]  # (B, 3, 3)
    wrist = mano_joints[:, 0, 0]
    mano_verts[:, 0] = (mano_verts[:, 0] - wrist[:, None, :]) @ R.transpose(0, 2, 1)
    mano_joints[:, 0] = (mano_joints[:, 0] - wrist[:, None, :]) @ R.transpose(0, 2, 1)


    print("Collecting correspondences for multi-frame alignment...")
    all_verts_wilor = []
    all_verts_mega = []

    for i in tqdm(range(num_frames), desc="Collecting correspondences"):

        depth = depths[i]
        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        homo_grid = np.stack([u + 0.5, v + 0.5, np.ones_like(u)], axis=-1).reshape(-1, 3)
        cam_points = (np.linalg.inv(K) @ homo_grid.T).T * depth.reshape(-1, 1)
        cam_coords = cam_points.reshape(h, w, 3)


        mano_mask = mano_masks[i] if mano_masks.ndim == 3 else mano_masks[i, 0]
        if mano_mask.shape != hand_masks[i].shape:
            mano_mask = np.array(Image.fromarray(mano_mask.astype(np.uint8)).resize(
                (hand_masks[i].shape[1], hand_masks[i].shape[0]), resample=Image.NEAREST
            ))

        valid_mask = alignment_depth_mask(depth, hand_masks[i], mano_mask)
        if np.count_nonzero(valid_mask) < 5:
            raise ValueError(
                f"Frame {i}: fewer than 5 valid depth pixels overlap the selected hand. "
                "Inspect depth, segmentation and WiLoR detection before MANO alignment."
            )


        all_verts_wilor.append(mano_verts[i, 0])
        all_verts_mega.append(cam_coords[valid_mask])


    s, R_align, ts = multi_frame_align_open3d(all_verts_wilor, all_verts_mega)


    for i in range(num_frames):
        mano_verts[i] = (s * mano_verts[i] @ R_align.T) + ts[i]
        mano_joints[i] = (s * mano_joints[i] @ R_align.T) + ts[i]


    mano_mesh = trimesh.load(mano_model_path / "mano_hand.obj", process=False)
    mano_faces = np.array(mano_mesh.faces)

    return {
        "mano_verts": mano_verts,
        "mano_joints": mano_joints,
        "mano_s": s,
        "mano_R": R_align,
        "mano_Ts": ts,
        "mano_faces": mano_faces,
        "mano_hand_pose": mano_hand_pose,
        "mano_betas": mano_betas,
        "mano_global_orient": mano_global_orient,
        "mano_masks": mano_masks,
    }


def process_mano_seq(
    data_dir: str,
    mano_model_path: Path = Path("assets/mano"),
):
    'Process one sequence and save its aligned single-hand arrays.'
    import torch
    torch.set_num_threads(1)

    seq_name = Path(data_dir).name

    if os.path.exists(Path(data_dir) / "hand_processed" / "optimized_mano_verts.npz"):
        print(f"Skipping {seq_name}, already processed.")
        return {"seq_name": seq_name, "status": "skipped"}

    try:
        print(f"Processing {seq_name}...")


        with open(mano_model_path / "mano_config.yaml") as f:
            cfg = yaml.safe_load(f)
        mano_cfg = {k.lower(): v for k, v in cfg["MANO"].items()}
        mano = MANO(**mano_cfg)


        mano_data_path = Path(data_dir) / "wilor"
        depth_dir = Path(data_dir) / "moge/depth"
        from preprocessing.single_hand import single_hand_mask_directory
        hand_mask_dir = single_hand_mask_directory(Path(data_dir) / "mask")

        result = process_mano_simple(
            mano_data_path=mano_data_path,
            mano=mano,
            depth_dir=depth_dir,
            hand_mask_dir=hand_mask_dir,
            mano_model_path=mano_model_path,
        )


        output_path = Path(data_dir) / "hand_processed"
        output_path.mkdir(exist_ok=True)
        np.savez(output_path / "optimized_mano_verts.npz", mano_verts=result["mano_verts"])
        np.savez(output_path / "optimized_mano_joints.npz", mano_joints=result["mano_joints"])
        np.savez(output_path / "mano_transforms.npz",
                 scales=result["mano_s"],
                 rotations=result["mano_R"],
                 translations=result["mano_Ts"])

        print(f"✅ {seq_name} completed")
        return {"seq_name": seq_name, "status": "success"}

    except Exception as e:
        print(f"❌ {seq_name} failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"seq_name": seq_name, "status": "failed", "error": str(e)}

if __name__ == "__main__":
    import tyro
    result = tyro.cli(process_mano_seq)
    if result.get("status") == "failed":
        raise SystemExit(1)
