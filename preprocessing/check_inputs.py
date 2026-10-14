"""Read-only checks for the inputs consumed by the current single-hand optimizer."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from preprocessing.filter_wilor import FIELDS
from preprocessing.single_hand import require_one_hand_axis, single_hand_mask_directory


def frame_files(directory, suffixes):
    files = [p for p in Path(directory).iterdir() if p.suffix.lower() in suffixes]
    result = {}
    for path in files:
        if not path.stem.isdigit():
            raise ValueError(f'Frame names must be numeric: {path.name}')
        index = int(path.stem)
        if index in result:
            raise ValueError(f'Duplicate frame ID {index} in {directory}')
        result[index] = path
    return result


def check_sequence(sequence_dir, foundationpose_dir):
    root, pose_root = Path(sequence_dir), Path(foundationpose_dir)
    single_hand_mask_directory(root / 'mask')
    frames = frame_files(root / 'rgb', {'.png', '.jpg'})
    if not frames or sorted(frames) != list(range(len(frames))):
        raise ValueError('Use the complete zero-based contiguous frame sequence; do not silently renumber a dataset subset')
    expected = set(frames)
    for relative in ('mask/hand', 'mask/object', 'moge/depth'):
        other = frame_files(root / relative, {'.png'})
        if set(other) != expected:
            raise ValueError(f'{relative} frame IDs differ from RGB')
        for index, path in other.items():
            with Image.open(frames[index]) as rgb, Image.open(path) as target:
                if rgb.size != target.size:
                    raise ValueError(f'{relative}/{path.name} resolution differs from RGB')
    camera = np.loadtxt(root / 'moge/cam_K.txt')
    if camera.shape != (3, 3) or not np.isfinite(camera).all() or camera[0, 0] <= 0 or camera[1, 1] <= 0:
        raise ValueError('Invalid MoGe camera intrinsics')
    for field in FIELDS:
        with np.load(root / 'wilor' / (field + '.npz'), allow_pickle=True) as data:
            values = data[field]
            if len(values) != len(frames):
                raise ValueError(f'{field} frame count differs from RGB')
            if field != 'mano_masks':
                require_one_hand_axis(values, field)
    for filename, key in [('optimized_mano_verts.npz', 'mano_verts'),
                          ('optimized_mano_joints.npz', 'mano_joints')]:
        with np.load(root / 'hand_processed' / filename) as data:
            values = data[key]
            require_one_hand_axis(values, key)
            if len(values) != len(frames) or not np.isfinite(values).all():
                raise ValueError(f'Invalid aligned MANO array: {filename}')
    with np.load(root / 'hand_processed/mano_transforms.npz') as data:
        for key in ('scales', 'rotations', 'translations'):
            if not np.isfinite(data[key]).all():
                raise ValueError(f'Non-finite MANO transform: {key}')
    poses = frame_files(pose_root / 'ob_in_cam', {'.txt'})
    if set(poses) != expected:
        raise ValueError('FoundationPose frame IDs differ from RGB')
    for path in poses.values():
        pose = np.loadtxt(path)
        if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]):
            raise ValueError(f'Invalid object-to-camera transform: {path.name}')
    mesh = pose_root / 'scaled_mesh.obj'
    if not mesh.is_file() or not mesh.stat().st_size:
        raise FileNotFoundError('FoundationPose did not export scaled_mesh.obj')
    return {'input_contract_passed': True, 'frames': len(frames), 'selected_hands': 1,
            'scope': 'File/shape/frame checks only; inspect visual alignment separately'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence-dir', type=Path, required=True)
    parser.add_argument('--foundationpose-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check_sequence(args.sequence_dir, args.foundationpose_dir), indent=2))


if __name__ == '__main__':
    main()
