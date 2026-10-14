"""Resolve public HO3D paths while accepting the first data release."""
from pathlib import Path

OBJECT_POSE_DIR = 'object_pose'
GENERATED_OBJECT_DIR = 'object'
_LEGACY_POSE_DIR = 'foundationpose/foundationpose_moge_scaled_nano_final_preg_5_30.0_global_scaleall'
_LEGACY_OBJECT_DIR = 'nano_final'


def _resolve(root, current, legacy):
    root = Path(root)
    for relative in (current, legacy):
        path = root / relative
        if path.is_dir():
            return path
    raise FileNotFoundError(f'Missing prepared data directory: {root / current}')


def object_pose_directory(sequence_dir):
    """Prefer the public layout; fall back only when that directory is absent."""
    return _resolve(sequence_dir, OBJECT_POSE_DIR, _LEGACY_POSE_DIR)


def generated_object_directory(scene_dir):
    return _resolve(scene_dir, GENERATED_OBJECT_DIR, _LEGACY_OBJECT_DIR)
