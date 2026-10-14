"""Input contracts for the single-hand release."""
from pathlib import Path


def single_hand_mask_directory(mask_root):
    root = Path(mask_root)
    if any((root / name).exists() for name in ('hand1', 'hand2')):
        raise ValueError('This release accepts one selected hand in mask/hand; separate hand1/hand2 masks are unsupported')
    hand = root / 'hand'
    if not hand.is_dir():
        raise FileNotFoundError(f'Missing selected-hand mask directory: {hand}')
    return hand


def require_one_hand_axis(array, name):
    """Validate an explicit (frames, hands, ...) tensor without discarding hands."""
    if array.ndim < 3 or array.shape[1] != 1:
        raise ValueError(f'{name} must contain exactly one selected hand on axis 1; received shape {array.shape}')
    return array
