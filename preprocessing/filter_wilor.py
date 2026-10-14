"""Normalize the fork's detections into one selected hand without overwriting raw outputs."""
import argparse
from pathlib import Path
import numpy as np

from preprocessing.single_hand import require_one_hand_axis, single_hand_mask_directory

FIELDS = ('mano_verts', 'mano_joints', 'mano_joints_2d', 'mano_global_orient',
          'mano_hand_pose', 'mano_betas', 'mano_masks')


def filter_sequence(sequence_dir, input_dir, output_dir):
    sequence = Path(sequence_dir).resolve()
    source, target = Path(input_dir).resolve(), Path(output_dir).resolve()
    masks = single_hand_mask_directory(sequence / 'mask')
    if source == target or source in target.parents or target in source.parents:
        raise ValueError('Raw and filtered WiLoR directories must be separate siblings')
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f'Filtered output already contains files; use a new output directory: {target}')
    mask_count = len(list(masks.glob('*.png')))
    if mask_count == 0:
        raise ValueError('Selected-hand mask directory contains no PNG frames')
    from optimization.data.loader import _load_npz_array
    arrays = {}
    for field in FIELDS:
        # The public fork stores 2D joints as one pickled Torch tensor per
        # frame/hand. Expand those payload dimensions before selecting a hand.
        arrays[field] = _load_npz_array(str(source / (field + '.npz')), field)
        if len(arrays[field]) != mask_count:
            raise ValueError(f'{field} frame count does not match the selected-hand masks')
    from preprocessing.hand_selection import HandSelector
    target.mkdir(parents=True, exist_ok=True)
    for field in FIELDS:
        np.savez(target / (field + '.npz'), **{field: arrays[field]})
    processor = HandSelector()
    if not processor.filter_wilor_by_hand_mask(str(target), str(sequence / 'mask')):
        raise RuntimeError('Single-hand selection failed; raw WiLoR outputs are preserved')
    for field in FIELDS[:-1]:
        with np.load(target / (field + '.npz'), allow_pickle=True) as data:
            require_one_hand_axis(data[field], field)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence-dir', type=Path, required=True)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(filter_sequence(args.sequence_dir, args.input_dir, args.output_dir))


if __name__ == '__main__':
    main()
