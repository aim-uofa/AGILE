"""Optimize a HO3D sequence using the released configuration."""
import argparse
import json
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def configuration(scene, data_root, mano_root, output):
    scenes = json.loads((PROJECT/'configs/ho3d_scenes.json').read_text())
    if scene not in scenes:
        raise ValueError(f'Unknown released HO3D scene: {scene}')
    values = json.loads((PROJECT/'configs/ho3d.json').read_text())
    values.update(data_dir=str(Path(data_root).expanduser().resolve()/'train'/scene),
                  mano_model_path=str(Path(mano_root).expanduser().resolve()),
                  result_dir=str(Path(output).expanduser().resolve()),
                  start_frame=scenes[scene]['contact_frame'])
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--data-root', type=Path, required=True, help='Extracted HO3D/preprocessed directory')
    parser.add_argument('--mano-root', type=Path, default=os.environ.get('HO3D_MANO_ROOT', 'assets/mano'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true', help='Print the effective configuration without importing CUDA modules')
    args = parser.parse_args()
    values = configuration(args.scene, args.data_root, args.mano_root, args.output)
    if args.dry_run:
        print(json.dumps(values, indent=2))
        return
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use a new output directory; existing results will not be overwritten')
    for path in [Path(values['data_dir']), args.mano_root/'mano_config.yaml',
                 args.mano_root/'mano_hand.obj', args.mano_root/'finger_vertices_mask.npz']:
        if not path.exists():
            raise FileNotFoundError(path)
    from optimization.config.config import Config
    from optimization.pipelines.trainer import Runner
    Runner(Config(**values)).train()


if __name__ == '__main__':
    main()
