"""Prepare one or more HO3D scenes using the public preprocessing forks."""
from dataclasses import dataclass
from pathlib import Path
import json
import os
import re
import shutil
import subprocess
import sys

from .check_inputs import check_sequence
from .layout import OBJECT_POSE_DIR, generated_object_directory

PROJECT = Path(__file__).resolve().parents[1]


@dataclass
class HO3DConfig:
    raw_data_root: str
    processed_data_root: str
    generated_mesh_root: str
    mano_model_root: str
    wilor_path: str
    moge_path: str
    foundationpose_path: str
    wilor_env: str = 'wilor'
    moge_env: str = 'moge'
    foundationpose_env: str = 'foundationpose'
    hoi_env: str = 'agile-optim'
    max_frames: int = 2048
    scenes: tuple[str, ...] = ()
    """Explicit split/scene names. An empty list processes all available scenes."""


def indexed_images(directory):
    paths = [p for p in Path(directory).iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}]
    result = {}
    for path in paths:
        if not path.stem.isdigit():
            raise ValueError(f'Expected a numeric frame ID: {path.name}')
        frame_id = int(path.stem)
        if frame_id in result:
            raise ValueError(f'Duplicate frame ID: {frame_id}')
        result[frame_id] = path
    if not result:
        raise ValueError(f'No frames in {directory}')
    return result


class HO3DPreprocessor:
    def __init__(self, config: HO3DConfig):
        if config.max_frames <= 0:
            raise ValueError('max_frames must be positive')
        self.config = config
        for field in ('raw_data_root', 'processed_data_root', 'generated_mesh_root',
                      'mano_model_root', 'wilor_path', 'moge_path', 'foundationpose_path'):
            setattr(config, field, str(Path(getattr(config, field)).expanduser().resolve()))

    def _run(self, environment, arguments, cwd=PROJECT):
        subprocess.run(['conda', 'run', '--no-capture-output', '-n', environment, 'python', *map(str, arguments)],
                       cwd=cwd, env={**os.environ, 'PYTHONNOUSERSITE': '1'}, check=True)

    def prepare_inputs(self, scene, split):
        if split not in ('train', 'evaluation') or not re.fullmatch(r'[A-Za-z0-9_-]+', scene):
            raise ValueError('Expected an HO3D train/evaluation split and a scene name')
        raw = Path(self.config.raw_data_root)
        sequence = Path(self.config.processed_data_root) / split / scene
        sources = {'rgb': raw / split / scene / 'rgb',
                   'mask/hand': raw / 'masks_SAM2' / split / scene / 'hand',
                   'mask/object': raw / 'masks_SAM2' / split / scene / 'object'}
        frames = {key: indexed_images(path) for key, path in sources.items()}
        ids = sorted(frames['rgb'])[:self.config.max_frames]
        if ids != list(range(len(ids))):
            raise ValueError('MoGe requires contiguous HO3D frame IDs starting at zero')
        for key in frames:
            if any(i not in frames[key] for i in ids):
                raise ValueError(f'Missing selected frame IDs in {key}')
        if sequence.exists() and any(sequence.iterdir()):
            raise FileExistsError(f'Use a fresh processed scene directory: {sequence}')
        # Check every selected image before creating any output.
        from PIL import Image
        for i in ids:
            with Image.open(frames['rgb'][i]) as image:
                size = image.size
            for key in ('mask/hand', 'mask/object'):
                with Image.open(frames[key][i]) as mask:
                    if mask.size != size:
                        raise ValueError(f'{key} frame {i} has a different image size')
        for key, items in frames.items():
            target = sequence / key
            target.mkdir(parents=True, exist_ok=True)
            for i in ids:
                source = items[i]
                shutil.copy2(source, target / f'{i:05d}{source.suffix.lower()}')
        (sequence / 'frames.json').write_text(json.dumps({'dataset': 'HO3D', 'split': split,
            'scene': scene, 'frame_ids': ids, 'source_rgb_names': [frames['rgb'][i].name for i in ids]}, indent=2)+'\n')
        return sequence

    def process_scene(self, scene, split):
        cfg = self.config
        mesh = generated_object_directory(Path(cfg.generated_mesh_root) / split / scene) / 'textured_mesh.obj'
        if not mesh.is_file():
            raise FileNotFoundError(f'Missing selected generated mesh: {mesh}')
        # FoundationPose's upstream implementation still interpolates paths into a shell.
        for value in (str(mesh), cfg.processed_data_root, cfg.foundationpose_path):
            if not re.fullmatch(r'[A-Za-z0-9_./-]+', value):
                raise ValueError('FoundationPose requires paths without spaces or shell characters')
        sequence = self.prepare_inputs(scene, split)
        self._run(cfg.wilor_env, ['demo_vid.py', '--img_folder', sequence / 'rgb',
                                 '--out_folder', sequence / 'wilor_raw'], cfg.wilor_path)
        self._run(cfg.hoi_env, ['-m', 'preprocessing.filter_wilor', '--sequence-dir', sequence,
                               '--input-dir', sequence / 'wilor_raw', '--output-dir', sequence / 'wilor'])
        self._run(cfg.moge_env, [PROJECT / 'preprocessing/export_moge.py', '--moge-repo', cfg.moge_path,
                                '--version', 'v2', '--img-path', sequence / 'rgb',
                                '--outdir', sequence, '--scene-name', 'moge'])
        self._run(cfg.hoi_env, ['-m', 'preprocessing.process_hand', '--data-dir', sequence,
                               '--mano-model-path', cfg.mano_model_root])
        output = sequence / OBJECT_POSE_DIR
        output.parent.mkdir(exist_ok=True)
        (Path(cfg.foundationpose_path) / 'tmp').mkdir(exist_ok=True)
        self._run(cfg.foundationpose_env, ['run_demo.py', '--mesh_file', mesh,
            '--depth_dir', sequence / 'moge', '--rgb_dir', sequence / 'rgb',
            '--mask_dir', sequence / 'mask', '--debug_dir', output,
            '--scale_mesh', '--periodic_register', '--register_interval', '5',
            '--register_cone_angle', '30.0', '--global_register_only', '--scale_all'], cfg.foundationpose_path)
        report = check_sequence(sequence, output)
        (sequence / 'preprocessing.json').write_text(json.dumps(report, indent=2)+'\n')
        return True

    def process_all_scenes(self):
        scenes = list(self.config.scenes)
        if not scenes:
            raw = Path(self.config.raw_data_root)
            scenes = [f'{split}/{p.name}' for split in ('train', 'evaluation')
                      if (raw / split).is_dir() for p in sorted((raw / split).iterdir())
                      if (p / 'rgb').is_dir()]
        if not scenes:
            raise ValueError('No HO3D scenes found')
        for item in scenes:
            parts = item.split('/')
            if len(parts) != 2:
                raise ValueError('Each scene must have the form train/SCENE or evaluation/SCENE')
            self.process_scene(parts[1], parts[0])


def main():
    import tyro
    HO3DPreprocessor(tyro.cli(HO3DConfig)).process_all_scenes()


if __name__ == '__main__':
    main()
