"""Standalone bridge worker for HOI's existing gsplat environment (Python 3.10)."""

import json
import os
import sys
from pathlib import Path


def main(path: Path, stage: str):
    handoff = json.loads(path.read_text())
    config = handoff["config"]
    repository = Path(config["repository"])
    sys.path.insert(0, str(repository))
    os.environ.update(
        HO3D_MANO_ROOT=config["mano_root"],
        HO3D_ROOT=config["raw_data_root"],
        DINO_MODEL_PATH=config["dino_model_path"],
    )
    scene, split = handoff["scene"], handoff["split"]
    data_dir = Path(handoff["processed_data_root"]) / split / scene
    if stage == "preprocess":
        from preprocessing.preprocess_ho3d import HO3DConfig, HO3DPreprocessor

        cfg = HO3DConfig(
            mano_model_root=config["mano_root"],
            raw_data_root=config["raw_data_root"],
            processed_data_root=handoff["processed_data_root"],
            generated_mesh_root=handoff["generated_mesh_root"],
            wilor_path=config["wilor_path"],
            moge_path=config["moge_path"],
            moge_env=config.get("moge_env") or "moge",
            foundationpose_path=config["foundationpose_path"],
            wilor_env=config["wilor_env"],
            foundationpose_env=config["foundationpose_env"],
            hoi_env=config["hoi_env"],
            max_frames=config["max_frames"],
        )
        if not HO3DPreprocessor(cfg).process_scene(scene, split):
            raise RuntimeError("HO3D preprocessing reported failure")
        from preprocessing.layout import generated_object_directory, object_pose_directory
        for relative in (
            "rgb", "wilor", "moge/cam_K.txt", "hand_processed",
        ):
            if not (data_dir / relative).exists():
                raise RuntimeError(f"Preprocessing did not produce {relative}")
        # Keep the selected, post-Blender mesh/UV/texture before the optimizer
        # constructs its lightweight proxy. No video texture fitting is enabled.
        pose_dir = object_pose_directory(data_dir)
        mesh_dir = generated_object_directory(Path(handoff['generated_mesh_root']) / split / scene)
        surface_path = mesh_dir / 'surface.json'
        if surface_path.is_file():
            surface = json.loads(surface_path.read_text())
            if not surface.get('blender') or not surface.get('target_triangles'):
                raise RuntimeError('Selected surface has no valid Blender processing record')
            from optimization.data.object_assets import prepare_object_asset, read_object_asset, load_mesh, file_hash
            selected_hash = handoff.get('asset', {}).get('mesh_sha256')
            if selected_hash and file_hash(mesh_dir/'textured_mesh.glb') != selected_hash:
                raise RuntimeError('Staged final asset differs from the selected generation delivery')
            canonical = pose_dir / 'scaled_mesh.obj'
            asset_path = data_dir / 'object_asset' / 'asset.json'
            if asset_path.exists():
                _, saved = read_object_asset(asset_path, load_mesh(canonical))
                if saved['sha256']['source/textured_mesh.glb'] != file_hash(mesh_dir/'textured_mesh.glb'):
                    raise RuntimeError('Registered object asset belongs to a different generation delivery')
            else:
                prepare_object_asset(mesh_dir / 'textured_mesh.glb', canonical, asset_path.parent,
                    stage='blender-final', provenance={'generation_selection': handoff.get('asset'), 'surface': surface},
                    supporting_files=[p for p in (surface_path, mesh_dir/'provenance.json', mesh_dir/'texture-provenance.json') if p.is_file()])
    elif stage == "train":
        from optimization.config.config import Config
        from optimization.pipelines.trainer import Runner
        from optimization.run_ho3d import configuration

        if split != 'train':
            raise ValueError('The released HO3D configuration supports the 18 training sequences')
        values = configuration(scene, handoff['processed_data_root'], config['mano_root'], handoff['result_dir'])
        frames = json.loads((data_dir/'frames.json').read_text())['frame_ids']
        if values['start_frame'] not in frames:
            raise ValueError('The prepared clip does not include the preset contact frame')
        result_dir = Path(handoff['result_dir'])
        if result_dir.exists() and any(result_dir.iterdir()):
            raise FileExistsError('Use a new result directory; existing results are preserved')
        Runner(Config(**values)).train()
        if not (Path(handoff["result_dir"]) / "aligned").is_dir():
            raise RuntimeError("Training did not produce aligned outputs")
    elif stage == "evaluate":
        from scripts.experiments.evaluate_protocol import evaluate_sequence

        manifest = json.loads((repository/'configs/ho3d_evaluation.json').read_text())
        result = evaluate_sequence('ho3d', scene, Path(handoff['result_dir']), manifest)
        if not result.get('success'):
            raise RuntimeError("HO3D evaluation failed")

        def serializable(value):
            if hasattr(value, "tolist"):
                return value.tolist()
            raise TypeError(type(value).__name__)

        (path.parent / "metrics.json").write_text(
            json.dumps(result, default=serializable, indent=2, allow_nan=False)
        )
    else:
        raise ValueError("Unknown HOI stage")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])
