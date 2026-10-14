"""The real bridge registers the selected post-Blender asset before optimization."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from optimization.data.object_assets import file_hash, read_object_asset, load_mesh
from tests.test_object_assets import textured_box


class AssetBridgeTests(unittest.TestCase):
    def test_preprocessing_registers_and_verifies_selected_delivery(self):
        repository=Path(__file__).resolve().parents[1]
        spec=importlib.util.spec_from_file_location('asset_bridge_worker',repository/'generation/src/agile3d/hoi_worker.py')
        worker=importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); data=root/'processed/train/SEQ'; meshes=root/'meshes/train/SEQ/object'
            meshes.mkdir(parents=True)
            mesh=textured_box(); mesh.export(meshes/'textured_mesh.glb')
            (meshes/'surface.json').write_text(json.dumps(dict(blender='4.5.4',target_triangles=len(mesh.faces))))
            for name in ('rgb','wilor','hand_processed'):(data/name).mkdir(parents=True)
            (data/'moge').mkdir(); (data/'moge/cam_K.txt').write_text('fixture')
            canonical=data/'object_pose/scaled_mesh.obj'
            canonical.parent.mkdir(parents=True)
            mesh.apply_scale(.8); mesh.export(canonical)
            config={key:'fixture' for key in ('mano_root','raw_data_root','dino_model_path','wilor_path','foundationpose_path',
                                             'wilor_env','megasam_env','foundationpose_env','hoi_env')}
            config.update(repository=str(repository),max_frames=2,moge_path="fixture")
            handoff=dict(config=config,scene='SEQ',split='train',generated_mesh_root=str(root/'meshes'),
                         processed_data_root=str(root/'processed'),asset={'mesh_sha256':file_hash(meshes/'textured_mesh.glb')})
            request=root/'handoff.json'; request.write_text(json.dumps(handoff))
            preprocessor=SimpleNamespace(HO3DConfig=lambda **kwargs: kwargs,
                HO3DPreprocessor=lambda cfg:SimpleNamespace(process_scene=lambda *a:True))
            with patch.dict(sys.modules,{'preprocessing.preprocess_ho3d':preprocessor}), patch.object(sys,'path',list(sys.path)), patch.dict('os.environ',{},clear=False):
                worker.main(request,'preprocess')
                manifest=data/'object_asset/asset.json'
                _, record=read_object_asset(manifest,load_mesh(canonical))
                self.assertEqual(record['stage'],'blender-final')
                self.assertEqual(record['sha256']['source/textured_mesh.glb'],handoff['asset']['mesh_sha256'])
                worker.main(request,'preprocess')  # repeat validates, without overwriting
                with (meshes/'textured_mesh.glb').open('ab') as stream:stream.write(b'changed')
                with self.assertRaisesRegex(RuntimeError,'differs from the selected'):
                    worker.main(request,'preprocess')


if __name__=='__main__':unittest.main()
