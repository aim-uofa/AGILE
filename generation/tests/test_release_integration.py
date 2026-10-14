"""Regression cases found while assembling the single-hand release."""
import json
import subprocess
import sys
import numpy as np
import pytest
import trimesh
from PIL import Image
from agile3d.hoi import HOIConfig, prepare_hoi
from agile3d.final_output import finalize_asset, prepare_delivery
from agile3d.store import digest
from agile3d.uv_orientation import artwork_orientation
from test_final_output import finish, add_variant, verdict


def hoi_config(tmp_path, monkeypatch, **changes):
    repo = tmp_path / 'optimization'; (repo / 'preprocessing').mkdir(parents=True)
    (repo / 'preprocessing/preprocess_ho3d.py').touch()
    monkeypatch.setattr(subprocess, 'check_output', lambda argv, **kw: 'test-revision\n' if 'rev-parse' in argv else '')
    (tmp_path / "demo_moge.py").touch()
    return HOIConfig(repository=str(repo), python=sys.executable, raw_data_root=str(tmp_path),
        mano_root=str(tmp_path), dino_model_path=str(tmp_path), wilor_path=str(tmp_path),
        moge_path=str(tmp_path), foundationpose_path=str(tmp_path), **changes)


def test_handoff_uses_selected_prior_mesh_and_requires_explicit_best_available(sample, tmp_path, monkeypatch):
    store, jid, _, source = sample
    job = finish(sample)
    add_variant(store, job)
    request = prepare_delivery(store, jid)
    chosen = request['candidates'][0]['id']
    delivery = finalize_asset(store, jid, selection=verdict(request, chosen))
    assert delivery['quality_status'] == 'best_available'
    cfg = hoi_config(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='allow_best_available'):
        prepare_hoi(store, jid, cfg, 'train', 'SYNTHETIC')
    cfg = cfg.model_copy(update={'allow_best_available': True})
    handoff = prepare_hoi(store, jid, cfg, 'train', 'SYNTHETIC')
    copied = __import__('pathlib').Path(handoff['generated_mesh_root']) / 'train/SYNTHETIC/object/textured_mesh.glb'
    assert digest(copied) == digest(source)
    assert handoff['asset']['selected_id'] == chosen
    assert handoff['asset']['quality_status'] == 'best_available'


@pytest.mark.parametrize('up', ['Z', 'Y'])
def test_inverted_uv_directions_are_measured_in_export_coordinate_system(tmp_path, up):
    vertices = np.array([[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]], dtype=float)
    if up == 'Y':
        vertices = vertices @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]).T
    uv = np.array([[.9, .9], [.1, .9], [.1, .1], [.9, .1]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2], [0, 2, 3]], process=False)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, image=Image.new('RGB', (8, 8), 'white'))
    path = tmp_path / 'prepared.glb'; mesh.export(path)
    before = digest(path)
    record = {'up_axis_in_export': up, 'islands': [{'id': 1, 'direction': 'front -Y',
        'uv_bounds': [[.1, .1], [.9, .9]], 'polygons': [uv.tolist()]}]}
    result = artwork_orientation(path, record)
    assert result[0]['artwork_top_toward_image'] == 'DOWN'
    assert result[0]['artwork_right_toward_image'] == 'LEFT'
    assert result[0]['planar_fit_relative_rms'] < 1e-6
    assert digest(path) == before
