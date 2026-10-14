import json
import os
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from agile3d.codex_backend import CodexBackend, MeshComponent, ProceduralMeshSpec
from agile3d.mesh import load_scene, parts
from agile3d.models import PipelineConfig, Review, SurfaceConfig
from agile3d.pipeline import create_job, queue_candidate, provide_texture, submit_geometry_review, verify_evidence, submit_review, export_asset
from agile3d.procedural import _shape, build_mesh
from agile3d.store import Store, digest
from agile3d.surface import apply_uv_texture, white_mesh, prepare_surface, mesh_uv_request


def component(**kwargs):
    return MeshComponent(name='body', kind='box', position=[0.,0.,0.], rotation_deg=[0.,0.,0.],
        size=[.4,.25,1.], color=[1.,1.,1.,1.], profile=[], torus_ratio=.2,
        label_text='', label_color=[1.,1.,1.,1.], texture_projection='box', **kwargs)


def test_loft_has_independent_depth_and_width_with_closed_rounded_surface():
    values=component().model_dump()
    values.update(kind='loft', sections=[[-.5,0.,0.,.4,.25,2.8],[-.2,.02,0.,.46,.42,2.6],
        [.2,.03,0.,.4,.32,2.4],[.5,.08,0.,.16,.15,2.]])
    shape=_shape(MeshComponent.model_validate(values))
    assert shape.is_watertight and shape.is_winding_consistent and shape.volume > 0
    middle=shape.vertices[np.abs(shape.vertices[:,2]+.18)<.04]
    top=shape.vertices[shape.vertices[:,2]>.48]
    assert np.ptp(middle[:,1]) > 2*np.ptp(top[:,1])
    assert len(shape.vertices)<10000
    values['sections'][1][0]=-.5
    with pytest.raises(ValueError,match='strictly upward'):MeshComponent.model_validate(values)


def test_apply_generated_map_preserves_existing_mesh_uv_and_positions(tmp_path):
    spec=ProceduralMeshSpec(schema_version=1,object_name='box',approximation_scope='Synthetic textured box for UV preservation.',assumptions=[],components=[component()])
    old=tmp_path/'old.png';Image.new('RGB',(768,512),'red').save(old)
    mesh=tmp_path/'prepared.glb';build_mesh(spec,mesh,old)
    atlas=tmp_path/'new.png';Image.new('RGB',(1024,1024),(21,63,185)).save(atlas)
    output=tmp_path/'new.glb';apply_uv_texture(mesh,atlas,output)
    before,after=parts(load_scene(mesh))[0],parts(load_scene(output))[0]
    assert np.array_equal(before.faces,after.faces)
    assert np.array_equal(before.vertices,after.vertices)
    assert np.array_equal(before.visual.uv,after.visual.uv)
    assert np.array_equal(np.asarray(after.visual.material.baseColorTexture),np.asarray(Image.open(atlas)))
    Image.new('RGB',(1024,768),'red').save(atlas)
    with pytest.raises(ValueError,match='square'):apply_uv_texture(mesh,atlas,output)


def test_white_gate_and_frozen_uv_texture_resume_no_geometry_recall(tmp_path,monkeypatch):
    calls=[]
    spec=ProceduralMeshSpec(schema_version=1,object_name='box',approximation_scope='Synthetic geometry for the new staged workflow.',assumptions=[],components=[component()])
    def propose(self,*args,**kwargs):
        assert kwargs['mesh_uv']
        calls.append('geometry');return spec,{'provider':'test'}
    def surface(mesh,directory,config,**kwargs):
        calls.append('surface');directory.mkdir()
        atlas=directory/'baked-albedo.png';Image.new('RGB',(768,512),'white').save(atlas)
        build_mesh(spec,directory/'prepared.glb',atlas)
        white_mesh(directory/'prepared.glb',directory/'white.glb')
        Image.new('RGB',(1024,1024),'white').save(directory/'uv-layout.png')
        Image.new('RGB',(768,560),'white').save(directory/'white-views.png')
        (directory/'surface.blend').write_bytes(b'test placeholder')
        record={'islands':[],'texture_baked':False}
        (directory/'surface.json').write_text(json.dumps(record));return record
    monkeypatch.setattr(CodexBackend,'propose_mesh',propose)
    monkeypatch.setattr('agile3d.surface.prepare_surface',surface)
    config=PipelineConfig.model_validate({'image':{'provider':'disabled'},'mesh':{'backend':'codex'},
        'surface':{'enabled':True},'texture':{'provider':'provided','layout':'mesh_uv'}})
    reference=tmp_path/'reference.png';Image.new('RGB',(100,100),'white').save(reference)
    store=Store(tmp_path/'runs');job=create_job(store,[reference],'test box',config)
    job=queue_candidate(store,job['id'],background=False)
    assert job['status']=='awaiting_geometry_review',job
    atlas=tmp_path/'atlas.png';Image.new('RGB',(512,512),'blue').save(atlas)
    with pytest.raises(ValueError,match='awaiting'):provide_texture(store,job['id'],atlas)
    review=Review(evidence_digest=job['attempts'][0]['evidence_digest'],decision='accept',score=8.,
        matches_reference=True,no_hand_geometry=True,consistent_views=True,feedback='synthetic test')
    with pytest.raises(ValueError,match='stale'):submit_geometry_review(store,job['id'],review.model_copy(update={'evidence_digest':'0'*64}))
    job=submit_geometry_review(store,job['id'],review)
    assert job['status']=='awaiting_texture',job
    art=job['attempts'][0]['artifacts'];provenance=tmp_path/'provenance.json'
    metadata={'image_sha256':digest(atlas),'prepared_mesh_sha256':art['prepared_mesh']['sha256'],
              'uv_layout_sha256':art['uv_layout']['sha256']}
    provenance.write_text(json.dumps(dict(metadata,uv_layout_sha256='wrong')))
    with pytest.raises(ValueError,match='frozen'):provide_texture(store,job['id'],atlas,provenance)
    provenance.write_text(json.dumps(metadata))
    job=provide_texture(store,job['id'],atlas,provenance)
    assert job['status']=='awaiting_review',job
    assert calls==['geometry','surface']
    verify_evidence(store,job)
    job=submit_review(store,job['id'],review.model_copy(update={'evidence_digest':job['attempts'][0]['evidence_digest']}))
    exported=export_asset(store,job['id'])
    names={Path(r['path']).name for r in exported['files']}
    assert {'uv-layout.png','surface.json','texture-provenance.json'} <= names


def test_uv_mode_requires_blender_and_forbids_old_adaptive_panels():
    with pytest.raises(ValueError,match='Blender'):
        PipelineConfig.model_validate({'image':{'provider':'disabled'},'mesh':{'backend':'codex'},
            'texture':{'provider':'provided','layout':'mesh_uv'}})
    with pytest.raises(ValueError,match='mesh_uv'):
        PipelineConfig.model_validate({'image':{'provider':'disabled'},'mesh':{'backend':'codex'},
            'surface':{'enabled':True},'texture':{'provider':'provided'}})


def test_uv_request_disambiguates_image_origin_and_island_order(tmp_path):
    mesh=tmp_path/'mesh';mesh.write_bytes(b'mesh')
    guide=tmp_path/'guide';guide.write_bytes(b'guide')
    record={'islands':[{'id':1,'direction':'back +Y','uv_bounds':[[.1,.2],[.4,.8]],'polygons':[]}]}
    request=mesh_uv_request('carton',record,mesh,guide)
    assert request['islands'][0]['image_bounds_left_top_right_bottom']==pytest.approx([.1,.2,.4,.8])
    assert 'polygons' not in request['islands'][0]
    assert 'TOP LEFT' in request['prompt']


@pytest.mark.skipif(not os.environ.get('AGILE_TEST_BLENDER'),reason='Set AGILE_TEST_BLENDER for the real Blender regression')
@pytest.mark.parametrize('uv_method', ['semantic', 'smart_project'])
def test_real_blender_small_scale_quadriflow_and_metallic_albedo_bake(tmp_path, uv_method):
    # A fully metallic glTF has no diffuse contribution. Its base-color texture
    # must nevertheless survive remeshing and baking; scene lighting is irrelevant.
    mesh=trimesh.creation.icosphere(subdivisions=3,radius=.003)
    texture=Image.new('RGB',(512,512),(225,65,22))
    uv=np.column_stack((np.arctan2(mesh.vertices[:,1],mesh.vertices[:,0])/(2*np.pi)+.5,
                        mesh.vertices[:,2]/.006+.5))
    mesh.visual=trimesh.visual.texture.TextureVisuals(uv=uv,
        material=trimesh.visual.material.PBRMaterial(baseColorTexture=texture,metallicFactor=1.))
    source=tmp_path/'source.glb';mesh.export(source)
    record=prepare_surface(source,tmp_path/'surface',SurfaceConfig(
        enabled=True,executable=os.environ['AGILE_TEST_BLENDER'],method='quadriflow',
        target_faces=300,texture_size=512,uv_method=uv_method),up_axis='Z',bake=True)
    assert 100 < record['target_triangles'] < record['source_triangles']
    assert record['quad_polygons'] > 100
    assert record['temporary_working_scale'] > 1000
    assert record['texture_baked'] and record['bake_type']=='base_color_via_emission'
    colors=np.asarray(Image.open(tmp_path/'surface/baked-albedo.png').convert('RGB'))
    occupied=colors[colors[:,:,0]>100]
    assert len(occupied)>10000
    assert np.median(occupied,axis=0)==pytest.approx([225,65,22],abs=8)
    assert record['source_alignment_check']['p95_relative_diagonal'] < .03
    assert (tmp_path/'surface/white-oblique-views.png').is_file()
