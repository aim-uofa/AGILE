import json

import numpy as np
import pytest
from PIL import Image

from agile3d.codex_backend import CodexBackend, MeshComponent, ProceduralMeshSpec
from agile3d.mesh import load_scene, parts, inspect_mesh, _colors
from agile3d.models import PipelineConfig, ImageConfig, MeshConfig, TextureConfig, QualityPolicy
from agile3d.pipeline import create_job, queue_candidate, provide_texture, verify_evidence
from agile3d.procedural import build_mesh
from agile3d.store import Store, digest
from agile3d.texture import validate_rectangles


def specification(kind="box", projection="box", profile=None):
    return ProceduralMeshSpec(schema_version=1, object_name="mapping test",
        approximation_scope="Synthetic structural geometry with a generated albedo atlas.", assumptions=[],
        components=[MeshComponent(name="body",kind=kind,position=[0.,0.,0.],rotation_deg=[0.,0.,0.],
            size=[.8,.5,1.],color=[.5,.5,.5,1.],profile=profile or [],torus_ratio=.2,
            label_text="",label_color=[1.,1.,1.,1.],texture_projection=projection)])


def atlas_file(tmp_path):
    colors=np.array([[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255]],dtype=np.uint8)
    pixels=np.zeros((512,768,3),dtype=np.uint8)
    for i,color in enumerate(colors):pixels[i//3*256:(i//3+1)*256,i%3*256:(i%3+1)*256]=color
    path=tmp_path/'atlas.png';Image.fromarray(pixels).save(path)
    return path, colors


def test_box_uv_faces_colors_orientation_and_geometry_roundtrip(tmp_path):
    atlas,colors=atlas_file(tmp_path);spec=specification()
    bare=tmp_path/'bare.glb';paint=tmp_path/'paint.glb'
    build_mesh(spec,bare);build_mesh(spec,paint,atlas)
    a,b=parts(load_scene(bare))[0],parts(load_scene(paint))[0]
    assert np.allclose(a.triangles,b.triangles)
    directions=[(0,-1,0),(0,1,0),(-1,0,0),(1,0,0),(0,0,1),(0,0,-1)]
    for panel,normal in enumerate(directions):
        face=int(np.argmax(b.face_normals@normal))
        assert np.allclose(_colors(b,face,np.array([[1/3]*3]),{}),colors[panel])
    for normal,sign in [((0,-1,0),1),((0,1,0),-1)]:
        faces=np.flatnonzero(b.face_normals@normal>.99);v=b.faces[faces].ravel()
        assert sign*np.corrcoef(b.vertices[v,0],b.visual.uv[v,0])[0,1]>.999
    assert inspect_mesh(paint,QualityPolicy(require_watertight=True))["passed"]


def test_cylinder_inner_outer_bottom_and_no_long_uv_seam(tmp_path):
    atlas,colors=atlas_file(tmp_path)
    profile=[[0.,-.5],[.5,-.5],[.5,.5],[.44,.5],[.44,-.42],[0.,-.42]]
    path=tmp_path/'mug.glb';build_mesh(specification('lathe_shell','cylindrical',profile),path,atlas)
    mesh=parts(load_scene(path))[0];norm=mesh.face_normals;cent=mesh.triangles_center
    outer=np.sum(norm[:,:2]*cent[:,:2],axis=1)>0
    for panel,normal in enumerate([(0,-1,0),(0,1,0),(-1,0,0),(1,0,0)]):
        candidates=np.flatnonzero(outer & (norm@normal>.99))
        assert len(candidates)
        assert np.allclose(_colors(mesh,int(candidates[0]),np.array([[1/3]*3]),{}),colors[panel])
    inner=np.flatnonzero((~outer)&(np.abs(norm[:,2])<.1))
    assert np.allclose(_colors(mesh,int(inner[0]),np.array([[1/3]*3]),{}),colors[4])
    assert np.ptp(mesh.visual.uv[mesh.faces],axis=1).max()<.51


def test_texture_stage_resumes_frozen_geometry_without_second_model_call(tmp_path,monkeypatch):
    calls=[]
    def propose(self,*args,**kwargs):
        assert kwargs['generated_texture'] is True
        calls.append(1);return specification(),{"provider":"test"}
    monkeypatch.setattr(CodexBackend,'propose_mesh',propose)
    ref=tmp_path/'ref.png';Image.new('RGB',(100,100),'white').save(ref)
    store=Store(tmp_path/'runs')
    config=PipelineConfig(image=ImageConfig(provider='disabled'),mesh=MeshConfig(backend='codex'),texture=TextureConfig(provider='provided'))
    job=create_job(store,[ref],'synthetic box',config)
    job=queue_candidate(store,job['id'],background=False)
    assert job['status']=='awaiting_texture',job
    original=job['attempts'][0]['artifacts']['geometry_spec']['sha256']
    atlas,_=atlas_file(tmp_path);provenance=tmp_path/'provenance.json'
    provenance.write_text(json.dumps({'image_sha256':'invalid'}))
    with pytest.raises(ValueError,match='does not match'):provide_texture(store,job['id'],atlas,provenance)
    assert store.load(job['id'])['status']=='awaiting_texture'
    provenance.write_text(json.dumps({'provider':'test-imagegen','image_sha256':digest(atlas)}))
    job=provide_texture(store,job['id'],atlas,provenance)
    assert job['status']=='awaiting_review',job
    assert len(job['attempts'])==1 and calls==[1]
    assert job['attempts'][0]['artifacts']['geometry_spec']['sha256']==original
    assert job['attempts'][0]['quality']['passed']
    verify_evidence(store,job)
    with pytest.raises(ValueError,match='awaiting'):provide_texture(store,job['id'],atlas,provenance)


def test_generated_atlas_rejects_unmapped_geometry(tmp_path):
    atlas,_=atlas_file(tmp_path)
    with pytest.raises(ValueError,match='no geometry component'):build_mesh(specification(projection='none'),tmp_path/'bad.glb',atlas)


def test_measured_unequal_atlas_rows_map_without_resampling_bitmap(tmp_path):
    atlas,colors=atlas_file(tmp_path)
    pixels=np.zeros((512,768,3),dtype=np.uint8)
    rects=[]
    for i,color in enumerate(colors):
        x=i%3*256;y=0 if i<3 else 320;bottom=320 if i<3 else 512
        pixels[y:bottom,x:x+256]=color
        rects.append([x/768,y/512,(x+256)/768,bottom/512])
    Image.fromarray(pixels).save(atlas);before=atlas.read_bytes()
    path=tmp_path/'unequal.glb';build_mesh(specification(),path,atlas,rects)
    mesh=parts(load_scene(path))[0]
    for i,normal in enumerate([(0,-1,0),(0,1,0),(-1,0,0),(1,0,0),(0,0,1),(0,0,-1)]):
        face=int(np.argmax(mesh.face_normals@normal))
        assert np.allclose(_colors(mesh,face,np.array([[1/3]*3]),{}),colors[i])
    assert atlas.read_bytes()==before
    with pytest.raises(ValueError,match='overlap'):validate_rectangles([[0,0,1,1]]*6)


def test_split_carton_flaps_share_one_top_artwork(tmp_path):
    atlas,_=atlas_file(tmp_path)
    pixels=np.asarray(Image.open(atlas)).copy()
    pixels[256:384,256:512]=[0,0,255]
    pixels[384:512,256:512]=[0,255,0]
    Image.fromarray(pixels).save(atlas)
    spec=specification();original=spec.components[0]
    spec.components=[original.model_copy(update={'name':name,'size':[1.,.5,.02],'position':[0.,y,0.]}) for name,y in [('front flap',-.25),('back flap',.25)]]
    output=tmp_path/'flaps.glb';build_mesh(spec,output,atlas)
    observed=[]
    for mesh in parts(load_scene(output)):
        top=int(np.argmax(mesh.face_normals[:,2]));color=_colors(mesh,top,np.array([[1/3]*3]),{})[0]
        observed.append((float(mesh.centroid[1]),color))
    observed.sort(key=lambda item:item[0])
    assert np.array_equal(observed[0][1],[0,255,0])
    assert np.array_equal(observed[1][1],[0,0,255])
