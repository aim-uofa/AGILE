"""Blender surface preparation, immutable mesh UVs, and local evidence."""
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

from .mesh import load_scene, parts, render_views
from .models import SurfaceConfig
from .store import atomic_json, digest


def uv_guide(record, output, size):
    # This is a deterministic mesh-coordinate diagram, not an edited texture.
    guide = Image.new('RGB', (size, size), '#f7f7f7')
    draw = ImageDraw.Draw(guide)
    palette = ['#f0d2bf', '#c7e1ee', '#cce4c6', '#eed0e3', '#eee1a9', '#d5cff0']
    font = ImageFont.load_default(size=max(12, size//64))
    for island in record['islands']:
        for polygon in island['polygons']:
            points = [(round(u*(size-1)), round((1-v)*(size-1))) for u,v in polygon]
            draw.polygon(points, fill=palette[(island['id']-1)%6], outline='#83939c', width=1)
    for island in record['islands']:
        lo, hi = np.asarray(island['uv_bounds'])
        if np.prod(hi-lo) < .0005:
            continue
        text = f"{island['id']} {island['direction'].split()[0]}"
        x, y = (lo+hi)/2
        px, py = x*(size-1), (1-y)*(size-1)
        box = draw.textbbox((px,py), text, font=font, anchor='mm')
        draw.rectangle((box[0]-2,box[1]-2,box[2]+2,box[3]+2), fill='white')
        draw.text((px,py), text, font=font, anchor='mm', fill='#142d40')
    guide.save(output)


def white_mesh(source, output):
    scene = load_scene(source)
    for mesh in scene.geometry.values():
        mesh.visual = trimesh.visual.texture.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=[215,215,215,255], roughnessFactor=.8))
    scene.export(output)


def oblique_views(source, output, up_axis):
    """Rigidly rotate a temporary scene to expose thickness and vessel interiors."""
    scene = load_scene(source)
    rx = lambda deg: trimesh.transformations.rotation_matrix(np.deg2rad(deg), [1, 0, 0])
    ry = lambda deg: trimesh.transformations.rotation_matrix(np.deg2rad(deg), [0, 1, 0])
    if up_axis not in ('Y', 'Z'):
        raise ValueError('Rendering up_axis must be Y or Z')
    scene.apply_transform(rx(25) @ ry(30) @ (rx(-90) if up_axis == 'Z' else np.eye(4)))
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        rotated = Path(temporary)/'rotated.glb'
        scene.export(rotated)
        render_views(rotated, output)


def prepare_surface(mesh: Path, directory: Path, config: SurfaceConfig, *, up_axis='Z', bake=False):
    directory.mkdir(parents=True, exist_ok=True)
    request = {'mesh': str(mesh.resolve()), 'output': str(directory.resolve()),
               'config': config.model_dump(), 'up_axis': up_axis, 'bake': bake,
               'worker_sha256': digest(Path(__file__).with_name('blender_worker.py'))}
    atomic_json(directory/'request.json', request)
    command = [config.executable, '--background', '--factory-startup', '--python-exit-code', '1',
               '--python', str(Path(__file__).with_name('blender_worker.py')), '--', str((directory/'request.json').resolve())]
    number = len(list(directory.glob('blender*.log'))) + 1
    log_path = directory/f'blender-{number:02d}.log'
    atomic_json(directory/f'request-{number:02d}.json', request)
    with log_path.open('wb') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                       timeout=config.timeout_seconds)
    record = json.loads((directory/'surface.json').read_text())
    original = trimesh.util.concatenate(parts(load_scene(mesh)))
    prepared = trimesh.util.concatenate(parts(load_scene(directory/'prepared.glb')))
    a, _ = trimesh.sample.sample_surface(original, 6000, seed=42)
    b, _ = trimesh.sample.sample_surface(prepared, 6000, seed=43)
    diagonal = float(np.linalg.norm(original.extents))
    distances = np.r_[cKDTree(a).query(b)[0], cKDTree(b).query(a)[0]]
    record['source_alignment_check'] = {'no_registration': True, 'samples_each': 6000,
        'mean_relative_diagonal': float(distances.mean()/diagonal),
        'p95_relative_diagonal': float(np.quantile(distances,.95)/diagonal),
        'source_bounds': original.bounds.tolist(), 'prepared_bounds': prepared.bounds.tolist()}
    # Detect coordinate-system mistakes and severely destructive remeshes.
    if np.quantile(distances,.95)/diagonal > .08:
        raise ValueError('Surface processing changed geometry excessively; inspect source and prepared meshes')
    uv_guide(record, directory/'uv-layout.png', config.texture_size)
    white_mesh(directory/'prepared.glb', directory/'white.glb')
    render_views(directory/'white.glb', directory/'white-views.png', up_axis=up_axis)
    oblique_views(directory/'white.glb', directory/'white-oblique-views.png', up_axis)
    render_views(directory/'prepared.glb', directory/'prepared-views.png', up_axis=up_axis)
    record['hashes'] = {name: digest(directory/name) for name in ['prepared.glb','uv-layout.png','baked-albedo.png','white-views.png','white-oblique-views.png']}
    atomic_json(directory/'surface.json', record)
    return record


def read_uv_texture(path: Path):
    with Image.open(path) as image:
        if image.width != image.height or not 512 <= image.width <= 4096:
            raise ValueError('Frozen UV texture must be square and 512..4096 pixels')
        if image.format != 'PNG':
            raise ValueError('Frozen UV texture must be PNG')
        return image.convert('RGB')


def apply_uv_texture(mesh: Path, atlas: Path, output: Path):
    """Use the mesh's frozen UVs exactly. Never adapt UVs to a generated picture."""
    image = read_uv_texture(atlas)
    scene = load_scene(mesh)
    for part in scene.geometry.values():
        uv = getattr(part.visual, 'uv', None)
        if uv is None or not np.isfinite(uv).all() or uv.min() < -.0001 or uv.max() > 1.0001:
            raise ValueError('Prepared mesh has missing or invalid packed UVs')
        part.visual = trimesh.visual.texture.TextureVisuals(uv=uv.copy(),
            material=trimesh.visual.material.PBRMaterial(baseColorTexture=image, metallicFactor=0., roughnessFactor=.5))
    scene.export(output)
    return {'texture_source': 'generated_frozen_uv', 'prepared_mesh_sha256': digest(mesh),
            'texture_sha256': digest(atlas), 'UV_changed_after_generation': False}


def mesh_uv_request(description, record, prepared, guide, *, baked=None, feedback=''):
    from .uv_orientation import artwork_orientation
    measured = artwork_orientation(prepared, record) if record.get('up_axis_in_export') and record['islands'] else []
    islands = [{k:v for k,v in island.items() if k != 'polygons'} for island in record['islands']]
    for island in islands:
        (u0, v0), (u1, v1) = island['uv_bounds']
        island['image_bounds_left_top_right_bottom'] = [u0, 1-v1, u1, 1-v0]
    return {'schema_version': 2, 'mode': 'paint_frozen_mesh_uv',
        'prepared_mesh_sha256': digest(prepared), 'uv_layout_sha256': digest(guide),
        'image_shape': 'square', 'islands': islands, 'artwork_orientation': measured,
        'prompt': 'Use case: product-mockup. Produce ONLY a square base-color texture map for this exact unwrapped 3D mesh. '
            f'Target: {description}. The UV guide is generated from the actual mesh, not a suggested panel layout. '
            'Preserve every island boundary, position, scale and orientation. Paint inside the actual islands with a small '
            'edge bleed. Remove diagram labels, wire lines and guide colors from the result. Do not draw objects, shadows, '
            'perspective views, a 3x2 contact sheet or a new UV layout. References show real appearance; exclude hands and '
            'background. Visible markings should match the reference; do not invent readable hidden fine print. '
            + ('The baked atlas is the edit target: refine appearance while keeping its surface correspondence. ' if baked else '')
            + 'UV labels describe surface directions. Use image_bounds_left_top_right_bottom '
            'with the image TOP LEFT as origin; UV v instead starts at the bottom. '
            'Do not assume left-to-right front/back ordering: each island has its own direction and position. '
            + ('MEASURED ARTWORK DIRECTIONS: ' + json.dumps(measured) + '. An island may need upside-down artwork in the atlas to appear upright on the mesh. Follow the measured directions without moving UV islands. ' if measured else '')
            + feedback,
        'baked_seed': str(baked) if baked else None,
        'scope': 'Geometry and UVs are frozen before image generation. Agent must assess the actual resulting mesh.'}
