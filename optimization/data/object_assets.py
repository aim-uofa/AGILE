"""Keep the final generated surface independent of the optimizer's proxy.

The asset-to-input transform is established BEFORE proxy decimation. It is
never estimated from a fitted pose, an unrelated mesh, or an optimization proxy.
No texture enhancement, resampling, or video-based appearance fitting occurs.
"""
from pathlib import Path
import copy
import hashlib
import json
import shlex
import shutil

import numpy as np
import trimesh


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def geometry_hash(mesh):
    digest = hashlib.sha256()
    for values, dtype in ((mesh.vertices, '<f8'), (mesh.faces, '<i8')):
        values = np.ascontiguousarray(values, dtype=dtype)
        digest.update(str(values.shape).encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def contained_file(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Missing or nonportable asset: ' + str(relative))
    return path


def texture_image(mesh):
    material = getattr(mesh.visual, 'material', None)
    image = getattr(material, 'image', None)
    return image if image is not None else getattr(material, 'baseColorTexture', None)


def mesh_summary(mesh):
    image = texture_image(mesh)
    return dict(vertices=len(mesh.vertices), triangles=len(mesh.faces),
                texture_size=list(image.size) if image is not None else None,
                geometry_sha256=geometry_hash(mesh))


def export_lossless_mesh(mesh, path):
    """Preserve decoded pixels in GLB or legacy OBJ without JPEG recompression."""
    mesh = mesh.copy()
    material = getattr(mesh.visual, 'material', None)
    if material is not None:
        mesh.visual.material = material = copy.deepcopy(material)
        for name in ('image', 'baseColorTexture', 'normalTexture', 'emissiveTexture',
                     'occlusionTexture', 'metallicRoughnessTexture'):
            image = getattr(material, name, None)
            if image is not None:
                image = image.copy()
                image.format = 'PNG'
                setattr(material, name, image)
        if Path(path).suffix.lower() == '.obj' and isinstance(material, trimesh.visual.material.PBRMaterial):
            # glTF's absent baseColorFactor means white. Trimesh's generic
            # SimpleMaterial default is gray, which would darken legacy OBJs.
            factor = material.baseColorFactor
            mesh.visual.material = trimesh.visual.material.SimpleMaterial(
                image=material.baseColorTexture,
                diffuse=factor if factor is not None else [255,255,255,255])
    mesh.export(path)


def export_lossless_glb(mesh, path):
    export_lossless_mesh(mesh, path)


def load_mesh(path):
    # force='mesh' could silently concatenate materials and lose their UVs.
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) != 1 or len(mesh.graph.nodes_geometry) != 1:
            raise ValueError('Expected a single object with one UV material')
        mesh = mesh.dump()[0]
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError('Missing triangle surface')
    if not np.isfinite(mesh.vertices).all():
        raise ValueError('Non-finite object geometry')
    return mesh


def fit_asset_to_input(asset, canonical, relative_tolerance=1e-5):
    """Recover a similarity from identical ordered surface/UV topology only.

    Comparing triangle corners supports UV-split vertex numbering. A topology
    change, UV change, reflection, or non-similarity must be handled explicitly
    upstream; it must not be disguised by an unconstrained registration here.
    """
    if asset.faces.shape != canonical.faces.shape:
        raise ValueError('Final asset and canonical input must have matching triangle topology')
    for mesh in (asset, canonical):
        uv = getattr(mesh.visual, 'uv', None)
        if uv is None or len(uv) != len(mesh.vertices) or not np.isfinite(uv).all():
            raise ValueError('A verified UV surface is required')
    if not np.allclose(asset.visual.uv[asset.faces], canonical.visual.uv[canonical.faces],
                       atol=2e-6, rtol=0):
        raise ValueError('Asset UV correspondence differs from the canonical input')
    x, y = asset.triangles.reshape(-1, 3), canonical.triangles.reshape(-1, 3)
    xc, yc = x - x.mean(0), y - y.mean(0)
    u, singular, vt = np.linalg.svd(xc.T @ yc)
    sign = np.ones(3)
    sign[-1] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ np.diag(sign) @ u.T
    denominator = np.sum(xc * xc)
    if denominator <= 0 or np.linalg.matrix_rank(xc) < 2:
        raise ValueError('Degenerate source surface')
    scale = float(np.sum(singular * sign) / denominator)
    translation = y.mean(0) - scale * x.mean(0) @ rotation.T
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = scale * rotation, translation
    errors = np.linalg.norm(x @ transform[:3, :3].T + translation - y, axis=1)
    diagonal = float(np.linalg.norm(canonical.extents))
    maximum = float(errors.max())
    if scale <= 0 or diagonal <= 0 or maximum > relative_tolerance * diagonal:
        raise ValueError(f'Final asset does not match input: max residual {maximum:g}, diagonal {diagonal:g}')
    return transform, dict(method='ordered-triangle-and-UV-similarity',
                           max_error_input_units=maximum, input_diagonal=diagonal,
                           relative_tolerance=relative_tolerance, scale=scale)


def asset_files(mesh_path, supporting_files=()):
    """Resolve the mesh and its local material images, without rewriting them."""
    mesh_path = Path(mesh_path).resolve()
    root = mesh_path.parent
    files = {mesh_path.name: mesh_path}
    if mesh_path.suffix.lower() == '.obj':
        for line in mesh_path.read_text().splitlines():
            if not line.startswith('mtllib '):
                continue
            value = line[7:].strip()
            names = [value] if (root / value).is_file() else shlex.split(value)
            for name in names:
                material = contained_file(root, name)
                files[name] = material
                for entry in material.read_text().splitlines():
                    words = shlex.split(entry, comments=True)
                    if words and (words[0].lower().startswith('map_') or words[0].lower() in ('bump', 'disp', 'norm')):
                        relative = str(Path(name).parent / words[-1])
                        files[relative] = contained_file(root, relative)
    elif mesh_path.suffix.lower() != '.glb':
        raise ValueError('Final assets must be self-contained GLB or OBJ/MTL/images')
    for path in supporting_files:
        path = Path(path).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Supporting files must belong to the asset directory')
        relative = str(path.relative_to(root))
        files[relative] = contained_file(root, relative)
    return files


def prepare_object_asset(mesh_path, canonical_mesh_path, output, *, stage, provenance,
                         supporting_files=()):
    """Freeze a selected postprocessed asset and its verified canonical mapping."""
    if stage not in ('blender-final', 'postprocessed-final') or not provenance:
        raise ValueError('Declare the final postprocessing stage and its provenance')
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    files = asset_files(mesh_path, supporting_files)
    mesh, canonical = load_mesh(mesh_path), load_mesh(canonical_mesh_path)
    if texture_image(mesh) is None:
        raise ValueError('Final asset has no UV-bound bitmap texture')
    transform, verification = fit_asset_to_input(mesh, canonical)
    output.mkdir(parents=True)
    for name, source in files.items():
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    manifest = dict(schema_version=1, kind='agile-final-object-asset', stage=stage,
                    mesh='source/' + Path(mesh_path).name,
                    asset_to_input=transform.tolist(), input_geometry_sha256=geometry_hash(canonical),
                    alignment=verification, source_mesh=mesh_summary(mesh),
                    canonical_input=mesh_summary(canonical), provenance=provenance,
                    texture_policy='preserve-selected-final-texture',
                    sha256={'source/' + name: file_hash(path) for name, path in files.items()})
    (output / 'asset.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return output / 'asset.json'


def read_object_asset(manifest_path, canonical_input=None):
    path = Path(manifest_path).resolve()
    record = json.loads(path.read_text())
    if record.get('schema_version') != 1 or record.get('kind') != 'agile-final-object-asset':
        raise ValueError('Unsupported object asset manifest')
    if record.get('stage') not in ('blender-final', 'postprocessed-final'):
        raise ValueError('Only selected postprocessed final assets can replace the display mesh')
    for relative, expected in record['sha256'].items():
        if file_hash(contained_file(path.parent, relative)) != expected:
            raise ValueError('Final asset hash mismatch: ' + relative)
    if record['mesh'] not in record['sha256']:
        raise ValueError('Final mesh is not covered by asset hashes')
    if canonical_input is not None and geometry_hash(canonical_input) != record['input_geometry_sha256']:
        raise ValueError('Final asset mapping belongs to a different canonical input')
    transform = np.asarray(record['asset_to_input'], dtype=np.float64)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1])
            or np.linalg.det(transform[:3, :3]) <= 0):
        raise ValueError('Invalid asset-to-input transform')
    mesh = load_mesh(contained_file(path.parent, record['mesh']))
    mesh.apply_transform(transform)
    return mesh, record


def copy_asset_package(manifest_path, output):
    path, output = Path(manifest_path).resolve(), Path(output)
    _, record = read_object_asset(path)
    output.mkdir(parents=True, exist_ok=True)
    for relative in record['sha256']:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(contained_file(path.parent, relative), target)
    shutil.copyfile(path, output / 'asset.json')


def write_native_object_assets(directory, proxy, canonical_input, *, asset_manifest=None,
                               appearance_optimized=False):
    """Write both unscaled canonical variants; saved poses carry the final scale."""
    directory = Path(directory)
    root = directory / 'object_assets'
    root.mkdir(parents=True, exist_ok=True)
    source = canonical_input.copy() if canonical_input is not None else proxy.copy()
    record = None
    if asset_manifest:
        if canonical_input is None:
            raise ValueError('A final asset requires its original canonical optimization input')
        source, record = read_object_asset(asset_manifest, canonical_input)
        copy_asset_package(asset_manifest, root / 'source_asset')
    # Do not silently throw away an explicitly requested appearance optimization.
    display = proxy.copy() if appearance_optimized else source.copy()
    export_lossless_glb(display, root / 'display.glb')
    export_lossless_glb(proxy, root / 'optimization_proxy.glb')
    export_lossless_mesh(display, directory / 'object_mesh.obj')  # legacy evaluators
    descriptor = dict(schema_version=1, object_scale_baked=False, default='display',
        texture_policy='optimized-proxy' if appearance_optimized else 'preserve-selected-final-texture',
        variants={
            'display': dict(mesh='object_assets/display.glb', **mesh_summary(display)),
            'optimization_proxy': dict(mesh='object_assets/optimization_proxy.glb', **mesh_summary(proxy))},
        source_stage=record['stage'] if record else 'optimization-input-unverified-stage',
        appearance_optimized=bool(appearance_optimized),
        source_asset_manifest='object_assets/source_asset/asset.json' if record else None)
    if appearance_optimized:
        export_lossless_glb(source, root / 'source_reference.glb')
        descriptor['variants']['source_reference'] = dict(
            mesh='object_assets/source_reference.glb', **mesh_summary(source))
    (directory / 'object_assets.json').write_text(json.dumps(descriptor, ensure_ascii=False, indent=2) + '\n')
    return descriptor
