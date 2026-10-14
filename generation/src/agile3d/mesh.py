"""Inspect real geometry and render portable orthographic review evidence on CPU.

Rendering preserves node transforms and samples base-color textures. It is not a
PBR renderer or a calibrated comparison against the input cameras.
"""

import io
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from .models import QualityPolicy


def load_scene(path: Path) -> trimesh.Scene:
    if path.suffix.lower() != ".glb":
        raise ValueError(
            "Use a self-contained GLB mesh (external resources are disabled)"
        )
    if path.stat().st_size > 512 * 1024**2:
        raise ValueError("GLB exceeds the 512 MiB input limit")
    scene = trimesh.load(
        io.BytesIO(path.read_bytes()),
        file_type="glb",
        force="scene",
        process=False,
        resolver={},
    )
    if not scene.geometry:
        raise ValueError("The GLB contains no geometry")
    return scene


def parts(scene: trimesh.Scene) -> list[trimesh.Trimesh]:
    result = []
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        mesh = scene.geometry[name].copy()
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("Only triangle meshes are supported")
        mesh.apply_transform(transform)
        result.append(mesh)
    return result


def texture_image(material):
    image = getattr(material, "baseColorTexture", None)
    return image if image is not None else getattr(material, "image", None)


def has_texture(mesh) -> bool:
    uv = getattr(mesh.visual, "uv", None)
    material = getattr(mesh.visual, "material", None)
    if uv is None or not np.isfinite(uv).all():
        return False
    materials = getattr(material, "materials", [material])
    return bool(materials) and all(texture_image(m) is not None for m in materials)


def inspect_mesh(path: Path, policy: QualityPolicy) -> dict:
    meshes = parts(load_scene(path))
    vertices = sum(len(m.vertices) for m in meshes)
    faces = sum(len(m.faces) for m in meshes)
    reasons, warnings = [], []
    finite = all(np.isfinite(m.vertices).all() for m in meshes)
    if not finite or not vertices or not faces:
        return {
            "passed": False,
            "reasons": ["Empty or non-finite geometry"],
            "warnings": [],
            "vertices": vertices,
            "faces": faces,
        }
    if faces > policy.max_faces:
        reasons.append(f"Face count {faces} exceeds {policy.max_faces}")
        # Do not perform unbounded topology/render work after this gate.
        return {
            "passed": False,
            "reasons": reasons,
            "warnings": [],
            "vertices": vertices,
            "faces": faces,
        }
    joined = trimesh.util.concatenate(meshes)
    bounds = joined.bounds
    diagonal = float(np.linalg.norm(bounds[1] - bounds[0]))
    if diagonal <= 1e-12:
        reasons.append("Mesh has zero spatial extent")
    degenerates = int(
        sum(
            np.count_nonzero(m.area_faces <= max(diagonal**2, 1e-24) * 1e-14)
            for m in meshes
        )
    )
    if degenerates:
        reasons.append(f"{degenerates} degenerate triangles")
    # Weld UV seams on a geometry-only copy for topology checks, leaving export intact.
    topology = trimesh.Trimesh(joined.vertices, joined.faces, process=True)
    components = len(topology.split(only_watertight=False))
    watertight = bool(topology.is_watertight)
    winding = bool(topology.is_winding_consistent)
    textured = all(has_texture(m) for m in meshes)
    if components > policy.max_components:
        reasons.append(
            f"{components} disconnected components exceed {policy.max_components}"
        )
    if not winding:
        reasons.append("Inconsistent triangle winding")
    if not watertight:
        (reasons if policy.require_watertight else warnings).append(
            "Mesh is not watertight"
        )
    if not textured:
        (reasons if policy.require_texture else warnings).append(
            "Some geometry has no UV base-color texture"
        )
    warnings.append(
        "Metric scale and reference-camera alignment are not inferred by this check"
    )
    return {
        "passed": not reasons,
        "reasons": reasons,
        "warnings": warnings,
        "vertices": vertices,
        "faces": faces,
        "components": components,
        "degenerate_faces": degenerates,
        "watertight": watertight,
        "winding_consistent": winding,
        "textured": textured,
        "bounds": bounds.tolist(),
        "units": "unknown",
    }


def _material(mesh, face_index):
    material = getattr(mesh.visual, "material", None)
    materials = getattr(material, "materials", None)
    if materials:
        indices = getattr(mesh.visual, "face_materials", None)
        return materials[int(indices[face_index]) if indices is not None else 0]
    return material


def _colors(mesh, face_index, weights, texture_cache):
    material = _material(mesh, face_index)
    uv = getattr(mesh.visual, "uv", None)
    texture = texture_image(material)
    if uv is not None and texture is not None:
        key = id(texture)
        if key not in texture_cache:
            texture_cache[key] = np.asarray(texture.convert("RGB"))
        pixels = texture_cache[key]
        coords = np.clip(weights @ uv[mesh.faces[face_index]], 0, 1)
        x = np.rint(coords[:, 0] * (pixels.shape[1] - 1)).astype(int)
        y = np.rint((1 - coords[:, 1]) * (pixels.shape[0] - 1)).astype(int)
        color = pixels[y, x].astype(float)
        factor = getattr(material, "baseColorFactor", None)
        if factor is not None:
            factor = np.array(factor[:3], dtype=float)
            color *= factor / 255 if factor.max() > 1 else factor
        return color
    if mesh.visual.kind == "vertex":
        return weights @ mesh.visual.vertex_colors[mesh.faces[face_index], :3]
    if mesh.visual.kind == "face":
        color = mesh.visual.face_colors[face_index, :3]
    else:
        color = np.asarray(getattr(material, "main_color", [180, 180, 180, 255]))[:3]
    return np.tile(color, (len(weights), 1))


def render_views(path: Path, output: Path, size: int = 256, up_axis: str = "Y") -> Path:
    meshes = parts(load_scene(path))
    all_vertices = np.concatenate([m.vertices for m in meshes])
    if not np.isfinite(all_vertices).all():
        raise ValueError("Cannot render non-finite vertices")
    center = (all_vertices.min(0) + all_vertices.max(0)) / 2
    extent = max(float(np.ptp(all_vertices, axis=0).max()), 1e-9)
    # Names describe canonical object axes, not the unknown reference-camera pose.
    views = [
        ("+Z", (1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ("-Z", (-1, 0, 0), (0, 1, 0), (0, 0, -1)),
        ("+X", (0, 0, -1), (0, 1, 0), (1, 0, 0)),
        ("-X", (0, 0, 1), (0, 1, 0), (-1, 0, 0)),
        ("+Y", (1, 0, 0), (0, 0, -1), (0, 1, 0)),
        ("-Y", (1, 0, 0), (0, 0, 1), (0, -1, 0)),
    ]
    if up_axis == "Z":
        views = [
            ("front -Y", (1, 0, 0), (0, 0, 1), (0, -1, 0)),
            ("back +Y", (-1, 0, 0), (0, 0, 1), (0, 1, 0)),
            ("left -X", (0, -1, 0), (0, 0, 1), (-1, 0, 0)),
            ("right +X", (0, 1, 0), (0, 0, 1), (1, 0, 0)),
            ("top +Z", (1, 0, 0), (0, 1, 0), (0, 0, 1)),
            ("bottom -Z", (1, 0, 0), (0, -1, 0), (0, 0, -1)),
        ]
    elif up_axis != "Y":
        raise ValueError("Rendering up_axis must be Y or Z")
    sheet = Image.new("RGB", (size * 3, (size + 24) * 2), "white")
    texture_cache = {}
    for index, (label, right, up, forward) in enumerate(views):
        pixels = np.full((size, size, 3), 242, np.uint8)
        zbuffer = np.full((size, size), -np.inf)
        matrix = np.array([right, up, forward]).T
        for mesh in meshes:
            # Trimesh validates tracked-array hashes on property access. Cache
            # this immutable render state once, not once per rasterized face.
            face_normals = np.asarray(mesh.face_normals).copy()
            faces = np.asarray(mesh.faces)
            projected = (mesh.vertices - center) @ matrix / extent
            projected[:, 0] = (0.5 + projected[:, 0] * 0.82) * (size - 1)
            projected[:, 1] = (0.5 - projected[:, 1] * 0.82) * (size - 1)
            for face_index, face in enumerate(faces):
                tri = projected[face]
                lo = np.maximum(np.floor(tri[:, :2].min(0)).astype(int), 0)
                hi = np.minimum(np.ceil(tri[:, :2].max(0)).astype(int), size - 1)
                if (hi < lo).any():
                    continue
                a, b, c = tri
                den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
                if abs(den) < 1e-10:
                    continue
                yy, xx = np.mgrid[lo[1] : hi[1] + 1, lo[0] : hi[0] + 1]
                w0 = ((b[1] - c[1]) * (xx - c[0]) + (c[0] - b[0]) * (yy - c[1])) / den
                w1 = ((c[1] - a[1]) * (xx - c[0]) + (a[0] - c[0]) * (yy - c[1])) / den
                w2 = 1 - w0 - w1
                depth = w0 * a[2] + w1 * b[2] + w2 * c[2]
                mask = (
                    (w0 >= -1e-6)
                    & (w1 >= -1e-6)
                    & (w2 >= -1e-6)
                    & (depth > zbuffer[yy, xx])
                )
                if not mask.any():
                    continue
                weights = np.stack([w0[mask], w1[mask], w2[mask]], axis=1)
                # Mild two-sided lighting keeps texture colors readable.
                shade = 0.55 + 0.45 * abs(float(face_normals[face_index] @ forward))
                pixels[yy[mask], xx[mask]] = np.clip(
                    _colors(mesh, face_index, weights, texture_cache) * shade, 0, 255
                )
                zbuffer[yy[mask], xx[mask]] = depth[mask]
        tile = Image.fromarray(pixels)
        x, y = (index % 3) * size, (index // 3) * (size + 24)
        sheet.paste(tile, (x, y))
        ImageDraw.Draw(sheet).text(
            (x + 8, y + size + 4), f"mesh axis {label}", fill="black"
        )
    sheet.save(output)
    return output


def export_obj(glb: Path, destination: Path):
    scene = load_scene(glb)
    obj, textures = trimesh.exchange.obj.export_obj(scene, return_texture=True)
    (destination / "textured_mesh.obj").write_text(obj)
    for name, content in (textures or {}).items():
        if Path(name).name != name:
            raise ValueError("Unexpected material output filename")
        target = destination / name
        target.write_bytes(content.encode() if isinstance(content, str) else content)
