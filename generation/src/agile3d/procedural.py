"""Build a textured mesh from Codex's finite declarative geometry specification.

No generated code is executed. Units are normalized and Z is up. Independent
components may intersect; this is an explicit procedural approximation, not a
Boolean solid or a photogrammetric reconstruction.
"""

from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

from .codex_backend import ProceduralMeshSpec


def _triangulate(polygon):
    points = np.asarray(polygon, dtype=float)
    area = np.sum(
        points[:, 0] * np.roll(points[:, 1], -1)
        - np.roll(points[:, 0], -1) * points[:, 1]
    )
    order = list(range(len(points)))
    if area < 0:
        order.reverse()
    triangles = []

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    while len(order) > 3:
        found = False
        for j, b in enumerate(order):
            a, c = order[j - 1], order[(j + 1) % len(order)]
            if cross(points[a], points[b], points[c]) <= 1e-12:
                continue
            others = [i for i in order if i not in (a, b, c)]
            if any(
                all(
                    cross(points[x], points[y], points[i]) >= -1e-12
                    for x, y in ((a, b), (b, c), (c, a))
                )
                for i in others
            ):
                continue
            triangles.append((a, b, c))
            order.pop(j)
            found = True
            break
        if not found:
            raise ValueError("Cannot triangulate the extrusion profile")
    triangles.append(tuple(order))
    return points, np.asarray(triangles)


def _shape(component):
    if component.kind == "loft":
        from scipy.interpolate import PchipInterpolator

        sections = np.asarray(component.sections, dtype=float)
        z = np.linspace(sections[0, 0], sections[-1, 0], min(128, len(sections) * 5))
        params = PchipInterpolator(sections[:, 0], sections[:, 1:], axis=0)(z)
        angle = np.arange(64) * (2 * np.pi / 64)
        rings = []
        for height, (cx, cy, rx, ry, power) in zip(z, params):
            c, s = np.cos(angle), np.sin(angle)
            rings.append(np.column_stack((cx + rx * np.sign(c) * abs(c)**(2/power),
                                          cy + ry * np.sign(s) * abs(s)**(2/power), np.full(64, height))))
        vertices = np.vstack(rings)
        faces = []
        for ring in range(len(rings) - 1):
            for i in range(64):
                a, b = ring*64+i, ring*64+(i+1)%64
                faces.extend(((a, b, b+64), (a, b+64, a+64)))
        for ring, reverse in [(0, True), (len(rings)-1, False)]:
            center = len(vertices)
            vertices = np.vstack((vertices, [params[ring, 0], params[ring, 1], z[ring]]))
            for i in range(64):
                tri = (center, ring*64+i, ring*64+(i+1)%64)
                faces.append(tri[::-1] if reverse else tri)
        mesh = trimesh.Trimesh(vertices, faces, process=False)
    elif component.kind == "box":
        mesh = trimesh.creation.box()
    elif component.kind == "ellipsoid":
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    elif component.kind in ("lathe", "lathe_shell"):
        profile = np.asarray(component.profile, dtype=float)
        if component.kind == "lathe_shell":
            profile = np.vstack((profile, profile[0]))
        else:
            if profile[0, 0] > 0:
                profile = np.vstack(([0, profile[0, 1]], profile))
            if profile[-1, 0] > 0:
                profile = np.vstack((profile, [0, profile[-1, 1]]))
        mesh = trimesh.creation.revolve(profile, sections=64)
    elif component.kind == "torus":
        major = 0.5 / (1 + component.torus_ratio)
        mesh = trimesh.creation.torus(
            major_radius=major,
            minor_radius=major * component.torus_ratio,
            major_sections=64,
            minor_sections=24,
        )
    else:
        points, faces = _triangulate(component.profile)
        mesh = trimesh.creation.extrude_triangulation(points, faces, height=1)
        mesh.apply_translation([0, 0, -0.5])
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    return mesh


def _texture(mesh, component):
    rgba = tuple(round(c * 255) for c in component.color)
    label_rgba = tuple(round(c * 255) for c in component.label_color)
    texture = Image.new("RGBA", (1024, 512), rgba)
    if component.label_text:
        draw = ImageDraw.Draw(texture)
        # Pillow ships a portable sized default font; no machine-specific fonts.
        font = ImageFont.load_default(size=36)
        words = component.label_text.split()
        lines, line = [], ""
        for word in words:
            candidate = (line + " " + word).strip()
            if draw.textlength(candidate, font=font) > 430 and line:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
        y = 256 - len(lines) * 24
        for line in lines:
            width = draw.textlength(line, font=font)
            draw.text((768 - width / 2, y), line, font=font, fill=label_rgba)
            y += 48
    # A blank half and a printed half keep labels off hidden/back-facing panels.
    mesh.unmerge_vertices()
    local = mesh.vertices.copy()
    normals = np.repeat(mesh.face_normals, 3, axis=0)
    uv = np.empty((len(local), 2))
    if component.kind in ("lathe", "lathe_shell"):
        angle = np.arctan2(local[:, 1], local[:, 0])
        front = local[:, 1] < 0
        u = np.clip((angle + np.pi) / np.pi, 0, 1)
        uv[:, 0] = np.where(front, 0.5 + 0.5 * u, 0.25)
        uv[:, 1] = np.clip(local[:, 2] + 0.5, 0, 1)
    else:
        # Printed panels are often thin in local Z and then rotated onto the
        # object. Choose their broad surface before the component transform.
        thin_axis = int(np.argmin(component.size))
        horizontal, vertical = [axis for axis in range(3) if axis != thin_axis]
        printed = np.abs(normals[:, thin_axis]) > 0.7
        uv[:, 0] = np.where(printed, 0.5 + 0.5 * (local[:, horizontal] + 0.5), 0.25)
        uv[:, 1] = np.clip(local[:, vertical] + 0.5, 0, 1)
    mesh.visual = trimesh.visual.texture.TextureVisuals(
        uv=np.clip(uv, 0, 1),
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=texture,
            metallicFactor=0.0,
            roughnessFactor=0.65,
            doubleSided=False,
        ),
    )


def build_mesh(spec: ProceduralMeshSpec, output: Path, texture_atlas: Path | None = None, texture_layout=None) -> dict:
    from .texture import apply_atlas, read_atlas

    atlas = read_atlas(texture_atlas) if texture_atlas else None
    mapped = 0
    scene = trimesh.Scene()
    prepared = []
    for index, component in enumerate(spec.components):
        mesh = _shape(component)
        if atlas is not None and component.texture_projection == "cylindrical":
            apply_atlas(mesh, component, atlas, texture_layout)
            mapped += 1
        elif atlas is None or component.texture_projection == "none":
            _texture(mesh, component)
        mesh.apply_scale(component.size)
        transform = trimesh.transformations.euler_matrix(
            *np.deg2rad(component.rotation_deg), axes="sxyz"
        )
        transform[:3, 3] = component.position
        mesh.apply_transform(transform)
        prepared.append((index, component, mesh))
    box_meshes = [m for _, c, m in prepared if c.texture_projection == "box"]
    box_bounds = None
    if atlas is not None and box_meshes:
        vertices = np.concatenate([m.vertices for m in box_meshes])
        box_bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
    for index, component, mesh in prepared:
        if atlas is not None and component.texture_projection == "box":
            apply_atlas(mesh, component, atlas, texture_layout, box_bounds)
            mapped += 1
        scene.add_geometry(
            mesh,
            node_name=f"part-{index:02d}",
            geom_name=f"{index:02d}-{component.name}",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if atlas is not None and not mapped:
        raise ValueError("Atlas supplied but no geometry component is mapped to it")
    scene.export(output, file_type="glb")
    return {
        "backend": "codex-procedural",
        "units": "normalized_unknown_scale",
        "up_axis": "Z",
        "components": len(spec.components),
        "texture_source": "generated_atlas" if atlas is not None else "procedural",
        "atlas_mapped_components": mapped,
        "box_projection_space": "shared object-space bounds of mapped structural parts",
        "approximation_scope": spec.approximation_scope,
        "assumptions": spec.assumptions,
        "limitations": [
            "Components are not Boolean-unioned",
            "Generated artwork approximates observations; it is not recovered ground truth"
            if atlas is not None else "Labels are procedural approximations, not recovered textures",
            "Occluded surfaces are inferred",
        ],
    }
