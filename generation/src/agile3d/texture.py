"""Map a generated six-panel albedo atlas onto real procedural mesh surfaces.

The image is not composited onto a preview. UVs and the bitmap are embedded in
the exported GLB. No learned 3D model or image-generation API is hidden here.
"""

from pathlib import Path

import numpy as np
from PIL import Image
import trimesh


ATLAS_PANELS = ["front", "back", "left", "right", "top_or_interior", "bottom"]


def read_atlas(path: Path) -> Image.Image:
    with Image.open(path) as image:
        if image.width < 768 or image.height < 512 or max(image.size) > 4096:
            raise ValueError("Atlas must be at least 768x512 and at most 4096 pixels per side")
        return image.convert("RGB")


def validate_rectangles(rectangles):
    if rectangles is None:
        return np.array([[i % 3 / 3, i // 3 / 2, (i % 3 + 1) / 3, (i // 3 + 1) / 2] for i in range(6)])
    values = np.asarray(rectangles, dtype=float)
    if values.shape != (6, 4) or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Supply six finite normalized [left, top, right, bottom] panel rectangles")
    if (values[:, 2:] <= values[:, :2]).any():
        raise ValueError("Atlas panel rectangles must have positive area")
    for i, a in enumerate(values):
        for b in values[i + 1:]:
            if min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1]):
                raise ValueError("Atlas panel rectangles must not overlap")
    return values


def atlas_uv(mesh, component, image_size, rectangles=None, projection_bounds=None):
    """Return per-corner UVs after splitting seams; preserve triangle positions."""
    normals = np.asarray(mesh.face_normals).copy()
    mesh.unmerge_vertices()
    vertices = np.asarray(mesh.vertices)
    bounds = mesh.bounds if projection_bounds is None else projection_bounds
    xyz = (vertices - bounds[0]) / np.maximum(bounds[1] - bounds[0], 1e-9)
    # The normal after nonuniform component scaling chooses the actual broad face.
    scaled = normals.copy() if projection_bounds is not None else normals / np.asarray(component.size)
    scaled /= np.maximum(np.linalg.norm(scaled, axis=1, keepdims=True), 1e-12)
    repeated = np.repeat(scaled, 3, axis=0)
    dominant = np.argmax(np.abs(repeated), axis=1)
    tile = np.zeros(len(vertices), dtype=int)
    uv = np.empty((len(vertices), 2))
    for axis, sign, panel, horizontal, vertical, flip_u, flip_v in [
        (1, -1, 0, 0, 2, False, False),
        (1, 1, 1, 0, 2, True, False),
        (0, -1, 2, 1, 2, True, False),
        (0, 1, 3, 1, 2, False, False),
        (2, 1, 4, 0, 1, False, False),
        (2, -1, 5, 0, 1, False, True),
    ]:
        mask = (dominant == axis) & (np.sign(repeated[:, axis]) == sign)
        tile[mask] = panel
        uv[mask, 0] = 1 - xyz[mask, horizontal] if flip_u else xyz[mask, horizontal]
        uv[mask, 1] = 1 - xyz[mask, vertical] if flip_v else xyz[mask, vertical]
    if component.texture_projection == "cylindrical":
        centers = vertices.reshape(-1, 3, 3).mean(axis=1)
        center_angle = np.arctan2(centers[:, 1], centers[:, 0])
        quarter = np.floor(((center_angle + 3 * np.pi / 4) % (2 * np.pi)) / (np.pi / 2)).astype(int)
        # Front -Y, right +X, back +Y, left -X, each centered on a panel.
        panel = np.array([0, 3, 1, 2])[quarter]
        panel_angle = -np.pi / 2 + quarter * np.pi / 2
        angle = np.arctan2(vertices[:, 1], vertices[:, 0])
        delta = (angle - np.repeat(panel_angle, 3) + np.pi) % (2 * np.pi) - np.pi
        walls = np.repeat(np.abs(normals[:, 2]) < 0.75, 3)
        radial_dot = np.sum(normals[:, :2] * centers[:, :2], axis=1)
        outer = walls & np.repeat(radial_dot > 0, 3)
        inner = walls & ~outer
        tile[outer] = np.repeat(panel, 3)[outer]
        uv[outer, 0] = delta[outer] / (np.pi / 2) + 0.5
        uv[outer, 1] = xyz[outer, 2]
        # The top/interior panel is specified as uniform material, not artwork.
        tile[inner] = 4
        uv[inner] = 0.5
    uv = np.clip(uv, 0, 1)
    # Sample one pixel inside each cell to avoid neighboring-panel bleed.
    width, height = image_size
    rectangles = validate_rectangles(rectangles)
    bounds = rectangles[tile]
    span = bounds[:, 2:] - bounds[:, :2]
    if (span * [width, height] < 4).any():
        raise ValueError("Each atlas panel must span at least four pixels per axis")
    inset = 1 / (span * [width, height])
    uv = inset + uv * (1 - 2 * inset)
    uv[:, 0] = bounds[:, 0] + uv[:, 0] * span[:, 0]
    uv[:, 1] = 1 - bounds[:, 3] + uv[:, 1] * span[:, 1]
    return uv


def apply_atlas(mesh, component, atlas, rectangles=None, projection_bounds=None):
    uv = atlas_uv(mesh, component, atlas.size, rectangles, projection_bounds)
    mesh.visual = trimesh.visual.texture.TextureVisuals(
        uv=uv,
        material=trimesh.visual.material.PBRMaterial(
            name="generated-albedo-atlas", baseColorTexture=atlas,
            baseColorFactor=[255, 255, 255, 255], metallicFactor=0.0,
            roughnessFactor=0.45, doubleSided=False,
        ),
    )


def texture_request(spec, description, feedback=""):
    components = [
        {"name": c.name, "projection": c.texture_projection, "size": c.size}
        for c in spec.components if c.texture_projection != "none"
    ]
    if not components:
        raise ValueError("Generated-texture geometry must contain a mapped structural surface")
    prompt = (
        "Use case: product-mockup. Asset type: flat base-color/albedo UV texture atlas.\n"
        f"Target observed object: {description}\n"
        "Inputs: the attached photographs are appearance references only. Reconstruct the "
        "visible artwork, colors, lettering and material pattern; remove hands, supports and "
        "background. Do not put a 3D object or a perspective product photograph in the output.\n"
        "Deliver ONE full-bleed image containing exactly THREE equal columns and TWO equal "
        "rows. Top row: FRONT, BACK, LEFT. Bottom row: RIGHT, TOP/INTERIOR, BOTTOM. These "
        "are directions for you, NOT text to print. No borders, gutters, legends, numbering "
        "or panel labels. Each cell is a flat normalized surface map that fills its entire "
        "rectangle; actual UV mapping restores the physical surface proportions. Keep "
        "printing upright in each side cell, with consistent artwork placement and scale.\n"
        "This is surface color only: no cast shadows, reflections, highlights, perspective, "
        "geometry outlines, drop shadows or studio background. White plastic regions must "
        "be plain white; uniform material fills the whole region. Preserve reference-visible "
        "brand words accurately. Unreadable fine print may be visually approximated; do not "
        "invent legible claims. Do not omit the rear panel merely because the front is clearer.\n"
        "For cylindrical bodies the four side cells represent consecutive quarter-wrap "
        "panels centered on front/back/left/right, not four repeated front labels. Make "
        "their adjoining colors/patterns continuous. Top/interior should be uniform inner "
        "material. For a box all six cells cover their entire respective face. For a "
        "bottle the side cells cover the bottle BODY, including blank shoulder and base, "
        "but excluding the separate cap. Inspect the input before deciding label coverage.\n"
        "Render the atlas as crisp, high-resolution printable artwork, ideally 1536x1024 "
        "or larger with the same 3:2 grid. The file will be embedded in a real GLB, not "
        "shown as a substitute for the final mesh."
    )
    if feedback.strip():
        prompt += "\nRepair the prior texture using this actual mesh review:\n" + feedback
    return {
        "schema_version": 1, "provider": "codex-desktop-image-tool",
        "model_identity": "not reported by the built-in tool",
        "layout": {"columns": 3, "rows": 2, "panels": ATLAS_PANELS},
        "mapped_components": components, "prompt": prompt,
        "scope": "Generate with the desktop image tool, then apply-texture; no image API is invoked by this module.",
    }
