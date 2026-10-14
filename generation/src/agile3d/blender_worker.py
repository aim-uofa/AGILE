"""Standalone Blender worker. No generated Python is executed.

Blender 4.5 API; failures are explicit, with no silent voxel-remesh fallback.
"""

import json
import math
import sys
from pathlib import Path

import bpy
import bmesh
import numpy as np
from mathutils import Matrix


DIRECTIONS = [
    ("front -Y", (1, 0, 0), (0, 0, 1)),
    ("back +Y", (-1, 0, 0), (0, 0, 1)),
    ("left -X", (0, -1, 0), (0, 0, 1)),
    ("right +X", (0, 1, 0), (0, 0, 1)),
    ("top +Z", (1, 0, 0), (0, 1, 0)),
    ("bottom -Z", (1, 0, 0), (0, -1, 0)),
]


def active(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def triangulated_count(obj):
    return sum(len(p.vertices) - 2 for p in obj.data.polygons)


def repair_volume(target, size, reason):
    active(target)
    target.data.remesh_voxel_size = size
    result = bpy.ops.object.voxel_remesh()
    if result != {"FINISHED"}:
        raise RuntimeError("Native volume repair failed")
    bpy.context.view_layer.update()
    return {"method": "voxel", "voxel_size": size, "reason": reason}


def semantic_uv(obj):
    """Mark seams using the actual surface, unwrap, orient and pack fixed islands."""
    active(obj)
    mesh = obj.data
    for layer in list(mesh.uv_layers):
        mesh.uv_layers.remove(layer)
    mesh.uv_layers.new(name="AGILE_UV")
    groups, edge_faces = [], {}
    for p in mesh.polygons:
        axis = max(range(3), key=lambda i: abs(p.normal[i]))
        group = {
            (0, False): 2,
            (0, True): 3,
            (1, False): 0,
            (1, True): 1,
            (2, False): 5,
            (2, True): 4,
        }[axis, p.normal[axis] >= 0]
        groups.append(group)
        for edge in p.edge_keys:
            edge_faces.setdefault(tuple(sorted(edge)), []).append(p.index)
    adjacency = [set() for _ in mesh.polygons]
    for edge in mesh.edges:
        faces = edge_faces[tuple(sorted(edge.vertices))]
        edge.use_seam = len(faces) != 2 or groups[faces[0]] != groups[faces[-1]]
        if not edge.use_seam:
            a, b = faces
            adjacency[a].add(b)
            adjacency[b].add(a)
    islands, remaining = [], set(range(len(mesh.polygons)))
    while remaining:
        seed = min(remaining)
        stack, faces = [seed], []
        remaining.remove(seed)
        while stack:
            f = stack.pop()
            faces.append(f)
            for neighbor in adjacency[f] & remaining:
                remaining.remove(neighbor)
                stack.append(neighbor)
        islands.append({"faces": sorted(faces), "direction": groups[seed]})
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.unwrap(method="ANGLE_BASED", correct_aspect=False, margin=0.01)
    bpy.ops.object.mode_set(mode="OBJECT")
    # Align each genuine unwrap with a documented canonical viewing direction.
    uv = mesh.uv_layers.active.data
    for island in islands:
        loops = [li for f in island["faces"] for li in mesh.polygons[f].loop_indices]
        points = np.array([tuple(uv[li].uv) for li in loops])
        coords = np.array(
            [tuple(mesh.vertices[mesh.loops[li].vertex_index].co) for li in loops]
        )
        _, right, up = DIRECTIONS[island["direction"]]
        target = coords @ np.array([right, up]).T
        centered = points - points.mean(0)
        u, _, vt = np.linalg.svd(centered.T @ (target - target.mean(0)))
        rotated = centered @ (u @ vt)
        for li, point in zip(loops, rotated):
            uv[li].uv = point
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.uv.select_all(action="SELECT")
    bpy.ops.uv.average_islands_scale()
    bpy.ops.uv.pack_islands(rotate=False, margin=0.012)
    bpy.ops.object.mode_set(mode="OBJECT")
    uv = mesh.uv_layers.active.data
    result = []
    for i, island in enumerate(islands):
        polygons = [
            [[float(v) for v in uv[li].uv] for li in mesh.polygons[f].loop_indices]
            for f in island["faces"]
        ]
        points = np.array([v for p in polygons for v in p])
        result.append(
            {
                "id": i + 1,
                "direction": DIRECTIONS[island["direction"]][0],
                "polygons": polygons,
                "face_count": len(island["faces"]),
                "area": float(sum(mesh.polygons[f].area for f in island["faces"])),
                "uv_bounds": [points.min(0).tolist(), points.max(0).tolist()],
            }
        )
    return result


def smart_uv(obj):
    """Blender Smart UV Project, followed by extraction of its actual islands."""
    active(obj)
    mesh = obj.data
    for layer in list(mesh.uv_layers):
        mesh.uv_layers.remove(layer)
    mesh.uv_layers.new(name="AGILE_UV")
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(
        angle_limit=math.radians(66), island_margin=0.02, correct_aspect=False
    )
    bpy.ops.object.mode_set(mode="OBJECT")
    uv = mesh.uv_layers.active.data
    edges = {}
    adjacency = [set() for _ in mesh.polygons]
    for face in mesh.polygons:
        corners = {
            mesh.loops[li].vertex_index: tuple(uv[li].uv) for li in face.loop_indices
        }
        for pair in face.edge_keys:
            key = tuple(sorted(pair))
            ends = np.array([corners[v] for v in key])
            edges.setdefault(key, []).append((face.index, ends))
    for faces in edges.values():
        if len(faces) == 2 and np.max(np.abs(faces[0][1] - faces[1][1])) < 1e-6:
            a, b = faces[0][0], faces[1][0]
            adjacency[a].add(b)
            adjacency[b].add(a)
    remaining = set(range(len(mesh.polygons)))
    result = []
    while remaining:
        first = min(remaining)
        stack = [first]
        remaining.remove(first)
        faces = []
        while stack:
            face = stack.pop()
            faces.append(face)
            for neighbor in sorted(adjacency[face] & remaining):
                remaining.remove(neighbor)
                stack.append(neighbor)
        polygons = [
            [[float(v) for v in uv[li].uv] for li in mesh.polygons[f].loop_indices]
            for f in sorted(faces)
        ]
        points = np.array([v for polygon in polygons for v in polygon])
        normal = sum(
            (np.array(mesh.polygons[f].normal) * mesh.polygons[f].area for f in faces),
            np.zeros(3),
        )
        axis = int(np.argmax(np.abs(normal)))
        direction = {
            (0, False): 2,
            (0, True): 3,
            (1, False): 0,
            (1, True): 1,
            (2, False): 5,
            (2, True): 4,
        }[axis, bool(normal[axis] >= 0)]
        result.append(
            {
                "id": len(result) + 1,
                "direction": DIRECTIONS[direction][0],
                "polygons": polygons,
                "face_count": len(faces),
                "area": float(sum(mesh.polygons[f].area for f in faces)),
                "uv_bounds": [points.min(0).tolist(), points.max(0).tolist()],
                "orientation": "Smart UV Project orientation; use baked correspondence, not a fixed panel order",
            }
        )
    return result


def target_material(obj, size):
    image = bpy.data.images.new("AGILE_albedo", width=size, height=size, alpha=False)
    image.generated_color = (0.8, 0.8, 0.8, 1)
    mat = bpy.data.materials.new("AGILE_albedo")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    nodes.active = tex
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = 0.5
    mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return image


def base_color_as_emission(source):
    """Bake albedo independently of metallic/roughness and scene illumination."""
    for index, material in enumerate(list(source.data.materials)):
        material = material.copy()
        source.data.materials[index] = material
        nodes, links = material.node_tree.nodes, material.node_tree.links
        bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None:
            raise ValueError(
                "Base-color baking requires an imported glTF Principled material"
            )
        emission = nodes.new("ShaderNodeEmission")
        color = bsdf.inputs["Base Color"]
        if color.is_linked:
            links.new(color.links[0].from_socket, emission.inputs["Color"])
        else:
            emission.inputs["Color"].default_value = color.default_value
        output = next(
            n for n in nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output
        )
        links.new(emission.outputs["Emission"], output.inputs["Surface"])


def main(request):
    out = Path(request["output"])
    out.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=request["mesh"])
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise ValueError("No mesh imported")
    # glTF import converts Y-up to Blender Z-up. Legacy Codex GLBs already use
    # Z-up coordinates, so undo this only for that documented input convention.
    if request["up_axis"] == "Z":
        rotation = Matrix.Rotation(-math.pi / 2, 4, "X")
        worlds = [(o, rotation @ o.matrix_world.copy()) for o in meshes]
        for o, matrix in worlds:
            o.parent = None
            o.matrix_world = matrix
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.join()
    source = bpy.context.object
    source.name = "original_source"
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    before = triangulated_count(source)
    coordinates = np.array([tuple(v.co) for v in source.data.vertices])
    diagonal = float(np.linalg.norm(np.ptp(coordinates, axis=0)))
    bpy.ops.object.duplicate()
    target = bpy.context.object
    target.name = "prepared_surface"
    # Merge UV-split duplicates on the target geometry only. Source UVs remain.
    bm = bmesh.new()
    bm.from_mesh(target.data)
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=diagonal * 1e-7)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    nonmanifold_edges = sum(not e.is_manifold for e in bm.edges)
    diagnostic = {
        "nonmanifold_edges": nonmanifold_edges,
        "nonmanifold_vertices": sum(not v.is_manifold for v in bm.verts),
        "noncontiguous_edges": sum(not e.is_contiguous for e in bm.edges),
        "zero_area_faces": sum(f.calc_area() < diagonal**2 * 1e-14 for f in bm.faces),
    }
    bm.to_mesh(target.data)
    bm.free()
    target.data.update()
    # Rebuild target connectivity to discard imported split-normal/custom layers
    # and invalidate evaluated-mesh caches before QuadriFlow. Source is intact.
    clean = bpy.data.meshes.new("clean_target_connectivity")
    clean.from_pydata(
        [tuple(v.co) for v in target.data.vertices],
        [],
        [tuple(p.vertices) for p in target.data.polygons],
    )
    clean.update()
    target.data = clean
    bpy.context.view_layer.update()
    diagnostic.update(vertices=len(clean.vertices), polygons=len(clean.polygons))
    (out / "pre-remesh-diagnostic.json").write_text(json.dumps(diagnostic))
    method = request["config"]["method"]
    repair = None
    working_scale = 1.0
    if method == "quadriflow":
        # Blender's QuadriFlow gate uses an absolute 1e-4 coordinate threshold
        # for short edges. Normalize only the temporary target, then restore its
        # exact coordinate system before unwrapping, baking and export.
        working_scale = 100.0 / float(np.ptp(coordinates, axis=0).max())
        target.data.transform(Matrix.Scale(working_scale, 4))
        target.data.update()
        if (
            any(
                diagnostic[k]
                for k in (
                    "nonmanifold_edges",
                    "nonmanifold_vertices",
                    "noncontiguous_edges",
                    "zero_area_faces",
                )
            )
            and request["config"]["repair_nonmanifold"]
        ):
            repair = repair_volume(
                target, 100.0 / 256, "Local geometry diagnostics require volume repair"
            )
        status = bpy.ops.object.quadriflow_remesh(
            target_faces=request["config"]["target_faces"],
            use_mesh_symmetry=False,
            use_preserve_sharp=False,
            use_preserve_boundary=True,
            smooth_normals=True,
            seed=0,
        )
        if (
            status != {"FINISHED"}
            and repair is None
            and request["config"]["repair_nonmanifold"]
        ):
            repair = repair_volume(
                target,
                100.0 / 256,
                "QuadriFlow rejected input despite passing local manifold checks",
            )
            status = bpy.ops.object.quadriflow_remesh(
                target_faces=request["config"]["target_faces"],
                use_mesh_symmetry=False,
                use_preserve_sharp=False,
                use_preserve_boundary=True,
                smooth_normals=True,
                seed=0,
            )
        if status != {"FINISHED"}:
            raise RuntimeError(
                "QuadriFlow did not finish; no fallback geometry was substituted"
            )
        target.data.transform(Matrix.Scale(1.0 / working_scale, 4))
        target.data.update()
    elif method == "decimate":
        modifier = target.modifiers.new("explicit_decimation", "DECIMATE")
        modifier.ratio = min(1.0, request["config"]["target_faces"] / max(1, before))
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    elif method != "preserve":
        raise ValueError("Unsupported surface method")
    uv_method = request["config"].get("uv_method", "semantic")
    islands = smart_uv(target) if uv_method == "smart_project" else semantic_uv(target)
    img = target_material(target, request["config"]["texture_size"])
    baked = False
    if request["bake"]:
        base_color_as_emission(source)
        scene = bpy.context.scene
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = 1
        scene.render.bake.use_pass_direct = False
        scene.render.bake.use_pass_indirect = False
        scene.render.bake.use_pass_color = True
        scene.render.bake.use_selected_to_active = True
        scene.render.bake.cage_extrusion = diagonal * 0.03
        scene.render.bake.max_ray_distance = diagonal * 0.08
        scene.render.bake.margin = 8
        source.select_set(True)
        target.select_set(True)
        bpy.context.view_layer.objects.active = target
        # The operator has its own defaults; scene.render.bake alone is not
        # sufficient in background execution (it can bake the target into itself).
        bpy.ops.object.bake(
            type="EMIT",
            use_selected_to_active=True,
            cage_extrusion=diagonal * 0.03,
            max_ray_distance=diagonal * 0.08,
            margin=8,
            use_clear=True,
        )
        pixels = np.asarray(img.pixels[:], dtype=float).reshape(-1, 4)[:, :3]
        if pixels.max() < 1e-7:
            raise RuntimeError(
                "Blender returned an entirely black color bake; refusing to export it as success"
            )
        baked = True
    img.filepath_raw = str(out / "baked-albedo.png")
    img.file_format = "PNG"
    img.save()
    active(target)
    bpy.ops.object.shade_smooth()
    bpy.ops.export_scene.gltf(
        filepath=str(out / "prepared.glb"),
        export_format="GLB",
        use_selection=True,
        export_yup=request["up_axis"] == "Y",
        export_animations=False,
    )
    source.hide_set(True)
    source.hide_render = True
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "surface.blend"))
    record = {
        "blender": bpy.app.version_string,
        "method": method,
        "source_triangles": before,
        "target_triangles": triangulated_count(target),
        "target_polygons": len(target.data.polygons),
        "quad_polygons": sum(len(p.vertices) == 4 for p in target.data.polygons),
        "nonmanifold_repair": repair,
        "temporary_working_scale": working_scale,
        "uv_islands": len(islands),
        "texture_baked": baked,
        "bake_type": "base_color_via_emission" if baked else None,
        "up_axis_in_export": request["up_axis"],
        "cage_extrusion_relative_diagonal": 0.03,
        "uv_method": uv_method,
        "unwrap": "Blender Smart UV Project (66 degrees, margin .02)"
        if uv_method == "smart_project"
        else "dominant-normal seams, angle-based unwrap, oriented and packed without rotation",
        "islands": islands,
    }
    (out / "surface.json").write_text(json.dumps(record))


if __name__ == "__main__":
    main(json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text()))
