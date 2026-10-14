import pytest
import trimesh
import numpy as np
from PIL import Image

from agile3d.codex_backend import MeshComponent, ProceduralMeshSpec
from agile3d.procedural import build_mesh
from agile3d.mesh import inspect_mesh, render_views
from agile3d.models import QualityPolicy


@pytest.mark.parametrize(
    "kind,profile",
    [
        ("box", []),
        ("ellipsoid", []),
        ("torus", []),
        ("lathe", [[0.4, -0.5], [0.5, 0.2], [0.2, 0.5]]),
        ("lathe", [[0.0, -0.5], [0.4, -0.5], [0.3, 0.5], [0.0, 0.5]]),
        (
            "lathe_shell",
            [
                [0.0, -0.5],
                [0.5, -0.5],
                [0.5, 0.5],
                [0.45, 0.5],
                [0.45, -0.43],
                [0.0, -0.43],
            ],
        ),
        (
            "extrusion",
            [
                [-0.5, -0.5],
                [0.5, -0.5],
                [0.5, 0.0],
                [0.0, 0.0],
                [0.0, 0.5],
                [-0.5, 0.5],
            ],
        ),
    ],
)
def test_geometry_and_texture_survive_export(tmp_path, kind, profile):
    component = MeshComponent(
        name="body",
        kind=kind,
        position=[0.0, 0.0, 0.0],
        rotation_deg=[0.0, 0.0, 0.0],
        size=[1.0, 1.0, 1.0],
        color=[0.8, 0.2, 0.1, 1.0],
        profile=profile,
        torus_ratio=0.25,
        label_text="Example",
        label_color=[1.0, 1.0, 1.0, 1.0],
    )
    spec = ProceduralMeshSpec(
        schema_version=1,
        object_name="example",
        approximation_scope="A synthetic test of topology and texture export.",
        assumptions=[],
        components=[component],
    )
    output = tmp_path / "mesh.glb"
    build_mesh(spec, output)
    quality = inspect_mesh(output, QualityPolicy(require_watertight=True))
    assert quality["passed"], quality
    assert quality["textured"]
    scene = trimesh.load(output, force="scene")
    assert scene.extents.max() <= 1.001


def test_label_is_visible_on_broad_face_of_rotated_thin_panel(tmp_path):
    values = dict(
        name="label",
        kind="box",
        position=[0.0, 0.0, 0.0],
        rotation_deg=[90.0, 0.0, 0.0],
        size=[1.0, 1.0, 0.01],
        color=[1.0, 1.0, 1.0, 1.0],
        profile=[],
        torus_ratio=0.2,
        label_color=[0.0, 0.0, 0.0, 1.0],
    )
    renders = []
    for index, text in enumerate(["", "VISIBLE LABEL"]):
        spec = ProceduralMeshSpec(
            schema_version=1,
            object_name="label",
            approximation_scope="Synthetic rotated label panel projection test.",
            assumptions=[],
            components=[MeshComponent(**values, label_text=text)],
        )
        path = tmp_path / f"{index}.glb"
        build_mesh(spec, path)
        picture = tmp_path / f"{index}.png"
        render_views(path, picture, size=128)
        renders.append(np.asarray(Image.open(picture)).astype(int))
    # From -Y, the broad XY panel rotated 90 degrees must show the text.
    difference = renders[0][152:280, 256:384] - renders[1][152:280, 256:384]
    assert np.count_nonzero(np.max(difference, axis=2) > 80) > 30
