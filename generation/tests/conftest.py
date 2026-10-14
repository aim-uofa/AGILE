import numpy as np
import pytest
import trimesh
from PIL import Image
from agile3d.models import ImageConfig, MeshConfig, PipelineConfig
from agile3d.pipeline import create_job
from agile3d.store import Store


@pytest.fixture
def sample(tmp_path):
    image = tmp_path / "views.png"
    pixels = np.zeros((128, 128, 3), dtype=np.uint8)
    pixels[:64, :64] = [240, 30, 30]
    pixels[:64, 64:] = [30, 240, 30]
    pixels[64:, :64] = [30, 30, 240]
    pixels[64:, 64:] = [220, 220, 20]
    Image.fromarray(pixels).save(image)
    mesh = trimesh.creation.box(extents=(1, 2, 1))
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [0, 1], [1, 0], [1, 1]]),
        image=Image.fromarray(pixels),
    )
    path = tmp_path / "input.glb"
    mesh.export(path)
    config = PipelineConfig(
        image=ImageConfig(provider="provided"), mesh=MeshConfig(backend="provided")
    )
    store = Store(tmp_path / "runs")
    job = create_job(
        store, [image], "Red, green and blue toy block, without hands", config
    )
    return store, job["id"], image, path
