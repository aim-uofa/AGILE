"""Offline workflow fixture: no API, GPU, weights, or ground-truth benchmark.

Run: python examples/offline_demo.py --output /ABS/DEMO
The job stops for a real review; this script never auto-accepts a candidate.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from agile3d.mesh import render_views
from agile3d.models import ImageConfig, MeshConfig, PipelineConfig
from agile3d.pipeline import create_job, queue_candidate
from agile3d.store import Store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    texture = Image.new("RGB", (128, 128), (240, 80, 40))
    pixels = np.asarray(texture).copy()
    pixels[32:96, 32:96] = [40, 120, 240]
    texture = Image.fromarray(pixels)
    mesh = trimesh.creation.box(extents=(1, 2, 1))
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [0, 1], [1, 0], [1, 1]]),
        image=texture,
    )
    mesh_file = args.output / "fixture.glb"
    mesh.export(mesh_file)
    rendered = args.output / "fixture-reference.png"
    render_views(mesh_file, rendered)
    with Image.open(rendered) as sheet:
        grid = Image.new("RGB", (512, 512), "white")
        for i, (x, y) in enumerate([(0, 0), (256, 0), (512, 0), (0, 280)]):
            grid.paste(
                sheet.crop((x, y, x + 256, y + 256)), ((i % 2) * 256, (i // 2) * 256)
            )
    image_file = args.output / "fixture-fourview.png"
    grid.save(image_file)
    config = PipelineConfig(
        image=ImageConfig(provider="provided"), mesh=MeshConfig(backend="provided")
    )
    store = Store(args.output / "runs")
    job = create_job(
        store,
        [rendered],
        "Offline fixture: orange and blue textured rectangular block",
        config,
    )
    result = queue_candidate(
        store, job["id"], image=image_file, mesh=mesh_file, background=False
    )
    print(
        json.dumps(
            {
                "job_id": job["id"],
                "status": result["status"],
                "job_directory": str(store.directory(job["id"])),
                "note": "Synthetic plumbing fixture; not a reconstruction quality result.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
