"""Standalone worker: uses Hunyuan's Python without importing agile3d or MCP.

The control environment and the CUDA environment can have different dependencies.
"""

import gc
import json
import sys
from pathlib import Path


def validate_model_selection(config):
    """Require user-selected weights before importing or starting a model worker."""
    missing = [
        name for name in ("shape_model", "subfolder", "texture_model")
        if not isinstance(config.get(name), str) or not config[name].strip()
    ]
    if missing:
        raise ValueError(
            "Set " + ", ".join("mesh." + name for name in missing)
            + " to your model locations before running"
        )


def load_texture_pipeline(pipeline_class, model_path, *, device, low_vram):
    """Load texture weights on CPU before installing upstream offload hooks.

    Upstream's from_pretrained creates both diffusion pipelines on CUDA before
    enable_model_cpu_offload can be called. Override the construction hook only
    in this worker; the installed Hunyuan source remains unchanged.
    """
    if not low_vram:
        paint = pipeline_class.from_pretrained(model_path)
        return paint

    class CpuLoadedPaintPipeline(pipeline_class):
        def load_models(self):
            target_device = device
            self.config.device = "cpu"
            try:
                super().load_models()
            finally:
                self.config.device = target_device
            # The wrapper does tensor preprocessing too, outside its diffusion
            # pipeline. Those tensors must use the offload execution device.
            for model in self.models.values():
                if hasattr(model, "device"):
                    model.device = target_device

    paint = CpuLoadedPaintPipeline.from_pretrained(model_path)
    paint.enable_model_cpu_offload(device=device)
    return paint


def main(request_path):
    request = json.loads(Path(request_path).read_text())
    config = request["config"]
    validate_model_selection(config)
    multiview = "mv" in (config["shape_model"] + config["subfolder"])
    if request.get("view_manifest") and not multiview:
        raise ValueError("Named-view subsets require a multi-view Hunyuan shape model")
    if config["repository"]:
        repository = Path(config["repository"]).expanduser().resolve()
        if not (repository / "hy3dgen").is_dir():
            raise ValueError("Hunyuan repository must contain hy3dgen/")
        sys.path.insert(0, str(repository))
    import torch
    from PIL import Image
    from hy3dgen.rembg import BackgroundRemover
    from hy3dgen.shapegen import (
        Hunyuan3DDiTFlowMatchingPipeline,
        FloaterRemover,
        DegenerateFaceRemover,
    )
    from hy3dgen.texgen import Hunyuan3DPaintPipeline

    remover = BackgroundRemover()
    if request.get("view_manifest"):
        # Direct script execution and normal package import both work in the
        # isolated Hunyuan interpreter, without importing the control runtime.
        sys.path.insert(0, str(Path(__file__).parent))
        from view_inputs import load_view_manifest

        inputs = load_view_manifest(
            request["view_manifest"], request["view_manifest_sha256"]
        )
        views = {name: remover(image) for name, image in inputs.items()}
    else:
        image = Image.open(request["image"]).convert("RGB")
        w, h = image.size
        boxes = [
            (0, 0, w // 2, h // 2),
            (w // 2, 0, w, h // 2),
            (0, h // 2, w // 2, h),
            (w // 2, h // 2, w, h),
        ]
        views = {
            name: remover(image.crop(box))
            for name, box in zip(("front", "back", "left", "right"), boxes)
        }
    del remover
    gc.collect()
    # Keep Hugging Face IDs intact; absolute paths are also accepted by upstream.
    shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        config["shape_model"],
        subfolder=config["subfolder"],
        use_safetensors=True,
        device=config["device"],
    )
    mesh = shape(
        image=views if multiview else views["front"],
        num_inference_steps=config["steps"],
        octree_resolution=config["octree_resolution"],
        generator=torch.Generator().manual_seed(request["seed"]),
    )[0]
    mesh = DegenerateFaceRemover()(FloaterRemover()(mesh))
    if config["low_vram"]:
        del shape
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    paint = load_texture_pipeline(
        Hunyuan3DPaintPipeline,
        config["texture_model"],
        device=config["device"],
        low_vram=config["low_vram"],
    )
    # Do not silently return a white mesh when texture generation fails.
    mesh = paint(mesh, image=list(views.values()))
    mesh.export(request["output"])


if __name__ == "__main__":
    main(sys.argv[1])
