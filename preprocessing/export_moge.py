"""Export MoGe-2/3 predictions using the AGILE RGB/depth/camera contract.

Run this module in the selected MoGe environment, not the optimizer environment.
Model imports are delayed so --help and the array helpers need no model package.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np


DEFAULT_MODELS = {
    "v2": "Ruicheng/moge-2-vitl-normal",
    "v3": "Ruicheng/moge-3-vitl",
}


def encode_depth_mm(depth_m: np.ndarray, validity: np.ndarray | None = None) -> np.ndarray:
    """Encode positive finite depth as uint16 mm; 0 and 65535 are invalid.

    Out-of-range predictions become 0, never clipped to an apparently valid
    surface. In particular, MoGe's masked +inf values must not become 65.535 m.
    """
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("Depth must be a two-dimensional image")
    valid = np.isfinite(depth) & (depth > 0) & (depth < 65.535)
    if validity is not None:
        mask = np.asarray(validity)
        if mask.shape != depth.shape:
            raise ValueError("Validity mask and depth must have the same shape")
        valid &= np.isfinite(mask) & (mask > 0)
    encoded = np.zeros(depth.shape, dtype=np.uint16)
    millimeters = np.rint(depth[valid] * 1000.0)
    encoded[valid] = np.where(
        (millimeters >= 1) & (millimeters < 65535), millimeters, 0
    ).astype(np.uint16)
    return encoded


def resize_nearest_2d(values: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """Resize depth/masks together without interpolating across invalid pixels."""
    values = np.asarray(values)
    if values.ndim != 2 or min(values.shape, default=0) <= 0:
        raise ValueError("Expected a nonempty two-dimensional image")
    height, width = output_shape
    if height <= 0 or width <= 0:
        raise ValueError("Output dimensions must be positive")
    rows = np.minimum(((np.arange(height) + 0.5) * values.shape[0] / height).astype(int), values.shape[0] - 1)
    cols = np.minimum(((np.arange(width) + 0.5) * values.shape[1] / width).astype(int), values.shape[1] - 1)
    return values[rows[:, None], cols[None, :]]


def normalized_intrinsics_to_pixels(intrinsics: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert normalized OpenCV intrinsics to the original RGB pixel grid."""
    camera = np.asarray(intrinsics, dtype=np.float64)
    if camera.shape != (3, 3) or not np.isfinite(camera).all():
        raise ValueError("Camera intrinsics must be a finite 3x3 matrix")
    if width <= 0 or height <= 0 or camera[0, 0] <= 0 or camera[1, 1] <= 0:
        raise ValueError("Image dimensions and focal lengths must be positive")
    if not np.allclose(camera[2], [0, 0, 1]):
        raise ValueError("Camera intrinsics must have bottom row [0, 0, 1]")
    return np.diag([width, height, 1.0]) @ camera


def prepare_prediction(prediction: dict, output_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Convert a NumPy model prediction into AGILE depth and pixel intrinsics."""
    if "mask" not in prediction:
        raise ValueError("MoGe must return a valid-pixel mask")
    encoded = encode_depth_mm(prediction["depth"], prediction["mask"])
    encoded = resize_nearest_2d(encoded, output_shape)
    height, width = output_shape
    camera = normalized_intrinsics_to_pixels(prediction["intrinsics"], width, height)
    return encoded, camera


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--img-path", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--version", choices=DEFAULT_MODELS, default="v2")
    parser.add_argument("--moge-repo", type=Path, help="Optional MoGe source checkout, including legacy non-installed forks")
    parser.add_argument("--pretrained", help="Hugging Face model ID or local checkpoint; defaults to the selected version's ViT-L")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-size", type=int, default=640, help="Maximum input long edge; 0 preserves input size")
    parser.add_argument("--resolution-level", type=int, choices=range(10), default=9)
    parser.add_argument("--refine-steps", type=int, default=3, help="Sparse refinement steps for v3 only")
    parser.add_argument("--fov-x", type=float, help="Known horizontal field of view in degrees, shared by every frame")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=None,
                        help="Override mixed precision; default is v2 FP16 / v3 FP32, matching their upstream infer defaults")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if not args.scene_name or Path(args.scene_name).name != args.scene_name or args.scene_name in (".", ".."):
        raise ValueError("scene-name must be one directory name")
    if args.max_size < 0 or args.refine_steps < 0:
        raise ValueError("max-size and refine-steps must be nonnegative")
    if args.fov_x is not None and not (0 < args.fov_x < 180):
        raise ValueError("fov-x must lie strictly between 0 and 180 degrees")
    if args.moge_repo is not None:
        repository = args.moge_repo.resolve(strict=True)
        if not (repository / "moge" / "model" / f"{args.version}.py").is_file():
            raise ValueError(f"MoGe repository does not contain moge/model/{args.version}.py: {repository}")
        sys.path.insert(0, str(repository))
    image_dir = args.img_path.resolve(strict=True)
    if not image_dir.is_dir():
        raise ValueError("img-path must be a directory")
    images = sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not images:
        raise ValueError("No input RGB images found")
    output = (args.outdir / args.scene_name).resolve()
    if output == image_dir or output in image_dir.parents or image_dir in output.parents:
        raise ValueError("Input and output directories must be separate and non-nested")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output must be new or empty: {output}")

    from PIL import Image
    import torch

    sizes = []
    for path in images:
        with Image.open(path) as source:
            sizes.append(source.size)
    if any(size != sizes[0] for size in sizes):
        raise ValueError("All frames must have the same size for AGILE's single cam_K.txt")
    model_module = importlib.import_module(f"moge.model.{args.version}")
    model_type = model_module.MoGeModel
    pretrained = args.pretrained or DEFAULT_MODELS[args.version]
    use_fp16 = args.fp16 if args.fp16 is not None else args.version == "v2"
    model = model_type.from_pretrained(pretrained).to(args.device).eval()
    for name in ("rgb", "depth", "valid_mask", "intrinsics"):
        (output / name).mkdir(parents=True, exist_ok=True)

    cameras, frames = [], []
    for index, path in enumerate(images):
        with Image.open(path) as source:
            rgb_image = source.convert("RGB")
        width, height = rgb_image.size
        inference_image = rgb_image.copy()
        if args.max_size and max(width, height) > args.max_size:
            inference_image.thumbnail((args.max_size, args.max_size), Image.Resampling.BOX)
        rgb = np.asarray(inference_image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(args.device)
        kwargs = {"resolution_level": args.resolution_level, "apply_mask": True, "fov_x": args.fov_x, "use_fp16": use_fp16}
        if args.version == "v3":
            kwargs["refine_steps"] = args.refine_steps
        prediction = model.infer(tensor, **kwargs)
        arrays = {name: prediction[name].detach().cpu().numpy() for name in ("depth", "mask", "intrinsics")}
        depth, camera = prepare_prediction(arrays, (height, width))
        if not np.any(depth):
            raise ValueError(f"No valid depth pixels for input {path.name}")
        filename = f"{index:05d}.png"
        rgb_image.save(output / "rgb" / filename)
        Image.fromarray(depth).save(output / "depth" / filename)
        Image.fromarray((depth > 0).astype(np.uint8) * 255).save(output / "valid_mask" / filename)
        np.savetxt(output / "intrinsics" / f"{index:05d}.txt", camera, fmt="%.8f")
        cameras.append(camera)
        frames.append({"source": path.name, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "output": filename, "valid_fraction": float(np.mean(depth > 0))})
        print(f"{index + 1}/{len(images)}: {path.name}, valid depth {frames[-1]['valid_fraction']:.1%}", flush=True)

    camera_array = np.stack(cameras)
    mean_camera = camera_array.mean(axis=0)
    np.savetxt(output / "cam_K.txt", mean_camera, fmt="%.8f")
    try:
        package_version = importlib.metadata.version("moge")
    except importlib.metadata.PackageNotFoundError:
        package_version = None
    metadata = {
        "model_version": args.version,
        "pretrained": str(pretrained),
        "moge_package_version": package_version,
        "model_source_sha256": hashlib.sha256(Path(model_module.__file__).read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
        "depth_unit": "millimeter",
        "invalid_depth": 0,
        "reserved_depth": 65535,
        "depth_resize": "nearest; validity preserved",
        "intrinsics_policy": "mean of per-frame pixel intrinsics (legacy AGILE contract)",
        "max_relative_focal_deviation": float(np.max(np.abs(camera_array[:, [0, 1], [0, 1]] / mean_camera[[0, 1], [0, 1]] - 1))),
        "fov_x_degrees": args.fov_x,
        "max_size": args.max_size,
        "resolution_level": args.resolution_level,
        "refine_steps": args.refine_steps if args.version == "v3" else None,
        "fp16": use_fp16,
        "frames": frames,
    }
    (output / "export_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(f"Export complete: {output}")
    if args.fov_x is None and len(images) > 1:
        print("Per-frame intrinsics are preserved in intrinsics/. cam_K.txt uses their mean for legacy readers; check focal variation before alignment.")


if __name__ == "__main__":
    main()
