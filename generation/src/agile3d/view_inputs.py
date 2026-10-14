"""Portable, hash-checked named views. No controller/CUDA dependencies."""

import hashlib
import json
from pathlib import Path

from PIL import Image

VIEW_NAMES = ("front", "back", "left", "right")


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_view_manifest(path, expected_sha256=None):
    path = Path(path).resolve()
    if expected_sha256 and file_digest(path) != expected_sha256:
        raise ValueError("View manifest hash mismatch")
    manifest = json.loads(path.read_text())
    entries = manifest.get("views", [])
    if manifest.get("schema_version") != 1 or not 2 <= len(entries) <= 3:
        raise ValueError("Partial conditioning requires exactly 2 or 3 named views")
    images = {}
    for entry in entries:
        name, filename = entry["name"], entry["file"]
        if name not in VIEW_NAMES or name in images:
            raise ValueError("View names must be distinct canonical directions")
        if Path(filename).name != filename or not filename.endswith(".png"):
            raise ValueError("Selected view files must be adjacent PNG basenames")
        image_path = (path.parent / filename).resolve()
        if (
            image_path.parent != path.parent
            or file_digest(image_path) != entry["sha256"]
        ):
            raise ValueError("Selected view file escaped its bundle or changed")
        with Image.open(image_path) as image:
            if (
                image.format != "PNG"
                or not 32 <= min(image.size) <= max(image.size) <= 4096
            ):
                raise ValueError("Invalid selected PNG dimensions")
            images[name] = image.convert("RGBA" if "A" in image.getbands() else "RGB")
    return {name: images[name] for name in VIEW_NAMES if name in images}
