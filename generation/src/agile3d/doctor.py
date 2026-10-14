"""Offline checks for every configured generation stage, before provider calls."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from pydantic import ValidationError

from .models import PipelineConfig


class PreflightError(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__(format_report(report))


def _placeholder(value):
    return not isinstance(value, str) or not value.strip() or any(
        marker in value.upper() for marker in ("YOUR_", "/ABS/", "<MODEL", "REPLACE_ME")
    )


def _executable(value):
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    if path.is_file() and os.access(path, os.X_OK):
        return str(path.resolve())
    return shutil.which(value)


def _installed(module):
    try:
        return importlib.util.find_spec(module) is not None
    except (ModuleNotFoundError, ValueError):
        return False


def inspect_config(config, *, frames=None):
    """Return actionable checks without accessing an API, CUDA, or model weights.

    A worker interpreter is queried only for Python version and module locations.
    Credentials are checked for presence; their values never enter this report.
    """
    checks = []

    def add(name, status, detail, action=None):
        checks.append({"name": name, "status": status, "detail": detail, **({"action": action} if action else {})})

    raw = config.model_dump() if isinstance(config, PipelineConfig) else config
    try:
        parsed = PipelineConfig.model_validate(raw)
        values = parsed.model_dump()
        add("config", "ok", "Configuration schema is valid.")
    except ValidationError as exc:
        values = PipelineConfig().model_dump()
        if isinstance(raw, dict):
            for key, value in raw.items():
                if key in values and isinstance(value, dict):
                    values[key].update(value)
        for error in exc.errors(include_input=False, include_url=False):
            field = ".".join(str(part) for part in error["loc"]) or "config"
            add(field, "error", error["msg"], "Correct the profile and run doctor again.")

    add("python", "ok" if sys.version_info >= (3, 10) else "error", "Python 3.10 or newer is required.")
    add("environment", "ok" if os.environ.get("CONDA_PREFIX") else "warning",
        "Conda environment is active." if os.environ.get("CONDA_PREFIX") else "No active Conda environment was detected; the current interpreter will be used.")

    def selected(name, value):
        valid = not _placeholder(value)
        add(name, "ok" if valid else "error", "Explicit selection is configured." if valid else "A model or path placeholder is unset.",
            None if valid else f"Set {name} in your profile.")
        return valid

    def executable(name, value):
        path = None if _placeholder(value) else _executable(value)
        add(name, "ok" if path else "error", f"Executable found: {path}" if path else "Executable was not found or is not executable.",
            None if path else f"Install the tool or set {name} to its executable path.")
        return path

    def credential(name):
        # Only valid environment variable names are reflected into diagnostics.
        import re
        if not isinstance(name, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
            return
        present = bool(os.environ.get(name))
        add(name, "ok" if present else "error", "Credential is present (value not displayed)." if present else "Credential is not set.",
            None if present else f"Set {name} in the active shell; do not save it in the profile.")

    def package(name):
        present = _installed(name)
        add(name, "ok" if present else "error", "Python module is available." if present else "Python module is missing.",
            None if present else "Install the generation package and the applicable extra in this Conda environment.")

    for name in ("numpy", "scipy", "PIL", "trimesh", "pydantic", "filelock"):
        package(name)
    agent, image, mesh, surface = (values[name] for name in ("agent", "image", "mesh", "surface"))
    if agent["backend"] == "vlm":
        selected("agent.model", agent.get("model"))
        selected("agent.base_url", agent.get("base_url"))
        credential(agent.get("api_key_env"))
        add("vlm_access", "warning", "Endpoint access, model support, quota and credentials are not tested offline.")
    else:
        executable("agent.executable", agent.get("executable"))
        add("codex_access", "warning", "Run codex login status to verify authentication. Doctor does not invoke Codex or inspect stored credentials.")
    if image["provider"] == "gemini":
        selected("image.model", image.get("model"))
        credential(image.get("api_key_env"))
        package("httpx" if image.get("api") == "interactions" else "google.genai")
        add("image_access", "warning", "Image API access, model support and quota are not tested offline.")
    elif image["provider"] == "provided":
        add("image_input", "warning", "This route pauses for a supplied multi-view image; no image API credential is required.")
    if surface["enabled"]:
        executable("surface.executable", surface.get("executable"))
    if mesh["backend"] == "hunyuan":
        python = executable("mesh.python", mesh.get("python") or sys.executable)
        repository = mesh.get("repository")
        repository_path = Path(repository).expanduser() if isinstance(repository, str) else None
        if repository is not None:
            valid = not _placeholder(repository) and (repository_path / "hy3dgen").is_dir()
            add("mesh.repository", "ok" if valid else "error", "Hunyuan source directory contains hy3dgen/." if valid else "Hunyuan source directory must contain hy3dgen/.",
                None if valid else "Install Hunyuan and set mesh.repository to its source checkout.")
        for name in ("shape_model", "subfolder", "texture_model"):
            value = mesh.get(name)
            if not selected("mesh." + name, value) or name == "subfolder":
                continue
            path = Path(value).expanduser()
            local = value.startswith(("/", "./", "../", "~")) or path.exists()
            if local:
                if name == "shape_model" and isinstance(mesh.get("subfolder"), str):
                    path = path / mesh["subfolder"]
                add("mesh." + name + ".files", "ok" if path.is_dir() and any(path.iterdir()) else "error",
                    "Local model directory exists and is not empty." if path.is_dir() and any(path.iterdir()) else "Local model directory is missing or empty.",
                    "Check the model files and, for shape weights, the configured subfolder.")
            else:
                add("mesh." + name + ".files", "warning", "Remote model identifier: availability, access and cached weights cannot be confirmed offline; a run may download weights.")
        if python:
            # No torch/hy3dgen import: finding top-level modules does not start CUDA
            # or invoke from_pretrained. Hide arbitrary interpreter stderr.
            probe = (
                "import importlib.util,json,sys; "
                "sys.path.insert(0,sys.argv[1]) if sys.argv[1] else None; "
                "print(json.dumps({'version':list(sys.version_info[:2]),"
                "'modules':{m:importlib.util.find_spec(m) is not None for m in "
                "['torch','numpy','PIL','diffusers','transformers','safetensors','hy3dgen']}}))"
            )
            try:
                result = subprocess.run([python, "-c", probe, str(repository_path or "")], capture_output=True, text=True, timeout=15,
                                        env={**os.environ, "PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
                found = json.loads(result.stdout) if result.returncode == 0 else None
                if not isinstance(found, dict):
                    raise ValueError("Invalid interpreter probe")
                missing = [name for name, available in found["modules"].items() if not available]
                valid = found["version"] >= [3, 10] and not missing
                add("mesh.runtime", "ok" if valid else "error", "Worker Python and core modules are available." if valid else "Worker needs Python >=3.10 and missing modules: " + (", ".join(missing) or "none; check Python version"),
                    None if valid else "Complete installation in the Hunyuan Conda environment.")
            except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired):
                add("mesh.runtime", "error", "Could not inspect the worker interpreter within 15 seconds.", "Check mesh.python and its Conda environment.")
        add("mesh.gpu", "warning", "CUDA compatibility, compiled extensions, free GPU memory and model loading require the documented runtime smoke test; doctor does not allocate a GPU.")
    elif mesh["backend"] == "provided":
        add("mesh_input", "warning", "This route pauses for a supplied GLB; Hunyuan is not required.")
    if values["texture"]["provider"] == "provided":
        add("texture_input", "warning", "This route pauses for a generated texture matching its saved UV request.")
    if frames is not None:
        from PIL import Image
        for index, frame in enumerate(frames):
            try:
                with Image.open(frame) as opened:
                    opened.verify()
                add(f"frame[{index}]", "ok", "Reference image is readable.")
            except (OSError, ValueError):
                add(f"frame[{index}]", "error", "Reference image is missing or invalid.", "Use an existing readable image file.")
    return {"status": "ready" if all(item["status"] != "error" for item in checks) else "blocked", "offline": True, "checks": checks}


def inspect_file(path, *, frames=None):
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {"status": "blocked", "offline": True, "checks": [{"name": "config", "status": "error", "detail": "Profile is not a readable JSON file."}]}
    return inspect_config(raw, frames=frames)


def format_report(report):
    lines = ["Generation preflight: " + report["status"], "Offline checks only; no provider calls or model downloads."]
    for item in report["checks"]:
        lines.append(f"[{item['status'].upper()}] {item['name']}: {item['detail']}" + (" " + item["action"] if item.get("action") else ""))
    return "\n".join(lines)


def require_ready(config, *, frames=None):
    report = inspect_config(config, frames=frames)
    if report["status"] != "ready":
        raise PreflightError(report)
    return report
