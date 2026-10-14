"""Explicit, bounded provider calls. No retry loop hides extra billed requests."""

import base64
import hashlib
import io
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from PIL import Image

from .models import ImageConfig, MeshConfig


MULTIVIEW_PROMPT = """Reconstruct only the target object from the reference images.
Remove hands, people, occluders and the scene background. Preserve the observed
shape, proportions, markings and materials. Do not invent text or accessories.
Return ONE square image arranged as a 2 by 2 grid of four equal square views:
top-left front, top-right back, bottom-left left, bottom-right right. Use the same
object, scale, vertical axis, and neutral lighting in all four views on a plain
white background, without captions, view labels, borders or shadows. Infer hidden
surfaces conservatively. The previous candidate, when supplied, is an editable
draft; the original references remain the source of truth.
"""


def generate_image(
    config: ImageConfig,
    references: list[Path],
    prompt: str,
    output: Path,
    previous: Path | None = None,
) -> dict:
    if config.provider == "provided":
        raise ValueError("This profile requires an imported four-view image")
    if config.provider == "disabled":
        raise ValueError("Image generation is disabled for this profile")
    if not config.model or not config.model.strip():
        raise ValueError("Set image.model to your image-generation model before running")
    key = os.environ.get(config.api_key_env)
    if not key:
        raise ValueError(f"Set {config.api_key_env} in the worker environment")
    paths = [*references, *([previous] if previous else [])]
    request_id = None
    if config.api == "interactions":
        # Call REST directly: google-genai 1.75 normalizes a zero retry budget
        # into one retry for Interactions. httpx sends exactly one POST.
        import httpx

        inputs = [{"type": "text", "text": prompt}]
        inputs += [
            {
                "type": "image",
                "mime_type": "image/png",
                "data": base64.b64encode(p.read_bytes()).decode(),
            }
            for p in paths
        ]
        base = (config.base_url or "https://generativelanguage.googleapis.com").rstrip(
            "/"
        )
        endpoint = base + (
            "/interactions"
            if base.endswith(("/v1", "/v1beta"))
            else "/v1beta/interactions"
        )
        with httpx.Client(timeout=config.timeout_seconds) as client:
            response = client.post(
                endpoint,
                headers={"x-goog-api-key": key},
                json={
                    "model": config.model,
                    "input": inputs,
                    "store": False,
                    "response_format": {
                        "type": "image",
                        "mime_type": "image/png",
                        "aspect_ratio": "1:1",
                    },
                },
            )
            response.raise_for_status()
            body = response.json()
        if body.get("status") != "completed":
            raise RuntimeError("Image interaction did not complete")
        blocks = list(body.get("outputs") or [])  # Earlier Interactions response.
        for step in body.get("steps") or []:
            if step.get("type") == "model_output":
                blocks.extend(step.get("content") or [])
        if body.get("output_image"):
            blocks.append(body["output_image"])
        images = [
            block
            for block in blocks
            if block.get("type", "image") == "image" and block.get("data")
        ]
        if not images:
            raise RuntimeError("Image provider returned no inline output image")
        data = base64.b64decode(images[-1]["data"], validate=True)
        request_id = body.get("id")
    else:
        from google import genai
        from google.genai import types

        options = {
            "timeout": config.timeout_seconds * 1000,
            "retry_options": {"attempts": 1},
        }
        if config.base_url:
            options["base_url"] = config.base_url
        client = genai.Client(api_key=key, http_options=options)
        try:
            images = [Image.open(p).copy() for p in paths]
            response = client.models.generate_content(
                model=config.model,
                contents=[prompt, *images],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(aspect_ratio="1:1"),
                ),
            )
            image_parts = [
                p.inline_data
                for p in (response.parts or [])
                if p.inline_data and p.inline_data.mime_type.startswith("image/")
            ]
            if not image_parts:
                raise RuntimeError("Image provider returned no output image")
            data = image_parts[0].data
            request_id = getattr(response, "response_id", None)
        finally:
            client.close()
    with Image.open(io.BytesIO(data)) as image:
        image.convert("RGB").save(output, format="PNG")
    return {
        "provider": "gemini",
        "model": config.model,
        "api": config.api,
        "request_id": request_id,
    }


def run_process(argv: list[str], log: Path, timeout: int, *, cwd=None, env=None):
    """Timeout terminates the whole local process group, including GPU children."""
    with log.open("ab") as stream:
        process = subprocess.Popen(
            argv,
            stdout=stream,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            env=env,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
    if code != 0:
        raise RuntimeError(f"Worker exited with code {code}; see {log.name}")


def generate_mesh(
    config: MeshConfig,
    image: Path,
    output: Path,
    seed: int,
    *,
    view_manifest: Path | None = None,
):
    if config.backend == "provided":
        raise ValueError("This profile requires an imported GLB mesh")
    if config.backend != "hunyuan":
        raise ValueError("This worker requires mesh.backend=hunyuan")
    from .hunyuan_worker import validate_model_selection

    validate_model_selection(config.model_dump())
    request = output.parent / "hunyuan-request.json"
    request.write_text(
        json.dumps(
            {
                "config": config.model_dump(),
                "image": str(image),
                "output": str(output),
                "seed": seed,
                **(
                    {
                        "view_manifest": str(view_manifest),
                        "view_manifest_sha256": hashlib.sha256(
                            view_manifest.read_bytes()
                        ).hexdigest(),
                    }
                    if view_manifest is not None
                    else {}
                ),
            },
            indent=2,
        )
    )
    run_process(
        [
            config.python or sys.executable,
            str(Path(__file__).with_name("hunyuan_worker.py")),
            str(request),
        ],
        output.parent / "hunyuan.log",
        config.timeout_seconds,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )
    if not output.is_file():
        raise RuntimeError("Hunyuan finished without producing a GLB")
