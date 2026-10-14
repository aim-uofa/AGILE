"""Bounded candidate loop with explicit review, durable stages and export gates."""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from . import providers
from .mesh import export_obj, inspect_mesh, render_views
from .models import PipelineConfig, Review
from .store import Store, atomic_json, now


def provide_texture(
    store: Store, job_id: str, image: Path, provenance: Path | None = None
) -> dict:
    """Freeze the generated atlas, then finish the SAME geometry candidate."""
    from .texture import read_atlas, validate_rectangles

    with store.lock(job_id):
        job = store.load(job_id)
        config = PipelineConfig.model_validate(job["config"])
        if job["status"] != "awaiting_texture" or config.texture.provider not in ("provided", "image_api"):
            raise ValueError("Candidate must be awaiting a generated texture atlas")
        if config.generation.texture_review:
            verify_evidence(store, job)
        # The old six-panel atlas and a packed UV square have different limits.
        # Validate before mutating durable state or consuming another stage.
        if config.texture.layout == "mesh_uv":
            from .surface import read_uv_texture

            read_uv_texture(image)
        else:
            read_atlas(image)
        attempt = job["attempts"][-1]
        directory = store.directory(job_id) / f"attempt-{attempt['number']:02d}"
        if config.generation.texture_review:
            from .generation import texture_directory

            directory = texture_directory(directory, attempt)
        for record in attempt["artifacts"].values():
            store.resolve(job_id, record)
        if "texture_atlas" in attempt["artifacts"]:
            raise ValueError("This candidate already has a frozen texture")
        metadata = {"provider": "provided", "model": "unreported"}
        if provenance is not None:
            if provenance.stat().st_size > 1024 * 1024:
                raise ValueError("Texture provenance exceeds 1 MiB")
            metadata = json.loads(provenance.read_text())
            if not isinstance(metadata, dict):
                raise ValueError("Texture provenance must be a JSON object")
            if (
                metadata.get("image_sha256")
                != hashlib.sha256(image.read_bytes()).hexdigest()
            ):
                raise ValueError(
                    "Texture provenance does not match the supplied bitmap"
                )
        validate_rectangles(metadata.get("panel_rectangles"))
        if config.texture.layout == "mesh_uv":
            if metadata.get("panel_rectangles") is not None:
                raise ValueError(
                    "Frozen mesh UVs cannot be replaced with panel rectangles"
                )
            for key in ("prepared_mesh", "uv_layout"):
                if metadata.get(key + "_sha256") != attempt["artifacts"][key]["sha256"]:
                    raise ValueError(
                        "Generated texture must name the frozen mesh and UV layout hashes"
                    )
            with Image.open(image) as generated:
                if generated.width != generated.height:
                    raise ValueError(
                        "Generated image must preserve the square UV layout"
                    )
        # Preserve the exact generated pixels and file bytes in the evidence.
        with Image.open(image) as source:
            if source.format != "PNG":
                raise ValueError(
                    "Supply a PNG atlas so the original generated bitmap is retained"
                )
        shutil.copyfile(image, directory / "texture-atlas.png")
        atomic_json(directory / "texture-provenance.json", metadata)
        for name, filename in (
            ("texture_atlas", "texture-atlas.png"),
            ("texture_provenance", "texture-provenance.json"),
        ):
            attempt["artifacts"][name] = store.artifact(job_id, directory / filename)
        token = uuid.uuid4().hex
        job.update(status="queued", queue_token=token)
        store.save(job)
    return run_candidate(store, job_id, token)


def copy_image(source: Path, destination: Path):
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((4096, 4096))
        image.save(destination, format="PNG")


def contact_sheet(paths: list[Path], output: Path):
    sheet = Image.new(
        "RGB", (320 * min(4, len(paths)), 264 * ((len(paths) + 3) // 4)), "white"
    )
    for i, path in enumerate(paths):
        with Image.open(path) as image:
            tile = ImageOps.contain(image.convert("RGB"), (320, 240))
        x, y = i % 4 * 320, i // 4 * 264
        sheet.paste(tile, (x + (320 - tile.width) // 2, y))
        ImageDraw.Draw(sheet).text((x + 5, y + 244), f"reference {i + 1}", fill="black")
    sheet.save(output)


def create_job(
    store: Store, references: list[Path], description: str, config: PipelineConfig
) -> dict:
    if not 1 <= len(references) <= 8:
        raise ValueError("Select 1 to 8 object reference images")
    if not 1 <= len(description.strip()) <= 4000:
        raise ValueError("Describe the target object in 1 to 4000 characters")
    job_id = uuid.uuid4().hex[:16]
    directory = store.directory(job_id)
    directory.mkdir()
    try:
        (directory / "references").mkdir()
        copied = []
        for i, source in enumerate(references):
            target = directory / "references" / f"{i + 1:02d}.png"
            copy_image(source, target)
            copied.append(store.artifact(job_id, target))
        contact_sheet(
            [store.resolve(job_id, a) for a in copied], directory / "references.png"
        )
        job = {
            "schema_version": 1,
            "id": job_id,
            "created_at": now(),
            "status": "ready",
            "description": description,
            "config": config.model_dump(),
            "references": copied,
            "reference_sheet": store.artifact(job_id, directory / "references.png"),
            "attempts": [],
            "accepted_attempt": None,
        }
        store.save(job)
        return job
    except BaseException:
        shutil.rmtree(directory)
        raise


def _error_message(exc: Exception) -> str:
    message = f"{type(exc).__name__}: {exc}"
    for name, value in os.environ.items():
        if len(value) > 7 and any(
            s in name.upper() for s in ("KEY", "TOKEN", "SECRET")
        ):
            message = message.replace(value, "[REDACTED]")
    return re.sub(r"\b(?:sk-|AIza)[A-Za-z0-9_-]+", "[REDACTED]", message)[:2000]


def queue_candidate(
    store: Store,
    job_id: str,
    *,
    image: Path | None = None,
    mesh: Path | None = None,
    resume: bool = False,
    retry_image: bool = False,
    background: bool = True,
) -> dict:
    with store.lock(job_id):
        job = store.load(job_id)
        config = PipelineConfig.model_validate(job["config"])
        if resume:
            if (
                job["status"] not in ("failed", "running", "queued")
                or not job["attempts"]
            ):
                raise ValueError(
                    "Only a failed or interrupted candidate can be resumed"
                )
            if image or mesh:
                raise ValueError(
                    "Resume reuses frozen inputs; start a new candidate to replace them"
                )
            attempt = job["attempts"][-1]
        else:
            if job["status"] not in ("ready", "failed"):
                raise ValueError(
                    f"Cannot start a candidate while job is {job['status']}"
                )
            if len(job["attempts"]) >= config.quality.max_attempts:
                raise ValueError(
                    "Attempt budget exhausted; resume an interrupted candidate or create a new job"
                )
            if config.generation.require_selection and "reference_selection" not in job:
                raise ValueError(
                    "This profile requires create-from-frames and frozen visual selection evidence"
                )
            if (
                config.image.provider == "provided"
                and image is None
                and not config.generation.multiview_review
            ):
                raise ValueError("Provide the four-view image for this offline profile")
            if config.generation.multiview_review and mesh is not None:
                raise ValueError(
                    "Use apply-mesh after grid review or subset selection; bind its provenance to the chosen inputs"
                )
            if (
                config.mesh.backend == "provided"
                and mesh is None
                and not config.generation.multiview_review
            ):
                raise ValueError("Provide a GLB for this offline profile")
            index = len(job["attempts"]) + 1
            directory = store.directory(job_id) / f"attempt-{index:02d}"
            directory.mkdir(exist_ok=True)
            attempt = {
                "number": index,
                "artifacts": {},
                "created_at": now(),
                "seed": config.mesh.seed + index - 1,
                "review": None,
                "image_call_started": False,
            }
            if image:
                copy_image(image, directory / "imported-image.png")
                attempt["input_image"] = store.artifact(
                    job_id, directory / "imported-image.png"
                )
            if mesh:
                if mesh.suffix.lower() != ".glb":
                    raise ValueError("Imported meshes must be self-contained GLB files")
                if mesh.stat().st_size > 512 * 1024**2:
                    raise ValueError("GLB exceeds the 512 MiB input limit")
                shutil.copyfile(mesh, directory / "imported-mesh.glb")
                attempt["input_mesh"] = store.artifact(
                    job_id, directory / "imported-mesh.glb"
                )
            job["attempts"].append(attempt)
        token = uuid.uuid4().hex
        job.update(status="queued", queue_token=token)
        attempt.pop("error", None)
        store.save(job)
    if background:
        args = [
            sys.executable,
            "-m",
            "agile3d.worker",
            "--root",
            str(store.root),
            "--job",
            job_id,
            "--token",
            token,
        ]
        if retry_image:
            args.append("--retry-image")
        try:
            with (store.directory(job_id) / "worker.log").open("ab") as stream:
                subprocess.Popen(
                    args,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
        except Exception as exc:
            with store.lock(job_id):
                job = store.load(job_id)
                job["status"] = "failed"
                job["attempts"][-1]["error"] = _error_message(exc)
                store.save(job)
            raise
        return job
    return run_candidate(store, job_id, token, retry_image=retry_image)


def run_candidate(store: Store, job_id: str, token: str, *, retry_image=False) -> dict:
    with store.lock(job_id):
        job = store.load(job_id)
        if job.get("queue_token") != token or job["status"] != "queued":
            raise ValueError("Candidate was already claimed or replaced")
        config = PipelineConfig.model_validate(job["config"])
        attempt = job["attempts"][-1]
        directory = store.directory(job_id) / f"attempt-{attempt['number']:02d}"
        artifacts = attempt["artifacts"]
        job.update(status="running", worker_pid=os.getpid())
        store.save(job)

        def checkpoint(name, path):
            artifacts[name] = store.artifact(job_id, path)
            store.save(job)

        try:
            references = [store.resolve(job_id, r) for r in job["references"]]
            for artifact in artifacts.values():
                store.resolve(job_id, artifact)
            if config.generation.multiview_review:
                from .generation import ensure_multiview

                if not ensure_multiview(store, job, config, directory, retry_image):
                    return job
            if "multiview" not in artifacts and config.image.provider != "disabled":
                attempt["phase"] = "image"
                store.save(job)
                image_path = directory / "multiview.png"
                if "input_image" in attempt:
                    shutil.copyfile(
                        store.resolve(job_id, attempt["input_image"]), image_path
                    )
                    attempt["image_source"] = {"provider": "provided"}
                else:
                    if attempt["image_call_started"] and not retry_image:
                        raise ValueError(
                            "Previous image request has unknown outcome. Resume with retry_image=true only if a new paid request is intended"
                        )
                    previous = job["attempts"][-2] if len(job["attempts"]) > 1 else None
                    feedback = (
                        (previous.get("review") or {}).get("feedback", "")
                        if previous
                        else ""
                    )
                    if previous:
                        feedback += "\nGeometry checks: " + json.dumps(
                            previous.get("quality", {})
                        )
                    prompt = (
                        providers.MULTIVIEW_PROMPT + "\nTarget: " + job["description"]
                    )
                    if feedback.strip():
                        prompt += (
                            "\nRepair the previous candidate using this review:\n"
                            + feedback
                        )
                    (directory / "prompt.txt").write_text(prompt)
                    checkpoint("prompt", directory / "prompt.txt")
                    old_image = None
                    if previous and "multiview" in previous["artifacts"]:
                        old_image = store.resolve(
                            job_id, previous["artifacts"]["multiview"]
                        )
                    attempt["image_call_started"] = True
                    store.save(job)  # Durable before the external side effect.
                    attempt["image_source"] = providers.generate_image(
                        config.image, references, prompt, image_path, old_image
                    )
                with Image.open(image_path) as generated:
                    w, h = generated.size
                    if min(w, h) < 64 or w != h or w % 2:
                        raise ValueError(
                            "Four-view image must be square, even-sized, and at least 64 pixels"
                        )
                checkpoint("multiview", image_path)
            if "mesh" not in artifacts and "source_mesh" not in artifacts:
                if config.mesh.backend == "provided" and "input_mesh" not in attempt:
                    from .generation import mesh_request

                    mesh_request(store, job, directory)
                    return job
                attempt["phase"] = "mesh"
                store.save(job)
                mesh_path = directory / (
                    "source-mesh.glb" if config.surface.enabled else "mesh.glb"
                )
                if "input_mesh" in attempt:
                    shutil.copyfile(
                        store.resolve(job_id, attempt["input_mesh"]), mesh_path
                    )
                elif config.mesh.backend == "codex":
                    from .codex_backend import (
                        ProceduralMeshSpec,
                    )
                    from .procedural import build_mesh
                    from .agent_backend import get_agent_backend

                    if "geometry_spec" in artifacts:
                        spec = ProceduralMeshSpec.model_validate_json(
                            store.resolve(
                                job_id, artifacts["geometry_spec"]
                            ).read_text()
                        )
                    else:
                        if attempt.get("codex_call_started"):
                            raise ValueError(
                                "Previous Codex geometry call has unknown outcome; inspect its logs before starting a new candidate"
                            )
                        previous = (
                            job["attempts"][-2] if len(job["attempts"]) > 1 else None
                        )
                        feedback = (
                            (previous.get("review") or {}).get("feedback", "")
                            if previous
                            else ""
                        )
                        if previous:
                            if config.agent.review_context != "images_only":
                                feedback += "\nGeometry checks: " + json.dumps(
                                    previous.get("quality", {})
                                )
                            if "geometry_spec" in previous["artifacts"]:
                                feedback += (
                                    "\nPrevious specification: "
                                    + store.resolve(
                                        job_id, previous["artifacts"]["geometry_spec"]
                                    ).read_text()
                                )
                        attempt["codex_call_started"] = True
                        store.save(job)
                        spec, metadata = get_agent_backend(config.agent).propose_mesh(
                            references,
                            job["description"],
                            directory / "codex",
                            feedback=feedback,
                            max_components=config.quality.max_components,
                            **(
                                {"generated_texture": True}
                                if config.texture.provider in ("provided", "image_api")
                                else {}
                            ),
                            **(
                                {"mesh_uv": True}
                                if config.texture.layout == "mesh_uv"
                                else {}
                            ),
                        )
                        atomic_json(directory / "geometry-spec.json", spec.model_dump())
                        attempt["mesh_source"] = metadata
                        checkpoint("geometry_spec", directory / "geometry-spec.json")
                    if (
                        config.texture.provider in ("provided", "image_api")
                        and config.texture.layout == "atlas"
                        and "texture_atlas" not in artifacts
                    ):
                        from .texture import texture_request

                        previous = (
                            job["attempts"][-2] if len(job["attempts"]) > 1 else None
                        )
                        feedback = (
                            (previous.get("review") or {}).get("feedback", "")
                            if previous
                            else ""
                        )
                        request = texture_request(spec, job["description"], feedback)
                        atomic_json(directory / "texture-request.json", request)
                        checkpoint(
                            "texture_request", directory / "texture-request.json"
                        )
                        build_mesh(spec, directory / "untextured-mesh.glb")
                        checkpoint("untextured_mesh", directory / "untextured-mesh.glb")
                        render_views(
                            directory / "untextured-mesh.glb",
                            directory / "untextured-views.png",
                            up_axis="Z",
                        )
                        checkpoint(
                            "untextured_views", directory / "untextured-views.png"
                        )
                        attempt["phase"] = "texture"
                        job["status"] = "awaiting_texture"
                        return job
                    atlas = (
                        store.resolve(job_id, artifacts["texture_atlas"])
                        if "texture_atlas" in artifacts
                        and config.texture.layout == "atlas"
                        else None
                    )
                    layout = (
                        json.loads(
                            store.resolve(
                                job_id, artifacts["texture_provenance"]
                            ).read_text()
                        ).get("panel_rectangles")
                        if atlas is not None
                        else None
                    )
                    attempt["procedural"] = build_mesh(
                        spec,
                        mesh_path,
                        **(
                            {"texture_atlas": atlas, "texture_layout": layout}
                            if atlas is not None
                            else {}
                        ),
                    )
                else:
                    providers.generate_mesh(
                        config.mesh,
                        store.resolve(job_id, artifacts["multiview"]),
                        mesh_path,
                        attempt["seed"],
                        **(
                            {
                                "view_manifest": store.resolve(
                                    job_id, artifacts["view_manifest"]
                                )
                            }
                            if "view_manifest" in artifacts
                            else {}
                        ),
                    )
                checkpoint(
                    "source_mesh" if config.surface.enabled else "mesh", mesh_path
                )
            if config.surface.enabled and "mesh" not in artifacts:
                from .surface import prepare_surface, apply_uv_texture, mesh_uv_request

                up_axis = "Z" if config.mesh.backend == "codex" else "Y"
                if "prepared_mesh" not in artifacts:
                    attempt["phase"] = "surface"
                    store.save(job)
                    surface = directory / "surface"
                    record = prepare_surface(
                        store.resolve(job_id, artifacts["source_mesh"]),
                        surface,
                        config.surface,
                        up_axis=up_axis,
                        bake=config.mesh.backend != "codex",
                    )
                    for key, filename in [
                        ("prepared_mesh", "prepared.glb"),
                        ("uv_layout", "uv-layout.png"),
                        ("surface_record", "surface.json"),
                        ("surface_blend", "surface.blend"),
                        ("baked_albedo", "baked-albedo.png"),
                        ("untextured_mesh", "white.glb"),
                        ("untextured_views", "white-views.png"),
                    ]:
                        checkpoint(key, surface / filename)
                    if (surface / "white-oblique-views.png").is_file():
                        checkpoint(
                            "untextured_oblique_views",
                            surface / "white-oblique-views.png",
                        )
                    attempt["surface"] = {
                        k: v for k, v in record.items() if k != "islands"
                    }
                if config.surface.geometry_review and not attempt.get(
                    "geometry_review"
                ):
                    policy = config.quality.model_copy(
                        update={"require_texture": False}
                    )
                    attempt["geometry_quality"] = inspect_mesh(
                        store.resolve(job_id, artifacts["prepared_mesh"]), policy
                    )
                    atomic_json(
                        directory / "geometry-quality.json", attempt["geometry_quality"]
                    )
                    checkpoint("geometry_quality", directory / "geometry-quality.json")
                    attempt["phase"] = "geometry_review"
                    attempt["evidence_digest"] = evidence_digest(job)
                    job["status"] = "awaiting_geometry_review"
                    return job
                if config.generation.texture_review:
                    from .generation import ensure_texture

                    if not ensure_texture(store, job, config, directory):
                        return job
                if config.texture.provider in ("provided", "image_api"):
                    if "texture_atlas" not in artifacts:
                        record = json.loads(
                            store.resolve(
                                job_id, artifacts["surface_record"]
                            ).read_text()
                        )
                        request = mesh_uv_request(
                            job["description"],
                            record,
                            store.resolve(job_id, artifacts["prepared_mesh"]),
                            store.resolve(job_id, artifacts["uv_layout"]),
                            baked=store.resolve(job_id, artifacts["baked_albedo"])
                            if record["texture_baked"]
                            else None,
                        )
                        atomic_json(directory / "texture-request.json", request)
                        checkpoint(
                            "texture_request", directory / "texture-request.json"
                        )
                        attempt["phase"] = "texture"
                        job["status"] = "awaiting_texture"
                        return job
                    texture_output = directory
                    if config.generation.texture_review:
                        from .generation import texture_directory

                        texture_output = texture_directory(directory, attempt)
                    attempt["texture_mapping"] = apply_uv_texture(
                        store.resolve(job_id, artifacts["prepared_mesh"]),
                        store.resolve(job_id, artifacts["texture_atlas"]),
                        texture_output / "mesh.glb",
                    )
                else:
                    shutil.copyfile(
                        store.resolve(job_id, artifacts["prepared_mesh"]),
                        directory / "mesh.glb",
                    )
                checkpoint(
                    "mesh",
                    (
                        texture_output
                        if config.texture.provider in ("provided", "image_api")
                        else directory
                    )
                    / "mesh.glb",
                )
            evidence_output = directory
            if config.generation.texture_review:
                from .generation import texture_directory

                evidence_output = texture_directory(directory, attempt)
            if "quality" not in artifacts:
                attempt["phase"] = "inspect"
                store.save(job)
                report = inspect_mesh(
                    store.resolve(job_id, artifacts["mesh"]), config.quality
                )
                attempt["quality"] = report
                atomic_json(evidence_output / "quality.json", report)
                checkpoint("quality", evidence_output / "quality.json")
            if "mesh_views" not in artifacts:
                attempt["phase"] = "render"
                store.save(job)
                report = attempt["quality"]
                if report["faces"] > config.quality.max_faces or "bounds" not in report:
                    image = Image.new("RGB", (768, 300), "white")
                    ImageDraw.Draw(image).text(
                        (20, 40),
                        "RENDER SKIPPED: invalid geometry or face limit exceeded.\nSee quality.json. This candidate cannot be accepted.",
                        fill="black",
                    )
                    image.save(evidence_output / "mesh-views.png")
                else:
                    render_views(
                        store.resolve(job_id, artifacts["mesh"]),
                        evidence_output / "mesh-views.png",
                        up_axis="Z"
                        if config.mesh.backend == "codex"
                        and (
                            config.texture.provider in ("provided", "image_api")
                            or config.surface.enabled
                        )
                        else "Y",
                    )
                checkpoint("mesh_views", evidence_output / "mesh-views.png")
            if config.generation.texture_review and not attempt.get("texture_review"):
                from .generation import texture_ready_for_review

                texture_ready_for_review(job)
                return job
            attempt["phase"] = "review"
            attempt["evidence_digest"] = evidence_digest(job)
            job["status"] = "awaiting_review"
        except Exception as exc:
            attempt["error"] = _error_message(exc)
            job["status"] = "failed"
        finally:
            job.pop("worker_pid", None)
            store.save(job)
        return job


def evidence_digest(job: dict) -> str:
    evidence = {
        "references": job["references"],
        "description": job["description"],
        "config": job["config"],
        "artifacts": job["attempts"][-1]["artifacts"],
    }
    if "reference_selection" in job:
        evidence["reference_selection"] = job["reference_selection"]
    return hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()


def verify_evidence(store: Store, job: dict):
    if "reference_selection" in job:
        store.resolve(job["id"], job["reference_selection"])
    for item in [
        *job["references"],
        job["reference_sheet"],
        *job["attempts"][-1]["artifacts"].values(),
    ]:
        store.resolve(job["id"], item)
    if job["attempts"][-1]["evidence_digest"] != evidence_digest(job):
        raise ValueError("The evidence or configuration changed after rendering")


def submit_geometry_review(store: Store, job_id: str, review: Review) -> dict:
    with store.lock(job_id):
        job = store.load(job_id)
        if job["status"] != "awaiting_geometry_review":
            raise ValueError("Candidate is not awaiting a white-geometry review")
        verify_evidence(store, job)
        attempt = job["attempts"][-1]
        config = PipelineConfig.model_validate(job["config"])
        if review.evidence_digest != attempt["evidence_digest"]:
            raise ValueError("Geometry review refers to stale evidence")
        if review.decision == "accept":
            if (
                not attempt["geometry_quality"]["passed"]
                or review.score < config.quality.min_score
            ):
                raise ValueError(
                    "White geometry did not pass local quality and visual gates"
                )
            attempt["geometry_review"] = review.model_dump()
            token = uuid.uuid4().hex
            job.update(status="queued", queue_token=token)
        else:
            attempt["review"] = review.model_dump()
            job["status"] = (
                "rejected"
                if review.decision == "reject"
                else (
                    "exhausted"
                    if len(job["attempts"]) >= config.quality.max_attempts
                    else "ready"
                )
            )
            token = None
        store.save(job)
    return run_candidate(store, job_id, token) if token else job


def submit_review(store: Store, job_id: str, review: Review) -> dict:
    with store.lock(job_id):
        job = store.load(job_id)
        if job["status"] != "awaiting_review":
            raise ValueError("Candidate is not awaiting review")
        verify_evidence(store, job)
        attempt = job["attempts"][-1]
        if review.evidence_digest != attempt["evidence_digest"]:
            raise ValueError("Review refers to stale or different evidence")
        config = PipelineConfig.model_validate(job["config"])
        if review.decision == "accept":
            quality = json.loads(
                store.resolve(job_id, attempt["artifacts"]["quality"]).read_text()
            )
            if not quality["passed"] or review.score < config.quality.min_score:
                raise ValueError(
                    "Acceptance failed geometry/texture gates or minimum review score"
                )
            job.update(status="accepted", accepted_attempt=attempt["number"])
        elif review.decision == "reject":
            job["status"] = "rejected"
        else:
            job["status"] = (
                "ready"
                if len(job["attempts"]) < config.quality.max_attempts
                else "exhausted"
            )
        attempt["review"] = review.model_dump()
        attempt["reviewed_at"] = now()
        store.save(job)
        return job


def export_asset(store: Store, job_id: str) -> dict:
    with store.lock(job_id):
        job = store.load(job_id)
        if job["status"] != "accepted":
            raise ValueError("Only an accepted candidate may be exported")
        verify_evidence(store, job)
        if "export" in job:
            for artifact in job["export"]["files"]:
                store.resolve(job_id, artifact)
            return job["export"]
        target = store.directory(job_id) / f"export-{uuid.uuid4().hex[:8]}"
        temporary = target.with_name("." + target.name)
        temporary.mkdir()
        attempt = job["attempts"][-1]
        try:
            source = store.resolve(job_id, attempt["artifacts"]["mesh"])
            shutil.copyfile(source, temporary / "textured_mesh.glb")
            export_obj(source, temporary)
            for key, name in (
                ("uv_layout", "uv-layout.png"),
                ("surface_record", "surface.json"),
                ("texture_provenance", "texture-provenance.json"),
            ):
                if key in attempt["artifacts"]:
                    shutil.copyfile(
                        store.resolve(job_id, attempt["artifacts"][key]),
                        temporary / name,
                    )
            metadata = {
                "job_id": job_id,
                "attempt": attempt["number"],
                "description": job["description"],
                "evidence_digest": attempt["evidence_digest"],
                "config": job["config"],
                "quality": attempt["quality"],
                "review": attempt["review"],
                "geometry_review": attempt.get("geometry_review"),
                "reference_selection": job.get("reference_selection"),
                "multiview_review": attempt.get("multiview_review"),
                "view_selection": attempt.get("view_selection"),
                "view_manifest": attempt["artifacts"].get("view_manifest"),
                "texture_review": attempt.get("texture_review"),
                "generation_trials": {
                    key: attempt.get(key, [])
                    for key in ("multiview_trials", "texture_trials")
                },
                "surface_artifacts": {
                    k: v
                    for k, v in attempt["artifacts"].items()
                    if k
                    in (
                        "source_mesh",
                        "prepared_mesh",
                        "uv_layout",
                        "surface_blend",
                        "texture_atlas",
                    )
                },
                "units": "unknown",
                "scale_note": "HOI preprocessing must estimate metric scale",
            }
            atomic_json(temporary / "provenance.json", metadata)
            temporary.rename(target)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        job["export"] = {
            "directory": str(target),
            "files": [store.artifact(job_id, p) for p in sorted(target.iterdir())],
        }
        store.save(job)
        return job["export"]
