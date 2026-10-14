"""Paper-stage rejection loops. All trials, critiques and edit inputs remain frozen."""

import json
import shutil
import uuid

from PIL import Image

from .models import PipelineConfig
from .stage_reviews import MultiviewReview, TextureReview
from .store import atomic_json, digest, now


def record(store, job, key, path):
    artifact = store.artifact(job["id"], path)
    job["attempts"][-1]["artifacts"][key] = artifact
    return artifact


def waiting(job, status):
    from .pipeline import evidence_digest

    job["status"] = status
    job["attempts"][-1]["phase"] = status.removeprefix("awaiting_")
    job["attempts"][-1]["evidence_digest"] = evidence_digest(job)


def advance(job):
    token = uuid.uuid4().hex
    job.update(status="queued", queue_token=token)
    return token


def image_valid(path):
    with Image.open(path) as im:
        if (
            im.format != "PNG"
            or im.width != im.height
            or im.width < 64
            or im.width % 2
            or im.width > 4096
        ):
            raise ValueError(
                "Four-view input must be a square even-sized PNG, 64..4096 pixels"
            )
        im.verify()


def ensure_multiview(store, job, config, directory, retry_image=False):
    """Called inside the job lock, before any mesh provider can execute."""
    from . import providers

    a = job["attempts"][-1]
    if a.get("view_selection"):
        if not {"view_manifest", "multiview"} <= a["artifacts"].keys():
            raise ValueError("Selected-view conditioning is incomplete")
        return True
    if a.get("multiview_review", {}).get("decision") == "accept":
        if "multiview" not in a["artifacts"]:
            raise ValueError("Accepted multi-view artifact is missing")
        return True
    trials = a.setdefault("multiview_trials", [])
    if not trials or trials[-1].get("review"):
        number = len(trials) + 1
        if number > config.generation.max_multiview_attempts:
            raise ValueError("Multi-view rejection budget exhausted")
        folder = directory / f"multiview-{number:02d}"
        folder.mkdir(exist_ok=True)
        previous_attempt = job["attempts"][-2] if len(job["attempts"]) > 1 else None
        feedback = (
            trials[-1]["review"]["feedback"]
            if trials
            else ((previous_attempt or {}).get("review") or {}).get("feedback", "")
        )
        previous_draft = (
            trials[-1].get("image")
            if trials
            else (previous_attempt or {}).get("artifacts", {}).get("multiview")
        )
        request = {
            "stage": "multiview",
            "trial": number,
            "reference_hashes": [r["sha256"] for r in job["references"]],
            "prompt": providers.MULTIVIEW_PROMPT
            + "\nTarget: "
            + job["description"]
            + "\nCritique of the preceding draft: "
            + feedback,
            "feedback": feedback,
            "layout": ["front", "back", "left", "right"],
            "scope": "Try a complete four-view image first; after the trial budget, select 2–3 coherent individual views if fallback is enabled.",
            "previous_draft": previous_draft,
            "attachments": (
                [{"role": "previous_draft", "artifact": previous_draft}]
                if previous_draft
                else []
            )
            + [
                {"role": "original_reference_sheet", "artifact": job["reference_sheet"]}
            ],
        }
        atomic_json(folder / "request.json", request)
        artifact = record(
            store, job, f"multiview_request_{number:02d}", folder / "request.json"
        )
        a["artifacts"]["multiview_request"] = artifact
        trials.append(
            {
                "number": number,
                "request": artifact,
                "created_at": now(),
                "call_started": False,
            }
        )
        store.save(job)
    trial = trials[-1]
    number = trial["number"]
    folder = directory / f"multiview-{number:02d}"
    output = folder / "image.png"
    if "image" not in trial:
        if number == 1 and "input_image" in a:
            shutil.copyfile(store.resolve(job["id"], a["input_image"]), output)
            trial["source"] = {"provider": "provided_at_candidate_start"}
        elif config.image.provider == "provided":
            waiting(job, "awaiting_multiview")
            return False
        else:
            if trial["call_started"] and not retry_image:
                raise ValueError(
                    "Multi-view image call outcome is unknown; inspect the record before an explicit retry"
                )
            trial["call_started"] = True
            store.save(job)
            request = json.loads(store.resolve(job["id"], trial["request"]).read_text())
            previous = (
                store.resolve(job["id"], request["previous_draft"])
                if request.get("previous_draft")
                else None
            )
            trial["source"] = providers.generate_image(
                config.image,
                [store.resolve(job["id"], r) for r in job["references"]],
                json.loads(store.resolve(job["id"], trial["request"]).read_text())[
                    "prompt"
                ],
                output,
                previous,
            )
        image_valid(output)
        trial["image"] = record(store, job, f"multiview_image_{number:02d}", output)
    waiting(job, "awaiting_multiview_review")
    return False


def provide_multiview(store, job_id, image, provenance=None):
    from .pipeline import verify_evidence

    image_valid(image)
    with store.lock(job_id):
        job = store.load(job_id)
        if job["status"] != "awaiting_multiview":
            raise ValueError("Candidate must be awaiting_multiview")
        verify_evidence(store, job)
        a = job["attempts"][-1]
        trial = a["multiview_trials"][-1]
        if "image" in trial:
            raise ValueError("This image trial is already frozen")
        folder = (
            store.directory(job_id)
            / f"attempt-{a['number']:02d}"
            / f"multiview-{trial['number']:02d}"
        )
        metadata = {"provider": "provided", "image_sha256": digest(image)}
        if provenance:
            if provenance.stat().st_size > 1024**2:
                raise ValueError("Provenance exceeds 1 MiB")
            metadata = json.loads(provenance.read_text())
            if metadata.get("image_sha256") != digest(image):
                raise ValueError("Multi-view provenance hash mismatch")
        shutil.copyfile(image, folder / "image.png")
        atomic_json(folder / "provenance.json", metadata)
        trial["image"] = record(
            store, job, f"multiview_image_{trial['number']:02d}", folder / "image.png"
        )
        trial["source"] = record(
            store,
            job,
            f"multiview_source_{trial['number']:02d}",
            folder / "provenance.json",
        )
        waiting(job, "awaiting_multiview_review")
        store.save(job)
        return job


def texture_directory(directory, attempt):
    return directory / f"texture-{attempt['texture_trials'][-1]['number']:02d}"


def mesh_request(store, job, directory):
    a = job["attempts"][-1]
    approved = a["artifacts"]["multiview"]
    request = {
        "stage": "mesh",
        "multiview": approved,
        "multiview_sha256": approved["sha256"],
        "instruction": "Lift this approved four-view image to a textured mesh. Return self-contained GLB and provenance with mesh_sha256 and multiview_sha256.",
    }
    if "view_manifest" in a["artifacts"]:
        request.update(
            mode="named_view_subset",
            view_manifest=a["artifacts"]["view_manifest"],
            view_manifest_sha256=a["artifacts"]["view_manifest"]["sha256"],
            instruction="Load ONLY the 2–3 named PNGs from the portable view manifest into the Hunyuan multi-view dictionary. Do not quarter-crop the preview, add blank views or duplicate views. Return GLB and provenance with mesh_sha256 and view_manifest_sha256.",
        )
    atomic_json(directory / "mesh-request.json", request)
    record(store, job, "mesh_request", directory / "mesh-request.json")
    waiting(job, "awaiting_mesh")


def provide_mesh(store, job_id, mesh, provenance):
    """Import lifting bound to an accepted grid or explicitly selected named views."""
    from .pipeline import run_candidate, verify_evidence

    if mesh.suffix.lower() != ".glb" or mesh.stat().st_size > 512 * 1024**2:
        raise ValueError("Supply a self-contained GLB up to 512 MiB")
    if provenance.stat().st_size > 1024**2:
        raise ValueError("Mesh provenance exceeds 1 MiB")
    metadata = json.loads(provenance.read_text())
    with store.lock(job_id):
        job = store.load(job_id)
        if job["status"] != "awaiting_mesh":
            raise ValueError("Candidate must be awaiting_mesh")
        verify_evidence(store, job)
        a = job["attempts"][-1]
        binding = "view_manifest" if "view_manifest" in a["artifacts"] else "multiview"
        if (
            not isinstance(metadata, dict)
            or metadata.get("mesh_sha256") != digest(mesh)
            or metadata.get(f"{binding}_sha256") != a["artifacts"][binding]["sha256"]
        ):
            raise ValueError(
                "Mesh provenance must match the GLB and its approved image or selected named-view manifest"
            )
        directory = store.directory(job_id) / f"attempt-{a['number']:02d}"
        shutil.copyfile(mesh, directory / "imported-mesh.glb")
        atomic_json(directory / "mesh-provenance.json", metadata)
        a["input_mesh"] = record(
            store, job, "imported_mesh", directory / "imported-mesh.glb"
        )
        record(store, job, "mesh_provenance", directory / "mesh-provenance.json")
        token = advance(job)
        store.save(job)
    return run_candidate(store, job_id, token)


def texture_request(store, job, config, directory, trial):
    from .surface import mesh_uv_request

    a = job["attempts"][-1]
    art = a["artifacts"]
    trials = a["texture_trials"]
    previous = trials[-2] if len(trials) > 1 else None
    seed = (previous or {}).get("artifacts", {}).get("texture_atlas")
    critique = (previous or {}).get("review", {})
    surface = json.loads(store.resolve(job["id"], art["surface_record"]).read_text())
    request = mesh_uv_request(
        job["description"],
        surface,
        store.resolve(job["id"], art["prepared_mesh"]),
        store.resolve(job["id"], art["uv_layout"]),
        baked=store.resolve(job["id"], seed) if seed else None,
        feedback="\nFrozen critique to address: "
        + json.dumps(critique, ensure_ascii=False),
    )
    request.update(
        stage="texture",
        trial=trial["number"],
        critique=critique,
        reference_hashes=[r["sha256"] for r in job["references"]],
        conditioning="selected_view_subset_and_originals"
        if "view_manifest" in art
        else "validated_multiview_and_originals"
        if "multiview" in art
        else "originals_pure_codex_variant",
    )
    attachments = []
    if seed:
        attachments.append({"role": "edit_target", "artifact": seed})
    if "multiview" in art:
        attachments.append(
            {
                "role": "selected_view_subset"
                if "view_manifest" in art
                else "validated_multiview",
                "artifact": art["multiview"],
            }
        )
        request["prompt"] += (
            " The selected appearance views are conditioning inputs (a full grid or a 2–3 view fallback). A fallback does not certify a complete four-view grid. Original observations remain the authority if a generated view contradicts them."
        )
    attachments += [
        {"role": "frozen_uv_guide", "artifact": art["uv_layout"]},
        {"role": "original_reference_sheet", "artifact": job["reference_sheet"]},
    ]
    request["attachments"] = attachments
    folder = texture_directory(directory, a)
    folder.mkdir(exist_ok=True)
    atomic_json(folder / "request.json", request)
    trial["request"] = record(
        store, job, f"texture_request_{trial['number']:02d}", folder / "request.json"
    )
    art["texture_request"] = trial["request"]
    waiting(job, "awaiting_texture")


def ensure_texture(store, job, config, directory):
    """Critique the baked seed first; generated edits reuse exactly one mesh/UV."""
    a = job["attempts"][-1]
    art = a["artifacts"]
    trials = a.setdefault("texture_trials", [])
    if a.get("texture_review", {}).get("decision") == "accept":
        return True
    if not trials:
        if config.mesh.backend != "codex":
            # Trial zero is the original ray-cast bake, not a generated edit.
            trial = {"number": 0, "source": "baked_seed", "created_at": now()}
            trials.append(trial)
            folder = texture_directory(directory, a)
            folder.mkdir(exist_ok=True)
            art["texture_atlas"] = art["baked_albedo"]
            metadata = {
                "provider": "blender_bake",
                "image_sha256": art["baked_albedo"]["sha256"],
                "prepared_mesh_sha256": art["prepared_mesh"]["sha256"],
                "uv_layout_sha256": art["uv_layout"]["sha256"],
            }
            atomic_json(folder / "provenance.json", metadata)
            art["texture_provenance"] = record(
                store, job, "texture_source_00", folder / "provenance.json"
            )
        else:
            trials.append({"number": 1, "source": "generated", "created_at": now()})
            texture_request(store, job, config, directory, trials[-1])
            return False
    elif trials[-1].get("review"):
        number = trials[-1]["number"] + 1
        if number > config.generation.max_texture_attempts:
            raise ValueError("Texture rejection budget exhausted")
        trials.append({"number": number, "source": "generated", "created_at": now()})
        texture_request(store, job, config, directory, trials[-1])
        return False
    if "texture_atlas" not in art:
        waiting(job, "awaiting_texture")
        return False
    return True


TEXTURE_KEYS = (
    "texture_atlas",
    "texture_provenance",
    "texture_request",
    "mesh",
    "quality",
    "mesh_views",
)


def texture_ready_for_review(job):
    a = job["attempts"][-1]
    trial = a["texture_trials"][-1]
    trial["artifacts"] = {
        k: a["artifacts"][k] for k in TEXTURE_KEYS if k in a["artifacts"]
    }
    for k, artifact in trial["artifacts"].items():
        a["artifacts"][f"texture_{trial['number']:02d}_{k}"] = artifact
    waiting(job, "awaiting_texture_review")


def submit_stage_review(store, job_id, review):
    from .pipeline import verify_evidence, run_candidate

    stage = "multiview" if isinstance(review, MultiviewReview) else "texture"
    if not isinstance(review, (MultiviewReview, TextureReview)):
        raise TypeError("Use a stage-specific review schema")
    token = None
    with store.lock(job_id):
        job = store.load(job_id)
        a = job["attempts"][-1]
        config = PipelineConfig.model_validate(job["config"])
        if job["status"] != f"awaiting_{stage}_review":
            raise ValueError("Wrong review stage")
        verify_evidence(store, job)
        if review.evidence_digest != a["evidence_digest"]:
            raise ValueError("Review refers to stale stage evidence")
        threshold = getattr(config.generation, f"min_{stage}_score")
        if review.decision == "accept" and not review.eligible(threshold):
            raise ValueError(
                "Acceptance contradicts stage-specific vetoes, view layout or score threshold"
            )
        if (
            review.decision == "accept"
            and stage == "texture"
            and not a["quality"]["passed"]
        ):
            raise ValueError(
                "Local mesh quality failed; texture score cannot override it"
            )
        trial = a[f"{stage}_trials"][-1]
        payload = {
            **review.model_dump(),
            "score_overall": review.score(),
            "threshold": threshold,
        }
        directory = (
            store.directory(job_id)
            / f"attempt-{a['number']:02d}"
            / f"{stage}-{trial['number']:02d}"
        )
        atomic_json(directory / "review.json", payload)
        record(
            store,
            job,
            f"{stage}_review_{trial['number']:02d}",
            directory / "review.json",
        )
        trial["review"] = payload
        if review.decision == "accept":
            a[f"{stage}_review"] = payload
            if stage == "multiview":
                a["artifacts"]["multiview"] = trial["image"]
            token = advance(job)
        elif (
            stage == "multiview"
            and config.generation.multiview_fallback == "best_subset"
        ):
            if trial["number"] >= config.generation.max_multiview_attempts:
                from .view_selection import selection_request

                selection_request(store, job, directory.parent)
            else:
                token = advance(job)
        elif review.decision == "reject":
            job["status"] = "rejected"
        elif trial["number"] >= getattr(config.generation, f"max_{stage}_attempts"):
            job.update(status="exhausted", exhausted_stage=stage)
        else:
            if stage == "texture":
                for key in TEXTURE_KEYS:
                    a["artifacts"].pop(key, None)
                a.pop("quality", None)
            token = advance(job)
        # Terminal records must remain independently verifiable too.
        from .pipeline import evidence_digest

        a["evidence_digest"] = evidence_digest(job)
        store.save(job)
    return run_candidate(store, job_id, token) if token else job


def create_from_frames(store, frames, description, config, max_selected=4):
    """Freeze selection evidence before the VLM call, then freeze selected job inputs."""
    from .agent_backend import get_agent_backend
    from .pipeline import copy_image, create_job

    if not 1 <= len(frames) <= 16 or not 1 <= max_selected <= min(4, len(frames)):
        raise ValueError("Supply 1..16 candidate frames and select 1..4 references")
    directory = store.root / "selections" / uuid.uuid4().hex[:16]
    directory.mkdir(parents=True)
    frozen = []
    for i, frame in enumerate(frames):
        p = directory / f"{i:03d}.png"
        copy_image(frame, p)
        frozen.append(p)
    inputs = [
        {"index": i, "path": str(p), "sha256": digest(p)} for i, p in enumerate(frozen)
    ]
    atomic_json(
        directory / "claim.json",
        {
            "status": "started",
            "inputs": inputs,
            "description": description,
            "max_selected": max_selected,
        },
    )
    backend = get_agent_backend(config.agent)
    try:
        result, metadata = backend.select_references(
            frozen, description, max_selected, directory / config.agent.backend
        )
    except BaseException as exc:
        atomic_json(
            directory / "claim.json", {"status": "outcome_unknown", "inputs": inputs}
        )
        # Preserve the original exception type for existing callers, while the
        # user CLI can locate evidence even before a job ID has been allocated.
        exc.agile_selection_directory = str(directory)
        raise
    selected = [frozen[i] for i in result.selected_indices]
    payload = {
        "selection": result.model_dump(),
        "candidate_frames": inputs,
        "metadata": metadata,
        "selected_sha256": [digest(p) for p in selected],
    }
    atomic_json(directory / "selection.json", payload)
    job = create_job(store, selected, description, config)
    dest = store.directory(job["id"]) / "reference-selection.json"
    atomic_json(dest, payload)
    job["reference_selection"] = store.artifact(job["id"], dest)
    store.save(job)
    atomic_json(
        directory / "claim.json",
        {"status": "completed", "job_id": job["id"], "selection": str(dest)},
    )
    return job
