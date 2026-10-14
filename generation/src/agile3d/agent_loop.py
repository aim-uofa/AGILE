"""Configured visual review and a bounded loop over the common AGILE state machine."""

from filelock import FileLock, Timeout

from .agent_backend import get_agent_backend
from .models import PipelineConfig
from .pipeline import (
    export_asset,
    queue_candidate,
    submit_review,
    submit_geometry_review,
    verify_evidence,
)
from .store import Store, atomic_json, now


def review_with_agent(store: Store, job_id: str) -> dict:
    """Claim one review durably; recovery never issues an implicit second call.

    The separate review lock stays held across the external request, while the
    ordinary job lock remains available to status readers and review submission.
    Claim/response files are execution records, not mutable evidence artifacts.
    """
    if store.load(job_id)["status"] == "awaiting_view_selection":
        return select_views_with_codex(store, job_id)
    if store.load(job_id)["status"] in (
        "awaiting_multiview_review",
        "awaiting_texture_review",
    ):
        return review_stage_with_codex(store, job_id)
    review_lock = FileLock(store.directory(job_id) / ".codex-review.lock", timeout=0)
    try:
        review_lock.acquire()
    except Timeout:
        raise ValueError("An agent review is already running for this job") from None
    try:
        with store.lock(job_id):
            job = store.load(job_id)
            geometry_stage = job["status"] == "awaiting_geometry_review"
            if job["status"] not in ("awaiting_review", "awaiting_geometry_review"):
                raise ValueError("Candidate must be awaiting review")
            verify_evidence(store, job)
            config = PipelineConfig.model_validate(job["config"])
            agent_name = config.agent.backend
            attempt = job["attempts"][-1]
            attempt_number = attempt["number"]
            directory = store.directory(job_id) / f"attempt-{attempt_number:02d}"
            call_directory = directory / (
                f"{agent_name}-white-review" if geometry_stage else agent_name
            )
            claim_path = directory / (
                f"{agent_name}-white-review-call.json"
                if geometry_stage
                else f"{agent_name}-review-call.json"
            )
            # Legacy calls may predate durable claims. Their outcome must also
            # be inspected rather than silently issuing a second paid request.
            if claim_path.exists() or any(
                call_directory.glob(f"{agent_name}-review-*")
            ):
                raise ValueError(
                    "Agent review was already started; inspect its saved response/logs. "
                    "Submit a valid saved response with review --file; no new call was made"
                )
            references = [store.resolve(job_id, r) for r in job["references"]]
            views = store.resolve(
                job_id,
                attempt["artifacts"][
                    "untextured_views" if geometry_stage else "mesh_views"
                ],
            )
            if geometry_stage and "untextured_oblique_views" in attempt["artifacts"]:
                views = [
                    views,
                    store.resolve(
                        job_id, attempt["artifacts"]["untextured_oblique_views"]
                    ),
                ]
            backend = get_agent_backend(config.agent)
            claim = {
                "status": "started",
                "started_at": now(),
                "attempt": attempt_number,
                "evidence_digest": attempt["evidence_digest"],
            }
            atomic_json(claim_path, claim)  # Durable before the external effect.
        try:
            images_only = config.agent.review_context == "images_only"
            review, metadata = backend.review_candidate(
                references,
                views,
                "The foreground object in the attached reference photographs."
                if images_only
                else job["description"],
                attempt["evidence_digest"],
                call_directory,
                minimum_score=config.quality.min_score,
                quality=None
                if images_only
                else attempt["geometry_quality" if geometry_stage else "quality"],
                **({"geometry_only": True} if geometry_stage else {}),
            )
            metadata["outbound_context_policy"] = {
                "review_context": config.agent.review_context,
                "QA_report_sent": not images_only,
                "private_target_description_sent": not images_only,
                "local_qa_gate_retained": True,
            }
            response_path = directory / (
                f"{agent_name}-white-review-response.json"
                if geometry_stage
                else f"{agent_name}-review-response.json"
            )
            atomic_json(response_path, review.model_dump())
            claim.update(
                status="response_received",
                response_file=response_path.name,
                response_received_at=now(),
                metadata=metadata,
            )
            atomic_json(claim_path, claim)
            job = (submit_geometry_review if geometry_stage else submit_review)(
                store, job_id, review
            )
            with store.lock(job_id):
                job = store.load(job_id)
                next(a for a in job["attempts"] if a["number"] == attempt_number)[
                    f"{agent_name}_geometry_review"
                    if geometry_stage
                    else f"{agent_name}_review"
                ] = metadata
                store.save(job)
            claim.update(status="completed", finished_at=now())
            atomic_json(claim_path, claim)
            return job
        except BaseException as exc:
            claim.update(
                status="submission_failed"
                if claim["status"] == "response_received"
                else "outcome_unknown",
                error_type=type(exc).__name__,
                finished_at=now(),
            )
            atomic_json(claim_path, claim)
            raise
    finally:
        review_lock.release()


def run_agent_loop(store: Store, job_id: str) -> dict:
    """Run automatic providers within frozen candidate and declared request budgets."""
    job = store.load(job_id)
    from .final_output import TERMINAL, finalize_asset

    config = PipelineConfig.model_validate(job["config"])
    if (
        (config.image.provider == "provided" or config.mesh.backend == "provided")
        and not (config.generation.multiview_review or config.generation.texture_review)
        and job["status"] not in TERMINAL
    ):
        raise ValueError(
            "Imported candidates use candidate + agent-review; supply each new asset explicitly"
        )
    review_states = (
        "awaiting_view_selection",
        "awaiting_review",
        "awaiting_geometry_review",
        "awaiting_multiview_review",
        "awaiting_texture_review",
    )
    while job["status"] in ("ready", "awaiting_texture", *review_states):
        if job["status"] == "awaiting_texture":
            if config.texture.provider != "image_api":
                break
            job = generate_texture_with_api(store, job_id)
        if job["status"] == "ready":
            job = queue_candidate(store, job_id, background=False)
        if job["status"] in review_states:
            job = review_with_agent(store, job_id)
    if job["status"] == "accepted":
        export_asset(store, job_id)
        job = store.load(job_id)
    if job["status"] in TERMINAL:
        finalize_asset(store, job_id)
        job = store.load(job_id)
        if (
            config.texture.provider == "image_api"
            and config.generation.final_texture_completion
            and job.get("delivery", {}).get("status") == "awaiting_generated_texture"
        ):
            generate_final_texture_with_api(store, job_id)
            job = store.load(job_id)
    return job


def select_views_with_codex(store, job_id):
    """One durable selection call; manual submit recovers a saved valid response."""
    import json
    from .view_selection import submit_view_selection

    with FileLock(store.directory(job_id) / ".codex-review.lock", timeout=0):
        with store.lock(job_id):
            job = store.load(job_id)
            if job["status"] != "awaiting_view_selection":
                raise ValueError("Candidate must be awaiting_view_selection")
            verify_evidence(store, job)
            a = job["attempts"][-1]
            directory = (
                store.directory(job_id)
                / f"attempt-{a['number']:02d}"
                / "view-selection"
            )
            config = PipelineConfig.model_validate(job["config"])
            agent_name = config.agent.backend
            claim_path = directory / f"{agent_name}-call.json"
            if claim_path.exists():
                raise ValueError(
                    "View selection already started; inspect saved response and use select-views to recover"
                )
            request = json.loads(
                store.resolve(
                    job_id, a["artifacts"]["view_selection_request"]
                ).read_text()
            )
            references = [store.resolve(job_id, r) for r in job["references"]]
            sheets = [store.resolve(job_id, r) for r in request["sheets"]]
            candidates = [
                {k: c[k] for k in ("id", "trial", "cell")}
                for c in request["candidates"]
            ]
            backend = get_agent_backend(config.agent)
            claim = {
                "status": "started",
                "started_at": now(),
                "evidence_digest": a["evidence_digest"],
            }
            atomic_json(claim_path, claim)
        try:
            review, metadata = backend.select_view_subset(
                references,
                sheets,
                candidates,
                a["evidence_digest"],
                directory / agent_name,
                request["max_views"],
            )
            atomic_json(directory / f"{agent_name}-response.json", review.model_dump())
            claim.update(status="response_received", metadata=metadata)
            atomic_json(claim_path, claim)
            job = submit_view_selection(store, job_id, review)
            claim.update(status="completed", finished_at=now())
            atomic_json(claim_path, claim)
            return job
        except BaseException as exc:
            claim.update(
                status="submission_failed"
                if claim["status"] == "response_received"
                else "outcome_unknown",
                error_type=type(exc).__name__,
            )
            atomic_json(claim_path, claim)
            raise


def review_stage_with_codex(store, job_id):
    """One durable claim per independent image/texture trial; never replay a call."""
    from .generation import submit_stage_review

    with FileLock(store.directory(job_id) / ".codex-review.lock", timeout=0):
        with store.lock(job_id):
            job = store.load(job_id)
            if job["status"] not in (
                "awaiting_multiview_review",
                "awaiting_texture_review",
            ):
                raise ValueError("Candidate is not awaiting a stage review")
            verify_evidence(store, job)
            stage = job["status"].removeprefix("awaiting_").removesuffix("_review")
            a = job["attempts"][-1]
            trial = a[f"{stage}_trials"][-1]
            directory = (
                store.directory(job_id)
                / f"attempt-{a['number']:02d}"
                / f"{stage}-{trial['number']:02d}"
            )
            config = PipelineConfig.model_validate(job["config"])
            agent_name = config.agent.backend
            claim_path = directory / f"{agent_name}-call.json"
            if claim_path.exists():
                raise ValueError(
                    "Stage review already started; inspect saved response and use stage-review to recover"
                )
            backend = get_agent_backend(config.agent)
            refs = [store.resolve(job_id, r) for r in job["references"]]
            claim = {
                "status": "started",
                "stage": stage,
                "trial": trial["number"],
                "evidence_digest": a["evidence_digest"],
                "started_at": now(),
            }
            atomic_json(claim_path, claim)
        try:
            if stage == "multiview":
                review, metadata = backend.review_multiview(
                    refs,
                    store.resolve(job_id, trial["image"]),
                    a["evidence_digest"],
                    directory / agent_name,
                    config.generation.min_multiview_score,
                )
            else:
                art = a["artifacts"]
                view = (
                    store.resolve(job_id, art["multiview"])
                    if "multiview" in art
                    else None
                )
                review, metadata = backend.review_texture(
                    refs,
                    view,
                    store.resolve(job_id, art["texture_atlas"]),
                    store.resolve(job_id, art["uv_layout"]),
                    store.resolve(job_id, art["mesh_views"]),
                    a["evidence_digest"],
                    directory / agent_name,
                    config.generation.min_texture_score,
                )
            atomic_json(directory / f"{agent_name}-response.json", review.model_dump())
            claim.update(status="response_received", metadata=metadata)
            atomic_json(claim_path, claim)
            job = submit_stage_review(store, job_id, review)
            claim.update(status="completed", finished_at=now())
            atomic_json(claim_path, claim)
            return job
        except BaseException as exc:
            claim.update(
                status="submission_failed"
                if claim["status"] == "response_received"
                else "outcome_unknown",
                error_type=type(exc).__name__,
            )
            atomic_json(claim_path, claim)
            raise


def generate_texture_with_api(store, job_id):
    """One explicit image-generation call per frozen texture request."""
    import json
    from .providers import generate_image
    from .pipeline import provide_texture
    from .store import digest

    with FileLock(store.directory(job_id) / ".codex-review.lock", timeout=0):
        with store.lock(job_id):
            job = store.load(job_id)
            config = PipelineConfig.model_validate(job["config"])
            if (
                job["status"] != "awaiting_texture"
                or config.texture.provider != "image_api"
            ):
                raise ValueError("Job is not awaiting an image API texture")
            verify_evidence(store, job)
            art = job["attempts"][-1]["artifacts"]
            request_path = store.resolve(job_id, art["texture_request"])
            request = json.loads(request_path.read_text())
            directory = request_path.parent / "image-api"
            directory.mkdir(exist_ok=True)
            claim_path = directory / "claim.json"
            if claim_path.exists():
                raise ValueError(
                    "Texture API call already started; inspect saved output and use apply-texture to recover"
                )
            attachments = request["attachments"]
            images = [store.resolve(job_id, item["artifact"]) for item in attachments]
            prompt = (
                request["prompt"]
                + "\nAttachments in order: "
                + json.dumps([item["role"] for item in attachments])
            )
            claim = {
                "status": "started",
                "started_at": now(),
                "request_sha256": digest(request_path),
            }
            atomic_json(claim_path, claim)
        try:
            output = directory / "generated.png"
            metadata = generate_image(config.image, images, prompt, output)
            metadata.update(
                image_sha256=digest(output),
                prepared_mesh_sha256=art["prepared_mesh"]["sha256"],
                uv_layout_sha256=art["uv_layout"]["sha256"],
            )
            provenance = directory / "provenance.json"
            atomic_json(provenance, metadata)
            claim.update(status="response_received", metadata=metadata)
            atomic_json(claim_path, claim)
            job = provide_texture(store, job_id, output, provenance)
            claim.update(status="completed", finished_at=now())
            atomic_json(claim_path, claim)
            return job
        except BaseException as exc:
            claim.update(
                status="submission_failed"
                if claim["status"] == "response_received"
                else "outcome_unknown",
                error_type=type(exc).__name__,
                finished_at=now(),
            )
            atomic_json(claim_path, claim)
            raise


# Existing clients retain their entry points; both aliases honor agent.backend.
review_with_codex = review_with_agent
run_codex_loop = run_agent_loop


def generate_final_texture_with_api(store, job_id):
    """Optional single completion call; it never changes a failed quality gate."""
    import json
    from .providers import generate_image
    from .final_output import provide_final_texture
    from .store import digest

    with FileLock(store.directory(job_id) / ".final-texture-api.lock", timeout=0):
        with store.lock(job_id):
            job = store.load(job_id)
            config = PipelineConfig.model_validate(job["config"])
            pending = job.get("delivery", {})
            if (
                config.texture.provider != "image_api"
                or not config.generation.final_texture_completion
                or pending.get("status") != "awaiting_generated_texture"
            ):
                raise ValueError(
                    "Final image API completion is not enabled or requested"
                )
            path = store.resolve(job_id, pending["request"])
            request = json.loads(path.read_text())
            directory = path.parent / "image-api-completion"
            directory.mkdir(exist_ok=True)
            claim_path = directory / "claim.json"
            if claim_path.exists():
                raise ValueError(
                    "Final texture API call already started; inspect saved output and use apply-final-texture to recover"
                )
            images = [store.resolve(job_id, ref) for ref in job["references"]]
            images.append(store.resolve(job_id, pending["uv_layout"]))
            prompt = (
                request["prompt"]
                + "\nAttachments: original reference images followed by the exact frozen UV guide."
            )
            claim = {
                "status": "started",
                "started_at": now(),
                "request_sha256": digest(path),
            }
            atomic_json(claim_path, claim)
        try:
            output = directory / "generated.png"
            metadata = generate_image(config.image, images, prompt, output)
            metadata.update(
                image_sha256=digest(output),
                prepared_mesh_sha256=request["prepared_mesh_sha256"],
                uv_layout_sha256=request["uv_layout_sha256"],
            )
            provenance = directory / "provenance.json"
            atomic_json(provenance, metadata)
            claim.update(status="response_received", metadata=metadata)
            atomic_json(claim_path, claim)
            result = provide_final_texture(store, job_id, output, provenance)
            claim.update(status="completed", finished_at=now())
            atomic_json(claim_path, claim)
            return result
        except BaseException as exc:
            claim.update(
                status="submission_failed"
                if claim["status"] == "response_received"
                else "outcome_unknown",
                error_type=type(exc).__name__,
                finished_at=now(),
            )
            atomic_json(claim_path, claim)
            raise
