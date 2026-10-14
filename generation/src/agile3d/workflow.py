"""Small user entry points around the existing reviewed generation state machine."""

from filelock import FileLock, Timeout

from .agent_loop import review_with_agent, run_agent_loop
from .doctor import PreflightError, require_ready
from .final_output import TERMINAL
from .generation import create_from_frames
from .models import PipelineConfig
from .pipeline import _error_message, queue_candidate


class WorkflowError(ValueError):
    """Keep the saved job discoverable when an external stage fails."""

    def __init__(self, store, job_id, cause):
        self.report = {
            "status": "needs_attention",
            "directory": str(store.directory(job_id)) if job_id else str(
                getattr(cause, "agile_selection_directory", store.root / "selections")
            ),
            "error": _error_message(cause),
            "next": "Inspect the saved job and provider records before recovery. No automatic retry or claim reset was performed.",
        }
        if job_id:
            self.report["job_id"] = job_id
            heading = f"Job: {job_id}"
        else:
            self.report.update(
                stage="reference_selection",
                next="Inspect the saved selection claim and provider records before starting another selection. No automatic retry or claim reset was performed.",
            )
            if hasattr(cause, "agile_selection_directory"):
                self.report["job_created"] = False
            heading = "Reference selection did not return a job ID."
        super().__init__(
            f"{heading}\nWorkspace: {self.report['directory']}\n"
            f"Error: {self.report['error']}\nNext: {self.report['next']}"
        )


def _manual_candidate(config):
    return (config.image.provider == "provided" or config.mesh.backend == "provided") and not (
        config.generation.multiview_review or config.generation.texture_review
    )


def generate(store, frames, description, config, max_selected=4):
    """Check the complete route before the first external selection request."""
    if not 1 <= len(frames) <= 16 or not 1 <= max_selected <= 4:
        raise ValueError("Supply 1..16 frames and set max-selected to 1..4.")
    if not 1 <= len(description.strip()) <= 4000:
        raise ValueError("Describe the target object in 1 to 4000 characters.")
    require_ready(config, frames=frames)
    config = PipelineConfig.model_validate(config)
    try:
        job = create_from_frames(store, frames, description, config, min(max_selected, len(frames)))
    except Exception as exc:
        raise WorkflowError(store, None, exc) from exc
    try:
        return _continue(store, job["id"], check=False)
    except Exception as exc:
        raise WorkflowError(store, job["id"], exc) from exc


def _unknown_candidate_call(attempt):
    artifacts = attempt.get("artifacts", {})
    if attempt.get("image_call_started") and "multiview" not in artifacts:
        return "image generation"
    if attempt.get("codex_call_started") and "geometry_spec" not in artifacts:
        return "Codex geometry"
    trials = attempt.get("multiview_trials", [])
    if trials and trials[-1].get("call_started") and "image" not in trials[-1]:
        return "multi-view generation"
    return None


def resume(store, job_id):
    """Continue saved work without overriding locks or replaying unknown requests."""
    # Validate the ID and saved job before adding recovery context.
    store.load(job_id)
    try:
        return _continue(store, job_id, check=True)
    except PreflightError:
        raise
    except Exception as exc:
        raise WorkflowError(store, job_id, exc) from exc


def _continue(store, job_id, *, check):
    # This lock also prevents two simple CLI invocations from issuing reviews.
    # Existing worker/review/provider locks remain authoritative for advanced tools.
    try:
        with FileLock(store.directory(job_id) / ".workflow.lock", timeout=0):
            job = store.load(job_id)
            if job["status"] in ("queued", "running"):
                raise ValueError(
                    "A worker is queued or running. Inspect status and worker.log; "
                    "resume will not replace it. If it was interrupted, verify that "
                    "no worker remains before using the advanced candidate --resume command."
                )
            if job.get("delivery", {}).get("status") == "delivered":
                for artifact in job["delivery"]["files"]:
                    store.resolve(job_id, artifact)
                return job
            config = PipelineConfig.model_validate(job["config"])
            if job["status"] == "failed" and job.get("attempts"):
                call = _unknown_candidate_call(job["attempts"][-1])
                if call:
                    raise ValueError(
                        f"The previous {call} request has an unknown outcome. "
                        "Inspect the saved response/logs before an explicit recovery; no new call was made."
                    )
            # A waiting imported stage does not issue calls. Let users inspect or
            # supply the requested file without needing unrelated API credentials.
            if job["status"] in ("awaiting_multiview", "awaiting_mesh") or (
                job["status"] == "awaiting_texture" and config.texture.provider != "image_api"
            ) or (job["status"] == "ready" and _manual_candidate(config)):
                return job
            if check:
                require_ready(config)
            if job["status"] == "failed" and job.get("attempts"):
                # Keep the original attempt, frozen assets and bounded budget.
                # No retry_image flag: all durable unknown-outcome checks apply.
                queue_candidate(store, job_id, resume=True, background=False)
            if _manual_candidate(config):
                job = store.load(job_id)
                if job["status"] in ("awaiting_review", "awaiting_geometry_review"):
                    job = review_with_agent(store, job_id)
                if job["status"] not in TERMINAL:
                    # A rejected imported asset needs new explicit input. Never
                    # reuse the old supplied mesh as a newly generated candidate.
                    return job
            return run_agent_loop(store, job_id)
    except Timeout:
        raise ValueError("This job already has an active workflow, worker or review; no duplicate was started.") from None


def next_action(job):
    """Describe the existing state instead of exposing internal scheduling steps."""
    status, job_id = job["status"], job["id"]
    delivery = job.get("delivery", {})
    if delivery.get("status") == "delivered":
        return "Review the exported asset and provenance before downstream use."
    if delivery.get("status") == "awaiting_generated_texture":
        return f"Supply the saved UV texture request with apply-final-texture {job_id}, then run resume {job_id}."
    actions = {
        "awaiting_multiview": f"Provide the requested image with apply-multiview {job_id} --image FILE, then run resume {job_id}.",
        "awaiting_mesh": f"Provide a GLB with apply-mesh {job_id} --mesh FILE --provenance FILE, then run resume {job_id}.",
        "awaiting_texture": f"Provide the saved UV texture request with apply-texture {job_id} --image FILE --provenance FILE, then run resume {job_id}.",
        "running": "The worker is active. Inspect status later; do not start a duplicate.",
        "queued": "A worker is queued. Inspect status later; do not start a duplicate.",
        "failed": f"Inspect the saved error and provider records, then run resume {job_id} after resolving the cause.",
    }
    if status == "ready" and _manual_candidate(PipelineConfig.model_validate(job["config"])):
        return f"Supply the offline inputs with candidate {job_id} --image FILE --mesh FILE, then use agent-review {job_id}."
    return actions.get(status, f"Run resume {job_id} to continue; saved review requests can also be submitted with the advanced commands.")


def format_status(job):
    lines = [f"Job: {job['id']}", f"State: {job['status']}"]
    if job.get("attempts"):
        attempt = job["attempts"][-1]
        lines.append(f"Attempt: {attempt['number']}" + (f" ({attempt['phase']})" if attempt.get("phase") else ""))
        if attempt.get("error"):
            lines.append("Last error: " + attempt["error"])
    delivery = job.get("delivery", {})
    if delivery.get("status") == "delivered":
        lines.extend(["Output: " + delivery["directory"], "Quality: " + delivery["quality_status"]])
    lines.append("Next: " + next_action(job))
    return "\n".join(lines)
