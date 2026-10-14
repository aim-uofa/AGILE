"""Exercise the real state machine, mesh builder, QA and export without APIs/GT."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace

from PIL import Image
import pytest

from agile3d import agent_loop, codex_backend, procedural, providers
from agile3d.codex_backend import MeshComponent, ProceduralMeshSpec
from agile3d.models import AgentConfig, ImageConfig, MeshConfig, PipelineConfig, QualityPolicy, Review
from agile3d.pipeline import create_job, export_asset, queue_candidate, submit_review
from agile3d.store import Store


@pytest.fixture
def codex_job(tmp_path, monkeypatch):
    references = []
    for i, color in enumerate(("red", "orange")):
        path = tmp_path / f"reference-{i}.png"
        Image.new("RGB", (48, 32), color).save(path)
        references.append(path)
    config = PipelineConfig(
        image=ImageConfig(provider="disabled"),
        mesh=MeshConfig(backend="codex"),
        quality=QualityPolicy(max_attempts=2),
        agent=AgentConfig(model="test-model", reasoning_effort="high"),
    )
    store = Store(tmp_path / "jobs")
    job = create_job(store, references, "A red rectangular object", config)
    state = SimpleNamespace(
        geometry_calls=[], review_calls=[], configs=[],
        decisions=["accept"], geometry_failure=None, review_failure=None, review_hook=None,
    )

    class FakeCodex:
        def __init__(self, config):
            state.configs.append(config)

        def propose_mesh(self, references, description, directory, feedback="", max_components=16):
            state.geometry_calls.append({"references": references, "description": description,
                                         "feedback": feedback, "directory": directory,
                                         "max_components": max_components})
            if state.geometry_failure:
                raise state.geometry_failure
            component = MeshComponent(
                name="observed-red-body", kind="box", position=[0.0, 0.0, 0.0],
                rotation_deg=[0.0, 0.0, 0.0], size=[0.6, 0.4, 1.0],
                color=[1.0, 0.0, 0.0, 1.0], profile=[], torus_ratio=0.2,
                label_text="", label_color=[1.0, 1.0, 1.0, 1.0],
            )
            spec = ProceduralMeshSpec(
                schema_version=1, object_name="red object",
                approximation_scope="A procedural shape used only to exercise the integration boundary.",
                assumptions=["Unobserved back is uniformly red."], components=[component],
            )
            return spec, {"provider": "fake-codex", "geometry_call": len(state.geometry_calls)}

        def review_candidate(self, references, views, description, evidence_digest,
                             directory, minimum_score, quality):
            # These must be real deterministic renders of the generated mesh.
            assert Path(views).is_file()
            with Image.open(views) as image:
                image.verify()
            state.review_calls.append({"references": references, "views": views,
                                       "quality": quality, "evidence_digest": evidence_digest})
            if state.review_hook:
                state.review_hook()
            if state.review_failure:
                raise state.review_failure
            index = len(state.review_calls) - 1
            decision = state.decisions[min(index, len(state.decisions) - 1)]
            return Review(
                evidence_digest=evidence_digest, decision=decision,
                score=8.0 if decision == "accept" else 5.0,
                matches_reference=True, no_hand_geometry=True, consistent_views=True,
                feedback="Shorten the body and preserve the observed red material.",
            ), {"provider": "fake-codex", "review_call": index + 1}

    monkeypatch.setattr(codex_backend, "CodexBackend", FakeCodex)

    def forbidden_provider(*args, **kwargs):
        pytest.fail("Pure Codex route called image generation or Hunyuan")

    monkeypatch.setattr(providers, "generate_image", forbidden_provider)
    monkeypatch.setattr(providers, "generate_mesh", forbidden_provider)
    return store, job["id"], state


def test_pure_codex_loop_builds_inspects_reviews_and_exports_without_other_providers(codex_job):
    store, job_id, state = codex_job
    job = agent_loop.run_codex_loop(store, job_id)
    assert job["status"] == "accepted", job
    assert job["accepted_attempt"] == 1
    attempt = job["attempts"][0]
    assert attempt["quality"]["passed"] and attempt["quality"]["textured"]
    assert "multiview" not in attempt["artifacts"]
    assert not (store.directory(job_id) / "attempt-01/multiview.png").exists()
    assert {"geometry_spec", "mesh", "quality", "mesh_views"} <= attempt["artifacts"].keys()
    assert len(state.geometry_calls) == len(state.review_calls) == 1
    assert state.geometry_calls[0]["references"] == state.review_calls[0]["references"]
    assert state.geometry_calls[0]["max_components"] == job["config"]["quality"]["max_components"]
    assert state.review_calls[0]["quality"]["faces"] == 12
    assert all(config.model == "test-model" and config.reasoning_effort == "high"
               for config in state.configs)
    exported = Path(job["export"]["directory"])
    assert (exported / "textured_mesh.obj").is_file()
    assert list(exported.glob("*.mtl")) and list(exported.glob("*.png"))
    # Re-entering a completed loop does not generate a new candidate or API call.
    assert agent_loop.run_codex_loop(store, job_id)["export"] == job["export"]
    assert len(state.geometry_calls) == len(state.review_calls) == 1


def test_retry_passes_review_quality_and_previous_spec_and_stops_at_budget_two(codex_job):
    store, job_id, state = codex_job
    state.decisions = ["retry", "retry"]
    job = agent_loop.run_codex_loop(store, job_id)
    assert job["status"] == "exhausted"
    assert len(job["attempts"]) == len(state.geometry_calls) == len(state.review_calls) == 2
    assert state.geometry_calls[0]["feedback"] == ""
    feedback = state.geometry_calls[1]["feedback"]
    assert "Shorten the body" in feedback
    assert "Geometry checks:" in feedback
    assert '"textured": true' in feedback
    prior_json = feedback.split("Previous specification: ", 1)[1]
    prior_spec = json.loads(prior_json)
    assert prior_spec["components"][0]["name"] == "observed-red-body"
    assert state.geometry_calls[0]["references"] == state.geometry_calls[1]["references"]
    assert job["attempts"][0]["seed"] != job["attempts"][1]["seed"]
    assert "export" not in job
    assert job["delivery"]["status"] == "delivered"
    assert job["delivery"]["quality_status"] == "best_available"
    assert agent_loop.run_codex_loop(store, job_id)["status"] == "exhausted"
    assert len(state.geometry_calls) == 2
    with pytest.raises(ValueError):
        queue_candidate(store, job_id, background=False)
    with pytest.raises(ValueError, match="accepted"):
        export_asset(store, job_id)


def test_second_candidate_can_be_accepted_and_only_it_is_exported(codex_job):
    store, job_id, state = codex_job
    state.decisions = ["retry", "accept"]
    job = agent_loop.run_codex_loop(store, job_id)
    assert job["status"] == "accepted"
    assert job["accepted_attempt"] == 2
    provenance = json.loads((Path(job["export"]["directory"]) / "provenance.json").read_text())
    assert provenance["attempt"] == 2
    assert job["attempts"][0]["codex_review"]["review_call"] == 1
    assert job["attempts"][1]["codex_review"]["review_call"] == 2


def test_agent_accept_cannot_override_a_real_deterministic_quality_failure(codex_job):
    store, job_id, state = codex_job
    job = store.load(job_id)
    job["config"]["quality"]["max_faces"] = 4  # The real box contains 12 faces.
    store.save(job)
    candidate = queue_candidate(store, job_id, background=False)
    assert candidate["status"] == "awaiting_review"
    assert candidate["attempts"][0]["quality"]["passed"] is False
    with pytest.raises(ValueError, match="gates"):
        agent_loop.review_with_codex(store, job_id)
    unchanged = store.load(job_id)
    assert unchanged["status"] == "awaiting_review"
    assert unchanged["accepted_attempt"] is None
    assert "export" not in unchanged
    assert len(state.geometry_calls) == len(state.review_calls) == 1


def test_unknown_codex_outcome_cannot_be_silently_repeated_on_resume(codex_job):
    store, job_id, state = codex_job
    state.geometry_failure = TimeoutError("Outcome of this Codex request is unknown")
    failed = agent_loop.run_codex_loop(store, job_id)
    assert failed["status"] == "failed"
    assert len(state.geometry_calls) == 1 and not state.review_calls
    assert failed["attempts"][0]["codex_call_started"] is True
    state.geometry_failure = None
    for retry_image in (False, True):
        resumed = queue_candidate(store, job_id, resume=True, retry_image=retry_image, background=False)
        assert resumed["status"] == "failed"
        assert "Previous Codex geometry call has unknown outcome" in resumed["attempts"][0]["error"]
        assert len(state.geometry_calls) == 1
        assert len(resumed["attempts"]) == 1


def test_unknown_review_outcome_is_claimed_before_call_and_cannot_repeat(codex_job):
    store, job_id, state = codex_job
    claim_path = store.directory(job_id) / "attempt-01/codex-review-call.json"

    def inspect_durable_claim():
        claim = json.loads(claim_path.read_text())
        assert claim["status"] == "started"
        assert claim["evidence_digest"] == store.load(job_id)["attempts"][0]["evidence_digest"]

    state.review_hook = inspect_durable_claim
    state.review_failure = TimeoutError("Review server outcome is unknown")
    with pytest.raises(TimeoutError):
        agent_loop.run_codex_loop(store, job_id)
    assert store.load(job_id)["status"] == "awaiting_review"
    assert json.loads(claim_path.read_text())["status"] == "outcome_unknown"
    state.review_failure = None
    with pytest.raises(ValueError, match="already started"):
        agent_loop.run_codex_loop(store, job_id)
    assert len(state.geometry_calls) == len(state.review_calls) == 1


def test_concurrent_reviews_do_not_make_duplicate_calls(codex_job):
    store, job_id, state = codex_job
    queue_candidate(store, job_id, background=False)
    started, release = Event(), Event()

    def pause_request():
        started.set()
        assert release.wait(timeout=5)

    state.review_hook = pause_request
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(agent_loop.review_with_codex, store, job_id)
        try:
            assert started.wait(timeout=5)
            with pytest.raises(ValueError, match="already running"):
                agent_loop.review_with_codex(store, job_id)
        finally:
            release.set()
        assert pending.result(timeout=5)["status"] == "accepted"
    assert len(state.review_calls) == 1


def test_legacy_review_log_also_blocks_unknown_reissue(codex_job):
    store, job_id, state = codex_job
    queue_candidate(store, job_id, background=False)
    old_call = store.directory(job_id) / "attempt-01/codex/codex-review-legacy"
    old_call.mkdir(parents=True)
    (old_call / "metadata.json").write_text('{"status":"timeout"}')
    with pytest.raises(ValueError, match="already started"):
        agent_loop.review_with_codex(store, job_id)
    assert not state.review_calls


def test_completed_response_survives_submission_failure_without_reissue(codex_job, monkeypatch):
    store, job_id, state = codex_job
    queue_candidate(store, job_id, background=False)
    evidence = store.load(job_id)["attempts"][0]["evidence_digest"]

    def interrupted_submit(*args):
        raise RuntimeError("Local interruption after the model responded")

    monkeypatch.setattr(agent_loop, "submit_review", interrupted_submit)
    with pytest.raises(RuntimeError, match="Local interruption"):
        agent_loop.review_with_codex(store, job_id)
    directory = store.directory(job_id) / "attempt-01"
    claim = json.loads((directory / "codex-review-call.json").read_text())
    assert claim["status"] == "submission_failed"
    with pytest.raises(ValueError, match="already started"):
        agent_loop.review_with_codex(store, job_id)
    recovered = Review.model_validate_json((directory / claim["response_file"]).read_text())
    assert recovered.evidence_digest == evidence
    assert submit_review(store, job_id, recovered)["status"] == "accepted"
    assert len(state.review_calls) == 1


def test_resume_after_builder_failure_reuses_checksummed_spec_without_new_codex_call(codex_job, monkeypatch):
    store, job_id, state = codex_job
    original_builder = procedural.build_mesh
    builder_calls = []

    def interrupt_once(spec, output):
        builder_calls.append(1)
        if len(builder_calls) == 1:
            raise RuntimeError("Local mesh builder interrupted after spec checkpoint")
        return original_builder(spec, output)

    monkeypatch.setattr(procedural, "build_mesh", interrupt_once)
    first = queue_candidate(store, job_id, background=False)
    assert first["status"] == "failed"
    assert "geometry_spec" in first["attempts"][0]["artifacts"]
    assert "mesh" not in first["attempts"][0]["artifacts"]
    resumed = queue_candidate(store, job_id, resume=True, background=False)
    assert resumed["status"] == "awaiting_review", resumed
    assert len(state.geometry_calls) == 1
    assert len(builder_calls) == 2
    assert len(resumed["attempts"]) == 1
    assert resumed["attempts"][0]["quality"]["passed"]


def test_review_metadata_stays_with_original_attempt_when_next_candidate_starts(codex_job, monkeypatch):
    store, job_id, state = codex_job
    state.decisions = ["retry"]
    queue_candidate(store, job_id, background=False)

    def submit_then_start_next(store, job_id, review):
        submitted = submit_review(store, job_id, review)
        assert submitted["status"] == "ready"
        # Simulate another worker claiming the next candidate in the gap between
        # review submission and persisting Codex's review metadata.
        queue_candidate(store, job_id, background=False)
        return submitted

    monkeypatch.setattr(agent_loop, "submit_review", submit_then_start_next)
    job = agent_loop.review_with_codex(store, job_id)
    assert len(job["attempts"]) == 2
    assert job["attempts"][0]["codex_review"]["review_call"] == 1
    assert "codex_review" not in job["attempts"][1]
