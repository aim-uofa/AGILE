"""User entry points preserve route isolation, billing guards and delivery gates."""

import json
import sys
from pathlib import Path

import pytest

from agile3d import agent_loop, doctor, workflow
from agile3d.cli import main
from agile3d.models import AgentConfig, ImageConfig, MeshConfig, PipelineConfig
from agile3d.models import Review
from agile3d.pipeline import queue_candidate, submit_review
from agile3d.store import Store
from test_agent_loop import codex_job  # Reuse real procedural geometry/QA fixtures.


def test_doctor_reports_all_unset_stages_without_printing_credentials(monkeypatch):
    secret = "fixture-credential-must-never-appear"
    monkeypatch.setenv("AGILE_VLM_API_KEY", secret)
    monkeypatch.setattr(doctor, "_installed", lambda name: True)
    raw = {
        "agent": {"backend": "vlm", "model": "YOUR_VISION_MODEL", "base_url": "https://YOUR_ENDPOINT/v1"},
        "mesh": {"python": "/ABS/env/bin/python", "repository": "/ABS/Hunyuan3D-2"},
    }
    result = doctor.inspect_config(raw)
    errors = {item["name"] for item in result["checks"] if item["status"] == "error"}
    assert {"agent.model", "agent.base_url", "image.model", "mesh.python", "mesh.repository",
            "mesh.shape_model", "mesh.subfolder", "mesh.texture_model"} <= errors
    assert secret not in json.dumps(result)
    assert "AGILE_VLM_API_KEY" in json.dumps(result)
    assert result["offline"] and result["status"] == "blocked"


@pytest.mark.parametrize("kind", ["codex", "provided"])
def test_routes_only_check_applicable_providers(monkeypatch, kind):
    monkeypatch.setattr(doctor, "_installed", lambda name: True)
    monkeypatch.setattr(doctor.subprocess, "run", lambda *a, **kw: pytest.fail("No Hunyuan interpreter should run"))
    config = PipelineConfig(
        agent=AgentConfig(executable=sys.executable),
        image=ImageConfig(provider="disabled" if kind == "codex" else "provided"),
        mesh=MeshConfig(backend=kind),
    )
    result = doctor.inspect_config(config)
    assert result["status"] == "ready", result
    names = {item["name"] for item in result["checks"]}
    assert not {"AGILE_IMAGE_API_KEY", "AGILE_VLM_API_KEY", "mesh.python", "surface.executable"} & names


def test_hunyuan_check_is_offline_and_does_not_import_cuda(tmp_path, monkeypatch):
    monkeypatch.setattr(doctor, "_installed", lambda name: True)
    repository = tmp_path / "hunyuan"
    (repository / "hy3dgen").mkdir(parents=True)
    weights = tmp_path / "shape" / "mv"
    weights.mkdir(parents=True)
    (weights / "model.safetensors").write_bytes(b"fixture")
    texture = tmp_path / "texture"
    texture.mkdir()
    (texture / "config.json").write_text("{}")
    calls = []

    def probe(args, **kwargs):
        calls.append((args, kwargs))
        assert "import torch" not in args[2]
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        assert kwargs["env"]["TRANSFORMERS_OFFLINE"] == "1"
        from types import SimpleNamespace
        return SimpleNamespace(returncode=0, stdout=json.dumps({"version": [3, 10], "modules": {"torch": True, "hy3dgen": True}}))

    monkeypatch.setattr(doctor.subprocess, "run", probe)
    config = PipelineConfig(image=ImageConfig(provider="provided"), agent=AgentConfig(executable=sys.executable),
                            mesh=MeshConfig(python=sys.executable, repository=str(repository),
                                            shape_model=str(weights.parent), subfolder="mv", texture_model=str(texture)))
    result = doctor.inspect_config(config)
    assert result["status"] == "ready", result
    assert len(calls) == 1
    assert any(item["name"] == "mesh.gpu" and item["status"] == "warning" for item in result["checks"])


def test_missing_later_dependency_blocks_first_vlm_selection(tmp_path, monkeypatch):
    monkeypatch.setenv("AGILE_VLM_API_KEY", "fixture-test-only")
    monkeypatch.setattr(doctor, "_installed", lambda name: True)
    monkeypatch.setattr(workflow, "create_from_frames", lambda *a, **kw: pytest.fail("Selection must not run before all stages pass"))
    from PIL import Image
    frame = tmp_path / "frame.png"
    Image.new("RGB", (32, 32), "red").save(frame)
    config = PipelineConfig(agent=AgentConfig(backend="vlm", model="fixture-vision", base_url="https://example.org/v1"),
                            image=ImageConfig(provider="provided"), mesh=MeshConfig(python="/ABS/env/bin/python"))
    store = Store(tmp_path / "runs")
    with pytest.raises(doctor.PreflightError, match="mesh.shape_model"):
        workflow.generate(store, [frame], "red object", config)
    assert not list(store.root.iterdir())


def test_simple_generate_automatically_delivers_and_resume_reuses_output(codex_job, monkeypatch):
    store, job_id, state = codex_job
    config = PipelineConfig.model_validate(store.load(job_id)["config"])
    monkeypatch.setattr(workflow, "require_ready", lambda *a, **kw: {"status": "ready"})

    def selection(store, frames, description, config, max_selected):
        assert max_selected == 1  # One frame works without extra selection flags.
        return store.load(job_id)

    monkeypatch.setattr(workflow, "create_from_frames", selection)
    job = workflow.generate(store, [Path("fixture.png")], "red object", config)
    assert job["delivery"]["status"] == "delivered"
    assert job["delivery"]["quality_status"] == "accepted"
    assert any(ref["path"].endswith(".obj") for ref in job["delivery"]["files"])
    assert "Output:" in workflow.format_status(job)
    monkeypatch.setattr(workflow, "require_ready", lambda *a, **kw: pytest.fail("Already delivered job needs no provider access"))
    assert workflow.resume(store, job_id)["delivery"] == job["delivery"]
    assert len(state.geometry_calls) == len(state.review_calls) == 1


@pytest.mark.parametrize("status", ["running", "queued"])
def test_resume_never_replaces_a_worker(sample, monkeypatch, status):
    store, job_id, *_ = sample
    job = store.load(job_id)
    job["status"] = status
    store.save(job)
    monkeypatch.setattr(workflow, "queue_candidate", lambda *a, **kw: pytest.fail("No duplicate worker"))
    with pytest.raises(ValueError, match="queued or running"):
        workflow.resume(store, job_id)
    assert store.load(job_id)["status"] == status


@pytest.mark.parametrize("flag", ["image_call_started", "codex_call_started"])
def test_resume_unknown_candidate_call_preserves_state(sample, monkeypatch, flag):
    store, job_id, *_ = sample
    job = store.load(job_id)
    job.update(status="failed", attempts=[{"number": 1, "artifacts": {}, flag: True}])
    store.save(job)
    before = (store.directory(job_id) / "job.json").read_bytes()
    monkeypatch.setattr(workflow, "queue_candidate", lambda *a, **kw: pytest.fail("No paid request replay"))
    with pytest.raises(ValueError, match="unknown outcome"):
        workflow.resume(store, job_id)
    assert (store.directory(job_id) / "job.json").read_bytes() == before


def test_resume_preserves_durable_unknown_review_guard(codex_job, monkeypatch):
    store, job_id, state = codex_job
    state.review_failure = RuntimeError("fixture connection dropped")
    with pytest.raises(RuntimeError, match="connection dropped"):
        agent_loop.run_agent_loop(store, job_id)
    monkeypatch.setattr(workflow, "require_ready", lambda *a, **kw: {"status": "ready"})
    state.review_failure = None
    with pytest.raises(ValueError, match="already started"):
        workflow.resume(store, job_id)
    assert len(state.geometry_calls) == len(state.review_calls) == 1


@pytest.mark.parametrize("decision", ["accept", "retry"])
def test_resume_reviews_legacy_imported_candidate_without_regenerating(sample, monkeypatch, decision):
    store, job_id, image, mesh = sample
    queue_candidate(store, job_id, image=image, mesh=mesh, background=False)
    monkeypatch.setattr(workflow, "require_ready", lambda *a, **kw: {"status": "ready"})

    def review(store, job_id):
        attempt = store.load(job_id)["attempts"][-1]
        return submit_review(store, job_id, Review(
            evidence_digest=attempt["evidence_digest"], decision=decision,
            score=8.0 if decision == "accept" else 4.0,
            matches_reference=True, no_hand_geometry=True, consistent_views=True,
            feedback="Fixture review of the imported geometry.",
        ))

    monkeypatch.setattr(workflow, "review_with_agent", review)
    monkeypatch.setattr(workflow, "queue_candidate", lambda *a, **kw: pytest.fail("No implicit reuse of imported candidates"))
    result = workflow.resume(store, job_id)
    assert len(result["attempts"]) == 1
    if decision == "accept":
        assert result["delivery"]["status"] == "delivered"
    else:
        assert result["status"] == "ready"
        assert "Supply the offline inputs" in workflow.next_action(result)


def test_generate_failed_review_shows_saved_job_and_keeps_unknown_claim(codex_job, monkeypatch):
    store, job_id, state = codex_job
    state.review_failure = RuntimeError("fixture connection dropped")
    monkeypatch.setattr(workflow, "require_ready", lambda *a, **kw: {"status": "ready"})
    monkeypatch.setattr(workflow, "create_from_frames", lambda *a, **kw: store.load(job_id))
    config = PipelineConfig.model_validate(store.load(job_id)["config"])
    with pytest.raises(workflow.WorkflowError) as failure:
        workflow.generate(store, [Path("fixture.png")], "red object", config)
    assert failure.value.report["job_id"] == job_id
    assert failure.value.report["directory"] == str(store.directory(job_id))
    assert "No automatic retry" in failure.value.report["next"]
    claim = store.directory(job_id) / "attempt-01/codex-review-call.json"
    assert json.loads(claim.read_text())["status"] == "outcome_unknown"
    assert len(state.review_calls) == 1


@pytest.mark.parametrize("as_json", [False, True])
def test_cli_selection_failure_keeps_exact_evidence_without_inventing_job(sample, monkeypatch, capsys, as_json):
    store, job_id, image, _ = sample
    config = store.load(job_id)["config"]
    profile = store.root.parent / "selection-profile.json"
    profile.write_text(json.dumps(config))
    calls = []

    class RejectedSelector:
        def select_references(self, *args, **kwargs):
            calls.append(args)
            raise RuntimeError("The selected model is not available for this account")

    monkeypatch.setattr("agile3d.agent_backend.get_agent_backend", lambda config: RejectedSelector())
    monkeypatch.setattr(workflow, "require_ready", lambda *a, **kw: {"status": "ready"})
    argv = ["--root", str(store.root), "generate", "--config", str(profile),
            "--description", "red box", "--frame", str(image)]
    with pytest.raises(SystemExit) as exit_info:
        main(argv + (["--json"] if as_json else []))
    assert exit_info.value.code == 2
    captured = capsys.readouterr()
    directories = list((store.root / "selections").iterdir())
    assert len(directories) == 1
    directory = directories[0]
    claim = json.loads((directory / "claim.json").read_text())
    assert claim["status"] == "outcome_unknown"
    assert (directory / "000.png").is_file()
    assert len(calls) == 1
    assert list(store.root.glob("*/job.json")) == [store.directory(job_id) / "job.json"]
    if as_json:
        result = json.loads(captured.out)
        assert result["directory"] == str(directory)
        assert result["job_created"] is False and "job_id" not in result
    else:
        assert str(directory) in captured.err
        assert "did not return a job ID" in captured.err
        assert "Traceback" not in captured.err


def test_status_keeps_json_compatibility_and_adds_human_output(sample, capsys):
    store, job_id, *_ = sample
    main(["--root", str(store.root), "status", job_id])
    assert json.loads(capsys.readouterr().out)["id"] == job_id
    main(["--root", str(store.root), "status", job_id, "--human"])
    output = capsys.readouterr().out
    assert f"Job: {job_id}" in output and "Next:" in output


def test_default_help_is_short_and_advanced_commands_remain_available(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "doctor" in output and "resume" in output
    assert "apply-final-texture" not in output
    with pytest.raises(SystemExit):
        main(["--help-advanced"])
    assert "apply-final-texture" in capsys.readouterr().out


def test_doctor_json_failure_is_machine_readable_and_creates_no_job(tmp_path, capsys):
    profile = tmp_path / "config.json"
    profile.write_text("{}")
    with pytest.raises(SystemExit) as exit_info:
        main(["--root", str(tmp_path / "runs"), "doctor", "--config", str(profile), "--json"])
    assert exit_info.value.code == 2
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"
    assert not (tmp_path / "runs").exists()
