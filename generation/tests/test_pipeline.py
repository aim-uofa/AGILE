import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import trimesh
from pydantic import ValidationError

from agile3d import providers
from agile3d.mesh import inspect_mesh, load_scene, parts
from agile3d.models import ImageConfig, QualityPolicy, Review
from agile3d.pipeline import export_asset, queue_candidate, submit_review


def candidate(sample):
    store, job_id, image, mesh = sample
    return queue_candidate(store, job_id, image=image, mesh=mesh, background=False)


def review_for(job, **changes):
    values = dict(
        evidence_digest=job["attempts"][-1]["evidence_digest"],
        decision="accept",
        score=8.0,
        matches_reference=True,
        no_hand_geometry=True,
        consistent_views=True,
        feedback="The geometry and base color match all visible reference surfaces.",
    )
    values.update(changes)
    return Review(**values)


def test_end_to_end_gated_export_and_roundtrip(sample):
    store, job_id, image, mesh = sample
    with pytest.raises(ValueError, match="accepted"):
        export_asset(store, job_id)
    job = candidate(sample)
    assert job["status"] == "awaiting_review", job
    assert job["attempts"][-1]["quality"]["passed"]
    submit_review(store, job_id, review_for(job))
    export = export_asset(store, job_id)
    assert export_asset(store, job_id) == export
    target = Path(export["directory"])
    assert (target / "textured_mesh.obj").is_file()
    assert list(target.glob("*.mtl"))
    assert list(target.glob("*.png"))
    obj_scene = trimesh.load(target / "textured_mesh.obj", force="scene", process=False)
    assert np.allclose(obj_scene.bounds, load_scene(mesh).bounds)
    assert sum(len(m.faces) for m in obj_scene.geometry.values()) == 12
    assert json.loads((target / "provenance.json").read_text())["units"] == "unknown"


def test_feedback_reaches_next_generation_and_budget_is_bounded(sample, monkeypatch):
    store, job_id, image, mesh = sample
    job = store.load(job_id)
    job["config"]["image"]["provider"] = "gemini"
    job["config"]["quality"]["max_attempts"] = 2
    store.save(job)
    calls = []

    def fake(config, references, prompt, output, previous):
        calls.append((prompt, previous))
        shutil.copyfile(image, output)
        return {"provider": "fake"}

    monkeypatch.setattr(providers, "generate_image", fake)
    first = queue_candidate(store, job_id, mesh=mesh, background=False)
    submit_review(
        store,
        job_id,
        review_for(
            first,
            decision="retry",
            score=0.0,
            feedback="Remove the extra handle and shorten the body.",
        ),
    )
    second = queue_candidate(store, job_id, mesh=mesh, background=False)
    assert "Remove the extra handle" in calls[1][0]
    assert calls[1][1].is_file()
    assert first["attempts"][0]["seed"] != second["attempts"][1]["seed"]
    last = submit_review(store, job_id, review_for(second, decision="retry", score=0.0))
    assert last["status"] == "exhausted"
    with pytest.raises(ValueError):
        queue_candidate(store, job_id, mesh=mesh, background=False)
    with pytest.raises(ValueError):
        export_asset(store, job_id)


def test_resume_after_mesh_failure_does_not_repeat_image_request(sample, monkeypatch):
    store, job_id, image, mesh = sample
    job = store.load(job_id)
    job["config"]["image"]["provider"] = "gemini"
    job["config"]["mesh"]["backend"] = "hunyuan"
    store.save(job)
    image_calls, mesh_calls = [], []

    def fake_image(config, references, prompt, output, previous):
        image_calls.append(1)
        shutil.copyfile(image, output)
        return {"provider": "fake"}

    def fake_mesh(config, input_image, output, seed):
        mesh_calls.append(1)
        if len(mesh_calls) == 1:
            raise RuntimeError("test interruption")
        shutil.copyfile(mesh, output)

    monkeypatch.setattr(providers, "generate_image", fake_image)
    monkeypatch.setattr(providers, "generate_mesh", fake_mesh)
    failed = queue_candidate(store, job_id, background=False)
    assert failed["status"] == "failed"
    resumed = queue_candidate(store, job_id, resume=True, background=False)
    assert resumed["status"] == "awaiting_review", resumed
    assert len(image_calls) == 1
    assert len(resumed["attempts"]) == 1
    assert any(
        "test interruption" in (event["error"] or "") for event in resumed["events"]
    )


def test_unknown_image_outcome_requires_explicit_new_call(sample, monkeypatch):
    store, job_id, image, mesh = sample
    job = store.load(job_id)
    job["config"]["image"]["provider"] = "gemini"
    store.save(job)
    calls = []

    def fail(*args):
        calls.append(1)
        raise TimeoutError("unknown provider outcome")

    monkeypatch.setattr(providers, "generate_image", fail)
    assert (
        queue_candidate(store, job_id, mesh=mesh, background=False)["status"]
        == "failed"
    )
    resumed = queue_candidate(store, job_id, resume=True, background=False)
    assert len(calls) == 1
    assert "retry_image=true" in resumed["attempts"][0]["error"]
    queue_candidate(store, job_id, resume=True, retry_image=True, background=False)
    assert len(calls) == 2


@pytest.mark.parametrize(
    "change",
    [
        {"score": float("nan")},
        {"score": float("inf")},
        {"score": 11.0},
        {"matches_reference": "false"},
        {"decision": "accept", "no_hand_geometry": False},
        {"unknown_field": True},
    ],
)
def test_reviews_are_strict(sample, change):
    job = candidate(sample)
    with pytest.raises(ValidationError):
        review_for(job, **change)


def test_stale_and_tampered_evidence_cannot_be_accepted(sample):
    store, job_id, image, mesh = sample
    job = candidate(sample)
    with pytest.raises(ValueError, match="stale"):
        submit_review(store, job_id, review_for(job, evidence_digest="0" * 64))
    evidence = store.resolve(job_id, job["attempts"][0]["artifacts"]["mesh_views"])
    evidence.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        submit_review(store, job_id, review_for(job))


def test_score_does_not_override_geometry_or_texture_gate(sample):
    store, job_id, image, mesh = sample
    trimesh.creation.box().export(mesh)  # Remove the texture.
    job = candidate(sample)
    assert not job["attempts"][0]["quality"]["passed"]
    with pytest.raises(ValueError, match="gates"):
        submit_review(store, job_id, review_for(job, score=10.0))
    assert store.load(job_id)["status"] == "awaiting_review"


def test_low_score_acceptance_is_rejected(sample):
    store, job_id, *_ = sample
    job = candidate(sample)
    with pytest.raises(ValueError, match="score"):
        submit_review(store, job_id, review_for(job, score=0.0))


def test_scene_instances_keep_transforms_and_materials(sample, tmp_path):
    store, job_id, image, mesh = sample
    geometry = parts(load_scene(mesh))[0]
    scene = trimesh.Scene()
    scene.add_geometry(geometry, node_name="first")
    transform = np.eye(4)
    transform[0, 3] = 3
    scene.add_geometry(geometry, node_name="second", transform=transform)
    scene.export(mesh)
    job = candidate(sample)
    assert job["status"] == "awaiting_review", job
    assert job["attempts"][0]["quality"]["faces"] == 24
    submit_review(store, job_id, review_for(job))
    exported = export_asset(store, job_id)
    reread = trimesh.load(
        Path(exported["directory"]) / "textured_mesh.obj", force="scene", process=False
    )
    assert np.allclose(reread.bounds, scene.bounds)
    assert sum(len(m.faces) for m in reread.geometry.values()) == 24


def test_import_rejects_path_traversal_and_invalid_ids(sample):
    store, *_ = sample
    with pytest.raises(ValueError):
        store.load("../some-job")
    with pytest.raises(ValueError):
        store.resolve("a" * 16, {"path": "../../outside", "sha256": ""})


def test_lock_blocks_duplicate_candidate(sample):
    from filelock import Timeout

    store, job_id, image, mesh = sample
    with store.lock(job_id):
        with pytest.raises(Timeout):
            queue_candidate(store, job_id, image=image, mesh=mesh, background=False)


def test_textureless_mesh_can_be_explicitly_evaluated_under_relaxed_policy(sample):
    store, job_id, image, mesh = sample
    trimesh.creation.box().export(mesh)
    report = inspect_mesh(mesh, QualityPolicy(require_texture=False))
    assert report["passed"] and not report["textured"]
    assert report["warnings"]


def test_plain_pbr_material_is_renderable_but_fails_texture_gate(sample):
    store, job_id, image, path = sample
    mesh = trimesh.creation.box()
    mesh.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(baseColorFactor=[200, 20, 20, 255])
    )
    mesh.export(path)
    job = candidate(sample)
    assert job["status"] == "awaiting_review", job
    assert not job["attempts"][0]["quality"]["textured"]


def test_config_rejects_embedded_credentials():
    with pytest.raises(ValidationError):
        ImageConfig(base_url="https://" + "user:password@example.org")
    with pytest.raises(ValidationError):
        ImageConfig(api_key="not-allowed")
