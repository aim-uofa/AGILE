"""Behavioral regressions for AGILE's two independent rejection samplers."""

import json
from pathlib import Path

import pytest
from PIL import Image

from agile3d.codex_backend import CodexBackend, ReferenceSelection
from agile3d.generation import (
    create_from_frames,
    provide_multiview,
    provide_mesh,
    submit_stage_review,
)
from agile3d.models import PipelineConfig, Review
from agile3d.pipeline import (
    create_job,
    queue_candidate,
    provide_texture,
    verify_evidence,
    submit_review,
    export_asset,
)
from agile3d.stage_reviews import MultiviewReview, TextureReview
from agile3d.store import Store, digest
from agile3d.procedural import build_mesh
from test_generated_texture import specification


def mv_review(job, decision="accept", **changes):
    values = dict(
        evidence_digest=job["attempts"][-1]["evidence_digest"],
        decision=decision,
        geometry_and_views=8.0,
        texture_and_material=8.0,
        geometric_details=8.0,
        feature_consistency=8.0,
        image_quality=8.0,
        has_view_labels=False,
        rotated_views=[],
        missing_views=[],
        view_positions=["front", "back", "left", "right"],
        layout_penalty=0.0,
        matches_reference=True,
        no_hand_artifacts=True,
        feedback="Correct the narrow side thickness.",
    )
    return MultiviewReview(**(values | changes))


def tx_review(job, decision="accept", **changes):
    values = dict(
        evidence_digest=job["attempts"][-1]["evidence_digest"],
        decision=decision,
        completeness=8.0,
        accuracy_no_fiction=8.0,
        correspondence=8.0,
        color_material=8.0,
        detail_fidelity=8.0,
        extra_penalty=0.0,
        invalid_texture=False,
        key_features_missing=False,
        severe_fiction=False,
        corrections=[],
        feedback="Correct the rear label on UV island 2.",
    )
    return TextureReview(**(values | changes))


@pytest.fixture
def paper(tmp_path, monkeypatch):
    calls = []

    def surface(source, directory, config, **kwargs):
        calls.append("surface")
        directory.mkdir()
        texture = directory / "baked-albedo.png"
        Image.new("RGB", (1024, 1024), "red").save(texture)
        # The fixture uses an already UV-mapped box, while the separate Blender
        # integration test executes real remeshing, unwrap and metallic bake.
        build_mesh(specification(), directory / "prepared.glb", texture)
        build_mesh(specification(), directory / "white.glb")
        for name in ["uv-layout.png", "white-views.png"]:
            Image.new("RGB", (512, 512), "white").save(directory / name)
        (directory / "surface.blend").write_bytes(b"fixture")
        record = {"texture_baked": True, "islands": []}
        (directory / "surface.json").write_text(json.dumps(record))
        return record

    monkeypatch.setattr("agile3d.surface.prepare_surface", surface)
    ref = tmp_path / "ref.png"
    Image.new("RGB", (512, 512), "red").save(ref)
    config = PipelineConfig.model_validate(
        {
            "image": {"provider": "provided"},
            "mesh": {"backend": "provided"},
            "surface": {
                "enabled": True,
                "geometry_review": False,
                "uv_method": "smart_project",
                "target_faces": 10000,
            },
            "texture": {"provider": "provided", "layout": "mesh_uv"},
            "generation": {
                "multiview_review": True,
                "texture_review": True,
                "max_multiview_attempts": 2,
                "max_texture_attempts": 2,
            },
        }
    )
    store = Store(tmp_path / "runs")
    job = create_job(store, [ref], "test carton", config)
    job = queue_candidate(store, job["id"], background=False)
    return store, job, ref, calls


def approve_image(paper):
    store, job, image, calls = paper
    job = provide_multiview(store, job["id"], image)
    job = submit_stage_review(store, job["id"], mv_review(job))
    assert job["status"] == "awaiting_mesh", job
    assert calls == []
    verify_evidence(store, job)
    return job


def lift(paper, tmp_path):
    store, _, image, calls = paper
    job = approve_image(paper)
    mesh = tmp_path / "lifted.glb"
    build_mesh(specification(), mesh)
    provenance = tmp_path / "lifted.json"
    metadata = {
        "mesh_sha256": digest(mesh),
        "multiview_sha256": job["attempts"][-1]["artifacts"]["multiview"]["sha256"],
    }
    provenance.write_text(json.dumps(metadata | {"multiview_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="approved"):
        provide_mesh(store, job["id"], mesh, provenance)
    provenance.write_text(json.dumps(metadata))
    job = provide_mesh(store, job["id"], mesh, provenance)
    assert job["status"] == "awaiting_texture_review", job
    assert job["attempts"][0]["texture_trials"][0]["number"] == 0
    return job


def attach_texture(store, job, tmp_path, color):
    atlas = tmp_path / f"{color}.png"
    Image.new("RGB", (512, 512), color).save(atlas)
    art = job["attempts"][-1]["artifacts"]
    provenance = tmp_path / f"{color}.json"
    provenance.write_text(
        json.dumps(
            {
                "image_sha256": digest(atlas),
                "prepared_mesh_sha256": art["prepared_mesh"]["sha256"],
                "uv_layout_sha256": art["uv_layout"]["sha256"],
            }
        )
    )
    return provide_texture(store, job["id"], atlas, provenance)


def test_multiview_rejections_prevent_lifting_and_preserve_trials(paper):
    store, job, image, calls = paper
    assert job["status"] == "awaiting_multiview"
    job = provide_multiview(store, job["id"], image)
    with pytest.raises(ValueError, match="stale"):
        submit_stage_review(store, job["id"], mv_review(job, evidence_digest="0" * 64))
    with pytest.raises(ValueError, match="veto"):
        submit_stage_review(store, job["id"], mv_review(job, has_view_labels=True))
    job = submit_stage_review(store, job["id"], mv_review(job, "retry"))
    assert job["status"] == "awaiting_multiview" and calls == []
    request = json.loads(
        store.resolve(
            job["id"], job["attempts"][0]["artifacts"]["multiview_request"]
        ).read_text()
    )
    assert "thickness" in request["prompt"]
    job = provide_multiview(store, job["id"], image)
    job = submit_stage_review(store, job["id"], mv_review(job, "retry"))
    assert job["status"] == "awaiting_view_selection"
    assert len(job["attempts"][0]["multiview_trials"]) == 2 and calls == []
    assert "source_mesh" not in job["attempts"][0]["artifacts"]
    verify_evidence(store, job)


def test_texture_retries_preserve_mesh_uv_bake_and_history(paper, tmp_path):
    store, _, _, calls = paper
    job = lift(paper, tmp_path)
    original = {
        k: job["attempts"][0]["artifacts"][k]["sha256"]
        for k in ["source_mesh", "prepared_mesh", "uv_layout"]
    }
    correction = {
        "severity": "major",
        "region": "rear island 2",
        "problem": "incorrect label",
        "edit_instruction": "Restore the observed white information panel.",
    }
    job = submit_stage_review(
        store, job["id"], tx_review(job, "retry", corrections=[correction])
    )
    assert job["status"] == "awaiting_texture"
    request = json.loads(
        store.resolve(
            job["id"], job["attempts"][0]["artifacts"]["texture_request"]
        ).read_text()
    )
    assert request["critique"]["corrections"] == [correction]
    assert [item["role"] for item in request["attachments"]] == [
        "edit_target",
        "validated_multiview",
        "frozen_uv_guide",
        "original_reference_sheet",
    ]
    job = attach_texture(store, job, tmp_path, "blue")
    assert job["status"] == "awaiting_texture_review"
    first = job["attempts"][0]["artifacts"]["texture_01_texture_atlas"]
    job = submit_stage_review(store, job["id"], tx_review(job, "retry"))
    job = attach_texture(store, job, tmp_path, "green")
    assert calls == ["surface"] and len(job["attempts"]) == 1
    assert original == {
        k: job["attempts"][0]["artifacts"][k]["sha256"] for k in original
    }
    assert store.resolve(job["id"], first).exists()
    job = submit_stage_review(store, job["id"], tx_review(job))
    assert job["status"] == "awaiting_review"
    review = Review(
        evidence_digest=job["attempts"][0]["evidence_digest"],
        decision="accept",
        score=8.0,
        matches_reference=True,
        no_hand_geometry=True,
        consistent_views=True,
        feedback="Synthetic test passes all gates.",
    )
    job = submit_review(store, job["id"], review)
    assert export_asset(store, job["id"])["files"]
    verify_evidence(store, job)
    store.resolve(job["id"], first).write_bytes(b"tampered previous trial")
    with pytest.raises(ValueError):
        verify_evidence(store, job)


def test_texture_budget_and_veto_cannot_be_overridden(paper, tmp_path):
    store, _, _, calls = paper
    job = lift(paper, tmp_path)
    with pytest.raises(ValueError, match="veto"):
        submit_stage_review(store, job["id"], tx_review(job, key_features_missing=True))
    for color in ["blue", "green"]:
        job = submit_stage_review(store, job["id"], tx_review(job, "retry"))
        job = attach_texture(store, job, tmp_path, color)
    job = submit_stage_review(store, job["id"], tx_review(job, "retry"))
    assert job["status"] == "exhausted" and job["exhausted_stage"] == "texture"
    assert calls == ["surface"]
    verify_evidence(store, job)


def test_selection_is_frozen_required_and_uses_only_selected_frames(
    paper, tmp_path, monkeypatch
):
    store, old, image, _ = paper
    config = PipelineConfig.model_validate(old["config"])
    config.generation.require_selection = True
    manual = create_job(store, [image], "test", config)
    with pytest.raises(ValueError, match="selection"):
        queue_candidate(store, manual["id"], background=False)

    def select(self, frames, description, max_selected, directory):
        assert max_selected == 1 and digest(frames[0]) == digest(image)
        return ReferenceSelection(
            selected_indices=[0],
            object_description="test box",
            reasoning="Only test reference available.",
        ), {"provider": "test"}

    monkeypatch.setattr(CodexBackend, "select_references", select)
    job = create_from_frames(store, [image], "test", config, max_selected=1)
    job = queue_candidate(store, job["id"], background=False)
    assert job["status"] == "awaiting_multiview"
    assert "reference_selection" in job
    verify_evidence(store, job)


def test_stage_weights_rotations_and_correspondence(paper):
    _, job, _, _ = paper
    review = mv_review(
        job,
        geometry_and_views=10.0,
        texture_and_material=5.0,
        geometric_details=5.0,
        feature_consistency=5.0,
        image_quality=5.0,
    )
    assert review.score() == 6.5
    assert mv_review(job, rotated_views=["left"]).score() == 5.0
    assert not mv_review(
        job, view_positions=["back", "front", "left", "right"]
    ).eligible(7.0)
    assert tx_review(job, severe_fiction=True).score() == 0.0
    assert tx_review(job, extra_penalty=2.0).score() == 6.0


def test_pure_codex_asks_for_generated_texture_before_any_texture_review(
    paper, tmp_path, monkeypatch
):
    store, old, image, calls = paper
    config = PipelineConfig.model_validate(old["config"])
    config.image.provider = "disabled"
    config.mesh.backend = "codex"
    config.generation.multiview_review = False
    geometry_calls = []

    def propose(self, *args, **kwargs):
        geometry_calls.append(1)
        return specification(), {"provider": "fixture"}

    monkeypatch.setattr(CodexBackend, "propose_mesh", propose)
    job = create_job(store, [image], "synthetic pure box", config)
    job = queue_candidate(store, job["id"], background=False)
    assert job["status"] == "awaiting_texture"
    assert job["attempts"][0]["texture_trials"][0]["number"] == 1
    job = attach_texture(store, job, tmp_path, "blue")
    assert job["status"] == "awaiting_texture_review"
    job = submit_stage_review(store, job["id"], tx_review(job, "retry"))
    assert (
        job["status"] == "awaiting_texture"
        and geometry_calls == [1]
        and calls == ["surface"]
    )


def test_stage_review_unknown_outcome_never_reissues_implicitly(paper, monkeypatch):
    from agile3d.agent_loop import review_with_codex

    store, job, image, _ = paper
    job = provide_multiview(store, job["id"], image)
    calls = []

    def review(self, *args, **kwargs):
        calls.append(1)
        raise TimeoutError("unknown remote outcome")

    monkeypatch.setattr(CodexBackend, "review_multiview", review)
    with pytest.raises(TimeoutError):
        review_with_codex(store, job["id"])
    with pytest.raises(ValueError, match="already started"):
        review_with_codex(store, job["id"])
    assert calls == [1]
    path = store.directory(job["id"]) / "attempt-01/multiview-01/codex-call.json"
    assert json.loads(path.read_text())["status"] == "outcome_unknown"


def test_automatic_lifting_runs_only_after_approved_multiview(
    paper, tmp_path, monkeypatch
):
    store, old, image, calls = paper
    config = PipelineConfig.model_validate(old["config"])
    config.mesh.backend = "hunyuan"
    lifted = []

    def generate(config, image_path, output, seed):
        lifted.append(digest(image_path))
        build_mesh(specification(), output)

    monkeypatch.setattr("agile3d.providers.generate_mesh", generate)
    job = create_job(store, [image], "synthetic automatic box", config)
    job = queue_candidate(store, job["id"], background=False)
    job = provide_multiview(store, job["id"], image)
    assert not lifted
    job = submit_stage_review(store, job["id"], mv_review(job))
    assert job["status"] == "awaiting_texture_review", job
    assert lifted == [job["attempts"][0]["artifacts"]["multiview"]["sha256"]]
    assert calls == ["surface"]


def test_new_stage_tools_over_real_mcp_stdio(paper, tmp_path):
    import asyncio
    import sys
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    store, job, image, calls = paper
    config = tmp_path / "profile.json"
    config.write_text(json.dumps(job["config"]))
    params = StdioServerParameters(
        env={"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
        command=sys.executable,
        args=[
            "-m",
            "agile3d.mcp_server",
            "--root",
            str(store.root),
            "--config",
            str(config),
            "--input-root",
            str(tmp_path),
        ],
    )

    def decoded(result):
        assert not result.isError, result.content
        return result.structuredContent or json.loads(result.content[0].text)

    async def exercise():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                names = {item.name for item in (await session.list_tools()).tools}
                assert {
                    "select_and_create_job",
                    "apply_generated_multiview",
                    "apply_lifted_mesh",
                    "review_multiview_candidate",
                    "review_texture_candidate",
                } <= names
                request = await session.call_tool(
                    "inspect_generation_request", {"job_id": job["id"]}
                )
                assert (
                    not request.isError
                    and sum(c.type == "image" for c in request.content) == 1
                )
                draft = decoded(
                    await session.call_tool(
                        "apply_generated_multiview",
                        {"job_id": job["id"], "image_path": str(image)},
                    )
                )
                evidence = await session.call_tool(
                    "inspect_candidate", {"job_id": job["id"]}
                )
                assert (
                    not evidence.isError
                    and sum(c.type == "image" for c in evidence.content) == 2
                )
                approved = decoded(
                    await session.call_tool(
                        "review_multiview_candidate",
                        {"job_id": job["id"], "review": mv_review(draft).model_dump()},
                    )
                )
                assert approved["status"] == "awaiting_mesh"
                request = await session.call_tool(
                    "inspect_generation_request", {"job_id": job["id"]}
                )
                assert not request.isError

    asyncio.run(exercise())
    assert calls == []
