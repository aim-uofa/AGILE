import copy
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from agile3d.agent_loop import review_with_codex
from agile3d.codex_backend import CodexBackend
from agile3d.generation import provide_multiview, provide_mesh, submit_stage_review
from agile3d.models import PipelineConfig, Review
from agile3d.pipeline import (
    create_job,
    queue_candidate,
    verify_evidence,
    submit_review,
    export_asset,
)
from agile3d.procedural import build_mesh
from agile3d.store import digest
from agile3d.view_inputs import load_view_manifest
from agile3d.view_selection import (
    ViewSubsetReview,
    submit_view_selection,
    recover_multiview,
    grid_crop_boxes,
)
from test_paper_flow import paper as paper, mv_review, tx_review
from test_generated_texture import specification


def test_off_center_grid_keeps_entire_objects_without_neighbor_fragments():
    image = Image.new("RGB", (320, 320), "white")
    for x in (40, 210):
        image.paste("red", (x, 8, x + 60, 174))
        image.paste("blue", (x, 185, x + 60, 310))
    boxes = grid_crop_boxes(image)
    assert 174 <= boxes[0][3] <= 185
    assert boxes[0][3] == boxes[2][1]
    assert image.crop(boxes[0]).getpixel((50, 173)) == (255, 0, 0)
    assert not np.any(np.all(np.asarray(image.crop(boxes[2])) == (255, 0, 0), axis=2))


def rejected_pool(paper, tmp_path, *, backend="provided", strategy="best_subset"):
    store, old, ref, calls = paper
    config = PipelineConfig.model_validate(old["config"])
    config.mesh.backend = backend
    config.generation.multiview_fallback = strategy
    job = create_job(store, [ref], "fixture", config)
    job = queue_candidate(store, job["id"], background=False)
    for number, colors in enumerate(
        [["red", "blue", "green", "yellow"], ["cyan", "magenta", "gray", "black"]], 1
    ):
        image = Image.new("RGB", (128, 128))
        for i, color in enumerate(colors):
            image.paste(
                color,
                ((i % 2) * 64, (i // 2) * 64, (i % 2 + 1) * 64, (i // 2 + 1) * 64),
            )
        path = tmp_path / f"trial-{number}.png"
        image.save(path)
        job = provide_multiview(store, job["id"], path)
        # An explicit full-grid rejection must also retry/fallback, not kill the
        # default route at the first image.
        decision = "reject" if strategy == "best_subset" else "retry"
        job = submit_stage_review(
            store, job["id"], mv_review(job, decision, matches_reference=False)
        )
    return job


def subset_review(job, count=2):
    directions = ["front", "back", "left", "right"]
    return ViewSubsetReview(
        evidence_digest=job["attempts"][-1]["evidence_digest"],
        assessments=[
            dict(
                candidate_id=f"t{t:02d}-p{i}",
                direction=d,
                quality=6.0,
                usable=True,
                reason="Recognizable fixture view; imperfect but useful.",
            )
            for t in (1, 2)
            for i, d in enumerate(directions)
        ],
        selected_ids=["t01-p0", "t02-p1", "t01-p2"][:count],
        consistency_score=7.0,
        reasoning="Complementary views across trials; prefer coherent geometry over a complete grid.",
    )


@pytest.mark.parametrize("count", [2, 3])
def test_cross_trial_subset_reaches_surface_and_export_without_accepting_grids(
    paper, tmp_path, count
):
    store, _, _, calls = paper
    job = rejected_pool(paper, tmp_path)
    assert job["status"] == "awaiting_view_selection" and not calls
    history = copy.deepcopy(job["attempts"][0]["multiview_trials"])
    job = submit_view_selection(store, job["id"], subset_review(job, count))
    a = job["attempts"][0]
    art = a["artifacts"]
    assert job["status"] == "awaiting_mesh" and not calls
    assert a["multiview_trials"] == history and not a["view_selection"]["grid_accepted"]
    assert "multiview_review" not in a
    manifest = store.resolve(job["id"], art["view_manifest"])
    views = load_view_manifest(manifest, art["view_manifest"]["sha256"])
    assert list(views) == ["front", "back", "left"][:count]
    assert views["front"].getpixel((20, 20)) == (255, 0, 0)
    assert views["back"].getpixel((20, 20)) == (255, 0, 255)
    mesh = tmp_path / "lifted.glb"
    build_mesh(specification(), mesh)
    metadata = tmp_path / "lifted.json"
    metadata.write_text(
        json.dumps(
            {
                "mesh_sha256": digest(mesh),
                "multiview_sha256": art["multiview"]["sha256"],
            }
        )
    )
    with pytest.raises(ValueError, match="manifest"):
        provide_mesh(store, job["id"], mesh, metadata)
    metadata.write_text(
        json.dumps(
            {"mesh_sha256": digest(mesh), "view_manifest_sha256": digest(manifest)}
        )
    )
    job = provide_mesh(store, job["id"], mesh, metadata)
    assert calls == ["surface"] and job["status"] == "awaiting_texture_review"
    job = submit_stage_review(store, job["id"], tx_review(job))
    assert job["status"] == "awaiting_review"
    job = submit_review(
        store,
        job["id"],
        Review(
            evidence_digest=job["attempts"][0]["evidence_digest"],
            decision="accept",
            score=8.0,
            matches_reference=True,
            no_hand_geometry=True,
            consistent_views=True,
            feedback="Fixture completes downstream gates.",
        ),
    )
    result = export_asset(store, job["id"])
    provenance = json.loads((Path(result["directory"]) / "provenance.json").read_text())
    assert provenance["view_selection"]["selected_count"] == count
    assert provenance["view_manifest"]["sha256"] == digest(manifest)
    assert provenance["multiview_review"] is None
    assert provenance["generation_trials"]["multiview_trials"] == history
    assert result["files"] and job["status"] == "accepted"
    verify_evidence(store, store.load(job["id"]))


def test_invalid_direction_duplicate_and_stale_selection_cannot_reach_lifting(
    paper, tmp_path
):
    store, _, _, calls = paper
    job = rejected_pool(paper, tmp_path)
    review = subset_review(job)
    for bad in [
        review.model_copy(update={"evidence_digest": "0" * 64}),
        review.model_copy(update={"selected_ids": ["t01-p0", "t02-p0"]}),
        review.model_copy(update={"selected_ids": ["t01-p0", "t01-p0"]}),
        review.model_copy(update={"assessments": review.assessments[:-1]}),
    ]:
        with pytest.raises(ValueError):
            submit_view_selection(store, job["id"], bad)
    assert store.load(job["id"])["status"] == "awaiting_view_selection" and not calls


def test_old_exhausted_job_can_adopt_fallback_without_resetting_trials(paper, tmp_path):
    store, _, _, calls = paper
    job = rejected_pool(paper, tmp_path, strategy="stop")
    assert job["status"] == "exhausted" and job["exhausted_stage"] == "multiview"
    previous = (store.directory(job["id"]) / "job.json").read_bytes()
    history = copy.deepcopy(job["attempts"][0]["multiview_trials"])
    job = recover_multiview(store, job["id"])
    assert job["status"] == "awaiting_view_selection" and not calls
    assert job["attempts"][0]["multiview_trials"] == history
    snapshot = store.resolve(
        job["id"], job["attempts"][0]["artifacts"]["before_view_fallback"]
    )
    assert snapshot.read_bytes() == previous
    verify_evidence(store, job)
    with pytest.raises(ValueError):
        recover_multiview(store, job["id"])


def test_automatic_loop_selects_subset_and_hands_only_manifest_to_hunyuan(
    paper, tmp_path, monkeypatch
):
    store, _, _, calls = paper
    job = rejected_pool(paper, tmp_path, backend="hunyuan")
    lifted = []

    def select(
        self, references, sheets, candidates, evidence_digest, directory, max_views
    ):
        assert len(sheets) == 2 and len(candidates) == 8
        return subset_review(store.load(job["id"])), {"provider": "fixture"}

    def generate(config, preview, output, seed, *, view_manifest):
        lifted.append(list(load_view_manifest(view_manifest)))
        build_mesh(specification(), output)

    monkeypatch.setattr(CodexBackend, "select_view_subset", select)
    monkeypatch.setattr("agile3d.providers.generate_mesh", generate)
    # Exercise automatic selection; the normal next stage remains the texture critic.
    job = review_with_codex(store, job["id"])
    assert (
        job["status"] == "awaiting_texture_review"
        and lifted == [["front", "back"]]
        and calls == ["surface"]
    )


def test_unknown_selection_outcome_does_not_repeat_call(paper, tmp_path, monkeypatch):
    store, _, _, _ = paper
    job = rejected_pool(paper, tmp_path)
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise TimeoutError("unknown external outcome")

    monkeypatch.setattr(CodexBackend, "select_view_subset", fail)
    with pytest.raises(TimeoutError):
        review_with_codex(store, job["id"])
    with pytest.raises(ValueError, match="already started"):
        review_with_codex(store, job["id"])
    assert calls == [1]


def test_manifest_rejects_tampered_images_and_path_escape(paper, tmp_path):
    store, _, _, _ = paper
    job = rejected_pool(paper, tmp_path)
    job = submit_view_selection(store, job["id"], subset_review(job))
    path = store.resolve(job["id"], job["attempts"][0]["artifacts"]["view_manifest"])
    manifest = json.loads(path.read_text())
    bad = copy.deepcopy(manifest)
    bad["views"][0]["file"] = "../escape.png"
    other = path.with_name("bad.json")
    other.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="basenames"):
        load_view_manifest(other)
    (path.parent / manifest["views"][0]["file"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        load_view_manifest(path)


def test_mcp_exposes_selection_and_actual_named_inputs(paper, tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    store, _, _, _ = paper
    job = rejected_pool(paper, tmp_path)
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(job["config"]))
    params = StdioServerParameters(
        env={"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
        command=sys.executable,
        args=[
            "-m",
            "agile3d.mcp_server",
            "--root",
            str(store.root),
            "--config",
            str(profile),
            "--input-root",
            str(tmp_path),
        ],
    )

    async def exercise():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                names = {t.name for t in (await session.list_tools()).tools}
                assert {"continue_with_view_subset", "select_multiview_subset"} <= names
                pool = await session.call_tool(
                    "inspect_candidate", {"job_id": job["id"]}
                )
                assert not pool.isError
                assert sum(c.type == "image" for c in pool.content) == 3
                result = await session.call_tool(
                    "select_multiview_subset",
                    {"job_id": job["id"], "review": subset_review(job).model_dump()},
                )
                assert not result.isError, result.content
                result = await session.call_tool(
                    "inspect_generation_request", {"job_id": job["id"]}
                )
                assert not result.isError
                request = json.loads(
                    next(c.text for c in result.content if c.type == "text")
                )
                assert list(request["named_view_paths"]) == ["front", "back"]
                assert sum(c.type == "image" for c in result.content) == 2
                assert Path(request["view_manifest_path"]).is_file()
                assert all(
                    Path(p).is_file() for p in request["named_view_paths"].values()
                )

    asyncio.run(exercise())


@pytest.mark.parametrize("count", [2, 3])
def test_standalone_worker_passes_only_selected_views_to_shape_and_texture(
    paper, tmp_path, monkeypatch, count
):
    from types import SimpleNamespace
    from agile3d import hunyuan_worker, providers
    from agile3d.models import MeshConfig

    store, _, _, _ = paper
    job = rejected_pool(paper, tmp_path)
    job = submit_view_selection(store, job["id"], subset_review(job, count))
    artifacts = job["attempts"][0]["artifacts"]
    manifest = store.resolve(job["id"], artifacts["view_manifest"])
    preview = store.resolve(job["id"], artifacts["multiview"])
    expected = list(load_view_manifest(manifest))
    calls = []

    class Mesh:
        def export(self, path):
            Path(path).write_bytes(b"fixture GLB")

    class Shape:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def __call__(self, *, image, **kwargs):
            assert isinstance(image, dict) and list(image) == expected
            calls.append(("shape", len(image)))
            return [Mesh()]

    class Paint:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def __call__(self, mesh, *, image):
            assert len(image) == count
            calls.append(("texture", len(image)))
            return mesh

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            Generator=lambda: SimpleNamespace(manual_seed=lambda seed: seed)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hy3dgen.rembg",
        SimpleNamespace(BackgroundRemover=lambda: lambda image: image),
    )
    monkeypatch.setitem(
        sys.modules,
        "hy3dgen.shapegen",
        SimpleNamespace(
            Hunyuan3DDiTFlowMatchingPipeline=Shape,
            FloaterRemover=lambda: lambda mesh: mesh,
            DegenerateFaceRemover=lambda: lambda mesh: mesh,
        ),
    )
    monkeypatch.setitem(
        sys.modules, "hy3dgen.texgen", SimpleNamespace(Hunyuan3DPaintPipeline=Paint)
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    def run_isolated(command, *args, env):
        assert env["PYTHONNOUSERSITE"] == "1"
        assert env["CUDA_VISIBLE_DEVICES"] == "2"
        hunyuan_worker.main(command[-1])
    monkeypatch.setattr(providers, "run_process", run_isolated)
    output = tmp_path / "worker.glb"
    providers.generate_mesh(
        MeshConfig(backend="hunyuan", low_vram=False, shape_model="fixture-2mv",
                   subfolder="fixture-mv", texture_model="fixture-texture"),
        preview,
        output,
        1234,
        view_manifest=manifest,
    )
    assert output.is_file() and calls == [("shape", count), ("texture", count)]
    request = json.loads((tmp_path / "hunyuan-request.json").read_text())
    assert request["view_manifest_sha256"] == digest(manifest)
