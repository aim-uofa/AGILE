import copy
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from agile3d.codex_backend import CodexBackend
from agile3d.final_output import (
    AssetSelection,
    finalize_asset,
    prepare_delivery,
    provide_final_texture,
)
from agile3d.mesh import load_scene, parts, texture_image
from agile3d.models import Review
from agile3d.pipeline import queue_candidate, submit_review, evidence_digest
from agile3d.store import digest


def finish(sample, *, decision="retry", score=5.0):
    store, jid, image, mesh = sample
    job = store.load(jid)
    job["config"]["quality"]["max_attempts"] = 1
    store.save(job)
    job = queue_candidate(store, jid, image=image, mesh=mesh, background=False)
    return submit_review(
        store,
        jid,
        Review(
            evidence_digest=job["attempts"][-1]["evidence_digest"],
            decision=decision,
            score=score,
            matches_reference=True,
            no_hand_geometry=True,
            consistent_views=True,
            feedback="Observed texture differs on the back.",
        ),
    )


def verdict(request, winner):
    return AssetSelection(
        evidence_digest=request["evidence_digest"],
        selected_id=winner,
        assessments=[
            dict(
                candidate_id=c["id"],
                geometry_fidelity=8.0 if c["id"] == winner else 5.0,
                texture_fidelity=8.0 if c["id"] == winner else 5.0,
                surface_usability=8.0,
                remaining_issues="Back markings remain imperfect.",
            )
            for c in request["candidates"]
        ],
        reasoning="The selected complete pair matches the reference silhouette and markings best.",
    )


def add_variant(store, job):
    """Earlier texture and later geometry are independently selectable complete pairs."""
    original = job["attempts"][0]
    next_attempt = copy.deepcopy(original)
    next_attempt["number"] = 2
    source = store.resolve(job["id"], original["artifacts"]["mesh"])
    scene = load_scene(source)
    for mesh in scene.geometry.values():
        mesh.vertices[:, 0] *= 1.3
    output = store.directory(job["id"]) / "variant.glb"
    scene.export(output)
    next_attempt["artifacts"] = {"mesh": store.artifact(job["id"], output)}
    next_attempt["review"] = dict(original["review"], score=4.0)
    job["attempts"].append(next_attempt)
    next_attempt["evidence_digest"] = evidence_digest(job)
    store.save(job)


def test_exhausted_job_delivers_embedded_texture_and_preserves_rejected_reviews(sample):
    store, jid, _, original = sample
    job = finish(sample)
    history = copy.deepcopy(job["attempts"])
    result = finalize_asset(store, jid)
    assert job["status"] == "exhausted"
    assert (
        result["status"] == "delivered" and result["quality_status"] == "best_available"
    )
    exported = Path(result["directory"])
    assert digest(exported / "textured_mesh.glb") == digest(original)
    assert (exported / "textured_mesh.obj").exists() and list(exported.glob("*.mtl"))
    source_mesh = parts(load_scene(original))[0]
    final_mesh = parts(load_scene(exported / "textured_mesh.glb"))[0]
    np.testing.assert_array_equal(source_mesh.visual.uv, final_mesh.visual.uv)
    np.testing.assert_array_equal(
        texture_image(source_mesh.visual.material),
        texture_image(final_mesh.visual.material),
    )
    saved = store.load(jid)
    assert saved["attempts"] == history and saved["status"] == "exhausted"
    assert saved["accepted_attempt"] is None and "export" not in saved
    assert finalize_asset(store, jid) == result


def test_comparison_can_choose_earlier_geometry_without_cross_uv_mixing(sample):
    store, jid, _, original = sample
    job = finish(sample)
    add_variant(store, job)
    request = prepare_delivery(store, jid)
    assert [c["id"] for c in request["candidates"]] == ["a01-mesh", "a02-mesh"]
    result = finalize_asset(store, jid, selection=verdict(request, "a01-mesh"))
    assert result["selected_id"] == "a01-mesh"
    assert digest(Path(result["directory"]) / "textured_mesh.glb") == digest(original)


def test_earlier_texture_is_kept_after_later_failed_edit(sample):
    store, jid, _, original = sample
    job = finish(sample)
    add_variant(store, job)
    a, later = job["attempts"]
    a["texture_trials"] = [
        {
            "number": 0,
            "artifacts": copy.deepcopy(a["artifacts"]),
            "review": {"score_overall": 8.0},
        },
        {
            "number": 1,
            "artifacts": later["artifacts"],
            "review": {"score_overall": 4.0},
        },
    ]
    a["artifacts"] = copy.deepcopy(later["artifacts"])
    job["attempts"] = [a]
    a["evidence_digest"] = evidence_digest(job)
    store.save(job)
    request = prepare_delivery(store, jid)
    result = finalize_asset(store, jid, selection=verdict(request, "a01-t00"))
    assert result["selected_id"] == "a01-t00"
    assert digest(Path(result["directory"]) / "textured_mesh.glb") == digest(original)


def test_saved_comparison_failure_falls_back_without_repeating_external_request(
    sample, monkeypatch
):
    store, jid, _, _ = sample
    add_variant(store, finish(sample))
    calls = []

    def unavailable(self, references, sheets, ids, evidence, directory):
        calls.append(ids)
        assert len(references) == 1 and len(sheets) == 2
        raise TimeoutError("Unknown service outcome")

    monkeypatch.setattr(CodexBackend, "select_final_asset", unavailable)
    result = finalize_asset(store, jid)
    assert result["status"] == "delivered"
    assert result["method"] == "comparison_with_prior_review_fallback"
    assert result["selected_id"] == "a01-mesh"
    assert finalize_asset(store, jid) == result and len(calls) == 1


def test_saved_codex_comparison_is_validated_and_all_candidates_are_sent(
    sample, monkeypatch
):
    store, jid, _, _ = sample
    add_variant(store, finish(sample))
    calls = []

    def compare(self, refs, sheets, ids, evidence, directory):
        calls.append(ids)
        answer = verdict(
            {"candidates": [{"id": i} for i in ids], "evidence_digest": evidence},
            "a01-mesh",
        )
        return answer, {"provider": "fixture"}

    monkeypatch.setattr(CodexBackend, "select_final_asset", compare)
    result = finalize_asset(store, jid)
    assert result["method"] == "codex_original_frame_comparison"
    assert calls == [["a01-mesh", "a02-mesh"]]


@pytest.mark.parametrize(
    "corruption", ["mesh", "reference", "snapshot", "score", "missing"]
)
def test_stale_tampered_or_incomplete_selection_cannot_be_exported(sample, corruption):
    store, jid, _, _ = sample
    finish(sample)
    request = prepare_delivery(store, jid)
    selection = verdict(request, "a01-mesh")
    job = store.load(jid)
    if corruption in ("mesh", "reference"):
        artifact = (
            request["candidates"][0]["artifacts"]["mesh"]
            if corruption == "mesh"
            else job["references"][0]
        )
        store.resolve(jid, artifact).write_bytes(b"changed")
    elif corruption == "snapshot":
        job["attempts"][0]["review"]["score"] = 9.0
        store.save(job)
    elif corruption == "score":
        selection.selected_id = "unavailable"
    else:
        selection.assessments = []
    with pytest.raises(ValueError):
        finalize_asset(store, jid, selection=selection)
    assert "delivery" not in store.load(jid)


def test_white_scaffold_requires_generated_uv_texture_even_after_geometry_exhaustion(
    sample, tmp_path
):
    store, jid, _, mesh = sample
    job = finish(sample)
    # Represent the pure Codex provided-texture route stopped at its white gate.
    job["config"]["mesh"]["backend"] = "codex"
    job["config"]["image"]["provider"] = "disabled"
    job["config"]["texture"] = {"provider": "provided", "layout": "mesh_uv"}
    job["config"]["surface"]["enabled"] = True
    a = job["attempts"][0]
    a["artifacts"] = {"prepared_mesh": a["artifacts"]["mesh"]}
    for name, path in [("uv_layout", "guide.png"), ("surface_record", "surface.json")]:
        target = store.directory(jid) / path
        if name == "uv_layout":
            Image.new("RGB", (512, 512), "white").save(target)
        else:
            target.write_text(json.dumps({"islands": []}))
        a["artifacts"][name] = store.artifact(jid, target)
    a["evidence_digest"] = evidence_digest(job)
    store.save(job)
    history = copy.deepcopy(a)
    result = finalize_asset(store, jid)
    assert result["status"] == "awaiting_generated_texture"
    request = json.loads(store.resolve(jid, result["request"]).read_text())
    texture = tmp_path / "generated.png"
    Image.new("RGB", (512, 512), "blue").save(texture)
    meta = tmp_path / "generated.json"
    metadata = {k: request[k] for k in ("prepared_mesh_sha256", "uv_layout_sha256")}
    metadata["image_sha256"] = digest(texture)
    meta.write_text(json.dumps(metadata))
    delivered = provide_final_texture(store, jid, texture, meta)
    assert (
        delivered["status"] == "delivered"
        and delivered["quality_status"] == "best_available"
    )
    assert store.load(jid)["attempts"][0] == history
    final_mesh = parts(load_scene(Path(delivered["directory"]) / "textured_mesh.glb"))[
        0
    ]
    np.testing.assert_array_equal(
        final_mesh.vertices, parts(load_scene(mesh))[0].vertices
    )
    assert texture_image(final_mesh.visual.material).getpixel((0, 0))[:3] == (0, 0, 255)


def test_export_recovers_after_local_checkpoint_interruption(sample, monkeypatch):
    store, jid, _, _ = sample
    finish(sample)
    save = store.save

    def interrupt(job):
        if job.get("delivery", {}).get("status") == "delivered":
            raise RuntimeError("Checkpoint interrupted")
        save(job)

    monkeypatch.setattr(store, "save", interrupt)
    with pytest.raises(RuntimeError, match="Checkpoint"):
        finalize_asset(store, jid)
    assert "delivery" not in store.load(jid)
    monkeypatch.setattr(store, "save", save)
    result = finalize_asset(store, jid)
    assert result["status"] == "delivered"
    assert len(list(store.directory(jid).glob("delivery-*/asset"))) == 1


def test_acceptance_and_delivery_selection_remain_independent(sample):
    store, jid, _, _ = sample
    job = finish(sample)
    add_variant(store, job)
    job["status"] = "accepted"
    job["accepted_attempt"] = 2
    job["attempts"][1]["review"]["decision"] = "accept"
    store.save(job)
    request = prepare_delivery(store, jid)
    result = finalize_asset(store, jid, selection=verdict(request, "a01-mesh"))
    assert result["quality_status"] == "best_available"
    assert (
        store.load(jid)["status"] == "accepted"
        and store.load(jid)["accepted_attempt"] == 2
    )


def test_identical_accepted_mesh_is_recognized_after_hash_deduplication(sample):
    store, jid, _, _ = sample
    job = finish(sample)
    second = copy.deepcopy(job["attempts"][0])
    second["number"] = 2
    second["review"]["decision"] = "accept"
    job["attempts"].append(second)
    job.update(status="accepted", accepted_attempt=2)
    store.save(job)
    result = finalize_asset(store, jid)
    assert result["quality_status"] == "accepted"


def test_bounded_tournament_compares_every_candidate_and_refinalists(
    sample, monkeypatch
):
    store, jid, _, _ = sample
    job = finish(sample)
    for number in range(2, 15):
        a = copy.deepcopy(job["attempts"][0])
        a["number"] = number
        scene = load_scene(store.resolve(jid, a["artifacts"]["mesh"]))
        for mesh in scene.geometry.values():
            mesh.vertices[:, 0] *= 1 + number / 20
        path = store.directory(jid) / f"variant-{number}.glb"
        scene.export(path)
        a["artifacts"] = {"mesh": store.artifact(jid, path)}
        job["attempts"].append(a)
    store.save(job)
    groups = []

    def compare(self, refs, sheets, ids, evidence, directory):
        assert len(refs) + len(sheets) <= 16 and len(ids) <= 12
        groups.append(ids)
        best = max(ids)
        return verdict(
            {"candidates": [{"id": i} for i in ids], "evidence_digest": evidence}, best
        ), {}

    monkeypatch.setattr(CodexBackend, "select_final_asset", compare)
    result = finalize_asset(store, jid)
    assert [len(group) for group in groups] == [12, 2, 2]
    assert result["selected_id"] == "a14-mesh"


def test_missing_geometry_reports_completion_needed_without_fake_output(sample):
    store, jid, _, _ = sample
    job = store.load(jid)
    job["status"] = "failed"
    store.save(job)
    result = finalize_asset(store, jid)
    assert result["status"] == "needs_geometry" and "directory" not in result


def test_cli_export_defaults_to_best_available_after_exhaustion(sample):
    import subprocess
    import sys

    store, jid, _, _ = sample
    finish(sample)
    output = subprocess.check_output(
        [sys.executable, "-m", "agile3d.cli", "--root", str(store.root), "export", jid],
        text=True,
    )
    result = json.loads(output)
    assert result["status"] == "delivered" and result["quality_status"] == "best_available"
    assert store.load(jid)["status"] == "exhausted"
