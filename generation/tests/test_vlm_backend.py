"""Offline VLM transport and full paper-route regressions; never bill providers."""

import base64
import copy
import io
import json
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest
from PIL import Image
from pydantic import ValidationError

from agile3d.agent_backend import get_agent_backend
from agile3d.agent_loop import run_agent_loop
from agile3d.codex_backend import CodexBackend
from agile3d.generation import create_from_frames
from agile3d.models import AgentConfig, PipelineConfig
from agile3d.procedural import build_mesh
from agile3d.store import Store
from agile3d.vlm_backend import VLMBackend, VLMCallError
from test_generated_texture import specification
from test_paper_flow import mv_review, tx_review
from test_paper_flow import paper as paper


def config(**changes):
    return AgentConfig(
        backend="vlm",
        base_url="https://example.test/v1",
        model="test-vision",
        **changes,
    )


def completion(value):
    return {
        "id": "mock-call",
        "model": "test-vision",
        "choices": [
            {"finish_reason": "stop", "message": {"content": json.dumps(value)}}
        ],
        "usage": {"total_tokens": 12},
    }


@pytest.fixture
def image(tmp_path, monkeypatch):
    monkeypatch.setenv("AGILE_VLM_API_KEY", "test-private-api-value")
    path = tmp_path / "evidence.jpg"
    Image.new("RGB", (40, 32), "red").save(path)
    return path


def select_value():
    return {
        "selected_indices": [0],
        "object_description": "red object",
        "reasoning": "visible front",
    }


def test_wire_images_and_schema_do_not_save_credentials(image, tmp_path, monkeypatch):
    seen = []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    class Opener:
        def open(self, request, timeout):
            seen.append(request)
            return Response(json.dumps(completion(select_value())).encode())

    monkeypatch.setattr("agile3d.vlm_backend.build_opener", lambda *args: Opener())
    result, metadata = get_agent_backend(config()).select_references(
        [image], "red object", 1, tmp_path / "calls"
    )
    assert result.selected_indices == [0]
    request = seen[0]
    assert request.full_url == "https://example.test/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer test-private-api-value"
    payload = json.loads(request.data)
    assert payload["response_format"]["json_schema"]["strict"]
    uri = payload["messages"][0]["content"][1]["image_url"]["url"]
    assert uri.startswith("data:image/png;base64,")
    with Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))) as photo:
        assert photo.size == (40, 32)
    for path in Path(metadata["call_directory"]).iterdir():
        assert "test-private-api-value" not in path.read_text()
        assert "data:image" not in path.read_text()
    assert metadata["provider"] == "vlm-api" and metadata["status"] == "completed"


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://example.test/v1",
        "https://key@example.test/v1",
        "https://example.test/v1?token=secret",
    ],
)
def test_reject_insecure_or_credential_endpoints(endpoint):
    with pytest.raises(ValidationError):
        AgentConfig(backend="vlm", model="test", base_url=endpoint)


def test_api_profile_cannot_fall_into_codex_geometry():
    with pytest.raises(ValidationError, match="Codex procedural"):
        PipelineConfig.model_validate(
            {
                "image": {"provider": "disabled"},
                "mesh": {"backend": "codex"},
                "agent": config().model_dump(),
            }
        )


def test_transient_retries_are_explicit_bounded_and_audited(
    image, tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr("agile3d.vlm_backend.time.sleep", lambda _: None)

    def post(*args):
        calls.append(1)
        if len(calls) < 3:
            raise HTTPError("https://example.test", 503, "unavailable", {}, None)
        return completion(select_value())

    backend = VLMBackend(config(max_retries=2))
    monkeypatch.setattr(backend, "_post", post)
    _, metadata = backend.select_references([image], "box", 1, tmp_path / "calls")
    assert len(calls) == 3 and len(metadata["attempts"]) == 3
    assert metadata["attempts"][0]["http_status"] == 503


@pytest.mark.parametrize(
    "outcome",
    ["timeout", "unauthorized", "refusal", "truncated", "invalid", "tool", "stale"],
)
def test_bad_or_unknown_outcomes_never_retry_or_apply(
    image, tmp_path, monkeypatch, outcome
):
    calls = []

    def post(*args):
        calls.append(1)
        if outcome == "timeout":
            raise URLError("test-private-api-value")
        if outcome == "unauthorized":
            raise HTTPError("url", 401, "test-private-api-value", {}, None)
        body = completion(
            {
                "evidence_digest": "0" * 64,
                "decision": "accept",
                "score": 8.0,
                "matches_reference": True,
                "no_hand_geometry": True,
                "consistent_views": True,
                "feedback": "good",
            }
        )
        choice = body["choices"][0]
        if outcome == "refusal":
            choice["message"]["refusal"] = "no"
        if outcome == "truncated":
            choice["finish_reason"] = "length"
        if outcome == "invalid":
            choice["message"]["content"] = "not JSON test-private-api-value"
        if outcome == "tool":
            choice["message"]["tool_calls"] = [{"function": {"name": "run"}}]
        return body

    backend = VLMBackend(config(max_retries=2))
    monkeypatch.setattr(backend, "_post", post)
    with pytest.raises(VLMCallError) as error:
        backend.review_candidate([image], image, "box", "1" * 64, tmp_path / "calls")
    assert len(calls) == 1 and "test-private-api-value" not in str(error.value)
    assert not list((tmp_path / "calls").rglob("response.json"))


def test_json_object_compatibility_still_validates_schema(image, tmp_path, monkeypatch):
    def post(endpoint, key, payload):
        assert payload["response_format"] == {"type": "json_object"}
        assert "Required JSON schema" in payload["messages"][0]["content"][0]["text"]
        return completion(select_value())

    backend = VLMBackend(config(response_format="json_object"))
    monkeypatch.setattr(backend, "_post", post)
    selection, _ = backend.select_references([image], "box", 1, tmp_path / "calls")
    assert selection.selected_indices == [0]


@pytest.mark.parametrize("texture_error", [False, True])
def test_full_paper_vlm_route_retries_texture_and_selects_without_codex(
    paper, tmp_path, monkeypatch, texture_error
):
    # Reuse the mocked Blender fixture, but execute the real jobs, mesh QA,
    # texture embedding, rendering, durable reviews and delivery state machine.
    _, _, reference, surface_calls = paper
    monkeypatch.setenv("AGILE_VLM_API_KEY", "test-private-api-value")
    monkeypatch.setattr(
        CodexBackend, "_call", lambda *a, **k: pytest.fail("Codex must never execute")
    )
    counts = Counter()
    generated = []

    def image_api(config, references, prompt, output, previous=None):
        generated.append(
            {"prompt": prompt, "references": list(references), "previous": previous}
        )
        if texture_error and "Attachments in order" in prompt:
            raise TimeoutError("Unknown image service outcome")
        Image.new("RGB", (1024, 1024), "blue").save(output)
        return {"provider": "mock-image-api", "model": "test-image"}

    def mesh_api(config, image, output, seed, **kwargs):
        build_mesh(specification(), output)
        return {"provider": "mock-hunyuan"}

    monkeypatch.setattr("agile3d.providers.generate_image", image_api)
    monkeypatch.setattr("agile3d.providers.generate_mesh", mesh_api)

    def post(self, endpoint, key, payload):
        name = payload["response_format"]["json_schema"]["name"]
        counts[name] += 1
        schema = payload["response_format"]["json_schema"]["schema"]
        evidence = (
            schema["properties"].get("evidence_digest", {}).get("enum", [None])[0]
        )
        job = {"attempts": [{"evidence_digest": evidence}]}
        text = payload["messages"][0]["content"][0]["text"]
        if name == "agile_selection":
            value = select_value()
        elif name == "agile_multiview_review":
            value = mv_review(
                job, "retry" if counts[name] == 1 else "accept"
            ).model_dump()
        elif name == "agile_texture_review":
            value = tx_review(
                job, "retry" if counts[name] % 2 else "accept"
            ).model_dump()
        elif name == "agile_review":
            geometry = "WHITE-GEOMETRY GATE" in text
            if not geometry:
                counts["final_review"] += 1
            value = {
                "evidence_digest": evidence,
                "decision": "accept"
                if geometry or counts["final_review"] > 1
                else "retry",
                "score": 8.0,
                "matches_reference": True,
                "no_hand_geometry": True,
                "consistent_views": True,
                "feedback": "Preserve the correct side and repair the back.",
            }
        elif name == "agile_final_selection":
            ids = json.loads(
                text.split("Candidate IDs in attachment order: ", 1)[1].split(
                    ". Select the best", 1
                )[0]
            )
            value = {
                "evidence_digest": evidence,
                "selected_id": ids[-1],
                "reasoning": "Best observed correspondence",
                "assessments": [
                    {
                        "candidate_id": item,
                        "geometry_fidelity": 7.0 + i / 10,
                        "texture_fidelity": 7.0 + i / 10,
                        "surface_usability": 8.0,
                        "remaining_issues": "Unseen underside",
                    }
                    for i, item in enumerate(ids)
                ],
            }
        else:
            pytest.fail(name)
        return completion(value)

    monkeypatch.setattr(VLMBackend, "_post", post)
    values = json.loads(
        (Path(__file__).parents[1] / "configs/agile-paper-vlm.json").read_text()
    )
    values["agent"] = config().model_dump()
    values["quality"]["max_attempts"] = 2
    values["image"]["model"] = "test-image"
    profile = PipelineConfig.model_validate(values)
    store = Store(tmp_path / "api-runs")
    job = create_from_frames(store, [reference], "test carton", profile, 1)
    if texture_error:
        with pytest.raises(TimeoutError):
            run_agent_loop(store, job["id"])
        assert store.load(job["id"])["status"] == "awaiting_texture"
        number = len(generated)
        with pytest.raises(ValueError, match="already started"):
            run_agent_loop(store, job["id"])
        assert len(generated) == number
        claims = list(store.root.rglob("image-api/claim.json"))
        assert len(claims) == 1
        assert json.loads(claims[0].read_text())["status"] == "outcome_unknown"
        return
    result = run_agent_loop(store, job["id"])
    assert result["status"] == "accepted", result
    assert result["delivery"]["status"] == "delivered"
    assert result["delivery"]["method"] == "vlm_original_frame_comparison"
    assert counts["agile_selection"] == 1
    assert counts["agile_multiview_review"] == 3  # One retry, then one per candidate.
    assert (
        counts["agile_texture_review"] == 4
    )  # Bake rejected, API edit accepted twice.
    assert counts["agile_review"] == 4 and counts["agile_final_selection"] == 1
    assert len(generated) == 5  # Three grids + two UV edits.
    texture_calls = [
        item for item in generated if "Attachments in order" in item["prompt"]
    ]
    assert len(texture_calls) == 2
    assert all(
        "edit_target" in item["prompt"] and "frozen_uv_guide" in item["prompt"]
        for item in texture_calls
    )
    calls_before = copy.deepcopy(counts)
    run_agent_loop(store, job["id"])
    assert counts == calls_before  # Completed calls/delivery are never replayed.
    assert not list(store.root.rglob("codex-call.json"))
    assert all(a["artifacts"]["texture_atlas"]["sha256"] for a in result["attempts"])


def test_vlm_subset_dispatch_preserves_selected_view_provenance(
    paper, tmp_path, monkeypatch
):
    from agile3d.agent_loop import review_with_agent
    from agile3d.generation import provide_multiview, submit_stage_review
    from agile3d.pipeline import create_job, queue_candidate

    _, original, reference, _ = paper
    monkeypatch.setenv("AGILE_VLM_API_KEY", "test-private-api-value")
    values = copy.deepcopy(original["config"])
    values["agent"] = config().model_dump()
    values["generation"]["multiview_fallback"] = "best_subset"
    store = Store(tmp_path / "subset-runs")
    job = create_job(store, [reference], "box", PipelineConfig.model_validate(values))
    job = queue_candidate(store, job["id"], background=False)
    for _ in range(2):
        job = provide_multiview(store, job["id"], reference)
        job = submit_stage_review(store, job["id"], mv_review(job, "retry"))
    assert job["status"] == "awaiting_view_selection"
    request = json.loads(
        store.resolve(
            job["id"], job["attempts"][-1]["artifacts"]["view_selection_request"]
        ).read_text()
    )
    ids = [item["id"] for item in request["candidates"]]

    def post(self, endpoint, key, payload):
        assert (
            payload["response_format"]["json_schema"]["name"] == "agile_view_selection"
        )
        return completion(
            {
                "evidence_digest": job["attempts"][-1]["evidence_digest"],
                "selected_ids": ids[:2],
                "consistency_score": 8.0,
                "reasoning": "Distinct compatible sides",
                "assessments": [
                    {
                        "candidate_id": item,
                        "direction": ("front", "back", "left", "right")[i % 4],
                        "quality": 8.0,
                        "usable": True,
                        "reason": "Visible surface",
                    }
                    for i, item in enumerate(ids)
                ],
            }
        )

    monkeypatch.setattr(VLMBackend, "_post", post)
    result = review_with_agent(store, job["id"])
    assert result["status"] == "awaiting_mesh"
    art = result["attempts"][-1]["artifacts"]
    manifest = json.loads(store.resolve(job["id"], art["view_manifest"]).read_text())
    assert len(manifest["views"]) == 2
    assert list(store.root.rglob("vlm-response.json"))
    assert not list(store.root.rglob("codex-call.json"))


@pytest.mark.parametrize("escaped", [False, True])
def test_response_cannot_echo_a_key_into_job_state(
    image, tmp_path, monkeypatch, escaped
):
    backend = VLMBackend(config())
    body = completion(select_value() | {"reasoning": "test-private-api-value"})
    if escaped:
        encoded = "".join("\\u%04x" % ord(char) for char in "test-private-api-value")
        body["choices"][0]["message"]["content"] = body["choices"][0]["message"][
            "content"
        ].replace("test-private-api-value", encoded)
    monkeypatch.setattr(backend, "_post", lambda *a: body)
    with pytest.raises(VLMCallError, match="credential-like"):
        backend.select_references([image], "box", 1, tmp_path / "calls")
    assert not list((tmp_path / "calls").rglob("response.json"))


def test_optional_final_texture_api_completes_without_accepting_failed_geometry(
    sample, monkeypatch
):
    from agile3d.pipeline import evidence_digest
    from test_final_output import finish

    store, jid, _, _ = sample
    job = finish(sample)
    values = job["config"]
    values["agent"] = config().model_dump()
    values["image"] = {"provider": "gemini", "model": "test-image"}
    values["surface"]["enabled"] = True
    values["texture"] = {"provider": "image_api", "layout": "mesh_uv"}
    values["generation"]["texture_review"] = True
    values["generation"]["final_texture_completion"] = True
    job["config"] = PipelineConfig.model_validate(values).model_dump()
    attempt = job["attempts"][0]
    from agile3d.mesh import load_scene
    import trimesh

    mesh = load_scene(store.resolve(jid, attempt["artifacts"]["mesh"]))
    for geometry in mesh.geometry.values():
        geometry.visual = trimesh.visual.TextureVisuals(
            uv=geometry.visual.uv,
            material=trimesh.visual.material.PBRMaterial(
                baseColorFactor=[255, 255, 255, 255]
            ),
        )
    untextured = store.directory(jid) / "prepared-white.glb"
    mesh.export(untextured)
    attempt["artifacts"] = {"prepared_mesh": store.artifact(jid, untextured)}
    for name, filename in (
        ("uv_layout", "guide.png"),
        ("surface_record", "surface.json"),
    ):
        path = store.directory(jid) / filename
        if name == "uv_layout":
            Image.new("RGB", (512, 512), "white").save(path)
        else:
            path.write_text(json.dumps({"islands": []}))
        attempt["artifacts"][name] = store.artifact(jid, path)
    attempt["evidence_digest"] = evidence_digest(job)
    store.save(job)
    history = copy.deepcopy(attempt)
    calls = []

    def generate(config, references, prompt, output, previous=None):
        calls.append(prompt)
        Image.new("RGB", (512, 512), "blue").save(output)
        return {"provider": "mock-image-api"}

    monkeypatch.setattr("agile3d.providers.generate_image", generate)
    result = run_agent_loop(store, jid)
    assert result["status"] == "exhausted" and result["accepted_attempt"] is None
    assert result["delivery"]["status"] == "delivered"
    assert result["delivery"]["quality_status"] == "best_available"
    assert result["attempts"][0] == history
    assert len(calls) == 1
    run_agent_loop(store, jid)
    assert len(calls) == 1
