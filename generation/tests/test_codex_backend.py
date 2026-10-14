"""Exercise the subprocess boundary without making paid Codex requests."""

import json
import time
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from agile3d.codex_backend import (
    CodexBackend,
    CodexCallError,
    CodexConfig,
    MeshComponent,
    ProceduralMeshSpec,
    audit_cli_events,
)


@pytest.fixture
def fake_codex(tmp_path, monkeypatch):
    executable = tmp_path / "fake-codex"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "Path('arguments.json').write_text(json.dumps(args))\n"
        "case = os.environ.get('AGILE_FAKE_CASE', 'ok')\n"
        "if case != 'blocked_stdin':\n"
        "    Path('received-prompt.txt').write_text(sys.stdin.read())\n"
        "print(json.dumps({'type':'thread.started','thread_id':'test-thread'}), flush=True)\n"
        "print(json.dumps({'type':'session.started','model':'fake-observed-model'}), flush=True)\n"
        "if case in ('timeout', 'blocked_stdin'):\n"
        "    child = subprocess.Popen([sys.executable, '-c', \"import time; time.sleep(.9); open('escaped-child.txt','w').write('bad')\"])\n"
        "    Path('child.pid').write_text(str(child.pid))\n"
        "    time.sleep(10)\n"
        "if case == 'nonzero':\n"
        "    print(os.environ.get('AGILE_FAKE_SECRET', ''), file=sys.stderr)\n"
        "    sys.exit(13)\n"
        "if case == 'missing': sys.exit(0)\n"
        "if case == 'flood': print('x' * 6000, flush=True)\n"
        "response = 'not JSON' if case == 'malformed' else os.environ['AGILE_FAKE_RESPONSE']\n"
        "Path(args[args.index('--output-last-message') + 1]).write_text(response)\n"
        "print(json.dumps({'type':'turn.completed','usage':{'input_tokens':12,'output_tokens':7}}), flush=True)\n"
    )
    executable.chmod(0o755)
    image = tmp_path / "reference.png"
    Image.new("RGB", (32, 32), "red").save(image)
    monkeypatch.setenv(
        "AGILE_FAKE_RESPONSE",
        json.dumps({"selected_indices": [0], "object_description": "red box", "reasoning": "clear shape"}),
    )
    return executable, image


def component(**changes):
    value = {
        "name": "body",
        "kind": "box",
        "position": [0.0, 0.0, 0.0],
        "rotation_deg": [0.0, 0.0, 0.0],
        "size": [0.6, 0.4, 1.0],
        "color": [1.0, 0.0, 0.0, 1.0],
        "profile": [],
        "torus_ratio": 0.2,
        "label_text": "",
        "label_color": [1.0, 1.0, 1.0, 1.0],
    }
    value.update(changes)
    return value


def review(**changes):
    value = {
        "evidence_digest": "a" * 64,
        "decision": "accept",
        "score": 8.0,
        "matches_reference": True,
        "no_hand_geometry": True,
        "consistent_views": True,
        "feedback": "The reference and six mesh views agree on shape and color.",
    }
    value.update(changes)
    return value


def test_review_schema_binds_exact_evidence_id(fake_codex, tmp_path, monkeypatch):
    executable, image = fake_codex
    monkeypatch.setenv('AGILE_FAKE_RESPONSE', json.dumps(review()))
    backend = CodexBackend(CodexConfig(executable=str(executable)))
    _, metadata = backend.review_candidate([image], image, 'synthetic object', 'a' * 64, tmp_path / 'bound')
    schema = json.loads((Path(metadata['call_directory']) / 'schema.json').read_text())
    assert schema['properties']['evidence_digest']['enum'] == ['a' * 64]


def test_selection_uses_cli_default_and_records_observed_model(fake_codex, tmp_path):
    executable, image = fake_codex
    result, metadata = CodexBackend(CodexConfig(executable=str(executable))).select_references(
        [image], "red box", 1, tmp_path / "calls"
    )
    assert result.selected_indices == [0]
    assert metadata["model"] == "fake-observed-model"
    assert metadata["model_source"] == "cli_event"
    assert metadata["usage"] == {"input_tokens": 12, "output_tokens": 7}
    call = Path(metadata["call_directory"])
    argv = json.loads((call / "arguments.json").read_text())
    assert "--model" not in argv and "--ignore-user-config" not in argv
    assert argv[argv.index("--sandbox") + 1] == "read-only"
    assert "--ephemeral" in argv and "--output-schema" in argv
    assert not any("bypass" in arg for arg in argv)
    assert argv[argv.index("--image") + 1] == str(image)
    assert json.loads((call / "metadata.json").read_text())["status"] == "completed"


def test_explicit_model_and_isolation_flags(fake_codex, tmp_path):
    executable, image = fake_codex
    _, metadata = CodexBackend(CodexConfig(
        executable=str(executable), model="user-chosen-model", ignore_user_config=True
    )).select_references([image], "box", 1, tmp_path / "calls")
    argv = json.loads((Path(metadata["call_directory"]) / "arguments.json").read_text())
    assert argv[argv.index("--model") + 1] == "user-chosen-model"
    assert "--ignore-user-config" in argv


def test_reasoning_and_retry_overrides_are_invocation_only(fake_codex, tmp_path):
    executable, image = fake_codex
    _, metadata = CodexBackend(CodexConfig(
        executable=str(executable), reasoning_effort="high", retry_provider="existing-custom",
        request_max_retries=0, stream_max_retries=0,
    )).select_references([image], "box", 1, tmp_path / "calls")
    argv = json.loads((Path(metadata["call_directory"]) / "arguments.json").read_text())
    assert 'model_reasoning_effort="high"' in argv
    assert "model_providers.existing-custom.request_max_retries=0" in argv
    assert "model_providers.existing-custom.stream_max_retries=0" in argv
    assert "--model" not in argv
    assert metadata["requested_reasoning_effort"] == "high"
    with pytest.raises(ValueError, match="provider ID"):
        CodexConfig(stream_max_retries=0)
    with pytest.raises(ValueError, match="built-in openai"):
        CodexConfig(retry_provider="openai", stream_max_retries=0)


@pytest.mark.parametrize("case, expected", [
    ("malformed", "malformed or schema-invalid"),
    ("missing", "without a structured response"),
    ("nonzero", "code 13"),
])
def test_failure_has_no_implicit_retry_and_redacts_logs(case, expected, fake_codex, tmp_path, monkeypatch):
    executable, image = fake_codex
    monkeypatch.setenv("AGILE_FAKE_CASE", case)
    monkeypatch.setenv("AGILE_FAKE_SECRET", "sk-" + "private-placeholder-not-a-real-credential")
    directory = tmp_path / "calls"
    with pytest.raises(CodexCallError, match=expected):
        CodexBackend(CodexConfig(executable=str(executable))).select_references([image], "box", 1, directory)
    calls = list(directory.iterdir())
    assert len(calls) == 1
    logs = (calls[0] / "stderr.log").read_text()
    assert "private-placeholder" not in logs
    if case == "nonzero":
        assert "[REDACTED]" in logs


@pytest.mark.parametrize("case", ["timeout", "blocked_stdin"])
def test_timeout_also_stops_child_process_and_blocked_stdin(case, fake_codex, tmp_path, monkeypatch):
    executable, image = fake_codex
    monkeypatch.setenv("AGILE_FAKE_CASE", case)
    started = time.monotonic()
    with pytest.raises(CodexCallError, match="timed out"):
        CodexBackend(CodexConfig(executable=str(executable), timeout_seconds=0.25)).select_references(
            [image], "x" * 50000 if case == "blocked_stdin" else "box", 1, tmp_path / "calls"
        )
    assert time.monotonic() - started < 3
    call = next((tmp_path / "calls").iterdir())
    assert json.loads((call / "metadata.json").read_text())["status"] == "timeout"
    time.sleep(1)
    assert not (call / "escaped-child.txt").exists()


def test_out_of_range_selection_is_not_marked_success(fake_codex, tmp_path, monkeypatch):
    executable, image = fake_codex
    monkeypatch.setenv("AGILE_FAKE_RESPONSE", json.dumps({
        "selected_indices": [4], "object_description": "box", "reasoning": "clear"
    }))
    with pytest.raises(CodexCallError, match="unavailable image"):
        CodexBackend(CodexConfig(executable=str(executable))).select_references([image], "box", 1, tmp_path / "calls")
    metadata = json.loads(next((tmp_path / "calls").glob("*/metadata.json")).read_text())
    assert metadata["status"] == "invalid_response"


@pytest.mark.parametrize("response, quality, expected", [
    (review(evidence_digest="b" * 64), {"passed": True}, "different evidence"),
    (review(score=6.0), {"passed": True}, "quality gates"),
    (review(), {"passed": False}, "quality gates"),
    (review(matches_reference=False), {"passed": True}, "schema-invalid"),
])
def test_review_cannot_accept_stale_or_failed_evidence(response, quality, expected, fake_codex, tmp_path, monkeypatch):
    executable, image = fake_codex
    monkeypatch.setenv("AGILE_FAKE_RESPONSE", json.dumps(response))
    with pytest.raises(CodexCallError, match=expected):
        CodexBackend(CodexConfig(executable=str(executable))).review_candidate(
            [image], image, "box", "a" * 64, tmp_path / "calls", quality=quality
        )


def test_procedural_data_is_validated_and_never_executed(fake_codex, tmp_path, monkeypatch):
    executable, image = fake_codex
    malicious_text = "__import__('pathlib').Path('executed.txt').touch()"
    spec = {
        "schema_version": 1,
        "object_name": "box",
        "approximation_scope": "Procedural shape approximation with flat materials.",
        "assumptions": ["Hidden rear face is unobserved."],
        "components": [component(label_text=malicious_text)],
    }
    monkeypatch.setenv("AGILE_FAKE_RESPONSE", json.dumps(spec))
    result, metadata = CodexBackend(CodexConfig(executable=str(executable))).propose_mesh(
        [image], "box", tmp_path / "calls", feedback="Make the body taller."
    )
    assert isinstance(result, ProceduralMeshSpec)
    assert result.components[0].label_text == malicious_text
    assert not list(tmp_path.rglob("executed.txt"))
    prompt = (Path(metadata["call_directory"]) / "prompt.txt").read_text()
    assert "Make the body taller" in prompt
    assert "without a learned 3D generator" in prompt
    schema = json.loads((Path(metadata["call_directory"]) / "schema.json").read_text())
    assert schema["properties"]["components"]["maxItems"] == 16


def test_geometry_enforces_requested_component_budget(fake_codex, tmp_path, monkeypatch):
    executable, image = fake_codex
    monkeypatch.setenv("AGILE_FAKE_RESPONSE", json.dumps({
        "schema_version": 1, "object_name": "box",
        "approximation_scope": "A procedural shape approximation with flat colors.",
        "assumptions": [], "components": [component(), component(name="extra")],
    }))
    with pytest.raises(CodexCallError, match="component budget"):
        CodexBackend(CodexConfig(executable=str(executable))).propose_mesh(
            [image], "box", tmp_path / "calls", max_components=1
        )
    call = next((tmp_path / "calls").iterdir())
    assert json.loads((call / "schema.json").read_text())["properties"]["components"]["maxItems"] == 1
    assert "at most 1 components" in (call / "prompt.txt").read_text()


def test_full_previous_spec_larger_than_old_cutoff_reaches_cli(fake_codex, tmp_path, monkeypatch):
    executable, image = fake_codex
    spec = {
        "schema_version": 1, "object_name": "box",
        "approximation_scope": "A procedural approximation whose full specification must survive feedback.",
        "assumptions": ["Unobserved side assumption " + "x" * 150 for _ in range(24)],
        "components": [component(name=f"panel-{i}", label_text="visible text " * 6) for i in range(16)],
    }
    ProceduralMeshSpec.model_validate(spec)
    prior_json = json.dumps(spec, indent=2)
    assert len(prior_json) > 12000
    feedback = "Repair label layout.\nPrevious specification: " + prior_json
    monkeypatch.setenv("AGILE_FAKE_RESPONSE", json.dumps(spec))
    _, metadata = CodexBackend(CodexConfig(executable=str(executable))).propose_mesh(
        [image], "box", tmp_path / "calls", feedback=feedback
    )
    transmitted = (Path(metadata["call_directory"]) / "received-prompt.txt").read_text()
    assert transmitted.endswith(feedback)
    assert json.loads(transmitted.split("Previous specification: ", 1)[1]) == spec


def test_oversized_feedback_is_rejected_before_a_call_without_truncation(fake_codex, tmp_path):
    executable, image = fake_codex
    calls = tmp_path / "calls"
    # The budget counts encoded bytes, including non-ASCII feedback.
    with pytest.raises(ValueError, match="96 KiB"):
        CodexBackend(CodexConfig(executable=str(executable))).propose_mesh(
            [image], "box", calls, feedback="界" * (32 * 1024 + 1)
        )
    assert not calls.exists()


@pytest.mark.parametrize("changes", [
    {"size": [0.0, 1.0, 1.0]},
    {"color": [2.0, 0.0, 0.0, 1.0]},
    {"position": [float("nan"), 0.0, 0.0]},
    {"kind": "lathe", "profile": [[0.1, 0.2], [0.5, -0.5]]},
    {"kind": "lathe", "profile": [[0.0, -0.5], [0.0, 0.5]]},
    {"kind": "extrusion", "profile": [[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]]},
    {"kind": "extrusion", "profile": [[-0.5, -0.5], [0.5, 0.5], [-0.3, 0.5], [0.5, -0.5]]},
    {"profile": [[0.0, 0.0]]},
    {"kind": "execute_python"},
])
def test_invalid_geometric_specifications_are_rejected(changes):
    with pytest.raises(ValidationError):
        MeshComponent.model_validate(component(**changes))


def test_hollow_lathe_shell_has_closed_nonmonotonic_cross_section():
    # Material boundary: outer bottom -> outer rim -> inner rim -> inner floor
    # -> center of interior floor -> underside center. Opening stays hollow.
    profile = [[0.5, -0.5], [0.5, 0.5], [0.44, 0.5],
               [0.44, -0.42], [0.0, -0.42], [0.0, -0.5]]
    value = MeshComponent.model_validate(component(kind="lathe_shell", profile=profile))
    assert value.profile == profile
    with pytest.raises(ValidationError, match="nonnegative"):
        MeshComponent.model_validate(component(kind="lathe_shell", profile=[[-0.1, -0.5], [0.5, 0.5], [0.0, 0.0]]))
    with pytest.raises(ValidationError, match="duplicate"):
        MeshComponent.model_validate(component(kind="lathe_shell", profile=profile + [profile[0]]))
    with pytest.raises(ValidationError, match="self-intersect"):
        MeshComponent.model_validate(component(kind="lathe_shell", profile=[[0.1, -0.5], [0.5, 0.5], [0.2, 0.5], [0.5, -0.5]]))


def test_limited_logs_do_not_block_process(fake_codex, tmp_path, monkeypatch):
    executable, image = fake_codex
    monkeypatch.setenv("AGILE_FAKE_CASE", "flood")
    _, metadata = CodexBackend(CodexConfig(executable=str(executable), max_log_bytes=1024)).select_references(
        [image], "box", 1, tmp_path / "calls"
    )
    assert metadata["stdout_truncated"] is True
    assert (Path(metadata["call_directory"]) / "stdout.log").stat().st_size <= 1024


def test_credential_like_response_is_scrubbed_and_not_returned(fake_codex, tmp_path, monkeypatch):
    executable, image = fake_codex
    monkeypatch.setenv("AGILE_FAKE_RESPONSE", json.dumps({
        "selected_indices": [0], "object_description": "box", "reasoning": "sk-this-is-a-placeholder-secret"
    }))
    with pytest.raises(CodexCallError, match="credential-like"):
        CodexBackend(CodexConfig(executable=str(executable))).select_references([image], "box", 1, tmp_path / "calls")
    response = next((tmp_path / "calls").glob("*/response.json")).read_text()
    assert "sk-this-is" not in response
    assert "[REDACTED]" in response


def test_event_audit_reports_tools_unknown_actions_and_truncated_logs():
    harmless = json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "tool_call is plain text here"}})
    assert audit_cli_events(harmless)["no_tool_calls_observed"] is True
    assert audit_cli_events(harmless)["audit_complete"] is False
    assert audit_cli_events(harmless + '\n{"type":"turn.completed"}')["audit_complete"] is True
    assert audit_cli_events(harmless, truncated=True)["audit_complete"] is False
    tool = json.dumps({"type": "item.completed", "item": {"id": "call1", "type": "command_execution"}})
    result = audit_cli_events(harmless + "\n" + tool)
    assert result["no_tool_calls_observed"] is False
    assert result["action_items"][0]["item_type"] == "command_execution"
    unknown = json.dumps({"type": "item.completed", "item": {"type": "new_unrecognized_action"}})
    assert audit_cli_events(unknown)["no_tool_calls_observed"] is False
