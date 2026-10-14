import importlib.util
import json
from pathlib import Path
import subprocess

spec = importlib.util.spec_from_file_location("hunyuan_pilot", Path(__file__).parents[1] / "examples/run_hunyuan_pilot.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_timeout_closes_running_metadata_and_generation(tmp_path):
    record = {"status": "running", "started_utc": "2026-01-01T00:00:00+00:00", "input_sha256": "abc"}
    path = tmp_path / "texture-metadata.json"
    path.write_text(json.dumps(record))
    (tmp_path / "generation.json").write_text(json.dumps({"status": "shape_completed_waiting_texture", "shape": {"status": "completed"}}))
    module.record_interruption(tmp_path, "texture", subprocess.TimeoutExpired("worker", 5))
    actual = json.loads(path.read_text())
    summary = json.loads((tmp_path / "generation.json").read_text())
    assert actual["status"] == summary["status"] == "failed"
    assert actual["input_sha256"] == "abc"
    assert actual["terminated_by_controller"]
    assert actual["error"].startswith("TimeoutExpired:")
    assert summary["shape"]["status"] == "completed"
    assert summary["texture"] == actual


def test_controller_does_not_overwrite_child_failure_or_success(tmp_path):
    path = tmp_path / "texture-metadata.json"
    for status in ("completed", "failed"):
        content = json.dumps({"status": status, "error": "original diagnostic"})
        path.write_text(content)
        module.record_interruption(tmp_path, "texture", RuntimeError("controller exit"))
        assert path.read_text() == content
