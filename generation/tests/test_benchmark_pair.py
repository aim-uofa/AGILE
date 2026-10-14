import importlib.util
import json
from pathlib import Path

from PIL import Image
import pytest
import trimesh

from agile3d.benchmark_metrics import EvaluationConfig
from agile3d.store import atomic_json, digest


spec = importlib.util.spec_from_file_location(
    "benchmark_pair", Path(__file__).parents[1] / "examples/benchmark_pair.py"
)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


@pytest.fixture
def paired_jobs(tmp_path, monkeypatch):
    image = tmp_path / "reference.jpg"
    Image.new("RGB", (32, 32), (150, 70, 10)).save(image)
    truth = tmp_path / "gt.obj"
    trimesh.creation.box().export(truth)
    data = {"scenes": [{"scene": "scene1", "reference_frames": [{"local_path": str(image), "sha256": digest(image)}],
                         "ground_truth": {"local_path": str(truth), "sha256": digest(truth),
                                          "unit_in_meters": 1.0}}]}
    atomic_json(tmp_path / "data.json", data)
    runs = []
    for method, status, count in [("hybrid", "exhausted", 2), ("procedural", "accepted", 1)]:
        directory = tmp_path / method
        directory.mkdir()
        copied = directory / "reference.png"
        Image.open(image).convert("RGB").save(copied)
        attempts = []
        for number in range(1, count + 1):
            mesh = directory / f"mesh-{number}.glb"
            trimesh.creation.box().export(mesh)
            attempts.append({"number": number,
                             "artifacts": {"mesh": {"path": mesh.name, "sha256": digest(mesh)}},
                             "quality": {"passed": True},
                             "review": {"decision": "accept" if status == "accepted" else "retry",
                                        "score": 8 if status == "accepted" else 5}})
        job = {"status": status, "accepted_attempt": 1 if status == "accepted" else None,
               "description": "The photographed object",
               "config": {"quality": {"max_attempts": 2, "min_score": 7}},
               "references": [{"path": copied.name, "sha256": digest(copied)}], "attempts": attempts}
        atomic_json(directory / "job.json", job)
        runs.append({"scene": "scene1", "method": method, "seed": 1234,
                     "job_manifest": str(directory / "job.json")})
    path = tmp_path / "experiment.json"
    atomic_json(path, {"data_manifest": "data.json", "methods": ["hybrid", "procedural"],
                       "seeds": [1234], "candidate_budget": 2, "runs": runs})
    calls = []

    def evaluator(mesh, truth, **kwargs):
        calls.append((mesh, truth, kwargs))
        # Deliberately make the first candidate better to catch GT cherry-picking.
        distance = 0.001 if mesh.stem.endswith("1") else 0.05
        return {"status": "ok", "modes": {"similarity": {
            "chamfer_l1_m": distance,
            "f_scores": [{"threshold_m": t, "f_score": 0.9 if distance < 0.01 else 0.1}
                         for t in (0.005, 0.01, 0.02)],
        }}}

    monkeypatch.setattr(collector, "evaluate_meshes", evaluator)
    return tmp_path, path, calls


def test_rejected_final_candidate_is_evaluated_without_selecting_gt_best(paired_jobs):
    root, path, calls = paired_jobs
    result = collector.collect_experiment(path, root / "results")
    assert len(calls) == 3
    assert len(result["candidate_rows"]) == 3
    primary = next(row for row in result["primary_rows"] if row["method"] == "hybrid")
    assert primary["attempt"] == 2
    assert primary["chamfer_mm"] == 50
    assert primary["accepted"] is False
    assert result["summaries"]["hybrid"]["accepted_runs"] == 0
    assert result["summaries"]["hybrid"]["geometrically_evaluated_runs"] == 1
    assert result["paired_differences"][0]["chamfer_mm_difference"] == -49
    assert result["protocol_fully_verified"] is True
    assert (root / "results/candidate-results.csv").exists()
    for _, _, kwargs in calls:
        assert kwargs["config"] == EvaluationConfig()
        assert kwargs["generated_unit_in_meters"] is None


def test_omitted_job_stays_in_scheduled_denominator(paired_jobs):
    root, path, _ = paired_jobs
    experiment = json.loads(path.read_text())
    experiment["runs"] = experiment["runs"][:1]
    atomic_json(path, experiment)
    result = collector.collect_experiment(path, root / "results")
    summary = result["summaries"]["procedural"]
    assert summary["scheduled_runs"] == 1
    assert summary["missing_or_invalid_geometry_runs"] == 1
    assert summary["conditional_on_evaluable_mesh"]["chamfer_mm"] is None
    assert summary["mean_f5mm_with_missing_or_invalid_geometry_as_zero"] == 0
    assert result["protocol_fully_verified"] is False


def test_modified_mesh_is_failure_even_if_another_candidate_is_available(paired_jobs):
    root, path, calls = paired_jobs
    (root / "hybrid/mesh-2.glb").write_bytes(b"changed output")
    result = collector.collect_experiment(path, root / "results")
    primary = next(row for row in result["primary_rows"] if row["method"] == "hybrid")
    assert primary["attempt"] == 2
    assert primary["geometry_eval_status"] == "failed"
    assert primary["chamfer_mm"] is None
    assert primary["metric_error"]["stage"] == "artifact_verification"
    assert len(calls) == 2


def test_protocol_mismatch_is_visible_without_erasing_mesh_scores(paired_jobs):
    root, path, _ = paired_jobs
    job_path = root / "procedural/job.json"
    job = json.loads(job_path.read_text())
    job["config"]["quality"]["max_attempts"] = 3
    job["description"] = "Different information was supplied"
    atomic_json(job_path, job)
    result = collector.collect_experiment(path, root / "results")
    assert result["protocol_fully_verified"] is False
    assert result["summaries"]["procedural"]["geometrically_evaluated_runs"] == 1
    deviations = next(j for j in result["jobs"] if j["method"] == "procedural")["protocol_deviations"]
    assert any("Candidate budget" in x for x in deviations)
    assert any("Target descriptions" in x for x in deviations)
    assert any("Quality policies" in x for x in deviations)


def test_reference_pixels_verified_across_jpeg_png_but_changed_image_flagged(paired_jobs):
    root, path, _ = paired_jobs
    job_path = root / "procedural/job.json"
    job = json.loads(job_path.read_text())
    image = root / "procedural/reference.png"
    Image.new("RGB", (32, 32), "blue").save(image)
    job["references"][0]["sha256"] = digest(image)
    atomic_json(job_path, job)
    result = collector.collect_experiment(path, root / "results")
    hybrid, procedural = result["jobs"]
    assert hybrid["reference_check"]["verified"] is True
    assert procedural["reference_check"]["verified"] is False
    assert result["protocol_fully_verified"] is False


def test_shared_codex_selection_is_verified_against_original_candidate_images(paired_jobs):
    root, path, _ = paired_jobs
    extra = root / "extra-reference.jpg"
    Image.new("RGB", (32, 32), "green").save(extra)
    data_path = root / "data.json"
    data = json.loads(data_path.read_text())
    data["scenes"][0]["reference_frames"].append({"local_path": str(extra), "sha256": digest(extra)})
    atomic_json(data_path, data)
    selection_path = root / "selection.json"
    atomic_json(selection_path, {"selection": {"selected_indices": [0]},
                                 "selected_paths": [str(root / "reference.jpg")]})
    experiment = json.loads(path.read_text())
    experiment["reference_selections"] = {"scene1": {"path": "selection.json", "sha256": digest(selection_path)}}
    atomic_json(path, experiment)
    result = collector.collect_experiment(path, root / "results")
    assert result["protocol_fully_verified"] is True
    assert result["shared_reference_selections"]["scene1"]["selected_indices"] == [0]
    selection_path.write_text("changed selection")
    with pytest.raises(ValueError, match="selection SHA-256"):
        collector.collect_experiment(path, root / "results")


def test_metrics_cache_is_invalidated_by_job_snapshot(paired_jobs):
    root, path, calls = paired_jobs
    collector.collect_experiment(path, root / "results")
    assert len(calls) == 3
    second = collector.collect_experiment(path, root / "results")
    assert len(calls) == 3
    assert all(row["metrics_cache_hit"] for row in second["candidate_rows"])
    job_path = root / "hybrid/job.json"
    job = json.loads(job_path.read_text())
    job["updated_at"] = "new snapshot"
    atomic_json(job_path, job)
    collector.collect_experiment(path, root / "results")
    assert len(calls) == 5


def test_nonterminal_jobs_do_not_receive_gt_scores_by_default(paired_jobs):
    root, path, calls = paired_jobs
    job_path = root / "hybrid/job.json"
    job = json.loads(job_path.read_text())
    job["status"] = "awaiting_review"
    atomic_json(job_path, job)
    result = collector.collect_experiment(path, root / "results")
    assert len(calls) == 1  # Only the other, terminal method is evaluated.
    row = next(r for r in result["primary_rows"] if r["method"] == "hybrid")
    assert row["chamfer_mm"] is None
    assert row["metric_error"]["stage"] == "pending_generation_review"
    assert result["summaries"]["hybrid"]["pending_runs"] == 1
    assert result["experiment_complete"] is False
