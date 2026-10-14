"""Evaluate frozen paired jobs after generation; this module never calls a model.

Input JSON specifies data_manifest, methods (two names), seeds, candidate_budget
and runs [{scene, method, seed, job_manifest}]. Paths resolve relative to that
JSON. Every dataset scene/method/seed cell remains in the result, including jobs
that are missing, rejected or failed. All attempted meshes are scored; the
accepted attempt or final attempted candidate is primary, never the GT-best one.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageOps

import agile3d.benchmark_metrics as benchmark_module
from agile3d.benchmark_metrics import EvaluationConfig, evaluate_meshes
from agile3d.store import atomic_json, digest


def _path(value, base):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _pixel_digest(path):
    # Mirrors create_job's lossless-to-PNG reference preparation, so JPG source
    # bytes and PNG copied bytes can still be compared without a false mismatch.
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((4096, 4096))
        value = str(image.size).encode() + image.tobytes()
    return hashlib.sha256(value).hexdigest()


def _artifact(directory, artifact):
    path = (directory / artifact["path"]).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("Job artifact escapes its job directory")
    if digest(path) != artifact["sha256"]:
        raise ValueError("Job artifact checksum mismatch")
    return path


def _failure(stage, message):
    return {"status": "failed", "error": {"stage": stage, "message": message}, "modes": {}}


def _row_metrics(metrics):
    similarity = metrics.get("modes", {}).get("similarity", {})
    scores = {round(x["threshold_m"], 8): x["f_score"] for x in similarity.get("f_scores", [])}
    chamfer = similarity.get("chamfer_l1_m")
    return {
        "geometry_eval_status": metrics["status"],
        "chamfer_mm": chamfer * 1000 if chamfer is not None else None,
        "f5mm": scores.get(0.005), "f10mm": scores.get(0.01), "f20mm": scores.get(0.02),
        "metric_error": metrics.get("error"),
    }


def _expected_references(scene, data_base):
    values = []
    for row in scene["reference_frames"]:
        path = _path(row.get("delivered_path") or row["local_path"], data_base)
        if digest(path) != row["sha256"]:
            raise ValueError("Source reference SHA-256 differs from the frozen data manifest")
        values.append(_pixel_digest(path))
    return values


def _apply_selections(scenes, selection_specs, spec_base, data_base):
    records = {}
    for name, record in selection_specs.items():
        if name not in scenes:
            raise ValueError("Reference selection names a scene outside the data manifest")
        path = _path(record["path"], spec_base)
        if digest(path) != record["sha256"]:
            raise ValueError("Reference selection SHA-256 differs from the frozen experiment")
        selection = json.loads(path.read_text())
        indices = selection["selection"]["selected_indices"]
        references = scenes[name]["reference_frames"]
        if (not indices or len(set(indices)) != len(indices)
                or any(not isinstance(i, int) or not 0 <= i < len(references) for i in indices)):
            raise ValueError("Invalid shared reference selection indices")
        selected = [references[i] for i in indices]
        selected_paths = selection["selected_paths"]
        if len(selected_paths) != len(selected):
            raise ValueError("Reference selection paths and indices disagree")
        for image, original in zip(selected_paths, selected):
            if digest(_path(image, path.parent)) != original["sha256"]:
                raise ValueError("Selected RGB differs from the frozen original reference")
        scenes[name] = {**scenes[name], "reference_frames": selected}
        records[name] = {"path": str(path), "sha256": record["sha256"],
                         "selected_indices": indices,
                         "object_description": selection["selection"].get("object_description"),
                         "selection_finished_at": selection.get("metadata", {}).get("finished_at")}
    return records


def _collect_job(run, scene, spec_base, data_base, budget, evaluator, destination,
                 evaluate_only_terminal=True):
    job_path = _path(run["job_manifest"], spec_base)
    job = json.loads(job_path.read_text())
    directory = job_path.parent
    problems = []
    configured_budget = job.get("config", {}).get("quality", {}).get("max_attempts")
    if configured_budget != budget:
        problems.append(f"Candidate budget differs: expected {budget}, configured {configured_budget}")
    attempts = job.get("attempts", [])
    numbers = [a["number"] for a in attempts]
    if len(numbers) != len(set(numbers)) or any(not isinstance(n, int) or n < 1 for n in numbers):
        raise ValueError("Attempt numbers must be distinct positive integers")
    if len(attempts) > budget:
        problems.append("Actual candidate count exceeds the declared common budget")
    try:
        actual = [_pixel_digest(_artifact(directory, r)) for r in job["references"]]
        expected = _expected_references(scene, data_base)
        if actual != expected:
            problems.append("Reference pixel hashes/order differ from the frozen dataset manifest")
        reference_check = {"verified": actual == expected, "actual_pixel_hashes": actual,
                           "expected_pixel_hashes": expected}
    except Exception as exc:
        reference_check = {"verified": False, "error": str(exc)}
        problems.append("Unable to verify frozen references")
    status = job.get("status", "unknown")
    terminal = status in {"accepted", "rejected", "exhausted", "failed"}
    if not terminal:
        problems.append("Generation/review has not reached a terminal state")
    accepted = job.get("accepted_attempt") if status == "accepted" else None
    if status == "accepted" and accepted not in numbers:
        raise ValueError("Accepted job does not identify an existing accepted attempt")
    selected = accepted or (attempts[-1].get("number") if attempts else None)
    truth = scene["ground_truth"]
    gt_path = _path(truth["local_path"], data_base)
    if digest(gt_path) != truth["sha256"]:
        raise ValueError("Ground-truth checksum differs from the frozen data manifest")
    rows = []
    for attempt in attempts or [{"number": None, "artifacts": {}}]:
        number = attempt.get("number")
        review = attempt.get("review") or {}
        row = {"scene": run["scene"], "method": run["method"], "seed": run["seed"],
               "attempt": number, "primary": number == selected, "job_status": status,
               "accepted": bool(accepted is not None and number == accepted),
               "review_decision": review.get("decision"), "review_score": review.get("score"),
               "quality_passed": attempt.get("quality", {}).get("passed"),
               "generation_error": attempt.get("error"), "job_manifest": str(job_path)}
        suffix = f"attempt-{number:02d}" if number is not None else "no-candidate"
        metric_path = destination / run["scene"] / run["method"] / f"seed-{run['seed']}" / (suffix + ".json")
        try:
            artifact = attempt.get("artifacts", {}).get("mesh")
            if evaluate_only_terminal and not terminal:
                metrics = _failure("pending_generation_review", "Job is not terminal; GT scoring withheld until candidate selection finishes")
            elif not artifact:
                metrics = _failure("generation", "No output mesh for this attempted candidate")
            else:
                mesh = _artifact(directory, artifact)
                row["mesh_path"] = str(mesh)
                # Unknown physical units remain unknown. Only an independently
                # frozen units declaration in the experiment spec can enable rigid.
                cache_input = {"job_manifest_sha256": digest(job_path), "mesh_sha256": digest(mesh),
                               "gt_sha256": digest(gt_path), "gt_unit_in_meters": truth["unit_in_meters"],
                               "generated_unit_in_meters": run.get("generated_unit_in_meters"),
                               "config": evaluator.__dict__,
                               "evaluator_source_sha256": digest(Path(benchmark_module.__file__))}
                cache_key = hashlib.sha256(json.dumps(cache_input, sort_keys=True).encode()).hexdigest()
                cached = json.loads(metric_path.read_text()) if metric_path.is_file() else {}
                if cached.get("evaluation_cache", {}).get("key") == cache_key:
                    metrics = cached
                    row["metrics_cache_hit"] = True
                else:
                    metrics = evaluate_meshes(
                        mesh, gt_path,
                        ground_truth_unit_in_meters=truth["unit_in_meters"],
                        generated_unit_in_meters=run.get("generated_unit_in_meters"),
                        config=evaluator,
                    )
                    metrics["evaluation_cache"] = {"key": cache_key, **cache_input}
                    row["metrics_cache_hit"] = False
        except Exception as exc:
            metrics = _failure("artifact_verification", str(exc))
        atomic_json(metric_path, metrics)
        row.update(_row_metrics(metrics), metrics_path=str(metric_path))
        rows.append(row)
    return rows, {"scene": run["scene"], "method": run["method"], "seed": run["seed"],
                  "job_status": status, "job_manifest": str(job_path),
                  "configured_candidate_budget": configured_budget, "candidate_count": len(attempts),
                  "selected_attempt": selected, "selection_policy": "accepted_attempt_else_last_attempt",
                  "reference_check": reference_check, "protocol_deviations": problems,
                  "config": job.get("config"), "description": job.get("description"),
                  "created_at": job.get("created_at"), "updated_at": job.get("updated_at")}


def collect_experiment(spec_path: str | Path, output_dir: str | Path,
                       evaluator: EvaluationConfig | None = None) -> dict:
    """Save JSON/CSV, with all scheduled cells and all generated candidate scores."""
    spec_path, output_dir = Path(spec_path).resolve(), Path(output_dir).resolve()
    spec = json.loads(spec_path.read_text())
    methods, seeds = spec["methods"], spec["seeds"]
    if len(methods) != 2 or len(set(methods)) != 2 or not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Specify two distinct methods and distinct requested seeds")
    if any(not isinstance(s, int) or s < 0 for s in seeds):
        raise ValueError("Requested seeds must be nonnegative integers")
    if not isinstance(spec["candidate_budget"], int) or spec["candidate_budget"] < 1:
        raise ValueError("candidate_budget must be a positive integer")
    data_path = _path(spec["data_manifest"], spec_path.parent)
    data = json.loads(data_path.read_text())
    scenes = {row["scene"]: row for row in data["scenes"]}
    selection_records = _apply_selections(scenes, spec.get("reference_selections", {}),
                                          spec_path.parent, data_path.parent)
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", value) for value in [*methods, *scenes]):
        raise ValueError("Method and scene names must be safe directory identifiers")
    evaluator = evaluator or EvaluationConfig()
    evaluator.validate()
    if tuple(evaluator.thresholds_m) != (0.005, 0.01, 0.02):
        raise ValueError("Paired pilot fixes F-score thresholds at 5, 10 and 20 mm")
    runs = {}
    for run in spec["runs"]:
        key = run["scene"], run["method"], run["seed"]
        if key in runs:
            raise ValueError("Duplicate scene/method/seed run")
        if key[0] not in scenes or key[1] not in methods or key[2] not in seeds:
            raise ValueError("Run is outside the frozen scene/method/seed grid")
        runs[key] = run
    candidate_rows, jobs = [], []
    for scene_name, scene in scenes.items():
        for seed in seeds:
            for method in methods:
                key = scene_name, method, seed
                try:
                    if key not in runs:
                        raise ValueError("No job supplied for this scheduled scene/method/seed")
                    rows, job = _collect_job(runs[key], scene, spec_path.parent, data_path.parent,
                                              spec["candidate_budget"], evaluator, output_dir / "metrics",
                                              evaluate_only_terminal=spec.get("evaluate_only_terminal_jobs", True))
                except Exception as exc:
                    metrics = _failure("job_collection", str(exc))
                    rows = [{"scene": scene_name, "method": method, "seed": seed, "attempt": None,
                             "primary": True, "accepted": False, "quality_passed": None,
                             "job_status": "unavailable", **_row_metrics(metrics)}]
                    job = {"scene": scene_name, "method": method, "seed": seed,
                           "job_status": "unavailable", "protocol_deviations": [str(exc)]}
                candidate_rows.extend(rows)
                jobs.append(job)
    primary = [row for row in candidate_rows if row["primary"]]
    summaries = {}
    for method in methods:
        scheduled = [row for row in primary if row["method"] == method]
        valid = [row for row in scheduled if row["geometry_eval_status"] == "ok"]
        accepted = sum(row["accepted"] for row in scheduled)
        summaries[method] = {
            "scheduled_runs": len(scheduled), "accepted_runs": accepted,
            "acceptance_rate": accepted / len(scheduled), "geometrically_evaluated_runs": len(valid),
            "missing_or_invalid_geometry_runs": len(scheduled) - len(valid),
            "pending_runs": sum(row["job_status"] not in {"accepted", "rejected", "exhausted", "failed"}
                                for row in scheduled),
            "conditional_on_evaluable_mesh": {field: float(np.mean([r[field] for r in valid])) if valid else None
                                               for field in ("chamfer_mm", "f5mm", "f10mm", "f20mm")},
            "mean_f5mm_with_missing_or_invalid_geometry_as_zero":
                sum(row["f5mm"] or 0.0 for row in scheduled) / len(scheduled),
            "interpretation": "Geometry scores include rejected candidates; they do not imply production acceptance.",
        }
    pairs = []
    for scene in scenes:
        for seed in seeds:
            paired_jobs = [j for j in jobs if j["scene"] == scene and j["seed"] == seed]
            if all("config" in j for j in paired_jobs):
                for key, getter in [("Target descriptions", lambda j: j["description"]),
                                    ("Quality policies", lambda j: j["config"].get("quality"))]:
                    if getter(paired_jobs[0]) != getter(paired_jobs[1]):
                        for job in paired_jobs:
                            job["protocol_deviations"].append(key + " differ between methods")
            a, b = [next(row for row in primary if (row["scene"], row["seed"], row["method"]) == (scene, seed, method))
                    for method in methods]
            valid = a["geometry_eval_status"] == b["geometry_eval_status"] == "ok"
            pairs.append({"scene": scene, "seed": seed, "both_meshes_evaluable": valid,
                          "difference_direction": f"{methods[1]} minus {methods[0]}",
                          "chamfer_mm_difference": b["chamfer_mm"] - a["chamfer_mm"] if valid else None,
                          "f5mm_difference": b["f5mm"] - a["f5mm"] if valid else None,
                          "both_accepted": a["accepted"] and b["accepted"]})
    report = {"schema_version": 1, "experiment_spec": str(spec_path), "experiment_sha256": digest(spec_path),
              "data_manifest": str(data_path), "data_manifest_sha256": digest(data_path),
              "shared_reference_selections": selection_records,
              "candidate_budget": spec["candidate_budget"], "requested_seeds": seeds,
              "evaluation_config": evaluator.__dict__, "methods": methods,
              "jobs": jobs, "primary_rows": primary, "candidate_rows": candidate_rows,
              "summaries": summaries, "paired_differences": pairs,
              "experiment_complete": all(j["job_status"] in {"accepted", "rejected", "exhausted", "failed"} for j in jobs),
              "protocol_fully_verified": all(not j["protocol_deviations"] for j in jobs),
              "limitations": ["Small selected training-subset pilot, not a held-out object benchmark or SOTA claim.",
                              "Final-candidate policy frozen before GT scores; all candidate metrics retained for diagnostics.",
                              "Shape scores use GT-dependent similarity alignment and do not measure recovered absolute scale.",
                              "No held-out image, texture, contact or HOI pose metrics are computed here."]}
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "paired-results.json", report)
    columns = ["scene", "method", "seed", "attempt", "primary", "job_status", "accepted",
               "quality_passed", "review_decision", "review_score", "geometry_eval_status",
               "chamfer_mm", "f5mm", "f10mm", "f20mm", "metrics_path"]
    with (output_dir / "candidate-results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(candidate_rows)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_spec", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = collect_experiment(args.experiment_spec, args.output)
    print(json.dumps({"output": str(args.output), "protocol_fully_verified": report["protocol_fully_verified"],
                      "summaries": report["summaries"]}, indent=2))


if __name__ == "__main__":
    main()
