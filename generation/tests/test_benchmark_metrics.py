import json

import numpy as np
import pytest
import trimesh

from agile3d.benchmark_metrics import (
    EvaluationConfig,
    evaluate_meshes,
    geometry_summary,
    sample_surface,
    surface_metrics,
    _fit_transform,
)


def _write(tmp_path, mesh, name):
    path = tmp_path / name
    mesh.export(path)
    return path


def _config(**kwargs):
    return EvaluationConfig(surface_samples=5000, registration_samples=900,
                            icp_iterations=40, icp_candidates=8, **kwargs)


def test_similarity_recovers_unknown_pose_and_scale_with_disjoint_score_samples(tmp_path):
    truth = trimesh.creation.box((0.07, 0.12, 0.04))
    generated = truth.copy()
    generated.apply_scale(7.3)
    transform = trimesh.transformations.euler_matrix(0.3, 0.5, 1.2)
    transform[:3, 3] = [1, -2, 3]
    generated.apply_transform(transform)
    result = evaluate_meshes(
        _write(tmp_path, generated, "generated.glb"),
        _write(tmp_path, truth, "truth.ply"),
        ground_truth_unit_in_meters=1.0, config=_config(),
    )
    assert result["status"] == "ok", result
    scores = result["modes"]["similarity"]
    assert scores["chamfer_l1_m"] < 0.0025
    assert scores["f_scores"][0]["f_score"] > 0.99
    assert scores["native_generated_units_to_meters_after_alignment"] == pytest.approx(1 / 7.3, rel=0.04)
    assert "rigid" not in result["modes"]
    assert "unknown" in result["rigid_unavailable_reason"]
    assert scores["registration"]["initial_rotations"] == 48
    assert scores["registration"]["refined_starts"] == 8
    assert scores["volume_ratio_to_gt"] == pytest.approx(1, rel=0.1)


def test_rigid_score_retains_physical_scale_error_and_gt_unit_conversion(tmp_path):
    # Reference file is millimeters; generated asset honestly declares meters.
    truth = trimesh.creation.box((70, 120, 40))
    generated = trimesh.creation.box((0.14, 0.24, 0.08))
    result = evaluate_meshes(
        _write(tmp_path, generated, "twice_size.glb"),
        _write(tmp_path, truth, "truth_mm.obj"),
        ground_truth_unit_in_meters=0.001, generated_unit_in_meters=1.0,
        config=_config(),
    )
    assert result["status"] == "ok", result
    assert result["modes"]["similarity"]["chamfer_l1_m"] < 0.0025
    assert result["modes"]["rigid"]["chamfer_l1_m"] > 0.02
    assert result["modes"]["rigid"]["fitted_scale_on_input_coordinates"] == pytest.approx(1)
    assert result["modes"]["rigid"]["volume_ratio_to_gt"] == pytest.approx(8, rel=1e-5)
    assert result["ground_truth"]["extents_m"] == pytest.approx([0.07, 0.12, 0.04])


def test_scene_instances_preserve_transforms_and_sampling_is_reproducible(tmp_path):
    mesh = trimesh.creation.box((0.07, 0.12, 0.04))
    scene = trimesh.Scene()
    shifted = np.eye(4)
    shifted[0, 3] = 0.09
    scene.add_geometry(mesh, node_name="first")
    scene.add_geometry(mesh, node_name="second", transform=shifted)
    path = _write(tmp_path, scene, "instances.glb")
    config = EvaluationConfig(surface_samples=1000, registration_samples=300,
                              icp_candidates=2, icp_iterations=12)
    first = evaluate_meshes(path, path, ground_truth_unit_in_meters=1, config=config)
    second = evaluate_meshes(path, path, ground_truth_unit_in_meters=1, config=config)
    assert first == second
    assert first["status"] == "ok"
    assert first["generated"]["components"] == 2
    assert first["generated"]["extents_native"] == pytest.approx([0.16, 0.12, 0.04])
    assert first["modes"]["similarity"]["chamfer_l1_m"] > 0  # independent score samples
    np.random.seed(42)
    expected = np.random.random()
    np.random.seed(42)
    a = sample_surface(mesh, 100, 1701)
    b = sample_surface(mesh, 100, 1701)
    assert np.array_equal(a, b)
    assert np.random.random() == expected


def test_distance_convention_and_directional_completeness():
    source = np.array([[0.0, 0, 0], [0.0, 0, 0]])
    target = np.array([[0.0, 0, 0], [0.01, 0, 0]])
    scores = surface_metrics(source, target, (0.005, 0.01))
    assert scores["chamfer_l1_m"] == pytest.approx(0.0025)
    assert scores["chamfer_l2_m2"] == pytest.approx(0.000025)
    assert scores["f_scores"][0] == {
        "threshold_m": 0.005, "precision": 1.0, "recall": 0.5, "f_score": 2 / 3,
    }
    assert scores["f_scores"][1]["f_score"] == 1


def test_invalid_inputs_are_failure_records_not_silently_skipped(tmp_path):
    mesh = trimesh.creation.box()
    path = _write(tmp_path, mesh, "valid.glb")
    missing = evaluate_meshes(tmp_path / "missing.glb", path, ground_truth_unit_in_meters=1)
    assert missing["status"] == "failed"
    assert missing["error"]["stage"] == "load_generated"
    assert missing["modes"] == {}
    invalid_units = evaluate_meshes(path, path, ground_truth_unit_in_meters=-1)
    assert invalid_units["status"] == "failed"
    assert invalid_units["error"]["stage"] == "configuration"
    json.dumps(missing, allow_nan=False)
    nan_units = evaluate_meshes(path, path, ground_truth_unit_in_meters=float("nan"))
    assert nan_units["status"] == "failed"
    json.dumps(nan_units, allow_nan=False)
    duplicate = trimesh.Trimesh(vertices=[[0, 0, 0], [0, 0, 0], [1, 0, 0]],
                                faces=[[0, 1, 2]], process=False)
    invalid = geometry_summary(duplicate)
    assert invalid["valid"] is False
    assert "degenerate_triangles" in invalid["problems"]


def test_open_surfaces_have_no_invented_volume():
    plane = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                            faces=[[0, 1, 2]], process=False)
    summary = geometry_summary(plane)
    assert summary["valid"] is True
    assert summary["watertight"] is False
    assert summary["volume_native_cubed"] is None


def test_similarity_registration_does_not_reflect_a_chiral_cloud():
    points = np.random.default_rng(12).normal(size=(100, 3))
    mirror = points * np.array([-1, 1, 1])
    transform = _fit_transform(points, mirror, scale=True)
    assert np.linalg.det(transform[:3, :3]) > 0
    aligned = points @ transform[:3, :3].T + transform[:3, 3]
    assert np.mean((aligned - mirror)**2) > 0.1
