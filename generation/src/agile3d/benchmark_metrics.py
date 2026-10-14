"""Independent, deterministic surface evaluation; never imported by generation.

Similarity registration is an evaluation oracle: it measures shape after fitting
pose and scale to the reference mesh, not recovery of physical size or pose.
Distances are between independently sampled surfaces, not exact triangle queries.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import io
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import trimesh


@dataclass(frozen=True)
class EvaluationConfig:
    surface_samples: int = 20000
    registration_samples: int = 3000
    seed: int = 1701
    icp_iterations: int = 60
    icp_candidates: int = 8
    thresholds_m: tuple[float, ...] = (0.005, 0.01, 0.02)

    def validate(self):
        if self.surface_samples < 100 or self.registration_samples < 100:
            raise ValueError("Use at least 100 samples for each surface")
        if not 1 <= self.icp_iterations <= 500:
            raise ValueError("icp_iterations must be in [1, 500]")
        if not 1 <= self.icp_candidates <= 48:
            raise ValueError("icp_candidates must be in [1, 48]")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if not self.thresholds_m or any(
            not np.isfinite(t) or t <= 0 for t in self.thresholds_m
        ):
            raise ValueError("F-score thresholds must be positive finite meters")


def _load_geometry(path: Path) -> trimesh.Trimesh:
    if path.suffix.lower() not in {".glb", ".obj", ".ply", ".stl"}:
        raise ValueError("Supported mesh inputs: self-contained GLB, OBJ, PLY, STL")
    if path.stat().st_size > 512 * 1024**2:
        raise ValueError("Input exceeds the 512 MiB mesh limit")
    scene = trimesh.load(
        io.BytesIO(path.read_bytes()),
        file_type=path.suffix[1:].lower(),
        force="scene",
        process=False,
        skip_materials=True,
        resolver={},
    )
    meshes = []
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        original = scene.geometry[name]
        if not isinstance(original, trimesh.Trimesh):
            raise ValueError("Point clouds and non-triangle geometry are unsupported")
        # Strip materials: evaluation needs geometry, not external texture files.
        mesh = trimesh.Trimesh(original.vertices, original.faces, process=False)
        mesh.apply_transform(transform)
        meshes.append(mesh)
    if not meshes:
        raise ValueError("Input contains no triangle geometry")
    return trimesh.util.concatenate(meshes)


def geometry_summary(mesh: trimesh.Trimesh) -> dict:
    """Validate geometry without silently repairing or discarding bad faces."""
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    result = {"vertices": len(vertices), "faces": len(faces), "valid": False}
    problems = []
    if not len(vertices) or not len(faces):
        problems.append("empty_mesh")
    if not np.isfinite(vertices).all():
        problems.append("nonfinite_vertices")
    if faces.ndim != 2 or faces.shape[1] != 3:
        problems.append("nontriangular_faces")
    elif len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)):
        problems.append("face_index_out_of_range")
    if problems:
        return {**result, "problems": problems}
    extents = np.ptp(vertices, axis=0)
    diagonal = float(np.linalg.norm(extents))
    if diagonal <= np.finfo(float).tiny:
        problems.append("zero_extent")
    areas = np.asarray(mesh.area_faces)
    degenerate = int(np.count_nonzero(areas <= diagonal**2 * 1e-14))
    if degenerate:
        problems.append("degenerate_triangles")
    if not np.isfinite(areas).all() or float(areas.sum()) <= 0:
        problems.append("invalid_surface_area")
    # Welding UV seams is for topology reporting only. Surface samples below use
    # the original triangles, including disconnected components and all surfaces.
    topology = trimesh.Trimesh(vertices, faces, process=True)
    watertight = bool(topology.is_watertight)
    winding = bool(topology.is_winding_consistent)
    signed_volume = float(topology.volume) if watertight and winding else None
    return {
        **result,
        "valid": not problems,
        "problems": problems,
        "degenerate_faces": degenerate,
        "extents_native": extents.tolist(),
        "bounds_native": mesh.bounds.tolist(),
        "bbox_diagonal_native": diagonal,
        "surface_area_native_squared": float(areas.sum()),
        "watertight": watertight,
        "winding_consistent": winding,
        "components": len(topology.split(only_watertight=False)),
        "signed_volume_native_cubed": signed_volume,
        "volume_native_cubed": abs(signed_volume) if signed_volume is not None else None,
        "volume_method": "absolute algebraic signed triangle-volume sum; no Boolean union or overlap correction",
        "volume_limitations": "Overlaps and self-intersections are not checked; watertight components do not establish a valid union solid.",
    }


def sample_surface(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    """Area-weighted uniform samples with a private RNG and no global RNG changes."""
    rng = np.random.default_rng(seed)
    areas = np.asarray(mesh.area_faces)
    indices = rng.choice(len(areas), count, p=areas / areas.sum())
    uv = rng.random((count, 2))
    root = np.sqrt(uv[:, 0])
    weights = np.column_stack((1 - root, root * (1 - uv[:, 1]), root * uv[:, 1]))
    return np.einsum("ni,nij->nj", weights, mesh.triangles[indices])


def _apply(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _fit_transform(source: np.ndarray, target: np.ndarray, scale: bool) -> np.ndarray:
    """Least-squares proper rigid/similarity fit; reflections are never allowed."""
    x_mean, y_mean = source.mean(0), target.mean(0)
    x, y = source - x_mean, target - y_mean
    u, singular, vt = np.linalg.svd(x.T @ y)
    sign = np.ones(3)
    sign[-1] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ np.diag(sign) @ u.T
    factor = float(np.sum(singular * sign) / np.sum(x * x)) if scale else 1.0
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("Registration produced an invalid scale")
    result = np.eye(4)
    result[:3, :3] = factor * rotation
    result[:3, 3] = y_mean - factor * rotation @ x_mean
    return result


def _axis_rotations() -> list[np.ndarray]:
    result = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            matrix = np.eye(3)[:, permutation] @ np.diag(signs)
            if np.linalg.det(matrix) > 0:
                result.append(matrix)
    return result


def _pca_basis(points: np.ndarray) -> np.ndarray:
    _, vectors = np.linalg.eigh(np.cov(points.T))
    basis = vectors[:, ::-1].copy()
    if np.linalg.det(basis) < 0:
        basis[:, -1] *= -1
    return basis


def _distances(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return cKDTree(target).query(source, workers=1)[0], cKDTree(source).query(target, workers=1)[0]


def _cost(source: np.ndarray, target: np.ndarray) -> float:
    forward, backward = _distances(source, target)
    return float((np.mean(forward**2) + np.mean(backward**2)) / 2)


def register_surfaces(source: np.ndarray, target: np.ndarray, *, similarity: bool,
                      config: EvaluationConfig) -> tuple[np.ndarray, dict]:
    """Fixed multistart symmetric ICP, selected only on registration samples.

    There are 24 proper signed axis permutations and 24 corresponding PCA starts.
    Rank starts by symmetric squared distance, then refine a fixed number by ICP.
    Each update fits *both* nearest-neighbor directions with no trimming.
    """
    source_center, target_center = source.mean(0), target.mean(0)
    rms_source = np.sqrt(np.mean(np.sum((source - source_center)**2, axis=1)))
    rms_target = np.sqrt(np.mean(np.sum((target - target_center)**2, axis=1)))
    factor = float(rms_target / rms_source) if similarity else 1.0
    source_basis, target_basis = _pca_basis(source), _pca_basis(target)
    rotations = _axis_rotations()
    rotations += [target_basis @ r @ source_basis.T for r in _axis_rotations()]
    starts = []
    for index, rotation in enumerate(rotations):
        transform = np.eye(4)
        transform[:3, :3] = factor * rotation
        transform[:3, 3] = target_center - factor * rotation @ source_center
        starts.append((_cost(_apply(source, transform), target), index, transform))
    starts.sort(key=lambda item: (item[0], item[1]))
    target_tree = cKDTree(target)
    best_cost, best_transform, best_start, best_iteration = np.inf, None, None, None
    diagnostics = []
    for initial_cost, index, initial in starts[:config.icp_candidates]:
        transform, previous_cost = initial.copy(), initial_cost
        candidate_best = initial_cost
        for iteration in range(config.icp_iterations + 1):
            current = _apply(source, transform)
            cost = _cost(current, target)
            candidate_best = min(candidate_best, cost)
            if cost < best_cost:
                best_cost, best_transform = cost, transform.copy()
                best_start, best_iteration = index, iteration
            if iteration == config.icp_iterations:
                break
            if iteration and abs(previous_cost - cost) <= 1e-7 * max(previous_cost, 1e-16):
                break
            previous_cost = cost
            nearest_target = target_tree.query(current, workers=1)[1]
            nearest_source = cKDTree(current).query(target, workers=1)[1]
            delta = _fit_transform(
                np.concatenate((current, current[nearest_source])),
                np.concatenate((target[nearest_target], target)),
                scale=similarity,
            )
            transform = delta @ transform
        diagnostics.append({"start": index, "iterations": iteration, "best_cost_m2": candidate_best})
    return best_transform, {
        "algorithm": "proper_similarity_symmetric_multistart_icp" if similarity else "proper_rigid_symmetric_multistart_icp",
        "initial_rotations": len(rotations),
        "refined_starts": len(diagnostics),
        "selected_start": best_start,
        "selected_iteration": best_iteration,
        "registration_chamfer_l2_m2": best_cost,
        "starts": diagnostics,
        "limitations": "Finite multistart local search; no guarantee of a global optimum. No reflections.",
    }


def surface_metrics(source: np.ndarray, target: np.ndarray,
                    thresholds_m: tuple[float, ...]) -> dict:
    forward, backward = _distances(source, target)
    f_scores = []
    for threshold in thresholds_m:
        precision, recall = float(np.mean(forward <= threshold)), float(np.mean(backward <= threshold))
        f_scores.append({
            "threshold_m": threshold,
            "precision": precision,
            "recall": recall,
            "f_score": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        })
    return {
        "chamfer_l1_m": float((forward.mean() + backward.mean()) / 2),
        "chamfer_l2_m2": float((np.mean(forward**2) + np.mean(backward**2)) / 2),
        "generated_to_gt_mean_m": float(forward.mean()),
        "gt_to_generated_mean_m": float(backward.mean()),
        "symmetric_distance_p95_m": float(np.percentile(np.concatenate((forward, backward)), 95)),
        "f_scores": f_scores,
    }


def _file_record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def evaluate_meshes(generated_path: str | Path, ground_truth_path: str | Path, *,
                    ground_truth_unit_in_meters: float,
                    generated_unit_in_meters: float | None = None,
                    config: EvaluationConfig | None = None) -> dict:
    """Return a JSON-safe success or failure record; failed meshes are not dropped.

    Mandatory GT conversion is meters per file coordinate unit (e.g. 0.001 for
    millimeters). A generated conversion must come from generation metadata, never
    from reference geometry. Unknown generated units permit only similarity scores.
    """
    config = config or EvaluationConfig()
    report = {
        "schema_version": 1,
        "status": "failed",
        "config": asdict(config),
        "ground_truth_unit_in_meters": ground_truth_unit_in_meters,
        "generated_unit_in_meters": generated_unit_in_meters,
        "distance_convention": "Chamfer = 0.5 * (mean generated-to-GT + mean GT-to-generated); L1 unsquared meters, L2 squared meters.",
        "sampling": "Independent uniform area-weighted samples, separate registration and final-score seeds; all triangles and components retained.",
        "modes": {},
    }
    stage = "configuration"
    try:
        config.validate()
        for factor in (ground_truth_unit_in_meters, generated_unit_in_meters):
            if factor is not None and (not np.isfinite(factor) or factor <= 0):
                raise ValueError("Unit conversion factors must be positive finite numbers")
        if ground_truth_unit_in_meters is None:
            raise ValueError("Ground-truth unit conversion is required")
        stage = "load_generated"
        generated_path, ground_truth_path = Path(generated_path), Path(ground_truth_path)
        generated = _load_geometry(generated_path)
        report["generated"] = {**_file_record(generated_path), **geometry_summary(generated)}
        if not report["generated"]["valid"]:
            raise ValueError("Invalid generated geometry: " + ", ".join(report["generated"]["problems"]))
        stage = "load_ground_truth"
        truth = _load_geometry(ground_truth_path)
        report["ground_truth"] = {**_file_record(ground_truth_path), **geometry_summary(truth)}
        if not report["ground_truth"]["valid"]:
            raise ValueError("Invalid ground-truth geometry: " + ", ".join(report["ground_truth"]["problems"]))
        truth.apply_scale(ground_truth_unit_in_meters)
        gt_diagonal = float(np.linalg.norm(truth.extents))
        gt_volume = report["ground_truth"]["volume_native_cubed"]
        gt_volume = gt_volume * ground_truth_unit_in_meters**3 if gt_volume is not None else None
        report["ground_truth"].update(extents_m=truth.extents.tolist(), bbox_diagonal_m=gt_diagonal, volume_m3=gt_volume)
        if generated_unit_in_meters is not None:
            generated.apply_scale(generated_unit_in_meters)
            report["generated"]["extents_m_before_alignment"] = generated.extents.tolist()
            raw_volume = report["generated"]["volume_native_cubed"]
            report["generated"]["volume_m3_before_alignment"] = raw_volume * generated_unit_in_meters**3 if raw_volume is not None else None
        # Random streams do not depend on the selected registration mode.
        p_reg = sample_surface(generated, config.registration_samples, config.seed)
        q_reg = sample_surface(truth, config.registration_samples, config.seed + 1)
        p_score = sample_surface(generated, config.surface_samples, config.seed + 2)
        q_score = sample_surface(truth, config.surface_samples, config.seed + 3)
        modes = ["similarity"] + (["rigid"] if generated_unit_in_meters is not None else [])
        if generated_unit_in_meters is None:
            report["rigid_unavailable_reason"] = "Generated metric units are unknown; assigning units from GT would hide scale error."
        for mode in modes:
            stage = "registration_" + mode
            transform, registration = register_surfaces(p_reg, q_reg, similarity=mode == "similarity", config=config)
            stage = "score_" + mode
            scores = surface_metrics(_apply(p_score, transform), q_score, config.thresholds_m)
            aligned = generated.copy()
            aligned.apply_transform(transform)
            scale = float(np.cbrt(np.linalg.det(transform[:3, :3])))
            native_to_meters = scale * (generated_unit_in_meters or 1.0)
            native_transform = transform.copy()
            native_transform[:3, :3] *= generated_unit_in_meters or 1.0
            volume = report["generated"]["volume_native_cubed"]
            volume = volume * native_to_meters**3 if volume is not None else None
            report["modes"][mode] = {
                **scores,
                "chamfer_l1_percent_gt_bbox_diagonal": scores["chamfer_l1_m"] / gt_diagonal * 100,
                "alignment_uses_ground_truth": True,
                "transform_native_generated_to_gt_meters": native_transform.tolist(),
                "fitted_scale_on_input_coordinates": scale,
                "native_generated_units_to_meters_after_alignment": native_to_meters,
                "aligned_extents_m": aligned.extents.tolist(),
                "extent_ratio_to_gt": (aligned.extents / np.maximum(truth.extents, 1e-15)).tolist(),
                "aligned_volume_m3": volume,
                "volume_ratio_to_gt": volume / gt_volume if volume is not None and gt_volume else None,
                "registration": registration,
            }
        report["status"] = "ok"
    # Evaluation is a batch boundary: malformed assets must yield a failed row,
    # while KeyboardInterrupt/SystemExit still propagate as normal.
    except Exception as exc:
        report["error"] = {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
    return _json_safe(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generated", type=Path)
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("--ground-truth-unit-in-meters", type=float, required=True)
    parser.add_argument("--generated-unit-in-meters", type=float)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--registration-samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_meshes(
        args.generated, args.ground_truth,
        ground_truth_unit_in_meters=args.ground_truth_unit_in_meters,
        generated_unit_in_meters=args.generated_unit_in_meters,
        config=EvaluationConfig(surface_samples=args.samples, registration_samples=args.registration_samples, seed=args.seed),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    raise SystemExit(0 if result["status"] == "ok" else 2)


if __name__ == "__main__":
    main()
