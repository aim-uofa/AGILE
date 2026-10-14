import copy
import itertools
import threading

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree as KDTree
import copy

from tqdm import tqdm

_RNG_LOCK = threading.Lock()


class RegistrationError(RuntimeError):
    pass


def calculate_metrics(aligned_mesh, target_mesh, is_sqrt=False):
    vertices_source = np.asarray(aligned_mesh.vertices) * 100
    vertices_target = np.asarray(target_mesh.vertices) * 100

    # dist_bidirectional = chamferDist(vertices_source, vertices_target, bidirectional=True,point_reduction = "mean") #* 0.001
    gen_points_kd_tree = KDTree(vertices_source)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(vertices_target)

    if is_sqrt:  # square-root chamfer
        gt_to_gen_chamfer = np.mean(one_distances)
    else:  # squared chamfer
        gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(vertices_target)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(vertices_source)

    if is_sqrt:
        gen_to_gt_chamfer = np.mean(two_distances)
    else:
        gen_to_gt_chamfer = np.mean(np.square(two_distances))

    chamfer_obj = gt_to_gen_chamfer + gen_to_gt_chamfer
    threshold = 0.5  # 5 mm
    precision_1 = np.mean(one_distances < threshold).astype(np.float32)
    precision_2 = np.mean(two_distances < threshold).astype(np.float32)
    fscore_obj_5 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)

    threshold = 1.0  # 10 mm
    precision_1 = np.mean(one_distances < threshold).astype(np.float32)
    precision_2 = np.mean(two_distances < threshold).astype(np.float32)
    fscore_obj_10 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)
    return chamfer_obj, fscore_obj_5, fscore_obj_10


def preprocess_point_cloud(pcd, voxel_size):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd_down, pcd_fpfh


def execute_global_registration(
    source_down, target_down, source_fpfh, target_fpfh, voxel_size
):
    distance_threshold = 0.01  # 1#voxel_size * 1.5
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return result


def refine_registration(
    source, target, source_fpfh, target_fpfh, voxel_size, init_alignment
):
    distance_threshold = 0.01  # voxel_size * 0.4
    # print(":: Point-to-plane ICP registration is applied on original point")
    # print("   clouds to refine the alignment. This time we use a strict")
    # print("   distance threshold %.3f." % distance_threshold)

    # result_ransac.transformation
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        distance_threshold,
        init_alignment,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(
            with_scaling=True
        ),
    )
    return result


def compute_icp_metrics(target_mesh, source_mesh, num_iters=60, no_tqdm=False,
                        is_sqrt=False, seed=1, return_details=False):
    """Deterministic Sim(3) registration; score the same full mesh vertices.

    Radius normalization makes rigid FPFH/RANSAC usable for unequal scales.
    Proper PCA orientation candidates cover cases where rigid RANSAC cannot
    initialize. Registration never mutates caller meshes or returns stale state.
    Open3D uses a global RNG, so its seeded operations are serialized in threads.
    """
    if num_iters < 0:
        raise ValueError('num_iters must be nonnegative')
    with _RNG_LOCK:
        o3d.utility.random.seed(seed)
        return _compute_icp_metrics(target_mesh, source_mesh, num_iters, no_tqdm,
                                    is_sqrt, seed, return_details)


def _compute_icp_metrics(target_mesh, source_mesh, num_iters, no_tqdm, is_sqrt,
                         seed, return_details):
    source_mesh, target_mesh = copy.deepcopy(source_mesh), copy.deepcopy(target_mesh)
    source_vertices = np.asarray(source_mesh.vertices)
    target_vertices = np.asarray(target_mesh.vertices)
    for label, vertices, mesh in [('source', source_vertices, source_mesh),
                                   ('target', target_vertices, target_mesh)]:
        if len(vertices) < 4 or not len(mesh.triangles) or not np.isfinite(vertices).all():
            raise RegistrationError(label + ' mesh is empty, too small, or non-finite')
        if np.linalg.matrix_rank(vertices - vertices.mean(0), tol=1e-10) < 2:
            raise RegistrationError(label + ' mesh is degenerate')
        if not np.isfinite(mesh.get_surface_area()) or mesh.get_surface_area() <= 0:
            raise RegistrationError(label + ' mesh has no usable triangle surface')
    source_center, target_center = source_vertices.mean(0), target_vertices.mean(0)
    va, vb = source_vertices - source_center, target_vertices - target_center
    source_mesh.vertices = o3d.utility.Vector3dVector(va)
    target_mesh.vertices = o3d.utility.Vector3dVector(vb)
    radius_scale = float(np.sqrt(np.mean(np.sum(vb * vb, axis=1)) /
                                 np.mean(np.sum(va * va, axis=1))))
    if not np.isfinite(radius_scale) or radius_scale <= 0:
        raise RegistrationError('Invalid radius scale')
    # Keep NAS surface-sampling budget and final CD/F-score formula.
    target = target_mesh.sample_points_uniformly(1000)
    source = source_mesh.sample_points_uniformly(1000)
    scaled_source = copy.deepcopy(source)
    scale_tf = np.eye(4)
    scale_tf[:3, :3] *= radius_scale
    scaled_source.transform(scale_tf)
    _, _, aa = np.linalg.svd(va, full_matrices=False)
    _, _, bb = np.linalg.svd(vb, full_matrices=False)
    candidates = [('identity', np.eye(4)), ('radius', scale_tf)]
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([-1, 1], repeat=3):
            p = np.eye(3)[list(perm)] * np.asarray(signs)[:, None]
            r = bb.T @ p @ aa
            if np.linalg.det(r) < 0:
                continue
            initial = np.eye(4)
            initial[:3, :3] = radius_scale * r
            candidates.append(('radius-pca', initial))
    errors, rejected, attempted = [], 0, 0
    best = None
    target_tree = KDTree(vb * 100)

    def score(tf):
        aligned = va @ tf[:3, :3].T + tf[:3, 3]
        d1 = target_tree.query(aligned * 100)[0]
        d2 = KDTree(aligned * 100).query(vb * 100)[0]
        cd = (d1.mean() + d2.mean()) if is_sqrt else (np.square(d1).mean() + np.square(d2).mean())
        def f(threshold):
            p, q = np.mean(d1 < threshold), np.mean(d2 < threshold)
            return 2 * p * q / (p + q + 1e-7)
        return float(cd), float(f(.5)), float(f(1.))

    def try_candidate(label, initial):
        nonlocal best, attempted, rejected
        attempted += 1
        try:
            coarse = o3d.pipelines.registration.registration_icp(
                source, target, .03, initial,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
            fine = o3d.pipelines.registration.registration_icp(
                source, target, .01, coarse.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
            tf = fine.transformation
            det = np.linalg.det(tf[:3, :3])
            scale = np.cbrt(det)
            # Reject collapsed fits and unusable correspondence sets explicitly.
            if (not np.isfinite(tf).all() or det <= 0 or
                    not radius_scale * .2 <= scale <= radius_scale * 5 or fine.fitness <= 0):
                rejected += 1
                return
            cd, f5, f10 = score(tf)
            if not np.isfinite([cd, f5, f10]).all():
                raise RegistrationError('Non-finite registration metric')
            if best is None or cd < best['cd']:
                best = {'cd': cd, 'f5': f5, 'f10': f10, 'transformation': tf.copy(),
                        'scale': float(scale), 'candidate': label,
                        'fitness': float(fine.fitness), 'inlier_rmse': float(fine.inlier_rmse)}
            initial_cd, initial_f5, initial_f10 = score(initial)
            if best is None or initial_cd < best['cd']:
                initial_scale = np.cbrt(np.linalg.det(initial[:3, :3]))
                if radius_scale * .2 <= initial_scale <= radius_scale * 5:
                    best = {'cd': initial_cd, 'f5': initial_f5, 'f10': initial_f10,
                            'transformation': initial.copy(), 'scale': float(initial_scale),
                            'candidate': label + '-initial', 'fitness': float(fine.fitness),
                            'inlier_rmse': float(fine.inlier_rmse)}
        except (RuntimeError, ValueError) as exc:
            errors.append(type(exc).__name__ + ': ' + str(exc))

    for label, initial in candidates:
        try_candidate(label, initial)
    voxel_size = .005
    try:
        source_down, source_fpfh = preprocess_point_cloud(scaled_source, voxel_size)
        target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
        if len(source_down.points) < 4 or len(target_down.points) < 4:
            raise RegistrationError('Insufficient downsampled points for FPFH')
        iterations = range(num_iters) if no_tqdm else tqdm(range(num_iters), desc='scale-normalized RANSAC')
        for _ in iterations:
            try:
                global_fit = execute_global_registration(
                    source_down, target_down, source_fpfh, target_fpfh, voxel_size)
                try_candidate('scale-normalized-ransac', global_fit.transformation @ scale_tf)
            except (RuntimeError, ValueError) as exc:
                errors.append(type(exc).__name__ + ': ' + str(exc))
    except (RuntimeError, ValueError) as exc:
        errors.append(type(exc).__name__ + ': ' + str(exc))
    if best is None:
        raise RegistrationError('All registration candidates failed (%s attempted, %s rejected): %s' %
                                (attempted, rejected, '; '.join(errors[:5])))
    full_tf = best['transformation'].copy()
    full_tf[:3, 3] += target_center - full_tf[:3, :3] @ source_center
    details = {'seed': seed, 'radius_initial_scale': radius_scale,
               'fitted_scale': best['scale'], 'best_candidate': best['candidate'],
               'fitness': best['fitness'], 'inlier_rmse_m': best['inlier_rmse'],
               'attempted_candidates': attempted, 'rejected_candidates': rejected,
               'failed_candidate_count': len(errors), 'candidate_errors': errors[:10],
               'centered_transformation': best['transformation'].tolist(),
               'transformation': full_tf.tolist(), 'ransac_iterations': num_iters,
               'scale_ratio_bounds_to_radius_initialization': [.2, 5.]}
    result = (best['cd'], best['f5'], best['f10'])
    return (*result, details) if return_details else result
