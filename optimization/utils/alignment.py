import numpy as np
import open3d as o3d


def align_with_shared_scale_rotation(A_seq, B_seq):
    """
    A_seq, B_seq: list of length T
        each element is (N_t, 3) numpy array
        (A_t and B_t must have the same N_t, but N_t can vary across t)

    return:
        s: global scale
        R: global rotation (3,3)
        t_seq: list of (3,) translations for each frame
    """
    assert len(A_seq) == len(B_seq)
    T = len(A_seq)

    A_centered_all = []
    B_centered_all = []
    mu_A_list = []
    mu_B_list = []
    total_points = 0

    for t in range(T):
        A_t = A_seq[t]
        B_t = B_seq[t]
        assert A_t.shape == B_t.shape

        mu_A_t = A_t.mean(axis=0)
        mu_B_t = B_t.mean(axis=0)
        mu_A_list.append(mu_A_t)
        mu_B_list.append(mu_B_t)

        A_centered_all.append(A_t - mu_A_t)
        B_centered_all.append(B_t - mu_B_t)

        total_points += A_t.shape[0]

    A_centered = np.concatenate(A_centered_all, axis=0)
    B_centered = np.concatenate(B_centered_all, axis=0)

    # 2. Estimate shared R, s
    cov = (B_centered.T @ A_centered) / total_points
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt

    var_A = (A_centered**2).sum() / total_points
    s = np.trace(np.diag(D) @ S) / var_A

    # 3. Per-frame translation
    t_seq = []
    for mu_A_t, mu_B_t in zip(mu_A_list, mu_B_list):
        t_seq.append(mu_B_t - s * (R @ mu_A_t))
    t_seq = np.stack(t_seq)

    return s, R, t_seq

def umeyama_weighted(A, B, weights=None):
    """
    Umeyama algorithm with weighting support for estimating global similarity transform (s, R)
    A, B: (N,3) numpy arrays (already centered)
    weights: (N,) weights, defaults to uniform
    """
    if weights is None:
        weights = np.ones(len(A))
    weights = weights / weights.sum()

    # Weighted mean
    mu_A = (A * weights[:, None]).sum(axis=0)
    mu_B = (B * weights[:, None]).sum(axis=0)

    A_c = A - mu_A
    B_c = B - mu_B

    C = (B_c.T @ (A_c * weights[:, None]))

    # SVD
    U, D, Vt = np.linalg.svd(C)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt

    # Scaling
    var_A = (weights * (A_c**2).sum(axis=1)).sum()
    s = np.trace(np.diag(D) @ S) / var_A

    return s, R


def multi_frame_align_open3d(A_seq, B_seq, max_iters=5, trim_ratio=0.8, alpha=0.0):
    'Align paired MANO and depth point clouds across time. Return a shared scale and rotation, and per-frame translations. Alpha controls translation smoothing.'
    assert len(A_seq) == len(B_seq)
    T = len(A_seq)


    s, R = 1.0, np.eye(3)
    t_seq = [np.zeros(3) for _ in range(T)]

    for it in range(max_iters):
        A_aligned_list, B_trimmed_list, frame_weights = [], [], []

        for t in range(T):
            A_t, B_t = A_seq[t], B_seq[t]

            # transform current frame's A
            A_t_aligned = s * (A_t @ R.T) + t_seq[t]


            pcA = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(A_t_aligned))
            pcB = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(B_t))


            pcA_tree = o3d.geometry.KDTreeFlann(pcA)
            matches = []
            for p in np.asarray(pcB.points):
                _, idx, dist = pcA_tree.search_knn_vector_3d(p, 1)
                matches.append((idx[0], p, dist[0]))

            matches.sort(key=lambda x: x[2])
            keep_n = int(len(matches) * trim_ratio)
            matches = matches[:keep_n]

            A_sel = np.asarray(pcA.points)[[m[0] for m in matches]]
            B_sel = np.array([m[1] for m in matches])


            if len(A_sel) > 3:
                A_aligned_list.append(A_sel)
                B_trimmed_list.append(B_sel)
                frame_weights.extend([len(A_sel)] * len(A_sel))

        if not A_aligned_list:
            break


        A_centered_all, B_centered_all = [], []
        for A_sel, B_sel in zip(A_aligned_list, B_trimmed_list):
            ma, mb = A_sel.mean(axis=0), B_sel.mean(axis=0)
            A_centered_all.append(A_sel - ma)
            B_centered_all.append(B_sel - mb)

        A_all = np.vstack(A_centered_all)
        B_all = np.vstack(B_centered_all)
        weights = np.repeat(frame_weights, 1)[:len(A_all)]


        s, R = umeyama_weighted(A_all, B_all, weights)


        t_seq_new = []
        prev_t = np.zeros(3)
        for i, (A_t, B_t) in enumerate(zip(A_seq, B_seq)):
            ma, mb = A_t.mean(axis=0), B_t.mean(axis=0)
            t_raw = mb - s * (R @ ma)

            if i == 0 or alpha == 0.0:
                t_final = t_raw
            else:

                t_final = (1 - alpha) * t_raw + alpha * prev_t

            t_seq_new.append(t_final)
            prev_t = t_final

        t_seq = t_seq_new

    return s, R, np.array(t_seq)

def multi_frame_align_open3d_with_normals(A_seq, B_seq, A_normals_seq, B_normals_seq,
                             max_iters=5, trim_ratio=0.8, alpha=0.0,
                             normal_threshold=0.5, normal_lambda=0.5):
    'Align MANO vertices and depth point clouds with their corresponding normals across time.'
    assert len(A_seq) == len(B_seq) == len(A_normals_seq) == len(B_normals_seq)
    T = len(A_seq)

    s, R = 1.0, np.eye(3)
    t_seq = [np.zeros(3) for _ in range(T)]

    for it in range(max_iters):
        A_aligned_list, B_trimmed_list, frame_weights = [], [], []

        for t in range(T):
            A_t, B_t = A_seq[t], B_seq[t]
            nA_t, nB_t = A_normals_seq[t], B_normals_seq[t]

            # transform current frame's A
            A_t_aligned = s * (A_t @ R.T) + t_seq[t]
            nA_t_aligned = (nA_t @ R.T)

            # KDTree on B
            pcdB = o3d.geometry.PointCloud()
            pcdB.points = o3d.utility.Vector3dVector(B_t)
            pcB_tree = o3d.geometry.KDTreeFlann(pcdB)  # CPU KDTree
            # pcB_tree = o3d.geometry.KDTreeFlann(o3d.utility.Vector3dVector(B_t)) # CUDA KDTree

            matches = []
            for iA, (pA, nA) in enumerate(zip(A_t_aligned, nA_t_aligned)):

                pA = np.asarray(pA).astype(np.float64).reshape(3,)
                if not np.all(np.isfinite(pA)):
                    raise ValueError(f"Invalid query point pA: {pA}")

                _, idx, dist = pcB_tree.search_knn_vector_3d(pA, 1)
                j = idx[0]
                pB, nB = B_t[j], nB_t[j]

                # Normal consistency (cos θ)
                cos_theta = np.dot(nA, nB) / (np.linalg.norm(nA)*np.linalg.norm(nB)+1e-8)
                if cos_theta < normal_threshold:   # Discard matches with inconsistent normals
                    continue

                # Weighted distance (optional)
                weight = 1.0 - normal_lambda * cos_theta
                matches.append((iA, j, dist * weight))

            matches.sort(key=lambda x: x[2])
            keep_n = int(len(matches) * trim_ratio)
            matches = matches[:keep_n]

            if matches:
                A_sel = A_t_aligned[[m[0] for m in matches]]
                B_sel = B_t[[m[1] for m in matches]]
                A_aligned_list.append(A_sel)
                B_trimmed_list.append(B_sel)
                frame_weights.extend([1] * len(A_sel))

        if not A_aligned_list:
            break

        # Concatenate
        A_all = np.vstack(A_aligned_list)
        B_all = np.vstack(B_trimmed_list)

        # Update s, R
        s, R = umeyama_weighted(A_all, B_all, np.array(frame_weights))

        # Update per-frame translation (with smoothing)
        t_seq_new = []
        prev_t = np.zeros(3)
        for i, (A_t, B_t) in enumerate(zip(A_seq, B_seq)):
            ma, mb = A_t.mean(axis=0), B_t.mean(axis=0)
            t_raw = mb - s * (R @ ma)

            if i == 0 or alpha == 0.0:
                t_final = t_raw
            else:
                t_final = (1 - alpha) * t_raw + alpha * prev_t

            t_seq_new.append(t_final)
            prev_t = t_final

        t_seq = t_seq_new

    return s, R, np.array(t_seq)

def compute_mano_normals(mano_verts, mano_faces):
    """
    Compute MANO hand mesh vertex normals.

    Args:
        mano_verts: (N,3) numpy array of vertices
        mano_faces: (M,3) numpy array of face indices

    Returns:
        verts: (N,3) vertex coordinates
        normals: (N,3) normalized vertex normals
    """
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(mano_verts)
    mesh.triangles = o3d.utility.Vector3iVector(mano_faces)

    # Estimate vertex normals
    mesh.compute_vertex_normals()
    verts = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)

    return verts, normals, mesh
import torch
from sklearn.neighbors import NearestNeighbors
