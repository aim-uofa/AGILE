"""Unsigned point-to-triangle distance for static canonical object surfaces."""
import numpy as np


class SurfaceDistance:
    def __init__(self, vertices, faces):
        import open3d as o3d
        self.o3d = o3d
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=1)
        self.scene.add_triangles(
            o3d.core.Tensor(np.asarray(vertices, dtype=np.float32)),
            o3d.core.Tensor(np.asarray(faces, dtype=np.uint32)))

    def millimetres(self, hand_vertices_in_camera, object_pose):
        # The object's optimized scale is already in the canonical vertices.
        # Inverse rigid transformation avoids rebuilding the acceleration tree.
        pose = np.asarray(object_pose)
        points = (np.asarray(hand_vertices_in_camera) - pose[:3, 3]) @ pose[:3, :3]
        values = self.scene.compute_distance(
            self.o3d.core.Tensor(points.astype(np.float32)), nthreads=1).numpy() * 1000
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('Invalid point-to-surface distance')
        return values


def summarize_distance(distances):
    return dict(minimum_mm=float(np.min(distances)), percentile_05_mm=float(np.percentile(distances, 5)),
                vertices_within_5mm=int(np.count_nonzero(distances <= 5)), hand_vertices=len(distances))
