"""Read artwork orientation from frozen mesh UVs; never rotate the atlas or UVs."""
import numpy as np
from scipy.spatial import cKDTree
from .mesh import load_scene, parts

_BASIS = {
    'front -Y': ((1, 0, 0), (0, 0, 1)), 'back +Y': ((-1, 0, 0), (0, 0, 1)),
    'left -X': ((0, -1, 0), (0, 0, 1)), 'right +X': ((0, 1, 0), (0, 0, 1)),
    'top +Z': ((1, 0, 0), (0, 1, 0)), 'bottom -Z': ((1, 0, 0), (0, -1, 0)),
}


def _direction(vector):
    length = np.linalg.norm(vector)
    if length < 1e-10:
        return 'undefined'
    unit = vector / length
    if np.max(np.abs(unit)) < .85:
        return 'diagonal'
    return ('RIGHT' if unit[0] > 0 else 'LEFT') if abs(unit[0]) > abs(unit[1]) else ('DOWN' if unit[1] > 0 else 'UP')


def artwork_orientation(prepared, record):
    """Return local planar fits; curved islands still require actual-mesh review.

    Surface directions are expressed in Blender Z-up coordinates. glTF Y-up
    exports are converted back in memory solely for this measurement.
    Polygon membership avoids mistaking overlapping island bounding boxes.
    """
    up = record['up_axis_in_export']
    if up not in ('Y', 'Z'):
        raise ValueError('Surface export axis must be Y or Z')
    transform = np.eye(3) if up == 'Z' else np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    meshes = [(m, np.asarray(m.vertices) @ transform.T) for m in parts(load_scene(prepared))]
    result = []
    for island in record['islands']:
        polygons = island.get('polygons', [])
        if not polygons or island['direction'] not in _BASIS:
            continue
        tree = cKDTree(np.concatenate([np.asarray(p) for p in polygons]))
        lo, hi = np.asarray(island['uv_bounds'])
        uv_rows, xyz_rows = [], []
        for mesh, vertices in meshes:
            uv = np.asarray(mesh.visual.uv)
            faces_uv = uv[mesh.faces]
            centers = faces_uv.mean(axis=1)
            possible = np.flatnonzero(np.all((centers >= lo - 2e-6) & (centers <= hi + 2e-6), axis=1))
            if not len(possible):
                continue
            distances = tree.query(faces_uv[possible].reshape(-1, 2))[0].reshape(-1, 3)
            indices = mesh.faces[possible[np.all(distances < 2e-6, axis=1)]].ravel()
            if len(indices):
                uv_rows.append(uv[indices]); xyz_rows.append(vertices[indices])
        if not uv_rows:
            continue
        uv, xyz = np.concatenate(uv_rows), np.concatenate(xyz_rows)
        image_uv = np.column_stack((uv[:, 0], 1 - uv[:, 1], np.ones(len(uv))))
        matrix, _, rank, _ = np.linalg.lstsq(image_uv, xyz, rcond=None)
        if rank < 3:
            continue
        right, top = _BASIS[island['direction']]
        top_uv = np.linalg.lstsq(matrix[:2].T, top, rcond=None)[0]
        right_uv = np.linalg.lstsq(matrix[:2].T, right, rcond=None)[0]
        scale = max(float(np.linalg.norm(np.ptp(xyz, axis=0))), 1e-12)
        residual = float(np.sqrt(np.mean((image_uv @ matrix - xyz) ** 2)) / scale)
        result.append({'id': island['id'], 'artwork_top_toward_image': _direction(top_uv),
                       'artwork_right_toward_image': _direction(right_uv),
                       'planar_fit_relative_rms': round(residual, 6),
                       'scope': 'Local fit in TOP-LEFT image coordinates; verify curved surfaces and resulting mesh'})
    return result
