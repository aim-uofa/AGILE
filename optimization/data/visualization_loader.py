"""Visualization adapter. Saved optimized vertices are the default source."""
from pathlib import Path
import numpy as np
from optimization.data.result_bundle import ResultBundle, _npz


def slice_rotations(rotations, indices, total, representation=None):
    r = np.asarray(rotations)
    if r.shape == (3,):
        return np.repeat(r[None], len(indices), axis=0)
    if r.shape == (3, 3) and (total != 3 or representation == 'shared-matrix'):
        return np.repeat(r[None], len(indices), axis=0)
    if r.shape == (3, 3) and total == 3 and representation is None:
        if np.allclose(r.T @ r, np.eye(3)) and np.isclose(np.linalg.det(r), 1):
            raise ValueError('Ambiguous 3x3 rotations: declare shared-matrix or per-frame-rotvec')
    if r.shape not in ((total, 3), (total, 3, 3)):
        raise ValueError('Rotation count does not match camera frames')
    return r[indices]


class VisualizationLoader(ResultBundle):
    def __init__(self, data_dir, downsample_factor=1, mano_model_path='assets/mano',
                 use_parametric_mano=False, start_frame=0, end_frame=-1,
                 object_mesh_path=None, max_frames=None, **kwargs):
        if object_mesh_path is not None:
            raise ValueError('Final-result visualization uses the exported mesh; create a separate bundle for comparisons')
        super().__init__(data_dir, start_frame, end_frame, max_frames, **kwargs)
        self.downsample_factor = downsample_factor
        self.use_parametric_mano = use_parametric_mano
        self.can_compute_parametric = False
        self.mano_model = None
        if use_parametric_mano:
            if self.manifest:
                raise ValueError('Portable bundles use saved final vertices; parametric reconstruction requires legacy parameters')
            import torch
            import yaml
            from optimization.models.mano_wrapper import MANO
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            cfg = yaml.safe_load((Path(mano_model_path) / 'mano_config.yaml').read_text())
            self.mano_model = MANO(**{k.lower(): v for k, v in cfg['MANO'].items()}).to(self.device)
            self._load_mano_data()

    def _load_mano_data(self):
        total = self._video.shape[0]
        for key in ('mano_betas', 'mano_hand_pose', 'mano_global_orient'):
            data = _npz(self.data_dir / (key + '.npz'))[key]
            if len(data) != total:
                raise ValueError('Parameter length mismatch: ' + key)
            setattr(self, key, data[self.indices])
        transforms = _npz(self.data_dir / 'mano_transforms.npz')
        self.mano_transforms = dict(scales=transforms['scales'],
            rotations=slice_rotations(transforms['rotations'], self.indices, total,
                                      str(transforms['rotation_representation']) if 'rotation_representation' in transforms else None),
            translations=transforms['translations'][self.indices])
        if len(transforms['translations']) != total:
            raise ValueError('Translation count mismatch')
        self.can_compute_parametric = self.mano_model is not None

    def _compute_mano_vertices_from_params(self, i):
        import torch
        from optimization.utils.geometry import rotation_vector_to_matrix_safe
        self._index(i)
        def tensor(x):
            return torch.as_tensor(x, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            output = self.mano_model(betas=tensor(self.mano_betas[i, 0])[None],
                                    hand_pose=tensor(self.mano_hand_pose[i, 0])[None], pose2rot=False)
            vertices = (output.vertices[0] - output.joints[0, 0]) @ tensor(self.mano_global_orient[i, 0, 0]).T
            rotation = tensor(self.mano_transforms['rotations'][i])
            if rotation.shape == (3,):
                rotation = rotation_vector_to_matrix_safe(rotation)
            vertices = tensor(self.mano_transforms['scales']) * vertices @ rotation.T
            vertices += tensor(self.mano_transforms['translations'][i])
        return vertices.cpu().numpy()

    def get_mano_vertices(self, i, world=False):
        if not self.use_parametric_mano:
            return super().get_mano_vertices(i, world=world)
        if not self.can_compute_parametric:
            raise ValueError('Requested parametric MANO is unavailable')
        v = self._compute_mano_vertices_from_params(i)
        return v @ self.cam_c2w[i, :3, :3].T + self.cam_c2w[i, :3, 3] if world else v

    def get_points(self, i, downsample_factor=None):
        return super().get_points(i, downsample_factor or self.downsample_factor)
