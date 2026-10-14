"""Portable, versioned optimization results and a bounded-memory legacy adapter.

All geometry returned by default is in OpenCV camera coordinates, in metres.
Object scale is applied once to the canonical mesh. Array positions, original
frame IDs and source timestamps are deliberately separate.
"""
from pathlib import Path
import hashlib
import json
import shutil
import threading
import zipfile
import copy

import numpy as np
from PIL import Image
import trimesh


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def texture_uint8(values):
    """Encode a texture without exposure adjustment or uint8 wraparound."""
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError('Texture contains non-finite values')
    if values.dtype == np.uint8:
        return values.copy()
    if values.min() < 0 or values.max() > 1:
        raise ValueError('Floating texture values must be normalized to [0, 1]')
    return np.rint(values * 255).astype(np.uint8)


class ZipFrames:
    """Read an NPZ video one frame at a time without materializing the video."""
    def __init__(self, path, key):
        self.zip = zipfile.ZipFile(path)
        self.stream = self.zip.open(key + '.npy')
        version = np.lib.format.read_magic(self.stream)
        self.shape, fortran, self.dtype = np.lib.format._read_array_header(self.stream, version)
        if fortran or self.dtype.hasobject:
            self.close()
            raise ValueError('RGBD must be a numeric C-order array')
        self.offset = self.stream.tell()
        self.frame_bytes = int(np.prod(self.shape[1:])) * self.dtype.itemsize
        self.lock = threading.Lock()
        self.cached_index = None
        self.cached_frame = None

    def __getitem__(self, index):
        if not 0 <= int(index) < self.shape[0]:
            raise IndexError(index)
        with self.lock:
            if self.cached_index == int(index):
                return self.cached_frame.copy()
            self.stream.seek(self.offset + int(index) * self.frame_bytes)
            block = self.stream.read(self.frame_bytes)
            if len(block) != self.frame_bytes:
                raise ValueError('Truncated RGBD array')
            self.cached_index = int(index)
            self.cached_frame = np.frombuffer(block, dtype=self.dtype).reshape(self.shape[1:]).copy()
            return self.cached_frame.copy()

    def close(self):
        self.stream.close()
        self.zip.close()

    def __del__(self):
        if hasattr(self, 'stream'):
            self.close()


class RGBDFrames:
    def __init__(self, bundle):
        self.bundle = bundle
        self.shape = (bundle.num_frames, bundle.height, bundle.width, 4)

    def __getitem__(self, key):
        first, rest = (key[0], key[1:]) if isinstance(key, tuple) else (key, ())
        if isinstance(first, slice):
            result = np.stack([self[i] for i in range(*first.indices(self.shape[0]))])
            return result[(slice(None), *rest)] if rest else result
        result = np.concatenate((self.bundle.get_rgb(first).astype(np.float32),
                                 self.bundle.get_depth(first)[..., None]), axis=-1)
        return result[rest] if rest else result


def _npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _rigid_valid(poses):
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError('Poses must have shape (N, 4, 4)')
    r = poses[:, :3, :3]
    return (np.isfinite(poses).all(axis=(1, 2)) &
            np.all(np.isclose(poses[:, 3], [0, 0, 0, 1], atol=1e-6), axis=1) &
            np.isclose(np.linalg.det(r), 1., atol=1e-3) &
            np.all(np.isclose(r.transpose(0, 2, 1) @ r, np.eye(3), atol=1e-3), axis=(1, 2)))


class ResultBundle:
    def __init__(self, data_dir, start_frame=0, end_frame=-1, max_frames=None,
                 source_fps=None, source_fps_provenance=None, object_variant='display', **unused):
        self.data_dir = self.aligned_dir = Path(data_dir).resolve()
        self._video = None
        self.manifest = None
        self.config = {}
        self.notices = []
        self.object_assets = None
        self.object_variant = object_variant
        manifest_path = self.data_dir / 'result.json'
        if manifest_path.exists():
            self.manifest = json.loads(manifest_path.read_text())
            m = self.manifest
            if m.get('schema_version') not in (1, 2) or m.get('coordinates') != 'opencv-camera-metres':
                raise ValueError('Unsupported result bundle schema or coordinates')
            if m.get('object_scale_baked') is not True:
                raise ValueError('Bundle must declare baked object scale')
            arrays = _npz(self.asset(m['geometry']))
            frames = m['frames']
            total = len(frames)
            ids = [f.get('frame_id') for f in frames]
            self.frame_ids_source = m['frame_ids_source']
            times = [f.get('timestamp_seconds') for f in frames]
            self.time_source = m['time_source']
            self.notices = list(m.get('notices', []))
            self.source = m.get('source', {})
            self.K = arrays['intrinsic']
            self.height, self.width = m['image_size']
            mesh_path = self.asset(m['object_mesh']) if m.get('object_mesh') else None
            self.object_assets = m.get('object_assets')
            scale = 1.
            self.optimized_object_scale = m.get('optimized_object_scale')
        else:
            camera_path = self.data_dir / 'camera_data.npz'
            with np.load(camera_path, allow_pickle=False) as z:
                camera = {k: z[k] for k in z.files if k != 'rgbd'}
                has_rgbd = 'rgbd' in z.files
            if not has_rgbd:
                raise ValueError('Legacy output has no RGBD; supply a complete result export')
            self._video = ZipFrames(camera_path, 'rgbd')
            total, self.height, self.width, channels = self._video.shape
            if channels != 4:
                raise ValueError('Expected RGBD channels')
            self.K = camera['intrinsic']
            config_path = self.data_dir.parent / 'config.json'
            if config_path.exists():
                self.config = json.loads(config_path.read_text())
            if 'prediction_frame_ids' in camera:
                ids = camera['prediction_frame_ids'].tolist()
                self.frame_ids_source = 'saved-prediction-frame-ids'
            elif 'start_frame' in self.config:
                c = self.config
                step = c.get('hold_mode_interval', 5) if c.get('hold_mode') else 1
                first = c['start_frame']
                if not c.get('test_first_frame') and (c.get('bidirectional_optimization') or c.get('reverse_frame_order')):
                    first = 0
                ids = (first + np.arange(total) * step).tolist()
                self.frame_ids_source = 'inferred-from-saved-config'
                self.notices.append('Original frame IDs inferred from the saved configuration.')
            else:
                ids = [None] * total
                self.frame_ids_source = 'unknown'
            times = camera.get('timestamps_seconds', [None] * total)
            self.time_source = str(camera.get('timestamp_source', 'unknown'))
            arrays = {'cam_c2w': camera.get('cam_c2w', np.tile(np.eye(4), (total, 1, 1)))}
            for filename, key in [('optimized_mano_verts', 'mano_verts'), ('optimized_mano_joints', 'mano_joints'), ('mano_faces', 'mano_faces')]:
                path = self.data_dir / (filename + '.npz')
                if path.exists():
                    arrays[key] = _npz(path)[key]
            pose_path = self.data_dir / 'object_poses.npz'
            poses = _npz(pose_path) if pose_path.exists() else {}
            arrays.update({k: v for k, v in poses.items() if k != 'object_scale'})
            scale = np.asarray(poses.get('object_scale', 1.))
            self.optimized_object_scale = scale.tolist()
            mesh_path = self.data_dir / 'object_mesh.obj'
            if not mesh_path.exists():
                mesh_path = None
            self.source = {'format': 'legacy-aligned', 'configuration': self.config,
                           'camera_members': self._video.zip.namelist()}
            export_metadata = self.data_dir / 'export_metadata.json'
            metadata = json.loads(export_metadata.read_text()) if export_metadata.exists() else {}
            self.source['export_metadata'] = metadata
            assets_path = self.data_dir / 'object_assets.json'
            if assets_path.exists():
                self.object_assets = json.loads(assets_path.read_text())
                if self.object_assets.get('source_stage') == 'optimization-input-unverified-stage':
                    self.notices.append('Full optimization input preserved; upstream final-asset provenance has not been registered.')
            if metadata.get('texture_encoding') == 'srgb-unmodified':
                pass
            elif self.config.get('optimize_object_texture'):
                self.notices.append('Legacy optimized texture may have been brightened/clipped; retained as saved.')
            else:
                self.notices.append('Legacy texture retained as saved; original color encoding is unverified.')
        if total < 1 or len(ids) != total or len(times) != total:
            raise ValueError('Empty result or inconsistent frame metadata')
        known_ids = all(i is not None for i in ids)
        if not known_ids and any(i is not None for i in ids):
            raise ValueError('Partial original frame IDs are ambiguous')
        if known_ids and (any(not isinstance(i, (int, np.integer)) for i in ids) or np.any(np.diff(ids) <= 0)):
            raise ValueError('Original frame IDs must be strictly increasing integers')
        if any(t is None for t in times) and any(t is not None for t in times):
            raise ValueError('Partial source timestamps are ambiguous')
        if any(t is None for t in times):
            if source_fps is not None:
                if not known_ids or not np.isfinite(source_fps) or source_fps <= 0 or not source_fps_provenance:
                    raise ValueError('FPS requires known IDs, a positive value and explicit provenance')
                times = np.asarray(ids) / float(source_fps)
                self.time_source = source_fps_provenance
            else:
                times = [None] * total
        elif not np.isfinite(times).all() or (total > 1 and np.any(np.diff(times) <= 0)):
            raise ValueError('Timestamps must be finite and strictly increasing')
        end = total if end_frame == -1 else min(end_frame, total)
        if max_frames is not None:
            if max_frames < 1:
                raise ValueError('max_frames must be positive')
            end = min(end, start_frame + max_frames)
        if not 0 <= start_frame < end <= total:
            raise ValueError('Invalid or empty frame range')
        self.start_frame, self.end_frame = start_frame, end
        self.indices = np.arange(start_frame, end)
        self.num_frames = len(self.indices)
        self.frame_ids = np.asarray(ids, dtype=object)[self.indices]
        self.timestamps = np.asarray(times, dtype=object)[self.indices]
        self.mano_verts = self.mano_joints = self.mano_faces = self.object_poses_in_cam = None
        self.cam_c2w = np.tile(np.eye(4), (self.num_frames, 1, 1))
        for key in ('mano_verts', 'mano_joints', 'object_poses_in_cam', 'cam_c2w'):
            if key in arrays:
                a = arrays[key]
                if len(a) != total or not np.isfinite(a).all():
                    raise ValueError('Invalid length/non-finite data in ' + key)
                setattr(self, key, a[self.indices])
        self.mano_faces = arrays.get('mano_faces')
        if self.mano_verts is not None:
            if self.mano_verts.ndim != 4 or self.mano_verts.shape[1] != 1 or self.mano_verts.shape[-1] != 3:
                raise ValueError('Single-hand vertices must have shape (N, 1, V, 3)')
            if self.mano_faces is not None and (self.mano_faces.ndim != 2 or self.mano_faces.shape[-1] != 3 or
                    not np.issubdtype(self.mano_faces.dtype, np.integer) or self.mano_faces.min() < 0 or
                    self.mano_faces.max() >= self.mano_verts.shape[2]):
                raise ValueError('Invalid hand triangle indices')
        if self.K.shape not in ((3, 3), (total, 3, 3)) or not np.isfinite(self.K).all():
            raise ValueError('Invalid camera intrinsics')
        if self.K.ndim == 3:
            self.K = self.K[self.indices]
        if np.any(self.K[..., 0, 0] <= 0) or np.any(self.K[..., 1, 1] <= 0):
            raise ValueError('Camera focal lengths must be positive')
        if not _rigid_valid(self.cam_c2w).all():
            raise ValueError('Camera poses must be rigid')
        self._mesh_scale = scale
        if self.object_assets:
            if self.object_assets.get('object_scale_baked') is not bool(self.manifest):
                raise ValueError('Object variants have inconsistent scale semantics')
            variants = self.object_assets['variants']
            if object_variant not in variants:
                raise ValueError('Unavailable object variant: ' + object_variant)
            mesh_path = self.asset(variants[object_variant]['mesh'])
        elif object_variant != 'display':
            raise ValueError('Legacy result has no separate optimization proxy')
        self.object_mesh = trimesh.load(mesh_path, force='mesh', process=False) if mesh_path else None
        if self.object_mesh is not None:
            if not np.isfinite(scale).all() or np.any(np.asarray(scale) <= 0) or np.asarray(scale).size not in (1, 3):
                raise ValueError('Invalid object scale')
            self.object_mesh.apply_scale(scale)
        self.valid_hand = (np.ptp(self.mano_verts, axis=-2).max(axis=(1, 2)) > 1e-7
                           if self.has_mano else np.zeros(self.num_frames, bool))
        self.valid_object = (_rigid_valid(self.object_poses_in_cam) if self.has_object
                             else np.zeros(self.num_frames, bool))
        if self.manifest:
            # An explicit invalid flag can hide a missing estimate even when its
            # placeholder transform happens to look like a valid rigid pose.
            self.valid_hand &= [bool(self.manifest['frames'][i].get('valid_hand', False)) for i in self.indices]
            self.valid_object &= [bool(self.manifest['frames'][i].get('valid_object', False)) for i in self.indices]
        self.rgbd = RGBDFrames(self)

    def load_object_variant(self, variant):
        """Return another saved variant in the same scaled canonical frame."""
        if variant == self.object_variant:
            return self.object_mesh.copy() if self.object_mesh is not None else None
        if not self.object_assets or variant not in self.object_assets['variants']:
            raise ValueError('Unavailable object variant: ' + variant)
        from optimization.data.object_assets import load_mesh
        mesh = load_mesh(self.asset(self.object_assets['variants'][variant]['mesh']))
        mesh.apply_scale(self._mesh_scale)
        return mesh

    def asset(self, relative):
        path = (self.data_dir / relative).resolve()
        if not path.is_relative_to(self.data_dir) or not path.is_file():
            raise ValueError('Missing or nonportable bundle asset: ' + str(relative))
        return path

    def _index(self, i):
        if not 0 <= i < self.num_frames:
            raise IndexError(i)
        return int(self.indices[i])

    def get_rgb(self, i):
        index = self._index(i)
        if self.manifest:
            with Image.open(self.asset(self.manifest['frames'][index]['rgb'])) as img:
                if img.size != (self.width, self.height):
                    raise ValueError('RGB dimensions differ from calibration')
                return np.asarray(img.convert('RGB'))
        a = self._video[index][..., :3]
        # Native AGILE exports keep RGB in byte range even in float RGBD.
        # Do not infer encoding per frame: an almost-black byte-range frame
        # containing only 0 and 1 must not suddenly become white.
        return np.rint(np.clip(a, 0, 255)).astype(np.uint8)

    def get_depth(self, i):
        index = self._index(i)
        if self.manifest:
            with np.load(self.asset(self.manifest['frames'][index]['depth']), allow_pickle=False) as z:
                depth = z['depth']
                if depth.shape != (self.height, self.width):
                    raise ValueError('Depth dimensions differ from calibration')
                return depth
        return self._video[index][..., 3].astype(np.float32)

    def get_camera_info(self, i):
        self._index(i)
        return dict(K=self.K[i].copy() if self.K.ndim == 3 else self.K.copy(),
                    cam_c2w=self.cam_c2w[i].copy(), height=self.height, width=self.width,
                    frame_id=self.frame_ids[i], timestamp_seconds=self.timestamps[i], rgbd=self.rgbd[i])

    def get_mano_vertices(self, i, world=False):
        self._index(i)
        if self.mano_verts is None:
            return None
        v = self.mano_verts[i, 0].copy()
        return v @ self.cam_c2w[i, :3, :3].T + self.cam_c2w[i, :3, 3] if world else v

    def get_object_pose(self, i, world=False):
        self._index(i)
        if self.object_poses_in_cam is None:
            return None
        pose = self.object_poses_in_cam[i].copy()
        return self.cam_c2w[i] @ pose if world else pose

    def get_points(self, i, downsample_factor=1):
        if downsample_factor < 1:
            raise ValueError('downsample_factor must be positive')
        k = self.K[i] if self.K.ndim == 3 else self.K
        depth = self.get_depth(i)[::downsample_factor, ::downsample_factor]
        color = self.get_rgb(i)[::downsample_factor, ::downsample_factor] / 255.
        yy, xx = np.mgrid[:self.height:downsample_factor, :self.width:downsample_factor]
        pixels = np.stack((xx, yy, np.ones_like(xx)), axis=-1)
        points = (pixels @ np.linalg.inv(k).T) * depth[..., None]
        valid = np.isfinite(depth) & (depth > 0) & (depth < 65.535)
        return ((points[valid] @ self.cam_c2w[i, :3, :3].T + self.cam_c2w[i, :3, 3]).astype(np.float32),
                color[valid].astype(np.float32))

    @property
    def has_mano(self):
        return self.mano_verts is not None and self.mano_faces is not None

    @property
    def has_object(self):
        return self.object_mesh is not None and self.object_poses_in_cam is not None

    def verify_assets(self):
        if not self.manifest:
            raise ValueError('Verification requires a versioned result bundle')
        for path, expected in self.manifest['sha256'].items():
            if sha256(self.asset(path)) != expected:
                raise ValueError('Asset hash mismatch: ' + path)
        return True


def pack_result(source, output, *, reference_rgb_dir=None, label=None,
                source_fps=None, source_fps_provenance=None, start=0, end=-1,
                max_frames=None, handedness=None, asset_manifest=None,
                canonical_input=None, native_object_poses=None):
    """Freeze a result into a self-contained bundle; never overwrite an export.

    Reference image filenames must be numeric ORIGINAL frame IDs. No nearest
    frame matching, resizing, pose interpolation or mesh decimation is performed.
    """
    loader = ResultBundle(source, start_frame=start, end_frame=end, max_frames=max_frames,
                          source_fps=source_fps, source_fps_provenance=source_fps_provenance)
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a new bundle directory: ' + str(output))
    from optimization.data.object_assets import (read_object_asset, load_mesh, copy_asset_package,
                                    mesh_summary, export_lossless_glb)
    if loader.manifest:
        loader.verify_assets()
    variants = {}
    asset_record = copy.deepcopy(loader.object_assets)
    scale = loader.optimized_object_scale
    source_asset_manifest = None
    if asset_record and asset_record.get('source_asset_manifest'):
        source_asset_manifest = loader.asset(asset_record['source_asset_manifest'])
    if asset_manifest:
        config = loader.config if not loader.manifest else loader.source.get('configuration', {})
        if config.get('optimize_object_texture') or config.get('optimize_vertex_colors'):
            raise ValueError('Cannot replace a result that explicitly optimized appearance')
        if (not asset_record or asset_record.get('appearance_optimized') is not False) and any(
                config.get(key) is not False for key in ('optimize_object_texture', 'optimize_vertex_colors')):
            raise ValueError('Historical appearance optimization policy is unknown; verify the original configuration first')
        if not canonical_input:
            raise ValueError('Restoring a final asset requires its original canonical input')
        final_mesh, record = read_object_asset(asset_manifest, load_mesh(canonical_input))
        if native_object_poses:
            poses = _npz(native_object_poses)
            indices = [loader.manifest['frames'][i]['source_array_index'] for i in loader.indices] if loader.manifest else loader.indices
            if not np.array_equal(poses['object_poses_in_cam'][indices], loader.object_poses_in_cam):
                raise ValueError('Native poses do not match the saved result')
            scale = np.asarray(poses['object_scale']).tolist()
        if scale is None:
            raise ValueError('Old bundle lost the original object scale; supply matching --native-object-poses')
        if np.asarray(scale).size not in (1, 3) or not np.isfinite(scale).all() or np.any(np.asarray(scale) <= 0):
            raise ValueError('Invalid restored object scale')
        final_mesh.apply_scale(scale)
        variants = {'display': final_mesh, 'optimization_proxy': loader.object_mesh.copy()}
        asset_record = dict(schema_version=1, object_scale_baked=True, default='display',
                            source_stage=record['stage'], appearance_optimized=False,
                            texture_policy='preserve-selected-final-texture',
                            restored_from_saved_result=True,
                            native_object_poses_sha256=sha256(native_object_poses) if native_object_poses else None,
                            canonical_input_sha256=sha256(canonical_input))
        source_asset_manifest = Path(asset_manifest)
        loader.notices = [n for n in loader.notices if not n.startswith('Legacy ')]
        loader.notices.append('Display uses the verified final generated asset; the earlier optimization export is retained as its proxy.')
    elif asset_record:
        variants = {name: loader.load_object_variant(name) for name in asset_record['variants']}
    elif loader.object_mesh is not None:
        variants = {'display': loader.object_mesh}
        asset_record = dict(schema_version=1, object_scale_baked=True, default='display',
                            source_stage='legacy-export-unverified-stage',
                            texture_policy='as-saved')
    references = {}
    if reference_rgb_dir:
        if any(x is None for x in loader.frame_ids):
            raise ValueError('Original RGB lookup requires known frame IDs')
        for p in Path(reference_rgb_dir).iterdir():
            if p.suffix.lower() in ('.jpg', '.jpeg', '.png') and p.stem.isdigit():
                frame_id = int(p.stem)
                if frame_id in references:
                    raise ValueError('Duplicate reference frame ID: ' + str(frame_id))
                references[frame_id] = p
        missing = set(loader.frame_ids) - references.keys()
        if missing:
            raise ValueError('Missing reference frame IDs: ' + str(sorted(missing)[:10]))
    output.mkdir(parents=True)
    (output / 'rgb').mkdir()
    (output / 'depth').mkdir()
    arrays = dict(intrinsic=loader.K, cam_c2w=loader.cam_c2w)
    for key in ('mano_verts', 'mano_joints', 'mano_faces', 'object_poses_in_cam'):
        if getattr(loader, key) is not None:
            arrays[key] = getattr(loader, key)
    np.savez_compressed(output / 'geometry.npz', **arrays)
    mesh = 'object.glb' if variants else None
    if variants:
        asset_record['object_scale_baked'] = True
        asset_record['variants'] = {}
        for name, value in variants.items():
            if name not in ('display', 'optimization_proxy', 'source_reference'):
                raise ValueError('Unknown object variant: ' + name)
            filename = 'object.glb' if name == 'display' else 'object_' + name + '.glb'
            export_lossless_glb(value, output / filename)
            asset_record['variants'][name] = dict(mesh=filename, **mesh_summary(value))
        if source_asset_manifest:
            copy_asset_package(source_asset_manifest, output / 'source_asset')
            asset_record['source_asset_manifest'] = 'source_asset/asset.json'
    frames = []
    for i in range(loader.num_frames):
        fid = loader.frame_ids[i]
        rgb_path = references.get(fid)
        if rgb_path is None and loader.manifest:
            rgb_path = loader.asset(loader.manifest['frames'][loader.indices[i]]['rgb'])
        if rgb_path:
            with Image.open(rgb_path) as img:
                if img.size != (loader.width, loader.height):
                    raise ValueError('Reference size differs from calibrated result: ' + str(rgb_path))
            name = f'rgb/{i:06d}{rgb_path.suffix.lower()}'
            shutil.copyfile(rgb_path, output / name)
        else:
            name = f'rgb/{i:06d}.png'
            Image.fromarray(loader.get_rgb(i)).save(output / name)
        depth_name = f'depth/{i:06d}.npz'
        np.savez_compressed(output / depth_name, depth=loader.get_depth(i))
        source_index = (loader.manifest['frames'][loader.indices[i]]['source_array_index']
                        if loader.manifest else int(loader.indices[i]))
        frames.append(dict(index=i, source_array_index=source_index,
                           frame_id=int(fid) if fid is not None else None,
                           timestamp_seconds=float(loader.timestamps[i]) if loader.timestamps[i] is not None else None,
                           rgb=name, depth=depth_name, valid_hand=bool(loader.valid_hand[i]),
                           valid_object=bool(loader.valid_object[i])))
    manifest = dict(schema_version=2, label=label or (loader.manifest or {}).get('label') or Path(source).parent.name,
                    coordinates='opencv-camera-metres', object_scale_baked=True,
                    camera_pose_convention='camera-to-world; world is source reconstruction frame, not a calibrated robot base',
                    validity_definition='finite nondegenerate hand vertices / finite rigid object pose; not accuracy or confidence',
                    joint_order='source-export-order; preserve without remapping',
                    handedness=handedness or (loader.manifest or {}).get('handedness', 'unknown'), geometry='geometry.npz', object_mesh=mesh,
                    object_assets=asset_record, optimized_object_scale=scale,
                    image_size=[loader.height, loader.width], frames=frames,
                    frame_ids_source=loader.frame_ids_source, time_source=loader.time_source,
                    rgb_source='original-reference-frames' if references else (loader.manifest or {}).get('rgb_source', 'saved-optimization-rgb'),
                    reference_frame_count=len(references) if references else (loader.manifest or {}).get('reference_frame_count'),
                    source=loader.source, notices=loader.notices,
                    sha256={str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*')) if p.is_file()})
    # Write the manifest last; incomplete exports have no result.json.
    (output / 'result.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return manifest
