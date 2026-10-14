"""Resume a fixed-appearance forward tail from an existing native result.

Legacy runs saved no training checkpoint. Their ordered hand/object PLY captures
retain the actual optimized geometry, even when a rotation conversion damaged
the exported pose. This command recovers the anchor pose from that capture,
checks the original proxy and MANO state, then continues normal per-frame work.
The anchor is retained; optimization starts at the following sampled frame.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from optimization.config.config import Config


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fit_ordered_rigid_pose(vertices, target, max_error_m=2e-6):
    """Fit corresponding vertices; reject changed topology, scale or ordering."""
    vertices, target = np.asarray(vertices, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if vertices.shape != target.shape or vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError('Training PLY and optimization proxy must have identical vertex order/count')
    if len(vertices) < 4 or not np.isfinite(vertices).all() or not np.isfinite(target).all():
        raise ValueError('Invalid anchor vertices')
    center, target_center = vertices.mean(0), target.mean(0)
    if np.linalg.matrix_rank(vertices - center) != 3:
        raise ValueError('Anchor needs non-planar object geometry')
    u, _, vt = np.linalg.svd((vertices - center).T @ (target - target_center))
    sign = np.ones(3)
    sign[-1] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ np.diag(sign) @ u.T
    translation = target_center - center @ rotation.T
    residual = np.linalg.norm(vertices @ rotation.T + translation - target, axis=1)
    error = float(residual.max())
    if error > max_error_m:
        raise ValueError(f'Anchor rigid fit residual {error:.6g}m exceeds {max_error_m:.6g}m')
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, translation
    return pose, error


def anchor_index(frame_ids, anchor_frame):
    ids = np.asarray(frame_ids, dtype=np.int64)
    if ids.ndim != 1 or len(ids) < 2 or np.any(np.diff(ids) <= 0):
        raise ValueError('Resume requires unique, increasing source frame IDs')
    match = np.flatnonzero(ids == anchor_frame)
    if len(match) != 1 or match[0] == len(ids) - 1:
        raise ValueError('Anchor must exist and have at least one following sampled frame')
    return int(match[0])


def native_proxy_path(aligned):
    """Select the unscaled optimization proxy, never a newer display export."""
    aligned = Path(aligned).resolve()
    descriptor_path = aligned / 'object_assets.json'
    if not descriptor_path.exists():
        path = aligned / 'object_mesh.obj'
        if not path.is_file():
            raise ValueError('Missing legacy optimization proxy')
        return path
    descriptor = json.loads(descriptor_path.read_text())
    if descriptor.get('schema_version') != 1 or descriptor.get('object_scale_baked') is not False:
        raise ValueError('Resume needs a supported, unscaled native asset descriptor')
    relative = descriptor.get('variants', {}).get('optimization_proxy', {}).get('mesh')
    if not isinstance(relative, str) or not relative:
        raise ValueError('Native asset descriptor has no optimization proxy')
    path = (aligned / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(aligned) or not path.is_file():
        raise ValueError('Native optimization proxy must be a contained existing file')
    return path


def preserve_anchor_pcd(source, result_dir):
    """Keep the verified boundary capture so this result can resume again."""
    source = Path(source)
    target = Path(result_dir) / 'pcd_vis' / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or file_hash(source) != file_hash(target):
            raise ValueError('Refusing to replace a different existing anchor PLY')
    else:
        shutil.copyfile(source, target)
    return target


def load_resume_config(source_config, result_dir, object_asset_manifest=None):
    """Keep every optimization option; only output and display-asset paths differ."""
    raw = json.loads(Path(source_config).read_text())
    supported = {f.name for f in fields(Config) if f.init}
    unknown = set(raw) - supported - {'_cached_result_dir'}
    if unknown:
        raise ValueError(f'Unknown original config fields: {sorted(unknown)}')
    cfg = Config(**{key: value for key, value in raw.items() if key in supported})
    for name in ('reverse_frame_order', 'test_first_frame', 'optimize_object_texture',
                 'optimize_vertex_colors', 'optimize_hand_scale', 'optimize_intrinsics',
                 'optimize_object_scale_not_aligning', 'hand_shared_rotation'):
        if getattr(cfg, name):
            raise ValueError(f'Legacy tail recovery cannot restore {name}; use a full training checkpoint')
    if not cfg.fix_hand_rotation_to_wilor:
        raise ValueError('Legacy tail recovery currently requires fix_hand_rotation_to_wilor')
    if cfg.mano_transforms_path:
        raise ValueError('External MANO transforms are not supported by legacy tail recovery')
    cfg.result_dir = str(Path(result_dir).resolve())
    if object_asset_manifest is not None:
        cfg.object_asset_manifest = str(Path(object_asset_manifest).resolve())
    return cfg, raw


def restore_runner(runner, source_run, anchor_frame):
    """Restore prefix and boundary state without alignment or first-frame fitting."""
    import torch
    import trimesh
    from optimization.data.object_assets import load_mesh
    from optimization.utils.geometry import so3_log_batch

    source_run = Path(source_run)
    aligned = source_run / 'aligned'
    cfg, device = runner.cfg, runner.device
    # Bidirectional runs export the full sequence, although their config still
    # records the original contact seed. Do not mutate that optimization config.
    parser_start = 0 if cfg.bidirectional_optimization else cfg.start_frame
    runner.parser, runner.trainset = runner._build_parser_and_dataset(
        cfg.data_dir, start_frame=parser_start, end_frame=cfg.end_frame)
    frame_ids = np.asarray(runner.parser.frame_ids, dtype=np.int64)
    n, index = len(frame_ids), anchor_index(frame_ids, anchor_frame)
    runner.num_frames = n

    with np.load(aligned / 'object_poses.npz', allow_pickle=False) as data:
        poses = np.asarray(data['object_poses_in_cam']).copy()
        object_scale = np.asarray(data['object_scale']).reshape(-1).copy()
    if poses.shape != (n, 4, 4) or not np.isfinite(poses[:index + 1]).all():
        raise ValueError('Original exported poses do not cover the expected full sampled sequence')
    with np.load(aligned / 'camera_data.npz', allow_pickle=False) as data:
        if 'prediction_frame_ids' in data:
            if not np.array_equal(data['prediction_frame_ids'], frame_ids):
                raise ValueError('Original frame IDs do not match the current dataset')
            frame_id_evidence = 'saved-prediction-frame-ids'
        else:
            if len(data['rgbd']) != n:
                raise ValueError('Legacy camera data length does not match the full sampled sequence')
            frame_id_evidence = 'original-config-sampling-and-export-length'
        if not np.allclose(data['intrinsic'], runner.parser.K, atol=1e-5, rtol=0):
            raise ValueError('Dataset camera intrinsics changed since the original run')
        if not np.array_equal(data['rgbd'][index], runner.parser.rgbd[index]):
            raise ValueError('Anchor RGB/depth input changed since the original run')

    with np.load(aligned / 'mano_transforms.npz', allow_pickle=False) as data:
        hand_scale = np.asarray(data['scales']).copy()
        hand_t = np.asarray(data['translations']).copy()
        if not np.allclose(data['rotations'][:index + 1], 0, atol=1e-7):
            raise ValueError('Fixed-WiLoR source contains nonidentity hand rotation offsets')
    if hand_t.shape != (n, 3) or hand_scale.size != 1:
        raise ValueError('Invalid original hand transforms')
    runner.parser.mano_transforms['scales'] = hand_scale
    for filename, key, current in (
        ('mano_global_orient.npz', 'mano_global_orient', runner.parser.mano_global_orient),
        ('mano_betas.npz', 'mano_betas', runner.parser.mano_betas),
    ):
        with np.load(aligned / filename, allow_pickle=False) as data:
            if not np.array_equal(data[key], current):
                raise ValueError(f'Dataset {key} changed since the original run')
    with np.load(aligned / 'mano_hand_pose.npz', allow_pickle=False) as data:
        source_hand_pose = np.asarray(data['mano_hand_pose']).copy()
    if not cfg.optimize_hand_pose and not np.array_equal(source_hand_pose, runner.parser.mano_hand_pose):
        raise ValueError('Dataset MANO hand pose changed since the original run')

    with torch.no_grad():
        runner.optimized_obj_scale.copy_(torch.as_tensor(object_scale, device=device))
    mesh = runner.scene_renderer.get_mesh_object('object')
    canonical = mesh.canonical_vertices.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    proxy_path = native_proxy_path(aligned)
    source_mesh = load_mesh(proxy_path)
    # Native exports keep object scale separate. The old OBJ was unscaled too.
    if source_mesh.vertices.shape != canonical.shape or not np.array_equal(source_mesh.faces, faces):
        raise ValueError('Regenerated optimization proxy topology differs from the original')
    proxy_error = float(np.max(np.abs(source_mesh.vertices - canonical)))
    if proxy_error > 2e-7:
        raise ValueError(f'Regenerated optimization proxy changed by {proxy_error:.6g}m')

    pcd_path = source_run / 'pcd_vis' / f'frame_{anchor_frame:04d}_ho_pcd.ply'
    pcd = trimesh.load(pcd_path, process=False)
    points = np.asarray(pcd.vertices).copy()
    hand_count = 778
    if points.shape != (hand_count + len(canonical), 3):
        raise ValueError('Training PLY must contain 778 ordered hand vertices followed by the object')
    colors = np.asarray(pcd.colors)[:, :3]
    if not (np.all(colors[:hand_count] == [255, 0, 0]) and np.all(colors[hand_count:] == [0, 255, 0])):
        raise ValueError('Training PLY hand/object color order does not match save_ho_pcd')
    points[:, 1:] *= -1  # save_ho_pcd writes an OpenGL-style Y/Z flip.
    anchor_pose, fit_error = fit_ordered_rigid_pose(canonical * object_scale, points[hand_count:])
    old_anchor = poses[index].copy()
    poses[index] = anchor_pose

    def tensor(value):
        return torch.as_tensor(value, dtype=torch.float32, device=device).clone()

    runner.optimized_hand_pose_offsets = torch.zeros((n, 6), device=device)
    runner.optimized_obj_pose_offsets = torch.zeros((n, 6), device=device)
    runner.optimized_obj_R = torch.zeros((n, 3, 3), device=device)
    runner.optimized_obj_t = torch.zeros((n, 3), device=device)
    runner.optimized_hand_R = torch.zeros((n, 3, 3), device=device)
    runner.optimized_hand_t = torch.zeros((n, 3), device=device)
    runner.optimized_obj_R[:index + 1] = tensor(poses[:index + 1, :3, :3])
    runner.optimized_obj_t[:index + 1] = tensor(poses[:index + 1, :3, 3])
    runner.optimized_hand_R[:index + 1] = tensor(runner.parser.mano_global_orient[:index + 1].reshape(-1, 3, 3))
    runner.optimized_hand_t[:index + 1] = tensor(hand_t[:index + 1])
    if cfg.optimize_hand_pose:
        runner.optimized_mano_hand_pose = torch.zeros((n, 15, 3, 3), device=device)
        runner.optimized_mano_hand_pose[:index + 1] = tensor(source_hand_pose[:index + 1, 0])
    runner.shared_hand_rvec = torch.nn.Parameter(torch.zeros(3, device=device))
    runner.set_obj_base_pose(index, runner.optimized_obj_R[index], runner.optimized_obj_t[index])
    runner.set_hand_base_pose(index, runner.optimized_hand_R[index], runner.optimized_hand_t[index],
                             compute_mano=True,
                             hand_pose_vecs=so3_log_batch(runner.optimized_mano_hand_pose[index])
                             if cfg.optimize_hand_pose else None)
    hand = runner.scene_renderer.get_mesh_object('hand')
    reconstructed_hand = hand.get_transformed_vertices().detach().cpu().numpy()
    hand_error = float(np.linalg.norm(reconstructed_hand - points[:hand_count], axis=1).max())
    if hand_error > 2e-6:
        raise ValueError(f'Restored anchor MANO differs from training PLY by {hand_error:.6g}m')
    runner.last_hand_verts_incam = tensor(points[:hand_count])
    seed_valid = bool(torch.any(runner.trainset[index]['mano_hand_pose'] != 0))
    if not seed_valid:
        raise ValueError('Legacy resume requires a valid hand at the anchor')
    runner.hand_valid_current_frame = runner.hand_valid_previous_frame = seed_valid
    runner.hand_appeared = seed_valid
    rendered = runner.render_meshes(
        camtoworlds=torch.eye(4, device=device)[None],
        Ks=runner.trainset[index]['K'][None].to(device),
        height=runner.trainset[index]['image'].shape[0],
        width=runner.trainset[index]['image'].shape[1],
        frame_id=index, render_objects=['hand'], hard_rasterization=True)
    runner.hand_zbuf = rendered['zbuf'].detach().clone()
    if cfg.hand_sdf_loss or cfg.ho_relpose_loss:
        runner.precompute_object_sdf(canonical=True)
    # Per-frame DINO correspondences and optimizer moments are freshly created
    # by train_one_frame, just as they were between frames in the source run.
    np.savez(Path(cfg.result_dir) / 'resume_state.npz', frame_ids=frame_ids[:index + 1],
             object_poses=poses[:index + 1], object_scale=object_scale, hand_scale=hand_scale,
             hand_rotations=runner.optimized_hand_R[:index + 1].cpu().numpy(),
             hand_translations=hand_t[:index + 1], anchor_hand_vertices=points[:hand_count])
    copied_anchor = preserve_anchor_pcd(pcd_path, cfg.result_dir)
    hashed_sources = [source_run / 'config.json', pcd_path, proxy_path,
                      aligned / 'object_poses.npz', aligned / 'mano_transforms.npz']
    if (aligned / 'object_assets.json').exists():
        hashed_sources.append(aligned / 'object_assets.json')
    evidence = dict(anchor_frame=int(anchor_frame), first_optimized_frame=int(frame_ids[index + 1]),
                    last_optimized_frame=int(frame_ids[-1]), optimized_tail_count=n - index - 1,
                    retained_prefix_count=index + 1, frame_ids_source=frame_id_evidence,
                    anchor_object_fit_max_error_m=fit_error, anchor_hand_max_error_m=hand_error,
                    proxy_max_coordinate_error_m=proxy_error, object_scale=object_scale.tolist(),
                    hand_scale=float(hand_scale), anchor_original_export_pose=old_anchor.tolist(),
                    anchor_recovered_training_pose=anchor_pose.tolist(),
                    source_files_sha256={str(path.relative_to(source_run)): file_hash(path) for path in hashed_sources},
                    retained_anchor_pcd=str(copied_anchor.relative_to(cfg.result_dir)),
                    random_state='Original run did not save RNG state; standard Runner seed 42 is reset. '
                                 'Configuration is unchanged, but feature sampling is not bitwise continuation.',
                    optimizer_state='Per-frame Adam/schedulers start afresh in the original algorithm too.',
                    alignment_repeated=False,
                    object_iterations=cfg.object_max_iter_per_frame,
                    hand_iterations=cfg.hand_max_iter_per_frame)
    return index + 1, evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--anchor-frame', type=int, required=True)
    parser.add_argument('--result-dir', type=Path, required=True)
    parser.add_argument('--object-asset-manifest', type=Path)
    args = parser.parse_args(argv)
    source, output = args.source_run.resolve(), args.result_dir.resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Resume output must be separate from the original run')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Resume output must be a new or empty directory')
    cfg, original = load_resume_config(source / 'config.json', output, args.object_asset_manifest)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / 'config.json', output / 'source_config.json')
    from optimization.pipelines.trainer import Runner
    runner = Runner(cfg)
    start, evidence = restore_runner(runner, source, args.anchor_frame)
    differences = {name: {'source': value, 'resume': getattr(cfg, name)} for name, value in original.items()
                   if name != '_cached_result_dir' and value != getattr(cfg, name)}
    if cfg.object_asset_manifest != original.get('object_asset_manifest'):
        differences['object_asset_manifest'] = {
            'source': original.get('object_asset_manifest'), 'resume': cfg.object_asset_manifest}
    if set(differences) - {'result_dir', 'object_asset_manifest'}:
        raise ValueError(f'Optimization configuration changed: {differences}')
    evidence.update(schema_version=1, source_run=str(source), result_dir=str(output),
                    config_differences=differences, status='running')
    provenance_path = output / 'resume_provenance.json'
    provenance_path.write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2), flush=True)
    runner._run_train_loop([], str(output), skip_align_ho=True,
                           first_frame_max_iter=cfg.object_max_iter_per_frame,
                           save_periodic=True, start_dataset_idx=start)
    runner._print_final_summary()
    evidence['status'] = 'complete'
    provenance_path.write_text(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
