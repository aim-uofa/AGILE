# Prepared input file reference

This reference documents the stored files for developers working with the data loaders. For downloading and using the data, follow [prepared data](data.md).

## File organization

Both layouts contain **MoGe depth, WiLoR hand estimates and FoundationPose object initialization**. Both include hand-to-depth alignment. They preserve different preparation batches and organize the inputs differently:

| Layout | Sequences |
| --- | --- |
| Component directories (`moge-wilor-foundationpose`) | ABF12, ABF14, GPMF12, GPMF14, MC1, MC4, SM2, SM4, ShSu10, ShSu12 |
| Pre-aligned NPZ package (`aligned-npz`) | BB12, BB13, GSF13, GSF14, MDF12, MDF14, SMu1, SMu40 |

| Component | Directory layout | NPZ package |
| --- | --- | --- |
| Images, depth and camera | `rgb/`, `moge/depth/`, `moge/cam_K.txt` | `camera_data.npz`: RGBD, intrinsics and camera transforms |
| WiLoR estimates | `wilor/` | Top-level MANO pose/shape/orientation arrays; `wilor/mano_joints_2d.npz` supplies 2D joints |
| Depth-aligned hand | `hand_processed/` | `optimized_mano_verts.npz`, `optimized_mano_joints.npz`, `mano_transforms.npz` |
| Initial object pose and mesh | `object_pose/ob_in_cam/`, prepared meshes and scale files | `object_poses.npz`, `object_mesh.obj` and associated material/texture files |
| Segmentation | `mask/hand/`, `mask/object/` | `mask/hand/`, `mask/object/` |

The NPZ layout is a package of several arrays plus meshes and masks, not a single archive containing final reconstruction results. Directory-based inputs also contain NPZ hand arrays.

## What “aligned” means

During preprocessing, WiLoR's MANO hand geometry is fitted to the visible hand point cloud obtained by back-projecting predicted depth inside the hand mask. The fit uses a **shared scale and rotation across the sequence, with a separate translation for each frame**. The resulting vertices, joints and transforms are saved as initialization for motion optimization.

This is alignment between estimated hand geometry and predicted depth. It does **not** mean alignment to ground-truth annotations or the hand-Sim3 alignment used by an evaluation metric. The `optimized_mano_*` filenames refer to this preprocessing fit, not completion of AGILE's subsequent motion optimization.

For all eight released NPZ sequences, `cam_c2w` is the identity for every frame: these inputs use a single camera coordinate frame, not a registered multi-camera scene. The loader is named `optimization/data/megasam.py` for historical reasons; the released NPZ inputs listed here were checked against their MoGe and FoundationPose preparation files.

Depth PNGs in the directory layout store millimeters and are converted to meters when loaded. The NPZ RGBD depth channel is already in meters. Both loaders provide the optimizer with its expected common input fields.

## Use the matching components

The optimizer detects `camera_data.npz` and chooses the NPZ loader automatically; otherwise it reads the component directories. Each preprocessed archive includes its corresponding WiLoR 2D joints. Older pinned releases stored the joints for these eight sequences in separate `initialization` archives, which the downloader also handles. See [sequence inputs](ho3d-inputs.json) for the exact component paths, contact frames and archive combinations.

Keep each sequence's prepared mesh, poses, hand transforms and camera data together. The two layouts are not interchangeable numerical copies: mixing components from different preparation batches can change initialization. The release retains the combinations used by the validated configurations. Combining the old initialization archive with its matching preprocessed archive changes only the packaging, not the extracted inputs.

All eight NPZ sequences contain 1,024 prepared frames. Other sequences have their own prepared ranges, and some raw videos extend beyond the released inputs. The sequence manifest records both counts; `end_frame=-1` uses the prepared range, not the unavailable raw tail.

The public [preprocessing entry point](../preprocessing/README.md) produces the directory layout. Download the released NPZ packages to use those historical prepared inputs unchanged.

[Back to prepared data](data.md)
