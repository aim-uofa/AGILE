# Preprocessing

Turn RGB frames, segmentation masks and a generated object mesh into the camera, depth, hand and object estimates consumed by AGILE. The current entry point prepares single-hand HO3D sequences.

For the released sequences, [download the prepared data](../optimization/README.md#download-data) and go directly to optimization. This guide is for running preprocessing yourself.

## Features

- Estimate the hand with WiLoR and select the hand matching the supplied mask.
- Estimate depth and camera intrinsics with MoGe, then align the hand geometry to the visible hand depth.
- Initialize the generated object's scale and per-frame pose with FoundationPose.
- Check frame coverage, image dimensions and array structure before passing data to optimization.

## Inputs

Provide RGB frames, hand and object masks, and a selected [generated object mesh](../generation/README.md). For example:

```text
HO3D/raw/train/MC1/rgb/00000.jpg
HO3D/raw/masks_SAM2/train/MC1/hand/00000.png
HO3D/raw/masks_SAM2/train/MC1/object/00000.png
HO3D/generated/train/MC1/object/textured_mesh.obj
```

Keep the OBJ's material and texture files beside it with their relative paths intact. Supply a mask for every selected RGB frame; this entry point consumes masks rather than generating them. Frame names must be numeric and contiguous, starting at zero, and RGB and masks must share the same resolution.

## Dependencies

Set up Linux, a CUDA-capable GPU and the `agile-optim` environment using [installation](../docs/installation.md). Obtain MANO and prepare its support files as described there.

| Component | Used for | Environment |
| --- | --- | --- |
| WiLoR | Hand geometry, articulation, orientation and 2D joints | `wilor` |
| MoGe | Depth and camera intrinsics | `moge` |
| FoundationPose | Object scale and pose initialization | `foundationpose` |
| AGILE + MANO | Hand selection, hand-to-depth alignment and output checks | `agile-optim` |

Install the three external tools and their required weights separately, using the public forks and revisions in [preprocessing-toolchain.json](../configs/preprocessing-toolchain.json). Their environments are invoked through `conda run`; the names above can be changed with the command-line options. Model weights are not included in this repository.

## Run

Run from the repository root. Replace `/ABS/...` with your local paths:

```bash
conda activate agile-optim

python -m preprocessing.preprocess_ho3d \
  --raw-data-root /ABS/HO3D/raw \
  --processed-data-root /ABS/HO3D/new-preprocessed \
  --generated-mesh-root /ABS/HO3D/generated \
  --mano-model-root /ABS/MANO \
  --wilor-path /ABS/WiLoR --wilor-env wilor \
  --moge-path /ABS/MoGe --moge-env moge \
  --foundationpose-path /ABS/FoundationPose --foundationpose-env foundationpose \
  --hoi-env agile-optim --scenes train/MC1 --max-frames 897
```

This example processes MC1's 897 frames. `--max-frames` limits the number of input frames; it is not a sampling interval. Use the intended range for your sequence and a new output scene directory. FoundationPose paths must not contain spaces or shell characters.

The stages run in order: input checks → hand estimation and selection → depth estimation → hand-to-depth alignment → object pose initialization → output validation.

## Outputs

The example writes `/ABS/HO3D/new-preprocessed/train/MC1/`:

| Output | Contents |
| --- | --- |
| `rgb/`, `mask/`, `frames.json` | Selected frames, supplied masks and source frame IDs |
| `wilor_raw/`, `wilor/` | Original predictions and the selected single-hand estimates |
| `moge/` | Depth PNGs, validity masks, camera intrinsics and depth export metadata |
| `hand_processed/` | Depth-aligned MANO vertices/joints and alignment transforms |
| `object_pose/` | Prepared object mesh, object-to-camera poses and scale files |
| `preprocessing.json` | File, frame and array validation report |

Depth PNGs store millimeters; loaders convert depth to meters. You can repeat the structural check without rerunning inference:

```bash
python -m preprocessing.check_inputs \
  --sequence-dir /ABS/HO3D/new-preprocessed/train/MC1 \
  --foundationpose-dir /ABS/HO3D/new-preprocessed/train/MC1/object_pose
```

Inspect the alignment visually as well: structural checks do not measure reconstruction accuracy. Continue with [optimization](../optimization/README.md#run-optimization), setting `--data-root /ABS/HO3D/new-preprocessed`.

For the released sequences, the [prepared data](../docs/data.md) already includes the selected inputs. See the [input file reference](../docs/data-layout.md) when developing a loader or inspecting individual preparation files.

[Back to the project overview](../README.md)
