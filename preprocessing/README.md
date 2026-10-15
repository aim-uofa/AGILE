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

Install the three external tools and their required weights separately. Use these public forks: the pipeline depends on their video/depth/pose entry points. Clone them from the AGILE repository root:

```bash
mkdir -p external
git clone https://github.com/Chuan-10/WiLoR.git external/WiLoR
git -C external/WiLoR checkout c851b1bae9d65420f00091b729ea0f26f45d69db
git clone https://github.com/Chuan-10/MoGe.git external/MoGe
git -C external/MoGe checkout 14cd910362d75dda4413dd72a2c5c61dc5e78901
git clone https://github.com/Chuan-10/FoundationPose.git external/FoundationPose
git -C external/FoundationPose checkout 07c8609ad5a6e2eea88023cbfc0eca433f076edf
```

Follow each checkout's installation and weight-download instructions: [WiLoR](https://github.com/Chuan-10/WiLoR/blob/c851b1bae9d65420f00091b729ea0f26f45d69db/README.md#original-installation), [MoGe](https://github.com/Chuan-10/MoGe/blob/14cd910362d75dda4413dd72a2c5c61dc5e78901/README.md), and [FoundationPose](https://github.com/Chuan-10/FoundationPose/blob/07c8609ad5a6e2eea88023cbfc0eca433f076edf/readme.md#env-setup-option-2-conda-experimental). Keep them in their own Conda environments; use WiLoR's full source installation, since the pipeline calls `demo_vid.py`.

Their environments are invoked through `conda run`; the names above can be changed with the command-line options. Model weights are not included in this repository. Return to the AGILE repository root and check the entry points before processing:

```bash
conda activate agile-optim
python tools/check_preprocessing_toolchain.py \
  --wilor-path external/WiLoR --moge-path external/MoGe \
  --foundationpose-path external/FoundationPose
```

This checks the CLI contract and source hashes against [preprocessing-toolchain.json](../configs/preprocessing-toolchain.json). It does not load models or validate CUDA inference.

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

## In-the-wild videos

Object generation already accepts frames from your own single-hand videos. The complete custom-video workflow, including frame and mask preparation, preprocessing, contact-frame selection and optimization configurations, is on the [release roadmap](../README.md#todo). The current documented preprocessing and optimization presets target HO3D; no end-to-end in-the-wild example is included yet.

[Back to the project overview](../README.md)
