# Motion optimization, evaluation and visualization

Turn prepared video inputs and an object mesh into a hand-object motion sequence. This module initializes the interaction at a contact frame, tracks in both directions, and exports geometry that can be evaluated, packaged and played back independently of the training data.

## Features

- **Contact-anchored tracking:** align object pose and scale at the selected contact frame, then track forward and backward using silhouettes, local image features and hand-object relative-motion constraints.
- **Hand translation refinement:** fit translation to 2D joints with temporal smoothing. The released configuration keeps WiLoR global rotation and articulated hand pose fixed.
- **Object pose recovery:** when silhouette overlap falls below the configured threshold, test prepared-pose candidates before continuing tracking.
- **Detailed output with a lightweight proxy:** optimize a simplified mesh while retaining the selected display asset and its texture for export and visualization.

The current runnable dataset configuration is [HO3D](../configs/ho3d.json). [Sequence settings](../configs/ho3d_scenes.json) specify contact frames and prepared ranges. Additional dataset configurations will be released progressively.

## Inputs and outputs

| Input | Purpose |
| --- | --- |
| Prepared sequence | RGB/depth, masks, camera parameters, WiLoR hand estimates and object pose initialization. Download the matching components using the [prepared data guide](../docs/data.md). |
| Prepared object mesh | Canonical geometry and appearance associated with the sequence's pose initialization. |
| MANO assets and DINO checkpoint | Hand geometry and local image features used during optimization. |
| Original HO3D annotations and object models | Required for evaluation only. |

The run directory contains the effective `config.json`, diagnostics and final native results under `aligned/`:

| Output | Contents |
| --- | --- |
| `aligned/camera_data.npz` | Camera intrinsics, camera-to-world transforms, RGBD and original prediction frame IDs. |
| `aligned/optimized_mano_verts.npz`, `aligned/optimized_mano_joints.npz` | Reconstructed hand vertices and joints for each sampled frame. |
| `aligned/object_poses.npz` | Object transforms and optimized scale; accompanying mesh/material files store canonical geometry and appearance. |
| `aligned/object_assets/` | Display and optimization mesh variants with their asset metadata. |
| `metrics_protocol.json` | Evaluation metrics, frame coverage and protocol definitions, written by the evaluator. |

Native geometry uses OpenCV camera coordinates in metres. The portable exporter below records coordinate conventions and frame IDs, and applies object scale once when packaging the meshes.

## Dependencies

Follow the [installation guide](../docs/installation.md) for Linux, Python 3.10, CUDA 11.8, PyTorch, PyTorch3D, MANO support files and the SDF extension. Use a CUDA-capable GPU for optimization. The same environment includes Viser and the rendering dependencies used below.

Set the asset paths before running:

```bash
export HO3D_MANO_ROOT=/ABS/MANO
export DINO_MODEL_PATH=/ABS/dinov3-vitl16-pretrain-lvd1689m
```

MANO and the DINO checkpoint are obtained separately. Downloaded prepared inputs do not require rerunning WiLoR, MoGe, FoundationPose or object generation. Run all commands below from the repository root.

## Download data

To use the completed reconstructions directly, follow the [optimized results guide](../docs/ho3d-results.md). It includes checksum-verified downloads, visualization and independent evaluation for all 18 sequences.

```bash
python tools/download_ho3d.py --scenes MC1 --output /ABS/data
```

The downloader verifies checksums and retrieves each sequence's self-contained prepared inputs and matching generated asset. Optimization reads `/ABS/data/HO3D/preprocessed/train/MC1`; generated source assets are available separately under `/ABS/data/HO3D/generated/train/MC1/object/`.

To regenerate a released sequence, follow the [preprocessing guide](../preprocessing/README.md), then point `--data-root` below to the new prepared root. Use the mesh and initialization produced together, rather than substituting a mesh with a different coordinate system or scale. The wrapper accepts only sequence IDs in `configs/ho3d_scenes.json`; an additional sequence needs its own configuration and contact frame.

## Run optimization

First inspect the effective configuration without loading CUDA modules:

```bash
python -m optimization.run_ho3d --scene MC1 \
  --data-root /ABS/data/HO3D/preprocessed --mano-root /ABS/MANO \
  --output /ABS/results/MC1 --dry-run
```

Then run the complete sequence on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m optimization.run_ho3d --scene MC1 \
  --data-root /ABS/data/HO3D/preprocessed --mano-root /ABS/MANO \
  --output /ABS/results/MC1
```

Choose a new output directory; existing results are not overwritten. The released configuration samples every fifth frame, starts at the configured contact frame and merges both tracking directions into the final `aligned/` export. `end_frame=-1` uses all prepared frames. For MC1, the complete export contains 180 sampled frames, from original frame 0 through 895.

Use this wrapper to apply the released configuration. Advanced research options are available through `python train.py --help`; changing them defines a different experiment.

## Evaluation

Obtain the original HO3D annotations and object models, and set their root directory. The layout must include `train/MC1/` and `models/` as provided by HO3D.

```bash
export HO3D_ROOT=/ABS/HO3D/raw
export HO3D_MANO_ROOT=/ABS/MANO

python -m scripts.experiments.evaluate_protocol --dataset ho3d --scene MC1 \
  --result-dir /ABS/results/MC1 --manifest-path configs/ho3d_evaluation.json
```

The evaluator checks that prediction frame IDs match the declared full-sequence range, then writes `metrics_protocol.json`. It reports the full prepared range and, where available, a separately named paper window.

| Metric | Meaning |
| --- | --- |
| `mpjpe_ra_r` | Wrist-aligned hand joint error in mm; lower is better. |
| `cd_icp`, `f5_icp`, `f10_icp` | Object shape Chamfer distance in cm² and F-scores at 5/10 mm after shape registration. Lower CD and higher F-scores are better. |
| `cd_right` | Wrist-relative interaction Chamfer distance: bidirectional squared distance sum in cm², without fitting rotation or scale. |
| `cd_interaction_hand_sim3_per_frame` | Interaction Chamfer distance after fitting a per-frame similarity transform to hand joints and applying it to the object, in cm². |

The two interaction metrics answer different questions and should not be interchanged. Compare the same metric over the same sequence and frame coverage. The output records protocol `evaluation-v2-gt-cv-robust-icp`, per-metric coverage and the evaluation code revision. A partial run cannot substitute for full-sequence evaluation.

## Export and visualize

Package a complete native export into a new, portable directory:

```bash
python -m scripts.vis.pack_result \
  --data-dir /ABS/results/MC1/aligned \
  --output /ABS/results/MC1-bundle --label MC1
```

The bundle contains `result.json`, `geometry.npz`, meshes/materials, RGB/depth frames and asset checksums. It preserves both display and optimization mesh variants when available. The bundle can be inspected without the original dataset or MANO model files.

Start the interactive viewer and open `http://127.0.0.1:8090`:

```bash
python -m scripts.vis.run_viser_hoi \
  --data-dir /ABS/results/MC1-bundle --port 8090
```

The viewer provides frame playback, textured mesh and depth-point toggles, the matching source image, and calibrated PNG/GIF exports. It also accepts a native `aligned/` directory. Use `--object-variant optimization_proxy` to inspect the mesh used by optimization. When running remotely, forward port 8090 to your local machine with SSH.

For a shareable comparison page, render the bundle with the saved camera calibration:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.vis.render_demo \
  --data-dir /ABS/results/MC1-bundle \
  --output /ABS/results/MC1-preview --presentation-fps 5
```

This writes reference, overlay and rendered frames, `comparison.json`, `index.html`, and an MP4 for evenly spaced playback. `--presentation-fps` supplies display timing when source timestamps are unavailable; it does not change reconstruction sampling. Keep bundle assets and their recorded transforms together when moving or reusing the results.

[Back to the project overview](../README.md)
