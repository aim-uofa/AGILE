# Prepared data

Download ready-to-use HO3D inputs and generated object assets for 18 sequences from [Chrisss/AGILE](https://huggingface.co/datasets/Chrisss/AGILE). These inputs preserve the per-sequence selections used by the validated HO3D experiments.

## What is included

- RGB frames and hand/object masks.
- MoGe depth and camera intrinsics.
- WiLoR hand estimates and hand-to-depth alignment.
- FoundationPose object initialization and the corresponding optimization mesh.
- Selected generated objects with their materials and textures.

Each sequence has one self-contained preprocessed archive, including its 2D hand keypoints, and one generated-object archive. The downloader retrieves the matching pair and verifies archive and file checksums. The optimizer loads the inputs automatically; users do not need to choose a storage format or combine initialization packages.

## Download and use

Run from the repository root after [installation](installation.md#download-only-environment). Downloading alone needs only Python and the Hugging Face client, with no GPU or account:

```bash
python tools/download_ho3d.py --scenes MC1 --output /ABS/data
```

Prepared inputs are placed under `HO3D/preprocessed/train/SCENE/`; selected generated assets are placed under `HO3D/generated/train/SCENE/object/`.

Continue with [optimization, evaluation and visualization](../optimization/README.md#run-optimization), using `--data-root /ABS/data/HO3D/preprocessed`. Downloaded inputs are already prepared, so you can skip preprocessing and object generation.

Keep each sequence's mesh, poses, hand estimates and camera data together. The generated display asset and the prepared optimization mesh have different roles; use the mesh associated with the prepared poses for optimization.

## Coverage and reproducibility

The release contains the prepared frame ranges used by the experiments. Some original videos extend beyond these ranges; `end_frame=-1` uses all available prepared frames. [Sequence inputs](ho3d-inputs.json) records contact frames, archive combinations and prepared/raw frame counts.

For a reproducible download, pass `--revision COMMIT` using the revision recorded in [data-release.json](data-release.json). The former eight small initialization archives are now included in their matching preprocessed archives. This packaging change preserves every extracted file path and byte, selected component and frame range. Older releases remain available by revision, and the manifest-based downloader can still read them. Historical validation records retain the exact revision used by each run.

Input identity alone does not specify a benchmark: results also depend on the optimization code, configuration and evaluation coverage. See [validation records](validation.json) for the tested versions and scope.

For loader development or custom data preparation, see the [input file reference](data-layout.md).

## Optimized reconstructions

Completed optimization outputs for all 18 sequences are available separately from the prepared inputs. Use `python tools/download_ho3d.py --kind optimized --scenes MC1 --output /ABS/data` to download a final sequence checkpoint. See the [result format, metrics and evaluation guide](ho3d-results.md).

[Back to the project overview](../README.md)
