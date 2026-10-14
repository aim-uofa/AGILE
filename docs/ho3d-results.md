# HO3D optimized results

Download the final reconstructions for all **18 HO3D sequences** without rerunning optimization. Each sequence checkpoint contains the complete sampled trajectory, hand geometry and parameters, object transforms and scale, textured meshes, camera calibration, RGBD, the effective configuration and evaluation metrics.

These are final per-sequence optimization outputs, not pretrained network weights or checkpoints for resuming Adam iterations.

## Download and visualize

Install the [optimization and visualization dependencies](installation.md), then run from the repository root:

```bash
python tools/download_ho3d.py --kind optimized --scenes MC1 --output /ABS/data
python -m scripts.vis.run_viser_hoi \
  --data-dir /ABS/data/HO3D/optimized/train/MC1/aligned --port 8090
```

You can also download an archive directly from [Hugging Face](https://huggingface.co/datasets/Chrisss/AGILE/tree/main/ho3d/optimized):

```bash
hf download Chrisss/AGILE --repo-type dataset \
  ho3d/optimized/MC1.tar.gz ho3d/optimized/SHA256SUMS --local-dir /ABS/downloads
cd /ABS/downloads
sha256sum --check --ignore-missing ho3d/optimized/SHA256SUMS
tar -xzf ho3d/optimized/MC1.tar.gz -C /ABS/data
```

Create `/ABS/data` first. The repository downloader above also validates every archive member and guards against unsafe extraction and accidental overwrites.

Omit `--scenes` to download all 18 sequences. Add `--revision COMMIT` to pin the Hugging Face release. The downloader verifies the archive and every extracted file and refuses to overwrite different local files. Visualization reads saved geometry and does not require the original MANO model, DINO weights or evaluation ground truth.

Each archive extracts to `HO3D/optimized/train/SCENE/`:

| File | Contents |
| --- | --- |
| `aligned/` | Native hand geometry and parameters, object poses and scale, display/proxy meshes, camera calibration, RGBD and original frame IDs. |
| `config.json` | The effective optimization settings; machine-specific paths are replaced with portable paths. |
| `metrics_protocol.json` | Original evaluation values, per-metric frame coverage and protocol definitions. |
| `verification.json` | Full-output, geometry, data-integrity and access-check summary. |
| `release.json` | Exact training/evaluation revisions, input-data revision and provenance hashes. |

Geometry is in OpenCV camera coordinates, in metres. The native object mesh is unscaled: apply the saved `object_scale` once, followed by `object_poses_in_cam`. The provided loader and viewer handle this convention. All numerical arrays and assets retain their original bytes.

## Results and evaluation

The [published results](https://huggingface.co/datasets/Chrisss/AGILE/blob/main/ho3d/optimized/results.json) list six metrics for every sequence and their equal-weight means. [The manifest](https://huggingface.co/datasets/Chrisss/AGILE/blob/main/ho3d/optimized/manifest.json) records archive and file checksums. All 18 runs passed full optimization, export, evaluation and independent geometry/data checks: **3,590 sampled frames and 3,437 valid ground-truth frames**.

| Metric | Paper, Table 1 | Released results |
| --- | ---: | ---: |
| MPJPE ↓ (mm) | 3.92 | 3.9453 |
| Object CD ↓ (cm²) | 0.27 | 0.2710 |
| F@5 ↑ (%) | 86.63 | 84.3044 |
| F@10 ↑ (%) | 97.77 | 98.5210 |
| Wrist-relative interaction CDₕ ↓ (cm²) | 15.81 | 15.7669 |
| Hand-Sim3 interaction CD ↓ (cm²) | — | 18.0258 |

Paper values are from [Table 1](https://arxiv.org/html/2602.04672v4#S4.T1). Current values use `evaluation-v2-gt-cv-robust-icp` on the full available prepared range, including frames before the contact anchor. Some original video tails are absent from the prepared inputs. Object shape uses the released robust registration procedure. These differences prevent a claim of an identical paper evaluation; completion of 18 runs is not a separate measurement of the paper's tracking success rate.

`cd_right` measures wrist-relative interaction without fitting rotation or scale. `cd_interaction_hand_sim3_per_frame` fits a similarity transform to the hand joints and is a different metric. Do not substitute the latter for the paper's CDₕ reference.

To independently evaluate a downloaded sequence, obtain the original HO3D annotations, object models and MANO support files as described in the [evaluation guide](../optimization/README.md#evaluation):

```bash
export HO3D_ROOT=/ABS/HO3D/raw
export HO3D_MANO_ROOT=/ABS/MANO
python -m scripts.experiments.evaluate_protocol --dataset ho3d --scene MC1 \
  --result-dir /ABS/data/HO3D/optimized/train/MC1 \
  --manifest-path configs/ho3d_evaluation.json \
  --output-json /ABS/results/MC1-reevaluated.json
```

The released records retain the actual source version for each sequence. Eleven accepted runs used `52b7db9`; seven used `da1799a`, which corrects backward data reading without changing optimization parameters or metric formulas. The input revision is `Chrisss/AGILE@e306a6b973ac88838d0ec5c0325cddd039b4c28e`.

Use is subject to the [AGILE license](../LICENSE) and the original third-party terms. MANO model files, external network weights and evaluation ground truth are not included.

[Back to the project overview](../README.md)
