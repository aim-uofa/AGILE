<div align="center">

# AGILE

### Hand-Object Interaction Reconstruction from Video via Agentic Generation

<p>
  <a href="https://agile-hoi.github.io"><img src="https://img.shields.io/badge/Project-Page-blue?style=for-the-badge" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2602.04672"><img src="https://img.shields.io/badge/arXiv-2602.04672-b31b1b?style=for-the-badge&logo=arxiv" alt="arXiv"></a>
  <a href="https://arxiv.org/pdf/2602.04672"><img src="https://img.shields.io/badge/Paper-PDF-green?style=for-the-badge" alt="Paper"></a>
  <a href="https://huggingface.co/datasets/Chrisss/AGILE"><img src="https://img.shields.io/badge/Hugging%20Face-Data-FFD21E?style=for-the-badge" alt="Hugging Face Data"></a>
</p>

[Prepared data](https://huggingface.co/datasets/Chrisss/AGILE/tree/main/ho3d/preprocessed) · [Generated objects](https://huggingface.co/datasets/Chrisss/AGILE/tree/main/ho3d/generated) · [Optimized results](https://huggingface.co/datasets/Chrisss/AGILE/tree/main/ho3d/optimized) · [Installation](docs/installation.md)

[Jin-Chuan Shi](https://chuan-10.github.io/)<sup>1\*</sup>, Binhong Ye<sup>1\*</sup>, Tao Liu<sup>1</sup>, Xiaoyang Liu<sup>1</sup>, Yangjinhui Xu<sup>1</sup>, Junzhe He<sup>1</sup>, Zeju Li<sup>1</sup>, [Hao Chen](https://stan-haochen.github.io/)<sup>1</sup>, [Chunhua Shen](https://cshen.github.io/)<sup>1,2</sup>

<sup>1</sup>State Key Lab of CAD & CG, Zhejiang University &nbsp; <sup>2</sup>Zhejiang University of Technology

<sup>\*</sup>Equal contribution

**ACM SIGGRAPH 2026, Conference Track**

<img src="assets/teaser.png" width="100%">

</div>

## News

- **[2026.10]** AGILE code and data released! Dataset configurations, preprocessed inputs and generated assets are being released progressively, starting with HO3D.
- **[2026.03]** AGILE is conditionally accepted to **ACM SIGGRAPH 2026** (Conference Track)!
- **[2026.03]** Project page and paper released.

## Overview

**AGILE** turns monocular videos of hand-object interaction into **textured 3D objects and reconstructed hand-object motion**. It combines agentic object generation with contact-anchored tracking, providing a workflow from video frames to meshes, motion, quantitative evaluation and interactive visualization.

## Highlights

- **Generate, review and refine.** A vision-language model selects complementary reference frames, reviews candidate geometry and appearance, and uses visual feedback to guide revisions. Run the workflow through configurable VLM and image-generation APIs, or use the optional Codex integration. Geometry and texture are reviewed separately, so texture refinement can preserve the approved shape and UV layout.
- **Two routes to object geometry.** Use multiview-to-3D generation, or let Codex construct a procedural mesh directly from reference images without a separate 3D generative model. Both routes support textured asset delivery; the Codex route uses texture images supplied through the agent client.
- **Track from a contact anchor.** Initialize object pose and scale at a contact frame, then track forward and backward. Silhouettes and local image features guide object motion, while contact-aware hand-object relative-motion constraints help stabilize the interaction.
- **Preserve detail while optimizing efficiently.** Use a lightweight mesh proxy for optimization and retain the selected textured mesh for visualization. Recovered poses and scale drive the display asset, keeping its appearance separate from the tracking workload.

## Modules

The modules can be used together or independently. Start with the released prepared data to run reconstruction, or use the generation and preprocessing tools to prepare new inputs.

| Module | What it does | Outputs |
| --- | --- | --- |
| [Preprocessing](preprocessing/README.md) | Takes RGB frames, hand/object masks and a selected object mesh; runs hand and depth estimation, aligns the hand to depth, initializes object poses and checks frame consistency. | Depth, hand estimates, camera parameters and initial object poses. |
| [Object generation](generation/README.md) | Selects reference views, builds and reviews candidates, prepares UVs, refines textures and exports the selected asset. Includes the paper's VLM/API workflow, Codex-assisted generation, Codex procedural geometry and external-asset workflows. | Textured GLB/OBJ meshes, materials, textures and candidate review records. |
| [Motion optimization](optimization/README.md#run-optimization) | Refines hand translation, object pose and initial object scale with the released configuration. Tracks in both directions and tests prepared-pose candidates when silhouette overlap collapses. | Per-frame hand geometry, object transforms and scale. |
| [Evaluation](optimization/README.md#evaluation) | Measures hand accuracy, object shape and hand-object interaction with explicit frame coverage and metric protocols. | MPJPE, object Chamfer distance and F-scores, wrist-relative and hand-aligned interaction Chamfer distances. |
| [Export and visualization](optimization/README.md#export-and-visualize) | Packages reconstruction results and assets; provides interactive 3D playback and camera-aligned renders and overlays. | Portable result bundles with meshes, joints, poses, camera calibration and frame IDs; images and animations. |

## Code and Data

The AGILE code is available in this repository. Prepared data, generated assets and optimized reconstructions are hosted on [Hugging Face](https://huggingface.co/datasets/Chrisss/AGILE).

The current data release includes **HO3D configurations, preprocessed inputs and generated objects for 18 sequences**. The downloader retrieves the matching components for each sequence and verifies their checksums, so you can start optimization without running preprocessing or generation. Additional dataset configurations and prepared results will be released progressively. See [prepared data](docs/data.md) for sequence combinations and the [data manifest](docs/data-release.json) for checksums.

**Optimized results for all 18 HO3D sequences** are also available. These sequence checkpoints contain the reconstructed hands, object poses and scale, meshes, camera data and evaluation metrics, ready for visualization and downstream use. See [download and evaluate results](docs/ho3d-results.md).

## HO3D Results

| Result | MPJPE (mm) ↓ | Object CD (cm²) ↓ | F@5 (%) ↑ | F@10 (%) ↑ | Interaction CDₕ (cm²) ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Paper, Table 1](https://arxiv.org/html/2602.04672v4#S4.T1) | 3.92 | 0.27 | 86.63 | 97.77 | 15.81 |
| Released code and data | 3.9453 | 0.2710 | 84.30 | 98.52 | 15.7669 |

Release scores average all **18 sequences** equally over the full available prepared range (3,590 sampled frames), using the [evaluation script included in this repository](optimization/README.md#evaluation). Interaction CDₕ measures object error relative to each hand's wrist; hand-aligned similarity-transform scores are reported separately. Prepared frame coverage and object-registration details differ from the original paper evaluation, so the paper row is a reference rather than an exact reproduction claim. [Per-sequence results and protocol](docs/ho3d-results.md).

## Getting Started

Clone the repository and run the following commands from its root. Replace `/ABS/...` with absolute paths on your machine.

```bash
git clone https://github.com/aim-uofa/AGILE.git
cd AGILE
```

1. Follow the [installation guide](docs/installation.md).
2. Download a prepared sequence:

   ```bash
   python tools/download_ho3d.py --scenes MC1 --output /ABS/data
   ```

3. Follow the [optimization guide](optimization/README.md) to reconstruct, evaluate and visualize the interaction.

To view completed reconstructions without running optimization, follow [download and visualize results](docs/ho3d-results.md#download-and-visualize). Downloads are public and do not require a Hugging Face account.

To prepare your own inputs, provide RGB frames and hand/object masks, [choose a generation route](generation/README.md#choose-a-route), then follow the [HO3D preprocessing guide](preprocessing/README.md). Generation accepts user-supplied frames; a complete in-the-wild video workflow with dedicated configurations and examples is planned. Generation uses Conda and provides `doctor`, `generate`, `status` and `resume` commands for setup and execution. See [release notes](docs/release-notes.md) for updates and [third-party notices](THIRD_PARTY_NOTICES.md) for dependencies.

## TODO

- [x] Release the preprocessing, object generation and optimization code.
- [x] Release evaluation, result packaging and visualization tools.
- [x] Release HO3D configurations, preprocessed inputs and generated object assets.
- [x] Release validated HO3D optimized reconstructions and per-sequence metrics.
- [ ] Release DexYCB optimization and evaluation configurations.
- [ ] Release DexYCB preprocessed inputs and generated object assets.
- [ ] Release the in-the-wild video workflow, including frame and mask preparation, hand/depth alignment, object initialization, optimization configurations and an end-to-end example.
- [ ] Release the retargeting module with documentation and examples.

## License

AGILE is available for **non-commercial research use** under the [AGILE license](LICENSE). For commercial use, contact [Professor Chunhua Shen](https://cshen.github.io/).

Third-party code, dependencies, model weights and datasets follow their original licenses. See [third-party notices](THIRD_PARTY_NOTICES.md).

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{shi2026agile,
  title={AGILE: Hand-Object Interaction Reconstruction from Video via Agentic Generation},
  author={Shi, Jin-Chuan and Ye, Binhong and Liu, Tao and Liu, Xiaoyang and Xu, Yangjinhui and He, Junzhe and Li, Zeju and Chen, Hao and Shen, Chunhua},
  journal={arXiv preprint arXiv:2602.04672},
  year={2026}
}
```

## Acknowledgements

We thank the authors of [HOLD](https://github.com/zc-alexfan/hold), [MagicHOI](https://github.com/byran-wang/MagicHOI), [WiLoR](https://github.com/rolpotamias/WiLoR), [FoundationPose](https://github.com/NVlabs/FoundationPose), [MoGe](https://github.com/microsoft/MoGe), [MegaSAM](https://github.com/mega-sam/mega-sam), and [Viser](https://github.com/nerfstudio-project/viser) for their excellent work.
