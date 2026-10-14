# Object generation

Turn video frames into a reviewed, textured 3D object for hand-object reconstruction. Choose the paper's API workflow, a Codex-assisted workflow, Codex procedural geometry, or assets from your own tools. If you use the released HO3D objects, go directly to [optimization](../optimization/README.md).

## Features

- **Select useful observations.** A VLM or Codex selects complementary reference frames and checks candidates against them.
- **Review geometry and appearance separately.** Views, untextured geometry and textures have their own checks and bounded revisions. Texture corrections preserve approved geometry and UVs.
- **Resume from saved stages.** Jobs retain requests, candidates, reviews and worker logs. Continue an interrupted job without discarding its completed stages.
- **Deliver reusable assets.** Export GLB and OBJ with materials, textures and provenance for preprocessing, reconstruction or other applications.

## Choose a route

All four routes use the same job commands and output format.

| Route | Profile | Geometry and review | What you need |
| --- | --- | --- | --- |
| **Paper workflow: VLM + APIs** | [agile-paper-vlm.json](configs/agile-paper-vlm.json) | A VLM selects and reviews references; an image API synthesizes views; a 3D model reconstructs the object. | VLM and image APIs, 3D worker and Blender. No Codex required. |
| **Codex-assisted generation** | [agile-paper-codex.json](configs/agile-paper-codex.json) | Codex selects and reviews references and candidates; a 3D model reconstructs supplied multiview images. | Codex, an image-generation tool, 3D worker and Blender. |
| **Codex procedural geometry** | [codex-paper-parallel.json](configs/codex-paper-parallel.json) | Codex constructs geometry directly from references and reviews the result. No 3D generative model is used. | Codex and Blender; an image-generation tool supplies the requested texture. |
| **External assets** | [agile-paper-external.json](configs/agile-paper-external.json) | Supply generated views, a mesh and textures from your own tools; the preset uses Codex for review. | Codex, Blender and your asset-generation tools. |

The paper workflow follows the stages in [Section 3.1](../assets/paper/AGILE_paper.pdf) and [Supplementary Section 8](../assets/paper/AGILE_supplementary.pdf). The Codex-assisted and procedural routes are additional workflows. “Pure Codex” refers to procedural **geometry**: the textured preset still accepts an image from your image tool. No route assumes a particular model or account; configure your own models and credentials.

## Install with Conda

Run commands from the repository root:

```bash
conda env create -f envs/generation.yml
conda activate agile-generation
unset PYTHONPATH PYTHONHOME
python -m pip install --no-user './generation[gemini]'
agile --help
```

This CPU environment runs the controller and API adapters. Install only the additional tools required by your route: [Codex, Blender and the separate CUDA worker](docs/setup.md). MCP is optional and is not required for CLI use.

## Generate an object

Copy one profile from the table, then configure its executable paths and, where applicable, API endpoints, model IDs and weights. For example, to use Codex procedural geometry:

```bash
cp generation/configs/codex-paper-parallel.json generation.local.json
agile doctor --config generation.local.json
```

`doctor` checks configuration, local dependencies, executables and required credential variables before a job can make API calls. It does not download weights or test a paid service. See [setup](docs/setup.md) for the route-specific fields.

```bash
agile --root /ABS/runs generate \
  --config generation.local.json \
  --description 'The target object without the hand' \
  --frame /ABS/frame1.png --frame /ABS/frame2.png --max-selected 2
```

`generate` checks setup, selects references, runs the available stages, and exports the final asset. It prints the job ID, current state and next action. For supplied-image or supplied-mesh routes, it pauses at the relevant request. Follow [asset submission](docs/workflow.md#submit-generated-assets), then continue:

```bash
agile --root /ABS/runs status JOB_ID --human
agile --root /ABS/runs resume JOB_ID
```

A normal run does not require separate finalization commands. `resume` keeps the saved configuration and completed work. It does not start a second worker or silently repeat a paid image request with an unknown outcome. The [workflow guide](docs/workflow.md) covers each route, retries and manual controls.

## Inputs and outputs

| Item | Contents |
| --- | --- |
| Input | Object description, route configuration and 1–16 RGB frame candidates. Selection keeps 1–4 complementary references. |
| Intermediate results | Selected references, view images, meshes, renders, UV guides, reviews and logs in the job directory. |
| Final asset | `delivery-<snapshot>/asset/`: selected GLB/OBJ, MTL, textures and provenance. |
| Quality status | `accepted` when required reviews pass; `best_available` when a usable asset is delivered with unmet criteria. |

Keep the final material and texture files together. Preprocessing expects `textured_mesh.obj` under:

```text
generated/train/SCENE/object/
```

The asset describes geometry and appearance. [Preprocessing](../preprocessing/README.md) initializes camera-frame pose and scale; [optimization](../optimization/README.md) recovers the motion. See [outputs](docs/outputs.md) for selection and export details.

## Guides

- [Setup](docs/setup.md): Conda, 3D worker installation, APIs and preflight checks.
- [Workflow](docs/workflow.md): all routes, supplied assets, reviews and recovery.
- [Outputs](docs/outputs.md): delivered files and quality states.
- [Agent API](docs/agent-api.md): optional MCP integration and reconstruction handoff.
- [Mesh evaluation](docs/evaluation.md): optional metrics against a reference mesh.

The four profiles above are the starting points. Other files in `configs/` are component examples, test profiles or advanced compatibility options.

[Back to the project overview](../README.md)
