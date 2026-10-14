# Generation workflow

Complete [setup](setup.md), choose a [route](../README.md#choose-a-route), and run `doctor` on your local profile. The main commands are the same for every route:

```bash
agile doctor --config generation.local.json
agile --root /ABS/runs generate \
  --config generation.local.json \
  --description 'The target object without the hand' \
  --frame /ABS/frame1.png --frame /ABS/frame2.png --max-selected 2
agile --root /ABS/runs status JOB_ID --human
agile --root /ABS/runs resume JOB_ID
```

Supply up to 16 candidate frames. The configured VLM or Codex selects 1–4 complementary references. Each job saves its configuration, references and decisions. `generate` advances automatic stages and finalizes usable candidates; `resume` continues the same job.

## Paper workflow: VLM and APIs

Profile: [agile-paper-vlm.json](../configs/agile-paper-vlm.json).

1. **Select references.** A VLM selects informative frames with complementary viewpoints and limited occlusion.
2. **Synthesize and review views.** An image API generates front, back, left and right views. The VLM checks their geometry, viewpoints and appearance against the observations. Feedback guides bounded revisions.
3. **Reconstruct and prepare geometry.** A 3D model reconstructs the accepted views. Blender prepares the surface and UVs; the VLM reviews renders of the untextured mesh.
4. **Refine and review texture.** Review the baked texture first. If needed, the image API revises it using the references and fixed UV guide. Mesh geometry and UVs stay unchanged.
5. **Select and export.** Compare complete candidates and export the selected asset with its quality status.

This follows the generation stages in [Paper Section 3.1](../../assets/paper/AGILE_paper.pdf) and [Supplementary Section 8](../../assets/paper/AGILE_supplementary.pdf). Configurable providers and models can produce different assets; the workflow does not guarantee identical benchmark scores.

## Codex-assisted generation

Profile: [agile-paper-codex.json](../configs/agile-paper-codex.json).

Codex selects reference frames and performs visual reviews. Your image-generation tool supplies the requested multiview images; the configured 3D worker reconstructs geometry. Blender prepares the mesh and UVs, and your image tool supplies texture revisions when requested. Codex reviews and selects the final result.

The supplied-image stages pause explicitly. Submit the requested file and provenance, then run `resume`. To automate image creation too, configure the [image API adapter](setup.md#image-api) while keeping `agent.backend=codex`.

## Codex procedural geometry

Profile: [codex-paper-parallel.json](../configs/codex-paper-parallel.json).

Codex builds procedural geometry directly from the selected references; no separate 3D generation model is called. Geometry checks and Blender surface/UV preparation follow. For a textured asset, supply the texture requested by the UV stage, then let Codex review the result.

This is the “pure Codex” geometry route. The supplied textured profile still needs an image-generation tool or a user-provided texture. The smaller [codex-procedural.json](../configs/codex-procedural.json) is an advanced geometry-only profile, not the textured reconstruction workflow.

## External assets

Profile: [agile-paper-external.json](../configs/agile-paper-external.json).

Use your own image and mesh tools. The job pauses for multiview images, a GLB and texture inputs, retaining the same review and provenance checks. The preset uses Codex for selection and review; replace its agent configuration with the VLM settings if preferred.

## Submit generated assets

Submit only the input requested by the current job state:

| Waiting status | Required input | Command after `agile --root /ABS/runs` |
| --- | --- | --- |
| `awaiting_multiview` | Square four-view image: front/back on the top row, left/right on the bottom row. | `apply-multiview JOB_ID --image /ABS/views.png --provenance /ABS/views.json` |
| `awaiting_mesh` | Self-contained GLB reconstructed from the approved views. | `apply-mesh JOB_ID --mesh /ABS/model.glb --provenance /ABS/mesh.json` |
| `awaiting_texture` | Square texture atlas matching the current UV guide. | `apply-texture JOB_ID --image /ABS/texture.png --provenance /ABS/texture.json` |

A texture atlas must be 512–4096 pixels per side. Keep the approved mesh and UVs unchanged during texture revisions. The current request supplies the required guide and geometry hashes. Provenance JSON binds the file to that request:

- Views: `image_sha256`.
- Mesh: `mesh_sha256` and either `multiview_sha256` or `view_manifest_sha256`, as required by the request.
- Texture: `image_sha256`, `prepared_mesh_sha256` and `uv_layout_sha256`.

Use hashes of the submitted files, then run `resume JOB_ID`. The [MCP interface](agent-api.md) lets an agent inspect requests and submit these inputs. API-generated inputs record provenance automatically.

## Reviews, retries and recovery

Each review stage has a finite attempt budget. Rejected candidates remain available for inspection; they are not relabeled as accepted. A provider or worker failure keeps its logs and current state.

The paper API profile stops if no complete four-view candidate passes. The Codex-assisted preset also allows `multiview_fallback=best_subset`: select 2–3 consistent named views from previous attempts. This is an additional recovery option; the worker receives the actual named images, not a preview grid.

Use `status JOB_ID --human` to see the next action, and `resume JOB_ID` after supplying an input. `resume` does not take over a `queued` or `running` job, even if its worker has exited. If a job is stuck in either state, first confirm that the original worker has ended, then use the advanced `candidate JOB_ID --resume` command; do not launch it while the original worker is active. Ambiguous image API outcomes also require inspection before a repeat request. All attempts stay in the job directory.

## Advanced controls

The lower-level commands remain available for custom integrations:

- `create-from-frames` creates the selected-reference job without advancing its stages.
- `agent-loop` and `agent-review` expose stage-level orchestration and review.
- `candidate --resume` is the lower-level worker recovery command; prefer `resume` for normal use.
- `inspect-final`, `finalize` and `export-accepted` provide manual output selection and export.

Use `agile COMMAND --help` for arguments. Earlier `codex-loop` and `codex-review` names remain aliases; the job's saved backend determines which agent is used. Normal `generate` / `resume` runs already finalize complete assets, so separate finalization commands are not required.

Continue with [outputs](outputs.md) to use the delivered mesh.
