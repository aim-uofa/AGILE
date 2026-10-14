# Generated assets

`generate` and `resume` select and export usable completed candidates automatically. Read the printed delivery path, or inspect the saved job:

```bash
agile --root /ABS/runs status JOB_ID --human
```

## Files and quality states

`delivery-<snapshot>/asset/` contains the selected GLB, OBJ, MTL, textures and provenance. The selection record identifies the source candidate and quality state:

- `accepted`: all required reviews passed.
- `best_available`: usable geometry was selected with one or more unmet review criteria.
- `needs_geometry`: no usable geometry exists; repair or resume the job before using it.

A complete file is not necessarily an accepted result. Inspect its renders and review record before reconstruction. All previous candidates and failed reviews remain in the job directory.

## Manual selection and export

Use these optional commands to inspect or override an already generated selection:

```bash
agile --root /ABS/runs inspect-final JOB_ID
agile --root /ABS/runs finalize JOB_ID --file /ABS/selection.json
agile --root /ABS/runs export-accepted JOB_ID
```

The selection JSON follows the schema and evidence returned by `inspect-final`. Without `--file`, `finalize` uses the saved VLM or Codex backend. `export-accepted` exports only a candidate that passed its required reviews.

If delivery requests a texture, inspect the request through the [agent API](agent-api.md), submit the matching atlas, and resume or finalize. The optional `generation.final_texture_completion=true` allows one additional image API call to texture a usable UV mesh after normal budgets are exhausted. It is disabled in the paper profile and never converts a failed review to accepted.

## Use for reconstruction

Place `textured_mesh.obj` and its exported material and texture files under:

```text
generated/train/SCENE/object/
```

Keep relative material and texture paths intact. Then follow [preprocessing](../../preprocessing/README.md) to initialize pose and scale, and [optimization](../../optimization/README.md) to reconstruct motion. The optional HOI handoff accepts reviewed assets by default; a `best_available` asset requires an explicit configuration option.
