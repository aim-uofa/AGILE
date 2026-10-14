# MCP agent interface

The [VLM API workflow](setup.md#vlm-api) runs directly from the CLI and does not require MCP. Use this interface when an agent client should inspect jobs, supply assets or invoke reconstruction stages.

The MCP server exposes the generation workflow to an agent client. Start it from the repository root with the generation environment active:

```bash
conda activate agile-generation
python -m pip install --no-user './generation[agent]'
agile-mcp --root /ABS/runs \
  --config generation/configs/codex-paper-parallel.json \
  --input-root /ABS/input
```

Connect the client's stdio transport to this command. Repeat `--input-root` for additional directories that the server may read. Put submitted frames, generated images, meshes and provenance files under these directories.

## Tools

| Task | Tools |
| --- | --- |
| Select references and create a job | `preview_frames`, `select_and_create_job`, `create_agile_job` |
| Start and inspect a candidate | `start_candidate`, `get_job`, `inspect_candidate` |
| Supply generated views or geometry | `inspect_generation_request`, `apply_generated_multiview`, `apply_lifted_mesh` |
| Review views | `review_multiview_candidate`, `select_multiview_subset` |
| Supply and review texture | `inspect_texture_request`, `apply_generated_texture`, `review_texture_candidate` |
| Review the mesh or final candidate | `review_candidate` |
| Select and export | `inspect_final_candidates`, `submit_final_selection`, `finalize_best_asset`, `export_accepted_asset` |
| Complete a final texture request | `inspect_final_texture_request`, `apply_final_generated_texture` |

Use the server's tool schemas for argument names and structured review fields. Inspect the evidence before submitting a review; the review must reference the current evidence digest. Poll `get_job` while a worker runs.

For supplied-image workflows, the client provides image generation. The VLM API profile instead uses its configured image API automatically. Follow the current request's inputs, UV layout and [provenance requirements](workflow.md#submit-generated-assets).

## Reconstruction handoff

The optional `--hoi-config generation/configs/hoi.example.json` enables `prepare_hoi_run` and `start_hoi_run_stage`. Edit its paths for your installation. Run stages in order: `preprocess`, `train`, `evaluate`.

This handoff supports the 18 released HO3D training sequences. Use their complete prepared frame ranges from [ho3d_scenes.json](../../configs/ho3d_scenes.json) for evaluation.
