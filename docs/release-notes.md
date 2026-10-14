# Release notes

## Available

- Preprocessing, object generation, optimization, evaluation and visualization tools.
- HO3D configurations and prepared inputs for 18 sequences, with checksums in the [data manifest](data-release.json).
- Validated optimized reconstructions and per-sequence metrics for all 18 HO3D sequences, with [download and evaluation instructions](ho3d-results.md).
- A configurable VLM API workflow for reference selection, multiview generation, geometry and texture review, and final asset selection. Codex geometry and review remain optional; see the [generation guide](../generation/README.md).

## Implementation updates

- Stable rotation conversion near 180° and direct rotation-matrix export.
- Explicit matrix loading for short sequences.
- Convergence checks using step size, objective change and gradients.
- MANO support-file preparation with a closed wrist mesh.
- Correct backward frame selection and ordering for aligned sequence inputs.

These changes can affect results relative to earlier implementations. Recorded tests and end-to-end checks are listed in [validation.json](validation.json).

## Data paths

Generated assets use `object/`, and directory-format object initialization uses `object_pose/`. File contents and per-sequence input selections are unchanged. Older pinned releases remain readable. See [prepared data](data.md).

Each of the 18 preprocessed archives now includes all of its sequence inputs. The eight former initialization archives have been merged into their matching preprocessed archives without changing extracted paths or file bytes. The prepared-input and generated-object collection contains 36 archives; optimized results are distributed in 18 additional archives with a separate manifest. Previous revisions are retained.

## Data coverage

Evaluation uses the available prepared frame range, which can end before the raw video. See [sequence definitions](../configs/ho3d_scenes.json) and the evaluator's frame-coverage records. Keep display meshes separate from optimization proxies.

DexYCB configurations, additional prepared data and the retargeting module are planned; see the [roadmap](../README.md#todo).
