# Generation setup

Use a Conda environment for the CPU controller and, for model-based geometry, a separate Conda environment for the CUDA worker. Keep both separate from `agile-optim`. Commands below start at the repository root.

## Controller

```bash
conda env create -f envs/generation.yml
conda activate agile-generation
unset PYTHONPATH PYTHONHOME
python -m pip install --no-user './generation[gemini]'
python -m pip check
agile --help
```

The `gemini` extra installs the image API adapter. For Codex or supplied-asset routes that do not call this adapter, installing `./generation` is sufficient. Add `./generation[agent]` only if you use the [MCP interface](agent-api.md).

Choose one of the four [route profiles](../README.md#choose-a-route), copy it to a local JSON file, and edit that copy. Models and credentials are not bundled.

## Codex and Blender

The Codex-assisted, procedural and external-asset presets use an authenticated [Codex CLI](https://developers.openai.com/codex/cli/). Install it using the official instructions, then run `codex login` and `codex login status`. Set `agent.executable` if `codex` is not on `PATH`. Leave `agent.model` unset to use your CLI's configured default, or choose a model available to your account. If the service rejects that model, update your local profile; offline preflight cannot establish account access.

Install [Blender](https://www.blender.org/download/) and set `surface.executable` to its executable if needed. Blender performs surface preparation, UV layout and mesh processing. Check both tools in the shell where you will run AGILE:

```bash
codex --version
blender --version
```

Codex procedural geometry does not require the Hunyuan worker. Its textured preset pauses for an image matching the generated UV guide; submit that image through the CLI or your MCP client.

## VLM API

The paper workflow uses `agent.backend=vlm`: an OpenAI-compatible Chat Completions endpoint with image inputs and JSON output. Configure:

| Field | Value |
| --- | --- |
| `agent.base_url` | API prefix, such as `https://YOUR_VLM_ENDPOINT/v1`; the adapter appends `/chat/completions`. |
| `agent.model` | Your vision-language model's identifier. |
| `agent.api_key_env` | Credential variable name, default `AGILE_VLM_API_KEY`. |
| `agent.response_format` | `json_schema`, or `json_object` if your provider does not support JSON Schema. Responses are validated locally. |
| `agent.timeout_seconds` | Timeout for each API attempt, not the whole job. |
| `agent.max_retries` | Bounded retries for transient HTTP failures; default `0`. |

Set credential values in your shell or secret manager. JSON profiles contain variable names only. Images, requests and decisions are saved with the job; credentials are not saved as configuration fields.

## Image API

The VLM reviews images. A separate image adapter generates multiview images and texture edits. The included adapter uses the Gemini image API protocol.

| Field | Value |
| --- | --- |
| `image.provider` | `gemini` for the included adapter. |
| `image.model` | Your image-generation model; it must accept reference images and return images. |
| `image.api_key_env` | Credential variable name, default `AGILE_IMAGE_API_KEY`. |
| `image.base_url` | Optional compatible image API endpoint. Independent of the VLM endpoint. |
| `texture.provider` | `image_api` to generate and revise texture through the same adapter. |
| `texture.layout` | `mesh_uv`, preserving the prepared mesh's UV layout. |

The Codex-assisted and external profiles use `provided` images and textures by default: your image tool supplies them at the requested stages. You can instead configure the image API adapter while retaining Codex for reviews. See [asset submission](workflow.md#submit-generated-assets).

## Hunyuan CUDA worker

Required only for `mesh.backend=hunyuan`. This Linux x86_64 recipe uses Python 3.10, PyTorch 2.7.0, CUDA 11.8 and the public [Hunyuan3D-2 source revision](https://github.com/Tencent-Hunyuan/Hunyuan3D-2/tree/f8db63096c8282cb27354314d896feba5ba6ff8a) used by the adapter. A CUDA 11.8 toolkit with `nvcc`, a C++ compiler and an NVIDIA driver compatible with CUDA 11.8 are required. The dependency constraints keep this worker independent of the optimizer.

```bash
conda env create -f envs/generation-hunyuan.yml
conda activate agile-hunyuan
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1

python -m pip install --no-user 'setuptools<81' wheel ninja pybind11
python -m pip install --no-user torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu118

mkdir -p external
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git external/Hunyuan3D-2
git -C external/Hunyuan3D-2 checkout f8db63096c8282cb27354314d896feba5ba6ff8a
python -m pip install --no-user \
  -c generation/requirements/hunyuan-cu118-py310.constraints.txt \
  -r external/Hunyuan3D-2/requirements.txt
python -m pip install --no-user --no-build-isolation --no-deps ./external/Hunyuan3D-2
```

Build the two upstream texture extensions with the same CUDA toolkit. Set `TORCH_CUDA_ARCH_LIST` for your GPU: `8.9` for RTX 4090, `8.6` for RTX 3090 or `8.0` for A100.

```bash
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=4
nvcc --version
python -m pip install --no-user --no-build-isolation --no-deps \
  ./external/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer
python -m pip install --no-user --no-build-isolation --no-deps \
  ./external/Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer
python -m pip check
python -c 'import torch, custom_rasterizer, mesh_processor; from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline; from hy3dgen.texgen import Hunyuan3DPaintPipeline; print(torch.__version__, torch.cuda.is_available())'
python -c 'import sys; print(sys.executable)'
```

Use the last path as `mesh.python`, then return to `conda activate agile-generation`. Set the worker fields in your local profile:

| Field | Value |
| --- | --- |
| `mesh.python` | Absolute Python path inside `agile-hunyuan`. |
| `mesh.repository` | Absolute path to `external/Hunyuan3D-2`, containing `hy3dgen/`. |
| `mesh.shape_model` | Your compatible geometry checkpoint: local directory or model repository ID. |
| `mesh.subfolder` | Geometry checkpoint subfolder. |
| `mesh.texture_model` | Your compatible texture checkpoint: local directory or model repository ID. |

Choose multiview-compatible geometry weights for the paper and Codex-assisted routes; single-view weights cannot use named-view fallback. Obtain weights through the [upstream model links](https://github.com/Tencent-Hunyuan/Hunyuan3D-2/tree/f8db63096c8282cb27354314d896feba5ba6ff8a#-models-zoo) and follow their terms. First inference can also download background-removal and auxiliary model files; cache them before an offline run. The upstream source documents approximately 16 GB VRAM for shape and texture generation; actual memory depends on the selected models, resolution and offload settings.

## Check before running

```bash
conda activate agile-generation
agile doctor --config generation.local.json
agile doctor --config generation.local.json --json
```

`doctor` checks the selected route's configuration, credential variables, executables, worker Python and local modules, and configured model paths. It does not contact APIs, authenticate Codex, load weights or allocate CUDA memory. Remote checkpoint IDs still require a cache or network access and any upstream authorization at inference time. Resolve reported errors before `generate`; supplied-image stages are intentional pauses, not installation failures.

## Tests

```bash
conda activate agile-generation
python -m pip install --no-user './generation[dev,gemini]'
PYTHONPATH=generation/src python -m pytest generation/tests
```

Set `AGILE_TEST_BLENDER` to include Blender integration tests. Ordinary tests use local fixtures or mocked services. See [recorded validation](../../docs/validation.json) for the scope of actual generation runs.
