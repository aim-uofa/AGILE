# Installation

Use Linux x86_64, Python 3.10 and the CUDA 11.8 toolkit for optimization. The toolkit must include `nvcc` to build native extensions. Keep generation and preprocessing dependencies in separate environments. Run commands from the repository root.

## Optimization environment

```bash
conda create -n agile-optim --override-channels -c https://conda.anaconda.org/conda-forge python=3.10 pip -y
conda activate agile-optim
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
python -m pip install --no-user torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install --no-user pip==25.2 setuptools==80.9.0 wheel==0.45.1 ninja==1.13.0 numpy==1.23.5 scipy==1.15.3 six==1.17.0
python -m pip install --no-user --no-build-isolation chumpy==0.70
python -m pip install --no-user -r requirements.txt
```

Keep the installation order and NumPy pin for Chumpy compatibility. Build the CUDA extensions with the same toolkit as PyTorch. Set `TORCH_CUDA_ARCH_LIST` for your GPU: `8.9` for RTX 4090, `8.6` for RTX 3090, or `8.0` for A100.

```bash
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CUB_HOME="$CUDA_HOME/include"
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=4
nvcc --version

mkdir -p external
git clone https://github.com/facebookresearch/pytorch3d.git external/pytorch3d
git -C external/pytorch3d checkout 5043d15361d16a7093b4b60572c5f730c6c83308
CUDA_VISIBLE_DEVICES="" FORCE_CUDA=1 python -m pip install --no-user --no-build-isolation ./external/pytorch3d

git clone https://github.com/eliphatfs/cumesh2sdf.git external/cumesh2sdf
git -C external/cumesh2sdf checkout 7789ade81bcd5fa77112d2cb5d825ed0e634e76b
CUDA_VISIBLE_DEVICES="" python -m pip install --no-user --no-build-isolation ./external/cumesh2sdf

git clone https://github.com/hassony2/manopth.git external/manopth
git -C external/manopth checkout 4f1dcad1201ff1bfca6e065a85f0e3456e1aa32b
python -m pip install --no-user --no-build-isolation ./external/manopth
python -m pip check
```

## Licensed assets

Obtain MANO separately, then generate local support files with the original released finger selection:

```bash
python tools/prepare_mano_assets.py --model-dir /ABS/MANO --finger-mask configs/finger_vertices.json
export HO3D_MANO_ROOT=/ABS/MANO
export DINO_MODEL_PATH=/ABS/dinov3-vitl16-pretrain-lvd1689m
export HO3D_ROOT=/ABS/HO3D/raw
```

DINO must be the Transformers-compatible ViT-L/16 checkpoint with four register tokens, including config, processor configuration and weights. Use its upstream access process. MANO pickle files, topology templates and learned weights are not included. Evaluation requires the original HO3D annotations and object models, acquired separately.

The preparation tool creates the 1552-face hand rendering mesh, including its 14-triangle wrist cap. It preserves existing support files; use a fresh destination when replacing an older template.

## Preprocessing and generation

Install the public forks at the revisions in `configs/preprocessing-toolchain.json`. Use separate WiLoR, MoGe and FoundationPose environments; the workflow invokes them with `conda run`. Follow [preprocessing](../preprocessing/README.md) for the expected masks and inputs. MoGe depth is stored in millimeters.

Use `agile-generation` for the CPU generation controller. Create it from the repository root:

```bash
conda env create -f envs/generation.yml
conda activate agile-generation
python -m pip install --no-user './generation[gemini]'
```

The [generation guide](../generation/README.md) covers the paper's VLM/API workflow, Codex-assisted generation, Codex procedural geometry and external assets. The [setup guide](../generation/docs/setup.md) includes the separate `agile-hunyuan` CUDA environment, Blender, API configuration and `agile doctor` preflight checks. Install MCP only for agent-client integration. These tools are not optimizer dependencies.

## Verification

```bash
conda activate agile-optim
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
python tools/check_release.py

conda activate agile-generation
python -m pip install --no-user './generation[dev,gemini]'
PYTHONPATH=generation/src python -m pytest generation/tests
```

See [validation.json](validation.json) for recorded tests and end-to-end validation status.
