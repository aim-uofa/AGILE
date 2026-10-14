"""Public profiles require explicit model choices before starting providers."""

import builtins
import json
from pathlib import Path

import pytest

from agile3d.hunyuan_worker import main as run_mesh_worker
from agile3d.models import ImageConfig, MeshConfig, PipelineConfig
from agile3d.providers import generate_image, generate_mesh


@pytest.mark.parametrize("model", [None, "   "])
def test_missing_image_model_fails_before_credentials_or_input_reads(tmp_path, model):
    output = tmp_path / "image.png"
    with pytest.raises(ValueError, match=r"Set image\.model"):
        generate_image(
            ImageConfig(model=model, api_key_env="AGILE_UNSET_TEST_KEY"),
            [tmp_path / "missing-reference.png"], "object", output,
        )
    assert not output.exists()


@pytest.mark.parametrize("backend", ["provided", "disabled"])
def test_external_image_routes_do_not_require_a_model(tmp_path, backend):
    config = ImageConfig(provider=backend)
    with pytest.raises(ValueError, match="imported|disabled"):
        generate_image(config, [], "object", tmp_path / "image.png")


@pytest.mark.parametrize("missing", ["shape_model", "subfolder", "texture_model"])
def test_missing_mesh_choice_never_launches_worker(tmp_path, monkeypatch, missing):
    config = dict(shape_model="fixture-shape", subfolder="fixture-mv", texture_model="fixture-texture")
    config[missing] = None

    def unexpected_launch(*args, **kwargs):
        pytest.fail("An unconfigured model must not start a worker")

    monkeypatch.setattr("agile3d.providers.run_process", unexpected_launch)
    with pytest.raises(ValueError, match="mesh\\." + missing):
        generate_mesh(MeshConfig(**config), tmp_path / "missing.png", tmp_path / "mesh.glb", 42)
    assert not list(tmp_path.iterdir())


def test_standalone_worker_validates_before_loading_cuda(tmp_path, monkeypatch):
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"config": MeshConfig().model_dump()}))
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "hy3dgen"}:
            pytest.fail("An unconfigured model must not load CUDA dependencies")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(ValueError, match="mesh.shape_model"):
        run_mesh_worker(request)


def test_public_profiles_leave_model_choices_to_the_user():
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    for path in config_dir.glob("*.json"):
        if path.name in {"hoi.example.json", "review.example.json"}:
            continue
        profile = PipelineConfig.model_validate_json(path.read_text())
        assert profile.agent.model == ("YOUR_VISION_MODEL" if profile.agent.backend == "vlm" else None), path.name
        assert profile.image.model is None, path.name
        assert profile.mesh.shape_model is None, path.name
        assert profile.mesh.subfolder is None, path.name
        assert profile.mesh.texture_model is None, path.name
