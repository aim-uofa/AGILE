from types import SimpleNamespace

from agile3d.hunyuan_worker import load_texture_pipeline


def test_low_memory_texture_loading_never_constructs_diffusion_models_on_cuda():
    events = []

    class Pipeline:
        @classmethod
        def from_pretrained(cls, path):
            assert path == "local-weights"
            instance = cls()
            instance.config = SimpleNamespace(device="cuda")
            instance.models = {}
            instance.load_models()
            return instance

        def load_models(self):
            # Mimic upstream's eager .to(config.device), the memory peak that
            # previously happened before CPU offload could be enabled.
            assert self.config.device == "cpu"
            self.models = {
                name: SimpleNamespace(device=self.config.device)
                for name in ("delight_model", "multiview_model")
            }
            events.append("weights_loaded_on_cpu")

        def enable_model_cpu_offload(self, *, device):
            assert self.config.device == device == "cuda:0"
            assert all(model.device == device for model in self.models.values())
            events.append("offload_installed")

    pipeline = load_texture_pipeline(
        Pipeline, "local-weights", device="cuda:0", low_vram=True
    )
    assert isinstance(pipeline, Pipeline)
    assert events == ["weights_loaded_on_cpu", "offload_installed"]
