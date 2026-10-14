"""Select the configured visual decision service without changing its schemas."""

from .review_backend import ReviewBackend


def get_agent_backend(config) -> ReviewBackend:
    if config.backend == "vlm":
        from .vlm_backend import VLMBackend

        return VLMBackend(config)
    from .codex_backend import CodexBackend, CodexConfig

    return CodexBackend(
        CodexConfig(
            **{
                key: getattr(config, key)
                for key in (
                    "executable",
                    "model",
                    "reasoning_effort",
                    "timeout_seconds",
                    "review_context",
                )
            }
        )
    )
