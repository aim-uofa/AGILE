"""Configuration contains environment variable names, never API credentials."""

from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)


class ImageConfig(StrictModel):
    provider: Literal["gemini", "provided", "disabled"] = "gemini"
    model: str | None = Field(default=None, min_length=1)
    api: Literal["generate_content", "interactions"] = "generate_content"
    api_key_env: str = Field(
        default="AGILE_IMAGE_API_KEY", pattern=r"^[A-Z][A-Z0-9_]*$"
    )
    base_url: str | None = None
    timeout_seconds: int = Field(default=180, ge=10, le=600)

    @model_validator(mode="after")
    def credential_free_endpoint(self):
        if self.base_url:
            url = urlsplit(self.base_url)
            if (
                url.scheme not in ("https", "http")
                or not url.hostname
                or url.username
                or url.password
                or url.query
                or url.fragment
            ):
                raise ValueError(
                    "base_url must be an HTTP(S) endpoint without credentials or query parameters"
                )
        return self


class MeshConfig(StrictModel):
    backend: Literal["hunyuan", "provided", "codex"] = "hunyuan"
    python: str | None = None
    repository: str | None = None
    shape_model: str | None = Field(default=None, min_length=1)
    subfolder: str | None = Field(default=None, min_length=1)
    texture_model: str | None = Field(default=None, min_length=1)
    steps: int = Field(default=30, ge=1, le=100)
    octree_resolution: int = Field(default=256, ge=128, le=512)
    seed: int = Field(default=1234, ge=0, le=2**31 - 1)
    device: str = "cuda"
    low_vram: bool = True
    timeout_seconds: int = Field(default=1800, ge=30, le=7200)


class QualityPolicy(StrictModel):
    max_attempts: int = Field(default=3, ge=1, le=10)
    min_score: float = Field(default=7.0, ge=0, le=10)
    require_texture: bool = True
    require_watertight: bool = False
    max_faces: int = Field(default=500_000, ge=4, le=2_000_000)
    max_components: int = Field(default=8, ge=1, le=1000)


class AgentConfig(StrictModel):
    backend: Literal["codex", "vlm"] = "codex"
    base_url: str | None = None
    api_key_env: str = Field(default="AGILE_VLM_API_KEY", pattern=r"^[A-Z][A-Z0-9_]*$")
    response_format: Literal["json_schema", "json_object"] = "json_schema"
    max_retries: int = Field(default=0, ge=0, le=2)
    max_output_tokens: int = Field(default=4096, ge=256, le=32768)
    executable: str = "codex"
    model: str | None = None
    reasoning_effort: (
        Literal["low", "medium", "high", "xhigh", "max", "ultra"] | None
    ) = None
    timeout_seconds: int = Field(default=600, ge=30, le=3600)
    review_context: Literal["full", "images_only"] = "full"

    @model_validator(mode="after")
    def valid_vlm_endpoint(self):
        if self.backend == "vlm":
            if not self.model or not self.model.strip() or not self.base_url:
                raise ValueError("VLM agents require explicit model and base_url")
            url = urlsplit(self.base_url)
            if (
                url.scheme not in ("https", "http")
                or not url.hostname
                or url.username
                or url.password
                or url.query
                or url.fragment
            ):
                raise ValueError(
                    "VLM base_url must be HTTP(S) without credentials, query or fragment"
                )
            if url.scheme == "http" and url.hostname not in (
                "localhost",
                "127.0.0.1",
                "::1",
            ):
                raise ValueError(
                    "Remote VLM endpoints require HTTPS; HTTP is only for loopback servers"
                )
            if self.reasoning_effort is not None:
                raise ValueError(
                    "reasoning_effort is Codex-specific; omit it for a VLM endpoint"
                )
        return self


class TextureConfig(StrictModel):
    """Choose manual atlas import, an explicit image API, or procedural textures."""

    provider: Literal["procedural", "provided", "image_api"] = "procedural"
    layout: Literal["atlas", "mesh_uv"] = "atlas"


class SurfaceConfig(StrictModel):
    enabled: bool = False
    executable: str = "blender"
    method: Literal["preserve", "quadriflow", "decimate"] = "preserve"
    target_faces: int = Field(default=3000, ge=100, le=50000)
    texture_size: int = Field(default=1024, ge=512, le=4096)
    timeout_seconds: int = Field(default=600, ge=30, le=3600)
    geometry_review: bool = True
    repair_nonmanifold: bool = True
    uv_method: Literal["semantic", "smart_project"] = "semantic"


class GenerationPolicy(StrictModel):
    """Independent rejection loops; thresholds and budgets are engineering settings."""

    require_selection: bool = False
    multiview_review: bool = False
    max_multiview_attempts: int = Field(default=3, ge=1, le=10)
    min_multiview_score: float = Field(default=7.0, ge=0, le=10)
    multiview_fallback: Literal["best_subset", "stop"] = "best_subset"
    fallback_max_views: Literal[2, 3] = 3
    texture_review: bool = False
    max_texture_attempts: int = Field(default=3, ge=1, le=10)
    min_texture_score: float = Field(default=7.0, ge=0, le=10)
    final_texture_completion: bool = False


class PipelineConfig(StrictModel):
    image: ImageConfig = Field(default_factory=ImageConfig)
    mesh: MeshConfig = Field(default_factory=MeshConfig)
    quality: QualityPolicy = Field(default_factory=QualityPolicy)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    texture: TextureConfig = Field(default_factory=TextureConfig)
    surface: SurfaceConfig = Field(default_factory=SurfaceConfig)
    generation: GenerationPolicy = Field(default_factory=GenerationPolicy)

    @model_validator(mode="after")
    def coherent_route(self):
        if self.texture.provider == "image_api" and (
            self.image.provider != "gemini" or not self.generation.texture_review
        ):
            raise ValueError(
                "image_api textures require an image API and texture_review=True"
            )
        if self.agent.backend == "vlm" and self.mesh.backend == "codex":
            raise ValueError(
                "VLM API profiles use Hunyuan or supplied meshes, not Codex procedural geometry"
            )
        if self.generation.multiview_review and self.image.provider == "disabled":
            raise ValueError("Multi-view validation requires an image generation route")
        if self.generation.texture_review and (
            not self.surface.enabled
            or self.texture.provider not in ("provided", "image_api")
            or self.texture.layout != "mesh_uv"
        ):
            raise ValueError(
                "Texture rejection sampling requires Blender and mesh_uv provided textures"
            )
        if self.texture.layout == "mesh_uv" and not self.surface.enabled:
            raise ValueError("mesh_uv requires the Blender surface stage")
        if (
            self.surface.enabled
            and self.texture.provider in ("provided", "image_api")
            and self.texture.layout != "mesh_uv"
        ):
            raise ValueError(
                "Blender generated textures must use the frozen mesh_uv layout"
            )
        if (
            self.texture.provider in ("provided", "image_api")
            and self.mesh.backend != "codex"
            and self.texture.layout != "mesh_uv"
        ):
            raise ValueError(
                "Generated texture atlases currently require Codex geometry"
            )
        if (self.mesh.backend == "codex") != (self.image.provider == "disabled"):
            raise ValueError(
                "Codex geometry requires image.provider=disabled; other routes require images"
            )
        return self


class Review(StrictModel):
    """The agent reviews references AND actual mesh views. Scores are not metrics."""

    evidence_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    decision: Literal["accept", "retry", "reject"]
    score: float = Field(ge=0, le=10)
    matches_reference: bool
    no_hand_geometry: bool
    consistent_views: bool
    feedback: str = Field(min_length=1, max_length=6000)

    @model_validator(mode="after")
    def consistent_acceptance(self):
        if self.decision == "accept" and not all(
            [self.matches_reference, self.no_hand_geometry, self.consistent_views]
        ):
            raise ValueError("Acceptance requires all three visual checks to pass")
        return self
